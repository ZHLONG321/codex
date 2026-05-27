use crate::common::ChatCompletionsChunk;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::rate_limits::parse_all_rate_limits;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

static NEXT_ITEM_ID: AtomicU64 = AtomicU64::new(1);

fn next_item_id() -> String {
    format!("chat_item_{}", NEXT_ITEM_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

const OPENAI_MODEL_HEADER: &str = "openai-model";
const REQUEST_ID_HEADER: &str = "x-request-id";

pub fn spawn_chat_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    turn_state: Option<Arc<OnceLock<String>>>,
) -> ResponseStream {
    let rate_limit_snapshots = parse_all_rate_limits(&stream_response.headers);
    let models_etag = stream_response
        .headers
        .get("X-Models-Etag")
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let server_model = stream_response
        .headers
        .get(OPENAI_MODEL_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    if let Some(turn_state) = turn_state.as_ref()
        && let Some(header_value) = stream_response
            .headers
            .get("x-codex-turn-state")
            .and_then(|v| v.to_str().ok())
    {
        let _ = turn_state.set(header_value.to_string());
    }
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        if let Some(model) = server_model {
            let _ = tx_event.send(Ok(ResponseEvent::ServerModel(model))).await;
        }
        for snapshot in rate_limit_snapshots {
            let _ = tx_event.send(Ok(ResponseEvent::RateLimits(snapshot))).await;
        }
        if let Some(etag) = models_etag {
            let _ = tx_event.send(Ok(ResponseEvent::ModelsEtag(etag))).await;
        }
        process_chat_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

struct ChatStreamState {
    reasoning_buffer: String,
    content_buffer: String,
    in_reasoning: bool,
    in_message: bool,
}

impl ChatStreamState {
    fn new() -> Self {
        Self {
            reasoning_buffer: String::new(),
            content_buffer: String::new(),
            in_reasoning: false,
            in_message: false,
        }
    }
}

const DONE_LINE: &str = "[DONE]";

pub async fn process_chat_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = ChatStreamState::new();
    let error = None;
    let mut last_server_model: Option<String> = None;

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }
        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Chat SSE Error: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                let error = error.unwrap_or(ApiError::Stream(
                    "stream closed before response completed".into(),
                ));
                let _ = tx_event.send(Err(error)).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream("idle timeout waiting for SSE".into())))
                    .await;
                return;
            }
        };

        trace!("Chat SSE event: {}", &sse.data);

        if sse.data.trim() == DONE_LINE {
            emit_finish(&tx_event, &mut state, None).await;
            return;
        }

        let chunk: ChatCompletionsChunk = match serde_json::from_str(&sse.data) {
            Ok(chunk) => chunk,
            Err(e) => {
                debug!("Failed to parse Chat Completions chunk: {e}, data: {}", &sse.data);
                continue;
            }
        };

        if let Some(ref model) = chunk.model
            && last_server_model.as_deref() != Some(model.as_str())
        {
            if tx_event
                .send(Ok(ResponseEvent::ServerModel(model.clone())))
                .await
                .is_err()
            {
                return;
            }
            last_server_model = Some(model.clone());
        }

        let Some(choice) = chunk.choices.into_iter().next() else {
            continue;
        };

        if let Some(ref reasoning) = choice.delta.reasoning_content {
            if !reasoning.is_empty() {
                if !state.in_reasoning {
                    state.in_reasoning = true;
                    let item = ResponseItem::Reasoning {
                        id: next_item_id(),
                        summary: Vec::new(),
                        content: None,
                        encrypted_content: None,
                    };
                    if tx_event.send(Ok(ResponseEvent::OutputItemAdded(item))).await.is_err() {
                        return;
                    }
                }
                state.reasoning_buffer.push_str(reasoning);
                if tx_event
                    .send(Ok(ResponseEvent::ReasoningContentDelta {
                        delta: reasoning.clone(),
                        content_index: 0,
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }

        if let Some(ref content) = choice.delta.content {
            if !content.is_empty() {
                if !state.in_message {
                    state.in_message = true;
                    let item = ResponseItem::Message {
                        id: Some(next_item_id()),
                        role: "assistant".to_string(),
                        content: Vec::new(),
                        phase: None,
                    };
                    if tx_event.send(Ok(ResponseEvent::OutputItemAdded(item))).await.is_err() {
                        return;
                    }
                }
                state.content_buffer.push_str(content);
                if tx_event.send(Ok(ResponseEvent::OutputTextDelta(content.clone()))).await.is_err() {
                    return;
                }
            }
        }

        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                if tx_event
                    .send(Ok(ResponseEvent::ToolCallInputDelta {
                        item_id: String::new(),
                        call_id: Some(tool_call.id),
                        delta: serde_json::to_string(&tool_call.function)
                            .unwrap_or_default(),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }

        if choice.finish_reason.is_some() {
            let usage = chunk.usage.map(TokenUsage::from);
            emit_finish(&tx_event, &mut state, usage).await;
            return;
        }
    }
}

async fn emit_finish(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut ChatStreamState,
    usage: Option<TokenUsage>,
) {
    if state.in_reasoning {
        let item = ResponseItem::Reasoning {
            id: next_item_id(),
            summary: Vec::new(),
            content: Some(vec![codex_protocol::models::ReasoningItemContent::ReasoningText {
                text: std::mem::take(&mut state.reasoning_buffer),
            }]),
            encrypted_content: None,
        };
        let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
    }

    if state.in_message {
        let item = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: std::mem::take(&mut state.content_buffer),
            }],
            phase: None,
        };
        let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
    }

    let _ = tx_event
        .send(Ok(ResponseEvent::Completed {
            response_id: String::new(),
            token_usage: usage,
            end_turn: None,
        }))
        .await;
}

pub fn non_streaming_response_to_events(
    response: crate::common::ChatCompletionsResponse,
) -> Vec<Result<ResponseEvent, ApiError>> {
    let mut events = Vec::new();

    if let Some(model) = response.model {
        events.push(Ok(ResponseEvent::ServerModel(model)));
    }

    let Some(choice) = response.choices.into_iter().next() else {
        return events;
    };

    if let Some(ref reasoning_content) = choice.message.reasoning_content {
        if !reasoning_content.is_empty() {
            let id = next_item_id();
            let reasoning_item = ResponseItem::Reasoning {
                id: id.clone(),
                summary: Vec::new(),
                content: Some(vec![codex_protocol::models::ReasoningItemContent::ReasoningText {
                    text: reasoning_content.clone(),
                }]),
                encrypted_content: None,
            };
            events.push(Ok(ResponseEvent::OutputItemAdded(reasoning_item.clone())));
            events.push(Ok(ResponseEvent::ReasoningContentDelta {
                delta: reasoning_content.clone(),
                content_index: 0,
            }));
            events.push(Ok(ResponseEvent::OutputItemDone(reasoning_item)));
        }
    }

    if let Some(ref content) = choice.message.content {
        if !content.is_empty() {
            let msg_id = Some(next_item_id());
            events.push(Ok(ResponseEvent::OutputItemAdded(ResponseItem::Message {
                id: msg_id.clone(),
                role: "assistant".to_string(),
                content: Vec::new(),
                phase: None,
            })));
            events.push(Ok(ResponseEvent::OutputTextDelta(content.clone())));
            let msg_done = ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: content.clone(),
                }],
                phase: None,
            };
            events.push(Ok(ResponseEvent::OutputItemDone(msg_done)));
        }
    }

    events.push(Ok(ResponseEvent::Completed {
        response_id: response.id.unwrap_or_default(),
        token_usage: response.usage.map(TokenUsage::from),
        end_turn: None,
    }));

    events
}
