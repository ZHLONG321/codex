use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::ChatClient;
use codex_api::ChatCompletionsRequest;
use codex_api::ChatMessage;
use codex_api::ChatOptions;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::RetryConfig;
use codex_client::ReqwestTransport;
use futures::StreamExt;
use futures::stream;
use http::HeaderMap;
use std::sync::Arc;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[derive(Clone, Default)]
struct DummyAuth;

impl AuthProvider for DummyAuth {
    fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
}

fn provider(base_url: &str) -> Provider {
    Provider {
        name: "test".to_string(),
        base_url: base_url.to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: RetryConfig {
            max_attempts: 1,
            base_delay: std::time::Duration::from_millis(1),
            retry_429: false,
            retry_5xx: true,
            retry_transport: true,
        },
        stream_idle_timeout: std::time::Duration::from_secs(5),
    }
}

/// Build an SSE stream body with reasoning_content followed by content.
fn sse_stream_with_reasoning() -> String {
    let chunks = vec![
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think about this..."},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"reasoning_content":"The answer is 42."},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"content":"The answer is 42."},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#,
    ];
    let mut body = String::new();
    for chunk in &chunks {
        body.push_str(&format!("data: {}\n\n", chunk));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

#[tokio::test]
async fn chat_streaming_emits_reasoning_content_events() {
    let server = MockServer::start().await;
    let base_url = format!("{}/v1", server.uri());

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .insert_header("openai-model", "deepseek-v4-flash")
                .set_body_string(sse_stream_with_reasoning()),
        )
        .mount(&server)
        .await;

    let transport = ReqwestTransport::new(reqwest::Client::new());
    let client = ChatClient::new(transport, provider(&base_url), Arc::new(DummyAuth));

    let request = ChatCompletionsRequest {
        model: "deepseek-v4-flash".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some("What is the answer?".to_string()),
            ..Default::default()
        }],
        tools: None,
        tool_choice: None,
        stream: true,
        stop: None,
        max_tokens: None,
        temperature: None,
        extra_body: Some(serde_json::json!({"thinking": {"type": "enabled"}})),
    };

    let options = ChatOptions::default();
    let stream = client
        .stream_request(request, options)
        .await
        .expect("stream_request should succeed");

    let events: Vec<_> = stream
        .filter_map(|e| async { e.ok() })
        .collect()
        .await;

    // Verify we got events
    assert!(!events.is_empty(), "should have received events");

    // Check for ServerModel event
    let has_server_model = events.iter().any(|e| {
        matches!(e, codex_api::ResponseEvent::ServerModel(m) if m == "deepseek-v4-flash")
    });
    assert!(has_server_model, "should have ServerModel event");

    // Check for ReasoningContentDelta events
    let reasoning_deltas: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            codex_api::ResponseEvent::ReasoningContentDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !reasoning_deltas.is_empty(),
        "should have ReasoningContentDelta events"
    );
    let full_reasoning: String = reasoning_deltas.join("");
    assert!(
        full_reasoning.contains("Let me think"),
        "reasoning should contain expected text"
    );

    // Check for OutputTextDelta events
    let content_deltas: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            codex_api::ResponseEvent::OutputTextDelta(delta) => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !content_deltas.is_empty(),
        "should have OutputTextDelta events"
    );
    let full_content: String = content_deltas.join("");
    assert!(
        full_content.contains("42"),
        "content should contain expected text"
    );

    // Check for Completed event
    let has_completed = events
        .iter()
        .any(|e| matches!(e, codex_api::ResponseEvent::Completed { .. }));
    assert!(has_completed, "should have Completed event");

    println!("PASS: reasoning_content streaming test verified {} events", events.len());
}

/// Non-streaming response with reasoning_content.
#[tokio::test]
async fn chat_non_streaming_returns_reasoning_content() {
    let server = MockServer::start().await;
    let base_url = format!("{}/v1", server.uri());

    let response_body = serde_json::json!({
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "deepseek-v4-flash",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "Let me reason about this step by step.",
                "content": "The answer is 42."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .insert_header("openai-model", "deepseek-v4-flash")
                .set_body_json(&response_body),
        )
        .mount(&server)
        .await;

    let transport = ReqwestTransport::new(reqwest::Client::new());
    let client = ChatClient::new(transport, provider(&base_url), Arc::new(DummyAuth));

    let request = ChatCompletionsRequest {
        model: "deepseek-v4-flash".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some("What is the answer?".to_string()),
            ..Default::default()
        }],
        tools: None,
        tool_choice: None,
        stream: false,
        stop: None,
        max_tokens: None,
        temperature: None,
        extra_body: Some(serde_json::json!({"thinking": {"type": "enabled"}})),
    };

    let options = ChatOptions::default();
    let stream = client
        .stream_request(request, options)
        .await
        .expect("stream_request should succeed");

    let events: Vec<_> = stream
        .filter_map(|e| async { e.ok() })
        .collect()
        .await;

    // Check for reasoning_content in events
    let has_reasoning = events.iter().any(|e| {
        matches!(e, codex_api::ResponseEvent::ReasoningContentDelta { delta, .. } if delta.contains("reason"))
    });
    assert!(has_reasoning, "should have ReasoningContentDelta event with reasoning text");

    // Check for content
    let has_content = events.iter().any(|e| {
        matches!(e, codex_api::ResponseEvent::OutputTextDelta(d) if d.contains("42"))
    });
    assert!(has_content, "should have OutputTextDelta with answer");

    // Check for Completed
    let has_completed = events
        .iter()
        .any(|e| matches!(e, codex_api::ResponseEvent::Completed { .. }));
    assert!(has_completed, "should have Completed event");

    println!("PASS: non-streaming reasoning_content test verified {} events", events.len());
}

/// Direct SSE parser test — bypasses HTTP to isolate the parsing logic.
/// Each SSE chunk is yielded as a separate byte chunk to simulate real streaming.
#[tokio::test]
async fn sse_parser_direct_with_reasoning_content() {
    let chunks = vec![
        format!("data: {}\n\n", r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think about this..."},"finish_reason":null}]}"#),
        format!("data: {}\n\n", r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"reasoning_content":"The answer is 42."},"finish_reason":null}]}"#),
        format!("data: {}\n\n", r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"content":"The answer is 42."},"finish_reason":null}]}"#),
        format!("data: {}\n\n", r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#),
        "data: [DONE]\n\n".to_string(),
    ];

    let byte_stream: codex_client::ByteStream = Box::pin(stream::iter(
        chunks.into_iter().map(|c| Ok(Bytes::from(c))),
    ));

    let (tx, rx) = tokio::sync::mpsc::channel(1600);
    codex_api::process_chat_sse(
        byte_stream,
        tx,
        std::time::Duration::from_secs(5),
        None,
    )
    .await;

    let mut events = Vec::new();
    let mut rx = rx;
    while let Some(event) = rx.recv().await {
        if let Ok(e) = event {
            events.push(e);
        }
    }

    let has_reasoning = events.iter().any(|e| {
        matches!(e, ResponseEvent::ReasoningContentDelta { .. })
    });
    assert!(has_reasoning, "direct SSE parser: should have ReasoningContentDelta");

    let has_content = events.iter().any(|e| {
        matches!(e, ResponseEvent::OutputTextDelta(d) if d.contains("42"))
    });
    assert!(has_content, "direct SSE parser: should have OutputTextDelta");

    let has_completed = events.iter().any(|e| {
        matches!(e, ResponseEvent::Completed { .. })
    });
    assert!(has_completed, "direct SSE parser: should have Completed");
}
