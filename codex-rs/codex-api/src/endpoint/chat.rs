use crate::auth::SharedAuthProvider;
use crate::common::ChatCompletionsRequest;
use crate::common::ChatCompletionsResponse;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::sse::non_streaming_response_to_events;
use crate::sse::spawn_chat_stream;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use std::sync::Arc;
use std::sync::OnceLock;
use tokio::sync::mpsc;
use tracing::instrument;

pub struct ChatClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

#[derive(Default)]
pub struct ChatOptions {
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_source: Option<SessionSource>,
    pub extra_headers: HeaderMap,
    pub compression: Compression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

impl<T: HttpTransport> ChatClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn codex_client::RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    #[instrument(
        name = "chat.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "chat_http",
            http.method = "POST",
            api.path = "chat/completions"
        )
    )]
    pub async fn stream_request(
        &self,
        request: ChatCompletionsRequest,
        options: ChatOptions,
    ) -> Result<ResponseStream, ApiError> {
        let ChatOptions {
            session_id,
            thread_id,
            session_source: _,
            extra_headers,
            compression,
            turn_state,
        } = options;

        let body = request.to_value()
            .map_err(|e| ApiError::Stream(format!("failed to encode chat request: {e}")))?;

        let mut headers = extra_headers;
        if let Some(ref thread_id) = thread_id {
            headers.insert("x-client-request-id", HeaderValue::from_str(thread_id).unwrap());
        }
        headers.extend(build_session_headers(session_id, thread_id));

        if request.stream {
            self.stream(body, headers, compression, turn_state).await
        } else {
            // Non-streaming: make the request and convert to events
            self.non_streaming(body, headers, compression).await
        }
    }

    #[instrument(
        name = "chat.stream",
        level = "info",
        skip_all,
        fields(
            transport = "chat_http",
            http.method = "POST",
            api.path = "chat/completions",
            turn.has_state = turn_state.is_some()
        )
    )]
    pub async fn stream(
        &self,
        body: serde_json::Value,
        extra_headers: HeaderMap,
        compression: Compression,
        turn_state: Option<Arc<OnceLock<String>>>,
    ) -> Result<ResponseStream, ApiError> {
        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        let stream_response = self
            .session
            .stream_with(
                Method::POST,
                Self::path(),
                extra_headers,
                Some(body),
                |req| {
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    req.compression = request_compression;
                },
            )
            .await?;

        Ok(spawn_chat_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            turn_state,
        ))
    }

    /// Non-streaming Chat Completions call. Returns the events as a ResponseStream
    /// by fully buffering the response, parsing it, and converting to events.
    async fn non_streaming(
        &self,
        body: serde_json::Value,
        extra_headers: HeaderMap,
        compression: Compression,
    ) -> Result<ResponseStream, ApiError> {
        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        let response = self
            .session
            .execute_with(
                Method::POST,
                Self::path(),
                extra_headers,
                Some(body),
                |req| {
                    req.compression = request_compression;
                },
            )
            .await?;

        let chat_response: ChatCompletionsResponse =
            serde_json::from_slice(&response.body)
                .map_err(|e| ApiError::Stream(format!("failed to parse chat response: {e}")))?;

        let events = non_streaming_response_to_events(chat_response);
        let (tx, rx) = mpsc::channel(events.len());
        for event in events {
            let _ = tx.send(event).await;
        }
        // Drop tx to close the channel after all events are sent
        drop(tx);

        Ok(ResponseStream {
            rx_event: rx,
            upstream_request_id: None,
        })
    }

    fn path() -> &'static str {
        "chat/completions"
    }
}
