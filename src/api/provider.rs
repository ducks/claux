use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::stream::ApiEvent;
use super::types::{Message, ToolDefinition};

/// An owned provider response stream.
///
/// Dropping the stream cancels the underlying HTTP body reader so callers
/// cannot accidentally leave a detached request consuming tokens.
pub struct ProviderStream {
    rx: mpsc::Receiver<ApiEvent>,
    cancel: CancellationToken,
    secret: String,
}

impl ProviderStream {
    pub(crate) fn new(rx: mpsc::Receiver<ApiEvent>, cancel: CancellationToken) -> Self {
        Self {
            rx,
            cancel,
            secret: String::new(),
        }
    }

    pub(crate) fn with_secret(mut self, secret: &str) -> Self {
        self.secret = secret.to_string();
        self
    }

    pub async fn recv(&mut self) -> Option<ApiEvent> {
        let mut event = self.rx.recv().await?;
        if let ApiEvent::Error(failure) = &mut event {
            failure.message = super::error::redact_credentials(&failure.message, &self.secret);
        }
        Some(event)
    }
}

impl Drop for ProviderStream {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Trait for LLM API providers (Anthropic, OpenAI-compatible, etc.)
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn set_model(&mut self, model: &str);

    /// Drop provider-side conversation state before loading another session.
    ///
    /// Stateless providers need no special handling. Stateful providers can
    /// override this to clear continuation cursors such as
    /// `previous_response_id`.
    fn reset_session(&mut self) {}

    /// Send a streaming request. Returns a channel of events.
    async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        max_tokens: u32,
        cancel: CancellationToken,
    ) -> Result<ProviderStream>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn streamed_error_redacts_the_request_key() {
        let (tx, rx) = mpsc::channel(1);
        tx.send(ApiEvent::Error(super::super::error::ApiFailure::other(
            "bad private-secret",
        )))
        .await
        .unwrap();
        let mut stream =
            ProviderStream::new(rx, CancellationToken::new()).with_secret("private-secret");
        let Some(ApiEvent::Error(error)) = stream.recv().await else {
            panic!("expected error");
        };
        assert_eq!(error.message, "bad [redacted]");
    }

    #[test]
    fn dropping_provider_stream_cancels_its_reader() {
        let (_tx, rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let stream = ProviderStream::new(rx, cancel.clone());

        drop(stream);

        assert!(cancel.is_cancelled());
    }
}
