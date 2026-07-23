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
}

impl ProviderStream {
    pub(crate) fn new(rx: mpsc::Receiver<ApiEvent>, cancel: CancellationToken) -> Self {
        Self { rx, cancel }
    }

    pub async fn recv(&mut self) -> Option<ApiEvent> {
        self.rx.recv().await
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
    fn model(&self) -> &str;
    fn set_model(&mut self, model: &str);

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

    #[test]
    fn dropping_provider_stream_cancels_its_reader() {
        let (_tx, rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let stream = ProviderStream::new(rx, cancel.clone());

        drop(stream);

        assert!(cancel.is_cancelled());
    }
}
