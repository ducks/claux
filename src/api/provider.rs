use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use super::stream::ApiEvent;
use super::types::{Message, ToolDefinition};

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
    ) -> Result<mpsc::Receiver<ApiEvent>>;
}
