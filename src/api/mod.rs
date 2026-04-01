pub(crate) mod stream;
pub(crate) mod types;

pub use stream::ApiEvent;
pub use types::*;

use anyhow::Result;
use serde_json::json;
use tokio::sync::mpsc;

/// Claude Messages API client with SSE streaming.
pub struct Client {
    api_key: String,
    model: String,
    api_url: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(api_key: String, model: &str) -> Self {
        Self {
            api_key,
            model: model.to_string(),
            api_url: "https://api.anthropic.com/v1/messages".to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
    }

    /// Send a streaming chat request. Returns a channel receiver of events.
    pub async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        max_tokens: u32,
    ) -> Result<mpsc::Receiver<ApiEvent>> {
        let (tx, rx) = mpsc::channel(256);

        let mut body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "stream": true,
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }

        let response = self
            .http
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error ({}): {}", status, error_text);
        }

        // Spawn a task to read the SSE stream
        let response = response;
        tokio::spawn(async move {
            if let Err(e) = stream::read_sse_stream(response, tx).await {
                tracing::error!("SSE stream error: {}", e);
            }
        });

        Ok(rx)
    }
}
