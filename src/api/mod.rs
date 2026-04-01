pub(crate) mod stream;
pub(crate) mod types;

pub use stream::ApiEvent;
pub use types::*;

use anyhow::Result;
use serde_json::json;
use tokio::sync::mpsc;

use crate::config::AuthMethod;

/// Claude Messages API client with SSE streaming.
pub struct Client {
    auth: AuthMethod,
    model: String,
    api_url: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(auth: AuthMethod, model: &str) -> Self {
        Self {
            auth,
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

        let mut request = self
            .http
            .post(&self.api_url)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        // Set auth header based on method
        request = match &self.auth {
            AuthMethod::ApiKey(key) => request.header("x-api-key", key),
            AuthMethod::OAuthToken(token) => {
                request.header("Authorization", format!("Bearer {}", token))
            }
        };

        let response = request.json(&body).send().await?;

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
