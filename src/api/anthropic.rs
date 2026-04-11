use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

use super::provider::Provider;
use super::stream::{self, ApiEvent};
use super::types::{Message, ToolDefinition};
use crate::config::AuthMethod;
use crate::context::SYSTEM_PROMPT_BLOCK_SEPARATOR;

/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    auth: AuthMethod,
    model: String,
    api_url: String,
    http: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(auth: AuthMethod, model: &str) -> Self {
        Self {
            auth,
            model: model.to_string(),
            api_url: "https://api.anthropic.com/v1/messages".to_string(),
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
    }

    async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        max_tokens: u32,
    ) -> Result<mpsc::Receiver<ApiEvent>> {
        let (tx, rx) = mpsc::channel(256);

        // Split system prompt into blocks matching Claude Code's 3-block array format.
        // Block 0: billing/version header
        // Block 1: identity + runtime context
        // Block 2: static instructions
        let system_blocks: Vec<serde_json::Value> = system
            .split(SYSTEM_PROMPT_BLOCK_SEPARATOR)
            .map(|block| {
                json!({
                    "type": "text",
                    "text": block,
                })
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
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

        request = match &self.auth {
            AuthMethod::ApiKey(key) => request.header("x-api-key", key),
            AuthMethod::OAuthToken(token) => request
                .header("Authorization", format!("Bearer {token}"))
                .header("anthropic-beta", "oauth-2025-04-20"),
        };

        let response = request.json(&body).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error ({status}): {error_text}");
        }

        tokio::spawn(async move {
            if let Err(e) = stream::read_sse_stream(response, tx).await {
                tracing::error!("SSE stream error: {}", e);
            }
        });

        Ok(rx)
    }
}
