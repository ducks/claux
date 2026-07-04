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

        add_cache_breakpoints(&mut body);

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

/// Insert ephemeral cache_control breakpoints into a request body.
///
/// The cached prefix is tools -> system -> messages, so one breakpoint on
/// the last system block caches everything stable for the session, and
/// moving breakpoints on the last two user messages cache the growing
/// conversation: each turn reads the cache entry written at the previous
/// user message and extends it. Two moving breakpoints (not one) so the
/// previous turn's boundary stays explicitly marked however many blocks
/// the last turn appended. 3 breakpoints total, under the API's limit of 4.
///
/// Prompts below the model's minimum cacheable length just ignore
/// cache_control, so this is safe to apply unconditionally.
fn add_cache_breakpoints(body: &mut serde_json::Value) {
    if let Some(blocks) = body["system"].as_array_mut() {
        if let Some(last) = blocks.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
    }

    if let Some(messages) = body["messages"].as_array_mut() {
        let last_two_user: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m["role"] == "user")
            .map(|(i, _)| i)
            .rev()
            .take(2)
            .collect();

        for idx in last_two_user {
            mark_last_block(&mut messages[idx]);
        }
    }
}

/// Set cache_control on the final content block of a message. Plain-string
/// content is lifted into an equivalent single text block, since
/// cache_control can only live on blocks.
fn mark_last_block(message: &mut serde_json::Value) {
    let content = &mut message["content"];

    if let Some(text) = content.as_str() {
        *content = json!([{
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }]);
        return;
    }

    if let Some(blocks) = content.as_array_mut() {
        if let Some(last) = blocks.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{ContentBlock, Message};

    fn body_with(messages: Vec<Message>) -> serde_json::Value {
        json!({
            "model": "claude-test",
            "system": [
                {"type": "text", "text": "block a"},
                {"type": "text", "text": "block b"},
            ],
            "messages": messages,
        })
    }

    #[test]
    fn marks_last_system_block_only() {
        let mut body = body_with(vec![Message::user("hi")]);
        add_cache_breakpoints(&mut body);

        let system = body["system"].as_array().unwrap();
        assert!(system[0].get("cache_control").is_none());
        assert_eq!(system[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn lifts_string_content_to_cached_text_block() {
        let mut body = body_with(vec![Message::user("hello")]);
        add_cache_breakpoints(&mut body);

        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hello");
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn marks_last_two_user_messages_not_assistant() {
        let mut body = body_with(vec![
            Message::user("first"),
            Message::assistant_text("reply one"),
            Message::user("second"),
            Message::assistant_text("reply two"),
            Message::user("third"),
        ]);
        add_cache_breakpoints(&mut body);

        let messages = body["messages"].as_array().unwrap();
        // Oldest user message: no breakpoint
        assert!(messages[0]["content"].is_string());
        // Assistant messages untouched
        assert!(messages[1]["content"].is_string());
        assert!(messages[3]["content"].is_string());
        // Last two user messages: marked
        assert_eq!(
            messages[2]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(
            messages[4]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn marks_final_block_of_tool_results_message() {
        // Tool-result turns are user-role messages with block content; the
        // breakpoint must land on the last block, and tool_result blocks
        // accept cache_control.
        let mut body = body_with(vec![Message::tool_results(vec![
            ContentBlock::ToolResult {
                tool_use_id: "tu_1".to_string(),
                content: "one".to_string(),
                is_error: None,
            },
            ContentBlock::ToolResult {
                tool_use_id: "tu_2".to_string(),
                content: "two".to_string(),
                is_error: None,
            },
        ])]);
        add_cache_breakpoints(&mut body);

        let content = body["messages"][0]["content"].as_array().unwrap();
        assert!(content[0].get("cache_control").is_none());
        assert_eq!(content[1]["cache_control"]["type"], "ephemeral");
        // The block is otherwise intact
        assert_eq!(content[1]["type"], "tool_result");
        assert_eq!(content[1]["tool_use_id"], "tu_2");
    }

    #[test]
    fn total_breakpoints_never_exceed_api_limit() {
        // Long tool-heavy conversation: still exactly 3 breakpoints
        // (1 system + 2 moving), under the API's limit of 4.
        let mut messages = Vec::new();
        for i in 0..10 {
            messages.push(Message::user(&format!("turn {i}")));
            messages.push(Message::assistant_text(&format!("reply {i}")));
        }
        let mut body = body_with(messages);
        add_cache_breakpoints(&mut body);

        let count = |v: &serde_json::Value| -> usize {
            let mut n = 0;
            if let Some(arr) = v.as_array() {
                for item in arr {
                    if item.get("cache_control").is_some() {
                        n += 1;
                    }
                    if let Some(content) = item["content"].as_array() {
                        n += content
                            .iter()
                            .filter(|b| b.get("cache_control").is_some())
                            .count();
                    }
                }
            }
            n
        };

        let total = count(&body["system"]) + count(&body["messages"]);
        assert_eq!(total, 3);
    }
}
