use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

use super::provider::Provider;
use super::stream::ApiEvent;
use super::types::{Message, MessageContent, ToolDefinition, Usage};

/// OpenAI-compatible API provider.
/// Works with Ollama, vLLM, LMStudio, OpenAI, and anything that speaks
/// the /v1/chat/completions streaming format.
pub struct OpenAICompatProvider {
    api_key: String,
    model: String,
    base_url: String,
    provider_name: String,
    http: reqwest::Client,
}

impl OpenAICompatProvider {
    pub fn new(base_url: &str, api_key: &str, model: &str, name: &str) -> Self {
        // Strip trailing slash
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url,
            provider_name: name.to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Convert our message format to OpenAI's format.
    fn convert_messages(messages: &[Message], system: &str) -> Vec<serde_json::Value> {
        let mut out = vec![json!({
            "role": "system",
            "content": system,
        })];

        for msg in messages {
            match &msg.content {
                MessageContent::Text(text) => {
                    out.push(json!({
                        "role": msg.role,
                        "content": text,
                    }));
                }
                MessageContent::Blocks(blocks) => {
                    // Flatten blocks into OpenAI format
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();
                    let mut tool_results = Vec::new();

                    for block in blocks {
                        match block {
                            super::types::ContentBlock::Text { text } => {
                                text_parts.push(text.clone());
                            }
                            super::types::ContentBlock::ToolUse { id, name, input } => {
                                tool_calls.push(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": serde_json::to_string(input).unwrap_or_default(),
                                    }
                                }));
                            }
                            super::types::ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } => {
                                tool_results.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": content,
                                }));
                            }
                        }
                    }

                    if !tool_calls.is_empty() {
                        let mut assistant_msg = json!({
                            "role": "assistant",
                        });
                        if !text_parts.is_empty() {
                            assistant_msg["content"] = json!(text_parts.join("\n"));
                        }
                        assistant_msg["tool_calls"] = json!(tool_calls);
                        out.push(assistant_msg);
                    } else if !tool_results.is_empty() {
                        for result in tool_results {
                            out.push(result);
                        }
                    } else if !text_parts.is_empty() {
                        out.push(json!({
                            "role": msg.role,
                            "content": text_parts.join("\n"),
                        }));
                    }
                }
            }
        }

        out
    }

    /// Convert our tool definitions to OpenAI function format.
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect()
    }
}

#[async_trait]
impl Provider for OpenAICompatProvider {
    fn name(&self) -> &str {
        &self.provider_name
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

        let url = format!("{}/chat/completions", self.base_url);
        let openai_messages = Self::convert_messages(messages, system);

        let mut body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": openai_messages,
            "stream": true,
        });

        if !tools.is_empty() {
            body["tools"] = json!(Self::convert_tools(tools));
        }

        let mut request = self
            .http
            .post(&url)
            .header("content-type", "application/json");

        if !self.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.api_key));
        }

        let response = request.json(&body).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error ({}): {}", status, error_text);
        }

        tokio::spawn(async move {
            if let Err(e) = read_openai_sse(response, tx).await {
                tracing::error!("OpenAI SSE stream error: {}", e);
            }
        });

        Ok(rx)
    }
}

/// Parse OpenAI-format SSE stream into ApiEvents.
async fn read_openai_sse(
    response: reqwest::Response,
    tx: mpsc::Sender<ApiEvent>,
) -> Result<()> {
    use futures_util::StreamExt as _;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    // Tool call accumulation
    let mut tool_calls: std::collections::HashMap<u32, (String, String, String)> =
        std::collections::HashMap::new(); // index -> (id, name, arguments)

    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };

            if data == "[DONE]" {
                let _ = tx
                    .send(ApiEvent::Usage(Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                    }))
                    .await;
                let _ = tx.send(ApiEvent::Done).await;
                return Ok(());
            }

            let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };

            // Check for usage in the chunk
            if let Some(usage) = event.get("usage") {
                input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(input_tokens as u64) as u32;
                output_tokens =
                    usage["completion_tokens"].as_u64().unwrap_or(output_tokens as u64) as u32;
            }

            let Some(choices) = event.get("choices").and_then(|c| c.as_array()) else {
                continue;
            };

            for choice in choices {
                let Some(delta) = choice.get("delta") else {
                    continue;
                };

                // Text content
                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        let _ = tx.send(ApiEvent::Text(content.to_string())).await;
                    }
                }

                // Tool calls
                if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tcs {
                        let index = tc["index"].as_u64().unwrap_or(0) as u32;

                        let entry = tool_calls.entry(index).or_insert_with(|| {
                            (String::new(), String::new(), String::new())
                        });

                        if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                            entry.0 = id.to_string();
                        }
                        if let Some(func) = tc.get("function") {
                            if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                entry.1 = name.to_string();
                            }
                            if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                                entry.2.push_str(args);
                            }
                        }
                    }
                }

                // Check finish reason
                if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                    if reason == "tool_calls" {
                        // Emit accumulated tool calls
                        let mut calls: Vec<(u32, (String, String, String))> =
                            tool_calls.drain().collect();
                        calls.sort_by_key(|(idx, _)| *idx);

                        for (_, (id, name, args)) in calls {
                            if let Ok(input) = serde_json::from_str(&args) {
                                let _ = tx
                                    .send(ApiEvent::ToolUse {
                                        id,
                                        name,
                                        input,
                                    })
                                    .await;
                            }
                        }
                    }
                }
            }
        }
    }

    // Stream ended
    // Emit any remaining tool calls
    if !tool_calls.is_empty() {
        let mut calls: Vec<(u32, (String, String, String))> = tool_calls.drain().collect();
        calls.sort_by_key(|(idx, _)| *idx);
        for (_, (id, name, args)) in calls {
            if let Ok(input) = serde_json::from_str(&args) {
                let _ = tx
                    .send(ApiEvent::ToolUse {
                        id,
                        name,
                        input,
                    })
                    .await;
            }
        }
    }

    let _ = tx.send(ApiEvent::Done).await;
    Ok(())
}
