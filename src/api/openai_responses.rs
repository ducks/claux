use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt as _;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::provider::{Provider, ProviderStream};
use super::stream::ApiEvent;
use super::types::{ContentBlock, Message, MessageContent, ToolDefinition, Usage};

#[derive(Default)]
struct ResponseCursor {
    response_id: Option<String>,
    next_message_index: usize,
}

/// Native OpenAI Responses API provider.
///
/// The cursor keeps OpenAI's reasoning items server-side between tool rounds
/// via `previous_response_id`. This is required for reasoning models and avoids
/// trying to squeeze Responses output items into Claux's provider-neutral
/// transcript format.
pub struct OpenAIResponsesProvider {
    api_key: String,
    model: String,
    base_url: String,
    provider_name: String,
    reasoning_effort: Option<String>,
    http: reqwest::Client,
    cursor: Arc<Mutex<ResponseCursor>>,
}

impl OpenAIResponsesProvider {
    pub fn new(
        base_url: &str,
        api_key: &str,
        model: &str,
        name: &str,
        reasoning_effort: Option<&str>,
    ) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            provider_name: name.to_string(),
            reasoning_effort: reasoning_effort.map(str::to_string),
            http: reqwest::Client::new(),
            cursor: Arc::new(Mutex::new(ResponseCursor::default())),
        }
    }

    fn build_input(&self, messages: &[Message]) -> (Vec<Value>, Option<String>) {
        let cursor = self.cursor.lock().expect("response cursor poisoned");
        if let Some(response_id) = &cursor.response_id {
            if cursor.next_message_index <= messages.len() {
                return (
                    convert_messages(&messages[cursor.next_message_index..], true),
                    Some(response_id.clone()),
                );
            }
        }
        (convert_messages(messages, false), None)
    }
}

#[async_trait]
impl Provider for OpenAIResponsesProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
        self.reset_session();
    }

    fn reset_session(&mut self) {
        *self.cursor.lock().expect("response cursor poisoned") = ResponseCursor::default();
    }

    async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        max_tokens: u32,
        cancel: CancellationToken,
    ) -> Result<ProviderStream> {
        let (tx, rx) = mpsc::channel(256);
        let (input, previous_response_id) = self.build_input(messages);
        let next_message_index = messages.len().saturating_add(1);

        let mut body = json!({
            "model": self.model,
            "instructions": system,
            "input": input,
            "max_output_tokens": max_tokens,
            "stream": true,
            "store": true,
        });
        if let Some(previous_response_id) = previous_response_id {
            body["previous_response_id"] = json!(previous_response_id);
        }
        if !tools.is_empty() {
            body["tools"] = json!(convert_tools(tools));
        }
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        }

        let url = format!("{}/responses", self.base_url);
        let mut request = self
            .http
            .post(&url)
            .header("content-type", "application/json");
        if !self.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.api_key));
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => anyhow::bail!("API request cancelled"),
            result = tokio::time::timeout(
                std::time::Duration::from_secs(60),
                request.json(&body).send(),
            ) => result
                .map_err(|_| anyhow::anyhow!("API request timed out waiting for response headers"))??,
        };

        if !response.status().is_success() {
            return Err(super::error::http_error(response, &self.provider_name, &self.model).await);
        }

        let stream_cancel = cancel.child_token();
        let reader_cancel = stream_cancel.clone();
        let error_tx = tx.clone();
        let cursor = self.cursor.clone();
        let provider_name = self.provider_name.clone();
        let model = self.model.clone();
        tokio::spawn(async move {
            if let Err(error) = read_responses_sse(
                response,
                tx,
                reader_cancel,
                cursor,
                next_message_index,
                &provider_name,
                &model,
            )
            .await
            {
                let message = format!("OpenAI Responses stream error: {error}");
                tracing::error!("{message}");
                let _ = error_tx.send(ApiEvent::Error(message)).await;
            }
        });

        Ok(ProviderStream::new(rx, stream_cancel))
    }
}

fn convert_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": false,
            })
        })
        .collect()
}

fn convert_messages(messages: &[Message], continuation: bool) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        match &message.content {
            MessageContent::Text(text) => {
                input.push(json!({ "role": message.role, "content": text }));
            }
            MessageContent::Blocks(blocks) => {
                let text = blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    input.push(json!({ "role": message.role, "content": text }));
                }

                for block in blocks {
                    match block {
                        ContentBlock::ToolUse {
                            name,
                            input: arguments,
                            ..
                        } if !continuation => {
                            input.push(json!({
                                "role": "assistant",
                                "content": format!(
                                    "[Tool call: {name}({})]",
                                    serde_json::to_string(arguments).unwrap_or_default()
                                ),
                            }));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } if continuation => {
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": content,
                            }));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            input.push(json!({
                                "role": "user",
                                "content": format!("[Tool result for {tool_use_id}: {content}]"),
                            }));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    input
}

async fn read_responses_sse(
    response: reqwest::Response,
    tx: mpsc::Sender<ApiEvent>,
    cancel: CancellationToken,
    cursor: Arc<Mutex<ResponseCursor>>,
    next_message_index: usize,
    provider: &str,
    model: &str,
) -> Result<()> {
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut completed = false;

    loop {
        let chunk = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            chunk = stream.next() => chunk,
        };
        let Some(chunk) = chunk else {
            break;
        };
        buffer.extend_from_slice(&chunk?);

        while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = buffer.drain(..=newline).collect();
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.strip_prefix("data:").map(str::trim_start) else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let event: Value = match serde_json::from_str(data) {
                Ok(event) => event,
                Err(_) => continue,
            };

            if event["type"] == "response.completed" {
                let response = &event["response"];
                if let Some(response_id) = response["id"].as_str() {
                    *cursor.lock().expect("response cursor poisoned") = ResponseCursor {
                        response_id: Some(response_id.to_string()),
                        next_message_index,
                    };
                }
                completed = true;
            }

            let terminal_error = matches!(
                event["type"].as_str(),
                Some("response.failed" | "response.incomplete" | "error" | "response.error")
            );
            for api_event in translate_event(&event, provider, model) {
                if tx.send(api_event).await.is_err() {
                    return Ok(());
                }
            }
            if terminal_error {
                return Ok(());
            }
        }
    }

    if completed {
        Ok(())
    } else {
        anyhow::bail!("stream ended before response.completed")
    }
}

fn translate_event(event: &Value, provider: &str, model: &str) -> Vec<ApiEvent> {
    match event["type"].as_str().unwrap_or_default() {
        "response.output_text.delta" => event["delta"]
            .as_str()
            .map(|text| vec![ApiEvent::Text(text.to_string())])
            .unwrap_or_default(),
        "response.output_item.done" if event["item"]["type"] == "function_call" => {
            let item = &event["item"];
            let Some(call_id) = item["call_id"].as_str() else {
                return vec![ApiEvent::Error(
                    "OpenAI function call omitted call_id".to_string(),
                )];
            };
            let Some(name) = item["name"].as_str() else {
                return vec![ApiEvent::Error(
                    "OpenAI function call omitted name".to_string(),
                )];
            };
            match item["arguments"]
                .as_str()
                .and_then(|arguments| serde_json::from_str(arguments).ok())
            {
                Some(input) => vec![ApiEvent::ToolUse {
                    id: call_id.to_string(),
                    name: name.to_string(),
                    input,
                }],
                None => vec![ApiEvent::Error(format!(
                    "OpenAI function {name} returned invalid arguments"
                ))],
            }
        }
        "response.completed" => {
            let usage = &event["response"]["usage"];
            vec![
                ApiEvent::Usage(Usage {
                    input_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as u32,
                    output_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as u32,
                    cache_read_tokens: usage["input_tokens_details"]["cached_tokens"]
                        .as_u64()
                        .unwrap_or(0) as u32,
                    cache_creation_tokens: 0,
                }),
                ApiEvent::Done,
            ]
        }
        "response.failed" | "response.incomplete" => {
            vec![ApiEvent::Error(super::error::stream_error(
                event, provider, model,
            ))]
        }
        "error" | "response.error" => vec![ApiEvent::Error(super::error::stream_error(
            event, provider, model,
        ))],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_tools_to_responses_shape() {
        let tools = convert_tools(&[ToolDefinition {
            name: "Read".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
        }]);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "Read");
        assert!(tools[0].get("function").is_none());
    }

    #[test]
    fn continuation_sends_only_function_outputs() {
        let messages = vec![
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "Read".to_string(),
                input: json!({"file_path": "/tmp/a"}),
            }]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "contents".to_string(),
                is_error: None,
            }]),
        ];
        let input = convert_messages(&messages, true);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["call_id"], "call_1");
    }

    #[test]
    fn resetting_session_drops_previous_response_cursor() {
        let mut provider = OpenAIResponsesProvider::new(
            "https://api.openai.com/v1",
            "key",
            "gpt-5.6-sol",
            "openai",
            None,
        );
        *provider.cursor.lock().unwrap() = ResponseCursor {
            response_id: Some("resp_previous_session".to_string()),
            next_message_index: 1,
        };

        provider.reset_session();

        let messages = vec![Message::user("new session")];
        let (input, previous_response_id) = provider.build_input(&messages);
        assert!(previous_response_id.is_none());
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["content"], "new session");
    }

    #[test]
    fn translates_text_tool_and_usage_events() {
        let text = translate_event(
            &json!({
                "type": "response.output_text.delta",
                "delta": "hello"
            }),
            "openai",
            "model",
        );
        assert!(matches!(&text[0], ApiEvent::Text(value) if value == "hello"));

        let tool = translate_event(
            &json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_7",
                    "name": "Read",
                    "arguments": "{\"file_path\":\"/tmp/a\"}"
                }
            }),
            "openai",
            "model",
        );
        assert!(matches!(
            &tool[0],
            ApiEvent::ToolUse { id, name, .. } if id == "call_7" && name == "Read"
        ));

        let completed = translate_event(
            &json!({
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 3}
                    }
                }
            }),
            "openai",
            "model",
        );
        assert!(matches!(
            &completed[0],
            ApiEvent::Usage(Usage {
                input_tokens: 12,
                output_tokens: 4,
                cache_read_tokens: 3,
                ..
            })
        ));
        assert!(matches!(completed[1], ApiEvent::Done));

        let failed = translate_event(
            &json!({
                "type": "response.failed",
                "response": {
                    "error": {"code": "server_error", "message": "Rate limited"},
                    "error_type": "rate_limit_exceeded"
                }
            }),
            "openrouter",
            "moonshotai/kimi-k3",
        );
        assert!(matches!(
            &failed[0],
            ApiEvent::Error(error)
                if error.contains("429 Too Many Requests")
                    && error.contains("moonshotai/kimi-k3")
        ));
    }

    #[tokio::test]
    async fn terminal_failure_emits_one_actionable_error() {
        let response = crate::test_support::sse_response(
            "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"upstream limit\"}}}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_responses_sse(
            response,
            tx,
            CancellationToken::new(),
            Arc::new(Mutex::new(ResponseCursor::default())),
            0,
            "openrouter",
            "deepseek/deepseek-r1",
        )
        .await
        .unwrap();

        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::Error(error))
                if error.contains("429 Too Many Requests")
                    && error.contains("deepseek/deepseek-r1")
        ));
        assert!(rx.recv().await.is_none());
    }
}
