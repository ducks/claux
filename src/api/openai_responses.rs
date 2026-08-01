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
    /// Count of messages already reflected server-side under `response_id`.
    ///
    /// This is the length of the history that was *actually sent*, never a
    /// prediction of what the turn will append. The engine appends a variable
    /// number of messages per round (an empty response appends none, steering
    /// appends several), so any predicted index desynchronizes from history
    /// and silently slices past unsent messages.
    sent_message_count: usize,
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

    /// Build the request input, continuing from the stored cursor when the
    /// history it describes is still a prefix of `messages`.
    ///
    /// Returns the input items, the `previous_response_id` to continue from,
    /// and the number of messages the request covers.
    ///
    /// Falls back to resending the whole conversation whenever continuation
    /// would be lossy: if history shrank (compaction rewrote it) or if the
    /// delta is empty. An empty delta paired with a live `previous_response_id`
    /// would ask the model to continue with no new input, silently discarding
    /// whatever the user just said.
    fn build_input(&self, messages: &[Message]) -> (Vec<Value>, Option<String>, usize) {
        let cursor = self.cursor.lock().expect("response cursor poisoned");
        if let Some(response_id) = &cursor.response_id {
            if cursor.sent_message_count < messages.len() {
                return (
                    convert_messages(&messages[cursor.sent_message_count..], true),
                    Some(response_id.clone()),
                    messages.len(),
                );
            }
        }
        (convert_messages(messages, false), None, messages.len())
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
        let (input, previous_response_id, sent_message_count) = self.build_input(messages);

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
                sent_message_count,
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
    sent_message_count: usize,
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
                        sent_message_count,
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
                    cache_creation_tokens: usage["input_tokens_details"]["cache_write_tokens"]
                        .as_u64()
                        .unwrap_or(0) as u32,
                    provider_cost_usd: usage["cost"]
                        .as_f64()
                        .filter(|cost| cost.is_finite() && *cost >= 0.0),
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
            sent_message_count: 1,
        };

        provider.reset_session();

        let messages = vec![Message::user("new session")];
        let (input, previous_response_id, _) = provider.build_input(&messages);
        assert!(previous_response_id.is_none());
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["content"], "new session");
    }

    fn provider_for_cursor_tests() -> OpenAIResponsesProvider {
        OpenAIResponsesProvider::new(
            "https://api.openai.com/v1",
            "key",
            "gpt-5.6-sol",
            "openai",
            None,
        )
    }

    /// Commit a cursor the way a completed response does: the request covered
    /// exactly `sent_message_count` messages.
    fn commit_cursor(provider: &OpenAIResponsesProvider, id: &str, sent_message_count: usize) {
        *provider.cursor.lock().unwrap() = ResponseCursor {
            response_id: Some(id.to_string()),
            sent_message_count,
        };
    }

    /// Drive two real `stream()` calls against a loopback server and return
    /// the request bodies. This exercises the full accounting path — cursor
    /// commit on `response.completed`, then the next request's slice — rather
    /// than poking at `build_input` with a hand-written cursor.
    async fn two_round_requests(first: &[Message], second: &[Message]) -> Vec<serde_json::Value> {
        let completed = |id: &str| {
            format!(
                "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"{id}\",\"usage\":{{}}}}}}\n\n"
            )
        };
        let server = crate::test_support::RecordingSseServer::start(vec![
            completed("resp_1"),
            completed("resp_2"),
        ])
        .await;
        let provider = OpenAIResponsesProvider::new(&server.base_url, "key", "m", "openai", None);

        for messages in [first, second] {
            let mut stream = provider
                .stream(messages, "system", &[], 1_000, CancellationToken::new())
                .await
                .unwrap();
            while stream.recv().await.is_some() {}
        }

        server.requests()
    }

    #[tokio::test]
    async fn a_turn_appending_no_assistant_message_does_not_drop_the_next_user_message() {
        // Regression: the cursor was derived from `messages.len() + 1`, which
        // assumed every round appends exactly one assistant message. An empty
        // model response appends none, so the next user message landed at the
        // index the cursor had already claimed and was sliced away — the
        // request went out with a live previous_response_id and an empty
        // input, silently discarding what the user typed.
        let requests = two_round_requests(
            &[Message::user("first")],
            &[Message::user("first"), Message::user("second")],
        )
        .await;

        let second = &requests[1];
        assert_eq!(second["previous_response_id"], "resp_1");
        let input = second["input"].as_array().unwrap();
        assert!(
            !input.is_empty(),
            "the user message must not be dropped: {second}"
        );
        assert_eq!(input[0]["content"], "second");
    }

    #[tokio::test]
    async fn steering_messages_appended_mid_turn_are_all_sent() {
        // Steering injects a variable number of user messages between rounds.
        // A fixed +1 stride skipped every message past the first.
        let requests = two_round_requests(
            &[Message::user("first")],
            &[
                Message::user("first"),
                Message::assistant_text("working on it"),
                Message::user("steer one"),
                Message::user("steer two"),
            ],
        )
        .await;

        let second = &requests[1];
        assert_eq!(second["previous_response_id"], "resp_1");
        let contents: Vec<&str> = second["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item["content"].as_str())
            .collect();
        assert_eq!(contents, ["working on it", "steer one", "steer two"]);
    }

    #[tokio::test]
    async fn a_tool_round_continues_from_the_previous_response() {
        // The case the cursor exists for: the assistant tool_use is already
        // server-side, so only the tool result should be sent.
        let requests = two_round_requests(
            &[Message::user("first")],
            &[
                Message::user("first"),
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
            ],
        )
        .await;

        let second = &requests[1];
        assert_eq!(second["previous_response_id"], "resp_1");
        let input = second["input"].as_array().unwrap();
        assert_eq!(input.len(), 1, "only the tool result is new: {second}");
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["call_id"], "call_1");
    }

    #[test]
    fn an_empty_delta_resends_the_conversation_instead_of_continuing_blindly() {
        // If history did not grow, continuing from previous_response_id would
        // send an empty input. Resend in full rather than ask the model to
        // continue from nothing.
        let provider = provider_for_cursor_tests();
        let messages = vec![Message::user("first")];
        let (_, _, sent) = provider.build_input(&messages);
        commit_cursor(&provider, "resp_1", sent);

        let (input, previous_response_id, _) = provider.build_input(&messages);

        assert!(previous_response_id.is_none());
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["content"], "first");
    }

    #[test]
    fn compaction_shrinking_history_falls_back_to_a_full_resend() {
        // Compaction rewrites history to something shorter. The stored cursor
        // then describes messages that no longer exist, so continuation would
        // be meaningless. (The engine also calls reset_session here; this
        // guards the provider independently.)
        let provider = provider_for_cursor_tests();
        commit_cursor(&provider, "resp_1", 12);

        let messages = vec![
            Message::user("Here is a summary of our conversation so far:"),
            Message::assistant_text("summary body"),
        ];
        let (input, previous_response_id, _) = provider.build_input(&messages);

        assert!(previous_response_id.is_none());
        assert_eq!(input.len(), 2);
        assert_eq!(
            input[0]["content"],
            "Here is a summary of our conversation so far:"
        );
    }

    #[tokio::test]
    async fn a_completed_response_records_the_count_of_messages_actually_sent() {
        let provider = provider_for_cursor_tests();
        let response = crate::test_support::sse_response(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_7\",\"usage\":{}}}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_responses_sse(
            response,
            tx,
            CancellationToken::new(),
            provider.cursor.clone(),
            3,
            "openai",
            "gpt-5.6-sol",
        )
        .await
        .unwrap();

        while rx.recv().await.is_some() {}
        let cursor = provider.cursor.lock().unwrap();
        assert_eq!(cursor.response_id.as_deref(), Some("resp_7"));
        assert_eq!(
            cursor.sent_message_count, 3,
            "the cursor must record what was sent, not a prediction"
        );
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
                        "input_tokens_details": {
                            "cached_tokens": 3,
                            "cache_write_tokens": 2
                        },
                        "cost": 0.0042
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
                cache_creation_tokens: 2,
                provider_cost_usd: Some(cost),
            }) if (*cost - 0.0042).abs() < f64::EPSILON
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
