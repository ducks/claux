use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::VecDeque;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::error::ApiFailure;
use super::provider::{Provider, ProviderStream};
use super::stream::{ApiEvent, Utf8LineDecoder};
use super::types::{ContentBlock, Message, MessageContent, ToolDefinition, Usage};

/// OpenAI-compatible API provider.
/// Works with Ollama, vLLM, LMStudio, OpenAI, and anything that speaks
/// the /v1/chat/completions streaming format.
pub struct OpenAICompatProvider {
    api_key: String,
    model: String,
    base_url: String,
    provider_name: String,
    reasoning_effort: Option<String>,
    prompt_caching: bool,
    allow_eof_without_finish_reason: bool,
    http: reqwest::Client,
}

impl OpenAICompatProvider {
    pub fn new(
        base_url: &str,
        api_key: &str,
        model: &str,
        name: &str,
        reasoning_effort: Option<&str>,
    ) -> Self {
        // Strip trailing slash
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url,
            provider_name: name.to_string(),
            reasoning_effort: reasoning_effort.map(str::to_string),
            prompt_caching: false,
            allow_eof_without_finish_reason: false,
            http: reqwest::Client::new(),
        }
    }

    pub fn with_prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = enabled;
        self
    }

    pub fn with_eof_without_finish_reason(mut self, enabled: bool) -> Self {
        self.allow_eof_without_finish_reason = enabled;
        self
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
                    let mut image_parts = Vec::new();
                    let mut tool_calls = Vec::new();
                    let mut tool_results = Vec::new();
                    let mut reasoning_text = String::new();
                    let mut reasoning_details = Vec::new();

                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                text_parts.push(text.clone());
                            }
                            ContentBlock::Image { source } => {
                                image_parts.push(json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!(
                                            "data:{};base64,{}",
                                            source.media_type, source.data
                                        )
                                    }
                                }));
                            }
                            ContentBlock::Reasoning { text, details } => {
                                if let Some(text) = text {
                                    reasoning_text.push_str(text);
                                }
                                reasoning_details.extend(details.iter().cloned());
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                tool_calls.push(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": serde_json::to_string(input).unwrap_or_default(),
                                    }
                                }));
                            }
                            ContentBlock::ToolResult {
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
                        if !reasoning_details.is_empty() {
                            assistant_msg["reasoning_details"] = json!(reasoning_details);
                        } else if !reasoning_text.is_empty() {
                            assistant_msg["reasoning"] = json!(reasoning_text);
                        }
                        assistant_msg["tool_calls"] = json!(tool_calls);
                        out.push(assistant_msg);
                    } else if !tool_results.is_empty() {
                        for result in tool_results {
                            out.push(result);
                        }
                    } else if !text_parts.is_empty() || !image_parts.is_empty() {
                        if image_parts.is_empty() {
                            out.push(json!({
                                "role": msg.role,
                                "content": text_parts.join("\n"),
                            }));
                        } else {
                            let mut content = text_parts
                                .into_iter()
                                .map(|text| json!({"type": "text", "text": text}))
                                .collect::<Vec<_>>();
                            content.extend(image_parts);
                            out.push(json!({
                                "role": msg.role,
                                "content": content,
                            }));
                        }
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

    fn request_body(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        max_tokens: u32,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": Self::convert_messages(messages, system),
            "stream": true,
            "stream_options": {
                "include_usage": true
            }
        });

        if !tools.is_empty() {
            body["tools"] = json!(Self::convert_tools(tools));
        }
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        }
        if self.prompt_caching {
            body["cache_control"] = json!({ "type": "ephemeral" });
        }

        body
    }
}

#[async_trait]
impl Provider for OpenAICompatProvider {
    fn name(&self) -> &str {
        &self.provider_name
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
        cancel: CancellationToken,
    ) -> Result<ProviderStream> {
        let (tx, rx) = mpsc::channel(256);

        let url = format!("{}/chat/completions", self.base_url);
        let body = self.request_body(messages, system, tools, max_tokens);

        tracing::debug!("OpenAI request: {} model={}", url, self.model);
        tracing::debug!(
            "API key present: {}, len: {}",
            !self.api_key.is_empty(),
            self.api_key.len()
        );

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
        let provider_name = self.provider_name.clone();
        let model = self.model.clone();
        let allow_eof_without_finish_reason = self.allow_eof_without_finish_reason;
        tokio::spawn(async move {
            if let Err(e) = read_openai_sse(
                response,
                tx,
                reader_cancel,
                &provider_name,
                &model,
                allow_eof_without_finish_reason,
            )
            .await
            {
                // Classify before wrapping: the prefix is for display, and
                // must not erase what the failure was.
                let failure =
                    super::error::classify_reader_error(&e).prefixed("OpenAI SSE stream error");
                tracing::error!("{failure}");
                let _ = error_tx.send(ApiEvent::Error(failure)).await;
            }
        });

        Ok(ProviderStream::new(rx, stream_cancel))
    }
}

type PendingToolCalls = std::collections::HashMap<u32, (String, String, String)>;

/// Merge a streamed fragment into the reasoning detail immediately before it.
///
/// OpenRouter streams reasoning detail payloads in pieces. Gemini in particular
/// can emit several `reasoning.text` objects with the same index before the
/// matching `reasoning.encrypted` signature. Replaying those text fragments as
/// separate blocks changes the signed sequence and Gemini rejects the next
/// request with "Corrupted thought signature." Only adjacent fragments with a
/// compatible identity are joined; distinct detail blocks retain their order
/// and representation.
fn merge_reasoning_detail_fragment(
    current: &mut serde_json::Value,
    incoming: &serde_json::Value,
) -> bool {
    let Some(current_object) = current.as_object_mut() else {
        return false;
    };
    let Some(incoming_object) = incoming.as_object() else {
        return false;
    };

    let Some(detail_type) = current_object.get("type").and_then(|value| value.as_str()) else {
        return false;
    };
    if incoming_object.get("type").and_then(|value| value.as_str()) != Some(detail_type) {
        return false;
    }

    let payload_field = match detail_type {
        "reasoning.text" => "text",
        "reasoning.encrypted" => "data",
        "reasoning.summary" => "summary",
        _ => return false,
    };

    for (field, incoming_value) in incoming_object {
        if field == payload_field {
            continue;
        }
        if current_object.get(field).is_some_and(|current_value| {
            !current_value.is_null() && !incoming_value.is_null() && current_value != incoming_value
        }) {
            return false;
        }
    }

    match (
        current_object
            .get(payload_field)
            .and_then(|value| value.as_str()),
        incoming_object
            .get(payload_field)
            .and_then(|value| value.as_str()),
    ) {
        (Some(current_payload), Some(incoming_payload)) => {
            let mut joined = String::with_capacity(current_payload.len() + incoming_payload.len());
            joined.push_str(current_payload);
            joined.push_str(incoming_payload);
            current_object.insert(payload_field.to_string(), serde_json::Value::String(joined));
        }
        (None, Some(_)) => {
            current_object.insert(
                payload_field.to_string(),
                incoming_object[payload_field].clone(),
            );
        }
        _ => {}
    }

    for (field, value) in incoming_object {
        if field == payload_field {
            continue;
        }
        if current_object
            .get(field)
            .is_none_or(serde_json::Value::is_null)
        {
            current_object.insert(field.clone(), value.clone());
        }
    }

    true
}

fn append_reasoning_detail_chunk(
    accumulated: &mut Vec<serde_json::Value>,
    incoming: Vec<serde_json::Value>,
) {
    for fragment in incoming {
        let merged = accumulated
            .last_mut()
            .is_some_and(|current| merge_reasoning_detail_fragment(current, &fragment));
        if !merged {
            accumulated.push(fragment);
        }
    }
}

fn take_reasoning_event(
    text: &mut String,
    details: &mut Vec<serde_json::Value>,
) -> Option<ApiEvent> {
    if text.is_empty() && details.is_empty() {
        return None;
    }

    Some(ApiEvent::Reasoning {
        text: (!text.is_empty()).then(|| std::mem::take(text)),
        details: std::mem::take(details),
    })
}

fn drain_tool_calls(tool_calls: &mut PendingToolCalls) -> Result<Vec<ApiEvent>> {
    let mut calls: Vec<(u32, (String, String, String))> = tool_calls.drain().collect();
    calls.sort_by_key(|(index, _)| *index);
    calls
        .into_iter()
        .map(|(_, (id, name, arguments))| {
            let input = serde_json::from_str(&arguments).map_err(|error| {
                anyhow::Error::new(ApiFailure::malformed_tool_arguments(format!(
                    "invalid arguments for tool call {name} ({id}): {error}"
                )))
            })?;
            Ok(ApiEvent::ToolUse { id, name, input })
        })
        .collect()
}

/// Parse OpenAI-format SSE stream into ApiEvents.
async fn read_openai_sse(
    response: reqwest::Response,
    tx: mpsc::Sender<ApiEvent>,
    cancel: CancellationToken,
    provider: &str,
    model: &str,
    allow_eof_without_finish_reason: bool,
) -> Result<()> {
    let mut diagnostics = OpenAIStreamDiagnostics::from_response(&response);
    let result = read_openai_sse_body(
        response,
        tx,
        cancel,
        provider,
        model,
        allow_eof_without_finish_reason,
        &mut diagnostics,
    )
    .await;
    result.map_err(|error| {
        let diagnostics = diagnostics.render();
        match error.downcast::<ApiFailure>() {
            Ok(failure) => anyhow::Error::new(ApiFailure::new(
                failure.kind,
                format!("{}; {diagnostics}", failure.message),
            )),
            Err(error) => anyhow::anyhow!("{error}; {diagnostics}"),
        }
    })
}

const MAX_DIAGNOSTIC_FRAMES: usize = 4;
const MAX_DIAGNOSTIC_FRAME_BYTES: usize = 2_048;

#[derive(Debug)]
struct OpenAIStreamDiagnostics {
    status: reqwest::StatusCode,
    headers: Vec<(String, String)>,
    final_frames: VecDeque<String>,
    partial_tail: Option<String>,
}

impl OpenAIStreamDiagnostics {
    fn from_response(response: &reqwest::Response) -> Self {
        const SAFE_HEADERS: &[&str] = &[
            "content-type",
            "server",
            "x-request-id",
            "request-id",
            "cf-ray",
        ];
        let headers = SAFE_HEADERS
            .iter()
            .filter_map(|name| {
                response
                    .headers()
                    .get(*name)
                    .and_then(|value| value.to_str().ok())
                    .map(|value| ((*name).to_string(), value.to_string()))
            })
            .collect();
        Self {
            status: response.status(),
            headers,
            final_frames: VecDeque::new(),
            partial_tail: None,
        }
    }

    fn record_frame(&mut self, data: &str) {
        let frame = crate::utils::truncate_str(data, MAX_DIAGNOSTIC_FRAME_BYTES).to_string();
        if self.final_frames.len() == MAX_DIAGNOSTIC_FRAMES {
            self.final_frames.pop_front();
        }
        self.final_frames.push_back(frame);
    }

    fn record_partial_tail(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let tail = String::from_utf8_lossy(bytes);
        self.partial_tail =
            Some(crate::utils::truncate_str(&tail, MAX_DIAGNOSTIC_FRAME_BYTES).to_string());
    }

    fn render(&self) -> String {
        serde_json::json!({
            "status": self.status.as_u16(),
            "headers": self.headers.iter().cloned().collect::<std::collections::BTreeMap<_, _>>(),
            "final_sse_frames": self.final_frames,
            "partial_tail": self.partial_tail,
        })
        .to_string()
    }
}

async fn read_openai_sse_body(
    response: reqwest::Response,
    tx: mpsc::Sender<ApiEvent>,
    cancel: CancellationToken,
    provider: &str,
    model: &str,
    allow_eof_without_finish_reason: bool,
    diagnostics: &mut OpenAIStreamDiagnostics,
) -> Result<()> {
    use futures_util::StreamExt as _;

    let mut stream = response.bytes_stream();
    let mut lines = Utf8LineDecoder::default();

    // Tool call accumulation
    let mut tool_calls = PendingToolCalls::new(); // index -> (id, name, arguments)

    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;
    let mut cache_read_tokens: u32 = 0;
    let mut cache_creation_tokens: u32 = 0;
    let mut provider_cost_usd: Option<f64> = None;
    let mut saw_finish_reason = false;
    let mut reasoning_text = String::new();
    let mut reasoning_details = Vec::new();

    loop {
        let chunk_result = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            chunk = stream.next() => chunk,
        };
        let Some(chunk_result) = chunk_result else {
            break;
        };
        let chunk = chunk_result?;
        for line in lines.push(&chunk)? {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            diagnostics.record_frame(data);

            if data == "[DONE]" {
                if let Some(event) =
                    take_reasoning_event(&mut reasoning_text, &mut reasoning_details)
                {
                    let _ = tx.send(event).await;
                }
                for event in drain_tool_calls(&mut tool_calls)? {
                    let _ = tx.send(event).await;
                }
                let _ = tx
                    .send(ApiEvent::Usage(Usage::from_openai_totals(
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_creation_tokens,
                        provider_cost_usd,
                    )))
                    .await;
                let _ = tx.send(ApiEvent::Done).await;
                return Ok(());
            }

            let event = serde_json::from_str::<serde_json::Value>(data)
                .map_err(|error| anyhow::anyhow!("invalid JSON in OpenAI SSE event: {error}"))?;

            if event["error"].is_object() {
                let failure = super::error::stream_error(&event, provider, model);
                let _ = tx.send(ApiEvent::Error(failure)).await;
                return Ok(());
            }

            // Some compatible gateways emit the billed cost as a standalone
            // terminal frame instead of nesting it under `usage`.
            provider_cost_usd = parse_nonnegative_number(event.get("cost")).or(provider_cost_usd);

            // Check for usage in the chunk
            if let Some(usage) = event.get("usage") {
                input_tokens = usage["prompt_tokens"]
                    .as_u64()
                    .unwrap_or(input_tokens as u64) as u32;
                output_tokens = usage["completion_tokens"]
                    .as_u64()
                    .unwrap_or(output_tokens as u64) as u32;
                cache_read_tokens = usage["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(cache_read_tokens as u64) as u32;
                cache_creation_tokens = usage["prompt_tokens_details"]["cache_write_tokens"]
                    .as_u64()
                    .unwrap_or(cache_creation_tokens as u64)
                    as u32;
                provider_cost_usd =
                    parse_nonnegative_number(usage.get("cost")).or(provider_cost_usd);
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

                if let Some(text) = delta
                    .get("reasoning")
                    .or_else(|| delta.get("reasoning_content"))
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.is_empty())
                {
                    reasoning_text.push_str(text);
                }
                let streamed_reasoning_details = delta
                    .get("reasoning_details")
                    .and_then(|value| value.as_array())
                    .cloned()
                    .unwrap_or_default();
                append_reasoning_detail_chunk(&mut reasoning_details, streamed_reasoning_details);

                // Tool calls
                if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tcs {
                        let index = tc["index"].as_u64().unwrap_or(0) as u32;

                        let entry = tool_calls
                            .entry(index)
                            .or_insert_with(|| (String::new(), String::new(), String::new()));

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
                    saw_finish_reason = true;
                    if let Some(event) =
                        take_reasoning_event(&mut reasoning_text, &mut reasoning_details)
                    {
                        let _ = tx.send(event).await;
                    }
                    match reason {
                        "tool_calls" => {
                            for event in drain_tool_calls(&mut tool_calls)? {
                                let _ = tx.send(event).await;
                            }
                        }
                        "stop" => {}
                        "length" => {
                            let _ = tx
                                .send(ApiEvent::Error(ApiFailure::output_limit_exceeded(
                                    "response reached its output token limit",
                                )))
                                .await;
                            return Ok(());
                        }
                        "content_filter" => {
                            let _ = tx
                                .send(ApiEvent::Error(ApiFailure::other(
                                    "response blocked by provider content filter",
                                )))
                                .await;
                            return Ok(());
                        }
                        other => {
                            let _ = tx
                                .send(ApiEvent::Error(ApiFailure::other(format!(
                                    "unsupported OpenAI finish reason: {other}"
                                ))))
                                .await;
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    diagnostics.record_partial_tail(lines.pending_bytes());
    let ended_mid_frame = !lines.pending_bytes().is_empty();
    lines.finish()?;
    if ended_mid_frame {
        anyhow::bail!("stream ended in the middle of an SSE frame");
    }
    if !saw_finish_reason && !allow_eof_without_finish_reason {
        anyhow::bail!("stream ended before a finish reason or [DONE] marker");
    }

    // Some compatible providers close cleanly after finish_reason instead of
    // sending [DONE]. Preserve that behavior, but only after a terminal event.
    if let Some(event) = take_reasoning_event(&mut reasoning_text, &mut reasoning_details) {
        let _ = tx.send(event).await;
    }
    for event in drain_tool_calls(&mut tool_calls)? {
        let _ = tx.send(event).await;
    }

    let _ = tx
        .send(ApiEvent::Usage(Usage::from_openai_totals(
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            provider_cost_usd,
        )))
        .await;
    let _ = tx.send(ApiEvent::Done).await;
    Ok(())
}

fn parse_nonnegative_number(value: Option<&serde_json::Value>) -> Option<f64> {
    let value = value?;
    let number = value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))?;
    (number.is_finite() && number >= 0.0).then_some(number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::ImageSource;

    #[test]
    fn requests_streamed_usage() {
        let provider =
            OpenAICompatProvider::new("https://api.openai.com/v1", "key", "model", "openai", None);
        let body = provider.request_body(&[Message::user("hello")], "system", &[], 1_000);

        assert_eq!(body["stream_options"]["include_usage"], true);
        assert!(body.get("reasoning").is_none());
        assert!(body.get("cache_control").is_none());
    }

    #[test]
    fn converts_image_blocks_to_openai_content_parts() {
        let messages = vec![Message::user_with_images(
            "describe it",
            vec![ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            }],
        )];
        let converted = OpenAICompatProvider::convert_messages(&messages, "system");
        let content = converted[1]["content"].as_array().unwrap();
        assert_eq!(content[0], json!({"type": "text", "text": "describe it"}));
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "data:image/png;base64,aGVsbG8="
        );
    }

    #[test]
    fn enables_provider_prompt_caching_when_configured() {
        let provider = OpenAICompatProvider::new(
            "https://openrouter.ai/api/v1",
            "key",
            "model",
            "openrouter",
            None,
        )
        .with_prompt_caching(true);
        let body = provider.request_body(&[Message::user("hello")], "system", &[], 1_000);

        assert_eq!(body["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn sends_reasoning_effort_and_replays_reasoning_details() {
        let provider = OpenAICompatProvider::new(
            "https://openrouter.ai/api/v1",
            "key",
            "model",
            "openrouter",
            Some("low"),
        );
        let messages = vec![Message::assistant_blocks(vec![
            ContentBlock::Reasoning {
                text: Some("private thought".to_string()),
                details: vec![json!({
                    "type": "reasoning.text",
                    "text": "preserved thought",
                    "index": 0
                })],
            },
            ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                input: json!({"file_path": "/tmp/test"}),
            },
        ])];

        let body = provider.request_body(&messages, "system", &[], 1_000);

        assert_eq!(body["reasoning"]["effort"], "low");
        assert_eq!(
            body["messages"][1]["reasoning_details"][0]["text"],
            "preserved thought"
        );
        assert!(body["messages"][1].get("reasoning").is_none());
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call-1");
    }

    #[tokio::test]
    async fn rejects_eof_before_finish_reason() {
        let response = crate::test_support::sse_response(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        let error = read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openai",
            "model",
            false,
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("before a finish reason"));
        assert!(error.to_string().contains("\"status\":200"));
        assert!(error
            .to_string()
            .contains("\\\"content\\\":\\\"partial\\\""));
        assert!(error.to_string().contains("\"partial_tail\":null"));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(text)) if text == "partial"));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn provider_policy_can_accept_clean_eof_without_finish_reason() {
        let response = crate::test_support::sse_response(
            "data: {\"choices\":[{\"delta\":{\"content\":\"complete\"},\"finish_reason\":null}]}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "compatible",
            "model",
            true,
        )
        .await
        .unwrap();

        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(text)) if text == "complete"));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Usage(_))));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Done)));
    }

    #[tokio::test]
    async fn accepts_clean_eof_after_finish_reason() {
        let response = crate::test_support::sse_response(
            "data: {\"choices\":[{\"delta\":{\"content\":\"complete\"},\"finish_reason\":\"stop\"}]}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openai",
            "model",
            false,
        )
        .await
        .unwrap();

        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(text)) if text == "complete"));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Usage(_))));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Done)));
    }

    #[tokio::test]
    async fn captures_provider_reported_cost_and_cache_usage() {
        let response = crate::test_support::sse_response(concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"complete\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":194,\"completion_tokens\":2,\"cost\":0.00095,\"prompt_tokens_details\":{\"cached_tokens\":40,\"cache_write_tokens\":10}}}\n\n",
            "data: [DONE]\n\n"
        ))
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openrouter",
            "deepseek/deepseek-v4-flash",
            false,
        )
        .await
        .unwrap();

        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(text)) if text == "complete"));
        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::Usage(Usage {
                input_tokens: 144,
                output_tokens: 2,
                cache_read_tokens: 40,
                cache_creation_tokens: 10,
                provider_cost_usd: Some(cost),
            })) if (cost - 0.00095).abs() < f64::EPSILON
        ));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Done)));
    }

    #[tokio::test]
    async fn captures_string_cost_from_standalone_terminal_frame() {
        let response = crate::test_support::sse_response(concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"complete\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}\n\n",
            "data: {\"choices\":[],\"cost\":\"0.001234\"}\n\n",
            "data: [DONE]\n\n"
        ))
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "compatible",
            "model",
            false,
        )
        .await
        .unwrap();

        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(_))));
        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::Usage(Usage {
                provider_cost_usd: Some(cost),
                ..
            })) if (cost - 0.001234).abs() < f64::EPSILON
        ));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Done)));
    }

    #[tokio::test]
    async fn reconstructs_streamed_reasoning_for_later_tool_rounds() {
        let response = crate::test_support::sse_response(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think \"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"carefully\",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        ))
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openrouter",
            "deepseek/deepseek-v4-flash-0731",
            false,
        )
        .await
        .unwrap();

        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::Reasoning { text: Some(text), details })
                if text == "think " && details[0]["text"] == "carefully"
        ));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Usage(_))));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Done)));
    }

    #[tokio::test]
    async fn coalesces_gemini_reasoning_fragments_before_replay() {
        let response = crate::test_support::sse_response(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"first \",\"format\":\"google-gemini-v1\",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"second \",\"format\":\"google-gemini-v1\",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"third\",\"format\":\"google-gemini-v1\",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"data\":\"signed-\",\"id\":\"call_123\",\"format\":\"google-gemini-v1\",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"data\":\"payload\",\"id\":\"call_123\",\"format\":\"google-gemini-v1\",\"index\":0}],\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"file_path\\\":\\\"/tmp/test\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        ))
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openrouter",
            "google/gemini-3.7-flash",
            false,
        )
        .await
        .unwrap();

        let Some(ApiEvent::Reasoning {
            text: None,
            details,
        }) = rx.recv().await
        else {
            panic!("expected reconstructed reasoning event");
        };
        assert_eq!(details.len(), 2);
        assert_eq!(details[0]["type"], "reasoning.text");
        assert_eq!(details[0]["text"], "first second third");
        assert_eq!(details[1]["type"], "reasoning.encrypted");
        assert_eq!(details[1]["data"], "signed-payload");
        assert_eq!(details[1]["id"], "call_123");

        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::ToolUse { id, name, input })
                if id == "call_123" && name == "Read" && input["file_path"] == "/tmp/test"
        ));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Usage(_))));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Done)));

        let provider = OpenAICompatProvider::new(
            "https://openrouter.ai/api/v1",
            "key",
            "google/gemini-3.7-flash",
            "openrouter",
            None,
        );
        let messages = vec![Message::assistant_blocks(vec![
            ContentBlock::Reasoning {
                text: None,
                details,
            },
            ContentBlock::ToolUse {
                id: "call_123".to_string(),
                name: "Read".to_string(),
                input: json!({"file_path": "/tmp/test"}),
            },
        ])];
        let body = provider.request_body(&messages, "system", &[], 1_000);

        assert_eq!(
            body["messages"][1]["reasoning_details"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            body["messages"][1]["reasoning_details"][0]["text"],
            "first second third"
        );
        assert_eq!(
            body["messages"][1]["reasoning_details"][1]["data"],
            "signed-payload"
        );
    }

    #[tokio::test]
    async fn output_length_is_an_error_not_successful_completion() {
        let response = crate::test_support::sse_response(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"length\"}]}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openai",
            "model",
            false,
        )
        .await
        .unwrap();

        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(text)) if text == "partial"));
        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::Error(error))
                if error.kind == crate::api::ApiFailureKind::OutputLimitExceeded
        ));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn malformed_tool_arguments_fail_the_stream() {
        let response = crate::test_support::sse_response(
            concat!(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\"}}]},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
            ),
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        let error = read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openai",
            "model",
            false,
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("invalid arguments"));
        assert_eq!(
            error
                .downcast_ref::<ApiFailure>()
                .map(|failure| failure.kind),
            Some(crate::api::ApiFailureKind::MalformedToolArguments)
        );
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn malformed_event_json_fails_the_stream() {
        let response = crate::test_support::sse_response("data: {not json}\n\n").await;
        let (tx, mut rx) = mpsc::channel(10);

        let error = read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openai",
            "model",
            false,
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("invalid JSON"));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn surfaces_streamed_rate_limit_details() {
        let response = crate::test_support::sse_response(
            "data: {\"error\":{\"code\":429,\"message\":\"upstream limit\",\"metadata\":{\"error_type\":\"rate_limit_exceeded\"}},\"choices\":[{\"delta\":{\"content\":\"\"},\"finish_reason\":\"error\"}]}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_openai_sse(
            response,
            tx,
            CancellationToken::new(),
            "openrouter",
            "poolside/laguna",
            false,
        )
        .await
        .unwrap();

        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::Error(error))
                if error.message.contains("429 Too Many Requests")
                    && error.message.contains("poolside/laguna")
                    && error.message.contains("choose another model/provider")
        ));
        assert!(rx.recv().await.is_none());
    }
}
