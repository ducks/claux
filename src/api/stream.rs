use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::types::Usage;

/// Events emitted by the SSE stream.
#[derive(Debug, Clone)]
pub enum ApiEvent {
    /// Streaming text from assistant
    Text(String),

    /// Provider reasoning state that must be replayed on later tool rounds.
    Reasoning {
        text: Option<String>,
        details: Vec<serde_json::Value>,
    },

    /// Tool use request
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    /// Usage information
    Usage(Usage),

    /// Stream complete
    Done,

    /// Error from API, classified so the turn loop can choose a recovery
    /// without pattern-matching provider prose.
    Error(super::error::ApiFailure),
}

/// Incrementally split an SSE byte stream into UTF-8 lines.
///
/// HTTP chunks may end in the middle of a multibyte character, so decoding
/// each chunk independently would replace valid text with U+FFFD.
#[derive(Default)]
pub(super) struct Utf8LineDecoder {
    buffer: Vec<u8>,
}

impl Utf8LineDecoder {
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>> {
        self.buffer.extend_from_slice(chunk);

        let mut lines = Vec::new();
        let mut start = 0;
        for (index, byte) in self.buffer.iter().enumerate() {
            if *byte != b'\n' {
                continue;
            }

            let mut raw_line = &self.buffer[start..index];
            if raw_line.last() == Some(&b'\r') {
                raw_line = &raw_line[..raw_line.len() - 1];
            }
            lines.push(std::str::from_utf8(raw_line)?.to_string());
            start = index + 1;
        }

        if start > 0 {
            self.buffer.drain(..start);
        }
        Ok(lines)
    }

    pub fn finish(&self) -> Result<()> {
        // A partial final line is left for the protocol parser to reject,
        // but invalid or truncated UTF-8 should be reported explicitly.
        std::str::from_utf8(&self.buffer)?;
        Ok(())
    }

    /// Bytes received after the final complete line. Protocol diagnostics use
    /// this to distinguish a clean EOF from a stream truncated mid-frame.
    pub(super) fn pending_bytes(&self) -> &[u8] {
        &self.buffer
    }
}

/// Read an SSE response and send parsed events to the channel.
pub async fn read_sse_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<ApiEvent>,
    cancel: CancellationToken,
    model: &str,
) -> Result<()> {
    use futures_util::StreamExt as _;

    let mut stream = response.bytes_stream();
    let mut lines = Utf8LineDecoder::default();

    // Tool use accumulation state
    let mut current_tool_id = String::new();
    let mut current_tool_name = String::new();
    let mut current_tool_input = String::new();
    let mut current_tool_initial_input = None;
    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;
    let mut cache_read_tokens: u32 = 0;
    let mut cache_creation_tokens: u32 = 0;

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

            if data == "[DONE]" {
                let _ = tx
                    .send(ApiEvent::Usage(Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_creation_tokens,
                        provider_cost_usd: None,
                    }))
                    .await;
                let _ = tx.send(ApiEvent::Done).await;
                return Ok(());
            }

            let event = serde_json::from_str::<serde_json::Value>(data)
                .context("invalid JSON in Anthropic SSE event")?;

            let event_type = event["type"].as_str().unwrap_or("");

            match event_type {
                "message_start" => {
                    if let Some(usage) = event.pointer("/message/usage") {
                        input_tokens = usage["input_tokens"].as_u64().unwrap_or(0) as u32;
                        output_tokens = usage["output_tokens"].as_u64().unwrap_or(0) as u32;
                        cache_read_tokens =
                            usage["cache_read_input_tokens"].as_u64().unwrap_or(0) as u32;
                        cache_creation_tokens =
                            usage["cache_creation_input_tokens"].as_u64().unwrap_or(0) as u32;
                    }
                }

                "message_delta" => {
                    if let Some(usage) = event.get("usage") {
                        output_tokens = usage["output_tokens"]
                            .as_u64()
                            .unwrap_or(output_tokens as u64)
                            as u32;
                    }
                }

                "content_block_start" => {
                    if let Some(cb) = event.get("content_block") {
                        if cb["type"].as_str() == Some("tool_use") {
                            current_tool_id = cb["id"].as_str().unwrap_or("").to_string();
                            current_tool_name = cb["name"].as_str().unwrap_or("").to_string();
                            current_tool_input.clear();
                            current_tool_initial_input = cb.get("input").cloned();
                        }
                    }
                }

                "content_block_delta" => {
                    if let Some(delta) = event.get("delta") {
                        match delta["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                if let Some(text) = delta["text"].as_str() {
                                    let _ = tx.send(ApiEvent::Text(text.to_string())).await;
                                }
                            }
                            "input_json_delta" => {
                                if let Some(json) = delta["partial_json"].as_str() {
                                    current_tool_input.push_str(json);
                                }
                            }
                            _ => {}
                        }
                    }
                }

                "content_block_stop" if !current_tool_name.is_empty() => {
                    let input = if current_tool_input.is_empty() {
                        current_tool_initial_input
                            .take()
                            .unwrap_or_else(|| serde_json::json!({}))
                    } else {
                        serde_json::from_str(&current_tool_input).with_context(|| {
                            format!(
                                "invalid arguments for Anthropic tool call {current_tool_name} \
                                 ({current_tool_id})"
                            )
                        })?
                    };
                    let _ = tx
                        .send(ApiEvent::ToolUse {
                            id: current_tool_id.clone(),
                            name: current_tool_name.clone(),
                            input,
                        })
                        .await;
                    current_tool_name.clear();
                    current_tool_input.clear();
                    current_tool_id.clear();
                    current_tool_initial_input = None;
                }

                "message_stop" => {
                    let _ = tx
                        .send(ApiEvent::Usage(Usage {
                            input_tokens,
                            output_tokens,
                            cache_read_tokens,
                            cache_creation_tokens,
                            provider_cost_usd: None,
                        }))
                        .await;
                    let _ = tx.send(ApiEvent::Done).await;
                    return Ok(());
                }

                "error" => {
                    let _ = tx
                        .send(ApiEvent::Error(super::error::stream_error(
                            &event,
                            "anthropic",
                            model,
                        )))
                        .await;
                    return Ok(());
                }

                _ => {}
            }
        }
    }

    lines.finish()?;
    anyhow::bail!("stream ended before message_stop")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_preserves_utf8_split_across_chunks() {
        let data = "data: café\n";
        let split = data.find('é').unwrap() + 1;
        let mut decoder = Utf8LineDecoder::default();

        assert!(decoder.push(&data.as_bytes()[..split]).unwrap().is_empty());
        assert_eq!(
            decoder.push(&data.as_bytes()[split..]).unwrap(),
            vec!["data: café"]
        );
        decoder.finish().unwrap();
    }

    #[test]
    fn decoder_rejects_invalid_utf8() {
        let mut decoder = Utf8LineDecoder::default();
        assert!(decoder.push(b"data: \xff\n").is_err());
    }

    #[tokio::test]
    async fn rejects_eof_before_message_stop() {
        let response = crate::test_support::sse_response(
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        let error = read_sse_stream(response, tx, CancellationToken::new(), "claude-test")
            .await
            .unwrap_err();

        assert!(error.to_string().contains("before message_stop"));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(text)) if text == "partial"));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn malformed_event_json_fails_the_stream() {
        let response = crate::test_support::sse_response("data: {not json}\n\n").await;
        let (tx, mut rx) = mpsc::channel(10);

        let error = read_sse_stream(response, tx, CancellationToken::new(), "claude-test")
            .await
            .unwrap_err();

        assert!(error.to_string().contains("invalid JSON"));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn malformed_tool_arguments_fail_the_stream() {
        let response = crate::test_support::sse_response(
            concat!(
                "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"Read\",\"input\":{}}}\n\n",
                "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\"}}\n\n",
                "data: {\"type\":\"content_block_stop\"}\n\n",
                "data: {\"type\":\"message_stop\"}\n\n"
            ),
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        let error = read_sse_stream(response, tx, CancellationToken::new(), "claude-test")
            .await
            .unwrap_err();

        assert!(error.to_string().contains("invalid arguments"));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn empty_object_tool_arguments_are_emitted() {
        let response = crate::test_support::sse_response(
            concat!(
                "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"Status\",\"input\":{}}}\n\n",
                "data: {\"type\":\"content_block_stop\"}\n\n",
                "data: {\"type\":\"message_stop\"}\n\n"
            ),
        )
        .await;
        let (tx, mut rx) = mpsc::channel(10);

        read_sse_stream(response, tx, CancellationToken::new(), "claude-test")
            .await
            .unwrap();

        assert!(matches!(
            rx.recv().await,
            Some(ApiEvent::ToolUse { id, name, input })
                if id == "tool-1" && name == "Status" && input == serde_json::json!({})
        ));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Usage(_))));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Done)));
    }
}
