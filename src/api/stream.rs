use anyhow::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::types::Usage;

/// Events emitted by the SSE stream.
#[derive(Debug, Clone)]
pub enum ApiEvent {
    /// Streaming text from assistant
    Text(String),

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

    /// Error from API
    Error(String),
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
}

/// Read an SSE response and send parsed events to the channel.
pub async fn read_sse_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<ApiEvent>,
    cancel: CancellationToken,
) -> Result<()> {
    use futures_util::StreamExt as _;

    let mut stream = response.bytes_stream();
    let mut lines = Utf8LineDecoder::default();

    // Tool use accumulation state
    let mut current_tool_id = String::new();
    let mut current_tool_name = String::new();
    let mut current_tool_input = String::new();
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
                    }))
                    .await;
                let _ = tx.send(ApiEvent::Done).await;
                return Ok(());
            }

            let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };

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

                "content_block_stop"
                    if !current_tool_name.is_empty() && !current_tool_input.is_empty() =>
                {
                    if let Ok(input) = serde_json::from_str(&current_tool_input) {
                        let _ = tx
                            .send(ApiEvent::ToolUse {
                                id: current_tool_id.clone(),
                                name: current_tool_name.clone(),
                                input,
                            })
                            .await;
                    }
                    current_tool_name.clear();
                    current_tool_input.clear();
                    current_tool_id.clear();
                }

                "message_stop" => {
                    let _ = tx
                        .send(ApiEvent::Usage(Usage {
                            input_tokens,
                            output_tokens,
                            cache_read_tokens,
                            cache_creation_tokens,
                        }))
                        .await;
                    let _ = tx.send(ApiEvent::Done).await;
                    return Ok(());
                }

                "error" => {
                    let msg = event["error"]["message"]
                        .as_str()
                        .unwrap_or("unknown error");
                    let _ = tx.send(ApiEvent::Error(msg.to_string())).await;
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

        let error = read_sse_stream(response, tx, CancellationToken::new())
            .await
            .unwrap_err();

        assert!(error.to_string().contains("before message_stop"));
        assert!(matches!(rx.recv().await, Some(ApiEvent::Text(text)) if text == "partial"));
        assert!(rx.recv().await.is_none());
    }
}
