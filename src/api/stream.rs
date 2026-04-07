use anyhow::Result;
use tokio::sync::mpsc;

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

/// Read an SSE response and send parsed events to the channel.
pub async fn read_sse_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<ApiEvent>,
) -> Result<()> {
    use futures_util::StreamExt as _;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    // Tool use accumulation state
    let mut current_tool_id = String::new();
    let mut current_tool_name = String::new();
    let mut current_tool_input = String::new();
    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;
    let mut cache_read_tokens: u32 = 0;
    let mut cache_creation_tokens: u32 = 0;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete lines
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

                "content_block_stop" => {
                    if !current_tool_name.is_empty() && !current_tool_input.is_empty() {
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

    // Stream ended without message_stop
    let _ = tx.send(ApiEvent::Done).await;
    Ok(())
}
