use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{Tool, ToolOutput};

mod html;
mod policy;

use html::strip_html;

pub struct WebFetchTool;

#[derive(Deserialize)]
struct Params {
    url: String,
    #[serde(default)]
    max_length: Option<usize>,
}

/// Max response size in characters (default 100k).
const DEFAULT_MAX_LENGTH: usize = 100_000;
const HARD_MAX_LENGTH: usize = 1_000_000;
const MAX_REDIRECTS: usize = 5;

impl WebFetchTool {
    pub fn new() -> Self {
        Self
    }

    async fn fetch(
        &self,
        mut url: reqwest::Url,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<reqwest::Response, String> {
        for redirect_count in 0..=MAX_REDIRECTS {
            let addresses = tokio::select! {
                result = policy::validate_destination(&url) => result?,
                _ = cancel.cancelled() => return Err("Interrupted by user.".to_string()),
            };
            let host = url
                .host_str()
                .ok_or_else(|| "URL must include a host".to_string())?;
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .resolve_to_addrs(host, &addresses)
                .user_agent("claux/1.0")
                .build()
                .map_err(|e| format!("Failed to configure HTTP client: {e}"))?;

            let response = tokio::select! {
                result = client.get(url.clone()).send() => {
                    result.map_err(|e| format!("Failed to fetch URL: {e}"))?
                }
                _ = cancel.cancelled() => return Err("Interrupted by user.".to_string()),
            };

            if !response.status().is_redirection() {
                return Ok(response);
            }
            if redirect_count == MAX_REDIRECTS {
                return Err(format!("Too many redirects (maximum {MAX_REDIRECTS})"));
            }

            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .ok_or_else(|| "Redirect response did not include a Location header".to_string())?
                .to_str()
                .map_err(|_| "Redirect Location header is not valid text".to_string())?;
            url = url
                .join(location)
                .map_err(|e| format!("Invalid redirect destination: {e}"))?;
        }

        unreachable!("redirect loop always returns")
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "WebFetch"
    }

    fn description(&self) -> &str {
        "Fetch the content of a URL. Returns the text content of the page. \
         Useful for reading documentation, API responses, or web pages."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum response length in characters (default 100000)",
                    "minimum": 1,
                    "maximum": HARD_MAX_LENGTH
                }
            },
            "required": ["url"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn summarize(&self, input: &Value) -> String {
        input["url"].as_str().unwrap_or("?").to_string()
    }

    async fn execute(
        &self,
        input: Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let max_length = params
            .max_length
            .unwrap_or(DEFAULT_MAX_LENGTH)
            .clamp(1, HARD_MAX_LENGTH);

        let url = match reqwest::Url::parse(&params.url) {
            Ok(url) => url,
            Err(e) => {
                return Ok(ToolOutput {
                    content: format!("Invalid URL: {e}"),
                    is_error: true,
                })
            }
        };
        let response = match self.fetch(url, &cancel).await {
            Ok(response) => response,
            Err(error) => {
                return Ok(ToolOutput {
                    content: error,
                    is_error: true,
                })
            }
        };

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolOutput {
                content: format!(
                    "HTTP {}: {}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("error")
                ),
                is_error: true,
            });
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let byte_limit = max_length.saturating_mul(4).min(HARD_MAX_LENGTH * 4);
        let (body, body_truncated) = match read_bounded_body(response, byte_limit, &cancel).await {
            Ok(body) => body,
            Err(error) => {
                return Ok(ToolOutput {
                    content: error,
                    is_error: true,
                })
            }
        };

        let text = if content_type.contains("text/html") {
            strip_html(&body)
        } else {
            body
        };

        let (mut text, text_truncated) = truncate_chars(text, max_length);
        if body_truncated || text_truncated {
            text.push_str("\n\n... (truncated)");
        }

        Ok(ToolOutput {
            content: text,
            is_error: false,
        })
    }
}

async fn read_bounded_body(
    response: reqwest::Response,
    byte_limit: usize,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<(String, bool), String> {
    use futures_util::StreamExt as _;

    let mut stream = response.bytes_stream();
    let mut bytes = Vec::with_capacity(byte_limit.min(64 * 1024));
    let mut truncated = false;

    loop {
        let chunk = tokio::select! {
            chunk = stream.next() => chunk,
            _ = cancel.cancelled() => return Err("Interrupted by user.".to_string()),
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|e| format!("Failed to read response body: {e}"))?;
        let remaining = byte_limit.saturating_sub(bytes.len());
        if chunk.len() > remaining {
            bytes.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        } else {
            bytes.extend_from_slice(&chunk);
        }
    }

    Ok((String::from_utf8_lossy(&bytes).into_owned(), truncated))
}

fn truncate_chars(text: String, max_length: usize) -> (String, bool) {
    let Some((byte_index, _)) = text.char_indices().nth(max_length) else {
        return (text, false);
    };
    (text[..byte_index].to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn truncates_unicode_at_character_boundary() {
        let (text, truncated) = truncate_chars("éclair".to_string(), 1);
        assert_eq!(text, "é");
        assert!(truncated);
    }

    #[tokio::test]
    async fn rejects_non_http_url() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(
                json!({"url": "ftp://example.com"}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("http"));
    }

    #[tokio::test]
    async fn rejects_invalid_url() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "not-a-url"}), CancellationToken::new())
            .await
            .unwrap();
        assert!(result.is_error);
    }
}
