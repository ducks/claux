use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{Tool, ToolOutput};

pub struct WebFetchTool {
    http: reqwest::Client,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::limited(5))
                .user_agent("claux/1.0")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

#[derive(Deserialize)]
struct Params {
    url: String,
    #[serde(default)]
    max_length: Option<usize>,
}

/// Max response size in characters (default 100k).
const DEFAULT_MAX_LENGTH: usize = 100_000;

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
                    "description": "Maximum response length in characters (default 100000)"
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

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let max_length = params.max_length.unwrap_or(DEFAULT_MAX_LENGTH);

        // Validate URL
        if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
            return Ok(ToolOutput {
                content: "URL must start with http:// or https://".to_string(),
                is_error: true,
            });
        }

        let response = match self.http.get(&params.url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolOutput {
                    content: format!("Failed to fetch URL: {e}"),
                    is_error: true,
                });
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

        // Get response body
        let body = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                return Ok(ToolOutput {
                    content: format!("Failed to read response body: {e}"),
                    is_error: true,
                });
            }
        };

        // If HTML, do a basic strip of tags to get readable text
        let text = if content_type.contains("text/html") {
            strip_html(&body)
        } else {
            body
        };

        // Truncate if needed
        let text = if text.len() > max_length {
            format!(
                "{}\n\n... (truncated, {} chars total)",
                &text[..max_length],
                text.len()
            )
        } else {
            text
        };

        Ok(ToolOutput {
            content: text,
            is_error: false,
        })
    }
}

/// Basic HTML tag stripping. Not a full parser — just removes tags and
/// decodes common entities to make HTML content readable.
fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut last_was_whitespace = false;

    let lower = html.to_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if !in_tag && chars[i] == '<' {
            // Check for script/style open/close
            let remaining: String = lower_chars[i..].iter().take(20).collect();
            if remaining.starts_with("<script") {
                in_script = true;
            } else if remaining.starts_with("</script") {
                in_script = false;
            } else if remaining.starts_with("<style") {
                in_style = true;
            } else if remaining.starts_with("</style") {
                in_style = false;
            }
            in_tag = true;
            i += 1;
            continue;
        }

        if in_tag {
            if chars[i] == '>' {
                in_tag = false;
            }
            i += 1;
            continue;
        }

        if in_script || in_style {
            i += 1;
            continue;
        }

        // Decode common HTML entities
        if chars[i] == '&' {
            let remaining: String = chars[i..].iter().take(10).collect();
            if remaining.starts_with("&amp;") {
                result.push('&');
                i += 5;
                last_was_whitespace = false;
                continue;
            } else if remaining.starts_with("&lt;") {
                result.push('<');
                i += 4;
                last_was_whitespace = false;
                continue;
            } else if remaining.starts_with("&gt;") {
                result.push('>');
                i += 4;
                last_was_whitespace = false;
                continue;
            } else if remaining.starts_with("&quot;") {
                result.push('"');
                i += 6;
                last_was_whitespace = false;
                continue;
            } else if remaining.starts_with("&#39;") || remaining.starts_with("&apos;") {
                result.push('\'');
                i += if remaining.starts_with("&#39;") { 5 } else { 6 };
                last_was_whitespace = false;
                continue;
            } else if remaining.starts_with("&nbsp;") {
                result.push(' ');
                i += 6;
                last_was_whitespace = true;
                continue;
            }
        }

        // Collapse whitespace
        if chars[i].is_whitespace() {
            if !last_was_whitespace {
                result.push(' ');
                last_was_whitespace = true;
            }
        } else {
            result.push(chars[i]);
            last_was_whitespace = false;
        }

        i += 1;
    }

    // Clean up excessive blank lines
    let mut cleaned = String::new();
    let mut blank_count = 0;
    for line in result.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank_count += 1;
            if blank_count <= 2 {
                cleaned.push('\n');
            }
        } else {
            blank_count = 0;
            cleaned.push_str(trimmed);
            cleaned.push('\n');
        }
    }

    cleaned.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_html_basic() {
        let html = "<p>Hello <b>world</b></p>";
        assert_eq!(strip_html(html), "Hello world");
    }

    #[test]
    fn strip_html_entities() {
        let html = "&amp; &lt; &gt; &quot;";
        assert_eq!(strip_html(html), "& < > \"");
    }

    #[test]
    fn strip_html_script_removed() {
        let html = "before<script>var x = 1;</script>after";
        assert_eq!(strip_html(html), "beforeafter");
    }

    #[test]
    fn strip_html_style_removed() {
        let html = "before<style>.foo { color: red; }</style>after";
        assert_eq!(strip_html(html), "beforeafter");
    }

    #[test]
    fn strip_html_whitespace_collapsed() {
        let html = "hello    \n\n\n   world";
        let result = strip_html(html);
        assert!(!result.contains("    "));
    }

    #[tokio::test]
    async fn rejects_non_http_url() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "ftp://example.com"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("http"));
    }

    #[tokio::test]
    async fn rejects_invalid_url() {
        let tool = WebFetchTool::new();
        let result = tool.execute(json!({"url": "not-a-url"})).await.unwrap();
        assert!(result.is_error);
    }
}
