use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;
use tokio::process::Command;

use super::{Tool, ToolOutput};

pub struct BashTool;

#[derive(Deserialize)]
struct Params {
    command: String,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default)]
    description: Option<String>,
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "Bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command. Use for git, build tools, or other CLI operations."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (max 600000, default 120000)"
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what the command does"
                }
            },
            "required": ["command"]
        })
    }

    fn is_read_only(&self) -> bool {
        false // conservative default; could be smarter with command analysis
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;

        let timeout_ms = params.timeout.unwrap_or(120_000).min(600_000);
        let timeout = Duration::from_millis(timeout_ms);

        let result = tokio::time::timeout(timeout, async {
            Command::new("sh")
                .arg("-c")
                .arg(&params.command)
                .output()
                .await
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let mut content = String::new();
                if !stdout.is_empty() {
                    content.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&stderr);
                }

                if !output.status.success() {
                    content.push_str(&format!("\nExit code: {}", output.status));
                }

                // Truncate very large output
                if content.len() > 100_000 {
                    content.truncate(100_000);
                    content.push_str("\n... (output truncated)");
                }

                Ok(ToolOutput {
                    content,
                    is_error: !output.status.success(),
                })
            }
            Ok(Err(e)) => Ok(ToolOutput {
                content: format!("Failed to execute command: {}", e),
                is_error: true,
            }),
            Err(_) => Ok(ToolOutput {
                content: format!("Command timed out after {}ms", timeout_ms),
                is_error: true,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.trim().contains("hello"));
    }

    #[tokio::test]
    async fn bash_exit_code() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "exit 1"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Exit code"));
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "echo err >&2"}))
            .await
            .unwrap();
        assert!(result.content.contains("err"));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "sleep 10", "timeout": 100}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }
}
