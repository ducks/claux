use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

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

    fn summarize(&self, input: &Value) -> String {
        let cmd = input["command"].as_str().unwrap_or("?");
        // Truncate long commands
        if cmd.len() > 80 {
            format!("{}...", crate::utils::truncate_str(cmd, 77))
        } else {
            cmd.to_string()
        }
    }

    async fn execute(&self, input: Value, cancel: CancellationToken) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;

        let timeout_ms = params.timeout.unwrap_or(120_000).min(600_000);
        let timeout = Duration::from_millis(timeout_ms);

        let mut child = match Command::new("sh")
            .arg("-c")
            .arg(&params.command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput {
                    content: format!("Failed to execute command: {e}"),
                    is_error: true,
                });
            }
        };

        // Take the pipes so we can read them concurrently with wait().
        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        // Spawn readers so partial output is captured even if we get cancelled
        // or time out mid-stream.
        let stdout_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(p) = stdout_pipe.as_mut() {
                let _ = p.read_to_end(&mut buf).await;
            }
            buf
        });
        let stderr_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(p) = stderr_pipe.as_mut() {
                let _ = p.read_to_end(&mut buf).await;
            }
            buf
        });

        let outcome = tokio::select! {
            status = child.wait() => Outcome::Finished(status),
            _ = cancel.cancelled() => Outcome::Cancelled,
            _ = tokio::time::sleep(timeout) => Outcome::TimedOut,
        };

        // For Cancelled / TimedOut, the child is still alive — kill it.
        if !matches!(outcome, Outcome::Finished(_)) {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }

        let stdout = stdout_task.await.unwrap_or_default();
        let stderr = stderr_task.await.unwrap_or_default();

        let stdout_s = String::from_utf8_lossy(&stdout);
        let stderr_s = String::from_utf8_lossy(&stderr);

        let mut content = String::new();
        if !stdout_s.is_empty() {
            content.push_str(&stdout_s);
        }
        if !stderr_s.is_empty() {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(&stderr_s);
        }

        let is_error = match &outcome {
            Outcome::Finished(Ok(status)) => {
                if !status.success() {
                    content.push_str(&format!("\nExit code: {status}"));
                }
                !status.success()
            }
            Outcome::Finished(Err(e)) => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&format!("wait error: {e}"));
                true
            }
            Outcome::Cancelled => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str("Interrupted by user.");
                true
            }
            Outcome::TimedOut => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&format!("Command timed out after {timeout_ms}ms"));
                true
            }
        };

        if content.len() > 100_000 {
            content.truncate(100_000);
            content.push_str("\n... (output truncated)");
        }

        Ok(ToolOutput { content, is_error })
    }
}

enum Outcome {
    Finished(std::io::Result<std::process::ExitStatus>),
    Cancelled,
    TimedOut,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> CancellationToken {
        CancellationToken::new()
    }

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "echo hello"}), token())
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.trim().contains("hello"));
    }

    #[tokio::test]
    async fn bash_exit_code() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "exit 1"}), token())
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Exit code"));
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "echo err >&2"}), token())
            .await
            .unwrap();
        assert!(result.content.contains("err"));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool;
        let result = tool
            .execute(json!({"command": "sleep 10", "timeout": 100}), token())
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }

    #[tokio::test]
    async fn bash_cancellation() {
        let tool = BashTool;
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            cancel_clone.cancel();
        });
        let result = tool
            .execute(json!({"command": "sleep 30", "timeout": 60000}), cancel)
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Interrupted"));
    }
}
