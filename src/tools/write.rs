use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{interrupted_output, Tool, ToolOutput};

pub struct WriteTool;

#[derive(Deserialize)]
struct Params {
    file_path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "Write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories if needed."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn summarize(&self, input: &Value) -> String {
        input["file_path"].as_str().unwrap_or("?").to_string()
    }

    async fn execute(
        &self,
        input: Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<ToolOutput> {
        if cancel.is_cancelled() {
            return Ok(interrupted_output());
        }
        let params: Params = serde_json::from_value(input)?;
        let path = crate::tools::read::expand_tilde(&params.file_path);

        // Create parent directories
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        if cancel.is_cancelled() {
            return Ok(interrupted_output());
        }
        std::fs::write(&path, &params.content)?;

        Ok(ToolOutput {
            content: format!("Successfully wrote to {}", params.file_path),
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn write_creates_the_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested/file.txt");

        let result = WriteTool
            .execute(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "content": "hello"
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(path).unwrap(), "hello");
    }

    #[tokio::test]
    async fn cancelled_write_does_not_create_the_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested/file.txt");
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = WriteTool
            .execute(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "content": "must not be written"
                }),
                cancel,
            )
            .await
            .unwrap();

        assert!(result.is_error);
        assert_eq!(result.content, "Interrupted by user.");
        assert!(!path.exists());
        assert!(!temp.path().join("nested").exists());
    }
}
