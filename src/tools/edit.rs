use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::fmt::Write;

use super::{Tool, ToolOutput};

pub struct EditTool;

#[derive(Deserialize)]
struct Params {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "Edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing old_string with new_string. The old_string must match exactly."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)"
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn summarize(&self, input: &Value) -> String {
        input["file_path"].as_str().unwrap_or("?").to_string()
    }

    #[allow(clippy::manual_find)]
    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let path = crate::tools::read::expand_tilde(&params.file_path);

        if !path.exists() {
            return Ok(ToolOutput {
                content: format!("File does not exist: {}", params.file_path),
                is_error: true,
            });
        }

        let content = std::fs::read_to_string(&path)?;
        let count = content.matches(&params.old_string).count();

        if count == 0 {
            return Ok(ToolOutput {
                content: format!("old_string not found in {}", params.file_path),
                is_error: true,
            });
        }

        if count > 1 && !params.replace_all {
            return Ok(ToolOutput {
                content: format!(
                    "old_string appears {count} times. Use replace_all or provide more context."
                ),
                is_error: true,
            });
        }

        let new_content = if params.replace_all {
            content.replace(&params.old_string, &params.new_string)
        } else {
            content.replacen(&params.old_string, &params.new_string, 1)
        };

        std::fs::write(&path, &new_content)?;

        // Show context around the edit
        let new_lines: Vec<&str> = new_content.lines().collect();
        for (i, line) in new_lines.iter().enumerate() {
            if line.contains(&params.new_string) {
                let start = i.saturating_sub(3);
                let end = (i + 4).min(new_lines.len());

                let mut result = format!("Updated {}. Snippet:\n", params.file_path);
                for (j, l) in new_lines[start..end].iter().enumerate() {
                    let _ = writeln!(result, "{}\t{}", start + j + 1, l);
                }
                return Ok(ToolOutput {
                    content: result,
                    is_error: false,
                });
            }
        }

        Ok(ToolOutput {
            content: format!("Updated {}", params.file_path),
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn edit_replaces_string() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello world").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(json!({
                "file_path": tmp.path().to_str().unwrap(),
                "old_string": "hello",
                "new_string": "goodbye"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(content, "goodbye world");
    }

    #[tokio::test]
    async fn edit_fails_on_missing_string() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello world").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(json!({
                "file_path": tmp.path().to_str().unwrap(),
                "old_string": "nonexistent",
                "new_string": "replacement"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn edit_fails_on_ambiguous_match() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "aaa bbb aaa").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(json!({
                "file_path": tmp.path().to_str().unwrap(),
                "old_string": "aaa",
                "new_string": "ccc"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("2 times"));
    }

    #[tokio::test]
    async fn edit_replace_all() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "aaa bbb aaa").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(json!({
                "file_path": tmp.path().to_str().unwrap(),
                "old_string": "aaa",
                "new_string": "ccc",
                "replace_all": true
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(content, "ccc bbb ccc");
    }

    #[tokio::test]
    async fn edit_nonexistent_file() {
        let tool = EditTool;
        let result = tool
            .execute(json!({
                "file_path": "/tmp/definitely_does_not_exist_12345",
                "old_string": "a",
                "new_string": "b"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
    }
}
