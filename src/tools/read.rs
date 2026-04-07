use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::fmt::Write;

use super::{Tool, ToolOutput};

pub struct ReadTool;

#[derive(Deserialize)]
struct Params {
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "Read"
    }

    fn description(&self) -> &str {
        "Read a file from the filesystem. Returns content with line numbers."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read"
                }
            },
            "required": ["file_path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn summarize(&self, input: &Value) -> String {
        let path = input["file_path"].as_str().unwrap_or("?");
        match input["offset"].as_u64() {
            Some(offset) => format!("{path} (from line {offset})"),
            None => path.to_string(),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let path = expand_tilde(&params.file_path);

        if !path.exists() {
            return Ok(ToolOutput {
                content: format!("File does not exist: {}", params.file_path),
                is_error: true,
            });
        }

        if !path.is_file() {
            return Ok(ToolOutput {
                content: format!("Not a file: {}", params.file_path),
                is_error: true,
            });
        }

        let content = std::fs::read_to_string(&path)?;
        let lines: Vec<&str> = content.lines().collect();

        let start = params.offset.unwrap_or(1).saturating_sub(1);
        let end = if let Some(limit) = params.limit {
            (start + limit).min(lines.len())
        } else {
            lines.len().min(start + 2000) // default limit
        };

        let mut result = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            let _ = writeln!(result, "{line_num}\t{line}");
        }

        Ok(ToolOutput {
            content: result,
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn read_existing_file() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "line one").unwrap();
        writeln!(tmp, "line two").unwrap();
        writeln!(tmp, "line three").unwrap();

        let tool = ReadTool;
        let result = tool
            .execute(json!({"file_path": tmp.path().to_str().unwrap()}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("line one"));
        assert!(result.content.contains("line two"));
        assert!(result.content.contains("1\t"));
    }

    #[tokio::test]
    async fn read_with_offset_and_limit() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(tmp, "line {}", i).unwrap();
        }

        let tool = ReadTool;
        let result = tool
            .execute(json!({
                "file_path": tmp.path().to_str().unwrap(),
                "offset": 3,
                "limit": 2
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("line 3"));
        assert!(result.content.contains("line 4"));
        assert!(!result.content.contains("line 5"));
    }

    #[tokio::test]
    async fn read_nonexistent_file() {
        let tool = ReadTool;
        let result = tool
            .execute(json!({"file_path": "/tmp/definitely_does_not_exist_12345"}))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("does not exist"));
    }

    #[test]
    fn expand_tilde_works() {
        let path = expand_tilde("~/test.txt");
        assert!(!path.to_str().unwrap().contains('~'));
        assert!(path.to_str().unwrap().contains("test.txt"));
    }

    #[test]
    fn expand_tilde_absolute_unchanged() {
        let path = expand_tilde("/tmp/test.txt");
        assert_eq!(path.to_str().unwrap(), "/tmp/test.txt");
    }
}

pub fn expand_tilde(path: &str) -> std::path::PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home).join(stripped);
        }
    }
    std::path::PathBuf::from(path)
}
