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
            let _ = writeln!(result, "{}\t{}", line_num, line);
        }

        Ok(ToolOutput {
            content: result,
            is_error: false,
        })
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
