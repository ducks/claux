use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Tool, ToolOutput};

pub struct GlobTool;

#[derive(Deserialize)]
struct Params {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "Glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns paths sorted by modification time."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '**/*.rs', 'src/**/*.ts')"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in"
                }
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn summarize(&self, input: &Value) -> String {
        let pattern = input["pattern"].as_str().unwrap_or("?");
        match input["path"].as_str() {
            Some(path) => format!("{} in {}", pattern, path),
            None => pattern.to_string(),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let base = params.path.as_deref().unwrap_or(".");
        let base = crate::tools::read::expand_tilde(base);
        let pattern = format!("{}/{}", base.display(), params.pattern);

        let mut paths: Vec<String> = Vec::new();
        for entry in glob::glob(&pattern)?.flatten() {
            let path_str = entry.to_string_lossy().to_string();
            // Skip hidden files and build directories
            if !path_str.contains("/.") && !path_str.contains("/target/") {
                paths.push(path_str);
            }
        }

        // Sort by modification time (most recent first)
        paths.sort_by_cached_key(|p| {
            std::fs::metadata(p)
                .and_then(|m| m.modified())
                .map(|t| {
                    std::time::SystemTime::now()
                        .duration_since(t)
                        .unwrap_or_default()
                })
                .unwrap_or_default()
        });

        if paths.is_empty() {
            return Ok(ToolOutput {
                content: "No matches found".to_string(),
                is_error: false,
            });
        }

        Ok(ToolOutput {
            content: paths.join("\n"),
            is_error: false,
        })
    }
}
