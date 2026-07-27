use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{Tool, ToolOutput};

pub struct GlobTool;

#[derive(Deserialize)]
struct Params {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    include_hidden: bool,
    #[serde(default)]
    include_build_dirs: bool,
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
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and directories (default: false)"
                },
                "include_build_dirs": {
                    "type": "boolean",
                    "description": "Include target build directories (default: false)"
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
            Some(path) => format!("{pattern} in {path}"),
            None => pattern.to_string(),
        }
    }

    async fn execute(
        &self,
        input: Value,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let base = params.path.as_deref().unwrap_or(".");
        let base = crate::tools::read::expand_tilde(base);
        let pattern = format!("{}/{}", base.display(), params.pattern);

        let mut paths: Vec<String> = Vec::new();
        for entry in glob::glob(&pattern)?.flatten() {
            let path_str = entry.to_string_lossy().to_string();
            if !super::search_filter::is_excluded(
                &entry,
                &base,
                params.include_hidden,
                params.include_build_dirs,
            ) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tokio_util::sync::CancellationToken;

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".github/workflows")).unwrap();
        fs::create_dir_all(dir.path().join("target/debug")).unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join(".github/workflows/ci.yml"), "ci").unwrap();
        fs::write(dir.path().join("target/debug/app"), "binary").unwrap();
        fs::write(dir.path().join("src/lib.rs"), "fn main() {}").unwrap();
        dir
    }

    #[tokio::test]
    async fn broad_search_excludes_hidden_and_build_directories() {
        let dir = fixture();
        let output = GlobTool
            .execute(
                json!({"pattern": "**/*", "path": dir.path()}),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("src/lib.rs"));
        assert!(!output.content.contains(".github"));
        assert!(!output.content.contains("target/debug"));
    }

    #[tokio::test]
    async fn flags_include_hidden_and_build_directories() {
        let dir = fixture();
        let output = GlobTool
            .execute(
                json!({
                    "pattern": "**/*",
                    "path": dir.path(),
                    "include_hidden": true,
                    "include_build_dirs": true
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(output.content.contains(".github/workflows/ci.yml"));
        assert!(output.content.contains("target/debug/app"));
    }

    #[tokio::test]
    async fn explicit_hidden_base_is_searched_without_a_flag() {
        let dir = fixture();
        let output = GlobTool
            .execute(
                json!({
                    "pattern": "**/*",
                    "path": dir.path().join(".github")
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("workflows/ci.yml"));
    }
}
