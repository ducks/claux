use anyhow::Result;
use async_trait::async_trait;
use grep_searcher::{SearcherBuilder, Sink, SinkMatch};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use walkdir::WalkDir;

use super::{Tool, ToolOutput};
use crate::sandbox::SandboxPolicy;

const MAX_RESULTS: usize = 1_000;
const MAX_OUTPUT_BYTES: usize = 256 * 1024;
const TRUNCATED_NOTICE: &str = "[results truncated: refine the pattern or path]";

pub struct GrepTool {
    sandbox_policy: Arc<SandboxPolicy>,
}

impl GrepTool {
    pub fn new(sandbox_policy: Arc<SandboxPolicy>) -> Self {
        Self { sandbox_policy }
    }
}

#[derive(Deserialize)]
struct Params {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    include_hidden: bool,
    #[serde(default)]
    include_build_dirs: bool,
    #[serde(default = "default_output_mode")]
    output_mode: String,
}

fn default_output_mode() -> String {
    "files_with_matches".to_string()
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "Grep"
    }

    fn description(&self) -> &str {
        "Search file contents using regex. Supports content, files_with_matches, and count modes."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob to filter files (e.g. '*.rs')"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and directories (default: false)"
                },
                "include_build_dirs": {
                    "type": "boolean",
                    "description": "Include target build directories (default: false)"
                },
                "output_mode": {
                    "type": "string",
                    "description": "Output: 'content', 'files_with_matches', or 'count'"
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
            Some(path) => format!("\"{pattern}\" in {path}"),
            None => format!("\"{pattern}\""),
        }
    }

    async fn execute(
        &self,
        input: Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let base = params.path.as_deref().unwrap_or(".");
        let requested_base = crate::tools::read::expand_tilde(base);
        let base = match self.sandbox_policy.authorize_read(&requested_base) {
            Ok(path) => path,
            Err(error) => return Ok(super::sandbox_denied_output(error)),
        };

        let matcher = grep_regex::RegexMatcherBuilder::new().build(&params.pattern)?;

        let mut searcher = SearcherBuilder::new().build();
        let mut results = Vec::new();
        let mut output_bytes = 0usize;
        let mut truncated = false;

        let walker = WalkDir::new(&base)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| {
                !super::search_filter::is_excluded(
                    entry.path(),
                    &base,
                    params.include_hidden,
                    params.include_build_dirs,
                )
            });

        for entry in walker.flatten() {
            if cancel.is_cancelled() {
                return Ok(ToolOutput {
                    content: "Search cancelled by user.".to_string(),
                    is_error: true,
                });
            }
            if results.len() >= MAX_RESULTS || output_bytes >= MAX_OUTPUT_BYTES {
                truncated = true;
                break;
            }
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let path = match self.sandbox_policy.authorize_read(path) {
                Ok(path) => path,
                Err(_) => continue,
            };

            // Apply glob filter
            if let Some(ref glob_pat) = params.glob {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Ok(pattern) = glob::Pattern::new(glob_pat) {
                        if !pattern.matches(name) {
                            continue;
                        }
                    }
                }
            }

            let mut file_matches: Vec<String> = Vec::new();
            let mut match_count = 0usize;

            struct CountSink<'a> {
                path: &'a std::path::Path,
                matches: &'a mut Vec<String>,
                count: &'a mut usize,
                mode: &'a str,
                remaining_results: usize,
                remaining_bytes: usize,
                truncated: &'a mut bool,
            }

            impl Sink for CountSink<'_> {
                type Error = std::io::Error;

                fn matched(
                    &mut self,
                    _searcher: &grep_searcher::Searcher,
                    mat: &SinkMatch<'_>,
                ) -> Result<bool, Self::Error> {
                    *self.count += 1;
                    if self.mode == "content" {
                        let line = std::str::from_utf8(mat.bytes()).unwrap_or("");
                        let rendered = format!(
                            "{}:{}:{}",
                            self.path.display(),
                            mat.line_number().unwrap_or(0),
                            line.trim_end()
                        );
                        if self.matches.len() >= self.remaining_results
                            || rendered.len() > self.remaining_bytes
                        {
                            *self.truncated = true;
                            return Ok(false);
                        }
                        self.remaining_bytes -= rendered.len();
                        self.matches.push(rendered);
                    }
                    Ok(self.mode == "content")
                }
            }

            let mut sink = CountSink {
                path: &path,
                matches: &mut file_matches,
                count: &mut match_count,
                mode: &params.output_mode,
                remaining_results: MAX_RESULTS.saturating_sub(results.len()),
                remaining_bytes: MAX_OUTPUT_BYTES.saturating_sub(output_bytes),
                truncated: &mut truncated,
            };

            let _ = searcher.search_path(&matcher, &path, &mut sink);

            if match_count > 0 {
                match params.output_mode.as_str() {
                    "files_with_matches" => results.push(path.display().to_string()),
                    "content" => results.extend(file_matches),
                    "count" => results.push(format!("{}:{}", path.display(), match_count)),
                    _ => {}
                }
                output_bytes = results.iter().map(String::len).sum::<usize>()
                    + results.len().saturating_sub(1);
            }
        }

        if results.is_empty() {
            return Ok(ToolOutput {
                content: "No matches found".to_string(),
                is_error: false,
            });
        }

        let mut content = results.join("\n");
        if truncated {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(TRUNCATED_NOTICE);
        }
        Ok(ToolOutput {
            content,
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tokio_util::sync::CancellationToken;

    fn tool() -> GrepTool {
        GrepTool::new(Arc::new(SandboxPolicy::unrestricted_for_tests()))
    }

    fn normalized(content: &str) -> String {
        content.replace('\\', "/")
    }

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".github/workflows")).unwrap();
        fs::create_dir_all(dir.path().join("target/debug")).unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join(".github/workflows/ci.yml"), "needle").unwrap();
        fs::write(dir.path().join("target/debug/log.txt"), "needle").unwrap();
        fs::write(dir.path().join("src/lib.rs"), "needle").unwrap();
        dir
    }

    #[tokio::test]
    async fn broad_search_excludes_hidden_and_build_directories() {
        let dir = fixture();
        let output = tool()
            .execute(
                json!({"pattern": "needle", "path": dir.path()}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let content = normalized(&output.content);

        assert!(content.contains("src/lib.rs"));
        assert!(!content.contains(".github"));
        assert!(!content.contains("target/debug"));
    }

    #[tokio::test]
    async fn flags_include_hidden_and_build_directories() {
        let dir = fixture();
        let output = tool()
            .execute(
                json!({
                    "pattern": "needle",
                    "path": dir.path(),
                    "include_hidden": true,
                    "include_build_dirs": true
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let content = normalized(&output.content);

        assert!(content.contains(".github/workflows/ci.yml"));
        assert!(content.contains("target/debug/log.txt"));
    }

    #[tokio::test]
    async fn explicit_hidden_base_is_searched_without_a_flag() {
        let dir = fixture();
        let output = tool()
            .execute(
                json!({
                    "pattern": "needle",
                    "path": dir.path().join(".github")
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(normalized(&output.content).contains("workflows/ci.yml"));
    }

    #[tokio::test]
    async fn broad_content_search_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let content = (0..(MAX_RESULTS + 50))
            .map(|index| format!("needle {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(dir.path().join("large.txt"), content).unwrap();

        let output = tool()
            .execute(
                json!({
                    "pattern": "needle",
                    "path": dir.path(),
                    "output_mode": "content"
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(output.content.contains(TRUNCATED_NOTICE));
        assert!(output.content.lines().count() <= MAX_RESULTS + 1);
        assert!(output.content.len() <= MAX_OUTPUT_BYTES + TRUNCATED_NOTICE.len() + 1);
    }
}
