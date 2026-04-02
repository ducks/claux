use anyhow::Result;
use async_trait::async_trait;
use grep_searcher::{SearcherBuilder, Sink, SinkMatch};
use serde::Deserialize;
use serde_json::{Value, json};
use walkdir::WalkDir;

use super::{Tool, ToolOutput};

pub struct GrepTool;

#[derive(Deserialize)]
struct Params {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
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
            Some(path) => format!("\"{}\" in {}", pattern, path),
            None => format!("\"{}\"", pattern),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let base = params.path.as_deref().unwrap_or(".");
        let base = crate::tools::read::expand_tilde(base);

        let matcher = grep_regex::RegexMatcherBuilder::new()
            .build(&params.pattern)?;

        let mut searcher = SearcherBuilder::new().build();
        let mut results = Vec::new();

        let walker = WalkDir::new(&base)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                let p = e.path().to_string_lossy();
                !p.contains("/.") && !p.contains("/target/")
            });

        for entry in walker.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

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
                        self.matches.push(format!(
                            "{}:{}:{}",
                            self.path.display(),
                            mat.line_number().unwrap_or(0),
                            line.trim_end()
                        ));
                    }
                    Ok(true)
                }
            }

            let mut sink = CountSink {
                path,
                matches: &mut file_matches,
                count: &mut match_count,
                mode: &params.output_mode,
            };

            let _ = searcher.search_path(&matcher, path, &mut sink);

            if match_count > 0 {
                match params.output_mode.as_str() {
                    "files_with_matches" => results.push(path.display().to_string()),
                    "content" => results.extend(file_matches),
                    "count" => results.push(format!("{}:{}", path.display(), match_count)),
                    _ => {}
                }
            }
        }

        if results.is_empty() {
            return Ok(ToolOutput {
                content: "No matches found".to_string(),
                is_error: false,
            });
        }

        Ok(ToolOutput {
            content: results.join("\n"),
            is_error: false,
        })
    }
}
