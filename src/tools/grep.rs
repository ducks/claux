use anyhow::Result;
use async_trait::async_trait;
use grep_searcher::{SearcherBuilder, Sink, SinkMatch};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use walkdir::WalkDir;

use super::{Tool, ToolOutput};
use crate::sandbox::SandboxPolicy;

#[cfg(test)]
const MAX_RESULTS: usize = 1_000;
#[cfg(test)]
const MAX_OUTPUT_BYTES: usize = 256 * 1024;
#[cfg(test)]
use super::search_budget::TRUNCATED_NOTICE;

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
        "Search file contents using regex. Supports content, files_with_matches, and count modes. Bounded by time, entries, bytes and results; partial results explain limits. Narrow the path for large trees."
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
        let policy = self.sandbox_policy.clone();
        let cancel = cancel.child_token();
        let _guard = cancel.clone().drop_guard();
        tokio::task::spawn_blocking(move || {
            search(
                params,
                &policy,
                cancel,
                super::search_budget::Limits::default(),
            )
        })
        .await?
    }
}

fn search(
    params: Params,
    policy: &SandboxPolicy,
    cancel: tokio_util::sync::CancellationToken,
    limits: super::search_budget::Limits,
) -> Result<ToolOutput> {
    use super::search_budget::{BoundedReader, Budget, Output};
    use grep_searcher::BinaryDetection;
    let budget = Budget::new(limits, cancel);
    let mut output = Output::default();
    if !budget.check() {
        return Ok(budget.finish(output));
    }
    anyhow::ensure!(
        matches!(
            params.output_mode.as_str(),
            "content" | "count" | "files_with_matches"
        ),
        "Invalid Grep output_mode"
    );
    let requested = crate::tools::read::expand_tilde(params.path.as_deref().unwrap_or("."));
    let base = match policy.authorize_read(&requested) {
        Ok(path) => path,
        Err(error) => return Ok(super::sandbox_denied_output(error)),
    };
    let matcher = grep_regex::RegexMatcherBuilder::new().build(&params.pattern)?;
    let filter = params.glob.as_deref().map(glob::Pattern::new).transpose()?;
    let mut searcher = SearcherBuilder::new()
        .line_number(true)
        .heap_limit(Some(limits.file_bytes as usize))
        .binary_detection(BinaryDetection::quit(b'\0'))
        .build();
    let mut walker = WalkDir::new(&base)
        .follow_links(false)
        .max_depth(limits.depth)
        .into_iter();
    while budget.check() {
        let Some(entry) = walker.next() else { break };
        if !budget.visit() {
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                budget.omitted.set(true);
                continue;
            }
        };
        if super::search_filter::is_excluded(
            entry.path(),
            &base,
            params.include_hidden,
            params.include_build_dirs,
        ) {
            if entry.file_type().is_dir() {
                walker.skip_current_dir();
            }
            continue;
        }
        if entry.file_type().is_dir() {
            if policy.authorize_read(entry.path()).is_err() {
                walker.skip_current_dir();
            }
            if entry.depth() == limits.depth {
                budget.omitted.set(true);
            }
            continue;
        }
        let path = match policy.authorize_read(entry.path()) {
            Ok(path) => path,
            Err(_) => continue,
        };
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => {
                budget.omitted.set(true);
                continue;
            }
        };
        // Never open FIFOs, sockets or device nodes during content searches.
        if !metadata.is_file() {
            continue;
        }
        if let Some(ref pattern) = filter {
            if !path
                .file_name()
                .is_some_and(|name| pattern.matches_path(std::path::Path::new(name)))
            {
                continue;
            }
        }
        if metadata.len() > limits.file_bytes {
            budget.omitted.set(true);
            continue;
        }
        let file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(_) => {
                budget.omitted.set(true);
                continue;
            }
        };
        let reader = BoundedReader {
            inner: file,
            budget: &budget,
            file_left: limits.file_bytes,
        };
        struct CountSink<'a> {
            path: &'a std::path::Path,
            count: usize,
            mode: &'a str,
            budget: &'a Budget,
            output: &'a mut Output,
        }
        impl Sink for CountSink<'_> {
            type Error = std::io::Error;
            fn matched(
                &mut self,
                _: &grep_searcher::Searcher,
                mat: &SinkMatch<'_>,
            ) -> std::io::Result<bool> {
                if !self.budget.check() {
                    return Ok(false);
                }
                self.count += 1;
                if self.mode == "content" {
                    if mat.bytes().len() > self.output.remaining_bytes(self.budget) {
                        self.budget.stop("output limit reached");
                        return Ok(false);
                    }
                    let line = std::str::from_utf8(mat.bytes()).unwrap_or("");
                    if !self.output.push(
                        format!(
                            "{}:{}:{}",
                            self.path.display(),
                            mat.line_number().unwrap_or(0),
                            line.trim_end()
                        ),
                        self.budget,
                    ) {
                        return Ok(false);
                    }
                }
                Ok(self.mode != "files_with_matches")
            }
        }
        let mut sink = CountSink {
            path: &path,
            count: 0,
            mode: &params.output_mode,
            budget: &budget,
            output: &mut output,
        };
        if searcher.search_reader(&matcher, reader, &mut sink).is_err() && budget.check() {
            budget.omitted.set(true);
        }
        let count = sink.count;
        if count > 0 {
            let rendered = match params.output_mode.as_str() {
                "files_with_matches" => Some(path.display().to_string()),
                "count" => Some(format!("{}:{count}", path.display())),
                _ => None,
            };
            if let Some(line) = rendered {
                if !output.push(line, &budget) {
                    break;
                }
            }
        }
    }
    Ok(budget.finish(output))
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
    fn bounded(input: Value, limits: super::super::search_budget::Limits) -> ToolOutput {
        search(
            serde_json::from_value(input).unwrap(),
            &SandboxPolicy::unrestricted_for_tests(),
            CancellationToken::new(),
            limits,
        )
        .unwrap()
    }

    #[test]
    fn nonmatching_tree_and_file_reads_are_bounded() {
        let dir = fixture();
        let result = bounded(
            json!({"pattern":"absent", "path":dir.path()}),
            super::super::search_budget::Limits {
                entries: 2,
                ..Default::default()
            },
        );
        assert!(result.content.contains("entry limit"));
        fs::write(dir.path().join("read.txt"), "a".repeat(100)).unwrap();
        let result = bounded(
            json!({"pattern":"absent", "path":dir.path().join("read.txt")}),
            super::super::search_budget::Limits {
                read_bytes: 8,
                ..Default::default()
            },
        );
        assert!(result.content.contains("read byte limit"));
        assert!(result.content.contains("Search incomplete"));
    }

    #[test]
    fn count_mode_counts_all_matching_lines() {
        let dir = fixture();
        fs::write(
            dir.path().join("count.txt"),
            "needle\nnope\nneedle\nneedle\n",
        )
        .unwrap();
        let result = bounded(
            json!({"pattern":"needle", "path":dir.path().join("count.txt"), "output_mode":"count"}),
            Default::default(),
        );
        assert!(result.content.ends_with(":3"));
    }

    #[test]
    fn skipped_large_files_and_long_lines_are_explicit() {
        let dir = fixture();
        fs::write(
            dir.path().join("long.txt"),
            format!("needle{}", "a".repeat(1000)),
        )
        .unwrap();
        let result = bounded(
            json!({"pattern":"needle", "path":dir.path().join("long.txt")}),
            super::super::search_budget::Limits {
                file_bytes: 100,
                ..Default::default()
            },
        );
        assert!(result.content.contains("Search incomplete"));
        assert!(result.content.contains("file size"));
        let result = bounded(
            json!({"pattern":"needle", "path":dir.path().join("long.txt"), "output_mode":"content"}),
            super::super::search_budget::Limits {
                output_bytes: 100,
                ..Default::default()
            },
        );
        assert!(result.content.contains(TRUNCATED_NOTICE));
        assert!(result.content.len() < 300);
    }

    #[tokio::test]
    async fn pre_cancelled_search_and_invalid_glob_fail_visibly() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = tool()
            .execute(
                json!({"pattern":"needle", "path":"/does-not-exist"}),
                cancel,
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("cancelled"));
        let dir = fixture();
        assert!(tool()
            .execute(
                json!({"pattern":"needle", "glob":"[", "path":dir.path()}),
                CancellationToken::new()
            )
            .await
            .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn directory_symlinks_and_special_files_are_not_followed() {
        let dir = fixture();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("private.txt"), "needle").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("escape")).unwrap();
        let _socket = std::os::unix::net::UnixListener::bind(dir.path().join("socket")).unwrap();
        let result = search(
            serde_json::from_value(json!({"pattern":"needle", "path":dir.path()})).unwrap(),
            &SandboxPolicy::workspace_only(dir.path()).unwrap(),
            CancellationToken::new(),
            Default::default(),
        )
        .unwrap();
        assert!(!result.content.contains("private.txt"));
        assert!(normalized(&result.content).contains("src/lib.rs"));
    }
    #[test]
    fn excluded_subtrees_do_not_consume_the_traversal_budget() {
        let dir = fixture();
        for index in 0..50 {
            fs::write(
                dir.path().join(".github").join(format!("{index}.txt")),
                "needle",
            )
            .unwrap();
            fs::write(
                dir.path().join("target").join(format!("{index}.txt")),
                "needle",
            )
            .unwrap();
        }
        let result = bounded(
            json!({"pattern":"needle", "path":dir.path()}),
            super::super::search_budget::Limits {
                entries: 10,
                ..Default::default()
            },
        );
        assert!(normalized(&result.content).contains("src/lib.rs"));
        assert!(!result.content.contains("Search incomplete"));
    }
}
