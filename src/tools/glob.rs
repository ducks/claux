use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use super::{Tool, ToolOutput};
use crate::sandbox::SandboxPolicy;

pub struct GlobTool {
    sandbox_policy: Arc<SandboxPolicy>,
}

impl GlobTool {
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
        "Find paths matching a glob pattern. Returns paths sorted by modification time. Searches are bounded; partial results explain limits. Prefer a narrow base directory."
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
        let policy = self.sandbox_policy.clone();
        let cancel = _cancel.child_token();
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
    use super::search_budget::{Budget, Output};
    let budget = Budget::new(limits, cancel);
    let mut output = Output::default();
    if !budget.check() {
        return Ok(budget.finish(output));
    }
    let requested = crate::tools::read::expand_tilde(params.path.as_deref().unwrap_or("."));
    let base = match policy.authorize_read(&requested) {
        Ok(path) => path,
        Err(error) => return Ok(super::sandbox_denied_output(error)),
    };
    if let Err(error) = policy.authorize_search_pattern(&params.pattern) {
        return Ok(super::sandbox_denied_output(error));
    }
    glob::Pattern::new(&params.pattern)?;
    // Start at a literal prefix rather than walking unrelated sibling trees.
    let mut prefix = std::path::PathBuf::new();
    let mut tail = std::path::PathBuf::new();
    let mut wildcard = false;
    for component in std::path::Path::new(&params.pattern).components() {
        wildcard |= component
            .as_os_str()
            .to_string_lossy()
            .contains(['*', '?', '[']);
        if wildcard {
            tail.push(component.as_os_str());
        } else {
            prefix.push(component.as_os_str());
        }
    }
    let requested_root = base.join(prefix);
    if !requested_root.exists() {
        return Ok(budget.finish(output));
    }
    let root = match policy.authorize_read(&requested_root) {
        Ok(path) => path,
        Err(error) => return Ok(super::sandbox_denied_output(error)),
    };
    if super::search_filter::is_excluded(
        &root,
        &base,
        params.include_hidden,
        params.include_build_dirs,
    ) {
        return Ok(budget.finish(output));
    }
    let pattern = glob::Pattern::new(&tail.to_string_lossy())?;
    let options = glob::MatchOptions {
        case_sensitive: true,
        require_literal_separator: true,
        require_literal_leading_dot: false,
    };
    let recursive = tail.components().any(|part| part.as_os_str() == "**");
    let requested_depth = tail.components().count();
    let depth = if recursive {
        limits.depth
    } else {
        requested_depth.min(limits.depth)
    };
    let mut walker = walkdir::WalkDir::new(&root)
        .follow_links(false)
        .max_depth(depth)
        .into_iter();
    let mut times = Vec::new();
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
        let authorized = match policy.authorize_read(entry.path()) {
            Ok(path) => path,
            Err(_) => {
                if entry.file_type().is_dir() {
                    walker.skip_current_dir();
                }
                continue;
            }
        };
        if (recursive || requested_depth > limits.depth)
            && entry.file_type().is_dir()
            && entry.depth() == limits.depth
        {
            budget.omitted.set(true);
        }
        let relative = entry.path().strip_prefix(&root)?;
        if !tail.as_os_str().is_empty() && relative.as_os_str().is_empty() {
            continue;
        }
        if !pattern.matches_path_with(relative, options) {
            continue;
        }
        let modified = std::fs::metadata(&authorized)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if !output.push(authorized.to_string_lossy().into_owned(), &budget) {
            break;
        }
        times.push(modified);
    }
    let mut entries: Vec<_> = times.into_iter().zip(output.lines.drain(..)).collect();
    entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    output.lines = entries.into_iter().map(|(_, path)| path).collect();
    Ok(budget.finish(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tokio_util::sync::CancellationToken;

    fn tool() -> GlobTool {
        GlobTool::new(Arc::new(SandboxPolicy::unrestricted_for_tests()))
    }

    fn normalized(content: &str) -> String {
        content.replace('\\', "/")
    }

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
        let output = tool()
            .execute(
                json!({"pattern": "**/*", "path": dir.path()}),
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
                    "pattern": "**/*",
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
        assert!(content.contains("target/debug/app"));
    }

    #[tokio::test]
    async fn explicit_hidden_base_is_searched_without_a_flag() {
        let dir = fixture();
        let output = tool()
            .execute(
                json!({
                    "pattern": "**/*",
                    "path": dir.path().join(".github")
                }),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(normalized(&output.content).contains("workflows/ci.yml"));
    }
    fn bounded(
        input: Value,
        limits: super::super::search_budget::Limits,
        cancel: CancellationToken,
    ) -> ToolOutput {
        search(
            serde_json::from_value(input).unwrap(),
            &SandboxPolicy::unrestricted_for_tests(),
            cancel,
            limits,
        )
        .unwrap()
    }

    #[test]
    fn no_match_search_is_entry_bounded_and_never_claims_complete() {
        let dir = fixture();
        let output = bounded(
            json!({"pattern":"**/*.absent", "path":dir.path()}),
            super::super::search_budget::Limits {
                entries: 2,
                ..Default::default()
            },
            CancellationToken::new(),
        );
        assert!(output.content.contains("entry limit"));
        assert!(output.content.contains("Search incomplete"));
        assert_ne!(output.content, "No matches found");
    }

    #[test]
    fn result_and_depth_limits_return_partial_results() {
        let dir = fixture();
        for name in ["a.rs", "b.rs", "c.rs"] {
            fs::write(dir.path().join(name), "").unwrap();
        }
        let result = bounded(
            json!({"pattern":"**/*.rs", "path":dir.path()}),
            super::super::search_budget::Limits {
                results: 2,
                ..Default::default()
            },
            CancellationToken::new(),
        );
        assert!(result
            .content
            .contains(super::super::search_budget::TRUNCATED_NOTICE));
        let result = bounded(
            json!({"pattern":"**/*.rs", "path":dir.path()}),
            super::super::search_budget::Limits {
                depth: 1,
                ..Default::default()
            },
            CancellationToken::new(),
        );
        assert!(!normalized(&result.content).contains("src/lib.rs"));
        assert!(result.content.contains("Search incomplete"));
    }

    #[test]
    fn literal_prefix_and_nonrecursive_patterns_avoid_unrelated_descendants() {
        let dir = fixture();
        for index in 0..100 {
            fs::write(dir.path().join("src").join(format!("{index}.txt")), "").unwrap();
        }
        fs::write(dir.path().join("wanted.txt"), "").unwrap();
        let limits = super::super::search_budget::Limits {
            entries: 10,
            ..Default::default()
        };
        let result = bounded(
            json!({"pattern":"*.txt", "path":dir.path()}),
            limits,
            CancellationToken::new(),
        );
        assert!(result.content.contains("wanted.txt"));
        assert!(!result.content.contains("Search incomplete"));
        let result = bounded(
            json!({"pattern":"src/lib.rs", "path":dir.path()}),
            limits,
            CancellationToken::new(),
        );
        assert!(normalized(&result.content).ends_with("src/lib.rs"));
    }

    #[test]
    fn cancelled_and_expired_searches_stop_before_filesystem_work() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = bounded(
            json!({"pattern":"**/*", "path":"/does-not-exist"}),
            Default::default(),
            cancel,
        );
        assert!(result.is_error);
        assert!(result.content.contains("cancelled"));
        let result = bounded(
            json!({"pattern":"**/*", "path":"/does-not-exist"}),
            super::super::search_budget::Limits {
                timeout: std::time::Duration::ZERO,
                ..Default::default()
            },
            CancellationToken::new(),
        );
        assert!(result.content.contains("time limit"));
    }

    #[test]
    fn matches_are_sorted_by_modification_time_and_patterns_stay_relative() {
        let dir = fixture();
        let old = dir.path().join("old.txt");
        let new = dir.path().join("new.txt");
        fs::write(&old, "").unwrap();
        fs::write(&new, "").unwrap();
        fs::File::open(&old)
            .unwrap()
            .set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(10))
            .unwrap();
        fs::File::open(&new)
            .unwrap()
            .set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(20))
            .unwrap();
        let result = bounded(
            json!({"pattern":"*.txt", "path":dir.path()}),
            Default::default(),
            CancellationToken::new(),
        );
        assert_eq!(
            result.content.lines().next().unwrap(),
            new.to_string_lossy()
        );
        let denied = search(
            serde_json::from_value(json!({"pattern":"../**/*", "path":dir.path()})).unwrap(),
            &SandboxPolicy::workspace_only(dir.path()).unwrap(),
            CancellationToken::new(),
            Default::default(),
        )
        .unwrap();
        assert!(denied.is_error);
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
            json!({"pattern":"**/*.rs", "path":dir.path()}),
            super::super::search_budget::Limits {
                entries: 10,
                ..Default::default()
            },
            CancellationToken::new(),
        );
        assert!(normalized(&result.content).contains("src/lib.rs"));
        assert!(!result.content.contains("Search incomplete"));
    }
}
