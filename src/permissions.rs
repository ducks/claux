use serde::{Deserialize, Serialize};

use crate::utils::diff::generate_diff;

/// How permissions are handled.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    /// Prompt for write operations, auto-allow reads
    #[default]
    Default,
    /// Auto-allow file edits, still prompt for bash
    AcceptEdits,
    /// Allow everything without prompting
    Bypass,
    /// Deny all write operations
    Plan,
}

/// Result of a permission check.
pub enum PermissionResult {
    Allow,
    Deny(String),
    Ask {
        message: String,
        diff: Option<String>,
    },
}

/// User's response to a permission prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionResponse {
    /// Allow this one time
    Allow,
    /// Deny this one time
    Deny,
    /// Always allow this tool for the rest of the session
    AlwaysAllow,
    /// Always allow this specific command (for Bash tool only)
    AlwaysAllowCommand(String),
}

pub struct PermissionChecker {
    mode: PermissionMode,
    /// Tools the user has "always allowed" this session
    session_allows: std::collections::HashSet<String>,
    /// Specific bash commands the user has "always allowed" this session
    bash_command_allows: std::collections::HashSet<String>,
}

impl PermissionChecker {
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            session_allows: std::collections::HashSet::new(),
            bash_command_allows: std::collections::HashSet::new(),
        }
    }

    /// Record that the user chose "always allow" for a tool.
    pub fn always_allow(&mut self, tool_name: &str) {
        self.session_allows.insert(tool_name.to_string());
    }

    /// Record that the user chose "always allow" for a specific bash command.
    pub fn always_allow_command(&mut self, cmd: &str) {
        self.bash_command_allows.insert(cmd.to_string());
    }

    /// Check if a specific bash command is always allowed.
    pub fn is_command_allowed(&self, cmd: &str) -> bool {
        self.bash_command_allows.contains(cmd)
    }

    /// Check whether a tool invocation should be allowed.
    pub fn check(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        is_read_only: bool,
    ) -> PermissionResult {
        // Session-level always-allow overrides
        if self.session_allows.contains(tool_name) {
            return PermissionResult::Allow;
        }

        // Command-specific allows for Bash
        if tool_name == "Bash" {
            if let Some(cmd) = input["command"].as_str() {
                if self.bash_command_allows.contains(cmd) {
                    return PermissionResult::Allow;
                }
            }
        }

        match self.mode {
            PermissionMode::Bypass => PermissionResult::Allow,

            PermissionMode::Plan => {
                if is_read_only {
                    PermissionResult::Allow
                } else {
                    PermissionResult::Deny("Plan mode: write operations are disabled".to_string())
                }
            }

            PermissionMode::AcceptEdits => {
                if is_read_only || tool_name == "Write" || tool_name == "Edit" {
                    PermissionResult::Allow
                } else if tool_name == "Bash" {
                    let cmd = input["command"].as_str().unwrap_or("");
                    PermissionResult::Ask {
                        message: format!("Allow bash: {}?", truncate(cmd, 80)),
                        diff: None,
                    }
                } else {
                    PermissionResult::Allow
                }
            }

            PermissionMode::Default => {
                if is_read_only {
                    // Follow Claude Code's lead: prompt for Read and Grep, auto-allow Glob
                    match tool_name {
                        "Read" => {
                            let path = input["file_path"].as_str().unwrap_or("?");
                            PermissionResult::Ask {
                                message: format!("read: {path}"),
                                diff: None,
                            }
                        }
                        "Grep" => {
                            let pattern = input["pattern"].as_str().unwrap_or("?");
                            let path = input["path"].as_str().unwrap_or("");
                            let msg = if path.is_empty() {
                                format!("grep: \"{pattern}\"")
                            } else {
                                format!("grep: \"{pattern}\" in {path}")
                            };
                            PermissionResult::Ask {
                                message: msg,
                                diff: None,
                            }
                        }
                        "Glob" => PermissionResult::Allow,
                        "WebFetch" => {
                            let url = input["url"].as_str().unwrap_or("?");
                            PermissionResult::Ask {
                                message: format!("fetch: {url}"),
                                diff: None,
                            }
                        }
                        _ => PermissionResult::Allow,
                    }
                } else {
                    match tool_name {
                        "Bash" => {
                            let cmd = input["command"].as_str().unwrap_or("");
                            PermissionResult::Ask {
                                message: format!("bash: {}", truncate(cmd, 80)),
                                diff: None,
                            }
                        }
                        "Write" => {
                            let path = input["file_path"].as_str().unwrap_or("?");
                            PermissionResult::Ask {
                                message: format!("write: {path}"),
                                diff: None,
                            }
                        }
                        "Edit" => {
                            let path = input["file_path"].as_str().unwrap_or("?");
                            let old_string = input["old_string"].as_str().unwrap_or("");
                            let new_string = input["new_string"].as_str().unwrap_or("");

                            let diff = if !old_string.is_empty() && !new_string.is_empty() {
                                Some(generate_diff(old_string, new_string, path))
                            } else {
                                None
                            };

                            PermissionResult::Ask {
                                message: format!("edit: {path}"),
                                diff,
                            }
                        }
                        _ => PermissionResult::Ask {
                            message: tool_name.to_string(),
                            diff: None,
                        },
                    }
                }
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bypass_allows_everything() {
        let checker = PermissionChecker::new(PermissionMode::Bypass);
        let input = json!({"command": "rm -rf /"});
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn plan_denies_writes() {
        let checker = PermissionChecker::new(PermissionMode::Plan);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Deny(_)
        ));
    }

    #[test]
    fn plan_allows_reads() {
        let checker = PermissionChecker::new(PermissionMode::Plan);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(
            checker.check("Read", &input, true),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_allows_read_only() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"pattern": "*.rs"});
        assert!(matches!(
            checker.check("Glob", &input, true),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_asks_for_bash() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn default_asks_for_write() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "/tmp/test", "content": "hello"});
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn accept_edits_allows_write_and_edit() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Allow
        ));
        assert!(matches!(
            checker.check("Edit", &input, false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn accept_edits_asks_for_bash() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"command": "rm -rf /"});
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn always_allow_overrides_mode() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});

        // First call should ask
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Ask { .. }
        ));

        // After always_allow, should allow
        checker.always_allow("Bash");
        assert!(matches!(
            checker.check("Bash", &input, false),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn always_allow_is_tool_specific() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        checker.always_allow("Bash");

        let input = json!({"file_path": "/tmp/test"});
        // Write should still ask
        assert!(matches!(
            checker.check("Write", &input, false),
            PermissionResult::Ask { .. }
        ));
    }

    #[test]
    fn ask_summary_contains_command() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});
        if let PermissionResult::Ask { message, diff: _ } = checker.check("Bash", &input, false) {
            assert!(message.contains("cargo test"));
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn ask_summary_contains_file_path() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "/home/ducks/important.rs"});
        if let PermissionResult::Ask { message, diff: _ } = checker.check("Edit", &input, false) {
            assert!(message.contains("important.rs"));
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn edit_permission_includes_diff_when_fields_present() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({
            "file_path": "src/main.rs",
            "old_string": "let x = 1",
            "new_string": "let x = 2"
        });

        if let PermissionResult::Ask { message, diff } = checker.check("Edit", &input, false) {
            assert!(message.contains("src/main.rs"));
            assert!(diff.is_some(), "Diff should be generated when old_string and new_string are provided");
            let diff_content = diff.unwrap();
            assert!(diff_content.contains("src/main.rs"));
            assert!(diff_content.contains("-let x = 1"));
            assert!(diff_content.contains("+let x = 2"));
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn edit_permission_no_diff_when_fields_missing() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "src/main.rs"});

        if let PermissionResult::Ask { message, diff } = checker.check("Edit", &input, false) {
            assert!(message.contains("src/main.rs"));
            assert!(diff.is_none(), "Diff should be None when old_string/new_string are missing");
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn default_prompts_for_read_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "src/secret.rs"});

        if let PermissionResult::Ask { message, diff } = checker.check("Read", &input, true) {
            assert!(message.contains("src/secret.rs"));
            assert!(diff.is_none());
        } else {
            panic!("expected Ask for Read tool");
        }
    }

    #[test]
    fn default_prompts_for_grep_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"pattern": "SECRET_KEY", "path": "src/"});

        if let PermissionResult::Ask { message, diff } = checker.check("Grep", &input, true) {
            assert!(message.contains("src/"));
            assert!(diff.is_none());
        } else {
            panic!("expected Ask for Grep tool");
        }
    }

    #[test]
    fn default_auto_allows_glob_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"pattern": "*.rs"});

        assert!(matches!(
            checker.check("Glob", &input, true),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn default_prompts_for_webfetch_tool() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"url": "https://example.com/api"});

        if let PermissionResult::Ask { message, diff } = checker.check("WebFetch", &input, true) {
            assert!(message.contains("https://example.com/api"));
            assert!(diff.is_none());
        } else {
            panic!("expected Ask for WebFetch tool");
        }
    }
}
