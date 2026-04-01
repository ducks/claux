use serde::{Deserialize, Serialize};

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
    Ask(String),
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
}

pub struct PermissionChecker {
    mode: PermissionMode,
    /// Tools the user has "always allowed" this session
    session_allows: std::collections::HashSet<String>,
}

impl PermissionChecker {
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            session_allows: std::collections::HashSet::new(),
        }
    }

    /// Record that the user chose "always allow" for a tool.
    pub fn always_allow(&mut self, tool_name: &str) {
        self.session_allows.insert(tool_name.to_string());
    }

    /// Check whether a tool invocation should be allowed.
    pub fn check(&self, tool_name: &str, input: &serde_json::Value, is_read_only: bool) -> PermissionResult {
        // Session-level always-allow overrides
        if self.session_allows.contains(tool_name) {
            return PermissionResult::Allow;
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
                    PermissionResult::Ask(format!("Allow bash: {}?", truncate(cmd, 80)))
                } else {
                    PermissionResult::Allow
                }
            }

            PermissionMode::Default => {
                if is_read_only {
                    PermissionResult::Allow
                } else {
                    let summary = match tool_name {
                        "Bash" => {
                            let cmd = input["command"].as_str().unwrap_or("");
                            format!("bash: {}", truncate(cmd, 80))
                        }
                        "Write" => {
                            let path = input["file_path"].as_str().unwrap_or("?");
                            format!("write: {}", path)
                        }
                        "Edit" => {
                            let path = input["file_path"].as_str().unwrap_or("?");
                            format!("edit: {}", path)
                        }
                        _ => format!("{}", tool_name),
                    };
                    PermissionResult::Ask(summary)
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
        assert!(matches!(checker.check("Bash", &input, false), PermissionResult::Allow));
    }

    #[test]
    fn plan_denies_writes() {
        let checker = PermissionChecker::new(PermissionMode::Plan);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(checker.check("Write", &input, false), PermissionResult::Deny(_)));
    }

    #[test]
    fn plan_allows_reads() {
        let checker = PermissionChecker::new(PermissionMode::Plan);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(checker.check("Read", &input, true), PermissionResult::Allow));
    }

    #[test]
    fn default_allows_read_only() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"pattern": "*.rs"});
        assert!(matches!(checker.check("Glob", &input, true), PermissionResult::Allow));
    }

    #[test]
    fn default_asks_for_bash() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});
        assert!(matches!(checker.check("Bash", &input, false), PermissionResult::Ask(_)));
    }

    #[test]
    fn default_asks_for_write() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "/tmp/test", "content": "hello"});
        assert!(matches!(checker.check("Write", &input, false), PermissionResult::Ask(_)));
    }

    #[test]
    fn accept_edits_allows_write_and_edit() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"file_path": "/tmp/test"});
        assert!(matches!(checker.check("Write", &input, false), PermissionResult::Allow));
        assert!(matches!(checker.check("Edit", &input, false), PermissionResult::Allow));
    }

    #[test]
    fn accept_edits_asks_for_bash() {
        let checker = PermissionChecker::new(PermissionMode::AcceptEdits);
        let input = json!({"command": "rm -rf /"});
        assert!(matches!(checker.check("Bash", &input, false), PermissionResult::Ask(_)));
    }

    #[test]
    fn always_allow_overrides_mode() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});

        // First call should ask
        assert!(matches!(checker.check("Bash", &input, false), PermissionResult::Ask(_)));

        // After always_allow, should allow
        checker.always_allow("Bash");
        assert!(matches!(checker.check("Bash", &input, false), PermissionResult::Allow));
    }

    #[test]
    fn always_allow_is_tool_specific() {
        let mut checker = PermissionChecker::new(PermissionMode::Default);
        checker.always_allow("Bash");

        let input = json!({"file_path": "/tmp/test"});
        // Write should still ask
        assert!(matches!(checker.check("Write", &input, false), PermissionResult::Ask(_)));
    }

    #[test]
    fn ask_summary_contains_command() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"command": "cargo test"});
        if let PermissionResult::Ask(summary) = checker.check("Bash", &input, false) {
            assert!(summary.contains("cargo test"));
        } else {
            panic!("expected Ask");
        }
    }

    #[test]
    fn ask_summary_contains_file_path() {
        let checker = PermissionChecker::new(PermissionMode::Default);
        let input = json!({"file_path": "/home/ducks/important.rs"});
        if let PermissionResult::Ask(summary) = checker.check("Edit", &input, false) {
            assert!(summary.contains("important.rs"));
        } else {
            panic!("expected Ask");
        }
    }
}
