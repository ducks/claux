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

pub struct PermissionChecker {
    mode: PermissionMode,
}

impl PermissionChecker {
    pub fn new(mode: PermissionMode) -> Self {
        Self { mode }
    }

    /// Check whether a tool invocation should be allowed.
    pub fn check(&self, tool_name: &str, input: &serde_json::Value, is_read_only: bool) -> PermissionResult {
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
