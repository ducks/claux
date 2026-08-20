//! Compact, tool-specific presentation for the chat transcript.
//!
//! Tools keep ownership of their generic summaries. The TUI uses raw input to
//! add the small bits of structure that make an execution trace scannable,
//! while unknown and plugin-provided tools retain the generic fallback.

use serde_json::Value;

const DETAIL_LIMIT: usize = 100;

#[derive(Debug, PartialEq)]
pub struct ToolPresentation {
    pub summary: String,
    pub detail: Option<String>,
}

pub fn present(name: &str, fallback: &str, input: &Value) -> ToolPresentation {
    match name.to_ascii_lowercase().as_str() {
        "bash" => bash(fallback, input),
        "read" => read(fallback, input),
        "edit" => edit(fallback, input),
        "write" => write(fallback, input),
        "agent" => agent(fallback, input),
        _ => generic(fallback),
    }
}

fn bash(fallback: &str, input: &Value) -> ToolPresentation {
    let command = string(input, "command").unwrap_or(fallback);
    let description = string(input, "description").filter(|value| !value.is_empty());
    let timeout = input.get("timeout").and_then(Value::as_u64);
    let detail = match (description, timeout) {
        (Some(description), Some(timeout)) => Some(format!(
            "{} · {} timeout",
            clipped(description),
            duration(timeout)
        )),
        (Some(description), None) => Some(clipped(description)),
        (None, Some(timeout)) => Some(format!("{} timeout", duration(timeout))),
        (None, None) => None,
    };
    ToolPresentation {
        summary: format!("$ {}", clipped(command)),
        detail,
    }
}

fn read(fallback: &str, input: &Value) -> ToolPresentation {
    let path = string(input, "file_path").unwrap_or(fallback);
    let offset = input.get("offset").and_then(Value::as_u64);
    let limit = input.get("limit").and_then(Value::as_u64);
    let detail = match (offset, limit) {
        (Some(offset), Some(limit)) if limit > 0 => Some(format!(
            "lines {offset}–{}",
            offset.saturating_add(limit - 1)
        )),
        (Some(offset), None) => Some(format!("from line {offset}")),
        (None, Some(limit)) => Some(format!("first {limit} lines")),
        _ => None,
    };
    ToolPresentation {
        summary: path.to_string(),
        detail,
    }
}

fn edit(fallback: &str, input: &Value) -> ToolPresentation {
    let path = string(input, "file_path").unwrap_or(fallback);
    let old_lines = string(input, "old_string").map(line_count);
    let new_lines = string(input, "new_string").map(line_count);
    let replace_all = input
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let detail = match (old_lines, new_lines) {
        (Some(old), Some(new)) => Some(format!(
            "replace {old} line{} with {new} line{}{}",
            plural(old),
            plural(new),
            if replace_all { " · all matches" } else { "" }
        )),
        _ if replace_all => Some("replace all matches".to_string()),
        _ => None,
    };
    ToolPresentation {
        summary: path.to_string(),
        detail,
    }
}

fn write(fallback: &str, input: &Value) -> ToolPresentation {
    let path = string(input, "file_path").unwrap_or(fallback);
    let detail = string(input, "content").map(|content| {
        let lines = line_count(content);
        format!("{lines} line{} · {} bytes", plural(lines), content.len())
    });
    ToolPresentation {
        summary: path.to_string(),
        detail,
    }
}

fn agent(fallback: &str, input: &Value) -> ToolPresentation {
    let description = string(input, "description")
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback);
    let detail = string(input, "prompt")
        .filter(|prompt| *prompt != description)
        .map(clipped);
    ToolPresentation {
        summary: description.to_string(),
        detail,
    }
}

fn generic(fallback: &str) -> ToolPresentation {
    ToolPresentation {
        summary: fallback.to_string(),
        detail: None,
    }
}

fn string<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(Value::as_str)
}

fn clipped(value: &str) -> String {
    let first_line = value.lines().next().unwrap_or(value);
    if first_line.chars().count() > DETAIL_LIMIT {
        format!(
            "{}…",
            crate::utils::truncate_str(first_line, DETAIL_LIMIT - 1)
        )
    } else {
        first_line.to_string()
    }
}

fn line_count(value: &str) -> usize {
    value.lines().count().max(1)
}

fn plural(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

fn duration(milliseconds: u64) -> String {
    if milliseconds.is_multiple_of(1_000) {
        format!("{}s", milliseconds / 1_000)
    } else {
        format!("{milliseconds}ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn presents_bash_command_and_execution_hint() {
        assert_eq!(
            present(
                "Bash",
                "fallback",
                &json!({
                    "command": "cargo test --workspace",
                    "description": "Run the full test suite",
                    "timeout": 120_000
                })
            ),
            ToolPresentation {
                summary: "$ cargo test --workspace".to_string(),
                detail: Some("Run the full test suite · 120s timeout".to_string()),
            }
        );
    }

    #[test]
    fn presents_file_operations() {
        assert_eq!(
            present(
                "Read",
                "fallback",
                &json!({"file_path": "src/main.rs", "offset": 10, "limit": 20})
            ),
            ToolPresentation {
                summary: "src/main.rs".to_string(),
                detail: Some("lines 10–29".to_string()),
            }
        );
        assert_eq!(
            present(
                "Edit",
                "fallback",
                &json!({
                    "file_path": "src/main.rs",
                    "old_string": "one\ntwo",
                    "new_string": "three",
                    "replace_all": true
                })
            ),
            ToolPresentation {
                summary: "src/main.rs".to_string(),
                detail: Some("replace 2 lines with 1 line · all matches".to_string()),
            }
        );
        assert_eq!(
            present(
                "Write",
                "fallback",
                &json!({"file_path": "README.md", "content": "one\ntwo\n"})
            ),
            ToolPresentation {
                summary: "README.md".to_string(),
                detail: Some("2 lines · 8 bytes".to_string()),
            }
        );
    }

    #[test]
    fn presents_agent_task_and_preserves_unknown_tool_fallback() {
        assert_eq!(
            present(
                "Agent",
                "fallback",
                &json!({"description": "Inspect parser", "prompt": "Find the parser bug and report its cause"})
            ),
            ToolPresentation {
                summary: "Inspect parser".to_string(),
                detail: Some("Find the parser bug and report its cause".to_string()),
            }
        );
        assert_eq!(
            present("plugin_tool", "plugin summary", &json!({"anything": true})),
            ToolPresentation {
                summary: "plugin summary".to_string(),
                detail: None,
            }
        );
    }
}
