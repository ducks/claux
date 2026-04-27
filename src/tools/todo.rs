use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::{Tool, ToolOutput};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl std::fmt::Display for TodoStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TodoStatus::Pending => write!(f, "pending"),
            TodoStatus::InProgress => write!(f, "in_progress"),
            TodoStatus::Completed => write!(f, "completed"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
    #[serde(rename = "activeForm")]
    pub active_form: String,
}

/// Shared todo state across tool calls within a session.
pub type TodoState = Arc<Mutex<Vec<TodoItem>>>;

pub fn new_todo_state() -> TodoState {
    Arc::new(Mutex::new(Vec::new()))
}

pub struct TodoWriteTool {
    state: TodoState,
}

impl TodoWriteTool {
    pub fn new(state: TodoState) -> Self {
        Self { state }
    }
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "TodoWrite"
    }

    fn description(&self) -> &str {
        "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user. It also helps the user understand the progress of the task and overall progress of their requests.\n\nTask States:\n- pending: Task not yet started\n- in_progress: Currently working on (limit to ONE task at a time)\n- completed: Task finished successfully\n\nEach task must have:\n- content: The imperative form describing what needs to be done (e.g., \"Run tests\")\n- status: One of pending, in_progress, completed\n- activeForm: The present continuous form shown during execution (e.g., \"Running tests\")\n\nMark tasks complete IMMEDIATELY after finishing. Do not batch up completions. Remove tasks that are no longer relevant."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "description": "The updated todo list",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "minLength": 1
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            },
                            "activeForm": {
                                "type": "string",
                                "minLength": 1
                            }
                        },
                        "required": ["content", "status", "activeForm"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["todos"],
            "additionalProperties": false
        })
    }

    fn is_read_only(&self) -> bool {
        true // doesn't modify files
    }

    fn summarize(&self, input: &Value) -> String {
        let count = input
            .get("todos")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        format!("Updating todo list ({count} items)")
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let todos: Vec<TodoItem> = serde_json::from_value(
            input
                .get("todos")
                .cloned()
                .unwrap_or(Value::Array(vec![])),
        )?;

        let old_todos = {
            let state = self.state.lock().unwrap();
            state.clone()
        };

        // If all todos are completed, clear the list
        let all_done = todos.iter().all(|t| t.status == TodoStatus::Completed);
        let new_todos = if all_done { Vec::new() } else { todos.clone() };

        {
            let mut state = self.state.lock().unwrap();
            *state = new_todos;
        }

        // Build a human-readable summary of changes
        let mut summary = String::new();

        // Show what changed
        let added: Vec<_> = todos
            .iter()
            .filter(|t| !old_todos.iter().any(|o| o.content == t.content))
            .collect();
        let completed: Vec<_> = todos
            .iter()
            .filter(|t| {
                t.status == TodoStatus::Completed
                    && old_todos
                        .iter()
                        .any(|o| o.content == t.content && o.status != TodoStatus::Completed)
            })
            .collect();
        let started: Vec<_> = todos
            .iter()
            .filter(|t| {
                t.status == TodoStatus::InProgress
                    && old_todos
                        .iter()
                        .any(|o| o.content == t.content && o.status != TodoStatus::InProgress)
            })
            .collect();

        if !added.is_empty() {
            summary.push_str("Added:\n");
            for t in &added {
                summary.push_str(&format!("  + {}\n", t.content));
            }
        }
        if !started.is_empty() {
            summary.push_str("Started:\n");
            for t in &started {
                summary.push_str(&format!("  > {}\n", t.active_form));
            }
        }
        if !completed.is_empty() {
            summary.push_str("Completed:\n");
            for t in &completed {
                summary.push_str(&format!("  ✓ {}\n", t.content));
            }
        }

        if all_done && !todos.is_empty() {
            summary.push_str("\nAll tasks completed. List cleared.");
        }

        // Show current state
        if !all_done {
            summary.push_str("\nCurrent tasks:\n");
            for t in &todos {
                let marker = match t.status {
                    TodoStatus::Pending => "○",
                    TodoStatus::InProgress => "◉",
                    TodoStatus::Completed => "✓",
                };
                summary.push_str(&format!("  {} {}\n", marker, t.content));
            }
        }

        Ok(ToolOutput {
            content: summary,
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool() -> (TodoWriteTool, TodoState) {
        let state = new_todo_state();
        let tool = TodoWriteTool::new(state.clone());
        (tool, state)
    }

    #[tokio::test]
    async fn test_add_todos() {
        let (tool, state) = make_tool();

        let input = serde_json::json!({
            "todos": [
                { "content": "Run tests", "status": "pending", "activeForm": "Running tests" },
                { "content": "Fix bugs", "status": "pending", "activeForm": "Fixing bugs" }
            ]
        });

        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Run tests"));

        let todos = state.lock().unwrap();
        assert_eq!(todos.len(), 2);
    }

    #[tokio::test]
    async fn test_complete_todo() {
        let (tool, state) = make_tool();

        // Add todos
        tool.execute(serde_json::json!({
            "todos": [
                { "content": "Run tests", "status": "in_progress", "activeForm": "Running tests" },
                { "content": "Fix bugs", "status": "pending", "activeForm": "Fixing bugs" }
            ]
        }))
        .await
        .unwrap();

        // Complete one
        let result = tool
            .execute(serde_json::json!({
                "todos": [
                    { "content": "Run tests", "status": "completed", "activeForm": "Running tests" },
                    { "content": "Fix bugs", "status": "in_progress", "activeForm": "Fixing bugs" }
                ]
            }))
            .await
            .unwrap();

        assert!(result.content.contains("Completed"));
        let todos = state.lock().unwrap();
        assert_eq!(todos.len(), 2);
    }

    #[tokio::test]
    async fn test_all_completed_clears_list() {
        let (tool, state) = make_tool();

        tool.execute(serde_json::json!({
            "todos": [
                { "content": "Run tests", "status": "completed", "activeForm": "Running tests" },
                { "content": "Fix bugs", "status": "completed", "activeForm": "Fixing bugs" }
            ]
        }))
        .await
        .unwrap();

        let todos = state.lock().unwrap();
        assert!(todos.is_empty());
    }

    #[tokio::test]
    async fn test_is_read_only() {
        let (tool, _) = make_tool();
        assert!(tool.is_read_only());
    }

    #[tokio::test]
    async fn test_schema_is_valid() {
        let (tool, _) = make_tool();
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["todos"].is_object());
    }
}
