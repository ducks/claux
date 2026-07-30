pub(crate) mod agent;
mod bash;
mod edit;
mod glob;
mod grep;
pub(crate) mod mcp;
pub(crate) mod read;
mod search_filter;
pub(crate) mod todo;
mod web_fetch;
mod write;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::api::ToolDefinition;
use crate::sandbox::SandboxPolicy;

/// Output from a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

fn interrupted_output() -> ToolOutput {
    ToolOutput {
        content: "Interrupted by user.".to_string(),
        is_error: true,
    }
}

fn sandbox_denied_output(error: anyhow::Error) -> ToolOutput {
    ToolOutput {
        content: error.to_string(),
        is_error: true,
    }
}

/// Every tool implements this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    fn is_read_only(&self) -> bool;

    /// Clear state that must not carry into another conversation.
    fn reset_session(&self) {}

    /// Update model-dependent tool state when the session model changes.
    fn set_model(&mut self, _model: &str) {}

    /// Short human-readable summary of what this invocation does.
    /// Shown to the user while the tool runs.
    fn summarize(&self, _input: &Value) -> String {
        self.name().to_string()
    }

    /// Execute the tool. `cancel` is signaled if the user has interrupted —
    /// long-running tools should monitor it and clean up. Tools without a
    /// natural interrupt point can ignore it.
    async fn execute(&self, input: Value, cancel: CancellationToken) -> Result<ToolOutput>;
}

/// Registry holding all available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a registry with Agent tool using a provider factory. The
    /// `permission_mode` is inherited by sub-agents the Agent tool spawns,
    /// so a sub-agent can't run with more authority than the parent session.
    pub fn new_with_agent_factory(
        factory: agent::ProviderFactory,
        model: String,
        permission_mode: crate::permissions::PermissionMode,
        sandbox_policy: Arc<SandboxPolicy>,
    ) -> Self {
        let todo_state = todo::new_todo_state();
        Self {
            tools: vec![
                Box::new(read::ReadTool::new(sandbox_policy.clone())),
                Box::new(write::WriteTool::new(sandbox_policy.clone())),
                Box::new(edit::EditTool::new(sandbox_policy.clone())),
                Box::new(glob::GlobTool::new(sandbox_policy.clone())),
                Box::new(grep::GrepTool::new(sandbox_policy.clone())),
                Box::new(bash::BashTool),
                Box::new(web_fetch::WebFetchTool::new()),
                Box::new(agent::AgentTool::new(
                    factory,
                    model,
                    permission_mode,
                    sandbox_policy,
                )),
                Box::new(todo::TodoWriteTool::new(todo_state)),
            ],
        }
    }

    /// Create a registry without Agent (for sub-agents to prevent recursion).
    pub fn without_agent(sandbox_policy: Arc<SandboxPolicy>) -> Self {
        let todo_state = todo::new_todo_state();
        Self {
            tools: vec![
                Box::new(read::ReadTool::new(sandbox_policy.clone())),
                Box::new(write::WriteTool::new(sandbox_policy.clone())),
                Box::new(edit::EditTool::new(sandbox_policy.clone())),
                Box::new(glob::GlobTool::new(sandbox_policy.clone())),
                Box::new(grep::GrepTool::new(sandbox_policy)),
                Box::new(bash::BashTool),
                Box::new(web_fetch::WebFetchTool::new()),
                Box::new(todo::TodoWriteTool::new(todo_state)),
            ],
        }
    }

    /// Create a basic registry (no Agent).
    #[cfg(test)]
    pub fn new() -> Self {
        Self::without_agent_for_tests()
    }

    #[cfg(test)]
    pub fn without_agent_for_tests() -> Self {
        Self::without_agent(Arc::new(SandboxPolicy::unrestricted_for_tests()))
    }

    /// Add external tools (e.g. from MCP servers).
    pub fn add_tools(&mut self, tools: Vec<Box<dyn Tool>>) {
        self.tools.extend(tools);
    }

    /// Clear conversation-scoped state held by registered tools.
    pub fn reset_session(&self) {
        for tool in &self.tools {
            tool.reset_session();
        }
    }

    /// Propagate a session model change to model-dependent tools.
    pub fn set_model(&mut self, model: &str) {
        for tool in &mut self.tools {
            tool.set_model(model);
        }
    }

    /// Get tool definitions for the API request.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Execute a tool by name.
    ///
    /// Never fails: an unknown tool name or a tool-level error is returned
    /// as an error ToolOutput so it becomes a tool_result block the model
    /// can see and recover from. Propagating an Err here would abort the
    /// whole turn and leave a dangling tool_use in history, which the API
    /// rejects on the next request.
    pub async fn execute(&self, name: &str, input: Value, cancel: CancellationToken) -> ToolOutput {
        let Some(tool) = self.tools.iter().find(|t| t.name() == name) else {
            let available: Vec<&str> = self.tools.iter().map(|t| t.name()).collect();
            return ToolOutput {
                content: format!(
                    "Unknown tool: {name}. Available tools: {}",
                    available.join(", ")
                ),
                is_error: true,
            };
        };

        match tool.execute(input, cancel).await {
            Ok(output) => output,
            Err(e) => ToolOutput {
                content: format!("Tool {name} failed: {e}"),
                is_error: true,
            },
        }
    }

    /// Get a human-readable summary of what the tool invocation will do.
    pub fn summarize(&self, name: &str, input: &Value) -> String {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.summarize(input))
            .unwrap_or_else(|| name.to_string())
    }

    /// Check if a tool is read-only.
    pub fn is_read_only(&self, name: &str) -> bool {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .is_some_and(|t| t.is_read_only())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn registry_has_core_tools() {
        let reg = ToolRegistry::new();
        let defs = reg.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Read"));
        assert!(names.contains(&"Write"));
        assert!(names.contains(&"Edit"));
        assert!(names.contains(&"Glob"));
        assert!(names.contains(&"Grep"));
        assert!(names.contains(&"Bash"));
    }

    #[test]
    fn registry_without_agent_has_no_agent() {
        let reg = ToolRegistry::without_agent_for_tests();
        let defs = reg.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(!names.contains(&"Agent"));
    }

    #[test]
    fn registry_with_agent_has_agent() {
        use crate::api::AnthropicProvider;
        use crate::config::AnthropicApiKey;

        let factory: agent::ProviderFactory = Box::new(|| {
            Box::new(AnthropicProvider::new(
                AnthropicApiKey::new("fake".into()),
                "model",
            ))
        });
        let reg = ToolRegistry::new_with_agent_factory(
            factory,
            "model".into(),
            crate::permissions::PermissionMode::Default,
            Arc::new(SandboxPolicy::unrestricted_for_tests()),
        );
        let defs = reg.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Agent"));
    }

    #[tokio::test]
    async fn native_tools_enforce_workspace_roots() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::write(workspace.join("inside.txt"), "inside").unwrap();
        let outside = parent.path().join("outside.txt");
        std::fs::write(&outside, "outside").unwrap();
        let registry = ToolRegistry::without_agent(Arc::new(
            SandboxPolicy::workspace_only(&workspace).unwrap(),
        ));
        let cancel = CancellationToken::new();

        let inside_read = registry
            .execute(
                "Read",
                json!({"file_path": workspace.join("inside.txt")}),
                cancel.clone(),
            )
            .await;
        assert!(!inside_read.is_error);

        let outside_read = registry
            .execute("Read", json!({"file_path": &outside}), cancel.clone())
            .await;
        assert!(outside_read.is_error);
        assert!(outside_read.content.contains("sandbox denied read"));

        let outside_write = registry
            .execute(
                "Write",
                json!({
                    "file_path": parent.path().join("created-outside.txt"),
                    "content": "must not be written"
                }),
                cancel,
            )
            .await;
        assert!(outside_write.is_error);
        assert!(outside_write.content.contains("sandbox denied write"));
        assert!(!parent.path().join("created-outside.txt").exists());
    }

    #[tokio::test]
    async fn unrestricted_native_tools_allow_paths_outside_the_workspace() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let outside = parent.path().join("outside.txt");
        std::fs::write(&outside, "outside").unwrap();
        let registry =
            ToolRegistry::without_agent(Arc::new(SandboxPolicy::unrestricted(&workspace).unwrap()));

        let read = registry
            .execute(
                "Read",
                json!({"file_path": &outside}),
                CancellationToken::new(),
            )
            .await;
        assert!(!read.is_error);
        assert!(read.content.contains("outside"));

        let created = parent.path().join("created-outside.txt");
        let write = registry
            .execute(
                "Write",
                json!({"file_path": &created, "content": "created"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!write.is_error);
        assert_eq!(std::fs::read_to_string(created).unwrap(), "created");
    }

    #[tokio::test]
    async fn glob_rejects_patterns_that_escape_the_workspace() {
        let workspace = tempfile::tempdir().unwrap();
        let registry = ToolRegistry::without_agent(Arc::new(
            SandboxPolicy::workspace_only(workspace.path()).unwrap(),
        ));

        let output = registry
            .execute(
                "Glob",
                json!({"pattern": "../**/*", "path": workspace.path()}),
                CancellationToken::new(),
            )
            .await;

        assert!(output.is_error);
        assert!(output
            .content
            .contains("search pattern escapes the authorized base"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn native_tools_reject_symlink_escapes() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let outside = parent.path().join("outside");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, workspace.join("escape")).unwrap();
        let registry = ToolRegistry::without_agent(Arc::new(
            SandboxPolicy::workspace_only(&workspace).unwrap(),
        ));

        let read = registry
            .execute(
                "Read",
                json!({"file_path": workspace.join("escape/secret.txt")}),
                CancellationToken::new(),
            )
            .await;
        assert!(read.is_error);
        assert!(read.content.contains("sandbox denied read"));

        let write = registry
            .execute(
                "Write",
                json!({
                    "file_path": workspace.join("escape/new.txt"),
                    "content": "must not be written"
                }),
                CancellationToken::new(),
            )
            .await;
        assert!(write.is_error);
        assert!(write.content.contains("sandbox denied write"));
        assert!(!outside.join("new.txt").exists());
    }

    #[test]
    fn read_tools_are_read_only() {
        let reg = ToolRegistry::new();
        assert!(reg.is_read_only("Read"));
        assert!(reg.is_read_only("Glob"));
        assert!(reg.is_read_only("Grep"));
    }

    #[test]
    fn write_tools_are_not_read_only() {
        let reg = ToolRegistry::new();
        assert!(!reg.is_read_only("Write"));
        assert!(!reg.is_read_only("Edit"));
        assert!(!reg.is_read_only("Bash"));
    }

    #[test]
    fn unknown_tool_is_not_read_only() {
        let reg = ToolRegistry::new();
        assert!(!reg.is_read_only("NonexistentTool"));
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_error_output() {
        let reg = ToolRegistry::new();
        let output = reg
            .execute(
                "FakeTool",
                serde_json::json!({}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("Unknown tool: FakeTool"));
        // The model should be able to self-correct from the message
        assert!(output.content.contains("Read"));
    }

    #[tokio::test]
    async fn execute_bad_params_returns_error_output() {
        let reg = ToolRegistry::new();
        // Read requires file_path; missing it must not abort the turn
        let output = reg
            .execute(
                "Read",
                serde_json::json!({}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("Read"));
    }

    #[test]
    fn all_tools_have_valid_schemas() {
        let reg = ToolRegistry::new();
        for def in reg.definitions() {
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            assert_eq!(def.input_schema["type"], "object");
            assert!(def.input_schema.get("properties").is_some());
        }
    }
}
