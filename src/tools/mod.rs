pub(crate) mod agent;
mod bash;
mod edit;
mod glob;
mod grep;
pub(crate) mod jobs;
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
use crate::command_sandbox::CommandSandbox;
use crate::sandbox::SandboxPolicy;

/// Output from a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub sub_agent: Option<Box<agent::SubAgentReport>>,
}

fn interrupted_output() -> ToolOutput {
    ToolOutput {
        sub_agent: None,
        content: "Interrupted by user.".to_string(),
        is_error: true,
    }
}

fn sandbox_denied_output(error: anyhow::Error) -> ToolOutput {
    ToolOutput {
        sub_agent: None,
        content: error.to_string(),
        is_error: true,
    }
}

/// Every tool implements this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    fn set_max_tokens(&mut self, _max_tokens: u32) {}
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    fn is_read_only(&self) -> bool;

    /// Present only for tools owned by an MCP connection.
    fn mcp_server(&self) -> Option<&str> {
        None
    }

    /// Clear state that must not carry into another conversation.
    fn reset_session(&self) {}

    /// Short human-readable summary of what this invocation does.
    /// Shown to the user while the tool runs.
    fn summarize(&self, _input: &Value) -> String {
        self.name().to_string()
    }

    /// Execute the tool. `cancel` is signaled if the user has interrupted —
    /// long-running tools should monitor it and clean up. Tools without a
    /// natural interrupt point can ignore it.
    async fn execute(&self, input: Value, cancel: CancellationToken) -> Result<ToolOutput>;

    async fn execute_with_progress(
        &self,
        input: Value,
        cancel: CancellationToken,
        _progress: Option<tokio::sync::watch::Sender<String>>,
    ) -> Result<ToolOutput> {
        self.execute(input, cancel).await
    }
}

/// Registry holding all available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    pub jobs: Arc<jobs::JobManager>,
    live_mcp: Option<(
        Vec<crate::config::McpServerConfig>,
        crate::config::ProjectTrust,
    )>,
}

impl ToolRegistry {
    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        for tool in &mut self.tools {
            tool.set_max_tokens(max_tokens);
        }
    }
    /// Create a registry with Agent tool using a provider factory. The
    /// permission policy (mode and rules) is inherited by sub-agents the
    /// Agent tool spawns, so a sub-agent can't run with more authority than
    /// the parent session.
    pub fn new_with_agent_factory(
        factory: agent::ProviderFactory,
        model: String,
        metadata: crate::model::ModelMetadata,
        permission_policy: crate::permissions::PermissionPolicy,
        trusted: bool,
        sandbox_policy: Arc<SandboxPolicy>,
        command_sandbox: Arc<CommandSandbox>,
    ) -> Self {
        let todo_state = todo::new_todo_state();
        let jobs = Arc::new(jobs::JobManager::default());
        Self {
            tools: vec![
                Box::new(read::ReadTool::new(sandbox_policy.clone())),
                Box::new(write::WriteTool::new(sandbox_policy.clone())),
                Box::new(edit::EditTool::new(sandbox_policy.clone())),
                Box::new(glob::GlobTool::new(sandbox_policy.clone())),
                Box::new(grep::GrepTool::new(sandbox_policy.clone())),
                Box::new(bash::BashTool::with_jobs(
                    command_sandbox.clone(),
                    jobs.clone(),
                )),
                Box::new(jobs::JobsTool(jobs.clone())),
                Box::new(web_fetch::WebFetchTool::new()),
                Box::new(agent::AgentTool::new(
                    factory,
                    model,
                    metadata,
                    permission_policy,
                    trusted,
                    sandbox_policy,
                    command_sandbox,
                )),
                Box::new(todo::TodoWriteTool::new(todo_state)),
            ],
            jobs,
            live_mcp: None,
        }
    }

    /// Create a registry without Agent (for sub-agents to prevent recursion).
    pub fn without_agent(
        sandbox_policy: Arc<SandboxPolicy>,
        command_sandbox: Arc<CommandSandbox>,
    ) -> Self {
        let todo_state = todo::new_todo_state();
        let jobs = Arc::new(jobs::JobManager::default());
        Self {
            tools: vec![
                Box::new(read::ReadTool::new(sandbox_policy.clone())),
                Box::new(write::WriteTool::new(sandbox_policy.clone())),
                Box::new(edit::EditTool::new(sandbox_policy.clone())),
                Box::new(glob::GlobTool::new(sandbox_policy.clone())),
                Box::new(grep::GrepTool::new(sandbox_policy)),
                Box::new(bash::BashTool::with_jobs(command_sandbox, jobs.clone())),
                Box::new(jobs::JobsTool(jobs.clone())),
                Box::new(web_fetch::WebFetchTool::new()),
                Box::new(todo::TodoWriteTool::new(todo_state)),
            ],
            jobs,
            live_mcp: None,
        }
    }

    /// Create a basic registry (no Agent).
    #[cfg(test)]
    pub fn new() -> Self {
        Self::without_agent_for_tests()
    }

    #[cfg(test)]
    pub fn without_agent_for_tests() -> Self {
        Self::without_agent(
            Arc::new(SandboxPolicy::unrestricted_for_tests()),
            Arc::new(CommandSandbox::unrestricted_for_tests()),
        )
    }

    /// Add external tools (e.g. from MCP servers).
    pub fn add_tools(&mut self, tools: Vec<Box<dyn Tool>>) -> Result<()> {
        let mut names: std::collections::HashSet<String> =
            self.tools.iter().map(|t| t.name().to_string()).collect();
        for tool in &tools {
            anyhow::ensure!(
                names.insert(tool.name().to_string()),
                "Duplicate tool name: {}",
                tool.name()
            );
        }
        self.tools.extend(tools);
        Ok(())
    }

    pub fn configure_live_mcp(&mut self, config: &crate::config::Config) {
        self.live_mcp = config
            .project_trust
            .clone()
            .map(|trust| (config.mcp_servers.clone(), trust));
    }

    fn mcp_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .tools
            .iter()
            .filter_map(|t| t.mcp_server().map(str::to_owned))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    fn live_mcp_config(&self, name: &str) -> Result<crate::config::McpServerConfig> {
        let (global, trust) = self
            .live_mcp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MCP configuration is unavailable"))?;
        let mut configs = global.clone();
        configs.extend(crate::config::load_mcp_json(trust));
        let mut matches = configs.into_iter().filter(|config| config.name == name);
        let config = matches.next().ok_or_else(|| {
            anyhow::anyhow!(
                "No configured MCP server by that name. Project .mcp.json requires project trust."
            )
        })?;
        anyhow::ensure!(
            matches.next().is_none(),
            "Duplicate MCP server names must be resolved"
        );
        Ok(config)
    }

    /// Owner-only management, invoked between turns by the slash-command UI.
    /// Global config is the session snapshot; trusted project config is read fresh.
    pub async fn manage_mcp(&mut self, args: &str) -> Result<(String, bool)> {
        let words: Vec<_> = args.split_whitespace().collect();
        match words.as_slice() {
            [] | ["list"] => {
                let names = self.mcp_names();
                Ok((
                    if names.is_empty() {
                        "No MCP servers with tools connected.".into()
                    } else {
                        format!("Connected MCP servers: {}", names.join(", "))
                    },
                    false,
                ))
            }
            ["load", name] => {
                anyhow::ensure!(
                    !self.mcp_names().iter().any(|n| n == name),
                    "MCP server already connected; unload it first"
                );
                anyhow::ensure!(
                    self.mcp_names().len() < 16,
                    "Live MCP server limit (16) reached"
                );
                let config = self.live_mcp_config(name)?;
                let tools = mcp::connect_live_server(&config).await?;
                anyhow::ensure!(!tools.is_empty(), "MCP server advertised no tools");
                let count = tools.len();
                self.add_tools(tools)?;
                Ok((format!("Loaded MCP server {name} ({count} tools)."), true))
            }
            ["unload", name] => {
                let before = self.tools.len();
                self.tools.retain(|tool| tool.mcp_server() != Some(*name));
                anyhow::ensure!(self.tools.len() < before, "MCP server is not connected");
                Ok((
                    format!("Unloaded MCP server {name}; connection shutdown requested."),
                    true,
                ))
            }
            _ => anyhow::bail!("Usage: /mcp [list | load NAME | unload NAME]"),
        }
    }

    /// Clear conversation-scoped state held by registered tools.
    pub fn reset_session(&self) {
        self.jobs.reset();
        for tool in &self.tools {
            tool.reset_session();
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
    #[cfg(test)]
    pub async fn execute(&self, name: &str, input: Value, cancel: CancellationToken) -> ToolOutput {
        self.execute_with_progress(name, input, cancel, None).await
    }

    pub async fn execute_with_progress(
        &self,
        name: &str,
        input: Value,
        cancel: CancellationToken,
        progress: Option<tokio::sync::watch::Sender<String>>,
    ) -> ToolOutput {
        let Some(tool) = self.tools.iter().find(|t| t.name() == name) else {
            let available: Vec<&str> = self.tools.iter().map(|t| t.name()).collect();
            return ToolOutput {
                sub_agent: None,
                content: format!(
                    "Unknown tool: {name}. Available tools: {}",
                    available.join(", ")
                ),
                is_error: true,
            };
        };

        match tool.execute_with_progress(input, cancel, progress).await {
            Ok(output) => output,
            Err(e) => ToolOutput {
                sub_agent: None,
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

    struct OwnedMcpTool {
        name: String,
        server: String,
    }

    #[async_trait]
    impl Tool for OwnedMcpTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "test"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        fn is_read_only(&self) -> bool {
            true
        }
        fn mcp_server(&self) -> Option<&str> {
            Some(&self.server)
        }
        async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolOutput> {
            Ok(ToolOutput {
                content: "ok".into(),
                is_error: false,
                sub_agent: None,
            })
        }
    }

    #[tokio::test]
    async fn live_mcp_unload_uses_ownership_and_preserves_other_tools() {
        let mut reg = ToolRegistry::without_agent_for_tests();
        for server in ["a", "a__b"] {
            reg.add_tools(vec![Box::new(OwnedMcpTool {
                name: format!("mcp__{server}__echo"),
                server: server.into(),
            })])
            .unwrap();
        }
        assert_eq!(reg.mcp_names(), vec!["a", "a__b"]);
        assert!(!reg.manage_mcp("list").await.unwrap().1);
        assert!(reg.manage_mcp("load a").await.is_err());
        assert!(reg.manage_mcp("unload a").await.unwrap().1);
        assert_eq!(reg.mcp_names(), vec!["a__b"]);
        assert!(reg.definitions().iter().any(|d| d.name == "Read"));
        assert!(reg.manage_mcp("unload a").await.is_err());
        assert!(reg.manage_mcp("load missing extra").await.is_err());
    }

    #[tokio::test]
    #[ignore = "requires python3 on PATH; manual stdio MCP smoke test"]
    async fn live_mcp_stdio_round_trip_and_reload() {
        use crate::config::{Config, McpServerConfig, ProjectTrust};
        let dir = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.project_trust = Some(ProjectTrust::for_test(dir.path().to_path_buf(), false));
        config.mcp_servers.push(McpServerConfig {
            name: "local-slug".into(),
            command: "python3".into(),
            args: vec![format!(
                "{}/examples/live-mcp/server.py",
                env!("CARGO_MANIFEST_DIR")
            )],
            timeout_seconds: Some(5),
            env: Default::default(),
        });
        let mut reg = ToolRegistry::without_agent_for_tests();
        reg.configure_live_mcp(&config);
        for _ in 0..2 {
            assert!(reg.manage_mcp("load local-slug").await.unwrap().1);
            let output = reg
                .execute(
                    "mcp__local-slug__slugify",
                    json!({"text": "Hello Live Tools"}),
                    CancellationToken::new(),
                )
                .await;
            assert!(!output.is_error, "{}", output.content);
            assert_eq!(output.content, "hello-live-tools");
            reg.manage_mcp("unload local-slug").await.unwrap();
            assert!(
                reg.execute(
                    "mcp__local-slug__slugify",
                    json!({}),
                    CancellationToken::new()
                )
                .await
                .is_error
            );
        }
    }

    #[test]
    fn live_mcp_project_reload_obeys_trust_and_rejects_duplicates() {
        use crate::config::{Config, ProjectTrust};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".mcp.json");
        std::fs::write(
            &path,
            r#"{"mcpServers":{"local":{"command":"version-one"}}}"#,
        )
        .unwrap();
        let mut config = Config::default();
        config.project_trust = Some(ProjectTrust::for_test(dir.path().to_path_buf(), false));
        let mut reg = ToolRegistry::without_agent_for_tests();
        reg.configure_live_mcp(&config);
        assert!(reg.live_mcp_config("local").is_err());
        config.project_trust = Some(ProjectTrust::for_test(dir.path().to_path_buf(), true));
        reg.configure_live_mcp(&config);
        assert_eq!(reg.live_mcp_config("local").unwrap().command, "version-one");
        std::fs::write(
            &path,
            r#"{"mcpServers":{"local":{"command":"version-two"}}}"#,
        )
        .unwrap();
        assert_eq!(reg.live_mcp_config("local").unwrap().command, "version-two");
        config
            .mcp_servers
            .push(reg.live_mcp_config("local").unwrap());
        reg.configure_live_mcp(&config);
        assert!(reg
            .live_mcp_config("local")
            .unwrap_err()
            .to_string()
            .contains("Duplicate"));
    }

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
            crate::model::built_in_metadata("model"),
            crate::permissions::PermissionPolicy::default(),
            true,
            Arc::new(SandboxPolicy::unrestricted_for_tests()),
            Arc::new(CommandSandbox::unrestricted_for_tests()),
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
        let registry = ToolRegistry::without_agent(
            Arc::new(SandboxPolicy::workspace_only(&workspace).unwrap()),
            Arc::new(CommandSandbox::unrestricted_for_tests()),
        );
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
        let registry = ToolRegistry::without_agent(
            Arc::new(SandboxPolicy::unrestricted(&workspace).unwrap()),
            Arc::new(CommandSandbox::unrestricted_for_tests()),
        );

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
        let registry = ToolRegistry::without_agent(
            Arc::new(SandboxPolicy::workspace_only(workspace.path()).unwrap()),
            Arc::new(CommandSandbox::unrestricted_for_tests()),
        );

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
        let registry = ToolRegistry::without_agent(
            Arc::new(SandboxPolicy::workspace_only(&workspace).unwrap()),
            Arc::new(CommandSandbox::unrestricted_for_tests()),
        );

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
