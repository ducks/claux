use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use super::{Tool, ToolOutput, ToolRegistry};
use crate::api::Provider;
use crate::command_sandbox::CommandSandbox;
use crate::context;
#[cfg(test)]
use crate::permissions::PermissionMode;
use crate::permissions::PermissionPolicy;
use crate::query::Engine;
use crate::sandbox::SandboxPolicy;

/// Factory function to create a provider for sub-agents.
pub type ProviderFactory = Box<dyn Fn() -> Box<dyn Provider> + Send + Sync>;

#[derive(Clone, Debug, serde::Serialize)]
pub struct SubAgentReport {
    pub parent_tool_use_id: String,
    pub usage: crate::cost::UsageSummary,
    pub model_rounds: Vec<crate::query::ModelTraceEntry>,
    pub tools: Vec<crate::query::ToolTraceEntry>,
    #[serde(skip)]
    pub cost: crate::cost::CostTracker,
}

pub struct AgentTool {
    max_tokens: u32,
    make_provider: ProviderFactory,
    model: String,
    metadata: crate::model::ModelMetadata,
    /// Permission mode and rules inherited from the parent session. A
    /// sub-agent runs non-interactively (no prompt to surface), so anything
    /// the parent's policy would prompt for is denied rather than auto-run;
    /// Plan's deny-all-writes, Bypass's allow-all, and deny rules are
    /// honored exactly.
    permission_policy: PermissionPolicy,
    /// Project trust inherited from the parent session. Sub-agents share the
    /// parent's working directory, so they must apply the same CLAUDE.md
    /// trust gating: an untrusted project must not inject its checked-in
    /// instructions into a sub-agent's prompt either.
    trusted: bool,
    sandbox_policy: Arc<SandboxPolicy>,
    command_sandbox: Arc<CommandSandbox>,
}

impl AgentTool {
    pub fn new(
        make_provider: ProviderFactory,
        model: String,
        metadata: crate::model::ModelMetadata,
        permission_policy: PermissionPolicy,
        trusted: bool,
        sandbox_policy: Arc<SandboxPolicy>,
        command_sandbox: Arc<CommandSandbox>,
    ) -> Self {
        Self {
            max_tokens: 16_384,
            make_provider,
            model,
            metadata,
            permission_policy,
            trusted,
            sandbox_policy,
            command_sandbox,
        }
    }
}

#[derive(Deserialize)]
struct Params {
    prompt: String,
    #[serde(default, rename = "description")]
    _description: Option<String>,
}

#[async_trait]
impl Tool for AgentTool {
    fn set_max_tokens(&mut self, max_tokens: u32) {
        self.max_tokens = max_tokens;
    }
    fn name(&self) -> &str {
        "Agent"
    }

    fn description(&self) -> &str {
        "Launch a sub-agent to handle a complex task. The agent gets its own conversation \
         context and a restricted set of tools (no nested agents). Use for independent \
         subtasks like research, file exploration, or multi-step operations."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent to perform"
                },
                "description": {
                    "type": "string",
                    "description": "Short description (3-5 words) of the task"
                }
            },
            "required": ["prompt"]
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn summarize(&self, input: &Value) -> String {
        input["description"]
            .as_str()
            .or_else(|| {
                input["prompt"].as_str().map(|p| {
                    if p.len() > 60 {
                        crate::utils::truncate_str(p, 57)
                    } else {
                        p
                    }
                })
            })
            .unwrap_or("sub-agent task")
            .to_string()
    }

    async fn execute(
        &self,
        input: Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(600);
        let sub_cancel = cancel.child_token();

        let mut provider = (self.make_provider)();
        provider.set_model(&self.model);
        let tools =
            ToolRegistry::without_agent(self.sandbox_policy.clone(), self.command_sandbox.clone());
        // Inherit the parent's permission mode. Previously hardcoded to
        // Bypass, which let a sub-agent run Bash/Write/Edit with no prompts
        // regardless of the mode the user chose - approving the Agent tool
        // once silently authorized everything it did. The sub-agent has no
        // interactive prompt, so run_turn (non-interactive) denies any tool
        // the mode would Ask about; Bypass still allows all, Plan still
        // denies all writes.
        let permissions = self.permission_policy.checker();

        let mut engine = Engine::new(provider, tools, permissions, &self.model);
        engine.disable_checkpoints();
        engine.set_max_tokens(self.max_tokens);
        engine.set_model_metadata(self.metadata);
        engine.set_auto_compact_threshold(0.8); // Default for sub-agents
        engine.set_max_rounds(50);

        let base_prompt =
            tokio::time::timeout_at(deadline, context::build_system_prompt(self.trusted))
                .await
                .map_err(|_| anyhow::anyhow!("Sub-agent context preparation timed out"))??;
        let agent_prompt = format!(
            "{base_prompt}\n\n# Agent Mode\n\
             You are a sub-agent spawned to handle a specific task. \
             Complete the task and provide a clear, concise result. \
             You do NOT have access to the Agent tool (no nested agents). \
             Focus on the task and return your findings."
        );
        engine.set_system_prompt(agent_prompt);

        // The cancellation token flows into the sub-agent's own turn loop,
        // so interrupting the parent cleanly interrupts the sub-agent's
        // in-flight tools too.
        let result = match tokio::time::timeout_at(
            deadline,
            engine.submit(&params.prompt, sub_cancel.clone()),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                sub_cancel.cancel();
                Err(anyhow::anyhow!(
                    "Sub-agent exceeded its 600-second deadline"
                ))
            }
        };
        let report = Some(Box::new(SubAgentReport {
            parent_tool_use_id: String::new(),
            usage: engine.cost.usage_summary(),
            model_rounds: engine.execution_timing().model_rounds,
            tools: engine.tool_trace().to_vec(),
            cost: engine.cost.clone(),
        }));
        match result {
            Ok(response) => {
                if cancel.is_cancelled() {
                    return Ok(ToolOutput {
                        sub_agent: report,
                        content: "Sub-agent interrupted by user.".to_string(),
                        is_error: true,
                    });
                }
                let cost_summary = engine.cost.format_summary();
                let mut content = response;
                if !cost_summary.is_empty() {
                    content.push_str(&format!("\n\n[Agent {cost_summary}]"));
                }
                Ok(ToolOutput {
                    sub_agent: report,
                    content,
                    is_error: false,
                })
            }
            Err(e) => Ok(ToolOutput {
                sub_agent: report,
                content: format!("Agent error: {e}"),
                is_error: true,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ApiEvent, Message, ToolDefinition};
    use tokio::sync::mpsc;

    /// Provider that, on the sub-agent's first turn, requests a Write to a
    /// concrete path, then ends the turn. Lets us prove the sub-agent's
    /// inherited permission mode actually gates the write.
    struct PathWriteProvider {
        path: String,
    }

    #[async_trait]
    impl Provider for PathWriteProvider {
        fn name(&self) -> &str {
            "path-write"
        }
        fn set_model(&mut self, _model: &str) {}
        async fn stream(
            &self,
            messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            max_tokens: u32,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<crate::api::ProviderStream> {
            let (tx, rx) = mpsc::channel(10);
            assert_eq!(max_tokens, 400);
            tx.send(ApiEvent::Usage(crate::api::Usage {
                input_tokens: 10,
                provider_cost_usd: Some(0.01),
                ..Default::default()
            }))
            .await
            .unwrap();
            if messages.len() <= 1 {
                let _ = tx
                    .send(ApiEvent::ToolUse {
                        id: "tu_1".into(),
                        name: "Write".into(),
                        input: json!({
                            "file_path": self.path,
                            "content": "written by sub-agent",
                        }),
                    })
                    .await;
            } else {
                tx.send(ApiEvent::Text("done".into())).await.unwrap();
            }
            let _ = tx.send(ApiEvent::Done).await;
            Ok(crate::api::ProviderStream::new(rx, cancel.child_token()))
        }
    }

    async fn run_subagent_write(
        mode: PermissionMode,
        path: &std::path::Path,
        sandbox_policy: Arc<SandboxPolicy>,
    ) {
        let path_str = path.to_str().unwrap().to_string();
        let factory: ProviderFactory = Box::new(move || {
            Box::new(PathWriteProvider {
                path: path_str.clone(),
            })
        });
        let mut tool = AgentTool::new(
            factory,
            "test".into(),
            crate::model::built_in_metadata("test"),
            PermissionPolicy::new(mode, Default::default()),
            true,
            sandbox_policy,
            Arc::new(CommandSandbox::unrestricted_for_tests()),
        );
        tool.set_max_tokens(400);
        let output = tool
            .execute(
                json!({ "prompt": "write the file" }),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("execute never returns Err");
        let report = output.sub_agent.unwrap();
        assert_eq!(report.usage.input_tokens, 20);
        assert_eq!(report.usage.cost_usd, Some(0.02));
        assert_eq!(report.usage.rounds, 2);
        assert_eq!(report.tools.len(), 1);
        assert_eq!(report.model_rounds.len(), 2);
    }

    #[tokio::test]
    async fn subagent_plan_mode_denies_write() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("should-not-exist.txt");

        run_subagent_write(
            PermissionMode::Plan,
            &target,
            Arc::new(SandboxPolicy::unrestricted_for_tests()),
        )
        .await;

        assert!(
            !target.exists(),
            "Plan mode must deny the sub-agent's write; the file was created"
        );
    }

    #[tokio::test]
    async fn subagent_bypass_mode_allows_write() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("written.txt");

        run_subagent_write(
            PermissionMode::Bypass,
            &target,
            Arc::new(SandboxPolicy::unrestricted_for_tests()),
        )
        .await;

        assert!(
            target.exists(),
            "Bypass mode should let the sub-agent write; the file is missing"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "written by sub-agent"
        );
    }

    #[tokio::test]
    async fn subagent_inherits_native_tool_filesystem_policy() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let target = parent.path().join("must-not-exist.txt");

        run_subagent_write(
            PermissionMode::Bypass,
            &target,
            Arc::new(SandboxPolicy::workspace_only(&workspace).unwrap()),
        )
        .await;

        assert!(
            !target.exists(),
            "sub-agent must not write outside its inherited workspace policy"
        );
    }
}
