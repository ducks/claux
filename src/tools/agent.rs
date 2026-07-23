use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{Tool, ToolOutput, ToolRegistry};
use crate::api::Provider;
use crate::context;
use crate::permissions::{PermissionChecker, PermissionMode};
use crate::query::Engine;

/// Factory function to create a provider for sub-agents.
pub type ProviderFactory = Box<dyn Fn() -> Box<dyn Provider> + Send + Sync>;

pub struct AgentTool {
    make_provider: ProviderFactory,
    model: String,
    /// Permission mode inherited from the parent session. A sub-agent runs
    /// non-interactively (no prompt to surface), so anything the parent's
    /// mode would prompt for is denied rather than auto-run — but Plan's
    /// deny-all-writes and Bypass's allow-all are honored exactly.
    permission_mode: PermissionMode,
}

impl AgentTool {
    pub fn new(
        make_provider: ProviderFactory,
        model: String,
        permission_mode: PermissionMode,
    ) -> Self {
        Self {
            make_provider,
            model,
            permission_mode,
        }
    }
}

#[derive(Deserialize)]
struct Params {
    prompt: String,
    #[serde(default)]
    description: Option<String>,
}

#[async_trait]
impl Tool for AgentTool {
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

        let provider = (self.make_provider)();
        let tools = ToolRegistry::without_agent();
        // Inherit the parent's permission mode. Previously hardcoded to
        // Bypass, which let a sub-agent run Bash/Write/Edit with no prompts
        // regardless of the mode the user chose - approving the Agent tool
        // once silently authorized everything it did. The sub-agent has no
        // interactive prompt, so run_turn (non-interactive) denies any tool
        // the mode would Ask about; Bypass still allows all, Plan still
        // denies all writes.
        let permissions = PermissionChecker::new(self.permission_mode);

        let mut engine = Engine::new(provider, tools, permissions, &self.model);
        engine.set_auto_compact_threshold(0.8); // Default for sub-agents

        let base_prompt = context::build_system_prompt().await?;
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
        match engine.submit(&params.prompt, cancel.clone()).await {
            Ok(response) => {
                if cancel.is_cancelled() {
                    return Ok(ToolOutput {
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
                    content,
                    is_error: false,
                })
            }
            Err(e) => Ok(ToolOutput {
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
        fn model(&self) -> &str {
            "test"
        }
        fn set_model(&mut self, _model: &str) {}
        async fn stream(
            &self,
            messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _max_tokens: u32,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<crate::api::ProviderStream> {
            let (tx, rx) = mpsc::channel(10);
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
            }
            let _ = tx.send(ApiEvent::Done).await;
            Ok(crate::api::ProviderStream::new(rx, cancel.child_token()))
        }
    }

    async fn run_subagent_write(mode: PermissionMode, path: &std::path::Path) {
        let path_str = path.to_str().unwrap().to_string();
        let factory: ProviderFactory = Box::new(move || {
            Box::new(PathWriteProvider {
                path: path_str.clone(),
            })
        });
        let tool = AgentTool::new(factory, "test".into(), mode);
        tool.execute(
            json!({ "prompt": "write the file" }),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("execute never returns Err");
    }

    #[tokio::test]
    async fn subagent_plan_mode_denies_write() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("should-not-exist.txt");

        run_subagent_write(PermissionMode::Plan, &target).await;

        assert!(
            !target.exists(),
            "Plan mode must deny the sub-agent's write; the file was created"
        );
    }

    #[tokio::test]
    async fn subagent_bypass_mode_allows_write() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("written.txt");

        run_subagent_write(PermissionMode::Bypass, &target).await;

        assert!(
            target.exists(),
            "Bypass mode should let the sub-agent write; the file is missing"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "written by sub-agent"
        );
    }
}
