use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Tool, ToolOutput, ToolRegistry};
use crate::api;
use crate::context;
use crate::permissions::{PermissionChecker, PermissionMode};
use crate::query::Engine;

pub struct AgentTool {
    api_key: String,
    model: String,
}

impl AgentTool {
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
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

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input)?;

        // Build a child engine with restricted tools (no Agent to prevent recursion)
        let client = api::Client::new(self.api_key.clone(), &self.model);
        let tools = ToolRegistry::without_agent();
        let permissions = PermissionChecker::new(PermissionMode::Bypass); // agents run unattended

        let mut engine = Engine::new(client, tools, permissions, &self.model);

        // Build a scoped system prompt
        let base_prompt = context::build_system_prompt().await?;
        let agent_prompt = format!(
            "{}\n\n# Agent Mode\n\
             You are a sub-agent spawned to handle a specific task. \
             Complete the task and provide a clear, concise result. \
             You do NOT have access to the Agent tool (no nested agents). \
             Focus on the task and return your findings.",
            base_prompt
        );
        engine.set_system_prompt(agent_prompt);

        // Run the agent's query
        match engine.submit(&params.prompt).await {
            Ok(response) => {
                // Merge cost tracking
                let cost_summary = engine.cost.format_summary();

                let mut content = response;
                if !cost_summary.is_empty() {
                    content.push_str(&format!("\n\n[Agent {}]", cost_summary));
                }

                Ok(ToolOutput {
                    content,
                    is_error: false,
                })
            }
            Err(e) => Ok(ToolOutput {
                content: format!("Agent error: {}", e),
                is_error: true,
            }),
        }
    }
}
