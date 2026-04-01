pub(crate) mod agent;
mod bash;
mod edit;
mod glob;
mod grep;
pub(crate) mod read;
mod write;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::api::ToolDefinition;

/// Output from a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

/// Every tool implements this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    fn is_read_only(&self) -> bool;

    async fn execute(&self, input: Value) -> Result<ToolOutput>;
}

/// Registry holding all available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a registry with all tools including Agent.
    pub fn new_with_agent(api_key: String, model: String) -> Self {
        Self {
            tools: vec![
                Box::new(read::ReadTool),
                Box::new(write::WriteTool),
                Box::new(edit::EditTool),
                Box::new(glob::GlobTool),
                Box::new(grep::GrepTool),
                Box::new(bash::BashTool),
                Box::new(agent::AgentTool::new(api_key, model)),
            ],
        }
    }

    /// Create a registry without Agent (for sub-agents to prevent recursion).
    pub fn without_agent() -> Self {
        Self {
            tools: vec![
                Box::new(read::ReadTool),
                Box::new(write::WriteTool),
                Box::new(edit::EditTool),
                Box::new(glob::GlobTool),
                Box::new(grep::GrepTool),
                Box::new(bash::BashTool),
            ],
        }
    }

    /// Create a basic registry (no Agent). Kept for backwards compat.
    pub fn new() -> Self {
        Self::without_agent()
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
    pub async fn execute(&self, name: &str, input: Value) -> Result<ToolOutput> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| anyhow::anyhow!("unknown tool: {}", name))?;

        tool.execute(input).await
    }

    /// Check if a tool is read-only.
    pub fn is_read_only(&self, name: &str) -> bool {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .is_some_and(|t| t.is_read_only())
    }
}
