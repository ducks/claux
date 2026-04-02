pub(crate) mod agent;
mod bash;
mod edit;
mod glob;
mod grep;
pub(crate) mod read;
mod web_fetch;
mod write;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::api::{Provider, ToolDefinition};

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

    /// Short human-readable summary of what this invocation does.
    /// Shown to the user while the tool runs.
    fn summarize(&self, input: &Value) -> String {
        self.name().to_string()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput>;
}

/// Registry holding all available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a registry with Agent tool using a provider factory.
    pub fn new_with_agent_factory(
        factory: agent::ProviderFactory,
        model: String,
    ) -> Self {
        Self {
            tools: vec![
                Box::new(read::ReadTool),
                Box::new(write::WriteTool),
                Box::new(edit::EditTool),
                Box::new(glob::GlobTool),
                Box::new(grep::GrepTool),
                Box::new(bash::BashTool),
                Box::new(web_fetch::WebFetchTool::new()),
                Box::new(agent::AgentTool::new(factory, model)),
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
                Box::new(web_fetch::WebFetchTool::new()),
            ],
        }
    }

    /// Create a basic registry (no Agent).
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
        let reg = ToolRegistry::without_agent();
        let defs = reg.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(!names.contains(&"Agent"));
    }

    #[test]
    fn registry_with_agent_has_agent() {
        use crate::api::AnthropicProvider;
        use crate::config::AuthMethod;

        let factory: agent::ProviderFactory = Box::new(|| {
            Box::new(AnthropicProvider::new(
                AuthMethod::ApiKey("fake".into()),
                "model",
            ))
        });
        let reg = ToolRegistry::new_with_agent_factory(factory, "model".into());
        let defs = reg.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Agent"));
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
    async fn execute_unknown_tool_errors() {
        let reg = ToolRegistry::new();
        let result = reg.execute("FakeTool", serde_json::json!({})).await;
        assert!(result.is_err());
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
