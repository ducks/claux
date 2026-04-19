use anyhow::Result;
use async_trait::async_trait;
use mcp_client::{
    ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
    Transport,
};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use super::{Tool, ToolOutput};
use crate::config::McpServerConfig;

type McpClientType = McpClient<
    tower::timeout::Timeout<
        McpService<
            <StdioTransport as Transport>::Handle,
        >,
    >,
>;

/// A tool backed by an MCP server.
/// Wraps one tool from an MCP server's tools/list response.
pub struct McpTool {
    server_name: String,
    tool_name: String,
    tool_description: String,
    tool_schema: Value,
    client: Arc<Mutex<McpClientType>>,
}

impl McpTool {
    pub fn new(
        server_name: String,
        tool_name: String,
        tool_description: String,
        tool_schema: Value,
        client: Arc<Mutex<McpClientType>>,
    ) -> Self {
        Self {
            server_name,
            tool_name,
            tool_description,
            tool_schema,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn input_schema(&self) -> Value {
        self.tool_schema.clone()
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn summarize(&self, _input: &Value) -> String {
        format!("mcp:{} {}", self.server_name, self.tool_name)
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let client = self.client.lock().await;
        let result = client.call_tool(&self.tool_name, input).await;

        match result {
            Ok(call_result) => {
                let text = call_result
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        mcp_spec::content::Content::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                Ok(ToolOutput {
                    content: text,
                    is_error: call_result.is_error.unwrap_or(false),
                })
            }
            Err(e) => Ok(ToolOutput {
                content: format!("MCP error: {e}"),
                is_error: true,
            }),
        }
    }
}

/// Connect to all configured MCP servers and return their tools.
pub async fn connect_mcp_servers(
    configs: &[McpServerConfig],
) -> Vec<Box<dyn Tool>> {
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();

    for config in configs {
        match connect_server(config).await {
            Ok(server_tools) => {
                tracing::info!(
                    "MCP server '{}': {} tools discovered",
                    config.name,
                    server_tools.len()
                );
                tools.extend(server_tools);
            }
            Err(e) => {
                tracing::error!("Failed to connect to MCP server '{}': {e}", config.name);
            }
        }
    }

    tools
}

async fn connect_server(config: &McpServerConfig) -> Result<Vec<Box<dyn Tool>>> {
    let transport = StdioTransport::new(
        &config.command,
        config.args.clone(),
        config.env.clone(),
    );

    let handle = transport.start().await
        .map_err(|e| anyhow::anyhow!("Failed to start MCP server '{}': {e}", config.name))?;

    let service = McpService::with_timeout(handle, Duration::from_secs(30));
    let mut client = McpClient::new(service);

    client
        .initialize(
            ClientInfo {
                name: "claux".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            ClientCapabilities::default(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize MCP server '{}': {e}", config.name))?;

    let tool_list = client
        .list_tools(None)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to list tools from MCP server '{}': {e}", config.name))?;

    let client = Arc::new(Mutex::new(client));

    let tools: Vec<Box<dyn Tool>> = tool_list
        .tools
        .into_iter()
        .map(|t| {
            let tool: Box<dyn Tool> = Box::new(McpTool::new(
                config.name.clone(),
                format!("mcp__{}__{}", config.name, t.name),
                t.description.clone(),
                t.input_schema,
                client.clone(),
            ));
            tool
        })
        .collect();

    Ok(tools)
}
