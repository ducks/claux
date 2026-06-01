use anyhow::Result;
use async_trait::async_trait;
use rmcp::{
    model::{CallToolRequestParams, ClientInfo, Implementation, RawContent},
    service::RunningService,
    transport::{ConfigureCommandExt, TokioChildProcess},
    RoleClient, ServiceExt,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::process::Command;

use super::{Tool, ToolOutput};
use crate::config::McpServerConfig;

type McpClient = RunningService<RoleClient, ClientInfo>;

/// A tool backed by an MCP server.
/// Wraps one tool from an MCP server's tools/list response.
pub struct McpTool {
    server_name: String,
    tool_name: String,
    upstream_name: String,
    tool_description: String,
    tool_schema: Value,
    client: Arc<McpClient>,
}

impl McpTool {
    pub fn new(
        server_name: String,
        tool_name: String,
        upstream_name: String,
        tool_description: String,
        tool_schema: Value,
        client: Arc<McpClient>,
    ) -> Self {
        Self {
            server_name,
            tool_name,
            upstream_name,
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

    async fn execute(
        &self,
        input: Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<ToolOutput> {
        // MCP tool arguments must be a JSON object (per spec). Coerce, and
        // pass None for non-object/null inputs so the server gets defaults.
        let arguments = match input {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return Ok(ToolOutput {
                    content: format!("MCP tool input must be a JSON object, got: {other}"),
                    is_error: true,
                });
            }
        };

        let mut params = CallToolRequestParams::new(self.upstream_name.clone());
        if let Some(args) = arguments {
            params = params.with_arguments(args);
        }

        let result = tokio::select! {
            r = self.client.call_tool(params) => r,
            _ = cancel.cancelled() => {
                return Ok(ToolOutput {
                    content: "Interrupted by user.".to_string(),
                    is_error: true,
                });
            }
        };

        match result {
            Ok(call_result) => {
                let text = call_result
                    .content
                    .iter()
                    .filter_map(|c| match &c.raw {
                        RawContent::Text(t) => Some(t.text.clone()),
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
pub async fn connect_mcp_servers(configs: &[McpServerConfig]) -> Vec<Box<dyn Tool>> {
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
    let command = config.command.clone();
    let args = config.args.clone();
    let env = config.env.clone();

    let transport = TokioChildProcess::new(Command::new(&command).configure(|cmd| {
        cmd.args(&args);
        for (k, v) in &env {
            cmd.env(k, v);
        }
    }))
    .map_err(|e| anyhow::anyhow!("Failed to start MCP server '{}': {e}", config.name))?;

    let mut client_info = ClientInfo::default();
    client_info.client_info = Implementation::new("claux", env!("CARGO_PKG_VERSION"));

    let client = client_info
        .serve(transport)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize MCP server '{}': {e}", config.name))?;

    let tool_list = client.list_all_tools().await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to list tools from MCP server '{}': {e}",
            config.name
        )
    })?;

    let client = Arc::new(client);

    let tools: Vec<Box<dyn Tool>> = tool_list
        .into_iter()
        .map(|t| {
            // The MCP-side tool name (what we send back to the server).
            let upstream_name = t.name.to_string();
            // The claux-side tool name (namespaced so multiple servers don't collide).
            let exposed_name = format!("mcp__{}__{}", config.name, upstream_name);
            let description = t.description.as_deref().unwrap_or("").to_string();
            // input_schema is Arc<JsonObject>; convert to Value::Object for
            // claux's Tool::input_schema(&self) -> Value contract.
            let schema = Value::Object((*t.input_schema).clone());

            let tool: Box<dyn Tool> = Box::new(McpTool::new(
                config.name.clone(),
                exposed_name,
                upstream_name,
                description,
                schema,
                client.clone(),
            ));
            tool
        })
        .collect();

    Ok(tools)
}
