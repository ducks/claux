use anyhow::Result;
use async_trait::async_trait;
use rmcp::{
    model::{
        CallToolRequest, CallToolRequestParams, ClientInfo, ClientRequest, Implementation,
        RawContent, ServerResult,
    },
    service::{PeerRequestOptions, RunningService},
    transport::{ConfigureCommandExt, TokioChildProcess},
    RoleClient, ServiceExt,
};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

use super::{Tool, ToolOutput};
use crate::config::McpServerConfig;

type McpClient = RunningService<RoleClient, ClientInfo>;

const MCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const MCP_CALL_TIMEOUT: Duration = Duration::from_secs(120);
const MCP_CANCEL_NOTIFY_TIMEOUT: Duration = Duration::from_secs(1);

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

        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let mut handle = match self
            .client
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await
        {
            Ok(handle) => handle,
            Err(error) => {
                return Ok(ToolOutput {
                    content: format!("MCP error: {error}"),
                    is_error: true,
                });
            }
        };

        let response = tokio::select! {
            response = &mut handle.rx => response,
            _ = cancel.cancelled() => {
                let suffix = match tokio::time::timeout(
                    MCP_CANCEL_NOTIFY_TIMEOUT,
                    handle.cancel(Some("cancelled by user".to_string())),
                )
                .await
                {
                    Ok(Ok(())) => {
                        "Cancellation was sent to the server; side effects may already have occurred."
                    }
                    Ok(Err(_)) | Err(_) => {
                        "The cancellation notification could not be delivered; the remote operation may still be running."
                    }
                };
                return Ok(ToolOutput {
                    content: format!("MCP request cancelled by user. {suffix}"),
                    is_error: true,
                });
            }
            _ = tokio::time::sleep(MCP_CALL_TIMEOUT) => {
                let suffix = match tokio::time::timeout(
                    MCP_CANCEL_NOTIFY_TIMEOUT,
                    handle.cancel(Some("request timed out".to_string())),
                )
                .await
                {
                    Ok(Ok(())) => {
                        "Cancellation was sent to the server; side effects may already have occurred."
                    }
                    Ok(Err(_)) | Err(_) => {
                        "The cancellation notification could not be delivered; the remote operation may still be running."
                    }
                };
                return Ok(ToolOutput {
                    content: format!("MCP request timed out after {MCP_CALL_TIMEOUT:?}. {suffix}"),
                    is_error: true,
                });
            }
        };

        match response {
            Ok(Ok(ServerResult::CallToolResult(call_result))) => {
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
            Ok(Ok(_)) => Ok(ToolOutput {
                content: "MCP error: server returned an unexpected response".to_string(),
                is_error: true,
            }),
            Ok(Err(error)) => Ok(ToolOutput {
                content: format!("MCP error: {error}"),
                is_error: true,
            }),
            Err(_) => Ok(ToolOutput {
                content: "MCP error: connection closed before the tool returned".to_string(),
                is_error: true,
            }),
        }
    }
}

/// Connect to all configured MCP servers and return their tools.
pub async fn connect_mcp_servers(configs: &[McpServerConfig]) -> Vec<Box<dyn Tool>> {
    connect_mcp_servers_with_timeout(configs, MCP_CONNECT_TIMEOUT).await
}

async fn connect_mcp_servers_with_timeout(
    configs: &[McpServerConfig],
    timeout: Duration,
) -> Vec<Box<dyn Tool>> {
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();

    let connections = configs.iter().map(|config| async move {
        let result = tokio::time::timeout(timeout, connect_server(config))
            .await
            .map_err(|_| {
                anyhow::anyhow!("Timed out after {timeout:?} while starting or discovering tools")
            })
            .and_then(|result| result);
        (config, result)
    });

    for (config, result) in futures_util::future::join_all(connections).await {
        match result {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn server_connections_are_concurrent_and_bounded() {
        let configs: Vec<McpServerConfig> = (0..10)
            .map(|index| McpServerConfig {
                name: format!("server-{index}"),
                command: "sleep".to_string(),
                args: vec!["5".to_string()],
                env: Default::default(),
            })
            .collect();
        let timeout = Duration::from_millis(50);
        let started = tokio::time::Instant::now();

        let tools = connect_mcp_servers_with_timeout(&configs, timeout).await;

        assert!(tools.is_empty());
        assert!(
            started.elapsed() < Duration::from_millis(300),
            "server timeouts should overlap, took {:?}",
            started.elapsed()
        );
    }
}
