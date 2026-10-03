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
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

use super::{Tool, ToolOutput};
use crate::command_sandbox::sanitize_command_environment;
use crate::config::McpServerConfig;

type McpClient = RunningService<RoleClient, ClientInfo>;

const MCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const MCP_CALL_TIMEOUT: Duration = Duration::from_secs(120);
const MCP_CANCEL_NOTIFY_TIMEOUT: Duration = Duration::from_secs(1);
const MAX_RESULT_BYTES: usize = 64 * 1024;
const MAX_SCHEMA_BYTES: usize = 32 * 1024;
const MAX_DESCRIPTION_BYTES: usize = 4096;

fn bounded(text: &str, limit: usize) -> &str {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn validate_schema(schema: &Value) -> Result<()> {
    anyhow::ensure!(
        serde_json::to_vec(schema)?.len() <= MAX_SCHEMA_BYTES,
        "MCP tool schema exceeds size limit"
    );
    anyhow::ensure!(
        schema["type"] == "object",
        "MCP tool input schema must have object type"
    );
    // Network and file reference resolution are disabled in Cargo features.
    jsonschema::validator_for(schema)
        .map_err(|_| anyhow::anyhow!("Invalid or unresolved MCP input schema"))?;
    Ok(())
}

fn render_content<'a>(content: impl IntoIterator<Item = &'a RawContent>) -> String {
    let mut output = String::new();
    for raw in content {
        let text = match raw {
            RawContent::Text(text) => text.text.as_str(),
            _ => "[Non-text MCP content omitted]",
        };
        if !output.is_empty() {
            output.push('\n');
        }
        let available = MAX_RESULT_BYTES.saturating_sub(output.len());
        output.push_str(bounded(text, available));
        if text.len() > available || output.len() >= MAX_RESULT_BYTES {
            output.push_str("\n[MCP output truncated]");
            break;
        }
    }
    output
}

/// A tool backed by an MCP server.
/// Wraps one tool from an MCP server's tools/list response.
pub struct McpTool {
    timeout: Duration,
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
            timeout: MCP_CALL_TIMEOUT,
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
                    sub_agent: None,
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
                    sub_agent: None,
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
                        "The cancellation notification could not be delivered."
                    }
                };
                return Ok(ToolOutput {
                    sub_agent: None,
                    content: format!(
                        "MCP request cancelled by user. {suffix} The server remains connected."
                    ),
                    is_error: true,
                });
            }
            _ = tokio::time::sleep(self.timeout) => {
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
                        "The cancellation notification could not be delivered."
                    }
                };
                return Ok(ToolOutput {
                    sub_agent: None,
                    content: format!(
                        "MCP request timed out after {:?}. {suffix} The server remains connected.", self.timeout
                    ),
                    is_error: true,
                });
            }
        };

        match response {
            Ok(Ok(ServerResult::CallToolResult(call_result))) => {
                let text = render_content(call_result.content.iter().map(|content| &content.raw));

                Ok(ToolOutput {
                    sub_agent: None,
                    content: text,
                    is_error: call_result.is_error.unwrap_or(false),
                })
            }
            Ok(Ok(_)) => Ok(ToolOutput {
                sub_agent: None,
                content: "MCP error: server returned an unexpected response".to_string(),
                is_error: true,
            }),
            Ok(Err(error)) => Ok(ToolOutput {
                sub_agent: None,
                content: format!("MCP error: {error}"),
                is_error: true,
            }),
            Err(_) => Ok(ToolOutput {
                sub_agent: None,
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
    connect_mcp_servers_with(configs, timeout, |config| async move {
        connect_server(&config).await
    })
    .await
}

async fn connect_mcp_servers_with<F, Fut>(
    configs: &[McpServerConfig],
    timeout: Duration,
    connector: F,
) -> Vec<Box<dyn Tool>>
where
    F: Fn(McpServerConfig) -> Fut,
    Fut: Future<Output = Result<Vec<Box<dyn Tool>>>>,
{
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();

    let connections = configs.iter().map(|config| {
        let connection = connector(config.clone());
        async move {
            let result = tokio::time::timeout(timeout, connection)
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "Timed out after {timeout:?} while starting or discovering tools"
                    )
                })
                .and_then(|result| result);
            (config, result)
        }
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
    anyhow::ensure!(valid_name(&config.name), "Invalid MCP server name");
    let timeout = Duration::from_secs(config.timeout_seconds.unwrap_or(120));
    anyhow::ensure!(
        !timeout.is_zero() && timeout <= Duration::from_secs(3600),
        "MCP timeout_seconds must be between 1 and 3600"
    );
    let command = config.command.clone();
    let args = config.args.clone();
    let env = config.env.clone();

    let transport = TokioChildProcess::new(Command::new(&command).configure(|cmd| {
        sanitize_command_environment(cmd);
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
            anyhow::ensure!(
                valid_name(&exposed_name),
                "Invalid or oversized exposed MCP tool name: {exposed_name}"
            );
            let description = bounded(
                t.description.as_deref().unwrap_or(""),
                MAX_DESCRIPTION_BYTES,
            )
            .to_string();
            // input_schema is Arc<JsonObject>; convert to Value::Object for
            // claux's Tool::input_schema(&self) -> Value contract.
            let schema = Value::Object((*t.input_schema).clone());
            validate_schema(&schema)?;

            let mut tool = McpTool::new(
                config.name.clone(),
                exposed_name,
                upstream_name,
                description,
                schema,
                client.clone(),
            );
            tool.timeout = timeout;
            Ok(Box::new(tool) as Box<dyn Tool>)
        })
        .collect::<Result<_>>()?;

    Ok(tools)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn results_names_and_schemas_are_bounded_and_validated() {
        let content = vec![RawContent::text("é".repeat(MAX_RESULT_BYTES))];
        let rendered = render_content(&content);
        assert!(rendered.len() <= MAX_RESULT_BYTES + 32);
        assert!(rendered.ends_with("[MCP output truncated]"));
        assert!(!valid_name("server/name"));
        assert!(!valid_name(&"x".repeat(65)));
        assert!(valid_name("mcp__server-1__lookup"));
        for schema in [
            serde_json::json!({"type":"array"}),
            serde_json::json!({"type":"object","required":42}),
            serde_json::json!({"type":"object","properties":{"x":{"type":"bogus"}}}),
            serde_json::json!({"type":"object","description":"x".repeat(MAX_SCHEMA_BYTES)}),
            serde_json::json!({"type":"object","$ref":"file:///etc/passwd"}),
        ] {
            assert!(validate_schema(&schema).is_err());
        }
        assert!(validate_schema(
            &serde_json::json!({"type":"object","properties":{"x":{"type":"string"}}})
        )
        .is_ok());
    }

    #[test]
    fn registry_rejects_collisions_without_partially_adding_tools() {
        let mut registry = super::super::ToolRegistry::new();
        let before = registry.definitions().len();
        let duplicate = Box::new(super::super::read::ReadTool::new(Arc::new(
            crate::sandbox::SandboxPolicy::unrestricted_for_tests(),
        )));
        assert!(registry.add_tools(vec![duplicate]).is_err());
        assert_eq!(registry.definitions().len(), before);
    }

    #[tokio::test]
    async fn cancelled_and_timed_out_calls_leave_server_usable() {
        struct Server;
        impl rmcp::ServerHandler for Server {
            async fn call_tool(
                &self,
                request: CallToolRequestParams,
                context: rmcp::service::RequestContext<rmcp::RoleServer>,
            ) -> std::result::Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
                if request
                    .arguments
                    .as_ref()
                    .is_some_and(|args| args.get("slow") == Some(&Value::Bool(true)))
                {
                    context.ct.cancelled().await;
                }
                Ok(rmcp::model::CallToolResult::success(vec![
                    rmcp::model::Content::text("ok"),
                ]))
            }
        }
        let (client_io, server_io) = tokio::io::duplex(8192);
        let (server, client) = tokio::join!(
            Server.serve(server_io),
            ClientInfo::default().serve(client_io)
        );
        let server = server.unwrap();
        let client = Arc::new(client.unwrap());
        let mut tool = McpTool::new(
            "test".into(),
            "mcp__test__tool".into(),
            "tool".into(),
            String::new(),
            serde_json::json!({"type":"object"}),
            client.clone(),
        );
        tool.timeout = Duration::from_millis(25);
        let timed_out = tool
            .execute(
                serde_json::json!({"slow":true}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(timed_out.is_error && timed_out.content.contains("timed out"));
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        assert!(
            tool.execute(serde_json::json!({"slow":true}), cancel)
                .await
                .unwrap()
                .is_error
        );
        tool.timeout = Duration::from_secs(2);
        let result = tool
            .execute(
                serde_json::json!({}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(result.content, "ok");
        client.cancellation_token().cancel();
        server.cancellation_token().cancel();
    }

    #[test]
    fn non_text_content_is_omitted_without_forwarding_base64() {
        let content = vec![
            RawContent::text("hello"),
            RawContent::image("aGVsbG8=", "image/png"),
        ];

        let rendered = render_content(&content);

        assert!(rendered.starts_with("hello\n"));
        assert!(rendered.contains("Non-text MCP content omitted"));
        assert!(!rendered.contains("aGVsbG8="));
    }

    #[tokio::test]
    async fn server_connections_are_concurrent_and_bounded() {
        let configs: Vec<McpServerConfig> = (0..10)
            .map(|index| McpServerConfig {
                timeout_seconds: None,
                name: format!("server-{index}"),
                command: "sleep".to_string(),
                args: vec!["5".to_string()],
                env: Default::default(),
            })
            .collect();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let tools = connect_mcp_servers_with(&configs, Duration::from_millis(10), |_| {
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            async move {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                std::future::pending::<Result<Vec<Box<dyn Tool>>>>().await
            }
        })
        .await;

        assert!(tools.is_empty());
        assert_eq!(peak.load(Ordering::SeqCst), configs.len());
    }
}
