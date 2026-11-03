//! Runtime assembly that is shared by one-shot and interactive frontends.

use crate::command_sandbox::CommandSandbox;
use crate::config::{self, Config};
use crate::tools::{self, Tool};

/// Global servers are owner-configured; trusted project definitions are
/// model-writable and must always start within the session's command sandbox.
pub async fn connect_mcp_tools(config: &Config, sandbox: &CommandSandbox) -> Vec<Box<dyn Tool>> {
    let trust = config
        .project_trust
        .as_ref()
        .expect("project trust is resolved while loading config");
    let project_servers = config::load_mcp_json(trust);
    if !project_servers.is_empty() {
        tracing::info!(
            "Loaded {} sandboxed MCP server(s) from .mcp.json",
            project_servers.len()
        );
    } else if !trust.is_trusted() && trust.project_file(".mcp.json").exists() {
        tracing::warn!("Ignoring untrusted project .mcp.json; pass --trust-project or add this directory to trusted_projects in the global config");
    }
    let (mut global, project) = tokio::join!(
        tools::mcp::connect_mcp_servers(&config.mcp_servers),
        tools::mcp::connect_project_mcp_servers(&project_servers, sandbox),
    );
    global.extend(project);
    global
}
