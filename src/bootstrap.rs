//! Runtime assembly that is shared by one-shot and interactive frontends.

use crate::config::{self, Config};
use crate::tools::{self, Tool};

/// Connect globally configured MCP servers and, for trusted projects only,
/// project-local `.mcp.json` servers. Connections are created exactly once
/// and the resulting tools can be moved into either frontend's registry.
pub async fn connect_mcp_tools(config: &Config) -> Vec<Box<dyn Tool>> {
    let mut servers = config.mcp_servers.clone();
    let trust = config
        .project_trust
        .as_ref()
        .expect("project trust is resolved while loading config");
    let project_servers = config::load_mcp_json(trust);

    if !project_servers.is_empty() {
        tracing::info!(
            "Loaded {} MCP server(s) from .mcp.json",
            project_servers.len()
        );
        servers.extend(project_servers);
    } else if !trust.is_trusted() && trust.project_file(".mcp.json").exists() {
        tracing::warn!(
            "Ignoring untrusted project .mcp.json; pass --trust-project or add this directory \
             to trusted_projects in the global config"
        );
    }

    if servers.is_empty() {
        return Vec::new();
    }

    tracing::info!("Connecting to {} MCP server(s)...", servers.len());
    tools::mcp::connect_mcp_servers(&servers).await
}
