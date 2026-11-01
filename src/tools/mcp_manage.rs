//! The query engine executes lifecycle operations after normal permission checks.
use super::{Tool, ToolOutput};
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

pub struct McpManageTool {
    description: String,
}

impl McpManageTool {
    pub fn new(config: &crate::config::Config) -> Self {
        let project = config
            .project_trust
            .as_ref()
            .map(|trust| trust.project_file(""))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));
        let directory = project.join(&config.generated_mcp_dir);
        let directory =
            serde_json::to_string(&directory.to_string_lossy()).expect("path serializes");
        Self { description: format!(
            "Load, list, or unload MCP servers during this conversation. Default generated server directory: {directory}. When creating a server, write its files under a subdirectory named after the server in that directory (for example <directory>/<name>/server.py), unless the user requests another location. Create directories through normal file tools as needed. Add its launch definition to the trusted project's .mcp.json, then load its name here. Successful discovery exposes its tools on the next round. On failure, repair the files and retry. Unload before changing/reloading a server. Model-loaded servers inherit Bash filesystem containment; their tools require normal mutation permissions regardless of read-only hints. Loading executes the configured program."
        ) }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    action: String,
    name: Option<String>,
}

pub(super) fn parse(input: Value) -> Result<String> {
    let params: Params = serde_json::from_value(input)?;
    match (params.action.as_str(), params.name) {
        ("list", None) => Ok("list".into()),
        ("load" | "unload", Some(name))
            if !name.is_empty()
                && name.len() <= 64
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') =>
        {
            Ok(format!("{} {name}", params.action))
        }
        _ => anyhow::bail!(
            "Use action=list without name, or action=load/unload with a valid server name"
        ),
    }
}

#[async_trait]
impl Tool for McpManageTool {
    fn name(&self) -> &str {
        "McpManage"
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object", "properties":{
            "action":{"type":"string", "enum":["list","load","unload"]},
            "name":{"type":"string", "description":"Configured MCP server name; required for load/unload"}
        },"required":["action"],"additionalProperties":false})
    }
    fn is_read_only(&self) -> bool {
        false
    }
    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolOutput> {
        anyhow::bail!("McpManage requires the session query engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_directory_is_resolved_against_captured_project() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::config::Config {
            project_trust: Some(crate::config::ProjectTrust::for_test(
                dir.path().into(),
                true,
            )),
            ..Default::default()
        };
        let expected = dir.path().join(".claux/tools");
        assert!(McpManageTool::new(&config)
            .description()
            .contains(expected.to_str().unwrap()));
        assert!(
            !expected.exists(),
            "configuring the default must not write files"
        );
        config.generated_mcp_dir = "custom-tools".into();
        assert!(McpManageTool::new(&config)
            .description()
            .contains(dir.path().join("custom-tools").to_str().unwrap()));
        config.generated_mcp_dir = dir.path().join("absolute-tools");
        assert!(McpManageTool::new(&config)
            .description()
            .contains(config.generated_mcp_dir.to_str().unwrap()));
    }

    #[test]
    fn rejects_invalid_lifecycle_requests() {
        for input in [
            json!({"action":"load"}),
            json!({"action":"list","name":"x"}),
            json!({"action":"reload","name":"x"}),
            json!({"action":"load","name":"x unload other"}),
            json!({"action":"load","name":"x","command":"sh"}),
        ] {
            assert!(parse(input).is_err());
        }
        assert_eq!(
            parse(json!({"action":"load","name":"my-tool"})).unwrap(),
            "load my-tool"
        );
    }
}
