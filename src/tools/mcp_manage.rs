//! The query engine executes lifecycle operations after normal permission checks.
use super::{Tool, ToolOutput};
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

pub struct McpManageTool;

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
        "Load, list, or unload MCP servers during this conversation. To create a tool, write a stdio MCP server and its definition in the trusted project's .mcp.json, then load its name here. Successful discovery exposes its tools on the next round. On failure, repair the files and retry. Unload before changing/reloading a server. Model-loaded servers inherit Bash filesystem containment; their tools require normal mutation permissions regardless of read-only hints. Loading executes the configured program."
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
