//! Model access to durable background tasks, scoped to this conversation.
use super::{Tool, ToolOutput};
use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

pub struct ManagedTaskTool {
    config: crate::config::Config,
    resolved: crate::config::ResolvedModel,
    cwd: PathBuf,
    owned: Mutex<HashSet<String>>,
}

impl ManagedTaskTool {
    pub fn new(
        config: crate::config::Config,
        resolved: crate::config::ResolvedModel,
        cwd: PathBuf,
    ) -> Self {
        Self {
            config,
            resolved,
            cwd,
            owned: Mutex::new(HashSet::new()),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    action: String,
    task_id: Option<String>,
    prompt: Option<String>,
}

#[async_trait]
impl Tool for ManagedTaskTool {
    fn name(&self) -> &str {
        "TaskManage"
    }
    fn description(&self) -> &str {
        "Start and manage durable background research tasks for this conversation. Workers run independently in the current workspace, with read-only tools and no nested agents or MCP. They do not receive this conversation: include needed context. Use status to inspect their report and result. Tasks remain running when the conversation ends; the user can also inspect them with /tasks. Only tasks started in this conversation are accessible."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{
            "action":{"type":"string","enum":["start","list","status","send","cancel"]},
            "task_id":{"type":"string","description":"Exact ID returned by start; required for status, send, and cancel"},
            "prompt":{"type":"string","description":"Self-contained instructions for start or send"}
        },"required":["action"],"additionalProperties":false})
    }
    fn is_read_only(&self) -> bool {
        false
    }
    fn summarize(&self, input: &Value) -> String {
        let action = input["action"].as_str().unwrap_or("manage");
        let detail = input["prompt"]
            .as_str()
            .or_else(|| input["task_id"].as_str())
            .unwrap_or("");
        format!(
            "Background task {action}: {}",
            detail.chars().take(80).collect::<String>()
        )
    }
    fn reset_session(&self) {
        self.owned.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
    async fn execute(&self, input: Value, cancel: CancellationToken) -> Result<ToolOutput> {
        if cancel.is_cancelled() {
            return Ok(super::interrupted_output());
        }
        let params: Params = serde_json::from_value(input)?;
        let mut owned = self.owned.lock().unwrap_or_else(|e| e.into_inner());
        let value = match (params.action.as_str(), params.task_id.as_deref(), params.prompt.as_deref()) {
            ("start", None, Some(prompt)) => {
                ensure!(owned.len() < 16, "This conversation has reached its 16-task limit");
                let state = crate::managed::start_model_task(&self.config, &self.resolved, &self.cwd, prompt)?;
                owned.insert(state["id"].as_str().context("Task has no ID")?.to_string());
                state
            }
            ("list", None, None) => {
                let mut ids: Vec<_> = owned.iter().collect();
                ids.sort();
                let states: Result<Vec<_>> = ids.into_iter().map(|id| crate::managed::model_task_status(id)).collect();
                json!(states?)
            }
            (action @ ("status" | "cancel"), Some(id), None) => {
                ensure!(owned.contains(id), "Task was not started in this conversation");
                if action == "status" { crate::managed::model_task_status(id)? } else { crate::managed::model_task_cancel(id)? }
            }
            ("send", Some(id), Some(prompt)) => {
                ensure!(owned.contains(id), "Task was not started in this conversation");
                crate::managed::model_task_send(id, prompt)?
            }
            _ => bail!("Use start with prompt; list; status/cancel with task_id; or send with task_id and prompt"),
        };
        Ok(ToolOutput {
            content: value.to_string(),
            is_error: false,
            sub_agent: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permissions::{PermissionChecker, PermissionMode, PermissionResult};

    #[tokio::test]
    async fn task_ids_are_scoped_to_the_current_conversation() -> Result<()> {
        let config = crate::config::Config::default();
        let resolved = config.default_resolved_model()?;
        let tool = ManagedTaskTool::new(config, resolved, std::env::current_dir()?);
        assert!(tool
            .execute(
                json!({"action":"status","task_id":"someone-elses-task"}),
                CancellationToken::new()
            )
            .await
            .is_err());
        tool.owned.lock().unwrap().insert("owned-task".into());
        tool.reset_session();
        assert!(tool
            .execute(
                json!({"action":"cancel","task_id":"owned-task"}),
                CancellationToken::new()
            )
            .await
            .is_err());
        assert!(tool
            .execute(
                json!({"action":"start","prompt":"x","cwd":"/"}),
                CancellationToken::new()
            )
            .await
            .is_err());
        Ok(())
    }

    #[test]
    fn task_tool_uses_normal_mutation_permissions() {
        let input = json!({"action":"start","prompt":"inspect"});
        assert!(matches!(
            PermissionChecker::new(PermissionMode::Default).check("TaskManage", &input, false),
            PermissionResult::Ask { .. }
        ));
        assert!(matches!(
            PermissionChecker::new(PermissionMode::Plan).check("TaskManage", &input, false),
            PermissionResult::Deny(_)
        ));
    }
}
