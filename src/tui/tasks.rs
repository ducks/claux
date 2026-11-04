//! Interactive controls for the same durable tasks exposed by `claux task`.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueItem {
    pub id: String,
    pub stage: String,
    pub model: String,
    pub summary: String,
    pub question: Option<String>,
    pub queued: u64,
}

impl QueueItem {
    pub fn attention(&self) -> bool {
        matches!(
            self.stage.as_str(),
            "needs_input" | "failed" | "interrupted" | "ready_for_review"
        )
    }

    pub fn label(&self) -> &str {
        match self.stage.as_str() {
            "needs_input" => "Needs an answer",
            "failed" => "Failed",
            "interrupted" => "Interrupted",
            "ready_for_review" => "Ready for review",
            "working" => "Working",
            "cancelled" => "Cancelled",
            _ => "Waiting",
        }
    }

    fn rank(&self) -> u8 {
        match self.stage.as_str() {
            "needs_input" => 0,
            "failed" | "interrupted" => 1,
            "ready_for_review" => 2,
            "working" => 3,
            _ => 4,
        }
    }
}

pub fn queue_items(records: Vec<Value>) -> Vec<QueueItem> {
    let mut items: Vec<_> = records
        .into_iter()
        .filter_map(|value| {
            let id = value["id"].as_str()?.to_owned();
            let stage = value["stage"].as_str().unwrap_or("unknown").to_owned();
            let report = &value["report"];
            let relevant = matches!(
                stage.as_str(),
                "needs_input" | "ready_for_review" | "failed"
            ) && report["state"].as_str() == Some(stage.as_str());
            Some(QueueItem {
                id,
                stage,
                model: value["model"].as_str().unwrap_or("unknown").to_owned(),
                summary: if relevant {
                    report["summary"].as_str().unwrap_or("").to_owned()
                } else {
                    String::new()
                },
                question: if relevant {
                    report["question"].as_str().map(str::to_owned)
                } else {
                    None
                },
                queued: value["queued_inputs"].as_u64().unwrap_or(0),
            })
        })
        .collect();
    items.sort_by(|a, b| a.rank().cmp(&b.rank()).then_with(|| a.id.cmp(&b.id)));
    items
}

const HELP: &str = "Background tasks:\n/tasks — list tasks\n/task start <instructions> — start in the current directory\n/task start-worktree <instructions> — start in a new Git worktree\n/task start-edit <instructions> — start in a worktree with edit permission\n/task open <id> — select a task and view recent activity\n/task refresh — refresh the selected task\n/task send <instructions> — queue a follow-up\n/task resume — resume queued work after a handoff\n/task cancel — stop the current turn (queued inputs are kept)\nTasks use the current model and default background permissions unless start-edit is chosen. They do not inherit this conversation; include the context they need.";

fn selected(id: &Option<String>) -> Result<&str> {
    id.as_deref()
        .context("Select a task first with /task open <id>.")
}

async fn run_cli(args: &crate::cli::Cli, task_args: &[&str]) -> Result<Value> {
    let mut command = tokio::process::Command::new(std::env::current_exe()?);
    if let Some(path) = args.config_path() {
        command.arg("--config").arg(path);
    }
    for (flag, value) in [
        ("--base-url", args.base_url.as_ref()),
        ("--protocol", args.protocol.as_ref()),
        ("--reasoning-effort", args.reasoning_effort.as_ref()),
        ("--native-fs-policy", args.native_fs_policy.as_ref()),
        ("--bash-fs-policy", args.bash_fs_policy.as_ref()),
    ] {
        if let Some(value) = value {
            command.arg(flag).arg(value);
        }
    }
    if args.trust_project {
        command.arg("--trust-project");
    }
    command
        .arg("task")
        .args(task_args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(20), command.output())
        .await
        .context("Task command timed out; check /tasks before retrying.")??;
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        bail!("{}", error.trim());
    }
    serde_json::from_slice(&output.stdout).context("Task command returned invalid output")
}

fn task_summary(value: &Value) -> String {
    let id = value["id"].as_str().unwrap_or("?");
    let stage = value["stage"].as_str().unwrap_or("unknown");
    let model = value["model"].as_str().unwrap_or("?");
    let queued = value["queued_inputs"].as_u64().unwrap_or(0);
    format!("{id}  {stage}  {model}  {queued} queued")
}

fn recent_activity(dir: &Path) -> Result<String> {
    let path = dir.join("events.jsonl");
    if !path.exists() {
        return Ok("No activity yet.".into());
    }
    let mut file = std::fs::File::open(path)?;
    let length = file.metadata()?.len();
    let offset = length.saturating_sub(64 * 1024);
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = Vec::new();
    file.take(64 * 1024).read_to_end(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes);
    let lines = text.lines().skip(usize::from(offset > 0));
    let mut entries = Vec::new();
    for line in lines {
        let Ok(event) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let kind = event["type"].as_str().unwrap_or("event");
        let detail = match kind {
            "text" => event["data"]["text"].as_str().unwrap_or(""),
            "tool_started" => event["data"]["name"].as_str().unwrap_or(""),
            _ => "",
        };
        let detail: String = detail.chars().take(500).collect();
        entries.push(format!("{kind}: {detail}"));
    }
    if entries.is_empty() {
        return Ok("No activity yet.".into());
    }
    Ok(entries
        .into_iter()
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n"))
}

pub async fn execute(
    input: &str,
    selected_task: &mut Option<String>,
    args: &crate::cli::Cli,
    binding: Option<&crate::config::ModelBinding>,
) -> Result<String> {
    if input == "/tasks" {
        let list = run_cli(args, &["list"]).await?;
        let tasks = list.as_array().context("Invalid task list")?;
        return Ok(if tasks.is_empty() {
            "No background tasks. Use /task start <instructions>.".into()
        } else {
            format!(
                "Background tasks (use /task open <id>):\n{}",
                tasks
                    .iter()
                    .map(task_summary)
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        });
    }
    let rest = input.strip_prefix("/task").unwrap_or("").trim();
    let (action, argument) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let argument = argument.trim();
    match action {
        "" | "help" => Ok(HELP.into()),
        "start" | "start-worktree" | "start-edit" => {
            ensure!(!argument.is_empty(), "Provide task instructions.\n{HELP}");
            let binding = binding.context("Current session has no model binding")?;
            let resolved = args.load_config()?.resolve_binding(binding)?;
            ensure!(resolved.binding == *binding, "The current session's model settings changed. Restore them before starting a task.");
            let profile = &binding.profile;
            let mut command = vec![
                "start",
                argument,
                "--model",
                profile.as_str(),
                "--structured-report",
            ];
            if matches!(action, "start-worktree" | "start-edit") {
                command.push("--worktree");
            }
            if action == "start-edit" {
                command.extend(["--permission-mode", "accept-edits"]);
            }
            let state = run_cli(args, &command).await?;
            let id = state["id"]
                .as_str()
                .context("Task did not return an ID")?
                .to_string();
            *selected_task = Some(id);
            Ok(format!("Started and selected task: {}\nUse /task refresh to inspect it and /task send <instructions> to queue a follow-up.", task_summary(&state)))
        }
        "open" | "refresh" => {
            let id = if action == "open" {
                ensure!(!argument.is_empty(), "Use /task open <id>.");
                argument
            } else {
                selected(selected_task)?
            };
            let state = run_cli(args, &["status", id]).await?;
            let dir = state["artifacts"]
                .as_str()
                .context("Task artifacts unavailable")?;
            let activity = recent_activity(Path::new(dir))?;
            let report = &state["report"];
            let summary = report["summary"].as_str().unwrap_or("No report yet.");
            let question = report["question"]
                .as_str()
                .map(|q| format!("\nQuestion: {q}"))
                .unwrap_or_default();
            let display = format!(
                "{}\nReport: {summary}{question}\nRecent activity:\n{activity}",
                task_summary(&state)
            );
            *selected_task = Some(id.to_string());
            Ok(display)
        }
        "send" => {
            ensure!(!argument.is_empty(), "Use /task send <instructions>.");
            let state = run_cli(args, &["send", selected(selected_task)?, argument]).await?;
            Ok(format!("Follow-up queued: {}", task_summary(&state)))
        }
        "resume" | "cancel" => {
            ensure!(
                argument.is_empty(),
                "Unexpected arguments to /task {action}"
            );
            let state = run_cli(args, &[action, selected(selected_task)?]).await?;
            Ok(format!(
                "Task {action}: {}",
                if action == "cancel" {
                    "requested".to_string()
                } else {
                    task_summary(&state)
                }
            ))
        }
        _ => bail!("Unknown task command.\n{HELP}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn recent_activity_shows_latest_events_with_bounded_details() -> Result<()> {
        let dir = tempfile::tempdir()?;
        assert_eq!(recent_activity(dir.path())?, "No activity yet.");
        let mut file = std::fs::File::create(dir.path().join("events.jsonl"))?;
        for index in 0..25 {
            writeln!(
                file,
                "{}",
                serde_json::json!({"type":"tool_started","data":{"name":format!("tool-{index}")}})
            )?;
        }
        writeln!(
            file,
            "{}",
            serde_json::json!({"type":"text","data":{"text":"x".repeat(1000)}})
        )?;
        let output = recent_activity(dir.path())?;
        assert!(!output.contains("tool-0"));
        assert!(output.contains("tool-24"));
        assert!(!output.contains(&"x".repeat(501)));
        assert_eq!(output.lines().count(), 20);
        Ok(())
    }

    #[test]
    fn task_summary_shows_state_model_and_queue() {
        let state = serde_json::json!({"id":"abc","stage":"needs_input","model":"test-model","queued_inputs":2});
        assert_eq!(
            task_summary(&state),
            "abc  needs_input  test-model  2 queued"
        );
        assert!(selected(&None).is_err());
    }

    #[test]
    fn attention_items_are_prioritized_and_stale_reports_are_not_shown() {
        let items = queue_items(vec![
            serde_json::json!({"id":"running", "stage":"working", "report":{"state":"needs_input", "question":"old question"}}),
            serde_json::json!({"id":"review", "stage":"ready_for_review", "report":{"state":"ready_for_review", "summary":"Ready"}}),
            serde_json::json!({"id":"failed", "stage":"failed", "report":{"state":"ready_for_review", "summary":"old success"}}),
            serde_json::json!({"id":"question", "stage":"needs_input", "report":{"state":"needs_input", "question":"Which target?"}}),
        ]);
        assert_eq!(
            items
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            ["question", "failed", "review", "running"]
        );
        assert_eq!(items.iter().filter(|item| item.attention()).count(), 3);
        assert_eq!(items[0].question.as_deref(), Some("Which target?"));
        assert!(items[1].summary.is_empty());
        assert!(items[3].question.is_none());
    }
}
