//! Experimental local task supervisor. File locks, not saved PIDs, own workers.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::{cli, config, query::StreamEvent, session};

#[derive(Subcommand)]
pub enum TaskCommand {
    /// Start an experimental background task; prints a JSON task record
    Start {
        prompt: String,
        #[arg(long)]
        cwd: Option<PathBuf>,
        /// Create an isolated worktree and branch from HEAD
        #[arg(long)]
        worktree: bool,
        /// Named model profile, resolved and pinned when the task is created
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 900, value_parser = clap::value_parser!(u64).range(1..=3600))]
        timeout: u64,
        #[arg(long, default_value = "default", value_parser = ["default", "accept-edits", "auto", "bypass", "plan"])]
        permission_mode: String,
        #[arg(long)]
        trust_project: bool,
    },
    /// List task records as JSON
    List,
    /// Inspect a task and its worker liveness
    Status { id: String },
    /// Replay JSON events; --follow streams until the worker exits
    Attach {
        id: String,
        #[arg(long)]
        follow: bool,
    },
    /// Continue an idle task in the same conversation
    Send { id: String, prompt: String },
    /// Retry a launch that failed before a turn began (never replays an evaluated turn)
    Retry { id: String },
    /// Cooperatively cancel the current turn; does not delete worktrees
    Cancel { id: String },
    /// Wait for the current turn and print its versioned JSON result
    Wait {
        id: String,
        #[arg(long, default_value_t = 3600, value_parser = clap::value_parser!(u64).range(1..=86400))]
        timeout: u64,
    },
    #[command(hide = true)]
    Worker { id: String },
}

#[derive(Serialize, Deserialize)]
struct Task {
    schema_version: u8,
    id: String,
    cwd: PathBuf,
    session: PathBuf,
    binding: config::ModelBinding,
    // Only public CLI settings. Credentials are resolved anew by the worker.
    options: Vec<String>,
    timeout: u64,
    branch: Option<String>,
}

fn root() -> Result<PathBuf> {
    Ok(dirs::data_local_dir()
        .context("no local data directory")?
        .join("claux/tasks"))
}

fn task_dir(id: &str) -> Result<PathBuf> {
    ensure!(
        id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "expected an exact task ID"
    );
    let dir = root()?.join(id);
    ensure!(dir.join("task.json").is_file(), "task not found");
    Ok(dir)
}

fn private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn private_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().context("missing parent")?)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}

fn read_task(dir: &Path) -> Result<Task> {
    Ok(serde_json::from_reader(File::open(dir.join("task.json"))?)?)
}

fn worker_active(dir: &Path) -> Result<bool> {
    let file = private_file(&dir.join("worker.lock"))?;
    match FileExt::try_lock_shared(&file) {
        Ok(()) => Ok(false), // close releases the probe lock
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(true),
        Err(e) => Err(e.into()),
    }
}

fn status(dir: &Path) -> Result<Value> {
    let task = read_task(dir)?;
    let state: Value = serde_json::from_reader(File::open(dir.join("state.json"))?)?;
    let active = worker_active(dir)?;
    let starting = state["status"] == "starting"
        && state["started_at_ms"].as_i64().is_some_and(|start| {
            chrono::Utc::now().timestamp_millis().saturating_sub(start) < 10_000
        });
    let interrupted =
        !active && !starting && matches!(state["status"].as_str(), Some("starting" | "running"));
    Ok(json!({"schema_version":1, "id":task.id, "cwd":task.cwd,
        "model":task.binding.model, "branch":task.branch,
        "session":task.session, "worker_active":active,
        "status":if interrupted { json!("interrupted") } else {state["status"].clone()},
        "artifacts":dir}))
}

fn event(dir: &Path, kind: &str, data: Value) -> Result<()> {
    let mut file = private_file(&dir.join("events.jsonl"))?;
    ensure!(
        file.metadata()?.len() < 32 * 1024 * 1024,
        "task event log reached 32 MiB limit"
    );
    file.seek(std::io::SeekFrom::End(0))?;
    serde_json::to_writer(
        &mut file,
        &json!({"schema_version":1,
        "time":chrono::Utc::now().to_rfc3339(), "type":kind, "data":data}),
    )?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn spawn_worker(dir: &Path, id: &str) -> Result<()> {
    let log = private_file(&dir.join("worker.log"))?;
    let mut command = Command::new(std::env::current_exe()?);
    command
        .args(["task", "worker", id])
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid is async-signal-safe and touches no Rust state after fork.
        unsafe {
            command.pre_exec(|| {
                nix::unistd::setsid()
                    .map(|_| ())
                    .map_err(std::io::Error::from)
            });
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x00000200 | 0x08000000); // new process group, no console
    }
    command.spawn()?;
    Ok(())
}

fn enqueue(dir: &Path, prompt: &str) -> Result<()> {
    ensure!(
        !prompt.trim().is_empty() && prompt.len() <= 40_000,
        "prompt must contain 1–40000 bytes"
    );
    ensure!(
        !worker_active(dir)?,
        "task is active; wait before sending a follow-up"
    );
    let state: Value = serde_json::from_reader(File::open(dir.join("state.json"))?)?;
    ensure!(
        state["status"] != "interrupted"
            && !matches!(state["status"].as_str(), Some("running" | "starting")),
        "previous worker was interrupted; recover its checkpoint before continuing"
    );
    ensure!(
        !dir.join("pending.json").exists(),
        "a turn is already pending"
    );
    // The caller holds control.lock, serializing send/cancel/start operations.
    let _ = fs::remove_file(dir.join("cancel"));
    write_json(&dir.join("pending.json"), &json!({"prompt":prompt}))?;
    write_json(
        &dir.join("state.json"),
        &json!({"status":"starting", "started_at_ms":chrono::Utc::now().timestamp_millis()}),
    )?;
    launch(dir)
}

fn launch(dir: &Path) -> Result<()> {
    if let Err(error) = spawn_worker(dir, &read_task(dir)?.id) {
        write_json(
            &dir.join("state.json"),
            &json!({"status":"failed", "reason":"spawn_failed"}),
        )?;
        return Err(error);
    }
    Ok(())
}

fn worker_slot(root: &Path) -> Result<File> {
    let slots = root.join("worker-slots");
    private_dir(&slots)?;
    for index in 0..4 {
        let lock = private_file(&slots.join(format!("{index}.lock")))?;
        match FileExt::try_lock_exclusive(&lock) {
            Ok(()) => return Ok(lock),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("four managed workers already active; retry after one finishes")
}

fn control_lock(dir: &Path) -> Result<File> {
    let file = private_file(&dir.join("control.lock"))?;
    FileExt::try_lock_exclusive(&file).context("another task command is in progress")?;
    Ok(file)
}

pub async fn handle(args: &cli::Cli, command: &TaskCommand) -> Result<()> {
    match command {
        TaskCommand::Start {
            prompt,
            cwd,
            worktree,
            model,
            timeout,
            permission_mode,
            trust_project,
        } => {
            ensure!(
                !prompt.trim().is_empty() && prompt.len() <= 40_000,
                "prompt must contain 1–40000 bytes"
            );
            let cwd = cwd.as_deref().unwrap_or(Path::new(".")).canonicalize()?;
            ensure!(cwd.is_dir(), "working directory is not a directory");
            // Resolve relative config paths before changing to the task worktree.
            let config_path = args.config_path().map(fs::canonicalize).transpose()?;
            let id = uuid::Uuid::new_v4().simple().to_string();
            private_dir(&root()?)?;
            let dir = root()?.join(&id);
            private_dir(&dir)?;
            let mut branch = None;
            let cwd = if *worktree {
                let name = format!("claux-task/{id}");
                let path = dir.join("worktree");
                let output = Command::new("git")
                    .current_dir(&cwd)
                    .args(["worktree", "add", "-b", &name])
                    .arg(&path)
                    .arg("HEAD")
                    .output()?;
                ensure!(
                    output.status.success(),
                    "git worktree creation failed; inspect repository state"
                );
                branch = Some(name);
                path
            } else {
                cwd
            };
            // Resolve project configuration in the actual task workspace.
            std::env::set_current_dir(&cwd)?;
            let mut options = vec!["claux".into()];
            for (flag, value) in [
                ("--base-url", args.base_url.as_ref()),
                ("--protocol", args.protocol.as_ref()),
                ("--reasoning-effort", args.reasoning_effort.as_ref()),
                ("--native-fs-policy", args.native_fs_policy.as_ref()),
                ("--bash-fs-policy", args.bash_fs_policy.as_ref()),
                ("--model", model.as_ref()),
            ] {
                if let Some(value) = value {
                    options.extend([flag.into(), value.clone()]);
                }
            }
            if let Some(path) = config_path {
                options.extend([
                    "--config".into(),
                    path.to_str()
                        .context("configuration path must be UTF-8")?
                        .into(),
                ]);
            }
            options.extend(["--permission-mode".into(), permission_mode.clone()]);
            if *trust_project || args.trust_project {
                options.push("--trust-project".into());
            }
            let cli = cli::Cli::try_parse_from(&options)?;
            let config = cli.load_config()?;
            let resolved = match model {
                Some(m) => config.resolve_model(m)?,
                None => config.default_resolved_model()?,
            };
            let (_, session) = session::create_session_with_model(&resolved)?;
            let task = Task {
                schema_version: 1,
                id,
                cwd,
                session,
                binding: resolved.binding,
                options,
                timeout: *timeout,
                branch,
            };
            write_json(&dir.join("task.json"), &task)?;
            write_json(&dir.join("state.json"), &json!({"status":"idle"}))?;
            let _control = control_lock(&dir)?;
            enqueue(&dir, prompt)?;
            println!("{}", status(&dir)?);
        }
        TaskCommand::List => {
            let mut records = Vec::new();
            if root()?.exists() {
                for entry in fs::read_dir(root()?)? {
                    let dir = entry?.path();
                    if dir.join("task.json").is_file() {
                        records.push(status(&dir)?);
                    }
                }
            }
            records.sort_by_key(|record| record["id"].as_str().unwrap_or_default().to_owned());
            println!("{}", serde_json::to_string(&records)?);
        }
        TaskCommand::Status { id } => println!("{}", status(&task_dir(id)?)?),
        TaskCommand::Send { id, prompt } => {
            let dir = task_dir(id)?;
            let _control = control_lock(&dir)?;
            enqueue(&dir, prompt)?;
            println!("{}", status(&dir)?);
        }
        TaskCommand::Retry { id } => {
            let dir = task_dir(id)?;
            let _control = control_lock(&dir)?;
            ensure!(!worker_active(&dir)?, "task worker is active");
            let state = status(&dir)?;
            ensure!(
                state["status"] == "failed" || state["status"] == "interrupted",
                "only failed or interrupted launches can be retried"
            );
            ensure!(
                dir.join("pending.json").is_file(),
                "turn already began; do not replay it—inspect the checkpoint instead"
            );
            let _ = fs::remove_file(dir.join("cancel"));
            write_json(
                &dir.join("state.json"),
                &json!({"status":"starting", "started_at_ms":chrono::Utc::now().timestamp_millis()}),
            )?;
            launch(&dir)?;
            println!("{}", status(&dir)?);
        }
        TaskCommand::Cancel { id } => {
            let dir = task_dir(id)?;
            let _control = control_lock(&dir)?;
            ensure!(
                worker_active(&dir)? || dir.join("pending.json").exists(),
                "task has no active turn"
            );
            private_file(&dir.join("cancel"))?;
            println!("{}", json!({"id":id,"status":"cancel_requested"}));
        }
        TaskCommand::Attach { id, follow } => {
            let dir = task_dir(id)?;
            let mut reader = BufReader::new(private_file(&dir.join("events.jsonl"))?);
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line)? > 0 {
                    print!("{line}");
                    std::io::stdout().flush()?;
                    continue;
                }
                if !follow || (!worker_active(&dir)? && status(&dir)?["status"] != "starting") {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
        TaskCommand::Wait { id, timeout } => {
            let dir = task_dir(id)?;
            let deadline = Instant::now() + Duration::from_secs(*timeout);
            loop {
                let state = status(&dir)?;
                if !worker_active(&dir)? && state["status"] != "starting" {
                    ensure!(
                        state["status"] == "idle",
                        "task did not complete; inspect task status and events"
                    );
                    println!("{}", fs::read_to_string(dir.join("result.json"))?.trim());
                    break;
                }
                ensure!(
                    Instant::now() < deadline,
                    "wait timed out; task continues in background"
                );
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
        TaskCommand::Worker { id } => {
            let dir = task_dir(id)?;
            let lock = private_file(&dir.join("worker.lock"))?;
            // Brief status probes must not make a newly spawned worker fail.
            // A duplicate worker waits, then exits if the first consumed the turn.
            FileExt::lock_exclusive(&lock)?;
            if !dir.join("pending.json").is_file() {
                return Ok(());
            }
            if work(&dir).await.is_err() {
                // Do not persist configuration errors or provider response bodies.
                write_json(&dir.join("state.json"), &json!({"status":"failed"}))?;
                let _ = event(
                    &dir,
                    "failed",
                    json!({"message":"worker failed; inspect configuration or checkpoint"}),
                );
                anyhow::bail!("background worker failed; inspect task state and checkpoint");
            }
        }
    }
    Ok(())
}

async fn work(dir: &Path) -> Result<()> {
    let _slot = worker_slot(&root()?)?;
    let task = read_task(dir)?;
    std::env::set_current_dir(&task.cwd)?;
    let args = cli::Cli::try_parse_from(&task.options)?;
    let mut config = args.load_config()?;
    config.permission_mode = serde_json::from_value(json!(args.permission_mode))?;
    crate::command_sandbox::configure_child_environment(
        config.sensitive_environment_names(),
        config.strip_agent_sockets,
    );
    let resolved = config.resolve_binding(&task.binding)?;
    ensure!(
        resolved.binding == task.binding,
        "task provider settings changed; create a new task"
    );
    let mut plugins = crate::plugin::PluginRegistry::new();
    for plugin in &config.plugins {
        plugins.add(Box::new(crate::plugin::CommandPlugin::new(
            &plugin.name,
            &plugin.command,
            &plugin.args,
            plugin.trigger.clone(),
        )));
    }
    let plugins = Arc::new(plugins);
    let mut engine = crate::build_engine(&config, &resolved, plugins.clone()).await?;
    engine.set_system_prompt(
        crate::context::build_system_prompt_for_model(
            &resolved.binding.model,
            Some(&plugins),
            &config::HookTrigger::OnContextBuild,
            resolved.binding.provider_kind == config::ProviderKind::Anthropic,
            config.is_project_trusted(),
        )
        .await?,
    );
    let (_, messages) = session::load_session(&task.session)?;
    engine.set_messages(messages);
    engine.set_archive(session::load_archive(&task.session)?);
    engine.set_transcript_checkpoint(dir.join("checkpoint.json"));
    let pending: Value = serde_json::from_reader(File::open(dir.join("pending.json"))?)?;
    let prompt = pending["prompt"]
        .as_str()
        .context("missing pending prompt")?;
    fs::remove_file(dir.join("pending.json"))?;
    write_json(&dir.join("state.json"), &json!({"status":"running"}))?;
    event(dir, "turn_started", json!({"model":engine.model()}))?;
    let cancel = crate::shutdown::one_shot_cancellation_token()?;
    if dir.join("cancel").exists() {
        cancel.cancel();
    }
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let monitor_cancel = cancel.clone();
    let monitor_dir = dir.to_owned();
    let timeout = task.timeout;
    let monitor = tokio::spawn(async move {
        let deadline = Instant::now() + Duration::from_secs(timeout);
        loop {
            if monitor_dir.join("cancel").exists() || Instant::now() >= deadline {
                monitor_cancel.cancel();
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });
    let consume = collect_events(dir.to_owned(), rx, cancel.clone());
    let (response, collected) =
        tokio::join!(engine.submit_streaming(prompt, tx, cancel.clone()), consume);
    monitor.abort();
    session::save_engine(&task.session, &engine).inspect_err(|_| {
        let _ = session::write_engine_recovery(&task.session, &engine);
    })?;
    collected?;
    let completed = response.is_ok() && !cancel.is_cancelled();
    let result = if completed {
        crate::output::OneShotOutput::new("", engine.model(), &engine.cost)
    } else {
        crate::output::OneShotOutput::failed(
            engine.model(),
            &engine.cost,
            if cancel.is_cancelled() {
                "turn cancelled or deadline reached"
            } else {
                "turn failed"
            },
            engine.last_failure(),
        )
    };
    // Final response text is reconstructed from stream events, with retry resets.
    let mut value = serde_json::to_value(result)?;
    if completed {
        let text = final_text(dir)?;
        value["result"] = json!(text);
        value["outcome"]["result"] = value["result"].clone();
    }
    write_json(&dir.join("result.json"), &value)?;
    event(
        dir,
        if completed {
            "turn_completed"
        } else {
            "turn_failed"
        },
        json!({"usage":engine.cost.usage_summary()}),
    )?;
    write_json(
        &dir.join("state.json"),
        &json!({"status":if completed {"idle"} else if cancel.is_cancelled() {"cancelled"} else {"failed"}}),
    )?;
    Ok(())
}

async fn collect_events(
    dir: PathBuf,
    mut rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    cancel: CancellationToken,
) -> Result<()> {
    while let Some(item) = rx.recv().await {
        let (kind, data) = match item {
            StreamEvent::Text(text) => ("text", json!({"text":text})),
            StreamEvent::Retry(_) => ("retry", json!({})),
            StreamEvent::ModelRequest => ("model_request", json!({})),
            StreamEvent::ToolStart { name, .. } => ("tool_started", json!({"name":name})),
            StreamEvent::ToolFinished {
                index, is_error, ..
            } => ("tool_finished", json!({"index":index,"is_error":is_error})),
            StreamEvent::PermissionRequest { respond, .. }
            | StreamEvent::PermissionRequestWithDiff { respond, .. } => {
                let _ = respond.send(crate::permissions::PermissionResponse::Deny);
                (
                    "permission_denied",
                    json!({"reason":"background task cannot prompt; configure permission rules explicitly"}),
                )
            }
            StreamEvent::Interrupted => ("interrupted", json!({})),
            StreamEvent::Error(_) => ("error", json!({"message":"engine reported an error"})),
            _ => continue,
        };
        if let Err(error) = event(&dir, kind, data) {
            cancel.cancel();
            return Err(error);
        }
    }
    Ok(())
}

fn final_text(dir: &Path) -> Result<String> {
    let mut text = String::new();
    for line in BufReader::new(File::open(dir.join("events.jsonl"))?).lines() {
        let event: Value = serde_json::from_str(&line?)?;
        match event["type"].as_str() {
            Some("turn_started" | "retry") => text.clear(),
            Some("text") => text.push_str(event["data"]["text"].as_str().unwrap_or_default()),
            _ => {}
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_tracks_liveness_without_pid_reuse() -> Result<()> {
        let dir = tempfile::tempdir()?;
        assert!(!worker_active(dir.path())?);
        let lock = private_file(&dir.path().join("worker.lock"))?;
        FileExt::try_lock_exclusive(&lock)?;
        assert!(worker_active(dir.path())?);
        drop(lock);
        assert!(!worker_active(dir.path())?);
        Ok(())
    }

    #[test]
    fn worker_capacity_releases_slots_on_exit() -> Result<()> {
        let root = tempfile::tempdir()?;
        let mut slots = Vec::new();
        for _ in 0..4 {
            slots.push(worker_slot(root.path())?);
        }
        assert!(worker_slot(root.path()).is_err());
        slots.pop();
        assert!(worker_slot(root.path()).is_ok());
        Ok(())
    }

    #[test]
    fn output_resets_on_retry_and_followup() -> Result<()> {
        let dir = tempfile::tempdir()?;
        event(dir.path(), "text", json!({"text":"old turn"}))?;
        event(dir.path(), "turn_started", json!({}))?;
        event(dir.path(), "text", json!({"text":"rejected"}))?;
        event(dir.path(), "retry", json!({}))?;
        event(dir.path(), "text", json!({"text":"accepted"}))?;
        assert_eq!(final_text(dir.path())?, "accepted");
        Ok(())
    }

    #[tokio::test]
    async fn background_permissions_are_denied() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (respond, response) = tokio::sync::oneshot::channel();
        tx.send(StreamEvent::PermissionRequest {
            tool_name: "Bash".into(),
            summary: "test".into(),
            input: json!({}),
            respond,
        })
        .await?;
        drop(tx);
        collect_events(dir.path().to_owned(), rx, CancellationToken::new()).await?;
        assert_eq!(
            response.await?,
            crate::permissions::PermissionResponse::Deny
        );
        Ok(())
    }

    #[test]
    fn rejects_non_exact_ids() {
        for id in ["../outside", "", "short", "/tmp/a"] {
            assert!(task_dir(id).is_err());
        }
    }

    #[test]
    fn parses_task_commands_and_bounds() {
        assert!(cli::Cli::try_parse_from([
            "claux",
            "task",
            "start",
            "hello",
            "--worktree",
            "--model",
            "test"
        ])
        .is_ok());
        assert!(
            cli::Cli::try_parse_from(["claux", "task", "start", "hello", "--timeout", "0"])
                .is_err()
        );
    }
}
