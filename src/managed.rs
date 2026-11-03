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
        /// Require a structured final report (waiting, needs_input, ready_for_review, failed)
        #[arg(long)]
        structured_report: bool,
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
    /// Resume queued follow-ups after a report handoff or cancellation
    Resume { id: String },
    /// Retry a launch that failed before a turn began (never replays an evaluated turn)
    Retry { id: String },
    /// Mark an inspected interrupted input abandoned; queued work remains stopped
    Recover {
        id: String,
        /// Exact interrupted input ID, after inspecting its effects and checkpoint
        #[arg(long)]
        abandon: String,
    },
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
    #[serde(default)]
    structured_report: bool,
    id: String,
    cwd: PathBuf,
    session: PathBuf,
    binding: config::ModelBinding,
    // Only public CLI settings. Credentials are resolved anew by the worker.
    options: Vec<String>,
    timeout: u64,
    branch: Option<String>,
}

// One atomic snapshot owns queue transitions. Claimed inputs are never replayed.
#[derive(Clone, Serialize, Deserialize)]
struct Input {
    id: String,
    prompt: String,
    phase: String,
}

fn inbox(dir: &Path) -> Result<Vec<Input>> {
    let path = dir.join("inbox.json");
    if path.exists() {
        return Ok(serde_json::from_reader(File::open(path)?)?);
    }
    // Read old tasks without modifying them. The first queue write migrates them.
    if dir.join("pending.json").exists() {
        let pending: Value = serde_json::from_reader(File::open(dir.join("pending.json"))?)?;
        return Ok(vec![Input {
            id: "legacy".into(),
            prompt: pending["prompt"]
                .as_str()
                .context("missing pending prompt")?
                .into(),
            phase: "queued".into(),
        }]);
    }
    Ok(Vec::new())
}

fn save_inbox(dir: &Path, inputs: &[Input]) -> Result<()> {
    write_json(&dir.join("inbox.json"), &inputs)?;
    // inbox.json becomes authoritative before removing the legacy pending file.
    if dir.join("pending.json").exists() {
        fs::remove_file(dir.join("pending.json"))?;
    }
    Ok(())
}

fn claim_input(dir: &Path) -> Result<Input> {
    let mut inputs = inbox(dir)?;
    ensure!(
        !inputs.iter().any(|i| i.phase == "claimed"),
        "previous turn may have run; inspect checkpoint before recovery"
    );
    let input = inputs
        .iter_mut()
        .find(|i| i.phase == "queued")
        .context("no queued input")?;
    input.phase = "claimed".into();
    let claimed = input.clone();
    save_inbox(dir, &inputs)?;
    Ok(claimed)
}

fn finish_input(dir: &Path, id: &str) -> Result<()> {
    let mut inputs = inbox(dir)?;
    let input = inputs
        .iter_mut()
        .find(|i| i.id == id && i.phase == "claimed")
        .context("missing claimed input")?;
    input.phase = "finished".into();
    save_inbox(dir, &inputs)
}

// Called with the control lock held after confirming the worker is inactive.
fn abandon_interrupted(dir: &Path, id: &str) -> Result<()> {
    let mut inputs = inbox(dir)?;
    let claimed: Vec<_> = inputs
        .iter()
        .filter(|input| input.phase == "claimed")
        .collect();
    ensure!(
        claimed.len() == 1 && claimed[0].id == id,
        "expected the exact ID of the interrupted input"
    );
    // If interrupted between writes, the remaining claim still blocks dispatch.
    write_json(&dir.join("state.json"), &json!({"status":"cancelled"}))?;
    inputs
        .iter_mut()
        .find(|input| input.id == id && input.phase == "claimed")
        .expect("claim checked above")
        .phase = "abandoned".into();
    save_inbox(dir, &inputs)
}

fn report(text: &str, structured: bool) -> Value {
    if !structured {
        return json!({"state":"waiting", "summary":"Turn ended; no structured task report requested", "advisory":true});
    }
    let parsed = serde_json::from_str::<Value>(text);
    if let Ok(value) = parsed {
        let state = value["state"].as_str().unwrap_or_default();
        let summary = value["summary"].as_str().unwrap_or_default();
        let question = value.get("question").and_then(Value::as_str);
        if matches!(
            state,
            "waiting" | "needs_input" | "ready_for_review" | "failed"
        ) && !summary.trim().is_empty()
            && summary.len() <= 8000
            && question.is_none_or(|q| q.len() <= 2000)
            && (state != "needs_input" || question.is_some_and(|q| !q.trim().is_empty()))
        {
            return json!({"state":state,"summary":summary,"question":question,"advisory":true});
        }
    }
    json!({"state":"needs_input", "summary":"Worker did not produce a valid structured report; inspect result.json", "question":"Review the worker result before resuming queued work", "invalid_report":true,"advisory":true})
}

const REPORT_INSTRUCTIONS: &str = "\n\nManaged task report contract: Finish your turn with a single JSON object, without markdown fences, with fields state, summary, and optional question. state must be waiting (this turn is done), needs_input (a specific human answer is required; include question), ready_for_review (a usable result awaits review), or failed (work could not be completed). Report actual evidence and uncertainty in summary; never imply that a successful model turn proves the work correct. This report does not grant permissions or authorize external actions.";

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

fn lock_is_contended(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::WouldBlock
        || error
            .raw_os_error()
            .is_some_and(|code| Some(code) == fs2::lock_contended_error().raw_os_error())
}

fn worker_active(dir: &Path) -> Result<bool> {
    let file = private_file(&dir.join("worker.lock"))?;
    match FileExt::try_lock_shared(&file) {
        Ok(()) => Ok(false), // close releases the probe lock
        Err(e) if lock_is_contended(&e) => Ok(true),
        Err(e) => Err(e.into()),
    }
}

fn status(dir: &Path) -> Result<Value> {
    let task = read_task(dir)?;
    let state: Value = serde_json::from_reader(File::open(dir.join("state.json"))?)?;
    let active = worker_active(dir)?;
    let inputs = inbox(dir)?;
    let queued = inputs.iter().filter(|i| i.phase == "queued").count();
    let claimed = inputs.iter().any(|i| i.phase == "claimed");
    let report = dir.join("report.json");
    let report: Value = if report.exists() {
        serde_json::from_reader(File::open(report)?)?
    } else {
        Value::Null
    };
    let starting = state["status"] == "starting"
        && state["started_at_ms"].as_i64().is_some_and(|start| {
            chrono::Utc::now().timestamp_millis().saturating_sub(start) < 10_000
        });
    let interrupted = !active
        && !starting
        && (claimed || matches!(state["status"].as_str(), Some("starting" | "running")));
    Ok(json!({"schema_version":1, "id":task.id, "cwd":task.cwd,
        "model":task.binding.model, "branch":task.branch,
        "session":task.session, "worker_active":active,
        "status":if interrupted { json!("interrupted") } else {state["status"].clone()},
        "queued_inputs":queued, "report":report,
        "inputs":inputs.iter().map(|i| json!({"id":i.id,"phase":i.phase})).collect::<Vec<_>>(),
        "stage":if interrupted { json!("interrupted") }
            else if active || starting { json!("working") }
            else if matches!(state["status"].as_str(), Some("failed" | "cancelled")) {state["status"].clone()}
            else {report.get("state").cloned().unwrap_or(json!("waiting"))},
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
    let active = worker_active(dir)?;
    let state = status(dir)?;
    ensure!(
        state["status"] != "interrupted",
        "previous worker was interrupted; recover its checkpoint before continuing"
    );
    let mut inputs = inbox(dir)?;
    ensure!(
        !inputs.iter().any(|i| i.phase == "claimed") || active,
        "previous turn may have run; inspect checkpoint before recovery"
    );
    ensure!(
        inputs.iter().filter(|i| i.phase == "queued").count() < 32,
        "task inbox has 32 queued inputs; wait for work to finish"
    );
    ensure!(
        inputs.len() < 256,
        "task reached 256 input receipts; create a new task"
    );
    inputs.push(Input {
        id: uuid::Uuid::new_v4().simple().to_string(),
        prompt: prompt.into(),
        phase: "queued".into(),
    });
    save_inbox(dir, &inputs)?;
    if active || state["status"] == "starting" {
        return Ok(());
    }
    start_queued(dir)
}

fn start_queued(dir: &Path) -> Result<()> {
    let inputs = inbox(dir)?;
    ensure!(
        !inputs.iter().any(|i| i.phase == "claimed"),
        "turn already began; inspect the checkpoint instead of replaying it"
    );
    ensure!(
        inputs.iter().any(|i| i.phase == "queued"),
        "no queued inputs"
    );
    let _ = fs::remove_file(dir.join("cancel"));
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
            Err(error) if lock_is_contended(&error) => continue,
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("four managed workers already active; retry after one finishes")
}

fn control_lock(dir: &Path) -> Result<File> {
    let file = private_file(&dir.join("control.lock"))?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => return Ok(file),
            Err(error) if lock_is_contended(&error) && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error).context("could not acquire task control lock"),
        }
    }
}

pub async fn handle(args: &cli::Cli, command: &TaskCommand) -> Result<()> {
    match command {
        TaskCommand::Start {
            prompt,
            cwd,
            worktree,
            structured_report,
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
                structured_report: *structured_report,
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
            start_queued(&dir)?;
            println!("{}", status(&dir)?);
        }
        TaskCommand::Recover { id, abandon } => {
            let dir = task_dir(id)?;
            let _control = control_lock(&dir)?;
            ensure!(!worker_active(&dir)?, "task worker is active");
            ensure!(
                status(&dir)?["status"] == "interrupted",
                "task is not interrupted"
            );
            abandon_interrupted(&dir, abandon)?;
            println!("{}", status(&dir)?);
        }
        TaskCommand::Resume { id } => {
            let dir = task_dir(id)?;
            let _control = control_lock(&dir)?;
            ensure!(!worker_active(&dir)?, "task worker is active");
            ensure!(
                !matches!(
                    status(&dir)?["status"].as_str(),
                    Some("interrupted" | "starting")
                ),
                "task is starting or interrupted; inspect its status before resuming"
            );
            start_queued(&dir)?;
            println!("{}", status(&dir)?);
        }
        TaskCommand::Cancel { id } => {
            let dir = task_dir(id)?;
            let _control = control_lock(&dir)?;
            ensure!(
                worker_active(&dir)? || inbox(&dir)?.iter().any(|i| i.phase == "queued"),
                "task has no active turn"
            );
            private_file(&dir.join("cancel"))?;
            if !worker_active(&dir)? && status(&dir)?["status"] != "starting" {
                write_json(&dir.join("state.json"), &json!({"status":"cancelled"}))?;
            }
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
            {
                let _control = control_lock(&dir)?;
                let state: Value = serde_json::from_reader(File::open(dir.join("state.json"))?)?;
                if state["status"] != "starting" {
                    return Ok(());
                }
            }
            loop {
                {
                    let _control = control_lock(&dir)?;
                    if dir.join("cancel").exists() {
                        write_json(&dir.join("state.json"), &json!({"status":"cancelled"}))?;
                        drop(lock);
                        return Ok(());
                    }
                    if !inbox(&dir)?.iter().any(|i| i.phase == "queued") {
                        drop(lock);
                        return Ok(());
                    }
                }
                if work(&dir).await.is_err() {
                    let _control = control_lock(&dir)?;
                    write_json(&dir.join("state.json"), &json!({"status":"failed"}))?;
                    let _ = event(
                        &dir,
                        "failed",
                        json!({"message":"worker failed; inspect configuration or checkpoint"}),
                    );
                    anyhow::bail!("background worker failed; inspect task state and checkpoint");
                }
                // Release worker ownership while still holding control, so send cannot
                // enqueue between the final empty check and worker shutdown.
                let _control = control_lock(&dir)?;
                let state: Value = serde_json::from_reader(File::open(dir.join("state.json"))?)?;
                if state["status"] != "idle" {
                    drop(lock);
                    return Ok(());
                }
                let handoff: Value = serde_json::from_reader(File::open(dir.join("report.json"))?)?;
                if matches!(
                    handoff["state"].as_str(),
                    Some("needs_input" | "ready_for_review" | "failed")
                ) || !inbox(&dir)?.iter().any(|i| i.phase == "queued")
                {
                    drop(lock);
                    return Ok(());
                }
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
    let mut system_prompt = crate::context::build_system_prompt_for_model(
        &resolved.binding.model,
        Some(&plugins),
        &config::HookTrigger::OnContextBuild,
        resolved.binding.provider_kind == config::ProviderKind::Anthropic,
        config.is_project_trusted(),
    )
    .await?;
    if task.structured_report {
        system_prompt.push_str(REPORT_INSTRUCTIONS);
    }
    engine.set_system_prompt(system_prompt);
    let (_, messages) = session::load_session(&task.session)?;
    engine.set_messages(messages);
    engine.set_archive(session::load_archive(&task.session)?);
    engine.set_transcript_checkpoint(dir.join("checkpoint.json"));
    let input = {
        let _control = control_lock(dir)?;
        if dir.join("cancel").exists() {
            write_json(&dir.join("state.json"), &json!({"status":"cancelled"}))?;
            return Ok(());
        }
        let input = claim_input(dir)?;
        write_json(&dir.join("state.json"), &json!({"status":"running"}))?;
        input
    };
    let prompt = &input.prompt;
    event(
        dir,
        "turn_started",
        json!({"model":engine.model(),"input_id":input.id}),
    )?;
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
    let handoff = if completed {
        report(
            value["result"].as_str().unwrap_or_default(),
            task.structured_report,
        )
    } else {
        json!({"state":"failed", "summary":"Turn failed or was cancelled; inspect the original outcome", "advisory":true})
    };
    value["task_report"] = handoff.clone();
    value["input_id"] = json!(input.id);
    write_json(&dir.join(format!("result-{}.json", input.id)), &value)?;
    write_json(&dir.join("result.json"), &value)?;
    write_json(&dir.join("report.json"), &handoff)?;
    event(
        dir,
        if completed {
            "turn_completed"
        } else {
            "turn_failed"
        },
        json!({"usage":engine.cost.usage_summary()}),
    )?;
    let _control = control_lock(dir)?;
    write_json(
        &dir.join("state.json"),
        &json!({"status":if completed {"idle"} else if cancel.is_cancelled() {"cancelled"} else {"failed"}}),
    )?;
    // A crash between these writes leaves a claimed input, which recovery can
    // resolve explicitly, rather than a running state with no claimed input.
    finish_input(dir, &input.id)?;
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
    fn control_lock_waits_for_a_concurrent_command() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let lock = control_lock(dir.path())?;
        let path = dir.path().to_owned();
        let (tx, rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let result = control_lock(&path);
            tx.send(result.is_ok()).unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(50)).is_err());
        drop(lock);
        assert!(rx.recv_timeout(Duration::from_secs(2))?);
        waiter.join().unwrap();
        Ok(())
    }

    #[test]
    fn inbox_claims_are_durable_and_never_automatically_replayed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let inputs = vec![
            Input {
                id: "a".into(),
                prompt: "first".into(),
                phase: "queued".into(),
            },
            Input {
                id: "b".into(),
                prompt: "second".into(),
                phase: "queued".into(),
            },
        ];
        save_inbox(dir.path(), &inputs)?;
        assert_eq!(claim_input(dir.path())?.id, "a");
        // Simulate a new worker opening persisted state after a crash.
        assert!(claim_input(dir.path()).is_err());
        assert_eq!(inbox(dir.path())?[1].phase, "queued");
        finish_input(dir.path(), "a")?;
        assert_eq!(claim_input(dir.path())?.id, "b");
        Ok(())
    }

    #[test]
    fn recovery_requires_exact_claim_and_preserves_queued_inputs() -> Result<()> {
        let dir = tempfile::tempdir()?;
        save_inbox(
            dir.path(),
            &[
                Input {
                    id: "a".into(),
                    prompt: "first".into(),
                    phase: "claimed".into(),
                },
                Input {
                    id: "b".into(),
                    prompt: "second".into(),
                    phase: "queued".into(),
                },
            ],
        )?;
        assert!(abandon_interrupted(dir.path(), "b").is_err());
        assert_eq!(inbox(dir.path())?[0].phase, "claimed");
        abandon_interrupted(dir.path(), "a")?;
        let inputs = inbox(dir.path())?;
        assert_eq!(inputs[0].phase, "abandoned");
        assert_eq!(inputs[1].phase, "queued");
        assert_eq!(claim_input(dir.path())?.id, "b");
        Ok(())
    }

    #[test]
    fn legacy_pending_migration_cannot_duplicate_an_input() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_json(
            &dir.path().join("pending.json"),
            &json!({"prompt":"legacy"}),
        )?;
        assert_eq!(claim_input(dir.path())?.prompt, "legacy");
        // A crash after committing the inbox but before deleting pending.json
        // must still treat inbox.json as authoritative.
        write_json(
            &dir.path().join("pending.json"),
            &json!({"prompt":"legacy"}),
        )?;
        assert!(claim_input(dir.path()).is_err());
        assert_eq!(inbox(dir.path())?.len(), 1);
        Ok(())
    }

    #[test]
    fn reports_require_valid_states_and_concrete_questions() {
        let valid = report(
            r#"{"state":"needs_input","summary":"Need a target","question":"Which cluster?"}"#,
            true,
        );
        assert_eq!(valid["question"], "Which cluster?");
        for text in [
            "done",
            r#"{"state":"success","summary":"done"}"#,
            r#"{"state":"needs_input","summary":"blocked"}"#,
            r#"{"state":"ready_for_review","summary":""}"#,
        ] {
            assert_eq!(report(text, true)["invalid_report"], true);
        }
        assert_eq!(
            report("arbitrary ordinary output", false)["state"],
            "waiting"
        );
        assert_eq!(
            report(
                r#"{"state":"ready_for_review","summary":"Tests passed"}"#,
                true
            )["advisory"],
            true
        );
    }

    #[test]
    fn recognizes_platform_lock_contention_without_masking_other_errors() {
        assert!(lock_is_contended(&fs2::lock_contended_error()));
        assert!(!lock_is_contended(&std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "permission denied",
        )));
        #[cfg(windows)]
        assert!(lock_is_contended(&std::io::Error::from_raw_os_error(33)));
    }

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
