use anyhow::Result;
use async_trait::async_trait;
#[cfg(windows)]
use process_wrap::tokio::JobObject;
use process_wrap::tokio::{CommandWrap, KillOnDrop};
use serde::Deserialize;
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput};
use crate::command_sandbox::CommandSandbox;

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);
/// Bytes of each stream retained in memory. Anything past this is drained
/// and counted, never stored, so a runaway command cannot exhaust memory
/// before the tool result is truncated for the model.
const OUTPUT_DRAIN_TIMEOUT: Duration = Duration::from_secs(1);
const CHILD_REAP_TIMEOUT: Duration = Duration::from_secs(1);
const CAPTURE_LIMIT: usize = 50_000;

#[derive(Default)]
struct Capture {
    bytes: Vec<u8>,
    total: usize,
}

impl Capture {
    fn append(&mut self, bytes: &[u8]) {
        self.total = self.total.saturating_add(bytes.len());
        let keep = bytes.len().min(CAPTURE_LIMIT - self.bytes.len());
        self.bytes.extend_from_slice(&bytes[..keep]);
    }

    fn render(&self, stream: &str) -> String {
        let truncated = self.total > self.bytes.len();
        let bytes = match std::str::from_utf8(&self.bytes) {
            Err(error) if truncated && error.error_len().is_none() => {
                &self.bytes[..error.valid_up_to()]
            }
            _ => &self.bytes,
        };
        let mut text = render_output(bytes, stream);
        if truncated {
            text.push_str(&format!(
                "\n... ({stream} truncated; {} bytes omitted)",
                self.total - bytes.len()
            ));
        }
        text
    }
}

fn capture_pipe<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    mut pipe: Option<R>,
    stream: &'static str,
    progress: Option<tokio::sync::watch::Sender<String>>,
) -> (JoinHandle<()>, Arc<Mutex<Capture>>) {
    let capture = Arc::new(Mutex::new(Capture::default()));
    let reader_capture = capture.clone();
    let task = tokio::spawn(async move {
        if let Some(pipe) = pipe.as_mut() {
            let mut chunk = [0; 8192];
            let mut tail = Vec::new();
            while let Ok(count) = pipe.read(&mut chunk).await {
                if count == 0 {
                    break;
                }
                reader_capture
                    .lock()
                    .expect("capture poisoned")
                    .append(&chunk[..count]);
                if let Some(progress) = &progress {
                    tail.extend_from_slice(&chunk[..count]);
                    if tail.len() > 4096 {
                        tail.drain(..tail.len() - 4096);
                    }
                    progress
                        .send_replace(format!("[{stream}]\n{}", String::from_utf8_lossy(&tail)));
                }
            }
        }
    });
    (task, capture)
}

pub struct BashTool {
    sandbox: Arc<CommandSandbox>,
    jobs: Option<Arc<super::jobs::JobManager>>,
}

impl BashTool {
    pub fn new(sandbox: Arc<CommandSandbox>) -> Self {
        Self {
            sandbox,
            jobs: None,
        }
    }

    pub fn with_jobs(sandbox: Arc<CommandSandbox>, jobs: Arc<super::jobs::JobManager>) -> Self {
        Self {
            sandbox,
            jobs: Some(jobs),
        }
    }
}

#[derive(Deserialize)]
struct Params {
    command: String,
    #[serde(default)]
    background: bool,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default)]
    #[serde(rename = "description")]
    _description: Option<String>,
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "Bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command. Set background=true only when the user requests background work, allowing conversation to continue while it runs. Interactive sessions only. Use Jobs to inspect completion or cancel; starting a job does not mean its command succeeded."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "background": {"type": "boolean", "description": "Run as a session-owned background job (default false). Same permissions, sandbox and timeout as foreground Bash; cancelled when the session closes."},
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (max 600000, default 120000)"
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what the command does"
                }
            },
            "required": ["command"]
        })
    }

    fn is_read_only(&self) -> bool {
        false // conservative default; could be smarter with command analysis
    }

    fn summarize(&self, input: &Value) -> String {
        let cmd = input["command"].as_str().unwrap_or("?");
        // Truncate long commands
        if cmd.len() > 80 {
            format!("{}...", crate::utils::truncate_str(cmd, 77))
        } else {
            cmd.to_string()
        }
    }

    async fn execute(&self, input: Value, cancel: CancellationToken) -> Result<ToolOutput> {
        self.execute_with_progress(input, cancel, None).await
    }

    async fn execute_with_progress(
        &self,
        input: Value,
        cancel: CancellationToken,
        progress: Option<tokio::sync::watch::Sender<String>>,
    ) -> Result<ToolOutput> {
        let params: Params = serde_json::from_value(input.clone())?;
        if cancel.is_cancelled() {
            return Ok(super::interrupted_output());
        }
        if params.background {
            let jobs = self.jobs.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Background jobs are unavailable here; use foreground Bash.")
            })?;
            return jobs.start(self.sandbox.clone(), input, &cancel);
        }

        let timeout_ms = params.timeout.unwrap_or(120_000).min(600_000);
        let timeout = Duration::from_millis(timeout_ms);

        let mut inner = self.sandbox.command(&params.command)?;
        inner.stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(unix)]
        inner.process_group(0);
        let mut command = CommandWrap::from(inner);
        // Commands commonly create descendants (shell pipelines, test runners,
        // build systems). Put the whole tree in one killable unit so cancelling
        // the tool cannot leave grandchildren alive with our pipes still open.
        command.wrap(KillOnDrop);
        #[cfg(windows)]
        command.wrap(JobObject);

        let mut child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput {
                    sub_agent: None,
                    content: format!("Failed to execute command: {e}"),
                    is_error: true,
                });
            }
        };
        let process_group = child.id();

        // Take the pipes so we can read them concurrently with wait().
        let (mut stdout_task, stdout) =
            capture_pipe(child.stdout().take(), "stdout", progress.clone());
        let (mut stderr_task, stderr) = capture_pipe(child.stderr().take(), "stderr", progress);

        let outcome = tokio::select! {
            status = wait_for_parent(&mut child) => Outcome::Finished(status),
            _ = cancel.cancelled() => Outcome::Cancelled,
            _ = tokio::time::sleep(timeout) => Outcome::TimedOut,
        };

        // A shell can exit successfully while a background descendant remains
        // alive and holds stdout/stderr open. Always terminate anything left
        // in the command's process tree before draining output. Persistent
        // services must detach into their own service manager and redirect
        // their streams rather than inheriting a tool invocation's pipes.
        let cleanup = terminate_process_tree(&mut child, process_group);

        if !matches!(outcome, Outcome::Finished(_)) {
            let _ = tokio::time::timeout(CHILD_REAP_TIMEOUT, wait_for_parent(&mut child)).await;
        }

        let stdout_abandoned = drain_reader(&mut stdout_task).await;
        let stderr_abandoned = drain_reader(&mut stderr_task).await;
        let output_abandoned = stdout_abandoned || stderr_abandoned;

        let stdout_s = stdout.lock().expect("capture poisoned").render("stdout");
        let stderr_s = stderr.lock().expect("capture poisoned").render("stderr");

        let mut content = String::new();
        if !stdout_s.is_empty() {
            content.push_str(&stdout_s);
        }
        if !stderr_s.is_empty() {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(&stderr_s);
        }

        let mut is_error = match &outcome {
            Outcome::Finished(Ok(status)) => {
                if !status.success() {
                    content.push_str(&format!("\nExit code: {status}"));
                }
                !status.success()
            }
            Outcome::Finished(Err(e)) => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&format!("wait error: {e}"));
                true
            }
            Outcome::Cancelled => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str("Interrupted by user.");
                true
            }
            Outcome::TimedOut => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&format!("Command timed out after {timeout_ms}ms"));
                true
            }
        };

        if matches!(outcome, Outcome::Finished(Ok(status)) if status.success())
            && matches!(cleanup, Ok(true))
        {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(
                "Background processes were terminated when the command exited. Use a service manager for persistent processes.",
            );
            is_error = true;
        }

        if let Err(error) = cleanup {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(&format!("Failed to clean up command process tree: {error}"));
            is_error = true;
        }

        if output_abandoned {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(
                "A detached process kept command output open; Claux stopped waiting for its output.",
            );
            is_error = true;
        }

        Ok(ToolOutput {
            sub_agent: None,
            content,
            is_error,
        })
    }
}

async fn wait_for_parent(
    child: &mut Box<dyn process_wrap::tokio::ChildWrapper>,
) -> std::io::Result<std::process::ExitStatus> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        tokio::time::sleep(CHILD_POLL_INTERVAL).await;
    }
}

#[cfg(unix)]
fn terminate_process_tree(
    _child: &mut Box<dyn process_wrap::tokio::ChildWrapper>,
    process_group: Option<u32>,
) -> std::io::Result<bool> {
    use nix::sys::signal::{killpg, Signal};
    use nix::unistd::Pid;

    let Some(pid) = process_group.and_then(|pid| i32::try_from(pid).ok()) else {
        return Ok(false);
    };
    match killpg(Pid::from_raw(pid), Signal::SIGKILL) {
        Ok(()) => Ok(true),
        Err(nix::errno::Errno::ESRCH) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

#[cfg(not(unix))]
fn terminate_process_tree(
    child: &mut Box<dyn process_wrap::tokio::ChildWrapper>,
    _process_group: Option<u32>,
) -> std::io::Result<bool> {
    child.start_kill()?;
    // TerminateJobObject also succeeds for an empty job. The wrapper does
    // not expose its active process count, so success is not evidence that
    // descendants remained. Still terminate the job to prevent leaks.
    Ok(false)
}

async fn drain_reader(task: &mut JoinHandle<()>) -> bool {
    match tokio::time::timeout(OUTPUT_DRAIN_TIMEOUT, &mut *task).await {
        Ok(_) => false,
        Err(_) => {
            task.abort();
            let _ = task.await;
            true
        }
    }
}

fn render_output(bytes: &[u8], stream: &str) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) if !bytes.contains(&0) => text.to_string(),
        _ => format!("[binary {stream} suppressed: {} bytes]", bytes.len()),
    }
}

enum Outcome {
    Finished(std::io::Result<std::process::ExitStatus>),
    Cancelled,
    TimedOut,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> CancellationToken {
        CancellationToken::new()
    }

    #[tokio::test]
    async fn live_output_is_bounded_and_arrives_before_completion() {
        let (tx, mut rx) = tokio::sync::watch::channel(String::new());
        let cancel = token();
        let tool = tool();
        let execution = tool.execute_with_progress(
            json!({
                "command": "head -c 20000 /dev/zero | tr '\\0' x; printf '\\nready\\n'; sleep 30"
            }),
            cancel.clone(),
            Some(tx),
        );
        let observer = async {
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    rx.changed().await.unwrap();
                    let preview = rx.borrow_and_update().clone();
                    assert!(preview.len() < 4200);
                    if preview.contains("ready") {
                        break;
                    }
                }
            })
            .await
            .unwrap();
            cancel.cancel();
        };
        let (output, ()) = tokio::join!(execution, observer);
        assert!(output.unwrap().content.contains("Interrupted by user"));
    }

    fn tool() -> BashTool {
        BashTool::new(Arc::new(CommandSandbox::unrestricted_for_tests()))
    }

    #[test]
    fn capture_is_bounded_and_truncates_unicode_safely() {
        let mut capture = Capture::default();
        for _ in 0..100 {
            capture.append("界".repeat(8192).as_bytes());
        }
        assert_eq!(capture.bytes.len(), CAPTURE_LIMIT);
        assert_eq!(capture.total, 100 * 8192 * 3);
        let rendered = capture.render("stdout");
        assert!(rendered.starts_with("界"));
        assert!(rendered.contains("stdout truncated"));
        assert!(!rendered.contains("binary"));
        assert!(!rendered.contains('\u{fffd}'));
    }

    #[tokio::test]
    async fn noisy_streams_are_drained_and_keep_failure_diagnostics() {
        let output = tool().execute(json!({
            "command": "head -c 200000 /dev/zero | tr '\\0' x; head -c 200000 /dev/zero | tr '\\0' y >&2; exit 7"
        }), token()).await.unwrap();
        assert!(output.is_error);
        assert!(output.content.contains("stdout truncated"));
        assert!(output.content.contains("stderr truncated"));
        assert!(output.content.contains("Exit code:"));
        assert!(output.content.len() < 101_000);
    }

    #[tokio::test(start_paused = true)]
    async fn abandoning_reader_preserves_partial_output() {
        use tokio::io::AsyncWriteExt;
        let (mut writer, reader) = tokio::io::duplex(64);
        let (mut task, capture) = capture_pipe(Some(reader), "stdout", None);
        writer.write_all(b"partial").await.unwrap();
        tokio::task::yield_now().await;
        assert!(drain_reader(&mut task).await);
        assert_eq!(capture.lock().unwrap().render("stdout"), "partial");
    }

    #[tokio::test]
    async fn bash_echo() {
        let tool = tool();
        let result = tool
            .execute(json!({"command": "echo hello"}), token())
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.trim().contains("hello"));
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn completed_windows_job_cleanup_does_not_report_residual_processes() {
        let mut inner = tokio::process::Command::new("cmd.exe");
        inner.args(["/C", "exit 0"]);
        let mut command = CommandWrap::from(inner);
        command.wrap(KillOnDrop).wrap(JobObject);
        let mut child = command.spawn().unwrap();
        let status = tokio::time::timeout(Duration::from_secs(10), wait_for_parent(&mut child))
            .await
            .expect("command must exit promptly")
            .unwrap();
        assert!(status.success());
        assert!(!terminate_process_tree(&mut child, None).unwrap());
    }

    #[tokio::test]
    async fn bash_output_beyond_the_capture_limit_is_drained_not_stored() {
        let tool = tool();
        // 8 MiB of output: far beyond the per-stream capture limit. The
        // command must finish (the pipe is drained), the result must stay
        // bounded, and the omitted bytes must be reported.
        let result = tool
            .execute(
                json!({"command": "head -c 8388608 /dev/zero | tr '\\0' 'x'"}),
                token(),
            )
            .await
            .unwrap();
        assert!(
            result.content.len() <= CAPTURE_LIMIT + 160,
            "{}",
            result.content.len()
        );
        assert!(
            result
                .content
                .contains("stdout truncated; 8338608 bytes omitted"),
            "{}",
            &result.content[result.content.len().saturating_sub(200)..]
        );
        assert!(!result.content.contains("timed out"));
    }

    #[tokio::test]
    async fn bash_exit_code() {
        let tool = tool();
        let result = tool
            .execute(json!({"command": "exit 1"}), token())
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Exit code"));
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tool = tool();
        let result = tool
            .execute(json!({"command": "echo err >&2"}), token())
            .await
            .unwrap();
        assert!(result.content.contains("err"));
    }

    #[test]
    fn preserves_utf8_output() {
        assert_eq!(render_output("héllo\n".as_bytes(), "stdout"), "héllo\n");
    }

    #[test]
    fn suppresses_invalid_utf8_output() {
        assert_eq!(
            render_output(&[0xff, 0xfe, 0xfd], "stdout"),
            "[binary stdout suppressed: 3 bytes]"
        );
    }

    #[test]
    fn suppresses_nul_containing_output() {
        assert_eq!(
            render_output(b"text\0more", "stderr"),
            "[binary stderr suppressed: 9 bytes]"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn binary_stdout_does_not_hide_text_stderr() {
        let result = tool()
            .execute(
                json!({"command": "printf '\\377'; printf 'warning' >&2"}),
                token(),
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(
            result.content,
            "[binary stdout suppressed: 1 bytes]\nwarning"
        );
        assert!(!result.content.contains('\u{fffd}'));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = tool();
        let result = tool
            .execute(json!({"command": "sleep 10", "timeout": 100}), token())
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }

    #[tokio::test]
    async fn bash_cancellation() {
        let tool = tool();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            cancel_clone.cancel();
        });
        let start = std::time::Instant::now();
        let result = tool
            .execute(
                json!({
                    "command": "trap '' HUP; sleep 30 & wait",
                    "timeout": 60000
                }),
                cancel,
            )
            .await
            .unwrap();
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "cancellation should kill the entire process tree (took {:?})",
            start.elapsed()
        );
        assert!(result.is_error);
        assert!(result.content.contains("Interrupted"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_process_cannot_hold_a_completed_command_open() {
        let start = std::time::Instant::now();
        let result = tool()
            .execute(
                json!({
                    "command": "nohup sleep 30 >/dev/null 2>&1 & printf ready",
                    "timeout": 60000
                }),
                token(),
            )
            .await
            .unwrap();

        assert!(
            start.elapsed() < Duration::from_secs(3),
            "background descendants must not hold the tool open (took {:?})",
            start.elapsed()
        );
        assert!(result.is_error);
        assert!(result.content.contains("ready"));
        assert!(result
            .content
            .contains("Background processes were terminated"));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn detached_process_cannot_hold_output_capture_open() {
        let dir = tempfile::tempdir().unwrap();
        let ready = dir.path().join("ready");
        // Wait until setsid has actually detached before letting the parent
        // exit and trigger process-group cleanup.
        let command = format!(
            "setsid sh -c 'touch \"{}\"; sleep 2' & while [ ! -e \"{}\" ]; do sleep 0.01; done; printf ready",
            ready.display(), ready.display()
        );
        let start = std::time::Instant::now();
        let result = tool()
            .execute(
                json!({
                    "command": command,
                    "timeout": 60000
                }),
                token(),
            )
            .await
            .unwrap();

        assert!(
            start.elapsed() < Duration::from_secs(3),
            "escaped descendants must not hold the tool open (took {:?})",
            start.elapsed()
        );
        assert!(result.is_error);
        assert!(result.content.contains("stopped waiting"));
    }
}
