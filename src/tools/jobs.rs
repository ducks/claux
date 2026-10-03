//! Session-owned background Bash jobs. No detached services or extra authority.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::{bash::BashTool, Tool, ToolOutput};
use crate::command_sandbox::CommandSandbox;

const MAX_RUNNING: usize = 4;
const MAX_RETAINED: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Cancelling,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn active(&self) -> bool {
        matches!(self, Self::Running | Self::Cancelling)
    }
    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Cancelling => "cancelling",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug)]
pub struct JobSnapshot {
    pub id: String,
    pub command: String,
    pub status: JobStatus,
    pub elapsed: Duration,
    pub output: String,
}

struct Job {
    id: String,
    command: String,
    started: Instant,
    cancel: CancellationToken,
    progress: watch::Receiver<String>,
    result: Mutex<Option<(ToolOutput, Duration)>>,
    notified: AtomicBool,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Job {
    fn snapshot(&self) -> JobSnapshot {
        let result = self.result.lock().expect("job result poisoned");
        let (status, elapsed, output) = match &*result {
            Some((output, elapsed)) => (
                if self.cancel.is_cancelled() {
                    JobStatus::Cancelled
                } else if output.is_error {
                    JobStatus::Failed
                } else {
                    JobStatus::Succeeded
                },
                *elapsed,
                output.content.clone(),
            ),
            None => (
                if self.cancel.is_cancelled() {
                    JobStatus::Cancelling
                } else {
                    JobStatus::Running
                },
                self.started.elapsed(),
                crate::utils::sanitize_terminal_text(&self.progress.borrow()),
            ),
        };
        JobSnapshot {
            id: self.id.clone(),
            command: self.command.clone(),
            status,
            elapsed,
            output,
        }
    }
}

#[derive(Default)]
pub struct JobManager {
    enabled: AtomicBool,
    jobs: Mutex<Vec<Arc<Job>>>,
}

impl JobManager {
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    pub fn start(
        &self,
        sandbox: Arc<CommandSandbox>,
        mut input: Value,
        cancel: &CancellationToken,
    ) -> Result<ToolOutput> {
        if !self.enabled.load(Ordering::Relaxed) {
            bail!("Background jobs require an interactive session. Run this Bash command in the foreground instead.");
        }
        if cancel.is_cancelled() {
            return Ok(super::interrupted_output());
        }
        let mut jobs = self.jobs.lock().expect("jobs poisoned");
        if !self.enabled.load(Ordering::Relaxed) {
            bail!("This session is closing; background jobs are disabled.");
        }
        if jobs
            .iter()
            .filter(|job| job.snapshot().status.active())
            .count()
            >= MAX_RUNNING
        {
            bail!("At most {MAX_RUNNING} background jobs may run. Inspect or cancel a job with Jobs first.");
        }
        if jobs.len() >= MAX_RETAINED {
            if let Some(index) = jobs.iter().position(|job| !job.snapshot().status.active()) {
                jobs.remove(index);
            }
        }
        let id = format!("job-{}", uuid::Uuid::new_v4().simple());
        let command = input["command"].as_str().unwrap_or_default().to_string();
        input["background"] = json!(false);
        let (progress, updates) = watch::channel(String::new());
        let job = Arc::new(Job {
            id: id.clone(),
            command: crate::utils::sanitize_terminal_text(&command),
            started: Instant::now(),
            cancel: CancellationToken::new(),
            progress: updates,
            result: Mutex::new(None),
            notified: AtomicBool::new(false),
            task: Mutex::new(None),
        });
        let running = job.clone();
        let task = tokio::spawn(async move {
            let mut output = BashTool::new(sandbox)
                .execute_with_progress(input, running.cancel.clone(), Some(progress))
                .await
                .unwrap_or_else(|error| ToolOutput {
                    sub_agent: None,
                    content: format!("Bash failed: {error}"),
                    is_error: true,
                });
            output.content = crate::utils::sanitize_terminal_text(&output.content);
            *running.result.lock().expect("job result poisoned") =
                Some((output, running.started.elapsed()));
        });
        *job.task.lock().expect("job task poisoned") = Some(task);
        jobs.push(job);
        Ok(ToolOutput { sub_agent: None, content: format!("Started background Bash {id}. This is a launch acknowledgement, not command success. Use Jobs to inspect output or cancel. Jobs stop when this session closes; the Bash timeout still applies."), is_error: false })
    }

    pub fn snapshots(&self) -> Vec<JobSnapshot> {
        self.jobs
            .lock()
            .expect("jobs poisoned")
            .iter()
            .map(|job| job.snapshot())
            .collect()
    }

    pub fn completions(&self) -> Vec<JobSnapshot> {
        self.jobs
            .lock()
            .expect("jobs poisoned")
            .iter()
            .filter_map(|job| {
                let snapshot = job.snapshot();
                (!snapshot.status.active() && !job.notified.swap(true, Ordering::Relaxed))
                    .then_some(snapshot)
            })
            .collect()
    }

    pub fn cancel(&self, id: &str) -> Result<()> {
        let jobs = self.jobs.lock().expect("jobs poisoned");
        let job = jobs
            .iter()
            .find(|job| job.id == id)
            .ok_or_else(|| anyhow::anyhow!("Unknown job: {id}"))?;
        if job.result.lock().expect("job result poisoned").is_none() {
            job.cancel.cancel();
        }
        Ok(())
    }

    pub fn cancel_all(&self) {
        for job in self.jobs.lock().expect("jobs poisoned").iter() {
            if job.result.lock().expect("job result poisoned").is_none() {
                job.cancel.cancel();
            }
        }
    }

    pub async fn shutdown(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.cancel_all();
        let tasks: Vec<_> = self
            .jobs
            .lock()
            .expect("jobs poisoned")
            .iter()
            .filter_map(|job| job.task.lock().expect("job task poisoned").take())
            .collect();
        futures_util::future::join_all(tasks.into_iter().map(|mut task| async move {
            if tokio::time::timeout(Duration::from_secs(4), &mut task)
                .await
                .is_err()
            {
                task.abort();
                let _ = task.await;
            }
        }))
        .await;
    }

    pub fn reset(&self) {
        self.cancel_all();
        self.jobs.lock().expect("jobs poisoned").clear();
    }

    pub fn command(&self, args: &str) -> Result<String> {
        let words: Vec<_> = args.split_whitespace().collect();
        match words.as_slice() {
            [] => {
                let jobs = self.snapshots();
                if jobs.is_empty() {
                    return Ok("No background jobs in this session.".into());
                }
                Ok(jobs
                    .iter()
                    .map(|job| {
                        format!(
                            "{} | {} | {}s | {}",
                            job.id,
                            job.status.label(),
                            job.elapsed.as_secs(),
                            crate::utils::truncate_str(&job.command, 120)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            ["cancel", id] => {
                self.cancel(id)?;
                Ok(format!("Cancellation requested for {id}."))
            }
            [id] => {
                let job = self
                    .snapshots()
                    .into_iter()
                    .find(|job| job.id == *id)
                    .ok_or_else(|| anyhow::anyhow!("Unknown job: {id}"))?;
                Ok(format!(
                    "{} | {} | {}s\n{}\n{}",
                    job.id,
                    job.status.label(),
                    job.elapsed.as_secs(),
                    job.command,
                    job.output
                ))
            }
            _ => bail!("Usage: /jobs [job-id | cancel job-id]"),
        }
    }
}

impl Drop for JobManager {
    fn drop(&mut self) {
        self.cancel_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox() -> Arc<CommandSandbox> {
        Arc::new(CommandSandbox::unrestricted_for_tests())
    }

    async fn finished(manager: &JobManager) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while manager.snapshots().iter().any(|job| job.status.active()) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("jobs must finish promptly");
    }

    #[tokio::test]
    async fn background_launch_returns_early_and_completion_is_reported_once() {
        let manager = JobManager::default();
        manager.enable();
        let cancel = CancellationToken::new();
        let output = manager
            .start(
                sandbox(),
                json!({"command":"printf ready; sleep 0.2; printf done"}),
                &cancel,
            )
            .unwrap();
        assert!(output.content.contains("launch acknowledgement"));
        assert_eq!(manager.snapshots()[0].status, JobStatus::Running);
        // Foreground turn cancellation must not kill an acknowledged job.
        cancel.cancel();
        finished(&manager).await;
        let jobs = manager.completions();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, JobStatus::Succeeded);
        assert_eq!(jobs[0].output, "readydone");
        assert!(manager.completions().is_empty());
        manager.cancel(&jobs[0].id).unwrap();
        assert_eq!(manager.snapshots()[0].status, JobStatus::Succeeded);
    }

    #[tokio::test]
    async fn concurrency_is_bounded_and_shutdown_cancels_every_job() {
        let manager = JobManager::default();
        manager.enable();
        for _ in 0..MAX_RUNNING {
            manager
                .start(
                    sandbox(),
                    json!({"command":"sleep 30"}),
                    &CancellationToken::new(),
                )
                .unwrap();
        }
        assert!(manager
            .start(
                sandbox(),
                json!({"command":"true"}),
                &CancellationToken::new()
            )
            .is_err());
        manager.shutdown().await;
        assert!(manager
            .snapshots()
            .iter()
            .all(|job| job.status == JobStatus::Cancelled));
        assert!(manager.cancel("missing").is_err());
    }

    #[tokio::test]
    async fn capture_and_retention_are_bounded_and_timeouts_are_failures() {
        let manager = JobManager::default();
        manager.enable();
        manager
            .start(
                sandbox(),
                json!({"command":"head -c 200000 /dev/zero | tr '\\0' x; sleep 30", "timeout":150}),
                &CancellationToken::new(),
            )
            .unwrap();
        finished(&manager).await;
        let job = manager.snapshots().remove(0);
        assert_eq!(job.status, JobStatus::Failed);
        assert!(job.output.contains("truncated"));
        assert!(job.output.contains("timed out"));
        assert!(job.output.len() < 101_000);
        for _ in 0..MAX_RETAINED {
            manager
                .start(
                    sandbox(),
                    json!({"command":"true"}),
                    &CancellationToken::new(),
                )
                .unwrap();
            finished(&manager).await;
        }
        assert_eq!(manager.snapshots().len(), MAX_RETAINED);
        assert!(!manager
            .snapshots()
            .iter()
            .any(|current| current.id == job.id));
    }

    #[tokio::test]
    async fn disabled_cancelled_and_failed_jobs_do_not_look_successful() {
        let manager = JobManager::default();
        assert!(manager
            .start(
                sandbox(),
                json!({"command":"true"}),
                &CancellationToken::new()
            )
            .is_err());
        manager.enable();
        let token = CancellationToken::new();
        token.cancel();
        assert!(
            manager
                .start(sandbox(), json!({"command":"true"}), &token)
                .unwrap()
                .is_error
        );
        assert!(manager.snapshots().is_empty());
        manager
            .start(
                sandbox(),
                json!({"command":"printf failure >&2; exit 7"}),
                &CancellationToken::new(),
            )
            .unwrap();
        finished(&manager).await;
        assert_eq!(manager.snapshots()[0].status, JobStatus::Failed);
        assert!(manager.snapshots()[0].output.contains("failure"));
    }

    #[tokio::test]
    async fn manager_drop_cancels_the_owned_command() {
        let manager = JobManager::default();
        manager.enable();
        manager
            .start(
                sandbox(),
                json!({"command":"printf ready; sleep 30"}),
                &CancellationToken::new(),
            )
            .unwrap();
        let job = manager.jobs.lock().unwrap()[0].clone();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !job.progress.borrow().contains("ready") {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        drop(manager);
        assert!(job.cancel.is_cancelled());
        let task = job.task.lock().unwrap().take().unwrap();
        tokio::time::timeout(Duration::from_secs(4), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.snapshot().status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn jobs_tool_inspects_cancels_and_rejects_invalid_requests() {
        let manager = Arc::new(JobManager::default());
        manager.enable();
        let tool = JobsTool(manager.clone());
        assert!(tool
            .execute(json!({"action":"list"}), CancellationToken::new())
            .await
            .unwrap()
            .content
            .contains("No background jobs"));
        manager
            .start(
                sandbox(),
                json!({"command":"sleep 30"}),
                &CancellationToken::new(),
            )
            .unwrap();
        let id = manager.snapshots()[0].id.clone();
        let output = tool
            .execute(
                json!({"action":"output", "job_id":id}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(output.content.contains("running"));
        assert!(output.content.contains("sleep 30"));
        assert!(tool
            .execute(
                json!({"action":"output", "job_id":"job-1\ncancel job-2"}),
                CancellationToken::new()
            )
            .await
            .is_err());
        assert!(tool
            .execute(json!({"action":"output"}), CancellationToken::new())
            .await
            .is_err());
        tool.execute(
            json!({"action":"cancel", "job_id":id}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        manager.shutdown().await;
        assert_eq!(manager.snapshots()[0].status, JobStatus::Cancelled);
        assert!(manager
            .start(
                sandbox(),
                json!({"command":"true"}),
                &CancellationToken::new()
            )
            .is_err());
        manager.reset();
        assert!(manager.snapshots().is_empty());
    }
}

pub struct JobsTool(pub Arc<JobManager>);

#[async_trait]
impl Tool for JobsTool {
    fn name(&self) -> &str {
        "Jobs"
    }
    fn description(&self) -> &str {
        "List session background Bash jobs, inspect bounded output, or cancel a job. Use this to check actual completion; launch acknowledgements do not mean success."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object", "properties":{"action":{"type":"string","enum":["list","output","cancel"]},"job_id":{"type":"string"}}, "required":["action"]})
    }
    fn is_read_only(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, cancel: CancellationToken) -> Result<ToolOutput> {
        if cancel.is_cancelled() {
            return Ok(super::interrupted_output());
        }
        let args = match input["action"].as_str() {
            Some("list") => String::new(),
            Some(action @ ("output" | "cancel")) => {
                let id = input["job_id"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("job_id is required"))?;
                if !id.starts_with("job-")
                    || id.len() != 36
                    || !id[4..].chars().all(|c| c.is_ascii_hexdigit())
                {
                    bail!("Invalid job ID");
                }
                if action == "cancel" {
                    format!("cancel {id}")
                } else {
                    id.into()
                }
            }
            _ => bail!("action must be list, output, or cancel"),
        };
        Ok(ToolOutput {
            sub_agent: None,
            content: self.0.command(&args)?,
            is_error: false,
        })
    }
}
