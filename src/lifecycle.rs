//! Bounded metadata-only lifecycle events for supervised one-shot runs.
use crate::query::StreamEvent;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    sync::Mutex,
    time::Instant,
};

struct State {
    file: File,
    sequence: u64,
}
pub struct LifecycleLog {
    state: Mutex<State>,
    started: Instant,
}
impl LifecycleLog {
    pub fn create(path: &Path) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(path).context("cannot create lifecycle log")?;
        Ok(Self {
            state: Mutex::new(State { file, sequence: 0 }),
            started: Instant::now(),
        })
    }
    fn write(&self, kind: &str, details: Value) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("lifecycle log lock poisoned"))?;
        let record = json!({"schema_version":1,"sequence":state.sequence,"elapsed_ms":self.started.elapsed().as_millis(),"event":kind,"details":details});
        serde_json::to_writer(&mut state.file, &record)?;
        state.file.write_all(b"\n")?;
        state.file.flush()?;
        state.sequence += 1;
        Ok(())
    }
    pub fn started(&self) -> Result<()> {
        self.write("turn_started", json!({}))
    }
    pub fn finished(&self, success: bool) -> Result<()> {
        self.write("turn_finished", json!({"success":success}))
    }
    pub fn record(&self, event: &StreamEvent) -> Result<()> {
        match event {
            StreamEvent::ModelRequest => self.write("model_request", json!({})),
            StreamEvent::ToolStart { name, .. } => {
                let tool = match name.as_str() {
                    "Read" | "Write" | "Edit" | "Bash" | "Glob" | "Grep" | "WebFetch" | "Agent"
                    | "TaskManage" | "McpManage" | "Jobs" | "TaskReport" | "TodoWrite" => {
                        name.as_str()
                    }
                    _ => "other",
                };
                self.write("tool_started", json!({"tool":tool}))
            }
            StreamEvent::ToolRunning { index } => {
                self.write("tool_running", json!({"index":index}))
            }
            StreamEvent::ToolFinished {
                index, is_error, ..
            } => self.write("tool_finished", json!({"index":index,"is_error":is_error})),
            StreamEvent::ToolResult { is_error, .. } => {
                self.write("tool_result", json!({"is_error":is_error}))
            }
            StreamEvent::Retry(_) => self.write("model_retry", json!({})),
            StreamEvent::Interrupted => self.write("interrupted", json!({})),
            StreamEvent::Error(_) => self.write("engine_error", json!({})),
            // Text, tool content, permission payloads, reasoning and notices
            // are intentionally excluded from this metadata-only log.
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn events_are_private_metadata_and_never_overwrite() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("events");
        let log = LifecycleLog::create(&path).unwrap();
        log.started().unwrap();
        log.record(&StreamEvent::ToolStart {
            name: "Write".into(),
            summary: "private-summary".into(),
            input: json!({"content":"private-content"}),
        })
        .unwrap();
        log.record(&StreamEvent::ToolResult {
            is_error: false,
            content: "private-output".into(),
        })
        .unwrap();
        log.record(&StreamEvent::Text("private-response".into()))
            .unwrap();
        log.finished(true).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("private"));
        let rows: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 4);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row["sequence"], index);
        }
        assert_eq!(rows[1]["details"]["tool"], "Write");
        assert!(LifecycleLog::create(&path).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o077,
                0
            );
        }
    }
}
