use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::api::types::Message;

/// A conversation session, persisted as JSONL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub cwd: String,
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// An entry in the session JSONL file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
    #[serde(rename = "meta")]
    Meta(SessionMeta),
    #[serde(rename = "message")]
    Message { message: Message },
}

/// Get the session storage directory.
fn sessions_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
    let dir = base.join("claux").join("sessions");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Create a new session and return its ID.
pub fn create_session(model: &str) -> Result<(String, PathBuf)> {
    let id = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let dir = sessions_dir()?;
    let path = dir.join(format!("{}.jsonl", id));

    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let meta = SessionMeta {
        id: id.clone(),
        cwd,
        model: model.to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    let entry = SessionEntry::Meta(meta);
    let line = serde_json::to_string(&entry)?;
    std::fs::write(&path, format!("{}\n", line))?;

    Ok((id, path))
}

/// Append a message to a session file.
pub fn append_message(path: &PathBuf, message: &Message) -> Result<()> {
    use std::io::Write;

    let entry = SessionEntry::Message {
        message: message.clone(),
    };
    let line = serde_json::to_string(&entry)?;

    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)?;
    writeln!(file, "{}", line)?;

    Ok(())
}

/// Load all messages from a session file.
pub fn load_session(path: &PathBuf) -> Result<(SessionMeta, Vec<Message>)> {
    let content = std::fs::read_to_string(path)?;
    let mut meta = None;
    let mut messages = Vec::new();

    for line in content.lines() {
        if line.is_empty() {
            continue;
        }
        let entry: SessionEntry = serde_json::from_str(line)?;
        match entry {
            SessionEntry::Meta(m) => meta = Some(m),
            SessionEntry::Message { message } => messages.push(message),
        }
    }

    let meta = meta.ok_or_else(|| anyhow::anyhow!("Session file missing metadata"))?;
    Ok((meta, messages))
}

/// List available sessions, most recent first.
pub fn list_sessions() -> Result<Vec<(String, PathBuf)>> {
    let dir = sessions_dir()?;
    let mut sessions: Vec<(String, PathBuf)> = Vec::new();

    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "jsonl") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                sessions.push((stem.to_string(), path));
            }
        }
    }

    sessions.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(sessions)
}
