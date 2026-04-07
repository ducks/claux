//! Session storage using SQLite.
//!
//! Provides persistent storage for chat sessions with fast random access,
//! metadata tracking, and querying capabilities.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::api::types::Message;
use crate::db::{Db, SessionInfo};

/// Get the database path.
fn db_path() -> Result<PathBuf> {
    let base = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
    let dir = base.join("claux");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("sessions.db"))
}

/// Get the database instance (lazy initialization).
fn get_db() -> Result<Db> {
    let path = db_path()?;
    Db::open(&path).context("Failed to open session database")
}

/// Create a new session and return its ID and a dummy path for compatibility.
pub fn create_session(model: &str) -> Result<(String, PathBuf)> {
    let id = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let db = get_db()?;
    db.create_session(&id, model)?;
    
    // Return a dummy path for API compatibility
    let dummy_path = PathBuf::from(format!("sqlite://{id}"));
    Ok((id, dummy_path))
}

/// Append a message to a session.
/// The session_id is extracted from the path's file stem for compatibility.
pub fn append_message(path: &std::path::Path, message: &Message) -> Result<()> {
    let session_id = extract_session_id(path);
    let db = get_db()?;
    db.append_message(&session_id, message)?;
    Ok(())
}

/// Load all messages from a session.
pub fn load_session(path: &std::path::Path) -> Result<(SessionMeta, Vec<Message>)> {
    let session_id = extract_session_id(path);
    let db = get_db()?;
    
    let session_info = db.get_session(&session_id)?
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;
    
    let messages = db.get_messages(&session_id)?;
    
    // Convert SessionInfo to SessionMeta for compatibility
    let meta = SessionMeta {
        id: session_info.id,
        cwd: String::new(), // Not tracked in SQLite version
        model: session_info.model,
        created_at: session_info.created_at.parse().unwrap_or_else(|_| chrono::Utc::now()),
        updated_at: session_info.last_active.parse().unwrap_or_else(|_| chrono::Utc::now()),
    };
    
    Ok((meta, messages))
}

/// List available sessions, most recent first.
pub fn list_sessions() -> Result<Vec<(String, PathBuf)>> {
    let db = get_db()?;
    let sessions = db.list_sessions()?;
    
    let result: Vec<(String, PathBuf)> = sessions
        .into_iter()
        .map(|s| {
            let dummy_path = PathBuf::from(format!("sqlite://{}", s.id));
            (s.id, dummy_path)
        })
        .collect();
    
    Ok(result)
}

/// Extract session ID from path (file stem for file paths, or after "sqlite://" for SQLite paths).
fn extract_session_id(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "default".to_string())
}

/// Update session statistics (message count, token count).
pub fn update_session_stats(session_id: &str, message_count: usize, token_count: usize) -> Result<()> {
    let db = get_db()?;
    db.update_session_stats(session_id, message_count, token_count)?;
    Ok(())
}

/// Search sessions by content.
pub fn search_sessions(query: &str) -> Result<Vec<SessionInfo>> {
    let db = get_db()?;
    db.search_sessions(query)
}

/// Session metadata (kept for API compatibility).
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub cwd: String,
    pub model: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}
