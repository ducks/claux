//! SQLite database for session storage.
//!
//! Provides persistent storage for chat sessions with support for:
//! - Fast random access to messages
//! - Session metadata (token count, last active, etc.)
//! - Querying and searching sessions

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::api::Message;

/// Database wrapper with connection pooling.
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

impl Db {
    /// Open or create the database at the given path.
    pub fn open(path: &PathBuf) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )
        .context("Failed to open database")?;

        // Enable WAL mode for better concurrent performance (ignore result)
        let _ = conn.execute("PRAGMA journal_mode = WAL", []);

        // Create tables if they don't exist
        Self::init_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Initialize the database schema.
    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                name TEXT DEFAULT '',
                project TEXT DEFAULT 'uncategorized',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0
            )",
            [],
        )?;

        // Migrations for existing databases (duplicate column errors are expected and ignored)
        match conn.execute("ALTER TABLE sessions ADD COLUMN name TEXT DEFAULT ''", []) {
            Ok(_) => tracing::info!("Migration: added 'name' column to sessions"),
            Err(e) if e.to_string().contains("duplicate column") => {}
            Err(e) => tracing::warn!("Migration failed (name column): {e}"),
        }
        match conn.execute(
            "ALTER TABLE sessions ADD COLUMN project TEXT DEFAULT 'uncategorized'",
            [],
        ) {
            Ok(_) => tracing::info!("Migration: added 'project' column to sessions"),
            Err(e) if e.to_string().contains("duplicate column") => {}
            Err(e) => tracing::warn!("Migration failed (project column): {e}"),
        }

        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Create indexes for fast queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active)",
            [],
        )?;

        Ok(())
    }

    /// Create a new session.
    pub fn create_session(
        &self,
        id: &str,
        model: &str,
        name: Option<&str>,
        project: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO sessions (id, model, name, project, created_at, last_active) VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            [id, model, name.unwrap_or(""), project.unwrap_or("uncategorized")],
        )?;
        Ok(())
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: &str) -> Result<Option<SessionInfo>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, model, name, project, created_at, last_active, message_count, token_count
             FROM sessions WHERE id = ?1",
        )?;

        let session = stmt.query_row([id], |row| {
            Ok(SessionInfo {
                id: row.get(0)?,
                model: row.get(1)?,
                name: row.get(2)?,
                project: row
                    .get::<_, Option<String>>(3)?
                    .unwrap_or_else(|| "uncategorized".to_string()),
                created_at: row.get(4)?,
                last_active: row.get(5)?,
                message_count: row.get(6)?,
                token_count: row.get(7)?,
            })
        });

        match session {
            Ok(s) => Ok(Some(s)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all sessions, ordered by last active.
    pub fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, model, name, project, created_at, last_active, message_count, token_count
             FROM sessions ORDER BY last_active DESC",
        )?;

        let sessions = stmt.query_map([], |row| {
            Ok(SessionInfo {
                id: row.get(0)?,
                model: row.get(1)?,
                name: row.get(2)?,
                project: row
                    .get::<_, Option<String>>(3)?
                    .unwrap_or_else(|| "uncategorized".to_string()),
                created_at: row.get(4)?,
                last_active: row.get(5)?,
                message_count: row.get(6)?,
                token_count: row.get(7)?,
            })
        })?;

        Ok(sessions.collect::<Result<Vec<_>, _>>()?)
    }

    /// Append a message to a session.
    pub fn append_message(&self, session_id: &str, message: &Message) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        // Serialize content to JSON
        let content_json = serde_json::to_string(&message.content)?;

        // Insert the message
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) 
             VALUES (?1, ?2, ?3, CURRENT_TIMESTAMP)",
            [session_id, &message.role, &content_json],
        )?;

        // Update session metadata
        conn.execute(
            "UPDATE sessions SET last_active = CURRENT_TIMESTAMP, 
             message_count = message_count + 1 
             WHERE id = ?1",
            [session_id],
        )?;

        Ok(())
    }

    /// Get all messages for a session.
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT role, content FROM messages WHERE session_id = ?1 ORDER BY created_at ASC",
        )?;

        let messages = stmt.query_map([session_id], |row| {
            let role: String = row.get(0)?;
            let content_json: String = row.get(1)?;
            let content: crate::api::types::MessageContent =
                serde_json::from_str(&content_json).map_err(|_| rusqlite::Error::InvalidQuery)?;
            Ok(Message { role, content })
        })?;

        Ok(messages.collect::<Result<Vec<_>, _>>()?)
    }

    /// Get the last N messages for a session.
    pub fn get_last_messages(&self, session_id: &str, limit: usize) -> Result<Vec<Message>> {
        let conn = self.conn.lock().unwrap();
        let limit_str = limit.to_string();
        let mut stmt = conn.prepare(
            "SELECT role, content FROM messages WHERE session_id = ?1 
             ORDER BY created_at DESC LIMIT ?2",
        )?;

        let messages = stmt.query_map((session_id, limit_str.as_str()), |row| {
            let role: String = row.get(0)?;
            let content_json: String = row.get(1)?;
            let content: crate::api::types::MessageContent =
                serde_json::from_str(&content_json).map_err(|_| rusqlite::Error::InvalidQuery)?;
            Ok(Message { role, content })
        })?;

        let mut msgs: Vec<Message> = messages.collect::<Result<Vec<_>, _>>()?;
        msgs.reverse(); // Reverse to get chronological order
        Ok(msgs)
    }

    /// Update message count and token count for a session.
    pub fn update_session_stats(
        &self,
        session_id: &str,
        message_count: usize,
        token_count: usize,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sessions SET message_count = ?1, token_count = ?2, last_active = CURRENT_TIMESTAMP 
             WHERE id = ?3",
            (message_count as i64, token_count as i64, session_id),
        )?;
        Ok(())
    }

    /// Delete a session and all its messages.
    pub fn delete_session(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM sessions WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Search sessions by content.
    pub fn search_sessions(&self, query: &str) -> Result<Vec<SessionInfo>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT s.id, s.model, s.name, s.project, s.created_at, s.last_active, s.message_count, s.token_count
             FROM sessions s
             JOIN messages m ON s.id = m.session_id
             WHERE m.content LIKE ?1
             ORDER BY s.last_active DESC"
        )?;

        let sessions = stmt.query_map([format!("%{query}%")], |row| {
            Ok(SessionInfo {
                id: row.get(0)?,
                model: row.get(1)?,
                name: row.get(2)?,
                project: row
                    .get::<_, Option<String>>(3)?
                    .unwrap_or_else(|| "uncategorized".to_string()),
                created_at: row.get(4)?,
                last_active: row.get(5)?,
                message_count: row.get(6)?,
                token_count: row.get(7)?,
            })
        })?;

        Ok(sessions.collect::<Result<Vec<_>, _>>()?)
    }
}

/// Session metadata.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub model: String,
    pub name: Option<String>,
    pub project: String,
    pub created_at: String,
    pub last_active: String,
    pub message_count: i64,
    pub token_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_get_session() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = Db::open(&db_path).unwrap();

        db.create_session("test-123", "claude-sonnet", None, None)
            .unwrap();

        let session = db.get_session("test-123").unwrap();
        assert!(session.is_some());
        let s = session.unwrap();
        assert_eq!(s.id, "test-123");
        assert_eq!(s.model, "claude-sonnet");
    }

    #[test]
    fn test_append_and_get_messages() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = Db::open(&db_path).unwrap();

        db.create_session("test-456", "claude-sonnet", None, None)
            .unwrap();

        let msg1 = Message {
            role: "user".to_string(),
            content: crate::api::types::MessageContent::Text("Hello".to_string()),
        };
        let msg2 = Message {
            role: "assistant".to_string(),
            content: crate::api::types::MessageContent::Text("Hi there!".to_string()),
        };

        db.append_message("test-456", &msg1).unwrap();
        db.append_message("test-456", &msg2).unwrap();

        let messages = db.get_messages("test-456").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
    }

    #[test]
    fn test_list_sessions() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = Db::open(&db_path).unwrap();

        db.create_session("session-1", "model-a", None, None)
            .unwrap();
        db.create_session("session-2", "model-b", None, None)
            .unwrap();

        let sessions = db.list_sessions().unwrap();
        assert_eq!(sessions.len(), 2);
    }
}
