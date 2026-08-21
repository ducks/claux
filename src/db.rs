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
use crate::config::ModelBinding;

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

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = std::fs::metadata(path)?.permissions();
            permissions.set_mode(0o600);
            std::fs::set_permissions(path, permissions)?;
        }

        // Enable WAL mode for better concurrent performance (ignore result)
        let _ = conn.execute("PRAGMA journal_mode = WAL", []);

        // SQLite does not enforce declared foreign keys unless each
        // connection opts in. Session deletion relies on ON DELETE CASCADE.
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

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
                token_count INTEGER DEFAULT 0,
                model_profile TEXT,
                model_binding TEXT
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
        match conn.execute("ALTER TABLE sessions ADD COLUMN model_profile TEXT", []) {
            Ok(_) => tracing::info!("Migration: added 'model_profile' column to sessions"),
            Err(e) if e.to_string().contains("duplicate column") => {}
            Err(e) => tracing::warn!("Migration failed (model_profile column): {e}"),
        }
        match conn.execute("ALTER TABLE sessions ADD COLUMN model_binding TEXT", []) {
            Ok(_) => tracing::info!("Migration: added 'model_binding' column to sessions"),
            Err(e) if e.to_string().contains("duplicate column") => {}
            Err(e) => tracing::warn!("Migration failed (model_binding column): {e}"),
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

        // Older Claux versions left foreign-key enforcement disabled, so
        // remove any unreachable transcript rows they may have accumulated.
        conn.execute(
            "DELETE FROM messages
             WHERE NOT EXISTS (
                 SELECT 1 FROM sessions WHERE sessions.id = messages.session_id
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
    #[cfg(test)]
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

    /// Create a session with an immutable, credential-free provider/model snapshot.
    pub fn create_session_with_binding(
        &self,
        id: &str,
        binding: &ModelBinding,
        name: Option<&str>,
        project: Option<&str>,
    ) -> Result<()> {
        let binding_json = serde_json::to_string(binding)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO sessions
             (id, model, model_profile, model_binding, name, project, created_at, last_active)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (
                id,
                &binding.model,
                &binding.profile,
                binding_json,
                name.unwrap_or(""),
                project.unwrap_or("uncategorized"),
            ),
        )?;
        Ok(())
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: &str) -> Result<Option<SessionInfo>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, model, name, project, message_count, model_binding
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
                message_count: row.get(4)?,
                model_binding: parse_model_binding(row.get(5)?),
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
            "SELECT id, model, name, project, message_count, model_binding
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
                message_count: row.get(4)?,
                model_binding: parse_model_binding(row.get(5)?),
            })
        })?;

        Ok(sessions.collect::<Result<Vec<_>, _>>()?)
    }

    /// Append a message to a session.
    #[cfg(test)]
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

    /// Replace a session's messages wholesale, in one transaction.
    ///
    /// Persistence snapshots the engine's full message list after each turn
    /// rather than appending: compaction rewrites history and steering
    /// inserts messages mid-turn, so append-only saves drift from the
    /// engine's actual state.
    pub fn replace_messages(&self, session_id: &str, messages: &[Message]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        tx.execute("DELETE FROM messages WHERE session_id = ?1", [session_id])?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO messages (session_id, role, content, created_at)
                 VALUES (?1, ?2, ?3, CURRENT_TIMESTAMP)",
            )?;
            for message in messages {
                let content_json = serde_json::to_string(&message.content)?;
                stmt.execute([session_id, &message.role, &content_json])?;
            }
        }
        tx.execute(
            "UPDATE sessions SET last_active = CURRENT_TIMESTAMP,
             message_count = ?1
             WHERE id = ?2",
            (messages.len() as i64, session_id),
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Persist a model change while retaining the session's provider transport.
    pub fn update_session_binding(&self, session_id: &str, binding: &ModelBinding) -> Result<()> {
        let binding_json = serde_json::to_string(binding)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sessions
             SET model = ?1, model_profile = ?2, model_binding = ?3,
                 last_active = CURRENT_TIMESTAMP
             WHERE id = ?4",
            (&binding.model, &binding.profile, binding_json, session_id),
        )?;
        Ok(())
    }

    /// Get all messages for a session.
    ///
    /// Ordered by insertion (id), not created_at: CURRENT_TIMESTAMP has
    /// one-second granularity, so a tool round inserting several messages
    /// in the same second would load back in unspecified order.
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT role, content FROM messages WHERE session_id = ?1 ORDER BY id ASC")?;

        let messages = stmt.query_map([session_id], |row| {
            let role: String = row.get(0)?;
            let content_json: String = row.get(1)?;
            let content: crate::api::types::MessageContent =
                serde_json::from_str(&content_json).map_err(|_| rusqlite::Error::InvalidQuery)?;
            Ok(Message { role, content })
        })?;

        Ok(messages.collect::<Result<Vec<_>, _>>()?)
    }

    /// Delete a session and all its messages.
    pub fn delete_session(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM sessions WHERE id = ?1", [id])?;
        Ok(())
    }
}

/// Session metadata.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub model: String,
    pub name: Option<String>,
    pub project: String,
    pub message_count: i64,
    pub model_binding: Option<ModelBinding>,
}

fn parse_model_binding(json: Option<String>) -> Option<ModelBinding> {
    json.and_then(|json| match serde_json::from_str(&json) {
        Ok(binding) => Some(binding),
        Err(error) => {
            tracing::warn!("ignoring invalid saved model binding: {error}");
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OpenAIProtocol, ProviderKind};
    use tempfile::TempDir;

    fn binding(model: &str) -> ModelBinding {
        ModelBinding {
            profile: "router-model".to_string(),
            display_name: "Router Model".to_string(),
            provider: "openrouter".to_string(),
            provider_kind: ProviderKind::Openai,
            provider_name: "openrouter".to_string(),
            model: model.to_string(),
            base_url: Some("https://openrouter.ai/api/v1".to_string()),
            protocol: OpenAIProtocol::ChatCompletions,
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            reasoning_effort: None,
            prompt_caching: false,
            allow_eof_without_finish_reason: false,
        }
    }

    #[cfg(unix)]
    #[test]
    fn database_file_is_private() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let _db = Db::open(&db_path).unwrap();

        assert_eq!(
            std::fs::metadata(&db_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

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
    fn model_binding_roundtrips_and_can_be_updated() {
        let temp_dir = TempDir::new().unwrap();
        let db = Db::open(&temp_dir.path().join("test.db")).unwrap();
        db.create_session_with_binding("bound", &binding("model-a"), None, None)
            .unwrap();

        let created = db.get_session("bound").unwrap().unwrap();
        assert_eq!(
            created.model_binding.as_ref().unwrap().profile,
            "router-model"
        );
        assert_eq!(created.model_binding.unwrap(), binding("model-a"));

        let mut updated = binding("model-b");
        updated.profile = "adhoc:model-b".to_string();
        db.update_session_binding("bound", &updated).unwrap();
        let loaded = db.get_session("bound").unwrap().unwrap();
        assert_eq!(loaded.model, "model-b");
        assert_eq!(loaded.model_binding.unwrap(), updated);
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

    fn tool_turn() -> Vec<Message> {
        use crate::api::types::{ContentBlock, MessageContent};
        vec![
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("run the tests".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "tu_1".to_string(),
                    name: "Bash".to_string(),
                    input: serde_json::json!({"command": "cargo test"}),
                }]),
            },
            Message {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_string(),
                    content: "ok".to_string(),
                    is_error: None,
                }]),
            },
            Message {
                role: "assistant".to_string(),
                content: MessageContent::Text("All green.".to_string()),
            },
        ]
    }

    #[test]
    fn test_replace_messages_roundtrips_tool_rounds_in_order() {
        let temp_dir = TempDir::new().unwrap();
        let db = Db::open(&temp_dir.path().join("test.db")).unwrap();
        db.create_session("s", "m", None, None).unwrap();

        let turn = tool_turn();
        db.replace_messages("s", &turn).unwrap();

        // All messages inserted within the same second must load back in
        // insertion order (regression: ORDER BY created_at ties).
        let loaded = db.get_messages("s").unwrap();
        assert_eq!(loaded.len(), 4);
        let roles: Vec<&str> = loaded.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "user", "assistant"]);
        assert_eq!(
            serde_json::to_string(&loaded[1].content).unwrap(),
            serde_json::to_string(&turn[1].content).unwrap(),
            "tool_use blocks must survive the round trip"
        );

        // Replacing is a snapshot, not an append
        let compacted = vec![Message {
            role: "user".to_string(),
            content: crate::api::types::MessageContent::Text("summary".to_string()),
        }];
        db.replace_messages("s", &compacted).unwrap();
        let loaded = db.get_messages("s").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(db.get_session("s").unwrap().unwrap().message_count, 1);
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

    #[test]
    fn deleting_session_cascades_to_messages() {
        let temp_dir = TempDir::new().unwrap();
        let db = Db::open(&temp_dir.path().join("test.db")).unwrap();
        db.create_session("session", "model", None, None).unwrap();
        db.append_message("session", &Message::user("private transcript"))
            .unwrap();

        db.delete_session("session").unwrap();

        let conn = db.conn.lock().unwrap();
        let message_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(message_count, 0);
    }

    #[test]
    fn opening_database_removes_legacy_orphan_messages() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "PRAGMA foreign_keys = OFF;
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    project TEXT DEFAULT 'uncategorized',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT 0
                );
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                INSERT INTO messages (session_id, role, content)
                VALUES ('missing', 'user', '\"private transcript\"');",
            )
            .unwrap();
        }

        let db = Db::open(&db_path).unwrap();

        let conn = db.conn.lock().unwrap();
        let message_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(message_count, 0);
    }
}
