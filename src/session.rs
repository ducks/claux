//! Session storage using SQLite.
//!
//! Provides persistent storage for chat sessions with fast random access,
//! metadata tracking, and querying capabilities.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::api::types::Message;
use crate::config::{ModelBinding, ResolvedModel};
use crate::db::Db;

/// Get the database path.
pub(crate) fn db_path() -> Result<PathBuf> {
    let base =
        dirs::data_local_dir().ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
    let dir = base.join("claux");
    prepare_storage_dir(&dir)?;
    Ok(dir.join("sessions.db"))
}

fn prepare_storage_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = std::fs::metadata(dir)?.permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(dir, permissions)?;
    }

    Ok(())
}

/// Get the database instance (lazy initialization).
fn get_db() -> Result<Db> {
    let path = db_path()?;
    Db::open(&path).context("Failed to open session database")
}

pub fn create_session_with_model(resolved: &ResolvedModel) -> Result<(String, PathBuf)> {
    let id = new_session_id();
    let db = get_db()?;
    db.create_session_with_binding(&id, &resolved.binding, None, None)?;
    Ok((id.clone(), PathBuf::from(format!("sqlite://{id}"))))
}

/// Generate a readable, collision-resistant session identifier.
pub(crate) fn new_session_id() -> String {
    format!(
        "{}-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        uuid::Uuid::new_v4().simple()
    )
}

/// Persist the full message list for a session, replacing what was stored.
///
/// Called after each turn with the engine's message list. Snapshotting
/// (rather than appending) keeps the store faithful to the engine even
/// when compaction rewrites history or steering inserts messages mid-turn.
pub fn save_messages(path: &std::path::Path, messages: &[Message]) -> Result<()> {
    let session_id = extract_session_id(path);
    let db = get_db()?;
    db.replace_messages(&session_id, messages)?;
    Ok(())
}

pub fn save_model_binding(path: &std::path::Path, binding: &ModelBinding) -> Result<()> {
    let session_id = extract_session_id(path);
    get_db()?.update_session_binding(&session_id, binding)
}

/// Load all messages from a session.
pub fn load_session(path: &std::path::Path) -> Result<(SessionMeta, Vec<Message>)> {
    let session_id = extract_session_id(path);
    let db = get_db()?;

    let session_info = db
        .get_session(&session_id)?
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    let messages = repair_history(db.get_messages(&session_id)?);

    // Convert SessionInfo to SessionMeta for compatibility
    let meta = SessionMeta {
        id: session_info.id,
        cwd: String::new(), // Not tracked in SQLite version
        model: session_info.model,
        model_binding: session_info.model_binding,
    };

    Ok((meta, messages))
}

/// Find a session by exact id or unique-enough prefix, most recent first.
pub fn find_session(prefix: &str) -> Result<Option<(String, PathBuf)>> {
    Ok(list_sessions()?
        .into_iter()
        .find(|(sid, _)| sid == prefix || sid.starts_with(prefix)))
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

/// Session metadata (kept for API compatibility).
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub cwd: String,
    pub model: String,
    pub model_binding: Option<ModelBinding>,
}

/// Make a loaded history API-valid: every tool_use must be followed by a
/// matching tool_result, and no tool_result may reference a tool_use that
/// isn't present.
///
/// Histories can violate this two ways: sessions saved by older claux
/// versions (which stored only the final message of each turn), and
/// sessions whose last turn was cut off mid-tools by a crash or kill.
/// The Anthropic API rejects such conversations outright, so resume must
/// repair them: missing results are synthesized, orphaned results are
/// dropped.
pub fn repair_history(messages: Vec<Message>) -> Vec<Message> {
    use crate::api::types::{ContentBlock, MessageContent};

    const LOST_RESULT: &str = "Tool result not saved before the session ended.";

    let synthetic = |id: &str| ContentBlock::ToolResult {
        tool_use_id: id.to_string(),
        content: LOST_RESULT.to_string(),
        is_error: Some(true),
    };

    let mut repaired: Vec<Message> = Vec::with_capacity(messages.len());
    // tool_use ids from the most recent assistant message, awaiting results
    let mut pending: Vec<String> = Vec::new();

    for msg in messages {
        let is_result_message = matches!(
            &msg.content,
            MessageContent::Blocks(blocks)
                if blocks.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
        );

        if is_result_message {
            let MessageContent::Blocks(blocks) = &msg.content else {
                unreachable!("is_result_message implies Blocks");
            };
            // Keep results that answer a pending tool_use; drop orphans.
            let mut kept: Vec<ContentBlock> = blocks
                .iter()
                .filter(|b| match b {
                    ContentBlock::ToolResult { tool_use_id, .. } => pending.contains(tool_use_id),
                    _ => true,
                })
                .cloned()
                .collect();
            for block in &kept {
                if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                    pending.retain(|id| id != tool_use_id);
                }
            }
            // Results lost for the remaining pending ids: synthesize them
            // into this same message so pairing stays adjacent.
            for id in pending.drain(..) {
                kept.push(synthetic(&id));
            }
            if kept.is_empty() {
                continue; // message was nothing but orphans
            }
            repaired.push(Message {
                role: msg.role.clone(),
                content: MessageContent::Blocks(kept),
            });
            continue;
        }

        // Any other message while results are pending means those results
        // were never saved; synthesize them before continuing.
        if !pending.is_empty() {
            repaired.push(Message::tool_results(
                pending.drain(..).map(|id| synthetic(&id)).collect(),
            ));
        }

        if let MessageContent::Blocks(blocks) = &msg.content {
            for block in blocks {
                if let ContentBlock::ToolUse { id, .. } = block {
                    pending.push(id.clone());
                }
            }
        }
        repaired.push(msg);
    }

    // History ends with unanswered tool_uses (killed mid-turn)
    if !pending.is_empty() {
        repaired.push(Message::tool_results(
            pending.drain(..).map(|id| synthetic(&id)).collect(),
        ));
    }

    repaired
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{ContentBlock, MessageContent};

    #[cfg(unix)]
    #[test]
    fn storage_directory_is_private_and_repairs_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("claux");
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();

        prepare_storage_dir(&dir).unwrap();

        assert_eq!(
            std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    #[test]
    fn session_ids_do_not_collide_within_the_same_second() {
        let ids: std::collections::HashSet<String> = (0..1_000).map(|_| new_session_id()).collect();
        assert_eq!(ids.len(), 1_000);
        assert!(ids.iter().all(|id| id.len() == 48));
    }

    fn tool_use_msg(id: &str) -> Message {
        Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: id.to_string(),
            name: "Bash".to_string(),
            input: serde_json::json!({"command": "true"}),
        }])
    }

    fn tool_result_msg(id: &str) -> Message {
        Message::tool_results(vec![ContentBlock::ToolResult {
            tool_use_id: id.to_string(),
            content: "ok".to_string(),
            is_error: None,
        }])
    }

    fn assert_valid_pairing(messages: &[Message]) {
        let mut seen = std::collections::HashSet::new();
        let mut pending: Vec<String> = Vec::new();
        for msg in messages {
            if let MessageContent::Blocks(blocks) = &msg.content {
                let has_results = blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
                if !has_results && !pending.is_empty() {
                    panic!("tool_uses {pending:?} not answered by the next message");
                }
                for block in blocks {
                    match block {
                        ContentBlock::ToolUse { id, .. } => {
                            seen.insert(id.clone());
                            pending.push(id.clone());
                        }
                        ContentBlock::ToolResult { tool_use_id, .. } => {
                            assert!(seen.contains(tool_use_id), "orphan result {tool_use_id}");
                            pending.retain(|p| p != tool_use_id);
                        }
                        ContentBlock::Text { .. } => {}
                    }
                }
            } else if !pending.is_empty() {
                panic!("tool_uses {pending:?} not answered by the next message");
            }
        }
        assert!(
            pending.is_empty(),
            "history ends with unanswered {pending:?}"
        );
    }

    #[test]
    fn repair_leaves_valid_history_untouched() {
        let history = vec![
            Message::user("hi"),
            tool_use_msg("tu_1"),
            tool_result_msg("tu_1"),
            Message::assistant_text("done"),
        ];
        let repaired = repair_history(history.clone());
        assert_eq!(repaired.len(), history.len());
        assert_valid_pairing(&repaired);
    }

    #[test]
    fn repair_synthesizes_result_for_trailing_tool_use() {
        // Session killed mid-turn: history ends on an unanswered tool_use
        let history = vec![Message::user("go"), tool_use_msg("tu_1")];
        let repaired = repair_history(history);
        assert_eq!(repaired.len(), 3);
        assert_valid_pairing(&repaired);
        let MessageContent::Blocks(blocks) = &repaired[2].content else {
            panic!("expected synthetic results message");
        };
        assert!(matches!(
            &blocks[0],
            ContentBlock::ToolResult {
                is_error: Some(true),
                ..
            }
        ));
    }

    #[test]
    fn repair_synthesizes_result_before_next_message() {
        // Result lost in the middle of a conversation
        let history = vec![
            Message::user("go"),
            tool_use_msg("tu_1"),
            Message::assistant_text("moving on"),
            Message::user("ok"),
        ];
        let repaired = repair_history(history);
        assert_eq!(repaired.len(), 5);
        assert_valid_pairing(&repaired);
    }

    #[test]
    fn repair_drops_orphan_results() {
        // Legacy lossy save: a result message whose tool_use was never stored
        let history = vec![
            Message::user("go"),
            tool_result_msg("tu_ghost"),
            Message::assistant_text("done"),
        ];
        let repaired = repair_history(history);
        assert_eq!(repaired.len(), 2, "orphan-only message must be dropped");
        assert_valid_pairing(&repaired);
    }

    #[test]
    fn repair_fills_partial_results() {
        // Two tool_uses, only one result saved
        let history = vec![
            Message::user("go"),
            Message::assistant_blocks(vec![
                ContentBlock::ToolUse {
                    id: "tu_1".to_string(),
                    name: "Read".to_string(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolUse {
                    id: "tu_2".to_string(),
                    name: "Read".to_string(),
                    input: serde_json::json!({}),
                },
            ]),
            tool_result_msg("tu_1"),
            Message::assistant_text("done"),
        ];
        let repaired = repair_history(history);
        assert_valid_pairing(&repaired);
    }
}
