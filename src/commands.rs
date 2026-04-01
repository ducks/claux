use anyhow::Result;

use crate::query::Engine;
use crate::session;

pub enum CommandResult {
    /// Print text to the user
    Text(String),
    /// Exit the REPL
    Exit,
    /// Async command that needs engine access (handled by caller)
    Async(AsyncCommand),
}

pub enum AsyncCommand {
    Compact,
    Resume(Option<String>),
    Model(Option<String>),
}

/// Parse a slash command. Returns None if input isn't a command.
pub fn parse_command(input: &str) -> Option<CommandResult> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let (cmd, args) = match trimmed.split_once(' ') {
        Some((c, a)) => (c, a.trim()),
        None => (trimmed, ""),
    };

    match cmd {
        "/help" => Some(CommandResult::Text(help_text())),
        "/exit" | "/quit" => Some(CommandResult::Exit),
        "/clear" => Some(CommandResult::Text("\x1b[2J\x1b[H".to_string())),
        "/compact" => Some(CommandResult::Async(AsyncCommand::Compact)),
        "/resume" => {
            let id = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            Some(CommandResult::Async(AsyncCommand::Resume(id)))
        }
        "/model" => {
            let model = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            Some(CommandResult::Async(AsyncCommand::Model(model)))
        }
        "/cost" => Some(CommandResult::Text("__cost__".to_string())),
        _ => Some(CommandResult::Text(format!(
            "Unknown command: {}. Type /help for available commands.",
            cmd
        ))),
    }
}

/// Execute an async command that needs engine access.
pub async fn execute_async(cmd: AsyncCommand, engine: &mut Engine) -> Result<String> {
    match cmd {
        AsyncCommand::Compact => engine.compact().await,
        AsyncCommand::Resume(id) => execute_resume(id, engine),
        AsyncCommand::Model(new_model) => execute_model(new_model, engine),
    }
}

/// Show cost info (separate since it only needs read access).
pub fn format_cost(engine: &Engine) -> String {
    engine.cost.format_summary()
}

fn execute_resume(id: Option<String>, engine: &mut Engine) -> Result<String> {
    match id {
        Some(session_id) => {
            let sessions = session::list_sessions()?;
            let found = sessions
                .iter()
                .find(|(sid, _)| sid == &session_id || sid.starts_with(&session_id));

            match found {
                Some((_, path)) => {
                    let (meta, messages) = session::load_session(path)?;
                    engine.set_messages(messages);
                    Ok(format!(
                        "Resumed session \x1b[33m{}\x1b[0m ({}, {} messages)",
                        meta.id,
                        meta.model,
                        engine.message_count()
                    ))
                }
                None => Ok(format!("Session not found: {}", session_id)),
            }
        }
        None => {
            // List recent sessions
            let sessions = session::list_sessions()?;
            if sessions.is_empty() {
                return Ok("No sessions found.".to_string());
            }

            let mut output = String::from("Recent sessions:\n");
            for (i, (id, path)) in sessions.iter().take(10).enumerate() {
                let meta_line = match session::load_session(path) {
                    Ok((meta, msgs)) => format!(
                        "  \x1b[33m{}\x1b[0m  {}  {} msgs  {}",
                        meta.id, meta.model, msgs.len(), meta.cwd
                    ),
                    Err(_) => format!("  \x1b[33m{}\x1b[0m  (error reading)", id),
                };
                output.push_str(&meta_line);
                if i < sessions.len().min(10) - 1 {
                    output.push('\n');
                }
            }
            output.push_str("\n\nUse /resume <id> to resume a session.");
            Ok(output)
        }
    }
}

fn execute_model(new_model: Option<String>, engine: &mut Engine) -> Result<String> {
    match new_model {
        Some(model) => {
            engine.set_model(&model);
            Ok(format!("Model set to \x1b[33m{}\x1b[0m", model))
        }
        None => Ok(format!(
            "Current model: \x1b[33m{}\x1b[0m\n\n\
             Available:\n  \
             claude-opus-4-20250514\n  \
             claude-sonnet-4-20250514\n  \
             claude-haiku-4-5-20251001\n\n\
             Use /model <name> to switch.",
            engine.model()
        )),
    }
}

fn help_text() -> String {
    r#"Available commands:
  /help           Show this help
  /cost           Show token usage and cost
  /compact        Summarize conversation to free context
  /model [name]   Show or switch model
  /resume [id]    List or resume past sessions
  /clear          Clear screen
  /exit           Exit claude-rs

Keyboard:
  Ctrl+C    Cancel current request
  Ctrl+D    Exit"#
        .to_string()
}
