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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_command_returns_none() {
        assert!(parse_command("hello world").is_none());
    }

    #[test]
    fn help_returns_text() {
        let result = parse_command("/help");
        assert!(matches!(result, Some(CommandResult::Text(_))));
    }

    #[test]
    fn exit_returns_exit() {
        assert!(matches!(parse_command("/exit"), Some(CommandResult::Exit)));
        assert!(matches!(parse_command("/quit"), Some(CommandResult::Exit)));
    }

    #[test]
    fn cost_returns_sentinel() {
        if let Some(CommandResult::Text(text)) = parse_command("/cost") {
            assert_eq!(text, "__cost__");
        } else {
            panic!("expected Text");
        }
    }

    #[test]
    fn compact_returns_async() {
        assert!(matches!(
            parse_command("/compact"),
            Some(CommandResult::Async(AsyncCommand::Compact))
        ));
    }

    #[test]
    fn model_no_args_returns_none_model() {
        if let Some(CommandResult::Async(AsyncCommand::Model(m))) = parse_command("/model") {
            assert!(m.is_none());
        } else {
            panic!("expected Model(None)");
        }
    }

    #[test]
    fn model_with_args() {
        if let Some(CommandResult::Async(AsyncCommand::Model(Some(m)))) =
            parse_command("/model claude-opus-4-20250514")
        {
            assert_eq!(m, "claude-opus-4-20250514");
        } else {
            panic!("expected Model(Some)");
        }
    }

    #[test]
    fn resume_no_args() {
        if let Some(CommandResult::Async(AsyncCommand::Resume(id))) = parse_command("/resume") {
            assert!(id.is_none());
        } else {
            panic!("expected Resume(None)");
        }
    }

    #[test]
    fn resume_with_id() {
        if let Some(CommandResult::Async(AsyncCommand::Resume(Some(id)))) =
            parse_command("/resume 20260401-143022")
        {
            assert_eq!(id, "20260401-143022");
        } else {
            panic!("expected Resume(Some)");
        }
    }

    #[test]
    fn unknown_command_returns_error_text() {
        if let Some(CommandResult::Text(text)) = parse_command("/bogus") {
            assert!(text.contains("Unknown command"));
        } else {
            panic!("expected Text");
        }
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
