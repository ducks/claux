/// Slash commands. Phase 1 has the essentials.

pub enum CommandResult {
    /// Print text to the user
    Text(String),
    /// Exit the REPL
    Exit,
    /// No output
    Skip,
}

pub fn handle_command(input: &str, cost: &crate::cost::CostTracker) -> Option<CommandResult> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let (cmd, _args) = match trimmed.split_once(' ') {
        Some((c, a)) => (c, a.trim()),
        None => (trimmed, ""),
    };

    match cmd {
        "/help" => Some(CommandResult::Text(help_text())),
        "/exit" | "/quit" => Some(CommandResult::Exit),
        "/cost" => Some(CommandResult::Text(cost.format_summary())),
        "/clear" => Some(CommandResult::Text("\x1b[2J\x1b[H".to_string())),
        "/model" => Some(CommandResult::Text(format!("Current model: (use --model flag to change)"))),
        _ => Some(CommandResult::Text(format!("Unknown command: {}. Type /help for available commands.", cmd))),
    }
}

fn help_text() -> String {
    r#"Available commands:
  /help     Show this help
  /cost     Show token usage and cost
  /clear    Clear screen
  /exit     Exit claude-rs
  /model    Show current model

Keyboard:
  Ctrl+C    Cancel current request
  Ctrl+D    Exit"#
        .to_string()
}
