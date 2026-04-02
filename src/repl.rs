use anyhow::Result;
use std::io::{BufRead, Write, stdout};
use tokio::sync::mpsc;

use crate::commands::{self, CommandResult};
use crate::config::Config;
use crate::context;
use crate::permissions::PermissionResponse;
use crate::query::{Engine, StreamEvent};
use crate::session;

/// Run the interactive REPL.
pub async fn run(mut engine: Engine, _config: &Config) -> Result<()> {
    // Build system prompt
    let system_prompt = context::build_system_prompt_for_model(engine.model()).await?;
    engine.set_system_prompt(system_prompt);

    // Create session
    let (_session_id, session_path) = session::create_session(engine.model())?;

    println!("\x1b[1;36mclaux\x1b[0m v{}", env!("CARGO_PKG_VERSION"));
    println!("Model: \x1b[33m{}\x1b[0m", engine.model());
    println!("Type /help for commands, Ctrl+D to exit.\n");

    loop {
        // Read user input
        let input = match read_input()? {
            Some(input) => input,
            None => break, // Ctrl+D
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check for slash commands
        if let Some(result) = commands::parse_command(trimmed) {
            match result {
                CommandResult::Text(ref text) if text == "__cost__" => {
                    println!("{}", commands::format_cost(&engine));
                }
                CommandResult::Text(text) => println!("{}", text),
                CommandResult::Exit => break,
                CommandResult::Async(async_cmd) => {
                    match commands::execute_async(async_cmd, &mut engine).await {
                        Ok(output) => println!("{}", output),
                        Err(e) => eprintln!("\x1b[31mError: {}\x1b[0m", e),
                    }
                }
            }
            continue;
        }

        // Save user message
        let user_msg = crate::api::Message::user(trimmed);
        let _ = session::append_message(&session_path, &user_msg);

        print!("\n\x1b[1;32m❯\x1b[0m ");
        stdout().flush()?;

        // Stream the response
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(256);

        // Spawn the display consumer
        let display_handle = tokio::spawn(async move {
            let mut in_tool = false;
            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Text(t) => {
                        if in_tool {
                            println!();
                            in_tool = false;
                        }
                        print!("{}", t);
                        let _ = stdout().flush();
                    }
                    StreamEvent::ToolStart { name, summary, .. } => {
                        print!("\n  \x1b[2m[{}]\x1b[0m {} ", name, summary);
                        let _ = stdout().flush();
                        in_tool = true;
                    }
                    StreamEvent::ToolResult { is_error, .. } => {
                        if is_error {
                            print!("\x1b[31m✗\x1b[0m");
                        } else {
                            print!("\x1b[32m✓\x1b[0m");
                        }
                        let _ = stdout().flush();
                        in_tool = false;
                    }
                    StreamEvent::PermissionRequest {
                        tool_name,
                        summary,
                        respond,
                    } => {
                        if in_tool {
                            println!();
                            in_tool = false;
                        }
                        let response = prompt_permission(&tool_name, &summary);
                        let _ = respond.send(response);
                    }
                    StreamEvent::Error(e) => {
                        eprintln!("\n\x1b[31mError: {}\x1b[0m", e);
                    }
                    StreamEvent::Done => {
                        println!("\n");
                        break;
                    }
                }
            }
        });

        // Run the query
        if let Err(e) = engine.submit_streaming(trimmed, tx).await {
            eprintln!("\n\x1b[31mError: {}\x1b[0m\n", e);
        }

        display_handle.await?;

        // Save assistant response
        if let Some(last) = engine.messages().last() {
            let _ = session::append_message(&session_path, last);
        }
    }

    println!("\n{}", engine.cost.format_summary());
    println!("Goodbye!");
    Ok(())
}

/// Prompt the user for permission to execute a tool.
fn prompt_permission(_tool_name: &str, summary: &str) -> PermissionResponse {
    print!(
        "\n  \x1b[33m⚡ {}\x1b[0m  \x1b[2m(y)es / (n)o / (a)lways\x1b[0m ",
        summary
    );
    let _ = stdout().flush();

    let mut input = String::new();
    if std::io::stdin().lock().read_line(&mut input).is_err() {
        return PermissionResponse::Deny;
    }

    match input.trim().to_lowercase().as_str() {
        "y" | "yes" | "" => PermissionResponse::Allow,
        "a" | "always" => PermissionResponse::AlwaysAllow,
        _ => PermissionResponse::Deny,
    }
}

/// Read a line of input from the user (cooked mode).
fn read_input() -> Result<Option<String>> {
    print!("\x1b[1;34m>\x1b[0m ");
    stdout().flush()?;

    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => Ok(None), // EOF (Ctrl+D)
        Ok(_) => Ok(Some(line)),
        Err(e) => Err(e.into()),
    }
}
