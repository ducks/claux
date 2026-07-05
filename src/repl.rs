use anyhow::Result;
use std::io::{stdout, Write};
use tokio::sync::mpsc;

use crate::commands::{self, CommandResult};
use crate::config::{Config, HookTrigger};
use crate::context;
use crate::permissions::PermissionResponse;
use crate::plugin::PluginRegistry;
use crate::query::{Engine, StreamEvent};
use crate::session;
use crate::utils::diff::colorize_diff;

/// Run the interactive REPL.
///
/// A single reader thread owns stdin and forwards whole lines over a
/// channel. Every consumer (the prompt, permission questions, mid-turn
/// steering) reads from that channel, so there is never more than one
/// reader racing for stdin. Lines typed while a turn is running are queued
/// as steering messages and delivered to the model between tool rounds;
/// lines typed after the turn ends become the next prompt.
pub async fn run(mut engine: Engine, config: &Config, plugins: &PluginRegistry) -> Result<()> {
    // Build system prompt
    let system_prompt = context::build_system_prompt_for_model(
        engine.model(),
        Some(plugins),
        &HookTrigger::OnContextBuild,
        config.is_anthropic(),
    )
    .await?;
    engine.set_system_prompt(system_prompt);

    // Create session
    let (_session_id, session_path) = session::create_session(engine.model())?;

    // Persistent stdin reader. Exits on EOF (Ctrl+D) or when the receiver
    // is dropped; the closed channel is the REPL's exit signal.
    let (stdin_tx, mut stdin_rx) = mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        loop {
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(0) | Err(_) => break, // EOF
                Ok(_) => {
                    if stdin_tx.send(line).is_err() {
                        break;
                    }
                }
            }
        }
    });

    println!("\x1b[1;36mclaux\x1b[0m v{}", env!("CARGO_PKG_VERSION"));
    println!("Model: \x1b[33m{}\x1b[0m", engine.model());
    println!("Type /help for commands, Ctrl+D (or double Ctrl+C) to exit.");
    println!("\x1b[2mWhile claux is working, type a message and press Enter to steer it.\x1b[0m\n");

    // Double-press guard: a single Ctrl+C warns instead of killing the app.
    // Registering the tokio handler also disables the default SIGINT kill.
    let mut ctrl_c = crate::utils::CtrlCArm::default();

    loop {
        // Read user input
        print!("\x1b[1;34m>\x1b[0m ");
        stdout().flush()?;
        let input = loop {
            tokio::select! {
                line = stdin_rx.recv() => break line,
                _ = tokio::signal::ctrl_c() => {
                    if ctrl_c.press() {
                        break None;
                    }
                    println!("\n  \x1b[2m(press Ctrl+C again to exit)\x1b[0m");
                    print!("\x1b[1;34m>\x1b[0m ");
                    stdout().flush()?;
                }
            }
        };
        let Some(input) = input else {
            break; // Ctrl+D or confirmed double Ctrl+C
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
                CommandResult::Text(text) => println!("{text}"),
                CommandResult::Exit => break,
                CommandResult::Async(async_cmd) => {
                    match commands::execute_async(async_cmd, &mut engine).await {
                        Ok(output) => println!("{output}"),
                        Err(e) => eprintln!("\x1b[31mError: {e}\x1b[0m"),
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

        let steering = engine.steering_queue();
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(256);
        let model_name = engine.model().to_string();

        // Show thinking indicator while waiting for first response
        print!("\n  \x1b[2mthinking...\x1b[0m");
        let _ = stdout().flush();

        // Drive the turn: the submit future, its stream events, and stdin
        // all race in one select loop. Typed lines become steering messages
        // while the turn runs; permission prompts consume the next line.
        let mut submit_result: Option<Result<()>> = None;
        let mut exit_app = false;
        {
            let submit_fut = engine.submit_streaming(trimmed, tx);
            tokio::pin!(submit_fut);

            let mut in_tool = false;
            let mut first_text = true;

            loop {
                tokio::select! {
                    res = &mut submit_fut, if submit_result.is_none() => {
                        submit_result = Some(res);
                    }
                    _ = tokio::signal::ctrl_c() => {
                        if ctrl_c.press() {
                            exit_app = true;
                            break;
                        }
                        println!("\n  \x1b[2m(press Ctrl+C again to exit claux; type a message + Enter to steer it instead)\x1b[0m");
                    }
                    event = rx.recv() => {
                        let Some(event) = event else {
                            break; // tx dropped: turn is over and events drained
                        };
                        match event {
                            StreamEvent::Text(t) => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                // Clear thinking indicator on first text and show model
                                if first_text {
                                    print!("\r\x1b[2m● {model_name} \x1b[0m");
                                    let _ = stdout().flush();
                                    first_text = false;
                                }
                                print!("{t}");
                                let _ = stdout().flush();
                            }
                            StreamEvent::ToolStart { name, summary, .. } => {
                                print!("\n  \x1b[2m[{name}]\x1b[0m {summary} ");
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
                                print_permission_prompt(&tool_name, &summary);
                                let line = stdin_rx.recv().await.unwrap_or_default();
                                let _ =
                                    respond.send(parse_permission_response(&tool_name, &summary, &line));
                            }
                            StreamEvent::PermissionRequestWithDiff {
                                tool_name,
                                summary,
                                diff,
                                respond,
                            } => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                print_permission_prompt_with_diff(&tool_name, &summary, &diff);
                                let line = stdin_rx.recv().await.unwrap_or_default();
                                let _ =
                                    respond.send(parse_permission_response(&tool_name, &summary, &line));
                            }
                            StreamEvent::Error(e) => {
                                eprintln!("\n\x1b[31mError: {e}\x1b[0m");
                            }
                            StreamEvent::Done => {
                                println!("\n");
                            }
                        }
                    }
                    Some(line) = stdin_rx.recv(), if submit_result.is_none() => {
                        let line = line.trim();
                        if !line.is_empty() {
                            steering
                                .lock()
                                .expect("steering queue poisoned")
                                .push_back(line.to_string());
                            println!("\n  \x1b[2m↳ queued for the model: {line}\x1b[0m");
                        }
                    }
                }
            }
        }

        if let Some(Err(e)) = submit_result {
            eprintln!("\n\x1b[31mError: {e}\x1b[0m\n");
        }

        // Save assistant response
        if let Some(last) = engine.messages().last() {
            let _ = session::append_message(&session_path, last);
        }

        if exit_app {
            println!("\n\x1b[2mTurn abandoned; exiting.\x1b[0m");
            break;
        }
    }

    println!("\n{}", engine.cost.format_summary());
    println!("Goodbye!");
    Ok(())
}

/// Print the permission question for a tool. The answer is read from the
/// stdin channel by the caller.
fn print_permission_prompt(tool_name: &str, summary: &str) {
    if tool_name == "Bash" {
        print!(
            "\n  \x1b[33m⚡ {summary}\x1b[0m  \x1b[2m(y)es / (n)o / (a)lways this command\x1b[0m "
        );
    } else {
        print!("\n  \x1b[33m⚡ {summary}\x1b[0m  \x1b[2m(y)es / (n)o / (a)lways\x1b[0m ");
    }
    let _ = stdout().flush();
}

/// Print the permission question with a diff preview.
fn print_permission_prompt_with_diff(tool_name: &str, summary: &str, diff: &str) {
    if tool_name == "Bash" {
        println!(
            "\n  \x1b[33m⚡ {summary}\x1b[0m  \x1b[2m(y)es / (n)o / (a)lways this command\x1b[0m"
        );
    } else {
        println!("\n  \x1b[33m⚡ {summary}\x1b[0m  \x1b[2m(y)es / (n)o / (a)lways\x1b[0m");
    }
    println!("\n  \x1b[2m--- Diff Preview ---\x1b[0m");

    let colored_diff = colorize_diff(diff);
    for line in colored_diff.lines() {
        println!("  {line}");
    }

    println!("  \x1b[2m--- End Diff ---\x1b[0m\n");

    print!("  Allow? ");
    let _ = stdout().flush();
}

/// Interpret a line of input as an answer to a permission prompt.
fn parse_permission_response(tool_name: &str, summary: &str, line: &str) -> PermissionResponse {
    let trimmed = line.trim().to_lowercase();

    // For Bash, "always" is command-specific: extract the command from the
    // summary (format: "bash: <command>")
    if tool_name == "Bash" && (trimmed == "a" || trimmed == "always") {
        if let Some(cmd) = summary.strip_prefix("bash: ") {
            return PermissionResponse::AlwaysAllowCommand(cmd.trim().to_string());
        }
        return PermissionResponse::AlwaysAllowCommand(summary.to_string());
    }

    match trimmed.as_str() {
        "y" | "yes" | "" => PermissionResponse::Allow,
        "a" | "always" => PermissionResponse::AlwaysAllow,
        _ => PermissionResponse::Deny,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_yes_variants() {
        for line in ["y\n", "yes\n", "\n", "Y\n"] {
            assert_eq!(
                parse_permission_response("Write", "write: /tmp/x", line),
                PermissionResponse::Allow
            );
        }
    }

    #[test]
    fn permission_deny_on_anything_else() {
        assert_eq!(
            parse_permission_response("Write", "write: /tmp/x", "n\n"),
            PermissionResponse::Deny
        );
        assert_eq!(
            parse_permission_response("Write", "write: /tmp/x", "wait, not that file\n"),
            PermissionResponse::Deny
        );
    }

    #[test]
    fn permission_always_is_command_specific_for_bash() {
        assert_eq!(
            parse_permission_response("Bash", "bash: cargo test", "a\n"),
            PermissionResponse::AlwaysAllowCommand("cargo test".to_string())
        );
    }

    #[test]
    fn permission_always_for_non_bash() {
        assert_eq!(
            parse_permission_response("Write", "write: /tmp/x", "a\n"),
            PermissionResponse::AlwaysAllow
        );
    }
}
