use anyhow::Result;
use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::commands::{self, CommandResult};
use crate::config::{Config, HookTrigger, ProviderKind, ResolvedModel};
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
pub async fn run(
    mut engine: Engine,
    config: &Config,
    plugins: Arc<PluginRegistry>,
    resumed: Option<String>,
    mut resolved_model: ResolvedModel,
) -> Result<()> {
    // Build system prompt
    let system_prompt = context::build_system_prompt_for_model(
        engine.model(),
        Some(&plugins),
        &HookTrigger::OnContextBuild,
        resolved_model.binding.provider_kind == ProviderKind::Anthropic,
        config.is_project_trusted(),
    )
    .await?;
    engine.set_system_prompt(system_prompt);

    // Continue the resumed session, or create a fresh one. Resuming used
    // to create a new session anyway, so every resumed conversation forked
    // into a duplicate and the original never grew.
    let mut session_path = match resumed {
        Some(id) => std::path::PathBuf::from(format!("sqlite://{id}")),
        None => session::create_session_with_model(&resolved_model)?.1,
    };

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

    // A resumed session drops you back into the conversation, not at a
    // blank prompt: replay the tail so you can see where you left off.
    if engine.message_count() > 0 {
        print!(
            "{}",
            replay_transcript(engine.messages(), engine.model(), 12)
        );
    }

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
        if let Some(result) = commands::parse_command(trimmed, commands::Surface::Repl) {
            match result {
                CommandResult::Text(ref text) if text == "__cost__" => {
                    println!("{}", commands::format_cost(&engine));
                }
                CommandResult::Text(ref text) if text == "__context__" => {
                    println!("{}", commands::format_context(&engine));
                }
                CommandResult::Text(text) => println!("{text}"),
                CommandResult::Exit => break,
                // /resume with an id: switch which session this REPL is
                // writing to, so the resumed conversation continues in
                // place instead of copying itself into the current session.
                CommandResult::Async(commands::AsyncCommand::Resume(Some(ref prefix))) => {
                    match session::find_session(prefix)? {
                        Some((sid, path)) => {
                            let (meta, messages) = session::load_session(&path)?;
                            resolved_model = match meta.model_binding.as_ref() {
                                Some(binding) => config.resolve_binding(binding)?,
                                None => config.resolve_model(&meta.model)?,
                            };
                            let mut resumed_engine =
                                crate::build_engine(config, &resolved_model, plugins.clone())
                                    .await?;
                            let system_prompt = context::build_system_prompt_for_model(
                                &resolved_model.binding.model,
                                Some(&plugins),
                                &HookTrigger::OnContextBuild,
                                resolved_model.binding.provider_kind == ProviderKind::Anthropic,
                                config.is_project_trusted(),
                            )
                            .await?;
                            resumed_engine.set_system_prompt(system_prompt);
                            resumed_engine.set_messages(messages);
                            engine = resumed_engine;
                            session_path = path;
                            println!(
                                "Resumed session \x1b[33m{sid}\x1b[0m ({} messages)",
                                engine.message_count()
                            );
                            print!(
                                "{}",
                                replay_transcript(engine.messages(), engine.model(), 12)
                            );
                        }
                        None => println!("Session not found: {prefix}"),
                    }
                }
                CommandResult::Async(commands::AsyncCommand::Model(selector)) => {
                    let Some(selector) = selector else {
                        println!(
                            "{}",
                            commands::format_model_choices(
                                engine.model_binding(),
                                &config.selectable_models()?,
                            )
                        );
                        continue;
                    };

                    match config.resolve_model(&selector) {
                        Ok(next_model) => {
                            let messages = engine.messages().to_vec();
                            let rebuilt = async {
                                let mut next_engine =
                                    crate::build_engine(config, &next_model, plugins.clone())
                                        .await?;
                                let system_prompt = context::build_system_prompt_for_model(
                                    &next_model.binding.model,
                                    Some(&plugins),
                                    &HookTrigger::OnContextBuild,
                                    next_model.binding.provider_kind == ProviderKind::Anthropic,
                                    config.is_project_trusted(),
                                )
                                .await?;
                                next_engine.set_system_prompt(system_prompt);
                                next_engine.set_messages(messages);
                                Ok::<_, anyhow::Error>(next_engine)
                            }
                            .await;
                            match rebuilt {
                                Ok(next_engine) => {
                                    engine = next_engine;
                                    resolved_model = next_model;
                                    session::save_model_binding(
                                        &session_path,
                                        &resolved_model.binding,
                                    )?;
                                    println!(
                                        "Model set to \x1b[33m{}\x1b[0m via {}",
                                        resolved_model.binding.display_name,
                                        resolved_model.binding.provider_name
                                    );
                                }
                                Err(error) => {
                                    eprintln!("\x1b[31mError: {error}\x1b[0m");
                                }
                            }
                        }
                        Err(error) => eprintln!("\x1b[31mError: {error}\x1b[0m"),
                    }
                }
                CommandResult::Async(async_cmd) => {
                    match commands::execute_async(async_cmd, &mut engine).await {
                        Ok(output) => println!("{output}"),
                        Err(e) => eprintln!("\x1b[31mError: {e}\x1b[0m"),
                    }
                    // Commands like /compact rewrite engine history
                    let _ = session::save_messages(&session_path, engine.messages());
                    if let Some(binding) = engine.model_binding() {
                        let _ = session::save_model_binding(&session_path, binding);
                    }
                }
            }
            continue;
        }

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
        let turn_cancel = tokio_util::sync::CancellationToken::new();
        {
            let submit_fut = engine.submit_streaming(trimmed, tx, turn_cancel.clone());
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
                        // First press interrupts the turn cleanly; the
                        // engine pairs dangling tool calls and returns.
                        turn_cancel.cancel();
                        println!("\n  \x1b[2m(interrupting... press Ctrl+C again quickly to exit claux)\x1b[0m");
                    }
                    event = rx.recv() => {
                        let Some(event) = event else {
                            break; // tx dropped: turn is over and events drained
                        };
                        match event {
                            StreamEvent::Notice(n) => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                println!("\n  \x1b[2m[{n}]\x1b[0m");
                            }
                            StreamEvent::ContextUsage(_) => {}
                            StreamEvent::Retry(n) => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                println!(
                                    "\n  \x1b[2m[discarding rejected response: {n}]\x1b[0m"
                                );
                                first_text = true;
                            }
                            StreamEvent::SteeringSent(t) => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                println!("\n  \x1b[2m↳ steering sent: {t}\x1b[0m");
                            }
                            StreamEvent::Interrupted => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                println!("\n  \x1b[33m⏹ interrupted\x1b[0m\n");
                            }
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
                                print!("{}", crate::utils::sanitize_terminal_text(&t));
                                let _ = stdout().flush();
                            }
                            StreamEvent::ToolStart { name, summary, .. } => {
                                let name = crate::utils::sanitize_terminal_text(&name);
                                let summary = crate::utils::sanitize_terminal_text(&summary);
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
                                input,
                                respond,
                            } => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                print_permission_prompt(&tool_name, &summary);
                                let line = stdin_rx.recv().await.unwrap_or_default();
                                let _ = respond.send(parse_permission_response(
                                    &tool_name, &input, &line,
                                ));
                            }
                            StreamEvent::PermissionRequestWithDiff {
                                tool_name,
                                summary,
                                diff,
                                input,
                                respond,
                            } => {
                                if in_tool {
                                    println!();
                                    in_tool = false;
                                }
                                print_permission_prompt_with_diff(&tool_name, &summary, &diff);
                                let line = stdin_rx.recv().await.unwrap_or_default();
                                let _ = respond.send(parse_permission_response(
                                    &tool_name, &input, &line,
                                ));
                            }
                            StreamEvent::Error(e) => {
                                eprintln!(
                                    "\n\x1b[31mError: {}\x1b[0m",
                                    crate::utils::sanitize_terminal_text(&e)
                                );
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

        // Snapshot the full conversation, tool rounds included. Previously
        // only the final assistant message was saved, so resumed sessions
        // lost everything the turn actually did.
        if let Err(e) = session::save_messages(&session_path, engine.messages()) {
            tracing::warn!("Failed to save session: {e}");
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

/// Render the tail of a conversation for display after a resume: user
/// prompts, assistant text, and tool names, with tool result payloads
/// omitted (they're context for the model, noise for the human).
fn replay_transcript(messages: &[crate::api::Message], model: &str, keep: usize) -> String {
    use crate::api::types::{ContentBlock, MessageContent};

    let mut out = String::new();
    let start = messages.len().saturating_sub(keep);
    if start > 0 {
        out.push_str(&format!(
            "  \x1b[2m… {start} earlier messages not shown (full history is in context)\x1b[0m\n"
        ));
    }

    for msg in &messages[start..] {
        match (msg.role.as_str(), &msg.content) {
            ("user", MessageContent::Text(t)) => {
                let t = crate::utils::sanitize_terminal_text(t);
                out.push_str(&format!("\x1b[1;34m>\x1b[0m {t}\n"));
            }
            ("assistant", MessageContent::Text(t)) => {
                let t = crate::utils::sanitize_terminal_text(t);
                out.push_str(&format!("\x1b[2m● {model}\x1b[0m {t}\n"));
            }
            ("assistant", MessageContent::Blocks(blocks)) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            let text = crate::utils::sanitize_terminal_text(text);
                            out.push_str(&format!("\x1b[2m● {model}\x1b[0m {text}\n"));
                        }
                        ContentBlock::Image { source } => {
                            out.push_str(&format!(
                                "  \x1b[2m[attached {} image]\x1b[0m\n",
                                source.media_type
                            ));
                        }
                        ContentBlock::Reasoning { .. } => {}
                        ContentBlock::ToolUse { name, .. } => {
                            let name = crate::utils::sanitize_terminal_text(name);
                            out.push_str(&format!("  \x1b[2m[{name}] ✓\x1b[0m\n"));
                        }
                        ContentBlock::ToolResult { .. } => {}
                    }
                }
            }
            // Tool-result messages: skipped, see doc comment
            _ => {}
        }
    }
    out.push('\n');
    out
}

/// Print the permission question for a tool. The answer is read from the
/// stdin channel by the caller.
fn print_permission_prompt(tool_name: &str, summary: &str) {
    let summary = crate::utils::sanitize_terminal_text(summary);
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
    let summary = crate::utils::sanitize_terminal_text(summary);
    let diff = crate::utils::sanitize_terminal_text(diff);
    if tool_name == "Bash" {
        println!(
            "\n  \x1b[33m⚡ {summary}\x1b[0m  \x1b[2m(y)es / (n)o / (a)lways this command\x1b[0m"
        );
    } else {
        println!("\n  \x1b[33m⚡ {summary}\x1b[0m  \x1b[2m(y)es / (n)o / (a)lways\x1b[0m");
    }
    println!("\n  \x1b[2m--- Diff Preview ---\x1b[0m");

    let colored_diff = colorize_diff(&diff);
    for line in colored_diff.lines() {
        println!("  {line}");
    }

    println!("  \x1b[2m--- End Diff ---\x1b[0m\n");

    print!("  Allow? ");
    let _ = stdout().flush();
}

/// Interpret a line of input as an answer to a permission prompt.
fn parse_permission_response(
    tool_name: &str,
    input: &serde_json::Value,
    line: &str,
) -> PermissionResponse {
    let trimmed = line.trim().to_lowercase();

    match trimmed.as_str() {
        "y" | "yes" | "" => PermissionResponse::Allow,
        "a" | "always" => PermissionResponse::always_allow_for(tool_name, input),
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
                parse_permission_response(
                    "Write",
                    &serde_json::json!({"file_path": "/tmp/x"}),
                    line
                ),
                PermissionResponse::Allow
            );
        }
    }

    #[test]
    fn permission_deny_on_anything_else() {
        assert_eq!(
            parse_permission_response("Write", &serde_json::json!({"file_path": "/tmp/x"}), "n\n"),
            PermissionResponse::Deny
        );
        assert_eq!(
            parse_permission_response(
                "Write",
                &serde_json::json!({"file_path": "/tmp/x"}),
                "wait, not that file\n"
            ),
            PermissionResponse::Deny
        );
    }

    #[test]
    fn permission_always_is_command_specific_for_bash() {
        assert_eq!(
            parse_permission_response("Bash", &serde_json::json!({"command": "cargo test"}), "a\n"),
            PermissionResponse::AlwaysAllowCommand("cargo test".to_string())
        );
    }

    #[test]
    fn permission_always_preserves_long_bash_command() {
        let command = format!("echo {}", "x".repeat(200));
        assert_eq!(
            parse_permission_response("Bash", &serde_json::json!({"command": command}), "a\n"),
            PermissionResponse::AlwaysAllowCommand(command)
        );
    }

    #[test]
    fn permission_always_for_non_bash() {
        assert_eq!(
            parse_permission_response("Write", &serde_json::json!({"file_path": "/tmp/x"}), "a\n"),
            PermissionResponse::AlwaysAllow
        );
    }

    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut in_escape = false;
        for c in s.chars() {
            match (in_escape, c) {
                (false, '\x1b') => in_escape = true,
                (false, _) => out.push(c),
                (true, 'm') => in_escape = false,
                (true, _) => {}
            }
        }
        out
    }

    #[test]
    fn replay_shows_prompts_responses_and_tools() {
        use crate::api::types::{ContentBlock, Message};

        let messages = vec![
            Message::user("run the tests"),
            Message::assistant_blocks(vec![
                ContentBlock::Text {
                    text: "Running them now.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tu_1".to_string(),
                    name: "Bash".to_string(),
                    input: serde_json::json!({"command": "cargo test"}),
                },
            ]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".to_string(),
                content: "huge noisy output".to_string(),
                is_error: None,
            }]),
            Message::assistant_text("All green."),
        ];

        let out = strip_ansi(&replay_transcript(&messages, "qwen", 12));
        assert!(out.contains("> run the tests"));
        assert!(out.contains("Running them now."));
        assert!(out.contains("[Bash]"));
        assert!(out.contains("All green."));
        assert!(
            !out.contains("huge noisy output"),
            "tool result payloads are omitted"
        );
        assert!(!out.contains("earlier messages"), "no truncation header");
    }

    #[test]
    fn replay_truncates_to_the_tail() {
        use crate::api::types::Message;

        let messages: Vec<Message> = (0..30).map(|i| Message::user(&format!("m{i}"))).collect();
        let out = strip_ansi(&replay_transcript(&messages, "qwen", 12));
        assert!(out.contains("… 18 earlier messages not shown"));
        assert!(!out.contains("> m17\n"), "older messages hidden");
        assert!(out.contains("> m29"), "latest message shown");
    }
}
