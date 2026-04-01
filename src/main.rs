mod api;
mod cli;
mod commands;
mod config;
mod context;
mod cost;
mod permissions;
mod query;
mod repl;
mod session;
mod tools;

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Cli::parse();

    // Init logging
    let filter = if args.debug {
        "claude_rs=debug"
    } else if args.verbose {
        "claude_rs=info"
    } else {
        "claude_rs=warn"
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    // Load config (global + project)
    let config = config::Config::load()?;

    // Resolve API key
    let api_key = config
        .resolve_api_key()
        .ok_or_else(|| anyhow::anyhow!("No API key found. Set ANTHROPIC_API_KEY or configure in ~/.claude-rs/config.toml"))?;

    let model = args
        .model
        .as_deref()
        .unwrap_or(&config.model)
        .to_string();

    // One-shot mode: --print / -p
    if let Some(ref prompt) = args.prompt {
        let client = api::Client::new(api_key, &model);
        let tool_registry = tools::ToolRegistry::new();
        let permission_checker = permissions::PermissionChecker::new(config.permission_mode);
        let mut engine = query::Engine::new(client, tool_registry, permission_checker, &model);

        let system_prompt = context::build_system_prompt().await?;
        engine.set_system_prompt(system_prompt);

        let response = engine.submit(prompt).await?;
        print!("{}", response);
        return Ok(());
    }

    // Interactive REPL
    let client = api::Client::new(api_key, &model);
    let tool_registry = tools::ToolRegistry::new();
    let permission_checker = permissions::PermissionChecker::new(config.permission_mode);
    let mut engine = query::Engine::new(client, tool_registry, permission_checker, &model);

    // Resume a previous session if requested
    if let Some(ref session_id) = args.resume {
        let sessions = session::list_sessions()?;
        let found = sessions
            .iter()
            .find(|(sid, _)| sid == session_id || sid.starts_with(session_id));

        match found {
            Some((_, path)) => {
                let (meta, messages) = session::load_session(path)?;
                engine.set_messages(messages);
                eprintln!(
                    "Resumed session {} ({}, {} messages)",
                    meta.id,
                    meta.model,
                    engine.message_count()
                );
            }
            None => {
                eprintln!("Session not found: {}. Starting new session.", session_id);
            }
        }
    }

    repl::run(engine, &config).await
}
