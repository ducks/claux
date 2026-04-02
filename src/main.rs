mod api;
mod cli;
mod commands;
mod compact;
mod config;
mod context;
mod cost;
mod permissions;
mod plugin;
mod query;
mod repl;
mod session;
mod tools;
mod tui;

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Cli::parse();

    // Init logging
    let filter = if args.debug {
        "claux=debug"
    } else if args.verbose {
        "claux=info"
    } else {
        "claux=warn"
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    // Load config (global + project)
    let config = config::Config::load()?;

    // Build plugin registry
    let mut plugin_registry = plugin::PluginRegistry::new();
    for plugin_config in &config.plugins {
        plugin_registry.add(Box::new(plugin::CommandPlugin::new(
            &plugin_config.name,
            &plugin_config.command,
            &plugin_config.args,
        )));
    }
    if !plugin_registry.is_empty() {
        tracing::info!("Loaded {} plugin(s)", plugin_registry.len());
    }

    let model = args
        .model
        .as_deref()
        .unwrap_or(&config.model)
        .to_string();

    tracing::debug!("Config loaded: openai_base_url={:?} openai_api_key_cmd={:?} model={}",
        config.openai_base_url, config.openai_api_key_cmd, config.model);

    // Build the provider
    let provider = build_provider(&config, &model)?;
    let provider_name = provider.name().to_string();
    tracing::info!("Provider: {} ({})", provider_name, model);

    // Build a factory for agent sub-providers
    let config_for_factory = config.clone();
    let model_for_factory = model.clone();
    let agent_factory: tools::agent::ProviderFactory = Box::new(move || {
        build_provider(&config_for_factory, &model_for_factory)
            .expect("failed to build agent provider")
    });

    // One-shot mode: --print / -p
    if let Some(ref prompt) = args.prompt {
        let tool_registry = tools::ToolRegistry::new_with_agent_factory(agent_factory, model.clone());
        let permission_checker = permissions::PermissionChecker::new(config.permission_mode);
        let mut engine = query::Engine::new(provider, tool_registry, permission_checker, &model);

        let system_prompt = context::build_system_prompt_for_model(&model, Some(&plugin_registry)).await?;
        engine.set_system_prompt(system_prompt);

        let response = engine.submit(prompt).await?;
        print!("{}", response);
        return Ok(());
    }

    // Interactive REPL
    let tool_registry = tools::ToolRegistry::new_with_agent_factory(agent_factory, model.clone());
    let permission_checker = permissions::PermissionChecker::new(config.permission_mode);
    let mut engine = query::Engine::new(provider, tool_registry, permission_checker, &model);

    // Build system prompt with plugins for REPL mode
    let system_prompt = context::build_system_prompt_for_model(&model, Some(&plugin_registry)).await?;
    engine.set_system_prompt(system_prompt);

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

    if args.tui {
        tui::run(engine, &config, &plugin_registry).await
    } else {
        repl::run(engine, &config, &plugin_registry).await
    }
}

/// Build a provider from config.
fn build_provider(
    config: &config::Config,
    model: &str,
) -> Result<Box<dyn api::Provider>> {
    // Check for OpenAI-compatible provider in config
    if let Some(ref base_url) = config.openai_base_url {
        let api_key = config.resolve_openai_key().unwrap_or_default();
        let name = config.openai_provider_name.as_deref().unwrap_or("openai");
        return Ok(Box::new(api::OpenAICompatProvider::new(
            base_url, &api_key, model, name,
        )));
    }

    // Default: Anthropic
    let auth = config
        .resolve_auth()
        .ok_or_else(|| anyhow::anyhow!(
            "No authentication found. Set ANTHROPIC_API_KEY, configure ~/.config/claux/config.toml, or run `claude login`."
        ))?;

    Ok(Box::new(api::AnthropicProvider::new(auth, model)))
}
