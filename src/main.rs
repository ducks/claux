#![allow(dead_code, clippy::if_same_then_else)]

mod api;
mod bootstrap;
mod checkpoint;
mod cli;
mod commands;
mod compact;
mod config;
mod context;
mod cost;
mod db;
#[cfg(test)]
mod evals;
mod onboarding;
mod permissions;
mod plugin;
mod query;
mod repl;
mod session;
#[cfg(test)]
mod test_support;
mod theme;
mod tools;
mod tui;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;

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

    if let Some(command) = &args.command {
        match command {
            cli::CliCommand::Config {
                command:
                    cli::ConfigCommand::Init {
                        provider,
                        model,
                        force,
                    },
            } => {
                let path = onboarding::init_config(*provider, model.as_deref(), *force)?;
                println!("Created {}", path.display());
                println!("Run `claux doctor` to verify the setup.");
                return Ok(());
            }
            cli::CliCommand::Doctor { offline } => {
                let config = config::Config::load(args.trust_project)?;
                let report = onboarding::doctor(&config, *offline).await;
                print!("{}", report.text);
                if !report.healthy {
                    anyhow::bail!("doctor found configuration errors");
                }
                return Ok(());
            }
        }
    }

    // Load config (global + project)
    let mut config = config::Config::load(args.trust_project)?;
    if let Some(ref mode) = args.permission_mode {
        config.permission_mode = serde_json::from_value(serde_json::Value::String(mode.clone()))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Invalid permission mode {mode:?}; expected default, accept-edits, bypass, or plan"
                )
            })?;
    }

    // Build plugin registry
    let mut plugin_registry = plugin::PluginRegistry::new();
    for plugin_config in &config.plugins {
        plugin_registry.add(Box::new(plugin::CommandPlugin::new(
            &plugin_config.name,
            &plugin_config.command,
            &plugin_config.args,
            plugin_config.trigger.clone(),
        )));
    }
    if !plugin_registry.is_empty() {
        tracing::info!(
            "Loaded {} plugin(s): {} context, {} tool-start, {} tool-complete, {} session-start, {} turn-end, {} permission-request",
            plugin_registry.len(),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnContextBuild),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnToolStart),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnToolComplete),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnSessionStart),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnTurnEnd),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnPermissionRequest),
        );
    }
    let plugin_registry = Arc::new(plugin_registry);

    let model = args.model.as_deref().unwrap_or(&config.model).to_string();

    tracing::debug!(
        "Config loaded: openai_base_url={:?} openai_api_key_cmd={:?} model={}",
        config.openai_base_url,
        config.openai_api_key_cmd,
        config.model
    );

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

    // Connect once; the tools are moved into whichever frontend is selected.
    let mcp_tools = bootstrap::connect_mcp_tools(&config).await;

    // One-shot mode: --print / -p
    if let Some(ref prompt) = args.prompt {
        let mut tool_registry = tools::ToolRegistry::new_with_agent_factory(
            agent_factory,
            model.clone(),
            config.permission_mode,
        );
        tool_registry.add_tools(mcp_tools);
        let permission_checker = permissions::PermissionChecker::new(config.permission_mode);
        let mut engine = query::Engine::new(provider, tool_registry, permission_checker, &model);
        engine.set_plugins(plugin_registry.clone());
        engine.set_auto_compact_threshold(config.auto_compact_threshold);
        engine.set_max_tokens(config.max_tokens);
        engine.set_model_pricing(config.model_pricing.get(&model).copied());

        let system_prompt = context::build_system_prompt_for_model(
            &model,
            Some(&plugin_registry),
            &config::HookTrigger::OnContextBuild,
            config.is_anthropic(),
        )
        .await?;
        engine.set_system_prompt(system_prompt);

        let response = engine
            .submit(prompt, tokio_util::sync::CancellationToken::new())
            .await?;
        print!("{response}");
        return Ok(());
    }

    // Interactive REPL
    let mut tool_registry = tools::ToolRegistry::new_with_agent_factory(
        agent_factory,
        model.clone(),
        config.permission_mode,
    );
    tool_registry.add_tools(mcp_tools);
    let permission_checker = permissions::PermissionChecker::new(config.permission_mode);
    let mut engine = query::Engine::new(provider, tool_registry, permission_checker, &model);
    engine.set_plugins(plugin_registry.clone());
    engine.set_auto_compact_threshold(config.auto_compact_threshold);
    engine.set_max_tokens(config.max_tokens);
    engine.set_model_pricing(config.model_pricing.get(&model).copied());

    // Run session-start hooks
    plugin::PluginRegistry::execute_side_effects(
        &plugin_registry,
        &config::HookTrigger::OnSessionStart,
        None,
    )
    .await?;

    // Resume a previous session if requested. The matched id is handed to
    // the REPL so it continues that session instead of forking a new one.
    let mut resumed_id: Option<String> = None;
    if let Some(ref session_id) = args.resume {
        match session::find_session(session_id)? {
            Some((sid, path)) => {
                let (meta, messages) = session::load_session(&path)?;
                engine.set_messages(messages);
                eprintln!(
                    "Resumed session {} ({}, {} messages)",
                    meta.id,
                    meta.model,
                    engine.message_count()
                );
                resumed_id = Some(sid);
            }
            None => {
                eprintln!("Session not found: {session_id}. Starting new session.");
            }
        }
    }

    if args.tui {
        tui::run(engine, &config, &plugin_registry).await
    } else {
        repl::run(engine, &config, &plugin_registry, resumed_id).await
    }
}

/// Build a provider from config.
fn build_provider(config: &config::Config, model: &str) -> Result<Box<dyn api::Provider>> {
    // Check for OpenAI-compatible provider in config
    if let Some(ref base_url) = config.openai_base_url {
        let api_key = config.resolve_openai_key().unwrap_or_default();
        if api_key.is_empty() && base_url.contains("api.openai.com") {
            anyhow::bail!(
                "No OpenAI API key found. Set OPENAI_API_KEY or configure \
                 openai_api_key_cmd in ~/.config/claux/config.toml. ChatGPT login \
                 credentials are not API credentials."
            );
        }
        let name = config.openai_provider_name.as_deref().unwrap_or("openai");
        return match config.openai_protocol {
            config::OpenAIProtocol::ChatCompletions => Ok(Box::new(
                api::OpenAICompatProvider::new(base_url, &api_key, model, name),
            )),
            config::OpenAIProtocol::Responses => Ok(Box::new(api::OpenAIResponsesProvider::new(
                base_url,
                &api_key,
                model,
                name,
                config.openai_reasoning_effort.as_deref(),
            ))),
        };
    }

    // Default: Anthropic
    let auth = config.resolve_auth().ok_or_else(|| {
        anyhow::anyhow!(
            "No authentication found. Set ANTHROPIC_API_KEY, or run \
             `claux config init` followed by `claux doctor`."
        )
    })?;

    Ok(Box::new(api::AnthropicProvider::new(auth, model)))
}
