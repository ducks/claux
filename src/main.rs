mod api;
mod auth;
mod bootstrap;
mod checkpoint;
mod cli;
mod command_sandbox;
mod commands;
mod compact;
mod config;
mod context;
mod cost;
mod db;
#[cfg(test)]
mod evals;
mod image_input;
mod model;
mod model_catalog;
mod onboarding;
mod output;
mod permissions;
mod plugin;
mod query;
mod repl;
mod sandbox;
mod session;
mod shutdown;
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

    if let Some(cli::CliCommand::SandboxExec { workspace, command }) = &args.command {
        return command_sandbox::run_helper(workspace, command);
    }
    if matches!(args.command, Some(cli::CliCommand::SandboxProbe)) {
        return command_sandbox::run_probe();
    }

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
            cli::CliCommand::Auth { command } => {
                match command {
                    cli::AuthCommand::Login {
                        provider: cli::AuthProvider::OpenRouter,
                        headless,
                        no_browser,
                    } => auth::login_openrouter(*headless, *no_browser).await?,
                    cli::AuthCommand::Status {
                        provider: cli::AuthProvider::OpenRouter,
                    } => auth::status_openrouter()?,
                    cli::AuthCommand::Logout {
                        provider: cli::AuthProvider::OpenRouter,
                    } => auth::logout_openrouter()?,
                    cli::AuthCommand::Token {
                        provider: cli::AuthProvider::OpenRouter,
                    } => auth::print_openrouter_token()?,
                }
                return Ok(());
            }
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
            cli::CliCommand::SandboxExec { .. } | cli::CliCommand::SandboxProbe => {
                unreachable!("handled before logging")
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

    let requested_model = match args.model.as_deref() {
        Some(model) => config.resolve_model(model)?,
        None => config.default_resolved_model()?,
    };

    tracing::debug!(
        "Config loaded: openai_base_url={:?} openai_api_key_cmd={:?} model={}",
        config.openai_base_url,
        config.openai_api_key_cmd,
        config.model
    );

    // One-shot mode: --print / -p
    if let Some(ref prompt) = args.prompt {
        let mut engine = build_engine(&config, &requested_model, plugin_registry.clone()).await?;

        let system_prompt = context::build_system_prompt_for_model(
            &requested_model.binding.model,
            Some(&plugin_registry),
            &config::HookTrigger::OnContextBuild,
            requested_model.binding.provider_kind == config::ProviderKind::Anthropic,
            config.is_project_trusted(),
        )
        .await?;
        engine.set_system_prompt(system_prompt);
        if let Some(path) = args.transcript.as_ref() {
            engine.set_transcript_checkpoint(path.clone());
        }

        let cancel = shutdown::one_shot_cancellation_token()?;
        let response = if args.image.is_empty() {
            engine.submit(prompt, cancel.clone()).await
        } else {
            let images = image_input::load_images(&args.image)?;
            engine
                .submit_message(
                    api::types::Message::user_with_images(prompt, images),
                    cancel.clone(),
                )
                .await
        };
        let response = shutdown::classify_one_shot_response(response, cancel.is_cancelled());
        if let Some(path) = args.transcript.as_deref() {
            let error = response.as_ref().err().map(ToString::to_string);
            let result = response.as_ref().ok().map(String::as_str);
            let transcript = output::OneShotTranscript::new(
                engine.model(),
                &engine.cost,
                engine.messages(),
                engine.tool_trace(),
                engine.execution_timing(),
                result,
                error.as_deref(),
            );
            output::write_transcript(path, &transcript)?;
        }
        let response = response?;
        match args.output_format.unwrap_or_default() {
            cli::OutputFormat::Text => print!("{response}"),
            cli::OutputFormat::Json => {
                let output = output::OneShotOutput::new(&response, engine.model(), &engine.cost);
                serde_json::to_writer(std::io::stdout().lock(), &output)?;
                println!();
            }
        }
        return Ok(());
    }

    // Run session-start hooks
    plugin::PluginRegistry::execute_side_effects(
        &plugin_registry,
        &config::HookTrigger::OnSessionStart,
        None,
    )
    .await?;

    if args.tui {
        let mut models = config.selectable_models()?;
        if let Some(cli_model) = args.model.as_deref() {
            models.retain(|configured| {
                configured.binding.profile != cli_model && configured.binding.model != cli_model
            });
            models.insert(0, requested_model.clone());
        }
        return tui::run(&config, plugin_registry, models).await;
    }

    // Resume a previous session if requested. The matched id is handed to
    // the REPL so it continues that session instead of forking a new one.
    let mut resumed_id: Option<String> = None;
    let mut resolved_model = requested_model;
    let mut resumed_messages = None;
    if let Some(ref session_id) = args.resume {
        match session::find_session(session_id)? {
            Some((sid, path)) => {
                let (meta, messages) = session::load_session(&path)?;
                resolved_model = match meta.model_binding.as_ref() {
                    Some(binding) => config.resolve_binding(binding)?,
                    None => config.resolve_model(&meta.model).map_err(|error| {
                        anyhow::anyhow!(
                            "Session {} uses legacy model '{}', which cannot be resolved: {error}. \
                             Add a matching model profile or start a new session.",
                            meta.id,
                            meta.model
                        )
                    })?,
                };
                eprintln!(
                    "Resumed session {} ({}, {} messages)",
                    meta.id,
                    meta.model,
                    messages.len()
                );
                resumed_messages = Some(messages);
                resumed_id = Some(sid);
            }
            None => {
                eprintln!("Session not found: {session_id}. Starting new session.");
            }
        }
    }

    let mut engine = build_engine(&config, &resolved_model, plugin_registry.clone()).await?;
    if let Some(messages) = resumed_messages {
        engine.set_messages(messages);
    }
    repl::run(engine, &config, plugin_registry, resumed_id, resolved_model).await
}

async fn build_engine(
    config: &config::Config,
    resolved: &config::ResolvedModel,
    plugins: Arc<plugin::PluginRegistry>,
) -> Result<query::Engine> {
    let model = &resolved.binding.model;
    let metadata = model_catalog::resolve(resolved).await;
    let provider = build_provider(resolved)?;
    tracing::info!(
        "Provider: {} ({}, profile {})",
        provider.name(),
        model,
        resolved.binding.profile
    );

    let resolved_for_factory = resolved.clone();
    let agent_factory: tools::agent::ProviderFactory = Box::new(move || {
        build_provider(&resolved_for_factory).expect("failed to build agent provider")
    });
    let sandbox_policy = Arc::new(sandbox::SandboxPolicy::from_native_tool_policy(
        config.native_tool_filesystem_policy,
        std::env::current_dir()?,
    )?);
    let command_sandbox = Arc::new(command_sandbox::CommandSandbox::new(
        config.bash_filesystem_policy,
        std::env::current_dir()?,
    )?);
    let mut tool_registry = tools::ToolRegistry::new_with_agent_factory(
        agent_factory,
        model.clone(),
        metadata,
        config.permission_mode,
        config.is_project_trusted(),
        sandbox_policy,
        command_sandbox,
    );
    tool_registry.add_tools(bootstrap::connect_mcp_tools(config).await);

    let permission_checker = permissions::PermissionChecker::new(config.permission_mode);
    let mut engine = query::Engine::new(provider, tool_registry, permission_checker, model);
    engine.set_model_binding(resolved.binding.clone());
    engine.set_plugins(plugins);
    engine.set_auto_compact_threshold(config.auto_compact_threshold);
    engine.set_max_tokens(config.max_tokens);
    engine.set_model_metadata(metadata);
    Ok(engine)
}

/// Build a provider from config.
fn build_provider(resolved: &config::ResolvedModel) -> Result<Box<dyn api::Provider>> {
    let binding = &resolved.binding;
    let api_key = resolved.resolve_api_key().unwrap_or_default();
    if api_key.is_empty() && resolved.requires_api_key() {
        let login_hint = if binding.provider_name.eq_ignore_ascii_case("openrouter")
            || binding
                .base_url
                .as_deref()
                .is_some_and(|url| url.contains("openrouter.ai"))
        {
            " or run `claux auth login openrouter`"
        } else {
            ""
        };
        anyhow::bail!(
            "No API key found for profile '{}' (provider '{}'). Set {}{} or update \
             ~/.config/claux/config.toml.",
            binding.profile,
            binding.provider_name,
            binding.api_key_env,
            login_hint,
        );
    }
    match binding.provider_kind {
        config::ProviderKind::Openai => {
            let base_url = binding.base_url.as_deref().ok_or_else(|| {
                anyhow::anyhow!("saved provider '{}' has no base URL", binding.provider)
            })?;
            match binding.protocol {
                config::OpenAIProtocol::ChatCompletions => Ok(Box::new(
                    api::OpenAICompatProvider::new(
                        base_url,
                        &api_key,
                        &binding.model,
                        &binding.provider_name,
                        binding.reasoning_effort.as_deref(),
                    )
                    .with_prompt_caching(binding.prompt_caching)
                    .with_eof_without_finish_reason(binding.allow_eof_without_finish_reason),
                )),
                config::OpenAIProtocol::Responses => {
                    Ok(Box::new(api::OpenAIResponsesProvider::new(
                        base_url,
                        &api_key,
                        &binding.model,
                        &binding.provider_name,
                        binding.reasoning_effort.as_deref(),
                    )))
                }
            }
        }
        config::ProviderKind::Anthropic => {
            if api_key.is_empty() {
                anyhow::bail!(
                    "No authentication found for profile '{}'. Set {}.",
                    binding.profile,
                    binding.api_key_env
                );
            }
            let key = config::AnthropicApiKey::new(api_key);
            Ok(Box::new(match binding.base_url.as_deref() {
                Some(base_url) => {
                    api::AnthropicProvider::with_base_url(key, &binding.model, base_url)
                }
                None => api::AnthropicProvider::new(key, &binding.model),
            }))
        }
    }
}
