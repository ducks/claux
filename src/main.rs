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
mod logging;
mod model;
mod model_catalog;
mod onboarding;
mod output;
mod permissions;
mod plugin;
mod providers;
mod query;
mod repl;
mod sandbox;
mod session;
mod shutdown;
#[cfg(test)]
mod test_support;
mod theme;
mod tokenizer_fingerprint;
mod tools;
mod tui;
mod usage;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match run().await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("Error: {error:?}");
            std::process::ExitCode::from(1)
        }
    }
}

async fn run() -> Result<std::process::ExitCode> {
    use std::process::ExitCode;
    let args = cli::Cli::parse();

    if let Some(cli::CliCommand::SandboxExec { workspace, command }) = &args.command {
        return command_sandbox::run_helper(workspace, command).map(|()| ExitCode::SUCCESS);
    }
    if matches!(args.command, Some(cli::CliCommand::SandboxProbe)) {
        return command_sandbox::run_probe().map(|()| ExitCode::SUCCESS);
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
        .with_writer(logging::writer)
        .init();

    if let Some(command) = &args.command {
        match command {
            cli::CliCommand::Archive { session: prefix } => {
                let (_, path) = session::find_session(prefix)?
                    .ok_or_else(|| anyhow::anyhow!("Session not found: {prefix}"))?;
                let archive = session::load_archive(&path)?;
                serde_json::to_writer_pretty(std::io::stdout().lock(), &archive)?;
                return Ok(ExitCode::SUCCESS);
            }
            cli::CliCommand::Auth { command } => {
                match command {
                    cli::AuthCommand::Login {
                        provider,
                        headless,
                        no_browser,
                    } => {
                        let descriptor = provider.0.descriptor();
                        match descriptor.auth {
                            providers::AuthFlow::OpenRouterPkce => {
                                auth::login_openrouter(*headless, *no_browser).await?
                            }
                            providers::AuthFlow::ApiKey => {
                                auth::login_api_key(descriptor.id, descriptor.label)?
                            }
                            providers::AuthFlow::None => {
                                anyhow::bail!("provider has no login integration")
                            }
                        }
                    }
                    cli::AuthCommand::Status { provider } => {
                        let descriptor = provider.0.descriptor();
                        auth::status_provider(descriptor.id, descriptor.label)?;
                    }
                    cli::AuthCommand::Logout { provider } => {
                        let descriptor = provider.0.descriptor();
                        auth::logout_provider(descriptor.id, descriptor.label)?;
                    }
                    cli::AuthCommand::Token { provider } => {
                        let descriptor = provider.0.descriptor();
                        auth::print_provider_token(descriptor.id, descriptor.label)?;
                    }
                }
                return Ok(ExitCode::SUCCESS);
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
                return Ok(ExitCode::SUCCESS);
            }
            cli::CliCommand::Doctor { offline } => {
                let config = config::Config::load(args.trust_project)?;
                let report = onboarding::doctor(&config, *offline).await;
                print!("{}", report.text);
                if !report.healthy {
                    anyhow::bail!("doctor found configuration errors");
                }
                return Ok(ExitCode::SUCCESS);
            }
            cli::CliCommand::Usage { command } => {
                match command {
                    cli::UsageCommand::Status { provider, json } => {
                        usage::status(provider.as_deref(), *json).await?
                    }
                }
                return Ok(ExitCode::SUCCESS);
            }
            cli::CliCommand::TokenizerFingerprint {
                models,
                format,
                json,
                report_output,
                resume_fingerprint,
            } => {
                let format = if *json {
                    cli::TokenizerOutputFormat::Json
                } else {
                    format.unwrap_or_default()
                };
                tokenizer_fingerprint::run(
                    models,
                    format,
                    report_output.as_deref(),
                    *resume_fingerprint,
                )
                .await?;
                return Ok(ExitCode::SUCCESS);
            }
            cli::CliCommand::SandboxExec { .. } | cli::CliCommand::SandboxProbe => {
                unreachable!("handled before logging")
            }
        }
    }

    // Load config (global + project)
    let mut config = config::Config::load(args.trust_project)?;
    command_sandbox::configure_child_environment(
        config.sensitive_environment_names(),
        config.strip_agent_sockets,
    );
    if let Some(ref mode) = args.permission_mode {
        config.permission_mode = serde_json::from_value(serde_json::Value::String(mode.clone()))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Invalid permission mode {mode:?}; expected default, accept-edits, auto, bypass, or plan"
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
            "Loaded {} plugin(s): {} context, {} tool-start, {} tool-complete, {} session-start, {} turn-end, {} permission-request, {} permission-check",
            plugin_registry.len(),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnContextBuild),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnToolStart),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnToolComplete),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnSessionStart),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnTurnEnd),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnPermissionRequest),
            plugin_registry.get_by_trigger(&config::HookTrigger::OnPermissionCheck),
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
        // Classify the failure for the transcript, the JSON output, and the
        // exit code. Cancellation wins because the engine reports a clean
        // interrupt rather than an error.
        let failure: Option<query::FailureRecord> = if response.is_ok() {
            None
        } else if cancel.is_cancelled() {
            Some(query::FailureRecord::cancelled(
                engine.last_failure().map(|f| f.attempts).unwrap_or(1),
            ))
        } else {
            Some(
                engine
                    .last_failure()
                    .cloned()
                    .unwrap_or_else(query::FailureRecord::unclassified),
            )
        };
        if let Some(path) = args.transcript.as_deref() {
            let error = response.as_ref().err().map(ToString::to_string);
            let outcome = match (&response, &error) {
                (Ok(result), _) => output::TranscriptOutcome::Completed { result },
                (Err(_), Some(message)) => output::TranscriptOutcome::Error {
                    message,
                    failure: failure.as_ref(),
                },
                (Err(_), None) => unreachable!("errors always render a message"),
            };
            let transcript = output::OneShotTranscript::new(
                engine.model(),
                &engine.cost,
                engine.messages(),
                engine.tool_trace(),
                engine.execution_timing(),
                outcome,
            )
            .with_archive(engine.archive());
            output::write_transcript(path, &transcript)?;
        }
        let json = matches!(
            args.output_format.unwrap_or_default(),
            cli::OutputFormat::Json
        );
        match response {
            Ok(response) => {
                if json {
                    let output =
                        output::OneShotOutput::new(&response, engine.model(), &engine.cost);
                    serde_json::to_writer(std::io::stdout().lock(), &output)?;
                    println!();
                } else {
                    print!("{response}");
                }
                return Ok(ExitCode::SUCCESS);
            }
            Err(error) => {
                let message = error.to_string();
                let failure = failure.expect("failed responses are classified");
                if json {
                    // Always give supervisors a machine-readable outcome, not
                    // just a stderr string and exit status.
                    let output = output::OneShotOutput::failed(
                        engine.model(),
                        &engine.cost,
                        &message,
                        Some(&failure),
                    );
                    serde_json::to_writer(std::io::stdout().lock(), &output)?;
                    println!();
                }
                eprintln!("Error: {error:?}");
                return Ok(ExitCode::from(failure.kind.exit_code()));
            }
        }
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
        return tui::run(&config, plugin_registry, models)
            .await
            .map(|()| ExitCode::SUCCESS);
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
                if meta.recovered_messages > 0 {
                    eprintln!("Warning: recovered session with {} unreadable message(s) replaced by placeholders; original rows retained in the database.", meta.recovered_messages);
                }
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
        if let Some(id) = resumed_id.as_ref() {
            engine.set_archive(session::load_archive(&std::path::PathBuf::from(format!(
                "sqlite://{id}"
            )))?);
        }
    }
    repl::run(engine, &config, plugin_registry, resumed_id, resolved_model)
        .await
        .map(|()| ExitCode::SUCCESS)
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
    let permission_policy =
        permissions::PermissionPolicy::new(config.permission_mode, config.permission_rules()?);
    let mut tool_registry = tools::ToolRegistry::new_with_agent_factory(
        agent_factory,
        model.clone(),
        metadata,
        permission_policy.clone(),
        config.is_project_trusted(),
        sandbox_policy,
        command_sandbox,
    );
    tool_registry.add_tools(bootstrap::connect_mcp_tools(config).await)?;

    let permission_checker = permission_policy.checker();
    let mut engine = query::Engine::new(provider, tool_registry, permission_checker, model);
    engine.set_model_binding(resolved.binding.clone());
    engine.set_plugins(plugins);
    engine.set_auto_compact_threshold(config.auto_compact_threshold);
    engine.set_max_tokens(config.max_tokens);
    engine.set_max_rounds(config.max_rounds);
    engine.set_model_metadata(metadata);
    Ok(engine)
}

/// Build a provider from config.
fn build_provider(resolved: &config::ResolvedModel) -> Result<Box<dyn api::Provider>> {
    let binding = &resolved.binding;
    let api_key = resolved.resolve_api_key().unwrap_or_default();
    if api_key.is_empty() && resolved.requires_api_key() {
        let login_hint = providers::for_binding(binding)
            .filter(|descriptor| descriptor.auth != providers::AuthFlow::None)
            .map(|descriptor| format!(" or run `claux auth login {}`", descriptor.id))
            .unwrap_or_default();
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
            if binding.reasoning_effort.is_some() {
                tracing::warn!(profile = %binding.profile,
                    "reasoning_effort is not supported by the Anthropic adapter and will be ignored");
            }
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
