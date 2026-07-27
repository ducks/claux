//! TUI module with screen-based architecture.
//!
//! Each screen (home, chat) is self-contained with its own state, drawing,
//! and key handling. The top-level loop dispatches between screens based
//! on the Action returned by each.

pub mod chat;
pub mod home;
mod input;
pub mod markdown;
mod screen;
mod terminal;
mod ui;

use anyhow::Result;
use std::sync::Arc;

use crate::config::{Config, HookTrigger, ModelBinding, ProviderKind, ResolvedModel};
use crate::context;
use crate::db::Db;
use crate::plugin::PluginRegistry;
use crate::query::Engine;
use crate::theme::Theme;

use screen::Action;
use terminal::TerminalGuard;

/// Run the TUI application. Provider construction is deferred until a
/// session is opened so Home remains available without startup credentials.
pub async fn run(
    config: &Config,
    plugins: Arc<PluginRegistry>,
    models: Vec<ResolvedModel>,
) -> Result<()> {
    if models.is_empty() {
        anyhow::bail!("no models configured; set `model` in ~/.config/claux/config.toml");
    }

    // Open database
    let db_path = crate::session::db_path()?;
    let db = Db::open(&db_path)?;

    let mut terminal_guard = TerminalGuard::enter()?;

    let theme = Theme::dark();
    let mut engine: Option<Engine> = None;
    let mut engine_binding: Option<ModelBinding> = None;
    let mut home_notice = None;

    let app_result: Result<()> = async {
        // Screen loop: home -> chat -> home -> ...
        let mut next_action = Action::Home;
        loop {
            tracing::debug!("TUI action: {next_action:?}");
            match next_action {
                Action::Home => {
                    let mut home_screen =
                        home::HomeScreen::new(Db::open(&db_path)?, theme, models.clone());
                    if let Some(notice) = home_notice.take() {
                        home_screen.set_notice(notice);
                    }
                    next_action = home_screen.run(terminal_guard.terminal_mut())?;
                }
                Action::Chat { session_id } => {
                    let session = db
                        .get_session(&session_id)?
                        .ok_or_else(|| anyhow::anyhow!("session {session_id} no longer exists"))?;
                    let resolved = match session.model_binding.as_ref() {
                        Some(binding) => config.resolve_binding(binding),
                        None => config.resolve_model(&session.model).map_err(|error| {
                            anyhow::anyhow!(
                                "Legacy session '{}' uses model '{}': {error}",
                                session.id,
                                session.model
                            )
                        }),
                    };
                    let resolved = match resolved {
                        Ok(resolved) => resolved,
                        Err(error) => {
                            home_notice = Some(format!(
                                "Cannot open '{}': {error}. Restore its profile or create a new chat.",
                                session.name.as_deref().unwrap_or(&session.id)
                            ));
                            next_action = Action::Home;
                            continue;
                        }
                    };
                    if engine_binding.as_ref() != Some(&resolved.binding) {
                        match crate::build_engine(config, &resolved, plugins.clone()).await {
                            Ok(new_engine) => {
                                engine = Some(new_engine);
                                engine_binding = Some(resolved.binding.clone());
                            }
                            Err(error) => {
                                home_notice = Some(format!(
                                    "Cannot open '{}': {error}",
                                    session.name.as_deref().unwrap_or(&session.id)
                                ));
                                next_action = Action::Home;
                                continue;
                            }
                        }
                    }
                    let engine = engine.as_mut().expect("engine initialized");
                    let system_prompt = context::build_system_prompt_for_model(
                        &resolved.binding.model,
                        Some(&plugins),
                        &HookTrigger::OnContextBuild,
                        resolved.binding.provider_kind == ProviderKind::Anthropic,
                    )
                    .await?;
                    engine.set_system_prompt(system_prompt);
                    next_action = chat::run(
                        engine,
                        &session_id,
                        &db,
                        terminal_guard.terminal_mut(),
                        theme,
                    )
                    .await?;
                    engine_binding = engine.model_binding().cloned();
                }
                Action::Quit => return Ok(()),
            }
        }
    }
    .await;

    let restore_result = terminal_guard.restore();
    app_result?;
    restore_result?;

    if let Some(engine) = engine {
        println!("{}", engine.cost.format_summary());
    }
    Ok(())
}
