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

use crate::config::{Config, HookTrigger};
use crate::context;
use crate::db::Db;
use crate::plugin::PluginRegistry;
use crate::query::Engine;
use crate::theme::Theme;

use screen::Action;
use terminal::TerminalGuard;

/// Run the TUI application. Provider construction is deferred until a
/// session is opened so Home remains available without startup credentials.
pub async fn run(config: &Config, plugins: Arc<PluginRegistry>, models: Vec<String>) -> Result<()> {
    if models.is_empty() {
        anyhow::bail!("no models configured; set `model` in ~/.config/claux/config.toml");
    }

    // Open database
    let db_path = crate::session::db_path()?;
    let db = Db::open(&db_path)?;

    let mut terminal_guard = TerminalGuard::enter()?;

    let theme = Theme::dark();
    let mut engine: Option<Engine> = None;

    let app_result: Result<()> = async {
        // Screen loop: home -> chat -> home -> ...
        let mut next_action = Action::Home;
        loop {
            tracing::debug!("TUI action: {next_action:?}");
            match next_action {
                Action::Home => {
                    let mut home_screen =
                        home::HomeScreen::new(Db::open(&db_path)?, theme, models.clone());
                    next_action = home_screen.run(terminal_guard.terminal_mut())?;
                }
                Action::Chat { session_id } => {
                    let session = db
                        .get_session(&session_id)?
                        .ok_or_else(|| anyhow::anyhow!("session {session_id} no longer exists"))?;
                    if engine.is_none() {
                        engine = Some(
                            crate::build_engine(config, &session.model, plugins.clone()).await?,
                        );
                    }
                    let engine = engine.as_mut().expect("engine initialized");
                    if engine.model() != session.model {
                        engine.set_model(&session.model);
                        engine.set_model_pricing(config.model_pricing.get(&session.model).copied());
                    }
                    let system_prompt = context::build_system_prompt_for_model(
                        &session.model,
                        Some(&plugins),
                        &HookTrigger::OnContextBuild,
                        config.is_anthropic(),
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
