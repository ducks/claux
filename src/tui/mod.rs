//! TUI module with screen-based architecture.
//!
//! Each screen (home, chat) is self-contained with its own state, drawing,
//! and key handling. The top-level loop dispatches between screens based
//! on the Action returned by each.

pub mod chat;
pub mod home;
pub mod markdown;
mod screen;
mod terminal;
mod ui;

use anyhow::Result;

use crate::config::{Config, HookTrigger};
use crate::context;
use crate::db::Db;
use crate::plugin::PluginRegistry;
use crate::query::Engine;
use crate::theme::Theme;

use screen::Action;
use terminal::TerminalGuard;

/// Run the TUI application.
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

    // Open database
    let db_path = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?
        .join("claux")
        .join("sessions.db");
    let db = Db::open(&db_path)?;

    let mut terminal_guard = TerminalGuard::enter()?;

    let theme = Theme::dark();
    let model = engine.model().to_string();

    let app_result: Result<()> = async {
        // Screen loop: home -> chat -> home -> ...
        let mut next_action = Action::Home;
        loop {
            match next_action {
                Action::Home => {
                    let mut home_screen = home::HomeScreen::new(Db::open(&db_path)?, theme, &model);
                    next_action = home_screen.run(terminal_guard.terminal_mut())?;
                }
                Action::Chat { session_id } => {
                    next_action = chat::run(
                        &mut engine,
                        &session_id,
                        &db,
                        terminal_guard.terminal_mut(),
                        theme,
                        plugins,
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

    println!("{}", engine.cost.format_summary());
    Ok(())
}
