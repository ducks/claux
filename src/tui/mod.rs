//! TUI module with screen-based architecture.
//!
//! Each screen (home, chat) is self-contained with its own state, drawing,
//! and key handling. The top-level loop dispatches between screens based
//! on the Action returned by each.

pub mod chat;
pub mod home;
pub mod markdown;
mod screen;
mod ui;

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;

use crate::config::{Config, HookTrigger};
use crate::context;
use crate::db::Db;
use crate::plugin::PluginRegistry;
use crate::query::Engine;
use crate::theme::Theme;

use screen::Action;

/// Run the TUI application.
pub async fn run(mut engine: Engine, _config: &Config, plugins: &PluginRegistry) -> Result<()> {
    // Build system prompt
    let system_prompt = context::build_system_prompt_for_model(
        engine.model(),
        Some(plugins),
        &HookTrigger::OnContextBuild,
    )
    .await?;
    engine.set_system_prompt(system_prompt);

    // Open database
    let db_path = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?
        .join("claux")
        .join("sessions.db");
    let db = Db::open(&db_path)?;

    // Set up terminal
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    let theme = Theme::dark();
    let model = engine.model().to_string();

    // Screen loop: home -> chat -> home -> ...
    let mut next_action = Action::Home;

    loop {
        match next_action {
            Action::Home => {
                let mut home_screen = home::HomeScreen::new(Db::open(&db_path)?, theme, &model);
                next_action = home_screen.run(&mut terminal)?;
            }
            Action::Chat { session_id } => {
                next_action =
                    chat::run(&mut engine, &session_id, &db, &mut terminal, theme, plugins).await?;
            }
            Action::Quit => break,
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;

    println!("{}", engine.cost.format_summary());
    Ok(())
}
