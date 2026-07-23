use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{stdout, Stdout};

pub type AppTerminal = Terminal<CrosstermBackend<Stdout>>;

/// Owns the process-wide terminal modes used by the TUI.
///
/// Cleanup is attempted both explicitly and from Drop, so errors, early
/// returns, and panics cannot normally leave the shell in raw/alternate mode.
pub struct TerminalGuard {
    terminal: AppTerminal,
    active: bool,
}

impl TerminalGuard {
    pub fn enter() -> Result<Self> {
        enable_raw_mode()?;
        if let Err(error) = execute!(stdout(), EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error.into());
        }

        let backend = CrosstermBackend::new(stdout());
        let terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = execute!(stdout(), LeaveAlternateScreen);
                let _ = disable_raw_mode();
                return Err(error.into());
            }
        };

        Ok(Self {
            terminal,
            active: true,
        })
    }

    pub fn terminal_mut(&mut self) -> &mut AppTerminal {
        &mut self.terminal
    }

    pub fn restore(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        // Attempt every cleanup operation even if an earlier one fails.
        let cursor_result = self.terminal.show_cursor();
        let raw_result = disable_raw_mode();
        let screen_result = execute!(stdout(), LeaveAlternateScreen);
        self.active = false;

        cursor_result?;
        raw_result?;
        screen_result?;
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}
