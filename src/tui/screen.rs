//! Screen abstraction for the TUI.
//!
//! Each screen is a self-contained unit with its own state, drawing, and key handling.
//! Screens return an action that the top-level loop uses to decide what to do next.

/// Action returned by any screen to the top-level loop.
#[derive(Debug)]
pub enum Action {
    /// Switch to the chat screen with the given session ID
    Chat { session_id: String },
    /// Return to the home screen
    Home,
    /// Quit the application
    Quit,
}
