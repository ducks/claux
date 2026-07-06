//! Chat screen: conversation with the LLM.
//!
//! This is the main interaction screen. Extracted from the original tui/mod.rs.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::Stdout;
use std::sync::{Arc, Mutex};

use crate::commands::{self, CommandResult};
use crate::db::Db;
use crate::permissions::PermissionResponse;
use crate::plugin::PluginRegistry;
use crate::query::{Engine, SteeringQueue, StreamEvent};
use crate::theme::{Theme, ThemeName};

use super::screen::Action;
use super::ui;

/// A displayed message in the chat.
#[derive(Debug, Clone)]
pub enum ChatMessage {
    /// User, assistant, system, or error text message.
    Text { role: String, content: String },
    /// A tool invocation with its result status.
    Tool {
        name: String,
        summary: String,
        status: ToolStatus,
    },
}

/// Status of a tool invocation in the UI.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolStatus {
    Running,
    Success,
    Error,
}

/// What the chat screen is doing.
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Input,
    Streaming,
    Permission,
}

/// Chat screen state.
pub struct ChatApp {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub cursor: usize,
    pub scroll: u16,
    pub manual_scroll: bool,
    pub mode: Mode,
    pub stream_buffer: String,
    pub status: String,
    pub permission_prompt: Option<String>,
    pub permission_details: Option<Vec<String>>,
    pub should_exit: bool,
    pub should_go_home: bool,
    pub model: String,
    pub total_lines: u16,
    pub thinking: bool,
    pub theme: Theme,
    /// Text being typed while a turn runs, before Enter queues it as a
    /// steering message. Behind Arc<Mutex> because the during-tool key
    /// watcher runs on a separate task.
    pub steer_buf: Arc<Mutex<String>>,
    /// Double-press state for Ctrl+C so one stray press can't kill the app.
    pub ctrl_c: crate::utils::CtrlCArm,
    /// Bumped whenever `messages` changes; keys the rendered-line cache in
    /// ui::draw_chat so history isn't re-rendered on every frame.
    pub messages_rev: u64,
    /// Cached rendering of `messages` (see ui::draw_chat). None until the
    /// first draw or after invalidation.
    pub history_cache: Option<super::ui::HistoryCache>,
    /// Displayed in the header. Defaults to CARGO_PKG_VERSION; overridable
    /// so snapshot tests can pin it to a stable value across version bumps.
    pub version: String,
}

impl ChatApp {
    pub fn new(model: &str, theme: Theme) -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            manual_scroll: false,
            mode: Mode::Input,
            stream_buffer: String::new(),
            status: String::new(),
            permission_prompt: None,
            permission_details: None,
            should_exit: false,
            should_go_home: false,
            model: model.to_string(),
            total_lines: 0,
            thinking: false,
            theme,
            steer_buf: Arc::new(Mutex::new(String::new())),
            ctrl_c: crate::utils::CtrlCArm::default(),
            messages_rev: 0,
            history_cache: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages_rev += 1;
        self.messages.push(ChatMessage::Text {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn add_tool(&mut self, name: &str, summary: &str, status: ToolStatus) {
        // A tool arriving means the model has responded; without this, a
        // turn that opens with tool calls (no text) leaves the "thinking"
        // spinner running under tool output and permission prompts.
        self.thinking = false;
        self.messages_rev += 1;
        self.messages.push(ChatMessage::Tool {
            name: name.to_string(),
            summary: summary.to_string(),
            status,
        });
    }

    /// Update the last tool message's status (e.g., from Running to Success/Error).
    pub fn update_last_tool_status(&mut self, new_status: ToolStatus) {
        self.messages_rev += 1;
        if let Some(ChatMessage::Tool { status, .. }) = self.messages.last_mut() {
            *status = new_status;
        }
    }

    /// Update a specific tool message's status by index. Tool results can
    /// arrive for bubbles other than the last (parallel read-only tools).
    pub fn set_tool_status_at(&mut self, idx: usize, new_status: ToolStatus) {
        self.messages_rev += 1;
        if let Some(ChatMessage::Tool { status, .. }) = self.messages.get_mut(idx) {
            *status = new_status;
        }
    }

    pub fn set_theme(&mut self, theme_name: ThemeName) {
        self.theme = Theme::from_name(theme_name);
        // Cached lines bake in theme colors
        self.messages_rev += 1;
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            Mode::Input => self.handle_input_key(key),
            Mode::Permission | Mode::Streaming => {}
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        // Any key other than Ctrl+C stands down a pending exit confirmation.
        let is_ctrl_c = key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c');
        if !is_ctrl_c && self.ctrl_c.is_armed() {
            self.ctrl_c.disarm();
            self.status = format!("{} | /help for commands", self.model);
        }

        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
                if self.ctrl_c.press() {
                    self.should_exit = true;
                } else {
                    self.status = "Press Ctrl+C again to exit".to_string();
                }
            }
            (KeyModifiers::CONTROL, KeyCode::Char('d')) => {
                self.should_exit = true;
            }
            (_, KeyCode::Enter) => {
                // Submit handled by caller
            }
            (_, KeyCode::Backspace) if self.cursor > 0 => {
                self.cursor -= 1;
                self.input.remove(self.cursor);
            }
            (_, KeyCode::Delete) if self.cursor < self.input.len() => {
                self.input.remove(self.cursor);
            }
            (_, KeyCode::Left) if self.cursor > 0 => {
                self.cursor -= 1;
            }
            (_, KeyCode::Right) if self.cursor < self.input.len() => {
                self.cursor += 1;
            }
            (_, KeyCode::Home) | (KeyModifiers::CONTROL, KeyCode::Char('a')) => {
                self.cursor = 0;
            }
            (_, KeyCode::End) | (KeyModifiers::CONTROL, KeyCode::Char('e')) => {
                self.cursor = self.input.len();
            }
            (KeyModifiers::CONTROL, KeyCode::Char('u')) => {
                self.input.clear();
                self.cursor = 0;
            }
            (_, KeyCode::Up) => {
                self.scroll = self.scroll.saturating_add(3);
                self.manual_scroll = true;
            }
            (_, KeyCode::Down) => {
                self.scroll = self.scroll.saturating_sub(3);
                if self.scroll == 0 {
                    self.manual_scroll = false;
                }
            }
            (_, KeyCode::Char(c)) => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
            }
            _ => {}
        }
    }

    pub fn take_input(&mut self) -> Option<String> {
        if self.input.trim().is_empty() {
            return None;
        }
        let input = self.input.clone();
        self.input.clear();
        self.cursor = 0;
        Some(input)
    }
}

/// Run the chat screen. Returns an Action when the user exits or goes home.
pub async fn run(
    engine: &mut Engine,
    session_id: &str,
    db: &Db,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    theme: Theme,
    _plugins: &PluginRegistry,
) -> Result<Action> {
    // Clear engine state and load this session's messages. repair_history
    // makes old or crash-interrupted saves API-valid (tool_use/tool_result
    // pairing) before the engine sends them anywhere.
    engine.messages_mut().clear();
    let existing_messages = crate::session::repair_history(db.get_messages(session_id)?);
    for msg in &existing_messages {
        engine.messages_mut().push(msg.clone());
    }

    let mut app = ChatApp::new(engine.model(), theme);
    app.status = format!("{} | /help for commands", engine.model());

    // Show existing messages in the UI
    for msg in &existing_messages {
        let role = &msg.role;
        let content = match &msg.content {
            crate::api::types::MessageContent::Text(t) => t.clone(),
            crate::api::types::MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    crate::api::ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };
        app.add_message(role, &content);
    }

    let mut needs_redraw = true;
    let mut pending_submit: Option<String> = None;

    loop {
        if needs_redraw {
            terminal.draw(|f| ui::draw_chat(f, &mut app))?;
            needs_redraw = false;
        }

        // Process pending submit
        if let Some(input) = pending_submit.take() {
            let trimmed = input.trim().to_string();

            // Check for /home command
            if trimmed == "/home" {
                return Ok(Action::Home);
            }

            // Check slash commands
            if let Some(result) = commands::parse_command(&trimmed) {
                match result {
                    CommandResult::Text(ref text) if text == "__cost__" => {
                        app.add_message("system", &commands::format_cost(engine));
                    }
                    CommandResult::Text(text) => {
                        app.add_message("system", &text);
                    }
                    CommandResult::Exit => {
                        return Ok(Action::Home);
                    }
                    CommandResult::Async(async_cmd) => match async_cmd {
                        commands::AsyncCommand::Theme(theme_name) => match theme_name {
                            Some(name) => {
                                let theme = match name.to_lowercase().as_str() {
                                    "dark" => ThemeName::Dark,
                                    "light" => ThemeName::Light,
                                    "ansi" => ThemeName::Ansi,
                                    "dracula" => ThemeName::Dracula,
                                    "nord" => ThemeName::Nord,
                                    "catppuccin" => ThemeName::Catppuccin,
                                    _ => {
                                        app.add_message("error", &format!(
                                                    "Unknown theme: {name}. Available: dark, light, ansi, dracula, nord, catppuccin"
                                                ));
                                        continue;
                                    }
                                };
                                app.set_theme(theme);
                                app.add_message("system", &format!("Theme set to: {name}"));
                            }
                            None => {
                                app.add_message("system",
                                            "Available themes: dark, light, ansi, dracula, nord, catppuccin\n\
                                             Use /theme <name> to switch.");
                            }
                        },
                        _ => {
                            match commands::execute_async(async_cmd, engine).await {
                                Ok(output) => app.add_message("system", &output),
                                Err(e) => app.add_message("error", &format!("Error: {e}")),
                            }
                            // Commands like /compact rewrite engine history
                            let _ = db.replace_messages(session_id, engine.messages());
                        }
                    },
                }
                app.scroll = 0;
                app.manual_scroll = false;
                needs_redraw = true;
                continue;
            }

            // Regular message -- start streaming
            app.add_message("user", &trimmed);
            app.mode = Mode::Streaming;
            app.stream_buffer.clear();
            app.thinking = true;
            app.scroll = 0;
            app.manual_scroll = false;

            app.status = format!("{} | thinking...", app.model);

            let submit_result =
                drive_streaming(engine, &trimmed, &mut app, terminal, &mut CrosstermKeys).await;

            if let Err(e) = submit_result {
                app.add_message("error", &format!("Error: {e}"));
            }

            // Snapshot the full conversation, tool rounds included, even on
            // error: the engine may have made progress worth keeping.
            // Previously only the user message and the final assistant
            // message were saved, so resumed sessions lost everything the
            // turn actually did.
            if let Err(e) = db.replace_messages(session_id, engine.messages()) {
                tracing::warn!("Failed to save session: {e}");
            }

            app.mode = Mode::Input;
            app.status = format!("{} | {}", engine.model(), engine.cost.format_summary());
            app.scroll = 0;
            app.manual_scroll = false;
            needs_redraw = true;
            continue;
        }

        // Poll terminal events
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Enter && app.mode == Mode::Input {
                    if let Some(input) = app.take_input() {
                        pending_submit = Some(input);
                    }
                } else {
                    app.handle_key(key);
                }
                needs_redraw = true;
            }
        }

        if app.should_exit {
            return Ok(Action::Quit);
        }
        if app.should_go_home {
            return Ok(Action::Home);
        }
    }
}

/// Drive one turn through the engine's event protocol.
///
/// The TUI no longer duplicates the turn loop: engine::run_turn owns the
/// conversation (assistant blocks, tool execution, steering, interrupt
/// pairing), and this function is a select over the submit future, the
/// event stream, and a 50ms draw/key tick. Permission prompts arrive as
/// events carrying the tool input and a oneshot responder.
async fn drive_streaming<B: ratatui::backend::Backend>(
    engine: &mut Engine,
    input: &str,
    app: &mut ChatApp,
    terminal: &mut Terminal<B>,
    keys: &mut dyn KeySource,
) -> Result<()> {
    let steering = engine.steering_queue();
    let steer_buf = app.steer_buf.clone();
    let cancel = tokio_util::sync::CancellationToken::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);

    let mut submit_result: Option<Result<()>> = None;
    {
        let submit_fut = engine.submit_streaming(input, tx, cancel.clone());
        tokio::pin!(submit_fut);

        // Tool bubbles awaiting results, oldest first: indices into
        // app.messages, matched FIFO with ToolResult events (the engine
        // emits results in tool_use order).
        let mut running_tools: std::collections::VecDeque<usize> =
            std::collections::VecDeque::new();

        // First tick after one period, not immediately: keys and draws
        // shouldn't race the event stream at t=0.
        let mut tick = tokio::time::interval_at(
            tokio::time::Instant::now() + std::time::Duration::from_millis(50),
            std::time::Duration::from_millis(50),
        );
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                res = &mut submit_fut, if submit_result.is_none() => {
                    submit_result = Some(res);
                }
                event = rx.recv() => {
                    let Some(event) = event else {
                        break; // tx dropped: turn over, events drained
                    };
                    match event {
                        StreamEvent::Text(t) => {
                            // Drawn by the tick, which coalesces chunks
                            app.stream_buffer.push_str(&t);
                            app.thinking = false;
                        }
                        StreamEvent::Notice(n) => {
                            flush_stream_buffer(app);
                            app.add_message("system", &n);
                        }
                        StreamEvent::SteeringSent(t) => {
                            flush_stream_buffer(app);
                            app.add_message("user", &t);
                        }
                        StreamEvent::ToolStart { name, summary, .. } => {
                            flush_stream_buffer(app);
                            app.add_tool(&name, &summary, ToolStatus::Running);
                            running_tools.push_back(app.messages.len() - 1);
                            terminal.draw(|f| ui::draw_chat(f, app))?;
                        }
                        StreamEvent::ToolResult { is_error, .. } => {
                            if let Some(idx) = running_tools.pop_front() {
                                app.set_tool_status_at(
                                    idx,
                                    if is_error { ToolStatus::Error } else { ToolStatus::Success },
                                );
                            }
                            terminal.draw(|f| ui::draw_chat(f, app))?;
                        }
                        StreamEvent::PermissionRequest { tool_name, summary, input, respond }
                        | StreamEvent::PermissionRequestWithDiff { tool_name, summary, input, respond, .. } => {
                            let response = prompt_permission_tui(
                                app,
                                terminal,
                                keys,

                                &tool_name,
                                &summary,
                                &input,
                                &steering,
                            )
                            .await?;
                            // Denying outright ends the turn (the engine
                            // pairs the rest of the batch as interrupted);
                            // DenyAndCancel instead queued steering, which
                            // the engine delivers immediately.
                            let end_turn = matches!(response, PermissionResponse::Deny);
                            let _ = respond.send(response);
                            if end_turn {
                                cancel.cancel();
                            }
                        }
                        StreamEvent::Interrupted => {
                            flush_stream_buffer(app);
                            while let Some(idx) = running_tools.pop_front() {
                                app.set_tool_status_at(idx, ToolStatus::Error);
                            }
                            app.add_message("system", "Interrupted by user.");
                        }
                        StreamEvent::Error(_) => {
                            // Surfaced through submit_result by the caller
                        }
                        StreamEvent::Done => {
                            flush_stream_buffer(app);
                        }
                    }
                }
                _ = tick.tick() => {
                    if poll_stream_key(keys, &steer_buf, &steering)? {
                        cancel.cancel();
                    }
                    // The tick is the only draw during streaming: it
                    // coalesces all chunks since the last frame, animates
                    // the spinner, and keeps the steering buffer visible.
                    terminal.draw(|f| ui::draw_chat(f, app))?;
                }
            }
        }
    }

    submit_result.unwrap_or(Ok(()))
}

/// Move any streamed-but-unflushed assistant text into a message bubble.
fn flush_stream_buffer(app: &mut ChatApp) {
    if !app.stream_buffer.is_empty() {
        let content = app.stream_buffer.clone();
        app.stream_buffer.clear();
        app.add_message("assistant", &content);
    }
}

/// Full-screen permission prompt. Returns the user's decision; typing a
/// message and pressing Enter queues it as steering and denies the tool
/// (the engine then skips the rest of the batch and delivers the message).
async fn prompt_permission_tui<B: ratatui::backend::Backend>(
    app: &mut ChatApp,
    terminal: &mut Terminal<B>,
    keys: &mut dyn KeySource,

    tool_name: &str,
    summary: &str,
    input: &serde_json::Value,
    steering: &SteeringQueue,
) -> Result<PermissionResponse> {
    app.permission_prompt = Some(summary.to_string());
    app.permission_details = Some(format_permission_details(tool_name, input));
    app.mode = Mode::Permission;
    terminal.draw(|f| ui::draw_chat(f, app))?;

    let mut perm_input = String::new();

    let response = loop {
        match keys.poll_key()? {
            None => {
                // Nothing typed: yield to the runtime before polling again
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Some(key) => {
                match key.code {
                    KeyCode::Char('y') | KeyCode::Enter if perm_input.is_empty() => {
                        break PermissionResponse::Allow;
                    }
                    KeyCode::Char('a') if perm_input.is_empty() => {
                        break PermissionResponse::AlwaysAllow;
                    }
                    KeyCode::Char('n') if perm_input.is_empty() => {
                        break PermissionResponse::Deny;
                    }
                    KeyCode::Esc if perm_input.is_empty() => {
                        break PermissionResponse::Deny;
                    }
                    KeyCode::Enter if !perm_input.is_empty() => {
                        // User typed a message: deny this tool and inject
                        // the message as steering. The engine delivers it
                        // right after the skipped batch.
                        steering
                            .lock()
                            .expect("steering queue poisoned")
                            .push_back(perm_input.clone());
                        break PermissionResponse::DenyAndCancel;
                    }
                    KeyCode::Backspace if !perm_input.is_empty() => {
                        perm_input.pop();
                    }
                    KeyCode::Char(c) => {
                        perm_input.push(c);
                    }
                    _ => {}
                }
                // Redraw so the typed text stays visible
                app.status = format!("{} | deny and message: {perm_input}", app.model);
                terminal.draw(|f| ui::draw_chat(f, app))?;
            }
        }
    };

    app.permission_prompt = None;
    app.permission_details = None;
    app.mode = Mode::Streaming;
    app.status = app.model.clone();
    Ok(response)
}

/// Key input for the streaming turn path. Production reads the real
/// terminal via crossterm; tests feed scripted keystrokes, which is what
/// makes the interactive flow (steering, permission prompts, Ctrl+C)
/// coverable by `cargo test`.
pub trait KeySource {
    fn poll_key(&mut self) -> Result<Option<KeyEvent>>;
}

/// Reads keys from the real terminal without blocking.
pub struct CrosstermKeys;

impl KeySource for CrosstermKeys {
    fn poll_key(&mut self) -> Result<Option<KeyEvent>> {
        if event::poll(std::time::Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                return Ok(Some(key));
            }
        }
        Ok(None)
    }
}

/// Non-blocking key poll while a turn is running. Ctrl+C cancels (returns
/// true). Everything else builds the steering buffer: printable characters
/// append, Backspace deletes, and Enter moves the buffer into the engine's
/// steering queue, which the turn loop drains before its next API call.
fn poll_stream_key(
    keys: &mut dyn KeySource,
    steer_buf: &Arc<Mutex<String>>,
    steering: &SteeringQueue,
) -> Result<bool> {
    {
        if let Some(key) = keys.poll_key()? {
            match (key.modifiers, key.code) {
                (KeyModifiers::CONTROL, KeyCode::Char('c')) => return Ok(true),
                (_, KeyCode::Enter) => {
                    let text = {
                        let mut buf = steer_buf.lock().expect("steer buffer poisoned");
                        std::mem::take(&mut *buf)
                    };
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        steering
                            .lock()
                            .expect("steering queue poisoned")
                            .push_back(text);
                    }
                }
                (_, KeyCode::Backspace) => {
                    steer_buf.lock().expect("steer buffer poisoned").pop();
                }
                (m, KeyCode::Char(c)) if m.is_empty() || m == KeyModifiers::SHIFT => {
                    steer_buf.lock().expect("steer buffer poisoned").push(c);
                }
                _ => {}
            }
        }
    }
    Ok(false)
}

fn format_permission_details(tool_name: &str, input: &serde_json::Value) -> Vec<String> {
    let mut lines = Vec::new();

    match tool_name {
        "Bash" => {
            if let Some(cmd) = input["command"].as_str() {
                lines.push("Command:".to_string());
                for line in cmd.lines() {
                    lines.push(format!("  {line}"));
                }
            }
        }
        "Write" => {
            if let Some(path) = input["file_path"].as_str() {
                lines.push(format!("File: {path}"));
            }
            if let Some(content) = input["content"].as_str() {
                let preview: Vec<&str> = content.lines().take(10).collect();
                lines.push("Content:".to_string());
                for line in &preview {
                    lines.push(format!("  {line}"));
                }
                let total = content.lines().count();
                if total > 10 {
                    lines.push(format!("  ... ({} more lines)", total - 10));
                }
            }
        }
        "Edit" => {
            if let Some(path) = input["file_path"].as_str() {
                lines.push(format!("File: {path}"));
            }
            if let Some(old) = input["old_string"].as_str() {
                lines.push("Replace:".to_string());
                for line in old.lines().take(5) {
                    lines.push(format!("  - {line}"));
                }
            }
            if let Some(new) = input["new_string"].as_str() {
                lines.push("With:".to_string());
                for line in new.lines().take(5) {
                    lines.push(format!("  + {line}"));
                }
            }
        }
        "Agent" => {
            if let Some(prompt) = input["prompt"].as_str() {
                lines.push("Task:".to_string());
                for line in prompt.lines().take(5) {
                    lines.push(format!("  {line}"));
                }
            }
        }
        "Read" => {
            if let Some(path) = input["file_path"].as_str() {
                lines.push(format!("File: {path}"));
            }
        }
        "Grep" => {
            if let Some(pattern) = input["pattern"].as_str() {
                lines.push(format!("Pattern: {pattern}"));
            }
            if let Some(path) = input["path"].as_str() {
                lines.push(format!("In: {path}"));
            }
        }
        "WebFetch" => {
            if let Some(url) = input["url"].as_str() {
                lines.push(format!("URL: {url}"));
            }
        }
        _ => {
            let json_str = serde_json::to_string_pretty(input).unwrap_or_default();
            for line in json_str.lines().take(8) {
                lines.push(format!("  {line}"));
            }
        }
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;

    fn test_app() -> ChatApp {
        ChatApp::new("test-model", Theme::dark())
    }

    fn ctrl_c_key() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    #[test]
    fn single_ctrl_c_warns_double_exits() {
        let mut app = test_app();
        app.handle_key(ctrl_c_key());
        assert!(!app.should_exit, "first Ctrl+C must not exit");
        assert!(app.status.contains("again"), "status should show the hint");
        app.handle_key(ctrl_c_key());
        assert!(app.should_exit, "second Ctrl+C must exit");
    }

    #[test]
    fn typing_disarms_pending_ctrl_c() {
        let mut app = test_app();
        app.handle_key(ctrl_c_key());
        app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(!app.status.contains("again"), "hint should clear");
        app.handle_key(ctrl_c_key());
        assert!(
            !app.should_exit,
            "Ctrl+C after typing re-arms instead of exiting"
        );
    }

    #[test]
    fn ctrl_d_still_exits_immediately() {
        let mut app = test_app();
        app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL));
        assert!(app.should_exit);
    }

    #[test]
    fn add_message_creates_text_variant() {
        let mut app = test_app();
        app.add_message("user", "hello");
        assert_eq!(app.messages.len(), 1);
        match &app.messages[0] {
            ChatMessage::Text { role, content } => {
                assert_eq!(role, "user");
                assert_eq!(content, "hello");
            }
            _ => panic!("expected Text variant"),
        }
    }

    #[test]
    fn add_tool_creates_tool_variant() {
        let mut app = test_app();
        app.add_tool("Bash", "cargo build", ToolStatus::Running);
        assert_eq!(app.messages.len(), 1);
        match &app.messages[0] {
            ChatMessage::Tool {
                name,
                summary,
                status,
            } => {
                assert_eq!(name, "Bash");
                assert_eq!(summary, "cargo build");
                assert_eq!(*status, ToolStatus::Running);
            }
            _ => panic!("expected Tool variant"),
        }
    }

    #[test]
    fn update_last_tool_status_changes_running_to_success() {
        let mut app = test_app();
        app.add_tool("Read", "/some/file", ToolStatus::Running);
        app.update_last_tool_status(ToolStatus::Success);
        match &app.messages[0] {
            ChatMessage::Tool { status, .. } => assert_eq!(*status, ToolStatus::Success),
            _ => panic!("expected Tool variant"),
        }
    }

    #[test]
    fn update_last_tool_status_changes_running_to_error() {
        let mut app = test_app();
        app.add_tool("Bash", "failing command", ToolStatus::Running);
        app.update_last_tool_status(ToolStatus::Error);
        match &app.messages[0] {
            ChatMessage::Tool { status, .. } => assert_eq!(*status, ToolStatus::Error),
            _ => panic!("expected Tool variant"),
        }
    }

    #[test]
    fn update_last_tool_status_ignores_text_messages() {
        let mut app = test_app();
        app.add_message("assistant", "some text");
        // Should not panic — just a no-op
        app.update_last_tool_status(ToolStatus::Success);
        match &app.messages[0] {
            ChatMessage::Text { content, .. } => assert_eq!(content, "some text"),
            _ => panic!("expected Text variant"),
        }
    }

    #[test]
    fn update_last_tool_status_targets_last_message_only() {
        let mut app = test_app();
        app.add_tool("Read", "first tool", ToolStatus::Success);
        app.add_tool("Bash", "second tool", ToolStatus::Running);
        app.update_last_tool_status(ToolStatus::Error);
        // First tool unchanged
        match &app.messages[0] {
            ChatMessage::Tool { status, .. } => assert_eq!(*status, ToolStatus::Success),
            _ => panic!("expected Tool variant"),
        }
        // Second tool updated
        match &app.messages[1] {
            ChatMessage::Tool { status, .. } => assert_eq!(*status, ToolStatus::Error),
            _ => panic!("expected Tool variant"),
        }
    }

    #[test]
    fn mixed_messages_preserve_order() {
        let mut app = test_app();
        app.add_message("user", "do something");
        app.add_tool("Bash", "ls", ToolStatus::Running);
        app.update_last_tool_status(ToolStatus::Success);
        app.add_message("assistant", "done");

        assert_eq!(app.messages.len(), 3);
        assert!(matches!(&app.messages[0], ChatMessage::Text { role, .. } if role == "user"));
        assert!(matches!(
            &app.messages[1],
            ChatMessage::Tool {
                status: ToolStatus::Success,
                ..
            }
        ));
        assert!(matches!(&app.messages[2], ChatMessage::Text { role, .. } if role == "assistant"));
    }
}

/// End-to-end interactive turn tests: a scripted provider drives the
/// engine, scripted keystrokes drive the UI, and a ratatui TestBackend
/// captures what would have been drawn. This is the automated coverage
/// for flows that previously only a human at a keyboard could exercise.
#[cfg(test)]
mod turn_tests {
    use super::*;
    use crate::permissions::PermissionMode;
    use crate::test_support::{scripted_engine, tool_use};
    use ratatui::backend::TestBackend;

    struct ScriptedKeys(std::collections::VecDeque<KeyEvent>);

    impl KeySource for ScriptedKeys {
        fn poll_key(&mut self) -> Result<Option<KeyEvent>> {
            Ok(self.0.pop_front())
        }
    }

    fn ch(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn enter() -> KeyEvent {
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
    }

    fn ctrl_c() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    /// Run one full turn through drive_streaming with scripted keys.
    async fn run_turn(
        engine: &mut Engine,
        keystrokes: Vec<KeyEvent>,
    ) -> (ChatApp, Terminal<TestBackend>) {
        let mut app = ChatApp::new("test-model", Theme::dark());
        app.mode = Mode::Streaming;
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut keys = ScriptedKeys(keystrokes.into());

        drive_streaming(engine, "go", &mut app, &mut terminal, &mut keys)
            .await
            .unwrap();

        // Final frame with the settled state
        app.mode = Mode::Input;
        terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
        (app, terminal)
    }

    fn tool_statuses(app: &ChatApp) -> Vec<ToolStatus> {
        app.messages
            .iter()
            .filter_map(|m| match m {
                ChatMessage::Tool { status, .. } => Some(status.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn turn_renders_text_and_tool_result() {
        let mut engine = scripted_engine(
            vec![tool_use(
                "tu_1",
                "Glob",
                serde_json::json!({"pattern": "*.zz"}),
            )],
            None,
            PermissionMode::Bypass,
        );

        let (app, terminal) = run_turn(&mut engine, vec![]).await;

        assert_eq!(tool_statuses(&app), vec![ToolStatus::Success]);
        let screen = buffer_text(&terminal);
        assert!(screen.contains("working on it"), "assistant text rendered");
        assert!(screen.contains("Glob"), "tool bubble rendered");
        // Engine side: user, assistant(text+tool_use), tool results
        assert_eq!(engine.messages().len(), 3);
    }

    #[tokio::test]
    async fn permission_y_allows_the_tool() {
        let mut engine = scripted_engine(
            vec![tool_use(
                "tu_1",
                "Bash",
                serde_json::json!({"command": "echo approved-ok"}),
            )],
            None,
            PermissionMode::Default, // Bash asks for confirmation
        );

        let (app, _terminal) = run_turn(&mut engine, vec![ch('y')]).await;

        assert_eq!(tool_statuses(&app), vec![ToolStatus::Success]);
        let crate::api::MessageContent::Blocks(blocks) = &engine.messages()[2].content else {
            panic!("expected tool results");
        };
        let crate::api::ContentBlock::ToolResult { content, .. } = &blocks[0] else {
            panic!("expected ToolResult");
        };
        assert!(
            content.contains("approved-ok"),
            "tool actually ran: {content}"
        );
    }

    #[tokio::test]
    async fn permission_n_denies_and_ends_turn() {
        let mut engine = scripted_engine(
            vec![tool_use(
                "tu_1",
                "Bash",
                serde_json::json!({"command": "echo never-runs"}),
            )],
            None,
            PermissionMode::Default,
        );

        let (app, _terminal) = run_turn(&mut engine, vec![ch('n')]).await;

        assert!(
            app.messages.iter().any(|m| matches!(
                m,
                ChatMessage::Text { role, content } if role == "system" && content.contains("Interrupted")
            )),
            "denying ends the turn: {:?}",
            app.messages
        );
        let crate::api::MessageContent::Blocks(blocks) = &engine.messages()[2].content else {
            panic!("expected tool results");
        };
        let crate::api::ContentBlock::ToolResult { content, .. } = &blocks[0] else {
            panic!("expected ToolResult");
        };
        assert!(content.contains("denied"), "tool denied: {content}");
    }

    #[tokio::test]
    async fn permission_typed_message_becomes_steering() {
        let mut engine = scripted_engine(
            vec![tool_use(
                "tu_1",
                "Bash",
                serde_json::json!({"command": "echo never-runs"}),
            )],
            None,
            PermissionMode::Default,
        );

        let (app, _terminal) =
            run_turn(&mut engine, vec![ch('f'), ch('i'), ch('x'), enter()]).await;

        // The typed message reached the model as a user message
        let steer_delivered = engine.messages().iter().any(|m| {
            matches!(&m.content, crate::api::MessageContent::Text(t) if t == "fix")
                && m.role == "user"
        });
        assert!(
            steer_delivered,
            "typed message injected: {:?}",
            engine.messages()
        );
        // And the UI shows it as a user bubble
        assert!(app.messages.iter().any(|m| matches!(
            m,
            ChatMessage::Text { role, content } if role == "user" && content == "fix"
        )));
    }

    #[tokio::test]
    async fn ctrl_c_cancels_a_running_tool() {
        let mut engine = scripted_engine(
            vec![tool_use(
                "tu_1",
                "Bash",
                serde_json::json!({"command": "sleep 5"}),
            )],
            None,
            PermissionMode::Bypass,
        );

        let start = std::time::Instant::now();
        let (app, _terminal) = run_turn(&mut engine, vec![ctrl_c()]).await;
        assert!(
            start.elapsed() < std::time::Duration::from_secs(3),
            "Ctrl+C should cut the tool short (took {:?})",
            start.elapsed()
        );
        assert!(app.messages.iter().any(|m| matches!(
            m,
            ChatMessage::Text { role, content } if role == "system" && content.contains("Interrupted")
        )));
    }

    #[tokio::test]
    async fn typing_mid_tool_steers_and_preempts() {
        let mut engine = scripted_engine(
            vec![tool_use(
                "tu_1",
                "Bash",
                serde_json::json!({"command": "sleep 5"}),
            )],
            None,
            PermissionMode::Bypass,
        );

        let start = std::time::Instant::now();
        let (_app, _terminal) = run_turn(&mut engine, vec![ch('n'), ch('o'), enter()]).await;
        assert!(
            start.elapsed() < std::time::Duration::from_secs(3),
            "steering should preempt the tool (took {:?})",
            start.elapsed()
        );

        let steer_delivered = engine.messages().iter().any(|m| {
            matches!(&m.content, crate::api::MessageContent::Text(t) if t == "no")
                && m.role == "user"
        });
        assert!(
            steer_delivered,
            "steering delivered: {:?}",
            engine.messages()
        );
    }
}

#[cfg(test)]
mod tuishot_shots {
    use super::*;
    use tuishot::Tuishot;

    /// Version string used in snapshot renders. Pinning this keeps version
    /// bumps from drifting every screenshot on every release.
    const SNAPSHOT_VERSION: &str = "TEST";

    fn sample_conversation() -> ChatApp {
        let theme = crate::theme::Theme::dark();
        let mut app = ChatApp::new("claude-sonnet-4-20250514", theme);
        app.version = SNAPSHOT_VERSION.to_string();

        app.add_message("user", "Can you read src/main.rs and explain what it does?");
        app.add_tool("Read", "src/main.rs (42 lines)", ToolStatus::Success);
        app.add_message(
            "assistant",
            "This is the entry point for **claux**. It parses CLI arguments via `clap`, \
             loads configuration from `~/.config/claux/config.toml`, and dispatches to \
             either the REPL or one-shot mode depending on the flags.\n\n\
             Key things:\n\
             - `--tui` launches the full-screen Ratatui interface\n\
             - `--resume <id>` reloads a previous session\n\
             - `-p <prompt>` runs a single query and exits",
        );

        app.status = "1.2k tokens".to_string();
        app
    }

    #[derive(Tuishot)]
    enum ChatShot {
        #[tuishot(
            name = "chat-conversation",
            description = "Mid-conversation with tool use and markdown"
        )]
        Conversation,

        #[tuishot(
            name = "chat-streaming",
            description = "Assistant mid-response with streaming cursor"
        )]
        Streaming,

        #[tuishot(
            name = "chat-permission",
            description = "Prompting for Bash permission"
        )]
        Permission,

        #[tuishot(name = "chat-empty", description = "Fresh chat, no messages")]
        Empty,
    }

    impl ChatShotRender for ChatShot {
        fn render(&self, buf: &mut ratatui::buffer::Buffer, area: ratatui::layout::Rect) {
            let theme = crate::theme::Theme::dark();
            let mut app = match self {
                ChatShot::Conversation => sample_conversation(),
                ChatShot::Streaming => {
                    let mut app = sample_conversation();
                    app.mode = Mode::Streaming;
                    app.stream_buffer = "Sure, let me look at the configuration handling next. \
                        The config module uses `toml` for parsing and supports both global \
                        and per-project overrides"
                        .to_string();
                    app.thinking = false;
                    app
                }
                ChatShot::Permission => {
                    let mut app = sample_conversation();
                    app.mode = Mode::Permission;
                    app.permission_prompt = Some("Allow Bash command?".to_string());
                    app.permission_details = Some(vec![
                        "Command:".to_string(),
                        "  cargo test --lib".to_string(),
                        "".to_string(),
                        "Working directory: /home/user/dev/claux".to_string(),
                    ]);
                    app
                }
                ChatShot::Empty => {
                    let mut app = ChatApp::new("claude-sonnet-4-20250514", theme);
                    app.version = SNAPSHOT_VERSION.to_string();
                    app
                }
            };
            let rendered = tuishot::render_to_buffer(area.width, area.height, |f| {
                ui::draw_chat(f, &mut app);
            });
            buf.clone_from(&rendered);
        }
    }

    #[test]
    fn capture_chat_screens() {
        ChatShot::check_all().expect("chat screen capture");
    }
}
