//! Chat screen: conversation with the LLM.
//!
//! This is the main interaction screen. Extracted from the original tui/mod.rs.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::Stdout;
use std::sync::{Arc, Mutex};

use crate::commands::{self, CommandResult};
use crate::config::ResolvedModel;
use crate::db::Db;
use crate::permissions::PermissionResponse;
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
        detail: Option<String>,
        status: ToolStatus,
        output: String,
    },
}

/// Status of a tool invocation in the UI.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolStatus {
    Queued,
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

pub struct Activity {
    pub label: String,
    pub started: std::time::Instant,
    pub updated: std::time::Instant,
}

/// Chat screen state.
pub struct ChatApp {
    pub jobs: Vec<crate::tools::jobs::JobSnapshot>,
    pub show_jobs: bool,
    pub selected_job: usize,
    pub job_scroll: u16,
    pub save_error: Option<String>,
    pub expand_tool_output: bool,
    pub activity: Option<Activity>,
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub cursor: usize,
    /// Slash-command completion. Holds only the selection and any Esc
    /// dismissal; the candidate list is derived from `input` each frame.
    pub completion: super::completion::CompletionState,
    pub scroll: u16,
    pub manual_scroll: bool,
    pub mode: Mode,
    pub stream_buffer: String,
    pub status: String,
    pub permission_prompt: Option<String>,
    pub permission_details: Option<Vec<String>>,
    pub permission_always_label: Option<String>,
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
            jobs: Vec::new(),
            show_jobs: false,
            selected_job: 0,
            job_scroll: 0,
            save_error: None,
            expand_tool_output: false,
            activity: None,
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            completion: Default::default(),
            scroll: 0,
            manual_scroll: false,
            mode: Mode::Input,
            stream_buffer: String::new(),
            status: String::new(),
            permission_prompt: None,
            permission_details: None,
            permission_always_label: None,
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
            role: crate::utils::sanitize_terminal_text(role),
            content: crate::utils::sanitize_terminal_text(content),
        });
    }

    fn refresh_jobs(&mut self, manager: &crate::tools::jobs::JobManager) {
        self.jobs = manager.snapshots();
        self.selected_job = self.selected_job.min(self.jobs.len().saturating_sub(1));
        for job in manager.completions() {
            self.add_message(
                "system",
                &format!(
                    "Background {} {} after {}s: {}\n{}\n/jobs {} to inspect",
                    job.id,
                    job.status.label(),
                    job.elapsed.as_secs(),
                    crate::utils::truncate_str(&job.command, 120),
                    crate::utils::tail_str(&job.output, 600),
                    job.id
                ),
            );
        }
    }

    fn job_key(&mut self, key: KeyEvent, manager: &crate::tools::jobs::JobManager) -> bool {
        if key.kind == event::KeyEventKind::Release {
            return false;
        }
        if key.code == KeyCode::F(6) {
            self.show_jobs = !self.show_jobs;
            return true;
        }
        if !self.show_jobs {
            return false;
        }
        match key.code {
            KeyCode::Esc => self.show_jobs = false,
            KeyCode::Up => {
                self.selected_job = self.selected_job.saturating_sub(1);
                self.job_scroll = 0;
            }
            KeyCode::Down => {
                self.selected_job = (self.selected_job + 1).min(self.jobs.len().saturating_sub(1));
                self.job_scroll = 0;
            }
            KeyCode::PageUp => self.job_scroll = self.job_scroll.saturating_add(10),
            KeyCode::PageDown => self.job_scroll = self.job_scroll.saturating_sub(10),
            KeyCode::Char('x') if key.modifiers.is_empty() => {
                if let Some(job) = self.jobs.get(self.selected_job) {
                    let _ = manager.cancel(&job.id);
                }
            }
            _ if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') => {
                return false
            }
            _ => {}
        }
        true
    }

    fn save_session(&mut self, db: &Db, session_id: &str, engine: &Engine) -> bool {
        self.record_save(db.save_conversation(
            session_id,
            engine.messages(),
            engine.model_binding(),
            engine.archive(),
        ))
    }

    fn record_save(&mut self, result: Result<()>) -> bool {
        match result {
            Ok(()) => {
                self.save_error = None;
                true
            }
            Err(error) => {
                let message = crate::utils::sanitize_terminal_text(&format!("{error:#}"));
                if self.save_error.as_ref() != Some(&message) {
                    self.add_message("error", &format!("Session is UNSAVED: {message}\nKeep this chat open. Ctrl+S retries saving."));
                }
                self.save_error = Some(message);
                false
            }
        }
    }

    pub fn set_activity(&mut self, label: &str) {
        let now = std::time::Instant::now();
        if let Some(activity) = &mut self.activity {
            if activity.label == label {
                activity.updated = now;
                return;
            }
        }
        self.activity = Some(Activity {
            label: crate::utils::sanitize_terminal_text(label),
            started: now,
            updated: now,
        });
    }

    fn refresh_tool_activity(&mut self, indices: &[usize]) {
        let running: Vec<_> = indices
            .iter()
            .filter_map(|&idx| match self.messages.get(idx) {
                Some(ChatMessage::Tool {
                    name,
                    status: ToolStatus::Running,
                    ..
                }) => Some(name.clone()),
                _ => None,
            })
            .collect();
        let queued = indices.iter().any(|&idx| {
            matches!(
                self.messages.get(idx),
                Some(ChatMessage::Tool {
                    status: ToolStatus::Queued,
                    ..
                })
            )
        });
        let label = match running.as_slice() {
            [] if queued => "Tools queued; waiting to start".to_string(),
            [] => "Waiting for next update".to_string(),
            [name] => format!("Running {name}"),
            _ => format!("Running {} tools", running.len()),
        };
        self.set_activity(&label);
    }

    #[cfg(test)]
    pub fn add_tool(&mut self, name: &str, summary: &str, status: ToolStatus) {
        self.add_tool_presentation(name, summary, None, status);
    }

    pub fn add_tool_with_input(
        &mut self,
        name: &str,
        summary: &str,
        input: &serde_json::Value,
        status: ToolStatus,
    ) {
        let presentation = super::tool_display::present(name, summary, input);
        self.add_tool_presentation(
            name,
            &presentation.summary,
            presentation.detail.as_deref(),
            status,
        );
    }

    fn add_tool_presentation(
        &mut self,
        name: &str,
        summary: &str,
        detail: Option<&str>,
        status: ToolStatus,
    ) {
        // A tool arriving means the model has responded; without this, a
        // turn that opens with tool calls (no text) leaves the "thinking"
        // spinner running under tool output and permission prompts.
        self.thinking = false;
        self.messages_rev += 1;
        self.messages.push(ChatMessage::Tool {
            name: crate::utils::sanitize_terminal_text(name),
            summary: crate::utils::sanitize_terminal_text(summary),
            detail: detail.map(crate::utils::sanitize_terminal_text),
            status,
            output: String::new(),
        });
    }

    /// Update the last tool message's status (e.g., from Running to Success/Error).
    #[cfg(test)]
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

    pub fn set_tool_output_at(&mut self, idx: usize, content: &str) {
        if let Some(ChatMessage::Tool { output, .. }) = self.messages.get_mut(idx) {
            let bounded = crate::utils::tail_str(content, 20_000);
            *output = crate::utils::sanitize_terminal_text(bounded);
            if bounded.len() < content.len() {
                output.insert_str(0, "[earlier output omitted]\n");
            }
            self.messages_rev += 1;
        }
    }

    pub fn set_theme(&mut self, theme_name: ThemeName) {
        self.theme = Theme::from_name(theme_name);
        // Cached lines bake in theme colors
        self.messages_rev += 1;
    }

    /// Whether the completion menu is currently showing.
    ///
    /// The event loop needs this because it routes Enter to submission before
    /// `handle_key` ever sees it; without asking, a bare `/` would be sent to
    /// the model instead of accepting the highlighted command.
    pub fn completion_is_open(&mut self) -> bool {
        self.completion.active(&self.input, self.cursor).is_some()
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            Mode::Input => self.handle_input_key(key),
            Mode::Permission | Mode::Streaming => {}
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('o') {
            self.expand_tool_output = !self.expand_tool_output;
            self.messages_rev += 1;
            return;
        }
        // Any key other than Ctrl+C stands down a pending exit confirmation.
        let is_ctrl_c = key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c');
        if !is_ctrl_c && self.ctrl_c.is_armed() {
            self.ctrl_c.disarm();
            self.status = format!("{} | /help for commands", self.model);
        }

        // While the completion menu is showing it claims a few keys that
        // otherwise scroll or submit. Everything else falls through to normal
        // editing, so typing is never trapped in a "completion mode".
        if let Some(active) = self.completion.active(&self.input, self.cursor) {
            match (key.modifiers, key.code) {
                (_, KeyCode::Up) => {
                    self.completion.move_selection(-1, active.matches.len());
                    return;
                }
                (_, KeyCode::Down) => {
                    self.completion.move_selection(1, active.matches.len());
                    return;
                }
                (_, KeyCode::Tab) | (_, KeyCode::Enter) => {
                    let (input, cursor) =
                        super::completion::apply(&self.input, self.cursor, active.selected_spec());
                    self.input = input;
                    self.cursor = cursor;
                    self.completion.reset();
                    return;
                }
                (_, KeyCode::Esc) => {
                    self.completion.dismiss(&active.token);
                    return;
                }
                _ => {}
            }
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
                super::input::backspace(&mut self.input, &mut self.cursor);
            }
            (_, KeyCode::Delete) if self.cursor < super::input::char_count(&self.input) => {
                super::input::delete(&mut self.input, self.cursor);
            }
            (_, KeyCode::Left) if self.cursor > 0 => {
                self.cursor -= 1;
            }
            (_, KeyCode::Right) if self.cursor < super::input::char_count(&self.input) => {
                self.cursor += 1;
            }
            (_, KeyCode::Home) | (KeyModifiers::CONTROL, KeyCode::Char('a')) => {
                self.cursor = 0;
            }
            (_, KeyCode::End) | (KeyModifiers::CONTROL, KeyCode::Char('e')) => {
                self.cursor = super::input::char_count(&self.input);
            }
            (KeyModifiers::CONTROL, KeyCode::Char('u')) => {
                self.input.clear();
                self.cursor = 0;
                self.completion.reset();
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
                super::input::insert(&mut self.input, &mut self.cursor, c);
            }
            _ => {}
        }
    }

    pub fn handle_paste(&mut self, text: &str) {
        let normalized = normalize_paste(text);
        super::input::insert_text(&mut self.input, &mut self.cursor, &normalized);
    }

    pub fn take_input(&mut self) -> Option<String> {
        if self.input.trim().is_empty() {
            return None;
        }
        let input = self.input.clone();
        self.input.clear();
        self.cursor = 0;
        self.completion.reset();
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
    models: &[ResolvedModel],
    shutdown: &tokio_util::sync::CancellationToken,
) -> Result<Action> {
    let jobs = engine.jobs();
    jobs.enable();
    let result = run_session(engine, session_id, db, terminal, theme, models, shutdown).await;
    jobs.shutdown().await;
    result
}

async fn run_session(
    engine: &mut Engine,
    session_id: &str,
    db: &Db,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    theme: Theme,
    models: &[ResolvedModel],
    shutdown: &tokio_util::sync::CancellationToken,
) -> Result<Action> {
    // Clear engine state and load this session's messages. repair_history
    // makes old or crash-interrupted saves API-valid (tool_use/tool_result
    // pairing) before the engine sends them anywhere.
    let (existing_messages, recovered) = db.get_messages_with_recovery(session_id)?;
    let existing_messages = crate::session::repair_history(existing_messages);
    engine.set_messages(existing_messages.clone());
    engine.set_archive(db.get_archive(session_id)?);

    let mut app = ChatApp::new(engine.model(), theme);
    app.status = format!(
        "{} | {} | /help for commands",
        engine.model(),
        engine.context_status()
    );

    // Show existing messages in the UI
    for msg in &existing_messages {
        let role = &msg.role;
        let content = match &msg.content {
            crate::api::types::MessageContent::Text(t) => t.clone(),
            crate::api::types::MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    crate::api::ContentBlock::Text { text } => Some(text.clone()),
                    crate::api::ContentBlock::Image { source } => {
                        Some(format!("[attached {} image]", source.media_type))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };
        app.add_message(role, &content);
    }

    if recovered > 0 {
        app.add_message("system", &format!("Recovered {recovered} unreadable message(s) as placeholders. Original rows remain in the database."));
    }
    let mut needs_redraw = true;
    let mut pending_submit: Option<String> = None;

    loop {
        let jobs = engine.jobs();
        app.refresh_jobs(&jobs);
        needs_redraw |= !app.jobs.is_empty();
        if shutdown.is_cancelled() {
            if !app.save_session(db, session_id, engine) {
                anyhow::bail!(
                    "Session {session_id} could not be saved during shutdown: {}",
                    app.save_error.as_deref().unwrap_or("unknown error")
                );
            }
            return Ok(Action::Quit);
        }
        if needs_redraw {
            terminal.draw(|f| ui::draw_chat(f, &mut app))?;
            needs_redraw = false;
        }

        // Process pending submit
        if let Some(input) = pending_submit.take() {
            let trimmed = input.trim().to_string();
            if trimmed == "/jobs" {
                app.show_jobs = true;
                needs_redraw = true;
                continue;
            }

            // Commands may clear history or switch sessions. Do not let an
            // outstanding save failure silently discard the only good copy.
            if trimmed.starts_with('/')
                && app.save_error.is_some()
                && !app.save_session(db, session_id, engine)
            {
                app.add_message(
                    "error",
                    "Command paused because this session is unsaved. Ctrl+S retries.",
                );
                needs_redraw = true;
                continue;
            }

            // /home is a screen transition, so it returns Action::Home rather
            // than routing through parse_command like the rest. It is still
            // declared in commands::COMMANDS (tui_only) so that /help lists it
            // and the completion menu offers it; parse_command's own /home arm
            // only ever runs in the REPL, where it explains the command is
            // TUI-only.
            if trimmed == "/home" {
                return Ok(Action::Home);
            }

            // Check slash commands
            if let Some(result) = commands::parse_command(&trimmed, commands::Surface::Tui) {
                match result {
                    CommandResult::Text(ref text) if text == "__cost__" => {
                        app.add_message("system", &commands::format_cost(engine));
                    }
                    CommandResult::Text(ref text) if text == "__context__" => {
                        app.add_message("system", &commands::format_context(engine));
                    }
                    CommandResult::Text(text) => {
                        app.add_message("system", &text);
                    }
                    CommandResult::Exit => {
                        return Ok(Action::Home);
                    }
                    CommandResult::Async(async_cmd) => match async_cmd {
                        commands::AsyncCommand::Resume(Some(prefix)) => {
                            // Resuming is a session switch, not a history
                            // mutation.  Returning to the top-level loop lets
                            // it reopen the selected session (and rebuild the
                            // provider when its binding differs) instead of
                            // loading another session's messages into the
                            // current one and saving them over its row.
                            match crate::session::find_session(&prefix)? {
                                Some((target_session_id, _)) => {
                                    return Ok(Action::Chat {
                                        session_id: target_session_id,
                                    });
                                }
                                None => app
                                    .add_message("error", &format!("Session not found: {prefix}")),
                            }
                        }
                        commands::AsyncCommand::Model(selector) => match selector {
                            Some(selector) => {
                                // The top-level TUI owns provider construction.
                                // Return there so switching profiles rebuilds
                                // the engine instead of merely changing a model
                                // string on the current provider.
                                if !app.save_session(db, session_id, engine) {
                                    needs_redraw = true;
                                    continue;
                                }
                                return Ok(Action::SwitchModel {
                                    session_id: session_id.to_string(),
                                    selector,
                                });
                            }
                            None => app.add_message(
                                "system",
                                &commands::format_model_choices(engine.model_binding(), models),
                            ),
                        },
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
                            app.save_session(db, session_id, engine);
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

            app.status = format!("{} | {}", app.model, engine.context_status());

            let submit_result = drive_streaming(
                engine,
                &trimmed,
                &mut app,
                terminal,
                &mut CrosstermKeys,
                shutdown,
            )
            .await;

            if let Err(e) = submit_result {
                app.add_message("error", &format!("Error: {e}"));
            }

            // Snapshot the full conversation, tool rounds included, even on
            // error: the engine may have made progress worth keeping.
            // Previously only the user message and the final assistant
            // message were saved, so resumed sessions lost everything the
            // turn actually did.
            app.save_session(db, session_id, engine);

            app.mode = Mode::Input;
            app.status = format!(
                "{} | {} | {}",
                engine.model(),
                engine.context_status(),
                engine.cost.format_summary()
            );
            app.scroll = 0;
            app.manual_scroll = false;
            needs_redraw = true;
            continue;
        }

        // Poll terminal events
        if event::poll(std::time::Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    if app.job_key(key, &jobs) {
                        needs_redraw = true;
                        continue;
                    }
                    if key.kind == event::KeyEventKind::Release {
                        continue;
                    }
                    if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('s') {
                        if app.save_session(db, session_id, engine) {
                            app.status = "Session saved".to_string();
                        }
                        needs_redraw = true;
                        continue;
                    }
                    // Enter submits, unless the completion menu has claimed it to
                    // accept the highlighted command. Ask the app rather than
                    // deciding here: the caller cannot see the menu state.
                    if key.code == KeyCode::Enter
                        && app.mode == Mode::Input
                        && !app.completion_is_open()
                    {
                        if let Some(input) = app.take_input() {
                            pending_submit = Some(input);
                        }
                    } else {
                        app.handle_key(key);
                    }
                    needs_redraw = true;
                }
                Event::Paste(text) if app.mode == Mode::Input && !app.show_jobs => {
                    app.handle_paste(&text);
                    needs_redraw = true;
                }
                _ => {}
            }
        }

        if app.should_exit {
            if app.save_error.is_some() && !app.save_session(db, session_id, engine) {
                app.should_exit = false;
                needs_redraw = true;
                continue;
            }
            return Ok(Action::Quit);
        }
        if app.should_go_home {
            if app.save_error.is_some() && !app.save_session(db, session_id, engine) {
                app.should_go_home = false;
                needs_redraw = true;
                continue;
            }
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
    shutdown: &tokio_util::sync::CancellationToken,
) -> Result<()> {
    app.set_activity("Waiting for model response");
    let steering = engine.steering_queue();
    let steer_buf = app.steer_buf.clone();
    let jobs = engine.jobs();
    let cancel = shutdown.child_token();
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
        let mut batch_tools = Vec::new();

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
                    if let Some(activity) = &mut app.activity {
                        activity.updated = std::time::Instant::now();
                    }
                    match event {
                        StreamEvent::ModelRequest => {
                            app.set_activity("Waiting for model response");
                        }
                        StreamEvent::Reasoning => {
                            app.set_activity("Model reasoning");
                        }
                        StreamEvent::Text(t) => {
                            app.set_activity("Receiving model response");
                            // Drawn by the tick, which coalesces chunks
                            app.stream_buffer
                                .push_str(&crate::utils::sanitize_terminal_text(&t));
                            app.thinking = false;
                        }
                        StreamEvent::Retry(n) => {
                            app.set_activity("Waiting for model retry");
                            // The rejected provider attempt is not part of
                            // conversation history, and its text has not been
                            // flushed: the engine only emits Retry for an
                            // attempt that never committed, and announcing a
                            // tool (the thing that flushes this buffer) is
                            // exactly what counts as committing. So clearing
                            // here cannot discard rendered output.
                            app.stream_buffer.clear();
                            app.thinking = true;
                            app.add_message("system", &n);
                        }
                        StreamEvent::Notice(n) => {
                            app.set_activity("Processing context");
                            flush_stream_buffer(app);
                            app.add_message("system", &n);
                        }
                        StreamEvent::ContextUsage(usage) => {
                            app.status = format!(
                                "{} | {}",
                                app.model,
                                usage.short_status()
                            );
                        }
                        StreamEvent::SteeringSent(t) => {
                            flush_stream_buffer(app);
                            app.add_message("user", &t);
                        }
                        StreamEvent::ToolStart { name, summary, input } => {
                            if running_tools.is_empty() {
                                batch_tools.clear();
                            }
                            flush_stream_buffer(app);
                            app.add_tool_with_input(
                                &name,
                                &summary,
                                &input,
                                ToolStatus::Queued,
                            );
                            running_tools.push_back(app.messages.len() - 1);
                            batch_tools.push(app.messages.len() - 1);
                            app.refresh_tool_activity(&batch_tools);
                            terminal.draw(|f| ui::draw_chat(f, app))?;
                        }
                        StreamEvent::ToolRunning { index } => {
                            app.activity = None;
                            if let Some(&idx) = batch_tools.get(index) {
                                app.set_tool_status_at(idx, ToolStatus::Running);
                            }
                            app.refresh_tool_activity(&batch_tools);
                        }
                        StreamEvent::ToolOutput { index, content } => {
                            if let Some(&idx) = batch_tools.get(index) {
                                app.set_tool_output_at(idx, &content);
                                app.refresh_tool_activity(&batch_tools);
                            }
                        }
                        StreamEvent::ToolFinished { index, is_error, content } => {
                            if let Some(&idx) = batch_tools.get(index) {
                                app.set_tool_output_at(idx, &content);
                                app.set_tool_status_at(idx, if is_error { ToolStatus::Error } else { ToolStatus::Success });
                            }
                            app.refresh_tool_activity(&batch_tools);
                        }
                        StreamEvent::ToolResult { is_error, content } => {
                            if let Some(idx) = running_tools.pop_front() {
                                app.set_tool_output_at(idx, &content);
                                app.set_tool_status_at(
                                    idx,
                                    if is_error { ToolStatus::Error } else { ToolStatus::Success },
                                );
                            }
                            app.refresh_tool_activity(&batch_tools);
                            terminal.draw(|f| ui::draw_chat(f, app))?;
                        }
                        StreamEvent::PermissionRequest { tool_name, summary, input, respond }
                        | StreamEvent::PermissionRequestWithDiff { tool_name, summary, input, respond, .. } => {
                            let response = tokio::select! {
                                _ = cancel.cancelled() => PermissionResponse::Deny,
                                response = prompt_permission_tui(
                                app,
                                terminal,
                                keys,

                                &tool_name,
                                &summary,
                                &input,
                                &steering,
                            ) => response?,
                            };
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
                            while let Some(idx) = running_tools.pop_front() {
                                app.set_tool_status_at(idx, ToolStatus::Error);
                            }
                            // Surfaced through submit_result by the caller
                        }
                        StreamEvent::Done => {
                            flush_stream_buffer(app);
                        }
                    }
                }
                _ = tick.tick() => {
                    app.refresh_jobs(&jobs);
                    let event = keys.poll_event()?;
                    let handled = match &event { Some(Event::Key(key)) => app.job_key(*key, &jobs), _ => app.show_jobs };
                    if !handled && poll_stream_key(&mut SingleEvent(event), &steer_buf, &steering)? {
                        cancel.cancel();
                        app.set_activity("Interrupting");
                    }
                    // The tick is the only draw during streaming: it
                    // coalesces all chunks since the last frame, animates
                    // the spinner, and keeps the steering buffer visible.
                    terminal.draw(|f| ui::draw_chat(f, app))?;
                }
            }
        }
    }

    app.activity = None;
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
    app.permission_prompt = Some(crate::utils::sanitize_terminal_text(summary));
    app.show_jobs = false;
    app.permission_details = Some(
        format_permission_details(tool_name, input)
            .into_iter()
            .map(|line| crate::utils::sanitize_terminal_text(&line))
            .collect(),
    );
    app.permission_always_label = Some(PermissionResponse::always_allow_label(tool_name, input));
    app.mode = Mode::Permission;
    terminal.draw(|f| ui::draw_chat(f, app))?;

    let mut perm_input = String::new();

    let response = loop {
        match keys.poll_event()? {
            None => {
                // Nothing typed: yield to the runtime before polling again
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Some(Event::Paste(text)) => {
                perm_input.push_str(&normalize_paste(&text));
                app.status = format!("{} | deny and message: {perm_input}", app.model);
                terminal.draw(|f| ui::draw_chat(f, app))?;
            }
            Some(Event::Key(key)) if key.kind != event::KeyEventKind::Release => {
                match key.code {
                    KeyCode::Char('y') | KeyCode::Enter if perm_input.is_empty() => {
                        break PermissionResponse::Allow;
                    }
                    KeyCode::Char('a') if perm_input.is_empty() => {
                        break PermissionResponse::always_allow_for(tool_name, input);
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
            Some(_) => {}
        }
    };

    app.permission_prompt = None;
    app.permission_details = None;
    app.permission_always_label = None;
    app.mode = Mode::Streaming;
    app.status = app.model.clone();
    Ok(response)
}

/// Key input for the streaming turn path. Production reads the real
/// terminal via crossterm; tests feed scripted keystrokes, which is what
/// makes the interactive flow (steering, permission prompts, Ctrl+C)
/// coverable by `cargo test`.
pub trait KeySource {
    fn poll_event(&mut self) -> Result<Option<Event>>;
}

struct SingleEvent(Option<Event>);
impl KeySource for SingleEvent {
    fn poll_event(&mut self) -> Result<Option<Event>> {
        Ok(self.0.take())
    }
}

/// Reads keys from the real terminal without blocking.
pub struct CrosstermKeys;

impl KeySource for CrosstermKeys {
    fn poll_event(&mut self) -> Result<Option<Event>> {
        if event::poll(std::time::Duration::from_millis(0))? {
            return Ok(Some(event::read()?));
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
        let input = keys.poll_event()?;
        if let Some(Event::Paste(text)) = &input {
            steer_buf
                .lock()
                .expect("steer buffer poisoned")
                .push_str(&normalize_paste(text));
        }
        if let Some(Event::Key(key)) = input {
            if key.kind == event::KeyEventKind::Release {
                return Ok(false);
            }
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

fn normalize_paste(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn format_permission_details(tool_name: &str, input: &serde_json::Value) -> Vec<String> {
    let mut lines = Vec::new();

    match tool_name {
        "Bash" => {
            if input["background"].as_bool() == Some(true) {
                lines.push(
                    "Background job: continues across turns; stops when this session closes."
                        .into(),
                );
            }
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

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_str(app: &mut ChatApp, text: &str) {
        for c in text.chars() {
            app.handle_key(key(KeyCode::Char(c)));
        }
    }

    #[test]
    fn output_is_bounded_sanitized_and_expandable_without_changing_input() {
        let mut app = test_app();
        app.add_tool("Bash", "echo", ToolStatus::Running);
        app.set_tool_output_at(0, &format!("{}\x1b[31mfinal\x07", "界".repeat(20_000)));
        let ChatMessage::Tool { output, .. } = &app.messages[0] else {
            panic!()
        };
        assert!(output.len() < 20_100);
        assert!(output.starts_with("[earlier output omitted]"));
        assert!(output.ends_with("final"));
        assert!(!output.contains('\x1b'));
        app.input = "draft".into();
        let revision = app.messages_rev;
        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(app.expand_tool_output);
        assert!(app.messages_rev > revision);
        assert_eq!(app.input, "draft");
    }

    #[test]
    fn tab_accepts_the_selected_completion() {
        let mut app = test_app();
        type_str(&mut app, "/comp");
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.input, "/compact");
        assert_eq!(app.cursor, 8);
    }

    /// Mirror the event loop's Enter routing: it decides between submitting
    /// and delegating *before* handle_key runs, so a test that only calls
    /// handle_key does not exercise the real path. An earlier version of these
    /// tests missed exactly that, and a bare `/` was submitted to the model.
    fn press_enter_via_event_loop(app: &mut ChatApp) -> Option<String> {
        if app.mode == Mode::Input && !app.completion_is_open() {
            app.take_input()
        } else {
            app.handle_key(key(KeyCode::Enter));
            None
        }
    }

    #[test]
    fn a_bare_slash_is_never_submitted() {
        // Regression: typing / and pressing Enter sent "/" to the model, which
        // came back as "Unknown command: /".
        let mut app = test_app();
        type_str(&mut app, "/");
        let submitted = press_enter_via_event_loop(&mut app);
        assert!(
            submitted.is_none(),
            "must not submit while the menu is open"
        );
        assert_eq!(app.input, "/help", "Enter accepts the first entry");
    }

    #[test]
    fn enter_submits_once_the_menu_has_closed() {
        let mut app = test_app();
        type_str(&mut app, "/");
        press_enter_via_event_loop(&mut app); // accepts /help, closing the menu
        let submitted = press_enter_via_event_loop(&mut app);
        assert_eq!(submitted.as_deref(), Some("/help"));
    }

    #[test]
    fn enter_submits_ordinary_text_untouched() {
        let mut app = test_app();
        type_str(&mut app, "hello there");
        let submitted = press_enter_via_event_loop(&mut app);
        assert_eq!(submitted.as_deref(), Some("hello there"));
    }

    #[test]
    fn enter_accepts_a_completion_instead_of_submitting() {
        // With the menu open, Enter picks the highlighted command. The caller
        // only sees a submit once the menu is gone.
        let mut app = test_app();
        type_str(&mut app, "/comp");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.input, "/compact");

        // Menu is now closed (exact match), so the next Enter is a real submit.
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.input, "/compact", "second Enter leaves the line intact");
    }

    #[test]
    fn arrows_move_the_selection_rather_than_scrolling() {
        let mut app = test_app();
        let before = app.scroll;
        type_str(&mut app, "/c");
        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.scroll, before, "the transcript must not scroll");

        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.input, "/context", "Down moved to the second entry");
    }

    #[test]
    fn arrows_still_scroll_when_no_menu_is_open() {
        let mut app = test_app();
        type_str(&mut app, "hello");
        app.handle_key(key(KeyCode::Up));
        assert!(app.scroll > 0, "ordinary input keeps arrow scrolling");
    }

    #[test]
    fn esc_dismisses_the_menu_and_typing_brings_it_back() {
        let mut app = test_app();
        type_str(&mut app, "/co");
        app.handle_key(key(KeyCode::Esc));

        // Dismissed: Tab is now an ordinary key, so the line is unchanged.
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.input, "/co");

        type_str(&mut app, "m");
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.input, "/compact", "typing re-arms completion");
    }

    #[test]
    fn submitting_clears_completion_state() {
        let mut app = test_app();
        type_str(&mut app, "/co");
        app.handle_key(key(KeyCode::Down));
        app.take_input();

        // A stale selection must not carry into the next command.
        type_str(&mut app, "/co");
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.input, "/cost", "selection reset to the first entry");
    }

    #[test]
    fn a_slash_mid_sentence_is_just_text() {
        let mut app = test_app();
        type_str(&mut app, "what about /co");
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.input, "what about /co", "no completion mid-line");
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
    fn input_editing_handles_multibyte_characters() {
        let mut app = test_app();
        for character in ['a', 'é', '界'] {
            app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }
        assert_eq!(app.input, "aé界");
        assert_eq!(app.cursor, 3);

        app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(app.input, "a界");
        assert_eq!(app.cursor, 1);

        app.handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(app.input, "a");
    }

    #[test]
    fn bracketed_paste_preserves_large_multiline_text() {
        let mut app = test_app();
        let pasted = "first line\r\nsecond line\rthird line\n".repeat(256);

        app.handle_paste(&pasted);

        assert_eq!(app.input, pasted.replace("\r\n", "\n").replace('\r', "\n"));
        assert_eq!(app.cursor, app.input.chars().count());
    }

    #[test]
    fn displayed_messages_strip_terminal_controls() {
        let mut app = test_app();
        app.add_message("assistant", "hello\x1b]52;c;secret\x07 world");
        app.add_tool("Bash\x1b[2J", "echo safe\rspoof", ToolStatus::Running);

        assert!(matches!(
            &app.messages[0],
            ChatMessage::Text { content, .. }
                if content == "hello]52;c;secret world" && !content.contains('\x1b')
        ));
        assert!(matches!(
            &app.messages[1],
            ChatMessage::Tool { name, summary, .. }
                if name == "Bash[2J" && summary == "echo safespoof"
        ));
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
                ..
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

    struct ScriptedEvents(std::collections::VecDeque<Event>);

    #[test]
    fn save_failure_remains_visible_until_real_database_retry_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sessions.db");
        let db = Db::open(&path).unwrap();
        db.create_session("s", "test", None, None).unwrap();
        let other = rusqlite::Connection::open(&path).unwrap();
        other.execute_batch("CREATE TRIGGER reject_save BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'disk unavailable'); END;").unwrap();
        let mut engine = scripted_engine(vec![], None, PermissionMode::Default);
        engine.set_messages(vec![crate::api::Message::user("keep this draft")]);
        let mut app = ChatApp::new("test", Theme::dark());
        assert!(!app.save_session(&db, "s", &engine));
        assert!(app
            .save_error
            .as_deref()
            .unwrap()
            .contains("disk unavailable"));
        app.status = "unrelated token update".into();
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
        assert!(buffer_text(&terminal).contains("UNSAVED"));
        let count = app.messages.len();
        assert!(!app.save_session(&db, "s", &engine));
        assert_eq!(
            app.messages.len(),
            count,
            "repeated failures do not flood chat"
        );
        assert_eq!(engine.messages().len(), 1);
        other.execute_batch("DROP TRIGGER reject_save").unwrap();
        assert!(app.save_session(&db, "s", &engine));
        assert!(app.save_error.is_none());
        assert_eq!(db.get_messages("s").unwrap().len(), 1);
        terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
        // Historical error stays in the transcript, but the footer recovers.
        let screen = buffer_text(&terminal);
        let footer = crate::utils::tail_str(&screen, 100);
        assert!(!footer.contains("UNSAVED"));
    }

    impl KeySource for ScriptedEvents {
        fn poll_event(&mut self) -> Result<Option<Event>> {
            Ok(self.0.pop_front())
        }
    }

    #[test]
    fn streaming_paste_preserves_multiline_unicode_until_explicit_enter() {
        let buffer = Arc::new(Mutex::new(String::from("prefix ")));
        let steering = SteeringQueue::default();
        let mut keys = ScriptedEvents(
            vec![
                Event::Paste("界\r\nsecond\rthird\n".into()),
                Event::Key(enter()),
            ]
            .into(),
        );
        assert!(!poll_stream_key(&mut keys, &buffer, &steering).unwrap());
        assert_eq!(*buffer.lock().unwrap(), "prefix 界\nsecond\nthird\n");
        assert!(steering.lock().unwrap().is_empty());
        assert!(!poll_stream_key(&mut keys, &buffer, &steering).unwrap());
        assert_eq!(
            steering.lock().unwrap().pop_front().unwrap(),
            "prefix 界\nsecond\nthird"
        );
        assert!(buffer.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn permission_paste_cannot_grant_approval_or_submit_embedded_newlines() {
        for pasted in ["y\n", "a\r\n", "n\n", "please inspect 界\nfirst"] {
            let mut app = ChatApp::new("test", Theme::dark());
            let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
            let mut keys =
                ScriptedEvents(vec![Event::Paste(pasted.into()), Event::Key(enter())].into());
            let steering = SteeringQueue::default();
            let response = prompt_permission_tui(
                &mut app,
                &mut terminal,
                &mut keys,
                "Bash",
                "echo test",
                &serde_json::json!({"command": "echo test"}),
                &steering,
            )
            .await
            .unwrap();
            assert_eq!(response, PermissionResponse::DenyAndCancel);
            assert_eq!(
                steering.lock().unwrap().pop_front().unwrap(),
                normalize_paste(pasted)
            );
            assert!(keys.0.is_empty(), "only the physical Enter submits");
        }
    }

    #[test]
    fn streaming_ignores_key_release_and_keeps_ctrl_c_after_paste() {
        let buffer = Arc::new(Mutex::new(String::new()));
        let steering = SteeringQueue::default();
        let mut keys = ScriptedEvents(
            vec![
                Event::Paste("draft".into()),
                Event::Key(KeyEvent::new_with_kind(
                    KeyCode::Enter,
                    KeyModifiers::NONE,
                    event::KeyEventKind::Release,
                )),
                Event::Key(ctrl_c()),
            ]
            .into(),
        );
        assert!(!poll_stream_key(&mut keys, &buffer, &steering).unwrap());
        assert!(!poll_stream_key(&mut keys, &buffer, &steering).unwrap());
        assert!(steering.lock().unwrap().is_empty());
        assert!(poll_stream_key(&mut keys, &buffer, &steering).unwrap());
    }

    impl KeySource for ScriptedKeys {
        fn poll_event(&mut self) -> Result<Option<Event>> {
            Ok(self.0.pop_front().map(Event::Key))
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

    /// Draw the chat screen with `input` typed, and return the rendered text.
    /// Exercises the real popup geometry rather than just the match list.
    fn render_with_input(input: &str, width: u16, height: u16) -> String {
        let mut app = ChatApp::new("test-model", Theme::dark());
        app.mode = Mode::Input;
        for c in input.chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
        buffer_text(&terminal)
    }

    #[test]
    fn the_menu_renders_every_match_not_just_the_first() {
        // Regression: /h matched both /help and /home, but only /help was drawn.
        let screen = render_with_input("/h", 100, 30);
        assert!(screen.contains("/help"), "missing /help:\n{screen}");
        assert!(screen.contains("/home"), "missing /home:\n{screen}");
    }

    #[test]
    fn a_bare_slash_renders_the_whole_command_list() {
        // Regression: a fixed 8-row cap meant typing `/` showed only the first
        // eight of eleven commands, and the window did not scroll until the
        // selection moved - so /home and the rest were invisible until the user
        // typed enough to filter them in.
        let screen = render_with_input("/", 100, 40);
        for spec in commands::COMMANDS {
            assert!(
                screen.contains(spec.name),
                "{} missing from the menu",
                spec.name
            );
        }
    }

    #[test]
    fn a_short_terminal_still_draws_a_usable_menu() {
        // The list cannot fit, so it scrolls rather than overflowing. The
        // selected entry must still be on screen.
        let screen = render_with_input("/", 100, 14);
        assert!(
            screen.contains("/help"),
            "the selected entry must be visible:\n{screen}"
        );
    }

    #[test]
    fn a_tiny_terminal_draws_no_menu_rather_than_garbage() {
        // Not enough room above the input for borders plus a row.
        let screen = render_with_input("/", 100, 6);
        assert!(!screen.contains("Show this help"), "menu should be skipped");
    }

    #[test]
    fn a_long_input_wraps_onto_visible_editor_rows() {
        // 40-wide terminal -> 38 columns inside the input borders.
        let text = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ";
        let mut app = type_input(text);
        let mut terminal = Terminal::new(TestBackend::new(40, 30)).unwrap();
        terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
        let rows: Vec<String> = terminal
            .backend()
            .buffer()
            .content()
            .chunks(40)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect();

        assert!(rows
            .iter()
            .any(|row| row.contains("abcdefghijklmnopqrstuvwxyz0123456789AB")));
        assert!(rows.iter().any(|row| row.contains("CDEFGHIJ")));
    }

    #[test]
    fn wrapped_input_keeps_the_cursor_visible_after_home() {
        let mut app = type_input(&"abcdefghij".repeat(20));
        app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        let mut terminal = Terminal::new(TestBackend::new(40, 30)).unwrap();
        terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
        let screen = buffer_text(&terminal);
        assert!(
            screen.contains("abcdefghij"),
            "head of the line must be visible after Home:\n{screen}"
        );
    }

    /// Build an app with `input` already typed, cursor at the end.
    fn type_input(input: &str) -> ChatApp {
        let mut app = ChatApp::new("test-model", Theme::dark());
        app.mode = Mode::Input;
        for c in input.chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        app
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

        drive_streaming(
            engine,
            "go",
            &mut app,
            &mut terminal,
            &mut keys,
            &tokio_util::sync::CancellationToken::new(),
        )
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
    async fn background_bash_keeps_permissions_and_pairs_the_launch_result() {
        for allow in [false, true] {
            let mut engine = scripted_engine(
                vec![tool_use(
                    "bg",
                    "Bash",
                    serde_json::json!({
                        "command": "printf ready; sleep 30", "background": true
                    }),
                )],
                None,
                PermissionMode::Default,
            );
            engine.jobs().enable();
            let (mut app, mut terminal) = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                run_turn(&mut engine, vec![ch(if allow { 'y' } else { 'n' })]),
            )
            .await
            .unwrap();
            assert_eq!(engine.jobs().snapshots().len(), usize::from(allow));
            let repaired = crate::session::repair_history(engine.messages().to_vec());
            assert_eq!(
                serde_json::to_value(&repaired).unwrap(),
                serde_json::to_value(engine.messages()).unwrap()
            );
            if allow {
                assert!(engine
                    .undo_last_turn()
                    .unwrap_err()
                    .to_string()
                    .contains("background jobs"));
                app.refresh_jobs(&engine.jobs());
                app.show_jobs = true;
                terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
                assert!(buffer_text(&terminal).contains("running"));
                assert!(app.job_key(ch('x'), &engine.jobs()));
                engine.jobs().shutdown().await;
                app.refresh_jobs(&engine.jobs());
                assert!(matches!(
                    app.jobs[0].status,
                    crate::tools::jobs::JobStatus::Cancelled
                ));
                let count = app.messages.len();
                app.refresh_jobs(&engine.jobs());
                assert_eq!(count, app.messages.len(), "completion is displayed once");
                assert!(app.job_key(
                    KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
                    &engine.jobs()
                ));
                assert!(!app.show_jobs);
            }
        }
    }

    #[test]
    fn dashboard_is_safe_when_empty_tiny_or_scrolled() {
        let mut app = ChatApp::new("test", Theme::dark());
        app.show_jobs = true;
        let jobs = crate::tools::jobs::JobManager::default();
        for key in [
            KeyCode::Up,
            KeyCode::Down,
            KeyCode::PageUp,
            KeyCode::PageDown,
            KeyCode::Char('x'),
        ] {
            assert!(app.job_key(KeyEvent::new(key, KeyModifiers::NONE), &jobs));
        }
        for (width, height) in [(1, 1), (20, 6), (100, 30)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| ui::draw_chat(f, &mut app)).unwrap();
        }
        assert_eq!(app.selected_job, 0);
    }

    #[tokio::test]
    async fn permission_always_is_command_specific_for_bash() {
        let mut app = ChatApp::new("test-model", Theme::dark());
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut keys = ScriptedKeys(vec![ch('a')].into());
        let steering = SteeringQueue::default();
        let command = format!("echo {}", "x".repeat(200));
        let input = serde_json::json!({"command": command});

        let response = prompt_permission_tui(
            &mut app,
            &mut terminal,
            &mut keys,
            "Bash",
            "bash: truncated display",
            &input,
            &steering,
        )
        .await
        .unwrap();

        assert_eq!(
            response,
            PermissionResponse::AlwaysAllowCommand(command),
            "the grant must use the complete raw command"
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

    #[tokio::test]
    async fn shutdown_cancels_tools_and_permission_waits_with_paired_history() {
        for mode in [PermissionMode::Bypass, PermissionMode::Default] {
            let mut engine = scripted_engine(
                vec![tool_use(
                    "shutdown-tool",
                    "Bash",
                    serde_json::json!({"command": "sleep 5"}),
                )],
                None,
                mode,
            );
            let shutdown = tokio_util::sync::CancellationToken::new();
            let mut app = ChatApp::new("test-model", Theme::dark());
            app.mode = Mode::Streaming;
            let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
            let mut keys = ScriptedKeys(Vec::new().into());
            let turn = drive_streaming(
                &mut engine,
                "go",
                &mut app,
                &mut terminal,
                &mut keys,
                &shutdown,
            );
            let signal = async {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                shutdown.cancel();
            };
            let (result, ()) = tokio::time::timeout(std::time::Duration::from_secs(3), async {
                tokio::join!(turn, signal)
            })
            .await
            .expect("shutdown should not wait for the command or permission");
            result.unwrap();
            assert_eq!(tool_statuses(&app), vec![ToolStatus::Error]);
            assert_eq!(
                serde_json::to_value(crate::session::repair_history(engine.messages().to_vec()))
                    .unwrap(),
                serde_json::to_value(engine.messages()).unwrap(),
                "shutdown must finish pairing tool uses before the session is saved",
            );
        }
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
            name = "chat-tool-running",
            description = "Silent Bash execution with elapsed and quiet time"
        )]
        ToolRunning,

        #[tuishot(
            name = "chat-tool-queued",
            description = "Tool announced but not executing yet"
        )]
        ToolQueued,

        #[tuishot(
            name = "chat-tool-output",
            description = "Live bounded Bash output preview"
        )]
        ToolOutput,

        #[tuishot(
            name = "chat-unsaved",
            description = "Persistent save failure and retry hint"
        )]
        Unsaved,

        #[tuishot(
            name = "chat-background-jobs",
            description = "Session job dashboard with live output and cancellation"
        )]
        Jobs,

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
            let now = std::time::Instant::now();
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
                    app.permission_always_label =
                        Some("(a)lways allow cargo test commands".to_string());
                    app
                }
                ChatShot::ToolRunning | ChatShot::ToolQueued | ChatShot::ToolOutput => {
                    let mut app = sample_conversation();
                    app.mode = Mode::Streaming;
                    let running = !matches!(self, ChatShot::ToolQueued);
                    app.add_tool_with_input(
                        "Bash",
                        "",
                        &serde_json::json!({
                            "command": "rg -n cloud-hosting ~/dev",
                            "description": "Find cloud-hosting references across repos"
                        }),
                        if running {
                            ToolStatus::Running
                        } else {
                            ToolStatus::Queued
                        },
                    );
                    let started = now - std::time::Duration::from_secs(45);
                    app.activity = Some(Activity {
                        label: if running {
                            "Running Bash"
                        } else {
                            "Tools queued; waiting to start"
                        }
                        .to_string(),
                        started,
                        updated: started,
                    });
                    if matches!(self, ChatShot::ToolOutput) {
                        app.set_tool_output_at(app.messages.len() - 1,
                            "[stdout]\nScanning repositories...\ncloud-hosting/README.md:12:Deployment\nclaux/config.toml:8:cloud-hosting\nStill scanning worktrees...");
                        app.activity.as_mut().unwrap().updated = now;
                    }
                    app
                }
                ChatShot::Empty => {
                    let mut app = ChatApp::new("claude-sonnet-4-20250514", theme);
                    app.version = SNAPSHOT_VERSION.to_string();
                    app
                }
                ChatShot::Unsaved => {
                    let mut app = sample_conversation();
                    app.record_save(Err(anyhow::anyhow!("database is read-only")));
                    app
                }
                ChatShot::Jobs => {
                    use crate::tools::jobs::{JobSnapshot, JobStatus};
                    let mut app = sample_conversation();
                    app.show_jobs = true;
                    app.jobs = vec![JobSnapshot { id: "job-a12b".into(), command: "cargo test".into(), status: JobStatus::Running,
                        elapsed: std::time::Duration::from_secs(42), output: "[stdout]\nrunning 548 tests\ntest jobs::launch_returns_early ... ok\ntest jobs::cancellation ... ok\nStill running...".into() },
                        JobSnapshot { id: "job-c34d".into(), command: "docker compose build".into(), status: JobStatus::Succeeded,
                        elapsed: std::time::Duration::from_secs(18), output: "Build complete".into() }];
                    app
                }
            };
            let rendered = tuishot::render_to_buffer(area.width, area.height, |f| {
                ui::draw_chat_at(f, &mut app, now);
            });
            buf.clone_from(&rendered);
        }
    }

    #[test]
    fn capture_chat_screens() {
        ChatShot::check_all().expect("chat screen capture");
    }
}
