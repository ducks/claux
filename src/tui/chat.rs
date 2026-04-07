//! Chat screen: conversation with the LLM.
//!
//! This is the main interaction screen. Extracted from the original tui/mod.rs.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::Stdout;

use crate::commands::{self, CommandResult};
use crate::db::Db;
use crate::permissions::PermissionResponse;
use crate::plugin::PluginRegistry;
use crate::query::Engine;
use crate::theme::{Theme, ThemeName};

use super::screen::Action;
use super::ui;

/// A displayed message in the chat.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn set_theme(&mut self, theme_name: ThemeName) {
        self.theme = Theme::from_name(theme_name);
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            Mode::Input => self.handle_input_key(key),
            Mode::Permission | Mode::Streaming => {}
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c'))
            | (KeyModifiers::CONTROL, KeyCode::Char('d')) => {
                self.should_exit = true;
            }
            (_, KeyCode::Enter) => {
                // Submit handled by caller
            }
            (_, KeyCode::Backspace) => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.input.remove(self.cursor);
                }
            }
            (_, KeyCode::Delete) => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                }
            }
            (_, KeyCode::Left) => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                }
            }
            (_, KeyCode::Right) => {
                if self.cursor < self.input.len() {
                    self.cursor += 1;
                }
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
    plugins: &PluginRegistry,
) -> Result<Action> {
    // Load existing messages for this session
    let existing_messages = db.get_messages(session_id)?;
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
                        _ => match commands::execute_async(async_cmd, engine).await {
                            Ok(output) => app.add_message("system", &output),
                            Err(e) => app.add_message("error", &format!("Error: {e}")),
                        },
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

            let user_msg = crate::api::Message::user(&trimmed);
            let _ = db.append_message(session_id, &user_msg);

            app.status = format!("{} | thinking...", app.model);

            let submit_result = drive_streaming(engine, &trimmed, &mut app, terminal).await;

            match submit_result {
                Ok(()) => {
                    if let Some(last) = engine.messages().last() {
                        let _ = db.append_message(session_id, last);
                    }
                }
                Err(e) => {
                    app.add_message("error", &format!("Error: {e}"));
                }
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

/// Drive the streaming query, handling both stream events and terminal events.
async fn drive_streaming(
    engine: &mut Engine,
    input: &str,
    app: &mut ChatApp,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
) -> Result<()> {
    engine.messages_mut().push(crate::api::Message::user(input));

    let tool_defs = engine.tool_definitions();
    let mut api_rx = engine.start_stream(&tool_defs).await?;

    let mut text_buf = String::new();
    let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();

    loop {
        loop {
            tokio::select! {
                Some(event) = api_rx.recv() => {
                    match event {
                        crate::api::ApiEvent::Text(t) => {
                            app.stream_buffer.push_str(&t);
                            text_buf.push_str(&t);
                            if app.thinking {
                                app.thinking = false;
                            }
                            terminal.draw(|f| ui::draw_chat(f, app))?;
                        }
                        crate::api::ApiEvent::ToolUse { id, name, input } => {
                            tool_uses.push((id, name, input));
                        }
                        crate::api::ApiEvent::Usage(usage) => {
                            engine.cost.add_usage(&usage);
                        }
                        crate::api::ApiEvent::Done => break,
                        crate::api::ApiEvent::Error(e) => {
                            return Err(anyhow::anyhow!("API error: {e}"));
                        }
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                    if event::poll(std::time::Duration::from_millis(0))? {
                        if let Event::Key(key) = event::read()? {
                            if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }

        // Record assistant message
        let mut blocks = Vec::new();
        if !text_buf.is_empty() {
            blocks.push(crate::api::ContentBlock::Text {
                text: text_buf.clone(),
            });
        }
        for (id, name, input) in &tool_uses {
            blocks.push(crate::api::ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            });
        }
        if !blocks.is_empty() {
            engine
                .messages_mut()
                .push(crate::api::Message::assistant_blocks(blocks));
        }

        if tool_uses.is_empty() {
            if !app.stream_buffer.is_empty() {
                let content = app.stream_buffer.clone();
                app.stream_buffer.clear();
                app.add_message("assistant", &content);
            }
            break;
        }

        // Execute tools
        let mut result_blocks = Vec::new();
        for (id, name, input) in &tool_uses {
            let summary = engine.summarize_tool(name, input);
            app.stream_buffer
                .push_str(&format!("\n  [{name}] {summary} "));
            terminal.draw(|f| ui::draw_chat(f, app))?;

            let is_read_only = engine.is_tool_read_only(name);
            let perm = engine.check_permission(name, input, is_read_only);

            let tool_output = match perm {
                crate::permissions::PermissionResult::Allow => {
                    engine.execute_tool(name, input.clone()).await?
                }
                crate::permissions::PermissionResult::Deny(reason) => crate::tools::ToolOutput {
                    content: format!("Permission denied: {reason}"),
                    is_error: true,
                },
                crate::permissions::PermissionResult::Ask {
                    message: summary, ..
                } => {
                    let details = format_permission_details(name, input);
                    app.permission_prompt = Some(summary.clone());
                    app.permission_details = Some(details);
                    app.mode = Mode::Permission;
                    terminal.draw(|f| ui::draw_chat(f, app))?;

                    let response = loop {
                        if event::poll(std::time::Duration::from_millis(50))? {
                            if let Event::Key(key) = event::read()? {
                                match key.code {
                                    KeyCode::Char('y') | KeyCode::Enter => {
                                        break PermissionResponse::Allow;
                                    }
                                    KeyCode::Char('a') => {
                                        break PermissionResponse::AlwaysAllow;
                                    }
                                    KeyCode::Char('n') | KeyCode::Esc => {
                                        break PermissionResponse::Deny;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    };

                    app.permission_prompt = None;
                    app.permission_details = None;
                    app.mode = Mode::Streaming;

                    match response {
                        PermissionResponse::Allow => {
                            engine.execute_tool(name, input.clone()).await?
                        }
                        PermissionResponse::AlwaysAllow => {
                            engine.always_allow_tool(name);
                            engine.execute_tool(name, input.clone()).await?
                        }
                        PermissionResponse::AlwaysAllowCommand(cmd) => {
                            engine.always_allow_tool(name);
                            engine.execute_tool(name, input.clone()).await?
                        }
                        PermissionResponse::Deny => crate::tools::ToolOutput {
                            content: "Permission denied by user.".to_string(),
                            is_error: true,
                        },
                    }
                }
            };

            let (content, _was_truncated) =
                crate::compact::truncate_tool_output(&tool_output.content);

            if tool_output.is_error {
                app.stream_buffer.push_str(" ✗\n");
            } else {
                app.stream_buffer.push_str(" ✓\n");
            }
            terminal.draw(|f| ui::draw_chat(f, app))?;

            result_blocks.push(crate::api::ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content,
                is_error: if tool_output.is_error {
                    Some(true)
                } else {
                    None
                },
            });
        }

        engine
            .messages_mut()
            .push(crate::api::Message::tool_results(result_blocks));

        text_buf.clear();
        tool_uses.clear();

        let tool_defs = engine.tool_definitions();
        api_rx = engine.start_stream(&tool_defs).await?;
    }

    Ok(())
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
        _ => {
            let json_str = serde_json::to_string_pretty(input).unwrap_or_default();
            for line in json_str.lines().take(8) {
                lines.push(format!("  {line}"));
            }
        }
    }

    lines
}
