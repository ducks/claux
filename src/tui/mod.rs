mod markdown;
mod ui;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;
// mpsc used indirectly via StreamEvent
#[allow(unused_imports)]
use tokio::sync::mpsc;

use crate::commands::{self, CommandResult};
use crate::config::{Config, HookTrigger};
use crate::context;
use crate::permissions::PermissionResponse;
use crate::plugin::PluginRegistry;
use crate::query::Engine;
use crate::session;

/// A displayed message in the chat.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// What the TUI is currently doing.
#[derive(Debug, Clone, PartialEq)]
enum Mode {
    /// Waiting for user input
    Input,
    /// Streaming a response from Claude
    Streaming,
    /// Waiting for permission confirmation
    Permission,
}

/// TUI application state.
pub struct App {
    /// Displayed messages
    messages: Vec<ChatMessage>,
    /// Current input buffer
    input: String,
    /// Cursor position in input
    cursor: usize,
    /// Scroll offset for messages
    scroll: u16,
    /// Whether the user has manually scrolled up
    manual_scroll: bool,
    /// Current mode
    mode: Mode,
    /// Current streaming text accumulator
    stream_buffer: String,
    /// Status line text
    status: String,
    /// Permission prompt summary (when in Permission mode)
    permission_prompt: Option<String>,
    /// Full permission details (tool name + input preview)
    permission_details: Option<Vec<String>>,
    /// Channel to send permission responses
    permission_respond: Option<tokio::sync::oneshot::Sender<PermissionResponse>>,
    /// Should we exit?
    should_exit: bool,
    /// Model name for display
    model: String,
    /// Total lines needed for messages (for scroll calculation)
    total_lines: u16,
}

impl App {
    fn new(model: &str) -> Self {
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
            permission_respond: None,
            should_exit: false,
            model: model.to_string(),
            total_lines: 0,
        }
    }

    fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            Mode::Input => self.handle_input_key(key),
            Mode::Permission => self.handle_permission_key(key),
            Mode::Streaming => {
                // Ctrl+C to cancel would go here
            }
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c'))
            | (KeyModifiers::CONTROL, KeyCode::Char('d')) => {
                self.should_exit = true;
            }
            (_, KeyCode::Enter) => {
                // Submit is handled by the caller
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

    fn handle_permission_key(&mut self, key: KeyEvent) {
        let response = match key.code {
            KeyCode::Char('y') | KeyCode::Enter => Some(PermissionResponse::Allow),
            KeyCode::Char('a') => Some(PermissionResponse::AlwaysAllow),
            KeyCode::Char('n') | KeyCode::Esc => Some(PermissionResponse::Deny),
            _ => None,
        };

        if let Some(resp) = response {
            if let Some(tx) = self.permission_respond.take() {
                let _ = tx.send(resp);
            }
            self.permission_prompt = None;
            self.mode = Mode::Streaming;
        }
    }

    fn take_input(&mut self) -> Option<String> {
        if self.input.trim().is_empty() {
            return None;
        }
        let input = self.input.clone();
        self.input.clear();
        self.cursor = 0;
        Some(input)
    }
}

/// Run the TUI.
pub async fn run(mut engine: Engine, _config: &Config, plugins: &PluginRegistry) -> Result<()> {
    let system_prompt = context::build_system_prompt_for_model(engine.model(), Some(plugins), &HookTrigger::OnContextBuild).await?;
    engine.set_system_prompt(system_prompt);

    let (_session_id, session_path) = session::create_session(engine.model())?;

    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(engine.model());
    app.status = format!("{} | /help for commands", engine.model());

    let mut needs_redraw = true;
    let mut pending_submit: Option<String> = None;

    loop {
        if needs_redraw {
            terminal.draw(|f| ui::draw(f, &mut app))?;
            needs_redraw = false;
        }

        // If we have a pending submit, start the query
        if let Some(input) = pending_submit.take() {
            let trimmed = input.trim().to_string();

            // Check slash commands
            if let Some(result) = commands::parse_command(&trimmed) {
                match result {
                    CommandResult::Text(ref text) if text == "__cost__" => {
                        app.add_message("system", &commands::format_cost(&engine));
                    }
                    CommandResult::Text(text) => {
                        app.add_message("system", &text);
                    }
                    CommandResult::Exit => {
                        app.should_exit = true;
                    }
                    CommandResult::Async(async_cmd) => {
                        match commands::execute_async(async_cmd, &mut engine).await {
                            Ok(output) => app.add_message("system", &output),
                            Err(e) => app.add_message("error", &format!("Error: {}", e)),
                        }
                    }
                }
                app.scroll = 0;
                app.manual_scroll = false;
                needs_redraw = true;
                continue;
            }

            // Regular message — start streaming
            app.add_message("user", &trimmed);
            app.mode = Mode::Streaming;
            app.stream_buffer.clear();
            app.scroll = 0;
            app.manual_scroll = false;

            let user_msg = crate::api::Message::user(&trimmed);
            let _ = session::append_message(&session_path, &user_msg);

            app.status = format!("{} | streaming...", app.model);

            let submit_result = drive_streaming(
                &mut engine,
                &trimmed,
                &mut app,
                &mut terminal,
            )
            .await;

            match submit_result {
                Ok(()) => {
                    // Save the assistant response
                    if let Some(last) = engine.messages().last() {
                        let _ = session::append_message(&session_path, last);
                    }
                }
                Err(e) => {
                    app.add_message("error", &format!("Error: {}", e));
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
            break;
        }
    }

    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;

    println!("{}", engine.cost.format_summary());
    Ok(())
}

/// Drive the streaming query, handling both stream events and terminal events.
async fn drive_streaming(
    engine: &mut Engine,
    input: &str,
    app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
) -> Result<()> {
    engine.messages_mut().push(crate::api::Message::user(input));

    // Start the API stream
    let tool_defs = engine.tool_definitions();
    let mut api_rx = engine.start_stream(&tool_defs).await?;

    let mut text_buf = String::new();
    let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();

    loop {
        // Inner loop: stream one API response
        loop {
            tokio::select! {
                Some(event) = api_rx.recv() => {
                    match event {
                        crate::api::ApiEvent::Text(t) => {
                            app.stream_buffer.push_str(&t);
                            text_buf.push_str(&t);
                            terminal.draw(|f| ui::draw(f, app))?;
                        }
                        crate::api::ApiEvent::ToolUse { id, name, input } => {
                            // Don't show tool starts yet — show them with results
                            // to keep summaries and checkmarks aligned
                            tool_uses.push((id, name, input));
                        }
                        crate::api::ApiEvent::Usage(usage) => {
                            engine.cost.add_usage(&usage);
                        }
                        crate::api::ApiEvent::Done => break,
                        crate::api::ApiEvent::Error(e) => {
                            return Err(anyhow::anyhow!("API error: {}", e));
                        }
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                    // Check for terminal events during streaming
                    if event::poll(std::time::Duration::from_millis(0))? {
                        if let Event::Key(key) = event::read()? {
                            if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
                                return Ok(()); // Cancel
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
            engine.messages_mut().push(crate::api::Message::assistant_blocks(blocks));
        }

        if tool_uses.is_empty() {
            // Done — finalize the assistant message display
            if !app.stream_buffer.is_empty() {
                let content = app.stream_buffer.clone();
                app.stream_buffer.clear();
                app.add_message("assistant", &content);
            }
            break;
        }

        // Execute tools — show summary before each, checkmark after
        let mut result_blocks = Vec::new();
        for (id, name, input) in &tool_uses {
            // Show tool summary before execution
            let summary = engine.summarize_tool(name, input);
            app.stream_buffer.push_str(&format!("\n  [{}] {} ", name, summary));
            terminal.draw(|f| ui::draw(f, app))?;

            let is_read_only = engine.is_tool_read_only(name);
            let perm = engine.check_permission(name, input, is_read_only);

            let tool_output = match perm {
                crate::permissions::PermissionResult::Allow => {
                    engine.execute_tool(name, input.clone()).await?
                }
                crate::permissions::PermissionResult::Deny(reason) => {
                    crate::tools::ToolOutput {
                        content: format!("Permission denied: {}", reason),
                        is_error: true,
                    }
                }
                crate::permissions::PermissionResult::Ask(summary) => {
                    // Build detailed preview of what the tool wants to do
                    let details = format_permission_details(name, input);
                    app.permission_prompt = Some(summary.clone());
                    app.permission_details = Some(details);
                    app.mode = Mode::Permission;
                    terminal.draw(|f| ui::draw(f, app))?;

                    // Wait for user input
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
                        PermissionResponse::Deny => crate::tools::ToolOutput {
                            content: "Permission denied by user.".to_string(),
                            is_error: true,
                        },
                    }
                }
            };

            // Update stream display
            if tool_output.is_error {
                app.stream_buffer.push_str(" ✗\n");
            } else {
                app.stream_buffer.push_str(" ✓\n");
            }
            terminal.draw(|f| ui::draw(f, app))?;

            result_blocks.push(crate::api::ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content: tool_output.content,
                is_error: if tool_output.is_error { Some(true) } else { None },
            });
        }

        engine.messages_mut().push(crate::api::Message::tool_results(result_blocks));

        // Reset for next turn
        text_buf.clear();
        tool_uses.clear();

        // Start next API call
        let tool_defs = engine.tool_definitions();
        api_rx = engine.start_stream(&tool_defs).await?;
    }

    Ok(())
}

/// Format detailed permission preview for a tool invocation.
fn format_permission_details(tool_name: &str, input: &serde_json::Value) -> Vec<String> {
    let mut lines = Vec::new();

    match tool_name {
        "Bash" => {
            if let Some(cmd) = input["command"].as_str() {
                lines.push("Command:".to_string());
                for line in cmd.lines() {
                    lines.push(format!("  {}", line));
                }
            }
            if let Some(desc) = input["description"].as_str() {
                lines.push(format!("Description: {}", desc));
            }
        }
        "Write" => {
            if let Some(path) = input["file_path"].as_str() {
                lines.push(format!("File: {}", path));
            }
            if let Some(content) = input["content"].as_str() {
                let preview: Vec<&str> = content.lines().take(10).collect();
                lines.push("Content:".to_string());
                for line in &preview {
                    lines.push(format!("  {}", line));
                }
                let total = content.lines().count();
                if total > 10 {
                    lines.push(format!("  ... ({} more lines)", total - 10));
                }
            }
        }
        "Edit" => {
            if let Some(path) = input["file_path"].as_str() {
                lines.push(format!("File: {}", path));
            }
            if let Some(old) = input["old_string"].as_str() {
                lines.push("Replace:".to_string());
                for line in old.lines().take(5) {
                    lines.push(format!("  - {}", line));
                }
            }
            if let Some(new) = input["new_string"].as_str() {
                lines.push("With:".to_string());
                for line in new.lines().take(5) {
                    lines.push(format!("  + {}", line));
                }
            }
        }
        "Agent" => {
            if let Some(prompt) = input["prompt"].as_str() {
                lines.push("Task:".to_string());
                for line in prompt.lines().take(5) {
                    lines.push(format!("  {}", line));
                }
            }
        }
        _ => {
            // Generic: show the JSON input compactly
            let json_str = serde_json::to_string_pretty(input).unwrap_or_default();
            for line in json_str.lines().take(8) {
                lines.push(format!("  {}", line));
            }
        }
    }

    lines
}
