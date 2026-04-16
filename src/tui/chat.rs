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
pub enum ChatMessage {
    /// User, assistant, system, or error text message.
    Text {
        role: String,
        content: String,
    },
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
        self.messages.push(ChatMessage::Text {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn add_tool(&mut self, name: &str, summary: &str, status: ToolStatus) {
        self.messages.push(ChatMessage::Tool {
            name: name.to_string(),
            summary: summary.to_string(),
            status,
        });
    }

    /// Update the last tool message's status (e.g., from Running to Success/Error).
    pub fn update_last_tool_status(&mut self, new_status: ToolStatus) {
        if let Some(ChatMessage::Tool { status, .. }) = self.messages.last_mut() {
            *status = new_status;
        }
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
    // Clear engine state and load this session's messages
    engine.messages_mut().clear();
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
                    // Redraw periodically so the spinner animates
                    if app.thinking {
                        terminal.draw(|f| ui::draw_chat(f, app))?;
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
            // Flush any pending streamed text before showing tool
            if !app.stream_buffer.is_empty() {
                let content = app.stream_buffer.clone();
                app.stream_buffer.clear();
                app.add_message("assistant", &content);
            }
            app.add_tool(&name, &summary, ToolStatus::Running);
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
                app.update_last_tool_status(ToolStatus::Error);
            } else {
                app.update_last_tool_status(ToolStatus::Success);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;

    fn test_app() -> ChatApp {
        ChatApp::new("test-model", Theme::dark())
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
        assert!(matches!(&app.messages[1], ChatMessage::Tool { status: ToolStatus::Success, .. }));
        assert!(matches!(&app.messages[2], ChatMessage::Text { role, .. } if role == "assistant"));
    }
}

#[cfg(test)]
mod tuishot_shots {
    use super::*;
    use tuishot::Tuishot;

    fn sample_conversation() -> ChatApp {
        let theme = crate::theme::Theme::dark();
        let mut app = ChatApp::new("claude-sonnet-4-20250514", theme);

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
        #[tuishot(name = "chat-conversation", description = "Mid-conversation with tool use and markdown")]
        Conversation,

        #[tuishot(name = "chat-streaming", description = "Assistant mid-response with streaming cursor")]
        Streaming,

        #[tuishot(name = "chat-permission", description = "Prompting for Bash permission")]
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
                        and per-project overrides".to_string();
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
                ChatShot::Empty => ChatApp::new("claude-sonnet-4-20250514", theme),
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
