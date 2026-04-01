use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::{App, Mode};

const FG: Color = Color::Rgb(213, 196, 161); // gruvbox fg2
const BG_DARK: Color = Color::Rgb(40, 40, 40); // gruvbox bg
const BLUE: Color = Color::Rgb(131, 165, 152); // gruvbox blue
const GREEN: Color = Color::Rgb(184, 187, 38); // gruvbox green
const YELLOW: Color = Color::Rgb(250, 189, 47); // gruvbox yellow
const RED: Color = Color::Rgb(251, 73, 52); // gruvbox red
const PURPLE: Color = Color::Rgb(211, 134, 155); // gruvbox purple
const GRAY: Color = Color::Rgb(146, 131, 116); // gruvbox gray

pub fn draw(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // Header
            Constraint::Min(1),    // Messages
            Constraint::Length(3), // Input
            Constraint::Length(1), // Status bar
        ])
        .split(f.area());

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" claude-rs ", Style::default().fg(PURPLE).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("v{}", env!("CARGO_PKG_VERSION")),
            Style::default().fg(GRAY),
        ),
    ]));
    f.render_widget(header, chunks[0]);

    // Messages area
    let msg_area = chunks[1];
    let _msg_width = msg_area.width.saturating_sub(2) as usize;

    let mut lines: Vec<Line> = Vec::new();

    for msg in &app.messages {
        let (prefix, style) = match msg.role.as_str() {
            "user" => (
                "> ",
                Style::default().fg(BLUE).add_modifier(Modifier::BOLD),
            ),
            "assistant" => ("", Style::default().fg(FG)),
            "system" => ("", Style::default().fg(YELLOW)),
            "error" => ("", Style::default().fg(RED)),
            _ => ("", Style::default().fg(FG)),
        };

        // Add a blank line before each message
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }

        // Role label
        if msg.role == "user" {
            lines.push(Line::from(Span::styled("You", style)));
        }

        // Wrap message content
        for line in msg.content.lines() {
            let display = if !prefix.is_empty() && lines.last().map_or(true, |l| l.spans.is_empty()) {
                format!("{}{}", prefix, line)
            } else {
                line.to_string()
            };

            let text_style = if msg.role == "user" {
                Style::default().fg(BLUE)
            } else {
                Style::default().fg(FG)
            };

            lines.push(Line::from(Span::styled(display, text_style)));
        }
    }

    // Streaming buffer (assistant response in progress)
    if !app.stream_buffer.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        for line in app.stream_buffer.lines() {
            lines.push(Line::from(Span::styled(
                line.to_string(),
                Style::default().fg(GREEN),
            )));
        }
        // Cursor indicator
        lines.push(Line::from(Span::styled("▊", Style::default().fg(GREEN))));
    }

    app.total_lines = lines.len() as u16;

    // Auto-scroll to bottom unless user scrolled up
    let visible_height = msg_area.height.saturating_sub(2);
    let max_scroll = app.total_lines.saturating_sub(visible_height);
    if !app.manual_scroll {
        app.scroll = 0; // 0 means bottom
    }

    // Calculate actual scroll offset (we scroll from bottom)
    let scroll_offset = if app.manual_scroll {
        max_scroll.saturating_sub(app.scroll.min(max_scroll))
    } else {
        max_scroll
    };

    let messages_widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(GRAY)),
        )
        .scroll((scroll_offset, 0));
    f.render_widget(messages_widget, msg_area);

    // Input area
    let input_style = match app.mode {
        Mode::Input => Style::default().fg(FG),
        Mode::Permission => Style::default().fg(YELLOW),
        Mode::Streaming => Style::default().fg(GRAY),
    };

    let input_text = if let Some(ref prompt) = app.permission_prompt {
        format!("⚡ {}  (y)es / (n)o / (a)lways", prompt)
    } else if app.mode == Mode::Streaming {
        "...".to_string()
    } else {
        app.input.clone()
    };

    let input_widget = Paragraph::new(input_text)
        .style(input_style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(if app.mode == Mode::Input {
                    BLUE
                } else if app.mode == Mode::Permission {
                    YELLOW
                } else {
                    GRAY
                }))
                .title(if app.mode == Mode::Permission {
                    " Permission "
                } else {
                    " > "
                }),
        );
    f.render_widget(input_widget, chunks[2]);

    // Set cursor position in input mode
    if app.mode == Mode::Input {
        f.set_cursor_position((
            chunks[2].x + app.cursor as u16 + 1,
            chunks[2].y + 1,
        ));
    }

    // Status bar
    let status = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {} ", app.status), Style::default().fg(GRAY)),
    ]));
    f.render_widget(status, chunks[3]);
}
