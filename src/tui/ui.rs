use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::{App, Mode};
use super::markdown;

pub fn draw(f: &mut Frame, app: &mut App) {
    // Expand input area when showing permission details
    let input_height = if app.permission_details.is_some() {
        let detail_lines = app.permission_details.as_ref().map(|d| d.len()).unwrap_or(0);
        (detail_lines as u16 + 4).min(f.area().height / 2) // +4 for borders, summary, controls
    } else {
        3
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),          // Header
            Constraint::Min(1),            // Messages
            Constraint::Length(input_height), // Input (expands for permissions)
            Constraint::Length(1),          // Status bar
        ])
        .split(f.area());

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" claux ", Style::default().fg(app.theme.assistant_bold).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("v{}", env!("CARGO_PKG_VERSION")),
            Style::default().fg(app.theme.dim),
        ),
    ]));
    f.render_widget(header, chunks[0]);

    // Messages area
    let msg_area = chunks[1];
    let _msg_width = msg_area.width.saturating_sub(2) as usize;

    let mut lines: Vec<Line> = Vec::new();

    for msg in &app.messages {
        // Add a blank line before each message
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }

        match msg.role.as_str() {
            "user" => {
                // Create a "bubble" effect with background color
                let bubble_lines: Vec<Line> = msg.content.lines().map(|line| {
                    Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default()
                            .fg(app.theme.user_message_fg)
                            .bg(app.theme.user_message_bg),
                    ))
                }).collect();
                
                // Add header line
                lines.push(Line::from(vec![
                    Span::styled("● ", Style::default().fg(app.theme.user)),
                    Span::styled("You", Style::default().fg(app.theme.user).add_modifier(Modifier::BOLD)),
                ]));
                
                // Add each line with background
                for line in bubble_lines {
                    lines.push(line);
                }
            }
            "assistant" => {
                lines.push(Line::from(vec![
                    Span::styled("● ", Style::default().fg(app.theme.assistant)),
                    Span::styled(format!("{} ", app.model), Style::default().fg(app.theme.assistant_bold).add_modifier(Modifier::BOLD)),
                ]));
                let rendered = markdown::render(&msg.content, Style::default().fg(app.theme.fg));
                // Indent assistant content to align with the dot
                for line in rendered {
                    let mut indented = vec![Span::raw("  ")];
                    indented.extend(line.spans);
                    lines.push(Line::from(indented));
                }
            }
            "system" => {
                lines.push(Line::from(Span::styled("● ", Style::default().fg(app.theme.warning))));
                for line in msg.content.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(app.theme.warning),
                    )));
                }
            }
            "error" => {
                lines.push(Line::from(Span::styled("● ", Style::default().fg(app.theme.error))));
                for line in msg.content.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(app.theme.error),
                    )));
                }
            }
            _ => {
                lines.push(Line::from(Span::styled("● ", Style::default().fg(app.theme.dim))));
                for line in msg.content.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(app.theme.fg),
                    )));
                }
            }
        }
    }

    // Streaming buffer (assistant response in progress)
    if !app.stream_buffer.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(vec![
            Span::styled("● ", Style::default().fg(app.theme.success)),
            Span::styled(format!("{} ", app.model), Style::default().fg(app.theme.success).add_modifier(Modifier::BOLD)),
        ]));
        let rendered = markdown::render(&app.stream_buffer, Style::default().fg(app.theme.success));
        for line in rendered {
            let mut indented = vec![Span::raw("  ")];
            indented.extend(line.spans);
            lines.push(Line::from(indented));
        }
        // Cursor indicator
        lines.push(Line::from(Span::styled("  ▊", Style::default().fg(app.theme.success))));
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
                .border_style(Style::default().fg(app.theme.dim)),
        )
        .scroll((scroll_offset, 0));
    f.render_widget(messages_widget, msg_area);

    // Input area
    let input_style = match app.mode {
        Mode::Input => Style::default().fg(app.theme.fg),
        Mode::Permission => Style::default().fg(app.theme.warning),
        Mode::Streaming => Style::default().fg(app.theme.dim),
    };

    if let (Some(ref prompt), Some(ref details)) = (&app.permission_prompt, &app.permission_details) {
        // Expanded permission panel
        let mut perm_lines: Vec<Line> = Vec::new();
        perm_lines.push(Line::from(vec![
            Span::styled("⚡ ", Style::default().fg(app.theme.warning)),
            Span::styled(prompt.as_str(), Style::default().fg(app.theme.warning).add_modifier(Modifier::BOLD)),
        ]));
        perm_lines.push(Line::from(""));
        for detail in details {
            let style = if detail.starts_with("  +") {
                Style::default().fg(app.theme.success)
            } else if detail.starts_with("  -") {
                Style::default().fg(app.theme.error)
            } else if detail.ends_with(':') {
                Style::default().fg(app.theme.fg).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.dim)
            };
            perm_lines.push(Line::from(Span::styled(detail.clone(), style)));
        }
        perm_lines.push(Line::from(""));
        // Show different options based on whether it's a bash command
        let help_text = if prompt.starts_with("bash:") {
            vec![
                Span::styled("  (y)es  ", Style::default().fg(app.theme.success)),
                Span::styled("(n)o  ", Style::default().fg(app.theme.error)),
                Span::styled("(a)lways this cmd  ", Style::default().fg(app.theme.warning)),
                Span::styled("(A)lways all bash", Style::default().fg(app.theme.info)),
            ]
        } else {
            vec![
                Span::styled("  (y)es  ", Style::default().fg(app.theme.success)),
                Span::styled("(n)o  ", Style::default().fg(app.theme.error)),
                Span::styled("(a)lways allow", Style::default().fg(app.theme.warning)),
            ]
        };
        perm_lines.push(Line::from(help_text));

        let perm_widget = Paragraph::new(perm_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(app.theme.warning))
                    .title(" Permission Required "),
            );
        f.render_widget(perm_widget, chunks[2]);
    } else {
        let input_text = if app.mode == Mode::Streaming {
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
                        app.theme.user
                    } else {
                        app.theme.dim
                    }))
                    .title(" > "),
            );
        f.render_widget(input_widget, chunks[2]);
    }

    // Set cursor position in input mode
    if app.mode == Mode::Input {
        f.set_cursor_position((
            chunks[2].x + app.cursor as u16 + 1,
            chunks[2].y + 1,
        ));
    }

    // Status bar
    let thinking_indicator = if app.thinking {
        // Pulsing spinner effect
        let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let idx = (chrono::Local::now().timestamp_millis() / 100) as usize % spinner.len();
        format!(" {} ", spinner[idx])
    } else {
        " ".to_string()
    };

    let status = Paragraph::new(Line::from(vec![
        Span::styled(thinking_indicator, Style::default().fg(app.theme.success)),
        Span::styled(format!(" {} ", app.status), Style::default().fg(app.theme.dim)),
    ]));
    f.render_widget(status, chunks[3]);
}
