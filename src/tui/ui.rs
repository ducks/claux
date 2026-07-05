//! Shared UI drawing helpers for the TUI.

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::chat::{ChatApp, ChatMessage, Mode, ToolStatus};
use super::markdown;

/// Rendered lines for `ChatApp::messages`, cached across frames.
///
/// Rendering history means a markdown parse per message plus a word-wrap
/// pass over everything; at a few hundred messages that costs tens of
/// milliseconds, and draw_chat runs on a 50ms tick while streaming. The
/// cache makes each frame O(streaming tail): it is rebuilt only when the
/// message list changes (`rev`) or the text width changes.
pub struct HistoryCache {
    pub rev: u64,
    pub width: u16,
    pub lines: Vec<Line<'static>>,
    /// Rendered row count of each line at `width`, so the draw path can
    /// select just the visible window without re-wrapping history.
    pub line_rows: Vec<u16>,
    pub rows: u16,
}

/// Exact rendered row count for one line word-wrapped at `width`.
/// WordWrapper wraps logical lines independently, so per-line counts of
/// separate slices add up exactly.
fn count_line_rows(line: &Line<'static>, width: u16) -> u16 {
    Paragraph::new(vec![line.clone()])
        .wrap(Wrap { trim: false })
        .line_count(width) as u16
}

/// Draw the chat screen.
pub fn draw_chat(f: &mut Frame, app: &mut ChatApp) {
    let input_height = if app.permission_details.is_some() {
        let detail_lines = app
            .permission_details
            .as_ref()
            .map(|d| d.len())
            .unwrap_or(0);
        // Content: summary + blank + details + blank + y/n/a options,
        // plus 2 border rows. Undersizing this clips the options line,
        // leaving the user with a question and no visible answers.
        (detail_lines as u16 + 6).min(f.area().height / 2)
    } else {
        3
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(f.area());

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " claux ",
            Style::default()
                .fg(app.theme.assistant_bold)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("v{}", app.version),
            Style::default().fg(app.theme.dim),
        ),
    ]));
    f.render_widget(header, chunks[0]);

    // Messages area
    let msg_area = chunks[1];
    let inner_width = msg_area.width.saturating_sub(2).max(1);

    // Rebuild the history cache only when messages or width changed
    let cache_valid = app
        .history_cache
        .as_ref()
        .is_some_and(|c| c.rev == app.messages_rev && c.width == inner_width);
    if !cache_valid {
        let lines = history_lines(app);
        let line_rows: Vec<u16> = lines
            .iter()
            .map(|l| count_line_rows(l, inner_width))
            .collect();
        let rows = line_rows.iter().sum();
        app.history_cache = Some(HistoryCache {
            rev: app.messages_rev,
            width: inner_width,
            lines,
            line_rows,
            rows,
        });
    }

    // Streaming buffer (the per-frame tail, rendered fresh each draw)
    let history_empty = app.messages.is_empty();
    let mut tail_lines: Vec<Line> = Vec::new();
    if !app.stream_buffer.is_empty() {
        if !history_empty {
            tail_lines.push(Line::from(""));
        }
        tail_lines.push(Line::from(vec![
            Span::styled("● ", Style::default().fg(app.theme.success)),
            Span::styled(
                format!("{} ", app.model),
                Style::default()
                    .fg(app.theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        let rendered = markdown::render(&app.stream_buffer, Style::default().fg(app.theme.success));
        for line in rendered {
            let mut indented = vec![Span::raw("  ")];
            indented.extend(line.spans);
            tail_lines.push(Line::from(indented));
        }
        tail_lines.push(Line::from(Span::styled(
            "  ▊",
            Style::default().fg(app.theme.success),
        )));
    }
    let tail_rows: Vec<u16> = tail_lines
        .iter()
        .map(|l| count_line_rows(l, inner_width))
        .collect();

    if !app.manual_scroll {
        app.scroll = 0;
    }
    let visible_height = msg_area.height;
    let manual_scroll = app.manual_scroll;
    let user_scroll = app.scroll;

    // Select only the logical lines that intersect the viewport and hand
    // the renderer a local offset. Passing everything makes the render
    // itself O(history): ratatui re-wraps every row above the scroll
    // offset each frame to find the window.
    let (render_lines, local_offset, total_rows) = {
        let cache = app
            .history_cache
            .as_ref()
            .expect("history cache just built");

        let total_rows = cache.rows + tail_rows.iter().sum::<u16>();
        let max_scroll = total_rows.saturating_sub(visible_height);
        let scroll_offset = if manual_scroll {
            max_scroll.saturating_sub(user_scroll.min(max_scroll))
        } else {
            max_scroll
        };

        let line_count = cache.lines.len() + tail_lines.len();
        let rows_at = |i: usize| -> u16 {
            if i < cache.line_rows.len() {
                cache.line_rows[i]
            } else {
                tail_rows[i - cache.line_rows.len()]
            }
        };

        // Skip whole lines that end above the viewport
        let mut first = 0usize;
        let mut skipped: u16 = 0;
        while first < line_count && skipped + rows_at(first) <= scroll_offset {
            skipped += rows_at(first);
            first += 1;
        }
        let local_offset = scroll_offset - skipped;

        // Take lines until the viewport is covered
        let mut render_lines: Vec<Line<'static>> = Vec::new();
        let mut covered: u16 = 0;
        let needed = local_offset.saturating_add(visible_height);
        let mut i = first;
        while i < line_count && covered < needed {
            let line = if i < cache.lines.len() {
                cache.lines[i].clone()
            } else {
                tail_lines[i - cache.lines.len()].clone()
            };
            covered += rows_at(i);
            render_lines.push(line);
            i += 1;
        }

        (render_lines, local_offset, total_rows)
    };

    app.total_lines = total_rows;

    let messages_widget = Paragraph::new(render_lines)
        .block(
            Block::default()
                .borders(Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(app.theme.dim)),
        )
        .wrap(Wrap { trim: false })
        .scroll((local_offset, 0));
    f.render_widget(messages_widget, msg_area);

    draw_input_and_status(f, app, &chunks);
}

/// Render `app.messages` into styled lines. Called only on cache misses.
fn history_lines(app: &ChatApp) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

    for msg in &app.messages {
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }

        match msg {
            ChatMessage::Text { role, content } => match role.as_str() {
                "user" => {
                    let bubble_lines: Vec<Line> = content
                        .lines()
                        .map(|line| {
                            Line::from(Span::styled(
                                format!("  {line}"),
                                Style::default()
                                    .fg(app.theme.user_message_fg)
                                    .bg(app.theme.user_message_bg),
                            ))
                        })
                        .collect();

                    lines.push(Line::from(vec![
                        Span::styled("● ", Style::default().fg(app.theme.user)),
                        Span::styled(
                            "You",
                            Style::default()
                                .fg(app.theme.user)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                    for line in bubble_lines {
                        lines.push(line);
                    }
                }
                "assistant" => {
                    lines.push(Line::from(vec![
                        Span::styled("● ", Style::default().fg(app.theme.assistant)),
                        Span::styled(
                            format!("{} ", app.model),
                            Style::default()
                                .fg(app.theme.assistant_bold)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                    let rendered = markdown::render(content, Style::default().fg(app.theme.fg));
                    for line in rendered {
                        let mut indented = vec![Span::raw("  ")];
                        indented.extend(line.spans);
                        lines.push(Line::from(indented));
                    }
                }
                "system" => {
                    lines.push(Line::from(Span::styled(
                        "● ",
                        Style::default().fg(app.theme.warning),
                    )));
                    for line in content.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("  {line}"),
                            Style::default().fg(app.theme.warning),
                        )));
                    }
                }
                "error" => {
                    lines.push(Line::from(Span::styled(
                        "● ",
                        Style::default().fg(app.theme.error),
                    )));
                    for line in content.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("  {line}"),
                            Style::default().fg(app.theme.error),
                        )));
                    }
                }
                _ => {
                    lines.push(Line::from(Span::styled(
                        "● ",
                        Style::default().fg(app.theme.dim),
                    )));
                    for line in content.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("  {line}"),
                            Style::default().fg(app.theme.fg),
                        )));
                    }
                }
            },
            ChatMessage::Tool {
                name,
                summary,
                status,
            } => {
                let (indicator, indicator_color) = match status {
                    ToolStatus::Running => ("⟳", app.theme.warning),
                    ToolStatus::Success => ("●", app.theme.tool_success),
                    ToolStatus::Error => ("✗", app.theme.tool_error),
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{indicator} "),
                        Style::default().fg(indicator_color),
                    ),
                    Span::styled(
                        format!("{name} "),
                        Style::default()
                            .fg(app.theme.tool_name)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(summary.clone(), Style::default().fg(app.theme.tool_summary)),
                ]));
            }
        }
    }

    lines
}

/// Draw the input (or permission) box and the status bar.
fn draw_input_and_status(f: &mut Frame, app: &mut ChatApp, chunks: &[ratatui::layout::Rect]) {
    // Input area
    if let (Some(ref prompt), Some(ref details)) = (&app.permission_prompt, &app.permission_details)
    {
        let mut perm_lines: Vec<Line> = Vec::new();
        perm_lines.push(Line::from(vec![
            Span::styled("⚡ ", Style::default().fg(app.theme.warning)),
            Span::styled(
                prompt.as_str(),
                Style::default()
                    .fg(app.theme.warning)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        perm_lines.push(Line::from(""));
        for detail in details {
            let style = if detail.starts_with("  +") {
                Style::default().fg(app.theme.success)
            } else if detail.starts_with("  -") {
                Style::default().fg(app.theme.error)
            } else if detail.ends_with(':') {
                Style::default()
                    .fg(app.theme.fg)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.dim)
            };
            perm_lines.push(Line::from(Span::styled(detail.clone(), style)));
        }
        perm_lines.push(Line::from(""));
        perm_lines.push(Line::from(vec![
            Span::styled("  (y)es  ", Style::default().fg(app.theme.success)),
            Span::styled("(n)o  ", Style::default().fg(app.theme.error)),
            Span::styled("(a)lways allow", Style::default().fg(app.theme.warning)),
        ]));

        let perm_widget = Paragraph::new(perm_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(app.theme.warning))
                .title(" Permission Required "),
        );
        f.render_widget(perm_widget, chunks[2]);
    } else {
        let input_text = if app.mode == Mode::Streaming {
            let steer = app.steer_buf.lock().expect("steer buffer poisoned");
            if steer.is_empty() {
                "... (type to steer, Enter to queue, Ctrl+C to interrupt)".to_string()
            } else {
                steer.clone()
            }
        } else {
            app.input.clone()
        };

        let input_widget = Paragraph::new(input_text)
            .style(Style::default().fg(app.theme.fg))
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

    if app.mode == Mode::Input {
        f.set_cursor_position((chunks[2].x + app.cursor as u16 + 1, chunks[2].y + 1));
    }

    // Status bar
    let thinking_indicator = if app.thinking {
        let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let idx = (chrono::Local::now().timestamp_millis() / 100) as usize % spinner.len();
        format!(" {} ", spinner[idx])
    } else {
        " ".to_string()
    };

    let status = Paragraph::new(Line::from(vec![
        Span::styled(thinking_indicator, Style::default().fg(app.theme.success)),
        Span::styled(
            format!(" {} ", app.status),
            Style::default().fg(app.theme.dim),
        ),
    ]));
    f.render_widget(status, chunks[3]);
}

#[cfg(test)]
mod perf_probe {
    use super::*;
    use crate::theme::Theme;

    #[test]
    fn time_draw_with_large_history() {
        let mut app = ChatApp::new("test-model", Theme::dark());
        for i in 0..150 {
            app.add_message(
                "user",
                &format!(
                    "question {i} about the codebase, with some length to it so wrapping happens"
                ),
            );
            app.add_message(
                "assistant",
                &format!("Here is a **detailed** answer {i} with `inline code`, a list:\n- point one about the design\n- point two with more words that will wrap on narrow terminals\n\nAnd a table:\n| a | b |\n|---|---|\n| {i} | value |\n\nPlus a trailing paragraph that is long enough to wrap at eighty columns for sure, repeating itself to add width and weight to the rendering cost."),
            );
            app.add_tool("Bash", "cargo test", ToolStatus::Success);
        }
        app.stream_buffer = "streaming tail ".repeat(20);
        app.mode = Mode::Streaming;

        // warm up
        let _ = tuishot::render_to_buffer(120, 40, |f| draw_chat(f, &mut app));

        let n = 20;
        let start = std::time::Instant::now();
        for _ in 0..n {
            let _ = tuishot::render_to_buffer(120, 40, |f| draw_chat(f, &mut app));
        }
        let per_draw = start.elapsed() / n;
        println!("per-draw: {per_draw:?} for {} messages", app.messages.len());

        // Regression guard: warm-cache draws must stay O(viewport), not
        // O(history). Before the history cache + viewport slicing this
        // measured ~58ms in a debug build; sliced it's ~3ms. The bound is
        // loose to tolerate slow CI, but tight enough to catch a return
        // to per-frame full-history rendering.
        assert!(
            per_draw < std::time::Duration::from_millis(25),
            "draw with large history too slow: {per_draw:?}"
        );
    }
}
