//! Simple markdown-to-ratatui renderer.
//!
//! Handles: code blocks, inline code, bold, headers, horizontal rules.
//! Not a full markdown parser — just enough to make LLM output readable.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

const FG: Color = Color::Rgb(213, 196, 161); // gruvbox fg2
const AQUA: Color = Color::Rgb(142, 192, 124); // gruvbox aqua
const ORANGE: Color = Color::Rgb(254, 128, 25); // gruvbox orange
const YELLOW: Color = Color::Rgb(250, 189, 47); // gruvbox yellow
const GRAY: Color = Color::Rgb(146, 131, 116); // gruvbox gray
const BG_CODE: Color = Color::Rgb(60, 56, 54); // gruvbox bg1

/// Parse markdown text into styled ratatui Lines.
pub fn render(text: &str, base_style: Style) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();

    for raw_line in text.lines() {
        // Code block toggle
        if raw_line.trim_start().starts_with("```") {
            if in_code_block {
                // Closing fence
                lines.push(Line::from(Span::styled(
                    "└─────────────────────────────",
                    Style::default().fg(GRAY),
                )));
                in_code_block = false;
                code_lang.clear();
            } else {
                // Opening fence
                code_lang = raw_line.trim_start().strip_prefix("```").unwrap_or("").to_string();
                let label = if code_lang.is_empty() {
                    "┌─ code ".to_string()
                } else {
                    format!("┌─ {} ", code_lang)
                };
                lines.push(Line::from(Span::styled(
                    format!("{}─────────────────────────", label),
                    Style::default().fg(GRAY),
                )));
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            // Code block content — monospace, dimmed background color
            lines.push(Line::from(Span::styled(
                format!("│ {}", raw_line),
                Style::default().fg(AQUA).bg(BG_CODE),
            )));
            continue;
        }

        // Headers
        if raw_line.starts_with("### ") {
            lines.push(Line::from(Span::styled(
                raw_line[4..].to_string(),
                Style::default().fg(ORANGE).add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if raw_line.starts_with("## ") {
            lines.push(Line::from(Span::styled(
                raw_line[3..].to_string(),
                Style::default().fg(ORANGE).add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if raw_line.starts_with("# ") {
            lines.push(Line::from(Span::styled(
                raw_line[2..].to_string(),
                Style::default()
                    .fg(YELLOW)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )));
            continue;
        }

        // Horizontal rule
        if raw_line.trim() == "---" || raw_line.trim() == "***" || raw_line.trim() == "___" {
            lines.push(Line::from(Span::styled(
                "────────────────────────────────",
                Style::default().fg(GRAY),
            )));
            continue;
        }

        // Inline formatting: parse spans within the line
        lines.push(parse_inline(raw_line, base_style));
    }

    // Close unclosed code block
    if in_code_block {
        lines.push(Line::from(Span::styled(
            "└─────────────────────────────",
            Style::default().fg(GRAY),
        )));
    }

    lines
}

/// Parse inline markdown: **bold**, `code`, *italic*.
fn parse_inline(line: &str, base_style: Style) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut chars = line.char_indices().peekable();
    let mut current = String::new();

    while let Some((i, ch)) = chars.next() {
        match ch {
            '`' => {
                // Inline code
                if !current.is_empty() {
                    spans.push(Span::styled(current.clone(), base_style));
                    current.clear();
                }
                let mut code = String::new();
                for (_, c) in chars.by_ref() {
                    if c == '`' {
                        break;
                    }
                    code.push(c);
                }
                if !code.is_empty() {
                    spans.push(Span::styled(
                        code,
                        Style::default().fg(AQUA).bg(BG_CODE),
                    ));
                }
            }
            '*' => {
                // Check for ** (bold) vs * (italic)
                if chars.peek().map(|(_, c)| *c) == Some('*') {
                    // Bold
                    chars.next(); // consume second *
                    if !current.is_empty() {
                        spans.push(Span::styled(current.clone(), base_style));
                        current.clear();
                    }
                    let mut bold = String::new();
                    loop {
                        match chars.next() {
                            Some((_, '*')) if chars.peek().map(|(_, c)| *c) == Some('*') => {
                                chars.next(); // consume closing **
                                break;
                            }
                            Some((_, c)) => bold.push(c),
                            None => break,
                        }
                    }
                    if !bold.is_empty() {
                        spans.push(Span::styled(
                            bold,
                            base_style.add_modifier(Modifier::BOLD),
                        ));
                    }
                } else {
                    // Italic
                    if !current.is_empty() {
                        spans.push(Span::styled(current.clone(), base_style));
                        current.clear();
                    }
                    let mut italic = String::new();
                    for (_, c) in chars.by_ref() {
                        if c == '*' {
                            break;
                        }
                        italic.push(c);
                    }
                    if !italic.is_empty() {
                        spans.push(Span::styled(
                            italic,
                            base_style.add_modifier(Modifier::ITALIC),
                        ));
                    }
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        spans.push(Span::styled(current, base_style));
    }

    if spans.is_empty() {
        Line::from("")
    } else {
        Line::from(spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_unchanged() {
        let lines = render("hello world", Style::default());
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn code_block_detected() {
        let text = "before\n```rust\nlet x = 1;\n```\nafter";
        let lines = render(text, Style::default());
        // before, opening fence, code line, closing fence, after
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn headers_styled() {
        let text = "# Big\n## Medium\n### Small";
        let lines = render(text, Style::default());
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn horizontal_rule() {
        let lines = render("---", Style::default());
        assert_eq!(lines.len(), 1);
        let content: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(content.contains("────"));
    }

    #[test]
    fn inline_code() {
        let line = parse_inline("use `cargo build` here", Style::default());
        assert!(line.spans.len() >= 3); // "use " + "cargo build" + " here"
    }

    #[test]
    fn bold_text() {
        let line = parse_inline("this is **bold** text", Style::default());
        assert!(line.spans.len() >= 3);
        // The bold span content should be "bold"
        let bold_span = &line.spans[1];
        assert_eq!(bold_span.content.as_ref(), "bold");
    }

    #[test]
    fn empty_input() {
        let lines = render("", Style::default());
        // Empty string has one empty line from .lines()
        assert!(lines.len() <= 1);
    }

    #[test]
    fn unclosed_code_block_closed() {
        let text = "```\ncode here\nmore code";
        let lines = render(text, Style::default());
        // opening fence, 2 code lines, auto-closing fence
        assert_eq!(lines.len(), 4);
    }
}
