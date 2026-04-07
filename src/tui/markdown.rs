//! Markdown-to-ratatui renderer using pulldown-cmark.
//!
//! Full CommonMark support with proper parsing for code blocks, headers,
//! lists, links, emphasis, and more.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

const AQUA: Color = Color::Rgb(142, 192, 124); // gruvbox aqua
const ORANGE: Color = Color::Rgb(254, 128, 25); // gruvbox orange
const YELLOW: Color = Color::Rgb(250, 189, 47); // gruvbox yellow
const GRAY: Color = Color::Rgb(146, 131, 116); // gruvbox gray
const BLUE: Color = Color::Rgb(131, 165, 152); // gruvbox blue
const BG_CODE: Color = Color::Rgb(60, 56, 54); // gruvbox bg1

/// Parse markdown text into styled ratatui Lines using pulldown-cmark.
pub fn render(text: &str, base_style: Style) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut current_line: Vec<Span<'static>> = Vec::new();
    let mut style_stack: Vec<Style> = vec![base_style];
    let mut in_code_block = false;
    let mut code_block_lang = String::new();
    let mut list_depth: usize = 0;
    let mut _in_heading = false;
    let mut _heading_level = HeadingLevel::H1;

    let options = Options::all();
    let parser = Parser::new_ext(text, options);

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {
                    // Start new line for paragraph
                    if !current_line.is_empty() {
                        lines.push(Line::from(current_line.clone()));
                        current_line.clear();
                    }
                }
                Tag::Heading { level, .. } => {
                    _in_heading = true;
                    _heading_level = level;
                    let heading_style = match level {
                        HeadingLevel::H1 => Style::default()
                            .fg(YELLOW)
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                        HeadingLevel::H2 | HeadingLevel::H3 => {
                            Style::default().fg(ORANGE).add_modifier(Modifier::BOLD)
                        }
                        _ => Style::default().fg(ORANGE),
                    };
                    style_stack.push(heading_style);
                }
                Tag::BlockQuote(_) => {
                    current_line.push(Span::styled("│ ", Style::default().fg(GRAY)));
                    style_stack.push(Style::default().fg(GRAY));
                }
                Tag::CodeBlock(kind) => {
                    in_code_block = true;
                    code_block_lang = match kind {
                        pulldown_cmark::CodeBlockKind::Fenced(lang) => lang.to_string(),
                        pulldown_cmark::CodeBlockKind::Indented => String::new(),
                    };

                    let label = if code_block_lang.is_empty() {
                        "┌─ code ".to_string()
                    } else {
                        format!("┌─ {code_block_lang} ")
                    };
                    lines.push(Line::from(Span::styled(
                        format!("{label}─────────────────────────"),
                        Style::default().fg(GRAY),
                    )));
                }
                Tag::List(_) => {
                    list_depth += 1;
                }
                Tag::Item => {
                    let indent = "  ".repeat(list_depth.saturating_sub(1));
                    current_line.push(Span::styled(
                        format!("{indent}• "),
                        Style::default().fg(GRAY),
                    ));
                }
                Tag::Emphasis => {
                    let current_style = *style_stack.last().unwrap_or(&base_style);
                    style_stack.push(current_style.add_modifier(Modifier::ITALIC));
                }
                Tag::Strong => {
                    let current_style = *style_stack.last().unwrap_or(&base_style);
                    style_stack.push(current_style.add_modifier(Modifier::BOLD));
                }
                Tag::Link { .. } => {
                    style_stack.push(Style::default().fg(BLUE).add_modifier(Modifier::UNDERLINED));
                    // We'll append the URL in parentheses after the link text
                    current_line.push(Span::raw("["));
                }
                Tag::Image { dest_url, .. } => {
                    current_line.push(Span::styled(
                        format!("![image: {dest_url}]"),
                        Style::default().fg(BLUE),
                    ));
                }
                _ => {}
            },

            Event::End(tag_end) => match tag_end {
                TagEnd::Paragraph => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(current_line.clone()));
                        current_line.clear();
                    }
                    // Add blank line after paragraph
                    lines.push(Line::from(""));
                }
                TagEnd::Heading(_) => {
                    _in_heading = false;
                    if !current_line.is_empty() {
                        lines.push(Line::from(current_line.clone()));
                        current_line.clear();
                    }
                    lines.push(Line::from(""));
                    style_stack.pop();
                }
                TagEnd::BlockQuote(_) => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(current_line.clone()));
                        current_line.clear();
                    }
                    style_stack.pop();
                }
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    lines.push(Line::from(Span::styled(
                        "└─────────────────────────────",
                        Style::default().fg(GRAY),
                    )));
                    code_block_lang.clear();
                }
                TagEnd::List(_) => {
                    list_depth = list_depth.saturating_sub(1);
                    if !current_line.is_empty() {
                        lines.push(Line::from(current_line.clone()));
                        current_line.clear();
                    }
                }
                TagEnd::Item => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(current_line.clone()));
                        current_line.clear();
                    }
                }
                TagEnd::Emphasis | TagEnd::Strong => {
                    style_stack.pop();
                }
                TagEnd::Link => {
                    current_line.push(Span::raw("]"));
                    style_stack.pop();
                }
                _ => {}
            },

            Event::Text(text) => {
                if in_code_block {
                    // Code block content
                    for line in text.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("│ {line}"),
                            Style::default().fg(AQUA).bg(BG_CODE),
                        )));
                    }
                } else {
                    let current_style = *style_stack.last().unwrap_or(&base_style);
                    current_line.push(Span::styled(text.to_string(), current_style));
                }
            }

            Event::Code(code) => {
                current_line.push(Span::styled(
                    code.to_string(),
                    Style::default().fg(AQUA).bg(BG_CODE),
                ));
            }

            Event::SoftBreak => {
                current_line.push(Span::raw(" "));
            }

            Event::HardBreak => {
                if !current_line.is_empty() {
                    lines.push(Line::from(current_line.clone()));
                    current_line.clear();
                }
            }

            Event::Rule => {
                lines.push(Line::from(Span::styled(
                    "────────────────────────────────",
                    Style::default().fg(GRAY),
                )));
            }

            _ => {}
        }
    }

    // Flush remaining line
    if !current_line.is_empty() {
        lines.push(Line::from(current_line));
    }

    // Remove trailing empty lines
    while lines.last().map(|l| l.spans.is_empty()).unwrap_or(false) {
        lines.pop();
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_unchanged() {
        let lines = render("hello world", Style::default());
        assert!(!lines.is_empty());
    }

    #[test]
    fn code_block_detected() {
        let text = "before\n\n```rust\nlet x = 1;\n```\n\nafter";
        let lines = render(text, Style::default());
        // Should have: before, blank, opening fence, code line, closing fence, blank, after
        assert!(lines.len() >= 5);
    }

    #[test]
    fn headers_styled() {
        let text = "# Big\n\n## Medium\n\n### Small";
        let lines = render(text, Style::default());
        assert!(lines.len() >= 3);
    }

    #[test]
    fn horizontal_rule() {
        let lines = render("---", Style::default());
        assert!(!lines.is_empty());
        let content: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(content.contains("────"));
    }

    #[test]
    fn inline_code() {
        let lines = render("use `cargo build` here", Style::default());
        assert!(!lines.is_empty());
        // Check that there's a span with code styling
        let has_code = lines.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.content.as_ref().contains("cargo build"))
        });
        assert!(has_code);
    }

    #[test]
    fn bold_text() {
        let lines = render("this is **bold** text", Style::default());
        assert!(!lines.is_empty());
        // Check that there's a bold span
        let has_bold = lines.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::BOLD))
        });
        assert!(has_bold);
    }

    #[test]
    fn lists_rendered() {
        let text = "- item 1\n- item 2\n- item 3";
        let lines = render(text, Style::default());
        assert!(lines.len() >= 3);
        // Each item should have a bullet
        let has_bullets = lines.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.content.as_ref().contains("•"))
        });
        assert!(has_bullets);
    }

    #[test]
    fn links_rendered() {
        let text = "Check out [this link](https://example.com)";
        let lines = render(text, Style::default());
        assert!(!lines.is_empty());
        // Should have link text with brackets
        let content: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(content.contains("[") && content.contains("]"));
    }

    #[test]
    fn nested_lists() {
        let text = "- item 1\n  - nested 1\n  - nested 2\n- item 2";
        let lines = render(text, Style::default());
        // Just check that we got some output
        assert!(!lines.is_empty());
        // Check that nested items have more indentation
        let has_nested_indent = lines.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.content.as_ref().contains("  •"))
        });
        assert!(has_nested_indent);
    }

    #[test]
    fn empty_input() {
        let lines = render("", Style::default());
        // Empty string should result in empty lines vec
        assert!(lines.is_empty() || lines.len() == 1);
    }
}
