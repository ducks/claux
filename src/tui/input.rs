//! Unicode-safe editing helpers for terminal inputs.

use unicode_width::UnicodeWidthChar;

pub fn char_count(input: &str) -> usize {
    input.chars().count()
}

fn byte_index(input: &str, cursor: usize) -> usize {
    input
        .char_indices()
        .nth(cursor)
        .map(|(index, _)| index)
        .unwrap_or(input.len())
}

pub fn insert(input: &mut String, cursor: &mut usize, character: char) {
    input.insert(byte_index(input, *cursor), character);
    *cursor += 1;
}

/// Insert a pasted string at the cursor without routing it through terminal
/// key events. Bracketed paste can contain newlines and thousands of
/// characters, so inserting it as one string avoids event-by-event overhead.
pub fn insert_text(input: &mut String, cursor: &mut usize, text: &str) {
    let byte_index = input
        .char_indices()
        .nth(*cursor)
        .map(|(index, _)| index)
        .unwrap_or(input.len());
    input.insert_str(byte_index, text);
    *cursor += text.chars().count();
}

pub fn backspace(input: &mut String, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }

    *cursor -= 1;
    input.remove(byte_index(input, *cursor));
}

pub fn delete(input: &mut String, cursor: usize) {
    if cursor < char_count(input) {
        input.remove(byte_index(input, cursor));
    }
}

pub fn display_width_before(input: &str, cursor: usize) -> usize {
    input
        .chars()
        .take(cursor)
        .map(|character| character.width().unwrap_or(0))
        .sum()
}

/// Display-only layout for the prompt editor. Soft wrapping never inserts
/// newlines into the submitted input.
pub struct VisualLayout {
    pub lines: Vec<String>,
    pub cursor_row: usize,
    pub cursor_col: usize,
}

/// Wrap input at terminal-cell boundaries and locate the insertion cursor.
/// Explicit newlines (from pasted text) remain hard line breaks; ordinary long
/// lines wrap visually, like an editor with `wrap` enabled.
pub fn visual_layout(input: &str, cursor: usize, width: usize) -> VisualLayout {
    let width = width.max(1);
    let mut lines = vec![String::new()];
    let mut row = 0;
    let mut col = 0;
    let mut cursor_position = None;

    for (index, character) in input.chars().enumerate() {
        if index == cursor {
            cursor_position = Some(normalize_cursor(row, col, width));
        }

        if character == '\n' {
            lines.push(String::new());
            row += 1;
            col = 0;
            continue;
        }

        let character_width = character.width().unwrap_or(0);
        if character_width > 0 && col + character_width > width {
            lines.push(String::new());
            row += 1;
            col = 0;
        }
        lines[row].push(character);
        col += character_width;
    }

    let (cursor_row, cursor_col) =
        cursor_position.unwrap_or_else(|| normalize_cursor(row, col, width));
    while lines.len() <= cursor_row {
        lines.push(String::new());
    }

    VisualLayout {
        lines,
        cursor_row,
        cursor_col,
    }
}

fn normalize_cursor(row: usize, col: usize, width: usize) -> (usize, usize) {
    if col >= width {
        (row + col / width, col % width)
    } else {
        (row, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_multibyte_characters_at_character_boundaries() {
        let mut input = "aé界".to_string();
        let mut cursor = 2;

        insert(&mut input, &mut cursor, '🙂');
        assert_eq!(input, "aé🙂界");
        assert_eq!(cursor, 3);

        backspace(&mut input, &mut cursor);
        assert_eq!(input, "aé界");
        assert_eq!(cursor, 2);

        delete(&mut input, cursor);
        assert_eq!(input, "aé");
    }

    #[test]
    fn display_width_handles_wide_and_combining_characters() {
        assert_eq!(display_width_before("a界e\u{301}", 4), 4);
    }

    #[test]
    fn insert_text_inserts_at_character_cursor() {
        let mut input = "a界c".to_string();
        let mut cursor = 2;
        insert_text(&mut input, &mut cursor, "é\n");
        assert_eq!(input, "a界é\nc");
        assert_eq!(cursor, 4);
    }

    #[test]
    fn visually_wraps_without_changing_the_input() {
        let input = "abcdefghij";
        let layout = visual_layout(input, input.chars().count(), 4);

        assert_eq!(layout.lines, ["abcd", "efgh", "ij"]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (2, 2));
        assert_eq!(input, "abcdefghij");
    }

    #[test]
    fn cursor_tracks_wrapped_rows_when_moving_through_input() {
        let at_end = visual_layout("abcdefghij", 10, 4);
        assert_eq!((at_end.cursor_row, at_end.cursor_col), (2, 2));

        let near_start = visual_layout("abcdefghij", 2, 4);
        assert_eq!((near_start.cursor_row, near_start.cursor_col), (0, 2));
    }

    #[test]
    fn explicit_newlines_are_hard_breaks() {
        let layout = visual_layout("abc\ndefgh", 9, 3);
        assert_eq!(layout.lines, ["abc", "def", "gh"]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (2, 2));
    }

    #[test]
    fn wide_characters_wrap_by_terminal_width() {
        let layout = visual_layout("a界bc", 4, 3);
        assert_eq!(layout.lines, ["a界", "bc"]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (1, 2));
    }
}
