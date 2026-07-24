//! Unicode-safe editing helpers for single-line terminal inputs.

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
}
