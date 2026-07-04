//! Utility modules for claux.

pub mod diff;

/// Largest prefix of `s` that fits in `max_bytes`, cut on a char boundary.
/// Byte-indexed slicing (`&s[..n]`) panics mid-codepoint; tool output,
/// commands, and git status all carry arbitrary UTF-8, so every truncation
/// must go through here or `tail_str`.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Largest suffix of `s` that fits in `max_bytes`, cut on a char boundary.
pub fn tail_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut start = s.len() - max_bytes;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_ascii() {
        assert_eq!(truncate_str("hello", 3), "hel");
        assert_eq!(truncate_str("hello", 5), "hello");
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_multibyte_boundary() {
        // "é" is 2 bytes; cutting at byte 1 must back off to 0
        assert_eq!(truncate_str("é", 1), "");
        // cutting "aéb" at byte 2 lands mid-'é', backs off to "a"
        assert_eq!(truncate_str("aéb", 2), "a");
        // 4-byte emoji
        assert_eq!(truncate_str("🦀🦀", 5), "🦀");
        assert_eq!(truncate_str("🦀🦀", 4), "🦀");
        assert_eq!(truncate_str("🦀🦀", 3), "");
    }

    #[test]
    fn tail_ascii() {
        assert_eq!(tail_str("hello", 3), "llo");
        assert_eq!(tail_str("hello", 5), "hello");
        assert_eq!(tail_str("hello", 10), "hello");
    }

    #[test]
    fn tail_multibyte_boundary() {
        // a suffix cut landing mid-'é' must move forward past it
        assert_eq!(tail_str("aéb", 2), "b");
        assert_eq!(tail_str("🦀🦀", 5), "🦀");
        assert_eq!(tail_str("🦀🦀", 3), "");
    }

    #[test]
    fn empty_string() {
        assert_eq!(truncate_str("", 5), "");
        assert_eq!(tail_str("", 5), "");
    }
}
