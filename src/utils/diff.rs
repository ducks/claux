//! Diff generation utilities for showing file changes before applying them.

use similar::TextDiff;

/// Generate a unified diff between old and new content.
///
/// # Arguments
/// * `old` - The original file content
/// * `new` - The proposed new content
/// * `path` - The file path to display in the diff header
///
/// # Returns
/// A string containing the unified diff with context lines.
pub fn generate_diff(old: &str, new: &str, path: &str) -> String {
    let diff = TextDiff::from_lines(old, new);

    let mut output = format!("--- a/{path}\n+++ b/{path}\n");

    for change in diff.unified_diff().context_radius(3).iter_hunks() {
        output.push_str(&format!("{change}"));
    }

    output
}

/// Colorize a diff string for terminal output.
///
/// Returns a string with ANSI color codes:
/// - Red for removed lines (-)
/// - Green for added lines (+)
/// - No color for context lines ( )
pub fn colorize_diff(diff: &str) -> String {
    let mut output = String::new();

    for line in diff.lines() {
        let colored_line = if line.starts_with('+') && !line.starts_with("+++") {
            format!("\x1b[32m{line}\x1b[0m") // Green for additions
        } else if line.starts_with('-') && !line.starts_with("---") {
            format!("\x1b[31m{line}\x1b[0m") // Red for deletions
        } else {
            line.to_string() // No color for context
        };

        output.push_str(&colored_line);
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_diff_simple() {
        let old = "hello\nworld\n";
        let new = "hello\nuniverse\n";
        let diff = generate_diff(old, new, "test.txt");

        assert!(diff.contains("--- a/test.txt"));
        assert!(diff.contains("+++ b/test.txt"));
        assert!(diff.contains("-world"));
        assert!(diff.contains("+universe"));
    }

    #[test]
    fn test_generate_diff_addition() {
        let old = "line1\nline2\n";
        let new = "line1\nline1.5\nline2\n";
        let diff = generate_diff(old, new, "test.txt");

        assert!(diff.contains("+line1.5"));
    }

    #[test]
    fn test_generate_diff_deletion() {
        let old = "line1\nline2\nline3\n";
        let new = "line1\nline3\n";
        let diff = generate_diff(old, new, "test.txt");

        assert!(diff.contains("-line2"));
    }

    #[test]
    fn test_colorize_diff() {
        let diff = "--- a/test.txt\n+++ b/test.txt\n-old\n+new\n";
        let colored = colorize_diff(&diff);

        assert!(colored.contains("\x1b[31m-old\x1b[0m"));
        assert!(colored.contains("\x1b[32m+new\x1b[0m"));
    }

    #[test]
    fn test_colorize_diff_headers() {
        let diff = "--- a/test.txt\n+++ b/test.txt\n";
        let colored = colorize_diff(&diff);

        // Headers should not be colored
        assert!(!colored.contains("\x1b[31m---"));
        assert!(!colored.contains("\x1b[32m+++"));
    }
}
