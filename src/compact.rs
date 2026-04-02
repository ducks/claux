//! Compaction strategies for managing context window usage.
//!
//! Mirrors Claude Code's multi-strategy compaction pipeline:
//! 1. Tool output truncation — cap large tool results before they enter history
//! 2. Snip compaction — collapse old messages keeping recent N
//! 3. Full summary — summarize entire conversation via API call
//! 4. Reactive compact — triggered on prompt-too-long (413) errors

use crate::api::types::{ContentBlock, Message, MessageContent};

/// Maximum characters for a single tool result before truncation.
const TOOL_OUTPUT_MAX_CHARS: usize = 30_000;

/// Rough estimate: 1 token ≈ 4 characters.
const CHARS_PER_TOKEN: usize = 4;

/// Estimate the token count of the conversation.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    let mut chars = 0;
    for msg in messages {
        match &msg.content {
            MessageContent::Text(text) => chars += text.len(),
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => chars += text.len(),
                        ContentBlock::ToolUse { input, name, .. } => {
                            chars += name.len() + input.to_string().len();
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            chars += content.len();
                        }
                    }
                }
            }
        }
    }
    chars / CHARS_PER_TOKEN
}

/// Truncate a tool output string if it exceeds the maximum.
/// Returns the (possibly truncated) string and whether it was truncated.
pub fn truncate_tool_output(output: &str) -> (String, bool) {
    if output.len() <= TOOL_OUTPUT_MAX_CHARS {
        return (output.to_string(), false);
    }

    // Keep first and last portions with a truncation marker
    let keep_start = TOOL_OUTPUT_MAX_CHARS * 2 / 3;
    let keep_end = TOOL_OUTPUT_MAX_CHARS / 6;

    let start = &output[..keep_start];
    let end = &output[output.len() - keep_end..];
    let truncated_chars = output.len() - keep_start - keep_end;

    let result = format!(
        "{}\n\n... ({} characters truncated) ...\n\n{}",
        start, truncated_chars, end
    );

    (result, true)
}

/// Snip compaction: collapse old messages into a brief marker,
/// keeping the most recent `keep_recent` messages intact.
/// Returns the new message list if snipping occurred, or None if
/// there weren't enough messages to snip.
pub fn snip_old_messages(messages: &[Message], keep_recent: usize) -> Option<Vec<Message>> {
    if messages.len() <= keep_recent + 2 {
        return None; // Not enough to snip
    }

    let snip_count = messages.len() - keep_recent;
    let snipped = &messages[..snip_count];
    let kept = &messages[snip_count..];

    // Count what we're removing
    let snip_tokens = estimate_tokens(snipped);

    let marker = Message::user(&format!(
        "[{} earlier messages snipped (~{} tokens). The conversation continues below.]",
        snip_count, snip_tokens
    ));

    let mut result = vec![marker];
    result.extend_from_slice(kept);

    Some(result)
}

/// Determine if compaction should be triggered based on estimated token usage.
/// Returns the recommended strategy.
pub enum CompactStrategy {
    /// No compaction needed
    None,
    /// Snip old messages (light)
    Snip,
    /// Full summarization (heavy)
    Summarize,
}

/// Check what compaction strategy to use based on token estimates.
/// Uses model context window as reference.
pub fn should_compact(messages: &[Message], context_window: usize) -> CompactStrategy {
    let tokens = estimate_tokens(messages);
    let threshold_snip = context_window * 60 / 100; // 60% — snip
    let threshold_summarize = context_window * 80 / 100; // 80% — full summary

    if tokens > threshold_summarize {
        CompactStrategy::Summarize
    } else if tokens > threshold_snip {
        CompactStrategy::Snip
    } else {
        CompactStrategy::None
    }
}

/// Context window sizes for known models.
pub fn context_window_for_model(model: &str) -> usize {
    if model.contains("opus") {
        200_000
    } else if model.contains("sonnet") {
        200_000
    } else if model.contains("haiku") {
        200_000
    } else if model.contains("gpt-4o") {
        128_000
    } else if model.contains("gpt-4") {
        128_000
    } else if model.contains("gpt-3.5") {
        16_000
    } else {
        // Conservative default for unknown models
        128_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(&[]), 0);
    }

    #[test]
    fn estimate_tokens_text() {
        let msgs = vec![Message::user("hello world")]; // 11 chars ≈ 2-3 tokens
        let tokens = estimate_tokens(&msgs);
        assert!(tokens > 0);
        assert!(tokens < 10);
    }

    #[test]
    fn truncate_short_output_unchanged() {
        let (result, truncated) = truncate_tool_output("short");
        assert_eq!(result, "short");
        assert!(!truncated);
    }

    #[test]
    fn truncate_long_output() {
        let long = "x".repeat(50_000);
        let (result, truncated) = truncate_tool_output(&long);
        assert!(truncated);
        assert!(result.len() < long.len());
        assert!(result.contains("truncated"));
    }

    #[test]
    fn snip_not_enough_messages() {
        let msgs = vec![Message::user("hi"), Message::assistant_text("hello")];
        assert!(snip_old_messages(&msgs, 5).is_none());
    }

    #[test]
    fn snip_keeps_recent() {
        let msgs: Vec<Message> = (0..20)
            .map(|i| Message::user(&format!("message {}", i)))
            .collect();

        let result = snip_old_messages(&msgs, 5).unwrap();
        // Should have: 1 snip marker + 5 recent
        assert_eq!(result.len(), 6);
        // Last message should be the original last
        if let MessageContent::Text(text) = &result.last().unwrap().content {
            assert_eq!(text, "message 19");
        }
    }

    #[test]
    fn should_compact_small_conversation() {
        let msgs = vec![Message::user("hi")];
        assert!(matches!(
            should_compact(&msgs, 200_000),
            CompactStrategy::None
        ));
    }

    #[test]
    fn context_window_known_models() {
        assert_eq!(context_window_for_model("claude-sonnet-4-20250514"), 200_000);
        assert_eq!(context_window_for_model("gpt-4o"), 128_000);
    }
}
