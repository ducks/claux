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

use std::collections::HashSet;
use std::sync::LazyLock;
use tiktoken_rs::{cl100k_base, CoreBPE};

/// Global tokenizer (initialized once, thread-safe).
static TOKENIZER: LazyLock<CoreBPE> =
    LazyLock::new(|| cl100k_base().expect("failed to initialize cl100k tokenizer"));

/// Empty set for encode's allowed_special parameter.
static NO_SPECIAL: LazyLock<HashSet<&'static str>> = LazyLock::new(HashSet::new);

/// Count tokens in a string using tiktoken.
fn count_tokens(text: &str) -> usize {
    TOKENIZER.encode(text, &NO_SPECIAL).0.len()
}

/// Estimate the token count of the conversation using tiktoken.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    let mut total = 0;

    for msg in messages {
        match &msg.content {
            MessageContent::Text(text) => {
                total += count_tokens(text);
            }
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            total += count_tokens(text);
                        }
                        ContentBlock::ToolUse { input, name, .. } => {
                            total += count_tokens(name);
                            total += count_tokens(&input.to_string());
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            total += count_tokens(content);
                        }
                    }
                }
            }
        }
    }

    total
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

    let start = crate::utils::truncate_str(output, keep_start);
    let end = crate::utils::tail_str(output, keep_end);
    let truncated_chars = output.len() - start.len() - end.len();

    let result = format!("{start}\n\n... ({truncated_chars} characters truncated) ...\n\n{end}");

    (result, true)
}

/// True if the message carries any tool_result blocks.
fn contains_tool_result(msg: &Message) -> bool {
    match &msg.content {
        MessageContent::Text(_) => false,
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. })),
    }
}

/// Snip compaction: collapse old messages into a brief marker,
/// keeping the most recent `keep_recent` messages intact.
/// Returns the new message list if snipping occurred, or None if
/// there weren't enough messages to snip.
pub fn snip_old_messages(messages: &[Message], keep_recent: usize) -> Option<Vec<Message>> {
    if messages.len() <= keep_recent + 2 {
        return None; // Not enough to snip
    }

    let mut snip_count = messages.len() - keep_recent;

    // The kept window must not open with tool_result blocks: their matching
    // tool_use would be on the snipped side of the cut, and the API rejects
    // a conversation containing tool_results with no preceding tool_use.
    // Walk the cut back (keeping more messages) until the boundary is safe.
    // tool_results immediately follow their tool_use message, so this only
    // ever backs up past complete tool rounds.
    while snip_count > 0 && contains_tool_result(&messages[snip_count]) {
        snip_count -= 1;
    }
    if snip_count == 0 {
        return None; // No safe cut point; nothing to snip
    }
    let snipped = &messages[..snip_count];
    let kept = &messages[snip_count..];

    // Count what we're removing
    let snip_tokens = estimate_tokens(snipped);

    let marker = Message::user(&format!(
        "[{snip_count} earlier messages snipped (~{snip_tokens} tokens). The conversation continues below.]"
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
    fn truncate_long_multibyte_output_no_panic() {
        // Regression: byte-indexed slicing panicked when a cut point landed
        // mid-codepoint. 4-byte chars guarantee both cut points do.
        let long = "🦀".repeat(15_000); // 60k bytes
        let (result, truncated) = truncate_tool_output(&long);
        assert!(truncated);
        assert!(result.contains("truncated"));
        // Both kept segments must still be valid crab-only text
        assert!(result.starts_with('🦀'));
        assert!(result.ends_with('🦀'));
    }

    #[test]
    fn snip_not_enough_messages() {
        let msgs = vec![Message::user("hi"), Message::assistant_text("hello")];
        assert!(snip_old_messages(&msgs, 5).is_none());
    }

    #[test]
    fn snip_keeps_recent() {
        let msgs: Vec<Message> = (0..20)
            .map(|i| Message::user(&format!("message {i}")))
            .collect();

        let result = snip_old_messages(&msgs, 5).unwrap();
        // Should have: 1 snip marker + 5 recent
        assert_eq!(result.len(), 6);
        // Last message should be the original last
        if let MessageContent::Text(text) = &result.last().unwrap().content {
            assert_eq!(text, "message 19");
        }
    }

    /// Every tool_result in the list must have a matching tool_use earlier.
    /// This is the invariant the Anthropic API enforces on requests.
    fn assert_no_orphaned_tool_results(messages: &[Message]) {
        let mut seen_tool_use_ids = std::collections::HashSet::new();
        for msg in messages {
            if let MessageContent::Blocks(blocks) = &msg.content {
                for block in blocks {
                    match block {
                        ContentBlock::ToolUse { id, .. } => {
                            seen_tool_use_ids.insert(id.clone());
                        }
                        ContentBlock::ToolResult { tool_use_id, .. } => {
                            assert!(
                                seen_tool_use_ids.contains(tool_use_id),
                                "orphaned tool_result: {tool_use_id}"
                            );
                        }
                        ContentBlock::Text { .. } => {}
                    }
                }
            }
        }
    }

    /// One user → assistant(tool_use) → user(tool_result) round.
    fn tool_round(n: usize) -> Vec<Message> {
        vec![
            Message::user(&format!("request {n}")),
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: format!("tu_{n}"),
                name: "Read".to_string(),
                input: serde_json::json!({"file_path": "/tmp/x"}),
            }]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: format!("tu_{n}"),
                content: "contents".to_string(),
                is_error: None,
            }]),
        ]
    }

    #[test]
    fn snip_never_orphans_tool_results() {
        // Regression: the cut point used to land on arbitrary messages. If
        // the kept window opened with a tool_result whose tool_use was
        // snipped, the next API request 400'd. Try every keep_recent value
        // against a conversation of tool rounds so cut points land on every
        // message kind.
        let mut msgs: Vec<Message> = Vec::new();
        for n in 0..6 {
            msgs.extend(tool_round(n));
        }

        for keep_recent in 1..msgs.len() {
            if let Some(snipped) = snip_old_messages(&msgs, keep_recent) {
                assert_no_orphaned_tool_results(&snipped);
            }
        }
    }

    #[test]
    fn snip_backs_up_to_include_tool_use() {
        // 18 messages of tool rounds; keep_recent=4 puts the naive cut at
        // index 14, which is a tool_result message (pattern repeats every 3:
        // user, assistant tool_use, user tool_result). The cut must back up
        // to keep the matching tool_use.
        let mut msgs: Vec<Message> = Vec::new();
        for n in 0..6 {
            msgs.extend(tool_round(n));
        }

        let snipped = snip_old_messages(&msgs, 4).unwrap();
        assert_no_orphaned_tool_results(&snipped);
        // Marker + at least keep_recent messages survive
        assert!(snipped.len() > 4);
        // First kept message after the marker is the assistant tool_use,
        // not its orphaned result
        if let MessageContent::Blocks(blocks) = &snipped[1].content {
            assert!(matches!(blocks[0], ContentBlock::ToolUse { .. }));
        } else {
            panic!("expected the kept window to open with the tool_use message");
        }
    }

    #[test]
    fn snip_returns_none_when_no_safe_cut() {
        // A conversation that is one giant unfinished tool cascade from
        // index 1 on: every candidate cut lands on a tool_result, walking
        // back to 0. Must return None rather than produce an invalid list.
        let mut msgs = vec![Message::user("start")];
        msgs.push(Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "tu_0".to_string(),
            name: "Read".to_string(),
            input: serde_json::json!({}),
        }]));
        for _ in 0..8 {
            msgs.push(Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "tu_0".to_string(),
                content: "x".to_string(),
                is_error: None,
            }]));
        }

        // keep_recent=2: naive cut at index 8, all tool_results back to
        // index 2, then index 1 is the tool_use... which is safe, actually.
        // Force the unsafe case by asking to cut inside the results run
        // starting at index 1.
        let all_results: Vec<Message> = msgs[2..].to_vec();
        assert!(snip_old_messages(&all_results, 2).is_none());
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
        assert_eq!(
            context_window_for_model("claude-sonnet-4-20250514"),
            200_000
        );
        assert_eq!(context_window_for_model("gpt-4o"), 128_000);
    }
}
