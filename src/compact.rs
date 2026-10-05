//! Context estimation and task-preserving compaction helpers.
//!
//! Tool outputs are capped before entering history. Compaction summarizes the
//! older conversation into a current task handoff and retains recent turns
//! verbatim. Older context is never discarded before summarization succeeds.

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

// BPE merging is quadratic for long unbroken runs (including encoded tool
// arguments). Bound each invocation, since estimates also run on the UI task.
const TOKEN_ESTIMATE_CHUNK_BYTES: usize = 1024;

/// Estimate tokens using bounded UTF-8 slices. Chunk boundaries can change the
/// estimate slightly; provider-reported usage remains authoritative.
pub fn count_tokens(text: &str) -> usize {
    let mut rest = text;
    let mut total = 0;
    while !rest.is_empty() {
        let chunk = crate::utils::truncate_str(rest, TOKEN_ESTIMATE_CHUNK_BYTES);
        total += TOKENIZER.encode(chunk, &NO_SPECIAL).0.len();
        rest = &rest[chunk.len()..];
    }
    total
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
                        // Image token accounting varies by provider and image
                        // dimensions. Reserve a conservative fixed amount
                        // without counting the much larger base64 transport.
                        ContentBlock::Image { .. } => total += 1_024,
                        ContentBlock::Reasoning { text, details } => {
                            if let Some(text) = text {
                                total += count_tokens(text);
                            }
                            if !details.is_empty() {
                                total += count_tokens(
                                    &serde_json::Value::Array(details.clone()).to_string(),
                                );
                            }
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

/// Keep the handoff focused on information the next model round needs to act.
/// Later user corrections take precedence over the retained original request.
pub const SUMMARY_PROMPT: &str = "Summarize the conversation into a compact task handoff.
Do not continue the task or call tools. Use these sections:
- Objective: the current user goal, including changes to the original request.
- Constraints: user requirements, prohibitions, preferences, and later corrections.
- Decisions: choices made and the reasons that still matter.
- Progress: completed work, changed file paths, test results, and failures.
- Outstanding work: remaining steps, blockers, and the immediate next action.
Preserve concrete paths, identifiers, and unresolved errors needed to resume.
Distinguish completed work from plans. Later user instructions supersede earlier
ones. Carry forward still-relevant details from any earlier handoff. Do not
invent missing facts; use 'None' for empty sections. Keep it concise.";

pub const HANDOFF_INTRO: &str = "The following handoff summarizes earlier conversation. Follow its current objective and later corrections, then incorporate the recent messages that follow.";

pub fn is_user_request(message: &Message) -> bool {
    message.role == "user"
        && match &message.content {
            MessageContent::Text(text) => !text.trim().is_empty(),
            MessageContent::Blocks(blocks) => {
                !blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
                    && blocks.iter().any(|block| {
                        matches!(
                            block,
                            ContentBlock::Text { .. } | ContentBlock::Image { .. }
                        )
                    })
            }
        }
}

/// Retain up to four recent user turns, or a suffix of a long tool loop.
/// Never split an outstanding tool batch across the summary and retained tail.
pub fn recent_tail_start(messages: &[Message], budget: usize) -> usize {
    let mut suffix_tokens = vec![0; messages.len() + 1];
    for index in (0..messages.len()).rev() {
        suffix_tokens[index] =
            suffix_tokens[index + 1] + estimate_tokens(&messages[index..index + 1]);
    }
    let mut pending = HashSet::new();
    let mut boundaries = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if index > 0 && pending.is_empty() {
            boundaries.push(index);
        }
        if let MessageContent::Blocks(blocks) = &message.content {
            for block in blocks {
                match block {
                    ContentBlock::ToolUse { id, .. } => {
                        pending.insert(id.clone());
                    }
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        pending.remove(tool_use_id);
                    }
                    _ => {}
                }
            }
        }
    }
    let user_boundaries: Vec<_> = boundaries
        .iter()
        .copied()
        .filter(|index| is_user_request(&messages[*index]))
        .collect();
    for index in user_boundaries.iter().rev().take(4).rev() {
        if suffix_tokens[*index] <= budget {
            return *index;
        }
    }
    boundaries
        .into_iter()
        .find(|index| suffix_tokens[*index] <= budget)
        .unwrap_or(messages.len())
}

/// Render tool activity as data rather than executable provider tool items.
/// Images remain intact in the retained tail; never summarize base64 transport.
pub fn summary_text(messages: &[Message]) -> String {
    let mut text = String::new();
    for message in messages {
        text.push_str(&format!("\n[{}]\n", message.role));
        match &message.content {
            MessageContent::Text(value) => text.push_str(value),
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text: value } => text.push_str(value),
                        ContentBlock::Image { .. } => text.push_str("[attached image]"),
                        _ => text.push_str(
                            &serde_json::to_string(block).expect("content block serializes"),
                        ),
                    }
                    text.push('\n');
                }
            }
        }
    }
    text
}

/// Split large excerpts at UTF-8 boundaries, with an explicit token bound.
pub fn text_chunks(text: &str, budget: usize, limit: usize) -> anyhow::Result<Vec<&str>> {
    anyhow::ensure!(budget > 0, "Summary token budget must be positive");
    let mut chunks = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        anyhow::ensure!(
            chunks.len() < limit,
            "Compact error: history exceeds {limit} summary chunks; history preserved"
        );
        let mut low = 0;
        let mut tokens = 0;
        while low < rest.len() {
            let piece = crate::utils::truncate_str(&rest[low..], TOKEN_ESTIMATE_CHUNK_BYTES);
            let piece_tokens = count_tokens(piece);
            if tokens + piece_tokens > budget {
                break;
            }
            tokens += piece_tokens;
            low += piece.len();
        }
        // A budget smaller than one estimator slice needs a partial slice.
        // Otherwise accept the slight underfill instead of repeatedly encoding
        // overlapping prefixes of the entire conversation.
        let mut high = if low == 0 {
            rest.len().min(TOKEN_ESTIMATE_CHUNK_BYTES)
        } else {
            low
        };
        while low < high {
            let mut end = low + (high - low).div_ceil(2);
            while !rest.is_char_boundary(end) {
                end -= 1;
            }
            if end <= low {
                break;
            }
            if count_tokens(&rest[..end]) <= budget {
                low = end;
            } else {
                high = end - 1;
            }
        }
        if low == 0 {
            low = rest.chars().next().unwrap().len_utf8();
        }
        if tokens == 0 {
            anyhow::ensure!(
                count_tokens(&rest[..low]) <= budget,
                "Summary token budget cannot fit a character"
            );
        }
        chunks.push(&rest[..low]);
        rest = &rest[low..];
    }
    Ok(chunks)
}

/// Context window sizes for known models.
#[cfg(test)]
pub fn context_window_for_model(model: &str) -> usize {
    crate::model::built_in_metadata(model).context_window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(&[]), 0);
    }

    #[test]
    fn long_unbroken_input_finishes_within_deadline() {
        const CHILD: &str = "CLAUX_TOKEN_ESTIMATE_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            assert_eq!(count_tokens(&"a".repeat(65_536)), 8192);
            assert!(count_tokens(&"界".repeat(21_846)) > 0);
            return;
        }
        // Isolate the regression so a quadratic tokenizer cannot hang the
        // test runner. The old implementation exceeds this bound by far.
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "compact::tests::long_unbroken_input_finishes_within_deadline",
            ])
            .env(CHILD, "1")
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("token estimation exceeded 15 seconds");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    fn short_token_estimates_match_tokenizer() {
        for text in ["hello world", "λ界 text and paths /tmp/source.rs\n", ""] {
            assert_eq!(
                count_tokens(text),
                TOKENIZER.encode(text, &NO_SPECIAL).0.len()
            );
        }
    }

    #[test]
    fn excerpt_limits_fail_instead_of_preparing_unbounded_chunks() {
        assert!(text_chunks("hello", 0, 32).is_err());
        assert!(text_chunks("hello", 10, 0).is_err());
        let text = "hello world ".repeat(1000);
        let error = text_chunks(&text, 64, 2).unwrap_err();
        assert!(error.to_string().contains("exceeds 2 summary chunks"));
        assert!(text_chunks("\u{10ffff}", 1, 32).is_err());
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
    fn recent_tail_preserves_images_and_excludes_tool_results_as_user_turns() {
        let request = Message::user_with_images(
            "fix this screenshot",
            vec![crate::api::types::ImageSource {
                source_type: "base64".into(),
                media_type: "image/png".into(),
                data: "image-data".into(),
            }],
        );
        let messages = vec![
            Message::user(&"old context ".repeat(100)),
            request.clone(),
            Message::assistant_text("working"),
        ];
        assert_eq!(
            serde_json::to_value(&messages[recent_tail_start(&messages, 2048)]).unwrap(),
            serde_json::to_value(request).unwrap()
        );
        assert_eq!(recent_tail_start(&[], 2048), 0);
    }

    #[test]
    fn recent_tail_does_not_orphan_tool_results() {
        let messages = vec![
            Message::user(&"old context ".repeat(100)),
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: "call".into(),
                name: "Read".into(),
                input: serde_json::json!({}),
            }]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "call".into(),
                content: "result".into(),
                is_error: None,
            }]),
            Message::assistant_text("done"),
        ];
        assert_eq!(recent_tail_start(&messages, 100), 1);
        assert_eq!(recent_tail_start(&messages, 1), 3);
    }

    #[test]
    fn oversized_excerpts_split_without_losing_unicode_or_content() {
        let text = "λ界 text and paths /tmp/source.rs\n".repeat(100);
        let chunks = text_chunks(&text, 64, 100).unwrap();
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| count_tokens(chunk) <= 64));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn context_window_known_models() {
        assert_eq!(
            context_window_for_model("claude-sonnet-4-20250514"),
            200_000
        );
        assert_eq!(context_window_for_model("gpt-4o"), 128_000);
        assert_eq!(context_window_for_model("gpt-5.6-sol"), 1_050_000);
        assert_eq!(context_window_for_model("gpt-5.3-codex"), 400_000);
    }
}
