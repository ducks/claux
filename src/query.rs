use anyhow::Result;
use tokio::sync::{mpsc, oneshot};

use crate::api::{ApiEvent, ContentBlock, Message, Provider};
use crate::compact::{self};
use crate::cost::CostTracker;
use crate::permissions::{PermissionChecker, PermissionResponse, PermissionResult};
use crate::tools::ToolRegistry;

/// The query engine: conversation loop that sends messages, streams responses,
/// dispatches tools, and continues until the assistant stops.
pub struct Engine {
    provider: Box<dyn Provider>,
    tools: ToolRegistry,
    permissions: PermissionChecker,
    messages: Vec<Message>,
    system_prompt: String,
    model: String,
    max_tokens: u32,
    auto_compact_threshold: f64,
    pub cost: CostTracker,
}

/// Events sent from the engine to the UI during streaming.
pub enum StreamEvent {
    Text(String),
    ToolStart {
        name: String,
        id: String,
        summary: String,
    },
    ToolResult {
        name: String,
        content: String,
        is_error: bool,
    },
    /// Permission prompt — UI must respond via the oneshot sender.
    PermissionRequest {
        tool_name: String,
        summary: String,
        respond: oneshot::Sender<PermissionResponse>,
    },
    /// Permission prompt with diff preview
    PermissionRequestWithDiff {
        tool_name: String,
        summary: String,
        diff: String,
        respond: oneshot::Sender<PermissionResponse>,
    },
    Error(String),
    Done,
}

impl Engine {
    pub fn new(
        provider: Box<dyn Provider>,
        tools: ToolRegistry,
        permissions: PermissionChecker,
        model: &str,
    ) -> Self {
        Self {
            provider,
            tools,
            permissions,
            messages: Vec::new(),
            system_prompt: String::new(),
            model: model.to_string(),
            max_tokens: 16384,
            auto_compact_threshold: 0.8,
            cost: CostTracker::new(model),
        }
    }

    /// Set the auto-compact threshold (0.0-1.0).
    pub fn set_auto_compact_threshold(&mut self, threshold: f64) {
        self.auto_compact_threshold = threshold.clamp(0.0, 1.0);
    }

    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
        self.provider.set_model(model);
        self.cost = CostTracker::new(model);
    }

    pub fn set_theme(&mut self, _theme: crate::theme::ThemeName) {
        // Theme is handled by the TUI layer, not the engine.
        // This method exists for command parsing consistency.
        // The actual theme switch happens in the TUI's execute_async handler.
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Get tool definitions for the API.
    pub fn tool_definitions(&self) -> Vec<crate::api::ToolDefinition> {
        self.tools.definitions()
    }

    /// Start a streaming API call. Returns the event receiver.
    pub async fn start_stream(
        &self,
        tool_defs: &[crate::api::ToolDefinition],
    ) -> Result<mpsc::Receiver<ApiEvent>> {
        self.provider
            .stream(
                &self.messages,
                &self.system_prompt,
                tool_defs,
                self.max_tokens,
            )
            .await
    }

    /// Check permission for a tool.
    pub fn check_permission(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        is_read_only: bool,
    ) -> PermissionResult {
        self.permissions.check(tool_name, input, is_read_only)
    }

    /// Get a human-readable summary of a tool invocation.
    pub fn summarize_tool(&self, name: &str, input: &serde_json::Value) -> String {
        self.tools.summarize(name, input)
    }

    /// Check if a tool is read-only.
    pub fn is_tool_read_only(&self, name: &str) -> bool {
        self.tools.is_read_only(name)
    }

    /// Execute a tool by name.
    pub async fn execute_tool(
        &self,
        name: &str,
        input: serde_json::Value,
    ) -> Result<crate::tools::ToolOutput> {
        self.tools.execute(name, input).await
    }

    /// Record a tool as always-allowed for the session.
    pub fn always_allow_tool(&mut self, name: &str) {
        self.permissions.always_allow(name);
    }

    /// Record a bash command as always-allowed for the session.
    pub fn always_allow_command(&mut self, cmd: &str) {
        self.permissions.always_allow_command(cmd);
    }

    /// Check if auto-compact is needed and perform it if so.
    /// Returns true if compaction was performed.
    pub async fn maybe_auto_compact(&mut self) -> Result<bool> {
        // Disabled if threshold is 0.0
        if self.auto_compact_threshold <= 0.0 {
            return Ok(false);
        }

        let ctx_window = compact::context_window_for_model(&self.model);
        let current_tokens = compact::estimate_tokens(&self.messages);
        let threshold_tokens = (ctx_window as f64 * self.auto_compact_threshold) as usize;

        if current_tokens > threshold_tokens {
            tracing::info!(
                "Auto-compact triggered: {} tokens > {} (threshold: {:.0}% of {})",
                current_tokens,
                threshold_tokens,
                self.auto_compact_threshold * 100.0,
                ctx_window
            );

            let result = self.compact().await?;
            tracing::info!("Auto-compact completed: {}", result);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Compact the conversation using the multi-strategy pipeline.
    /// Strategies (in order of aggressiveness):
    /// 1. Snip — collapse old messages, keep recent ones
    /// 2. Summarize — send conversation to API for full summary
    pub async fn compact(&mut self) -> Result<String> {
        if self.messages.is_empty() {
            return Ok("Nothing to compact.".to_string());
        }

        let old_count = self.messages.len();
        let old_tokens = compact::estimate_tokens(&self.messages);

        // Try snip first (cheaper, no API call)
        if let Some(snipped) = compact::snip_old_messages(&self.messages, 10) {
            let new_tokens = compact::estimate_tokens(&snipped);
            self.messages = snipped;
            tracing::info!(
                "Snip compaction: {} msgs → {}, ~{} → ~{} tokens",
                old_count,
                self.messages.len(),
                old_tokens,
                new_tokens
            );

            // If snip freed enough, we're done
            let ctx_window = compact::context_window_for_model(&self.model);
            if new_tokens < ctx_window * 70 / 100 {
                return Ok(format!(
                    "Snipped {} old messages (~{} tokens freed)",
                    old_count - self.messages.len() + 1, // +1 for snip marker
                    old_tokens - new_tokens
                ));
            }
        }

        // Full summarization
        self.summarize_conversation().await
    }

    /// Full API-based conversation summary.
    async fn summarize_conversation(&mut self) -> Result<String> {
        let summary_prompt = "Summarize the conversation so far in a concise paragraph. \
            Focus on what was discussed, what decisions were made, what files were modified, \
            and any outstanding tasks. Be specific about file paths and changes.";

        let mut summary_messages = self.messages.clone();
        summary_messages.push(Message::user(summary_prompt));

        let mut rx = self
            .provider
            .stream(&summary_messages, &self.system_prompt, &[], self.max_tokens)
            .await?;

        let mut summary = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                ApiEvent::Text(t) => summary.push_str(&t),
                ApiEvent::Usage(usage) => self.cost.add_usage(&usage),
                ApiEvent::Done => break,
                ApiEvent::Error(e) => return Err(anyhow::anyhow!("Compact error: {e}")),
                _ => {}
            }
        }

        let old_count = self.messages.len();

        self.messages = vec![
            Message::user("Here is a summary of our conversation so far:"),
            Message::assistant_text(&summary),
        ];

        Ok(format!(
            "Compacted {old_count} messages into summary.\n\n\x1b[2m{summary}\x1b[0m"
        ))
    }

    /// Check if an API error is a prompt-too-long error (413 or specific error message).
    fn is_prompt_too_long(err: &str) -> bool {
        err.contains("413")
            || err.contains("prompt is too long")
            || err.contains("maximum context length")
            || err.contains("max_tokens")
            || err.contains("context_length_exceeded")
    }

    /// Check if an API error is a max-output-tokens error.
    fn is_max_output_tokens(err: &str) -> bool {
        err.contains("max_output_tokens") || err.contains("max_tokens_exceeded")
    }

    /// Submit a user message and run the full turn loop (chat → tools → chat → ...).
    /// Includes error recovery for prompt-too-long and max-output-tokens.
    /// Returns the final assistant text response.
    pub async fn submit(&mut self, user_input: &str) -> Result<String> {
        let _ = self.maybe_auto_compact().await; // Ignore result for non-streaming
        self.messages.push(Message::user(user_input));

        let mut full_response = String::new();
        let mut recovery_attempts = 0;
        const MAX_RECOVERY: u32 = 3;

        // Turn loop: keep going as long as the assistant requests tool use
        loop {
            let tool_defs = self.tools.definitions();
            let stream_result = self
                .provider
                .stream(
                    &self.messages,
                    &self.system_prompt,
                    &tool_defs,
                    self.max_tokens,
                )
                .await;

            // Handle connection-level errors (413, etc.)
            let mut rx = match stream_result {
                Ok(rx) => rx,
                Err(e) => {
                    let err_str = e.to_string();
                    if Self::is_prompt_too_long(&err_str) && recovery_attempts < MAX_RECOVERY {
                        recovery_attempts += 1;
                        tracing::warn!(
                            "Prompt too long (attempt {}), compacting and retrying",
                            recovery_attempts
                        );
                        self.compact().await?;
                        continue;
                    }
                    return Err(e);
                }
            };

            let mut text_buf = String::new();
            let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut had_error = false;

            while let Some(event) = rx.recv().await {
                match event {
                    ApiEvent::Text(t) => {
                        text_buf.push_str(&t);
                    }
                    ApiEvent::ToolUse { id, name, input } => {
                        tool_uses.push((id, name, input));
                    }
                    ApiEvent::Usage(usage) => {
                        self.cost.add_usage(&usage);
                    }
                    ApiEvent::Done => break,
                    ApiEvent::Error(e) => {
                        // Check for recoverable stream-level errors
                        if Self::is_prompt_too_long(&e) && recovery_attempts < MAX_RECOVERY {
                            recovery_attempts += 1;
                            tracing::warn!(
                                "Prompt too long during stream (attempt {}), compacting",
                                recovery_attempts
                            );
                            had_error = true;
                            break;
                        }
                        if Self::is_max_output_tokens(&e) && self.max_tokens < 64_000 {
                            tracing::warn!(
                                "Max output tokens hit, escalating {} -> {}",
                                self.max_tokens,
                                self.max_tokens * 2
                            );
                            self.max_tokens = (self.max_tokens * 2).min(64_000);
                            had_error = true;
                            break;
                        }
                        return Err(anyhow::anyhow!("API error: {e}"));
                    }
                }
            }

            // If we hit a recoverable error, compact/adjust and retry
            if had_error {
                if recovery_attempts > 0 {
                    self.compact().await?;
                }
                continue;
            }

            // Build the assistant message
            let mut blocks = Vec::new();
            if !text_buf.is_empty() {
                blocks.push(ContentBlock::Text {
                    text: text_buf.clone(),
                });
                full_response.push_str(&text_buf);
            }
            for (id, name, input) in &tool_uses {
                blocks.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }

            if !blocks.is_empty() {
                self.messages.push(Message::assistant_blocks(blocks));
            }

            // If no tool use, we're done
            if tool_uses.is_empty() {
                break;
            }

            // Execute tools and collect results (with output truncation)
            // Partition tools into parallel (read-only) and sequential (write) groups
            let result_blocks = self.execute_tools_parallel(&tool_uses).await?;

            self.messages.push(Message::tool_results(result_blocks));
        }

        Ok(full_response)
    }

    /// Execute tools with parallel execution for read-only tools.
    /// Returns result blocks in the same order as tool_uses.
    async fn execute_tools_parallel(
        &mut self,
        tool_uses: &[(String, String, serde_json::Value)],
    ) -> Result<Vec<ContentBlock>> {
        // Partition tools into read-only (parallel) and write (sequential)
        let mut parallel_tools = Vec::new();
        let mut sequential_tools = Vec::new();

        for (idx, (id, name, input)) in tool_uses.iter().enumerate() {
            let is_read_only = self.tools.is_read_only(name);
            let perm = self.permissions.check(name, input, is_read_only);

            // Only read-only AND auto-allowed tools can run in parallel
            let can_parallelize = is_read_only && matches!(perm, PermissionResult::Allow);

            if can_parallelize {
                parallel_tools.push((idx, id.clone(), name.clone(), input.clone()));
            } else {
                sequential_tools.push((idx, id.clone(), name.clone(), input.clone(), perm));
            }
        }

        // Execute parallel tools concurrently
        // Since all tool implementations use &self (immutable), we can safely
        // execute multiple read-only tools in parallel
        let tools_ref = &self.tools;
        let parallel_futures: Vec<_> = parallel_tools
            .iter()
            .map(|(idx, id, name, input)| async move {
                let result = tools_ref.execute(name, input.clone()).await;
                (*idx, id.clone(), result)
            })
            .collect();

        let parallel_results = futures_util::future::join_all(parallel_futures).await;

        // Execute sequential tools one by one
        let mut sequential_results = Vec::new();
        for (idx, id, name, input, perm) in sequential_tools {
            let tool_output = match perm {
                PermissionResult::Allow => self.tools.execute(&name, input.clone()).await?,
                PermissionResult::Deny(reason) => crate::tools::ToolOutput {
                    content: format!("Permission denied: {reason}"),
                    is_error: true,
                },
                PermissionResult::Ask { message, diff: _ } => {
                    // For non-streaming mode, just auto-allow (this path shouldn't normally be reached)
                    eprintln!("  [tool] {message} — auto-allowing");
                    self.tools.execute(&name, input.clone()).await?
                }
            };
            sequential_results.push((idx, id, Ok(tool_output)));
        }

        // Combine and sort results back into original order
        let mut all_results: Vec<(usize, String, Result<crate::tools::ToolOutput>)> = Vec::new();
        all_results.extend(parallel_results);
        all_results.extend(sequential_results);
        all_results.sort_by_key(|(idx, _, _)| *idx);

        // Build result blocks
        let mut result_blocks = Vec::new();
        for (_, id, result) in all_results {
            let tool_output = result?;

            // Truncate large tool outputs to avoid context overflow
            let (content, was_truncated) = compact::truncate_tool_output(&tool_output.content);
            if was_truncated {
                tracing::debug!("Truncated tool output for {}", id);
            }

            result_blocks.push(ContentBlock::ToolResult {
                tool_use_id: id,
                content,
                is_error: if tool_output.is_error {
                    Some(true)
                } else {
                    None
                },
            });
        }

        Ok(result_blocks)
    }

    /// Submit with streaming callbacks (for the REPL).
    pub async fn submit_streaming(
        &mut self,
        user_input: &str,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        let compacted = self.maybe_auto_compact().await?;
        if compacted {
            let _ = tx
                .send(StreamEvent::Text(
                    "\n[conversation auto-compacted to free context]\n".to_string(),
                ))
                .await;
        }
        self.messages.push(Message::user(user_input));

        let mut recovery_attempts = 0;
        const MAX_RECOVERY: u32 = 3;

        loop {
            let tool_defs = self.tools.definitions();
            let stream_result = self
                .provider
                .stream(
                    &self.messages,
                    &self.system_prompt,
                    &tool_defs,
                    self.max_tokens,
                )
                .await;

            let mut rx = match stream_result {
                Ok(rx) => rx,
                Err(e) => {
                    let err_str = e.to_string();
                    if Self::is_prompt_too_long(&err_str) && recovery_attempts < MAX_RECOVERY {
                        recovery_attempts += 1;
                        let _ = tx
                            .send(StreamEvent::Text(
                                "\n[compacting conversation...]\n".to_string(),
                            ))
                            .await;
                        self.compact().await?;
                        continue;
                    }
                    let _ = tx.send(StreamEvent::Error(err_str.clone())).await;
                    return Err(e);
                }
            };

            let mut text_buf = String::new();
            let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut had_error = false;

            while let Some(event) = rx.recv().await {
                match event {
                    ApiEvent::Text(t) => {
                        let _ = tx.send(StreamEvent::Text(t.clone())).await;
                        text_buf.push_str(&t);
                    }
                    ApiEvent::ToolUse { id, name, input } => {
                        let summary = self.tools.summarize(&name, &input);
                        let _ = tx
                            .send(StreamEvent::ToolStart {
                                name: name.clone(),
                                id: id.clone(),
                                summary,
                            })
                            .await;
                        tool_uses.push((id, name, input));
                    }
                    ApiEvent::Usage(usage) => {
                        self.cost.add_usage(&usage);
                    }
                    ApiEvent::Done => break,
                    ApiEvent::Error(e) => {
                        if Self::is_prompt_too_long(&e) && recovery_attempts < MAX_RECOVERY {
                            recovery_attempts += 1;
                            let _ = tx
                                .send(StreamEvent::Text(
                                    "\n[compacting conversation...]\n".to_string(),
                                ))
                                .await;
                            self.compact().await?;
                            had_error = true;
                            break;
                        }
                        if Self::is_max_output_tokens(&e) && self.max_tokens < 64_000 {
                            self.max_tokens = (self.max_tokens * 2).min(64_000);
                            had_error = true;
                            break;
                        }
                        let _ = tx.send(StreamEvent::Error(e.clone())).await;
                        return Err(anyhow::anyhow!("API error: {e}"));
                    }
                }
            }

            if had_error {
                continue;
            }

            // Record assistant message
            let mut blocks = Vec::new();
            if !text_buf.is_empty() {
                blocks.push(ContentBlock::Text {
                    text: text_buf.clone(),
                });
            }
            for (id, name, input) in &tool_uses {
                blocks.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }
            if !blocks.is_empty() {
                self.messages.push(Message::assistant_blocks(blocks));
            }

            if tool_uses.is_empty() {
                let _ = tx.send(StreamEvent::Done).await;
                break;
            }

            // Execute tools
            let mut result_blocks = Vec::new();
            for (id, name, input) in &tool_uses {
                let is_read_only = self.tools.is_read_only(name);
                let perm = self.permissions.check(name, input, is_read_only);

                let tool_output = match perm {
                    PermissionResult::Allow => self.tools.execute(name, input.clone()).await?,
                    PermissionResult::Deny(reason) => crate::tools::ToolOutput {
                        content: format!("Permission denied: {reason}"),
                        is_error: true,
                    },
                    PermissionResult::Ask { message, diff } => {
                        // Send permission request to UI, wait for response
                        let (resp_tx, resp_rx) = oneshot::channel();

                        let event = if let Some(d) = diff {
                            StreamEvent::PermissionRequestWithDiff {
                                tool_name: name.clone(),
                                summary: message,
                                diff: d,
                                respond: resp_tx,
                            }
                        } else {
                            StreamEvent::PermissionRequest {
                                tool_name: name.clone(),
                                summary: message,
                                respond: resp_tx,
                            }
                        };

                        let _ = tx.send(event).await;

                        match resp_rx.await {
                            Ok(PermissionResponse::Allow) => {
                                self.tools.execute(name, input.clone()).await?
                            }
                            Ok(PermissionResponse::AlwaysAllow) => {
                                self.permissions.always_allow(name);
                                self.tools.execute(name, input.clone()).await?
                            }
                            Ok(PermissionResponse::AlwaysAllowCommand(ref cmd)) => {
                                self.permissions.always_allow_command(cmd);
                                self.tools.execute(name, input.clone()).await?
                            }
                            Ok(PermissionResponse::Deny) | Err(_) => crate::tools::ToolOutput {
                                content: "Permission denied by user.".to_string(),
                                is_error: true,
                            },
                        }
                    }
                };

                // Truncate large tool outputs
                let (content, was_truncated) = compact::truncate_tool_output(&tool_output.content);
                if was_truncated {
                    tracing::debug!("Truncated tool output for {}", name);
                }

                let _ = tx
                    .send(StreamEvent::ToolResult {
                        name: name.clone(),
                        content: content.clone(),
                        is_error: tool_output.is_error,
                    })
                    .await;

                result_blocks.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content,
                    is_error: if tool_output.is_error {
                        Some(true)
                    } else {
                        None
                    },
                });
            }

            self.messages.push(Message::tool_results(result_blocks));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolDefinition, Usage};
    use crate::permissions::PermissionMode;
    use std::time::Instant;

    // Mock provider for testing
    struct MockProvider;

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        fn model(&self) -> &str {
            "test-model"
        }

        fn set_model(&mut self, _model: &str) {
            // No-op for mock
        }

        async fn stream(
            &self,
            _messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _max_tokens: u32,
        ) -> Result<mpsc::Receiver<ApiEvent>> {
            let (tx, rx) = mpsc::channel(10);
            // Return empty stream for testing
            drop(tx);
            Ok(rx)
        }
    }

    #[tokio::test]
    async fn test_parallel_tool_execution() {
        // Create a mock engine with read-only tools
        let provider = Box::new(MockProvider);
        let tools = ToolRegistry::without_agent();
        let permissions = PermissionChecker::new(PermissionMode::Bypass);

        let mut engine = Engine {
            provider,
            tools,
            permissions,
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            max_tokens: 1000,
            auto_compact_threshold: 0.8,
            cost: CostTracker::new("test"),
        };

        // Create multiple read-only tool uses (Read and Glob)
        let tool_uses = vec![
            (
                "test1".to_string(),
                "Read".to_string(),
                serde_json::json!({"file_path": "/dev/null"}),
            ),
            (
                "test2".to_string(),
                "Glob".to_string(),
                serde_json::json!({"pattern": "*.rs"}),
            ),
            (
                "test3".to_string(),
                "Read".to_string(),
                serde_json::json!({"file_path": "/dev/null"}),
            ),
        ];

        let start = Instant::now();
        let result = engine.execute_tools_parallel(&tool_uses).await;
        let duration = start.elapsed();

        assert!(result.is_ok(), "Parallel execution should succeed");
        let blocks = result.unwrap();
        assert_eq!(blocks.len(), 3, "Should have 3 result blocks");

        // Verify results are in correct order
        for (i, block) in blocks.iter().enumerate() {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                let expected_id = format!("test{}", i + 1);
                assert_eq!(
                    tool_use_id, &expected_id,
                    "Results should be in original order"
                );
            } else {
                panic!("Expected ToolResult block");
            }
        }

        println!("Parallel execution took: {:?}", duration);
    }

    #[tokio::test]
    async fn test_mixed_readonly_and_write_tools() {
        let provider = Box::new(MockProvider);
        let tools = ToolRegistry::without_agent();
        let permissions = PermissionChecker::new(PermissionMode::Bypass);

        let mut engine = Engine {
            provider,
            tools,
            permissions,
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            max_tokens: 1000,
            auto_compact_threshold: 0.8,
            cost: CostTracker::new("test"),
        };

        // Mix read-only and write tools
        let tool_uses = vec![
            (
                "test1".to_string(),
                "Read".to_string(), // read-only
                serde_json::json!({"file_path": "/dev/null"}),
            ),
            (
                "test2".to_string(),
                "Bash".to_string(), // write (not read-only)
                serde_json::json!({"command": "echo test"}),
            ),
            (
                "test3".to_string(),
                "Glob".to_string(), // read-only
                serde_json::json!({"pattern": "*.rs"}),
            ),
        ];

        let result = engine.execute_tools_parallel(&tool_uses).await;

        assert!(result.is_ok(), "Mixed execution should succeed");
        let blocks = result.unwrap();
        assert_eq!(blocks.len(), 3, "Should have 3 result blocks");

        // Verify order is maintained
        for (i, block) in blocks.iter().enumerate() {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                let expected_id = format!("test{}", i + 1);
                assert_eq!(tool_use_id, &expected_id, "Results should maintain order");
            }
        }
    }
}
