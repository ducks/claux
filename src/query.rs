use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

use crate::api::{ApiEvent, ContentBlock, Message, Provider};
use crate::compact::{self};
use crate::cost::CostTracker;
use crate::permissions::{PermissionChecker, PermissionResponse, PermissionResult};
use crate::tools::ToolRegistry;

/// Queue of user messages typed while a turn is running ("steering").
/// UIs push into it from input handlers; the turn loop drains it before
/// each API call and injects the entries as user messages, so the model
/// hears the user without the tool sequence being aborted.
pub type SteeringQueue = Arc<Mutex<VecDeque<String>>>;

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
    steering: SteeringQueue,
    pub cost: CostTracker,
}

/// Events sent from the engine to the UI during streaming.
pub enum StreamEvent {
    Text(String),
    /// Engine status line (compaction). Display-only: never part of the
    /// assistant's response text.
    Notice(String),
    /// A steering message was delivered into the conversation. UIs render
    /// it as the user message it now is.
    SteeringSent(String),
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
    /// `input` is the raw tool input so UIs can render rich details.
    PermissionRequest {
        tool_name: String,
        summary: String,
        input: serde_json::Value,
        respond: oneshot::Sender<PermissionResponse>,
    },
    /// Permission prompt with diff preview
    PermissionRequestWithDiff {
        tool_name: String,
        summary: String,
        diff: String,
        input: serde_json::Value,
        respond: oneshot::Sender<PermissionResponse>,
    },
    /// The turn was cancelled; dangling tool_uses were paired with
    /// synthetic interrupted results and the turn ended cleanly.
    Interrupted,
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
            steering: SteeringQueue::default(),
            cost: CostTracker::new(model),
        }
    }

    /// Test constructor: a bare engine over any provider, with the standard
    /// tool registry (minus Agent) and the given permission mode.
    #[cfg(test)]
    pub(crate) fn for_tests(
        provider: Box<dyn Provider>,
        steering: SteeringQueue,
        mode: crate::permissions::PermissionMode,
    ) -> Self {
        Self {
            provider,
            tools: ToolRegistry::without_agent(),
            permissions: PermissionChecker::new(mode),
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            max_tokens: 1000,
            auto_compact_threshold: 0.8,
            steering,
            cost: CostTracker::new("test"),
        }
    }

    /// Clone a handle to the steering queue. UIs (or their input threads)
    /// push typed-mid-turn messages through this handle.
    pub fn steering_queue(&self) -> SteeringQueue {
        self.steering.clone()
    }

    /// Drain queued steering messages into the conversation as user
    /// messages. Returns the drained texts so the caller can display them.
    /// Call between turn-loop iterations, after tool results are pushed.
    pub fn inject_steering(&mut self) -> Vec<String> {
        let drained: Vec<String> = {
            let mut q = self.steering.lock().expect("steering queue poisoned");
            q.drain(..).collect()
        };
        for text in &drained {
            self.messages.push(Message::user(text));
        }
        drained
    }

    /// True if a steering message is waiting. Tool batches check this
    /// between tools to decide whether to skip the rest of the batch.
    pub fn steering_pending(&self) -> bool {
        !self
            .steering
            .lock()
            .expect("steering queue poisoned")
            .is_empty()
    }

    /// Synthetic tool_result content for tools skipped because the user
    /// sent a steering message before they ran.
    pub const SKIPPED_FOR_STEERING: &'static str =
        "Skipped: superseded by a new user message before this tool ran.";

    /// Execute a tool, cancelling it if a steering message arrives while it
    /// runs or the turn itself is cancelled. Mirrors Claude Code's
    /// submit-interrupt: a mid-batch user message shouldn't wait out a
    /// doomed cargo test. The watcher polls the queue at 50ms, the same
    /// cadence the TUI polls the keyboard; turn cancellation propagates
    /// through the child token immediately.
    async fn execute_tool_steerable(
        &self,
        name: &str,
        input: serde_json::Value,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> crate::tools::ToolOutput {
        let token = cancel.child_token();
        let steering = self.steering.clone();
        let watch_token = token.clone();
        let watcher = tokio::spawn(async move {
            loop {
                if !steering.lock().expect("steering queue poisoned").is_empty() {
                    watch_token.cancel();
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });

        let output = self.tools.execute(name, input, token).await;
        watcher.abort();
        output
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

    /// Execute a tool by name. Pass `CancellationToken::new()` (non-cancellable)
    /// if the caller doesn't need to interrupt; pass a real token to support
    /// mid-execution cancellation. Failures (unknown tool, bad params, tool
    /// errors) come back as error ToolOutputs, never as Err — see
    /// ToolRegistry::execute.
    pub async fn execute_tool(
        &self,
        name: &str,
        input: serde_json::Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> crate::tools::ToolOutput {
        self.tools.execute(name, input, cancel).await
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

    /// Content used when pairing a tool_use whose execution was cut off by
    /// turn cancellation.
    pub const INTERRUPTED_BY_USER: &'static str = "Interrupted by user.";

    /// Submit a user message and run the full turn loop, returning the
    /// final assistant text. Non-interactive: tools that would ask for
    /// confirmation are denied. This is a thin collector over the same
    /// run_turn that powers submit_streaming, so the two can't drift.
    /// Cancelling `cancel` ends the turn cleanly (tool_uses paired with
    /// interrupted results).
    pub async fn submit(
        &mut self,
        user_input: &str,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<String> {
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(256);

        let collector = tokio::spawn(async move {
            let mut text = String::new();
            while let Some(event) = rx.recv().await {
                if let StreamEvent::Text(t) = event {
                    text.push_str(&t);
                }
            }
            text
        });

        let result = self.run_turn(user_input, tx, false, cancel).await;
        let text = collector.await.unwrap_or_default();
        result?;
        Ok(text)
    }

    /// Submit with streaming callbacks (for the REPL and TUI). Interactive:
    /// tools that need confirmation emit PermissionRequest events and wait.
    pub async fn submit_streaming(
        &mut self,
        user_input: &str,
        tx: mpsc::Sender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        self.run_turn(user_input, tx, true, cancel).await
    }

    /// The turn loop: chat -> tools -> chat -> ... until the assistant
    /// stops requesting tools. Handles steering injection, recoverable API
    /// errors (prompt-too-long -> compact, max-output-tokens -> escalate),
    /// tool execution, and cancellation. `interactive` decides what happens
    /// when a tool needs user confirmation: emit a PermissionRequest event
    /// and wait, or deny with a pointer at permission_mode config.
    async fn run_turn(
        &mut self,
        user_input: &str,
        tx: mpsc::Sender<StreamEvent>,
        interactive: bool,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        let compacted = self.maybe_auto_compact().await?;
        if compacted {
            let _ = tx
                .send(StreamEvent::Notice(
                    "conversation auto-compacted to free context".to_string(),
                ))
                .await;
        }
        self.messages.push(Message::user(user_input));

        let mut recovery_attempts = 0;
        const MAX_RECOVERY: u32 = 3;

        loop {
            // Deliver any steering messages queued since the last API call,
            // and tell the UI they're now in the conversation.
            for text in self.inject_steering() {
                let _ = tx.send(StreamEvent::SteeringSent(text)).await;
            }

            if cancel.is_cancelled() {
                let _ = tx.send(StreamEvent::Interrupted).await;
                return Ok(());
            }

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
                            .send(StreamEvent::Notice(
                                "compacting conversation...".to_string(),
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
            let mut stream_interrupted = false;

            loop {
                let event = tokio::select! {
                    event = rx.recv() => match event {
                        Some(event) => event,
                        None => break,
                    },
                    _ = cancel.cancelled() => {
                        stream_interrupted = true;
                        break;
                    }
                };
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
                                .send(StreamEvent::Notice(
                                    "compacting conversation...".to_string(),
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

            // Cancelled mid-stream: pair every received tool_use with a
            // synthetic interrupted result so the conversation stays
            // API-valid, then end the turn.
            if stream_interrupted {
                if !tool_uses.is_empty() {
                    let mut result_blocks = Vec::with_capacity(tool_uses.len());
                    for (id, name, _) in &tool_uses {
                        let _ = tx
                            .send(StreamEvent::ToolResult {
                                name: name.clone(),
                                content: Self::INTERRUPTED_BY_USER.to_string(),
                                is_error: true,
                            })
                            .await;
                        result_blocks.push(ContentBlock::ToolResult {
                            tool_use_id: id.clone(),
                            content: Self::INTERRUPTED_BY_USER.to_string(),
                            is_error: Some(true),
                        });
                    }
                    self.messages.push(Message::tool_results(result_blocks));
                }
                let _ = tx.send(StreamEvent::Interrupted).await;
                return Ok(());
            }

            if tool_uses.is_empty() {
                let _ = tx.send(StreamEvent::Done).await;
                break;
            }

            let (result_blocks, interrupted) = self
                .execute_tool_batch(&tool_uses, &tx, interactive, &cancel)
                .await;
            self.messages.push(Message::tool_results(result_blocks));

            if interrupted {
                let _ = tx.send(StreamEvent::Interrupted).await;
                return Ok(());
            }
        }

        Ok(())
    }

    /// Execute one batch of tool calls.
    ///
    /// Read-only tools that are auto-allowed run concurrently; everything
    /// else runs sequentially in order (permission prompts are inherently
    /// serial). A pending steering message supersedes the batch: tools not
    /// yet started get synthetic skipped results, and running tools are
    /// cancelled by their steering watchers. Result blocks come back in
    /// the original tool_use order.
    async fn execute_tool_batch(
        &mut self,
        tool_uses: &[(String, String, serde_json::Value)],
        tx: &mpsc::Sender<StreamEvent>,
        interactive: bool,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> (Vec<ContentBlock>, bool) {
        let mut outputs: Vec<Option<crate::tools::ToolOutput>> =
            (0..tool_uses.len()).map(|_| None).collect();

        // Classify up front. Only read-only AND auto-allowed tools run in
        // parallel; the permission check is repeated for sequential tools
        // below because an AlwaysAllow answer during the batch can change
        // later results.
        let parallel: Vec<usize> = tool_uses
            .iter()
            .enumerate()
            .filter(|(_, (_, name, input))| {
                let ro = self.tools.is_read_only(name);
                ro && matches!(
                    self.permissions.check(name, input, ro),
                    PermissionResult::Allow
                )
            })
            .map(|(idx, _)| idx)
            .collect();

        let mut interrupted = false;

        // Phase 1: run the parallel group concurrently. Tool impls take
        // &self, so concurrent immutable borrows are safe.
        if !self.steering_pending() && !cancel.is_cancelled() && !parallel.is_empty() {
            let this: &Self = &*self;
            let futures: Vec<_> = parallel
                .iter()
                .map(|&idx| {
                    let (_, name, input) = &tool_uses[idx];
                    async move {
                        (
                            idx,
                            this.execute_tool_steerable(name, input.clone(), cancel)
                                .await,
                        )
                    }
                })
                .collect();
            for (idx, output) in futures_util::future::join_all(futures).await {
                outputs[idx] = Some(output);
            }
        }

        // Phase 2: everything not yet run, in order.
        for (idx, (_, name, input)) in tool_uses.iter().enumerate() {
            if outputs[idx].is_some() {
                continue;
            }

            // Turn cancelled: pair the remaining tools with interrupted
            // results and end the turn after this batch.
            if cancel.is_cancelled() {
                interrupted = true;
                outputs[idx] = Some(crate::tools::ToolOutput {
                    content: Self::INTERRUPTED_BY_USER.to_string(),
                    is_error: true,
                });
                continue;
            }

            // A steering message supersedes the rest of the batch: give
            // the remaining tools synthetic results so the model reads the
            // user's correction instead of finishing an abandoned plan.
            if self.steering_pending() {
                outputs[idx] = Some(crate::tools::ToolOutput {
                    content: Self::SKIPPED_FOR_STEERING.to_string(),
                    is_error: true,
                });
                continue;
            }

            let is_read_only = self.tools.is_read_only(name);
            let perm = self.permissions.check(name, input, is_read_only);

            let output = match perm {
                PermissionResult::Allow => {
                    self.execute_tool_steerable(name, input.clone(), cancel)
                        .await
                }
                PermissionResult::Deny(reason) => crate::tools::ToolOutput {
                    content: format!("Permission denied: {reason}"),
                    is_error: true,
                },
                PermissionResult::Ask { message, diff } => {
                    if !interactive {
                        // One-shot mode has no prompt to ask the user, so a
                        // tool requiring confirmation must be denied rather
                        // than silently auto-allowed.
                        crate::tools::ToolOutput {
                            content: format!(
                                "Permission denied: {message} (one-shot mode has no prompt; set permission_mode in config.toml to allow)"
                            ),
                            is_error: true,
                        }
                    } else {
                        self.ask_permission(name, input, message, diff, tx, cancel)
                            .await
                    }
                }
            };
            outputs[idx] = Some(output);
        }

        if cancel.is_cancelled() {
            interrupted = true;
        }

        // Phase 3: truncate, emit events, and build blocks in order.
        let mut result_blocks = Vec::with_capacity(tool_uses.len());
        for (idx, (id, name, _)) in tool_uses.iter().enumerate() {
            let output = outputs[idx].take().expect("every tool got an output");
            let (content, was_truncated) = compact::truncate_tool_output(&output.content);
            if was_truncated {
                tracing::debug!("Truncated tool output for {}", name);
            }

            let _ = tx
                .send(StreamEvent::ToolResult {
                    name: name.clone(),
                    content: content.clone(),
                    is_error: output.is_error,
                })
                .await;

            result_blocks.push(ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content,
                is_error: if output.is_error { Some(true) } else { None },
            });
        }

        (result_blocks, interrupted)
    }

    /// Ask the UI for permission and run (or deny) the tool accordingly.
    async fn ask_permission(
        &mut self,
        name: &str,
        input: &serde_json::Value,
        message: String,
        diff: Option<String>,
        tx: &mpsc::Sender<StreamEvent>,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> crate::tools::ToolOutput {
        let (resp_tx, resp_rx) = oneshot::channel();

        let event = if let Some(d) = diff {
            StreamEvent::PermissionRequestWithDiff {
                tool_name: name.to_string(),
                summary: message,
                diff: d,
                input: input.clone(),
                respond: resp_tx,
            }
        } else {
            StreamEvent::PermissionRequest {
                tool_name: name.to_string(),
                summary: message,
                input: input.clone(),
                respond: resp_tx,
            }
        };

        let _ = tx.send(event).await;

        match resp_rx.await {
            Ok(PermissionResponse::Allow) => {
                self.execute_tool_steerable(name, input.clone(), cancel)
                    .await
            }
            Ok(PermissionResponse::AlwaysAllow) => {
                self.permissions.always_allow(name);
                self.execute_tool_steerable(name, input.clone(), cancel)
                    .await
            }
            Ok(PermissionResponse::AlwaysAllowCommand(ref cmd)) => {
                self.permissions.always_allow_command(cmd);
                self.execute_tool_steerable(name, input.clone(), cancel)
                    .await
            }
            // DenyAndCancel queues the typed message as steering; the
            // steering_pending check skips the rest of the batch.
            Ok(PermissionResponse::Deny) | Ok(PermissionResponse::DenyAndCancel) | Err(_) => {
                crate::tools::ToolOutput {
                    content: "Permission denied by user.".to_string(),
                    is_error: true,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolDefinition;
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
            steering: SteeringQueue::default(),
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
        let (batch_tx, mut batch_rx) = mpsc::channel(64);
        let drain = tokio::spawn(async move { while batch_rx.recv().await.is_some() {} });
        let (blocks, _interrupted) = engine
            .execute_tool_batch(
                &tool_uses,
                &batch_tx,
                false,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await;
        drop(batch_tx);
        drain.await.unwrap();
        let duration = start.elapsed();

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

        println!("Parallel execution took: {duration:?}");
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
            steering: SteeringQueue::default(),
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

        let (batch_tx, mut batch_rx) = mpsc::channel(64);
        let drain = tokio::spawn(async move { while batch_rx.recv().await.is_some() {} });
        let (blocks, _interrupted) = engine
            .execute_tool_batch(
                &tool_uses,
                &batch_tx,
                false,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await;
        drop(batch_tx);
        drain.await.unwrap();

        assert_eq!(blocks.len(), 3, "Should have 3 result blocks");

        // Verify order is maintained
        for (i, block) in blocks.iter().enumerate() {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                let expected_id = format!("test{}", i + 1);
                assert_eq!(tool_use_id, &expected_id, "Results should maintain order");
            }
        }
    }

    /// Bypass-mode scripted engine; see crate::test_support.
    fn steering_engine(
        first_round: Vec<(String, String, serde_json::Value)>,
        push_on_first_call: Option<String>,
    ) -> Engine {
        crate::test_support::scripted_engine(
            first_round,
            push_on_first_call,
            PermissionMode::Bypass,
        )
    }

    async fn run_streaming(engine: &mut Engine, prompt: &str) {
        let (tx, mut rx) = mpsc::channel(64);
        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
        engine
            .submit_streaming(prompt, tx, tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();
        drain.await.unwrap();
    }

    #[tokio::test]
    async fn test_steering_message_injected_after_tool_results() {
        let mut engine = steering_engine(
            vec![(
                "tu_1".to_string(),
                "Glob".to_string(),
                serde_json::json!({"pattern": "*.does-not-exist"}),
            )],
            Some("also check the auth module".to_string()),
        );

        run_streaming(&mut engine, "do a deep review").await;

        // Expected: user prompt, assistant(tool_use), user(tool_results),
        // then the steering text as its own user message before round two.
        let msgs = engine.messages();
        assert_eq!(msgs.len(), 4, "got: {msgs:?}");
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[2].role, "user"); // tool results
        assert_eq!(msgs[3].role, "user");
        match &msgs[3].content {
            crate::api::MessageContent::Text(t) => {
                assert_eq!(t, "also check the auth module")
            }
            other => panic!("expected steering text message, got {other:?}"),
        }
        // Queue fully drained
        assert!(engine.steering_queue().lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_pending_steering_skips_whole_batch() {
        let mut engine = steering_engine(
            vec![
                (
                    "tu_1".to_string(),
                    "Glob".to_string(),
                    serde_json::json!({"pattern": "*.a"}),
                ),
                (
                    "tu_2".to_string(),
                    "Glob".to_string(),
                    serde_json::json!({"pattern": "*.b"}),
                ),
            ],
            Some("wrong direction, stop".to_string()),
        );

        run_streaming(&mut engine, "explore").await;

        // Both tools were superseded by the steering message: their
        // tool_results are synthetic skips, not Glob output.
        let msgs = engine.messages();
        let crate::api::MessageContent::Blocks(blocks) = &msgs[2].content else {
            panic!("expected tool results, got {msgs:?}");
        };
        assert_eq!(blocks.len(), 2);
        for block in blocks {
            match block {
                ContentBlock::ToolResult {
                    content, is_error, ..
                } => {
                    assert_eq!(content, Engine::SKIPPED_FOR_STEERING);
                    assert_eq!(*is_error, Some(true));
                }
                other => panic!("expected ToolResult, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn test_steering_cancels_running_tool() {
        // A slow tool (sleep 5) must be cancelled when steering arrives
        // ~200ms in, not waited out.
        let mut engine = steering_engine(
            vec![(
                "tu_1".to_string(),
                "Bash".to_string(),
                serde_json::json!({"command": "sleep 5"}),
            )],
            None,
        );

        let steering = engine.steering_queue();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            steering
                .lock()
                .unwrap()
                .push_back("no, run it in nix-shell instead".to_string());
        });

        let start = std::time::Instant::now();
        run_streaming(&mut engine, "run the tests").await;
        assert!(
            start.elapsed() < std::time::Duration::from_secs(3),
            "steering should cancel the running tool, not wait it out (took {:?})",
            start.elapsed()
        );

        // The steering message made it into the conversation.
        let last = engine.messages().last().unwrap();
        match &last.content {
            crate::api::MessageContent::Text(t) => {
                assert_eq!(t, "no, run it in nix-shell instead")
            }
            other => panic!("expected steering message last, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_cancellation_ends_turn_with_paired_results() {
        // Cancelling mid-tool must cut the running tool short, pair every
        // tool_use with a result, emit Interrupted, and return Ok.
        let mut engine = steering_engine(
            vec![(
                "tu_1".to_string(),
                "Bash".to_string(),
                serde_json::json!({"command": "sleep 5"}),
            )],
            None,
        );

        let cancel = tokio_util::sync::CancellationToken::new();
        let canceller = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            canceller.cancel();
        });

        let (tx, mut rx) = mpsc::channel(64);
        let events = tokio::spawn(async move {
            let mut interrupted = false;
            while let Some(ev) = rx.recv().await {
                if matches!(ev, StreamEvent::Interrupted) {
                    interrupted = true;
                }
            }
            interrupted
        });

        let start = std::time::Instant::now();
        engine.submit_streaming("run it", tx, cancel).await.unwrap();
        assert!(
            start.elapsed() < std::time::Duration::from_secs(3),
            "cancellation should not wait out the tool (took {:?})",
            start.elapsed()
        );
        assert!(events.await.unwrap(), "Interrupted event must be emitted");

        // Every tool_use is paired: the last message holds the results
        let msgs = engine.messages();
        let crate::api::MessageContent::Blocks(blocks) = &msgs.last().unwrap().content else {
            panic!("expected tool results last, got {msgs:?}");
        };
        assert!(matches!(
            &blocks[0],
            ContentBlock::ToolResult {
                is_error: Some(true),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_submit_returns_text_without_notices() {
        // submit() is a collector over the unified turn loop. Steering
        // delivery generates a Notice event; the returned text must be the
        // assistant's words only.
        let mut engine = steering_engine(
            vec![(
                "tu_1".to_string(),
                "Glob".to_string(),
                serde_json::json!({"pattern": "*.x"}),
            )],
            Some("check auth too".to_string()),
        );

        let text = engine
            .submit("go", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(text, "working on it", "notices must not leak into text");
        // The steering message still made it into the conversation
        let last = engine.messages().last().unwrap();
        match &last.content {
            crate::api::MessageContent::Text(t) => assert_eq!(t, "check auth too"),
            other => panic!("expected steering message last, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_steering_preempts_in_non_streaming_submit() {
        // Before unification, steering preemption only existed in the
        // streaming path; submit() (one-shot, sub-agents) waited out the
        // whole batch. Both entry points now share run_turn.
        let mut engine = steering_engine(
            vec![(
                "tu_1".to_string(),
                "Bash".to_string(),
                serde_json::json!({"command": "sleep 5"}),
            )],
            None,
        );

        let steering = engine.steering_queue();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            steering
                .lock()
                .unwrap()
                .push_back("stop, wrong command".to_string());
        });

        let start = std::time::Instant::now();
        engine
            .submit("run it", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();
        assert!(
            start.elapsed() < std::time::Duration::from_secs(3),
            "steering should cancel the running tool via submit() too (took {:?})",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn test_unknown_tool_yields_error_block_not_abort() {
        // A hallucinated tool name must produce an error tool_result the
        // model can recover from. Aborting the turn here left a dangling
        // tool_use in history, which the API rejects on the next request.
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
            steering: SteeringQueue::default(),
            cost: CostTracker::new("test"),
        };

        let tool_uses = vec![(
            "test1".to_string(),
            "TaskCreate".to_string(), // not in the registry
            serde_json::json!({"subject": "x"}),
        )];

        let (batch_tx, mut batch_rx) = mpsc::channel(64);
        let drain = tokio::spawn(async move { while batch_rx.recv().await.is_some() {} });
        let (blocks, _interrupted) = engine
            .execute_tool_batch(
                &tool_uses,
                &batch_tx,
                false,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await;
        drop(batch_tx);
        drain.await.unwrap();
        assert_eq!(blocks.len(), 1, "every tool_use must get a tool_result");

        match &blocks[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                is_error,
                content,
            } => {
                assert_eq!(tool_use_id, "test1");
                assert_eq!(*is_error, Some(true));
                assert!(content.contains("Unknown tool"));
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[tokio::test]
    async fn test_ask_permission_denies_in_non_streaming_mode() {
        // Non-interactive batches have no prompt to fall back
        // on, so a tool that would normally ask for confirmation must be denied,
        // not silently auto-allowed.
        let provider = Box::new(MockProvider);
        let tools = ToolRegistry::without_agent();
        let permissions = PermissionChecker::new(PermissionMode::Default);

        let mut engine = Engine {
            provider,
            tools,
            permissions,
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            max_tokens: 1000,
            auto_compact_threshold: 0.8,
            steering: SteeringQueue::default(),
            cost: CostTracker::new("test"),
        };

        // Under PermissionMode::Default, Read asks for confirmation.
        let tool_uses = vec![(
            "test1".to_string(),
            "Read".to_string(),
            serde_json::json!({"file_path": "/etc/hosts"}),
        )];

        let (batch_tx, mut batch_rx) = mpsc::channel(64);
        let drain = tokio::spawn(async move { while batch_rx.recv().await.is_some() {} });
        let (blocks, _interrupted) = engine
            .execute_tool_batch(
                &tool_uses,
                &batch_tx,
                false,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await;
        drop(batch_tx);
        drain.await.unwrap();
        assert_eq!(blocks.len(), 1);

        match &blocks[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert_eq!(
                    *is_error,
                    Some(true),
                    "Ask-permission tool must be denied, not executed, in non-streaming mode"
                );
                assert!(
                    content.contains("Permission denied"),
                    "expected a permission-denied message, got: {content}"
                );
            }
            _ => panic!("Expected ToolResult block"),
        }
    }
}
