use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

#[cfg(test)]
use crate::api::ProviderStream;
use crate::api::{ApiEvent, ApiFailure, ApiFailureKind, ContentBlock, Message, Provider};
use crate::checkpoint::{PendingCheckpoint, TurnCheckpoint};
use crate::compact::{self};
use crate::config::{HookTrigger, ModelBinding};
use crate::cost::CostTracker;
use crate::permissions::{PermissionChecker, PermissionResponse, PermissionResult};
use crate::plugin::PluginRegistry;
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
    model_binding: Option<ModelBinding>,
    max_tokens: u32,
    context_window: usize,
    auto_compact_threshold: f64,
    steering: SteeringQueue,
    plugins: Option<Arc<PluginRegistry>>,
    checkpoint_enabled: bool,
    pending_checkpoint: Option<PendingCheckpoint>,
    last_checkpoint: Option<TurnCheckpoint>,
    pub cost: CostTracker,
}

/// Events sent from the engine to the UI during streaming.
pub enum StreamEvent {
    Text(String),
    /// The current provider attempt was rejected before any tools ran.
    /// UIs must discard uncommitted text from that attempt before showing
    /// the retry notice.
    Retry(String),
    /// Engine status line (compaction). Display-only: never part of the
    /// assistant's response text.
    Notice(String),
    /// A steering message was delivered into the conversation. UIs render
    /// it as the user message it now is.
    SteeringSent(String),
    ToolStart {
        name: String,
        summary: String,
    },
    ToolResult {
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
            model_binding: None,
            max_tokens: 16384,
            context_window: crate::model::built_in_metadata(model).context_window,
            auto_compact_threshold: 0.8,
            steering: SteeringQueue::default(),
            plugins: None,
            checkpoint_enabled: true,
            pending_checkpoint: None,
            last_checkpoint: None,
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
            tools: ToolRegistry::without_agent_for_tests(),
            permissions: PermissionChecker::new(mode),
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            model_binding: None,
            max_tokens: 1000,
            context_window: crate::model::built_in_metadata("test").context_window,
            auto_compact_threshold: 0.8,
            steering,
            plugins: None,
            checkpoint_enabled: false,
            pending_checkpoint: None,
            last_checkpoint: None,
            cost: CostTracker::new("test"),
        }
    }

    /// Attach lifecycle hooks to the engine so every frontend observes the
    /// same tool, permission, and turn events.
    pub fn set_plugins(&mut self, plugins: Arc<PluginRegistry>) {
        self.plugins = Some(plugins);
    }

    async fn fire_hook(&self, trigger: &HookTrigger) {
        if let Some(plugins) = &self.plugins {
            if let Err(error) = plugins.execute_side_effects(trigger, None).await {
                tracing::warn!("plugin hook {trigger:?} failed: {error}");
            }
        }
    }

    fn begin_checkpoint(&mut self) {
        if !self.checkpoint_enabled {
            return;
        }
        self.last_checkpoint = None;
        self.pending_checkpoint = match PendingCheckpoint::capture() {
            Ok(checkpoint) => Some(checkpoint),
            Err(error) => {
                tracing::debug!("turn checkpoint unavailable: {error}");
                None
            }
        };
    }

    fn finish_checkpoint(&mut self) {
        let Some(pending) = self.pending_checkpoint.take() else {
            return;
        };
        match pending.finish() {
            Ok(checkpoint) => self.last_checkpoint = Some(checkpoint),
            Err(error) => tracing::warn!("could not finish turn checkpoint: {error}"),
        }
    }

    pub fn last_turn_diff(&self) -> String {
        self.last_checkpoint
            .as_ref()
            .map(TurnCheckpoint::diff)
            .unwrap_or_else(|| {
                "No turn checkpoint is available (checkpoints require a Git worktree).".to_string()
            })
    }

    pub fn undo_last_turn(&mut self) -> Result<String> {
        let checkpoint = self.last_checkpoint.as_ref().ok_or_else(|| {
            anyhow::anyhow!("No turn checkpoint is available (checkpoints require a Git worktree).")
        })?;
        let result = checkpoint.undo()?;
        self.last_checkpoint = None;
        self.provider.reset_session();
        self.messages.push(Message::user(
            "[Claux checkpoint] The user invoked /undo-turn. The previous turn's \
             checkpointed filesystem changes were reverted. Re-read affected files \
             before relying on the previous turn's results.",
        ));
        Ok(result)
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

    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        self.max_tokens = max_tokens.max(1);
    }

    pub fn set_model_metadata(&mut self, metadata: crate::model::ModelMetadata) {
        self.context_window = metadata.context_window;
        self.cost.set_pricing_override(metadata.pricing);
    }

    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    #[cfg(test)]
    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.provider.reset_session();
        self.permissions.reset_session();
        self.tools.reset_session();
        self.cost.reset_usage();
        self.steering
            .lock()
            .expect("steering queue poisoned")
            .clear();
        self.messages = messages;
        self.pending_checkpoint = None;
        self.last_checkpoint = None;
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model_binding(&mut self, binding: ModelBinding) {
        self.model_binding = Some(binding);
    }

    pub fn model_binding(&self) -> Option<&ModelBinding> {
        self.model_binding.as_ref()
    }

    pub fn set_theme(&mut self, _theme: crate::theme::ThemeName) {
        // Theme is handled by the TUI layer, not the engine.
        // This method exists for command parsing consistency.
        // The actual theme switch happens in the TUI's execute_async handler.
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Check if auto-compact is needed and perform it if so.
    /// Returns true if compaction was performed.
    pub async fn maybe_auto_compact(&mut self) -> Result<bool> {
        // Disabled if threshold is 0.0
        if self.auto_compact_threshold <= 0.0 {
            return Ok(false);
        }

        let current_tokens = compact::estimate_tokens(&self.messages);
        let threshold_tokens = (self.context_window as f64 * self.auto_compact_threshold) as usize;

        if current_tokens > threshold_tokens {
            tracing::info!(
                "Auto-compact triggered: {} tokens > {} (threshold: {:.0}% of {})",
                current_tokens,
                threshold_tokens,
                self.auto_compact_threshold * 100.0,
                self.context_window
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
        let mut summary_source = None;

        // Try snip first (cheaper, no API call)
        if let Some(snipped) = compact::snip_old_messages(&self.messages, 10) {
            let new_tokens = compact::estimate_tokens(&snipped);
            tracing::info!(
                "Snip compaction: {} msgs → {}, ~{} → ~{} tokens",
                old_count,
                snipped.len(),
                old_tokens,
                new_tokens
            );

            // If snip freed enough, we're done
            if new_tokens < old_tokens && new_tokens < self.context_window * 70 / 100 {
                let new_count = snipped.len();
                self.commit_compacted_messages(snipped);
                return Ok(format!(
                    "Snipped {} old messages (~{} tokens freed)",
                    old_count - new_count + 1, // +1 for snip marker
                    old_tokens - new_tokens
                ));
            }

            summary_source = Some(snipped);
        }

        // Full summarization. Keep the current history untouched until the
        // provider completes so a failed compact cannot discard context.
        let summary_source = summary_source.unwrap_or_else(|| self.messages.clone());
        self.summarize_conversation(summary_source).await
    }

    /// Full API-based conversation summary.
    async fn summarize_conversation(&mut self, messages: Vec<Message>) -> Result<String> {
        let summary_prompt = "Summarize the conversation so far in a concise paragraph. \
            Focus on what was discussed, what decisions were made, what files were modified, \
            and any outstanding tasks. Be specific about file paths and changes.";

        let old_count = messages.len();
        let mut summary_messages = messages;
        summary_messages.push(Message::user(summary_prompt));

        let mut rx = self
            .provider
            .stream(
                &summary_messages,
                &self.system_prompt,
                &[],
                self.max_tokens,
                tokio_util::sync::CancellationToken::new(),
            )
            .await?;

        let mut summary = String::new();
        let mut completed = false;
        while let Some(event) = rx.recv().await {
            match event {
                ApiEvent::Text(t) => summary.push_str(&t),
                ApiEvent::Usage(usage) => self.cost.add_usage(&usage),
                ApiEvent::Done => {
                    completed = true;
                    break;
                }
                ApiEvent::Error(failure) => {
                    let message = format!("Compact error: {}", failure.message);
                    return Err(anyhow::Error::new(ApiFailure::new(failure.kind, message)));
                }
                _ => {}
            }
        }
        if !completed {
            anyhow::bail!("Compact error: API stream ended without completion");
        }

        self.commit_compacted_messages(vec![
            Message::user("Here is a summary of our conversation so far:"),
            Message::assistant_text(&summary),
        ]);

        Ok(format!(
            "Compacted {old_count} messages into summary.\n\n\x1b[2m{summary}\x1b[0m"
        ))
    }

    /// Replacing history invalidates provider state indexed into the previous
    /// message vector (notably OpenAI Responses' `previous_response_id`
    /// cursor). Reset only at the successful mutation boundary so failed
    /// compaction leaves both history and provider state usable.
    fn commit_compacted_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
        self.provider.reset_session();
    }

    /// Recover the failure classification from a `stream()` error.
    ///
    /// Providers return `anyhow::Error` when the request fails before a stream
    /// exists; the underlying `ApiFailure` carries the classification, so
    /// downcast rather than inspecting the rendered message.
    fn failure_kind(error: &anyhow::Error) -> ApiFailureKind {
        error
            .downcast_ref::<ApiFailure>()
            .map(|failure| failure.kind)
            .unwrap_or(ApiFailureKind::Other)
    }

    fn malformed_tool_retry_prompt(err: &str) -> String {
        let detail = crate::utils::truncate_str(err, 512);
        format!(
            "Your previous response was rejected before any tools executed because one or more \
             tool calls contained invalid JSON arguments ({detail}). Reissue the entire intended \
             tool-call batch with valid JSON arguments. Do not assume any tool from the rejected \
             response ran."
        )
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
        self.begin_checkpoint();

        let collector = tokio::spawn(async move {
            let mut text = String::new();
            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Text(t) => text.push_str(&t),
                    StreamEvent::Retry(_) => text.clear(),
                    _ => {}
                }
            }
            text
        });

        let result = self.run_turn(user_input, tx, false, cancel).await;
        let text = collector.await.unwrap_or_default();
        self.finish_checkpoint();
        self.fire_hook(&HookTrigger::OnTurnEnd).await;
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
        self.begin_checkpoint();
        let result = self.run_turn(user_input, tx, true, cancel).await;
        self.finish_checkpoint();
        self.fire_hook(&HookTrigger::OnTurnEnd).await;
        result
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
        let mut malformed_tool_retries = 0;
        const MAX_MALFORMED_TOOL_RETRIES: u32 = 1;
        let mut retry_prompt: Option<String> = None;

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
            let effective_system_prompt = retry_prompt
                .as_ref()
                .map(|prompt| format!("{}\n\n{prompt}", self.system_prompt));
            let stream_result = self
                .provider
                .stream(
                    &self.messages,
                    effective_system_prompt
                        .as_deref()
                        .unwrap_or(&self.system_prompt),
                    &tool_defs,
                    self.max_tokens,
                    cancel.clone(),
                )
                .await;

            let mut rx = match stream_result {
                Ok(rx) => rx,
                Err(e) => {
                    if cancel.is_cancelled() {
                        let _ = tx.send(StreamEvent::Interrupted).await;
                        return Ok(());
                    }
                    let err_str = e.to_string();
                    match Self::failure_kind(&e) {
                        ApiFailureKind::MalformedToolArguments
                            if malformed_tool_retries < MAX_MALFORMED_TOOL_RETRIES =>
                        {
                            malformed_tool_retries += 1;
                            retry_prompt = Some(Self::malformed_tool_retry_prompt(&err_str));
                            let _ = tx
                                .send(StreamEvent::Retry(
                                    "model returned malformed tool arguments; retrying once"
                                        .to_string(),
                                ))
                                .await;
                            continue;
                        }
                        ApiFailureKind::OutputLimitExceeded if self.max_tokens < 64_000 => {
                            self.max_tokens = (self.max_tokens * 2).min(64_000);
                            continue;
                        }
                        ApiFailureKind::ContextExceeded if recovery_attempts < MAX_RECOVERY => {
                            recovery_attempts += 1;
                            let _ = tx
                                .send(StreamEvent::Notice(
                                    "compacting conversation...".to_string(),
                                ))
                                .await;
                            self.compact().await?;
                            continue;
                        }
                        _ => {}
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
                        None => {
                            let error = "API stream ended without completion".to_string();
                            let _ = tx.send(StreamEvent::Error(error.clone())).await;
                            return Err(anyhow::anyhow!(error));
                        }
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
                        self.fire_hook(&HookTrigger::OnToolStart).await;
                        let summary = self.tools.summarize(&name, &input);
                        let _ = tx
                            .send(StreamEvent::ToolStart {
                                name: name.clone(),
                                summary,
                            })
                            .await;
                        tool_uses.push((id, name, input));
                    }
                    ApiEvent::Usage(usage) => {
                        self.cost.add_usage(&usage);
                    }
                    ApiEvent::Done => break,
                    ApiEvent::Error(failure) => {
                        match failure.kind {
                            // `tool_uses.is_empty()` keeps the retry safe: a
                            // provider that emits tool calls one at a time may
                            // already have surfaced some of the batch, and
                            // reissuing would double-execute them.
                            ApiFailureKind::MalformedToolArguments
                                if tool_uses.is_empty()
                                    && malformed_tool_retries < MAX_MALFORMED_TOOL_RETRIES =>
                            {
                                malformed_tool_retries += 1;
                                retry_prompt =
                                    Some(Self::malformed_tool_retry_prompt(&failure.message));
                                let _ = tx
                                    .send(StreamEvent::Retry(
                                        "model returned malformed tool arguments; retrying once"
                                            .to_string(),
                                    ))
                                    .await;
                                had_error = true;
                                break;
                            }
                            ApiFailureKind::OutputLimitExceeded if self.max_tokens < 64_000 => {
                                self.max_tokens = (self.max_tokens * 2).min(64_000);
                                had_error = true;
                                break;
                            }
                            ApiFailureKind::ContextExceeded if recovery_attempts < MAX_RECOVERY => {
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
                            _ => {}
                        }
                        let _ = tx.send(StreamEvent::Error(failure.message.clone())).await;
                        // Keep the detail in the rendered message: `context`
                        // alone would leave `to_string()` as just "API error"
                        // and push the cause into the error source, which
                        // callers that print the error would drop.
                        let message = format!("API error: {}", failure.message);
                        return Err(anyhow::Error::new(ApiFailure::new(failure.kind, message)));
                    }
                }
            }

            if had_error {
                continue;
            }

            // A complete response ends the retry scope. Any correction was
            // request-local and must not become conversation history.
            malformed_tool_retries = 0;
            retry_prompt = None;

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
                    for (id, _, _) in &tool_uses {
                        self.fire_hook(&HookTrigger::OnToolComplete).await;
                        let _ = tx.send(StreamEvent::ToolResult { is_error: true }).await;
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

            self.fire_hook(&HookTrigger::OnToolComplete).await;
            let _ = tx
                .send(StreamEvent::ToolResult {
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
        self.fire_hook(&HookTrigger::OnPermissionRequest).await;
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
                match PermissionResponse::always_allow_for(name, input) {
                    PermissionResponse::AlwaysAllow => self.permissions.always_allow(name),
                    PermissionResponse::AlwaysAllowCommand(command) => {
                        self.permissions.always_allow_command(&command);
                    }
                    _ => {}
                }
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
    use crate::api::{MessageContent, ToolDefinition};
    use crate::permissions::PermissionMode;
    use crate::plugin::Plugin;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    // Mock provider for testing
    struct MockProvider;

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
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
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<ProviderStream> {
            let (tx, rx) = mpsc::channel(10);
            // Return empty stream for testing
            drop(tx);
            Ok(ProviderStream::new(rx, cancel.child_token()))
        }
    }

    struct TruncatedProvider;

    #[async_trait::async_trait]
    impl Provider for TruncatedProvider {
        fn name(&self) -> &str {
            "truncated"
        }

        fn set_model(&mut self, _model: &str) {}

        async fn stream(
            &self,
            _messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _max_tokens: u32,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<ProviderStream> {
            let (tx, rx) = mpsc::channel(10);
            let _ = tx
                .send(ApiEvent::Text("partial response".to_string()))
                .await;
            drop(tx);
            Ok(ProviderStream::new(rx, cancel.child_token()))
        }
    }

    struct MalformedToolProvider {
        calls: Arc<AtomicUsize>,
        systems: Arc<Mutex<Vec<String>>>,
        recover: bool,
    }

    #[async_trait::async_trait]
    impl Provider for MalformedToolProvider {
        fn name(&self) -> &str {
            "malformed-tool"
        }

        fn set_model(&mut self, _model: &str) {}

        async fn stream(
            &self,
            _messages: &[Message],
            system: &str,
            _tools: &[ToolDefinition],
            _max_tokens: u32,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<ProviderStream> {
            self.systems.lock().unwrap().push(system.to_string());
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = mpsc::channel(10);
            if attempt == 0 {
                tx.send(ApiEvent::Text("rejected preamble".to_string()))
                    .await
                    .unwrap();
            }
            if self.recover && attempt > 0 {
                tx.send(ApiEvent::Text("recovered response".to_string()))
                    .await
                    .unwrap();
                tx.send(ApiEvent::Done).await.unwrap();
            } else {
                tx.send(ApiEvent::Error(ApiFailure::malformed_tool_arguments(
                    "OpenAI SSE stream error: invalid arguments for tool call Read \
                     (call_3): EOF while parsing a value",
                )))
                .await
                .unwrap();
            }
            drop(tx);
            Ok(ProviderStream::new(rx, cancel.child_token()))
        }
    }

    struct ResetTrackingProvider {
        resets: Arc<std::sync::atomic::AtomicUsize>,
    }

    struct CompactionTrackingProvider {
        resets: Arc<AtomicUsize>,
        complete: bool,
    }

    #[async_trait::async_trait]
    impl Provider for CompactionTrackingProvider {
        fn name(&self) -> &str {
            "compaction-tracking"
        }

        fn set_model(&mut self, _model: &str) {}

        fn reset_session(&mut self) {
            self.resets.fetch_add(1, Ordering::SeqCst);
        }

        async fn stream(
            &self,
            _messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _max_tokens: u32,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<ProviderStream> {
            let (tx, rx) = mpsc::channel(2);
            tx.send(ApiEvent::Text("compacted summary".to_string()))
                .await
                .unwrap();
            if self.complete {
                tx.send(ApiEvent::Done).await.unwrap();
            }
            drop(tx);
            Ok(ProviderStream::new(rx, cancel.child_token()))
        }
    }

    /// Provider that fails every request with a given classified failure,
    /// counting attempts. Lets the recovery tests assert on what the turn
    /// loop *did* rather than on how an error string was spelled.
    struct FailingProvider {
        failure: ApiFailure,
        calls: Arc<AtomicUsize>,
        max_tokens_seen: Arc<Mutex<Vec<u32>>>,
    }

    #[async_trait::async_trait]
    impl Provider for FailingProvider {
        fn name(&self) -> &str {
            "failing"
        }

        fn set_model(&mut self, _model: &str) {}

        async fn stream(
            &self,
            _messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            max_tokens: u32,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<ProviderStream> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.max_tokens_seen.lock().unwrap().push(max_tokens);
            let (tx, rx) = mpsc::channel(2);
            tx.send(ApiEvent::Error(self.failure.clone()))
                .await
                .unwrap();
            drop(tx);
            Ok(ProviderStream::new(rx, cancel.child_token()))
        }
    }

    /// Run one turn against a provider that always fails with `failure`,
    /// returning (attempt count, max_tokens seen per attempt).
    async fn recovery_attempts_for(failure: ApiFailure) -> (usize, Vec<u32>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let max_tokens_seen = Arc::new(Mutex::new(Vec::new()));
        let provider = Box::new(FailingProvider {
            failure,
            calls: calls.clone(),
            max_tokens_seen: max_tokens_seen.clone(),
        });
        let mut engine =
            Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Bypass);

        let _ = engine
            .submit("go", tokio_util::sync::CancellationToken::new())
            .await;

        let seen = max_tokens_seen.lock().unwrap().clone();
        (calls.load(Ordering::SeqCst), seen)
    }

    #[tokio::test]
    async fn an_output_limit_escalates_max_tokens_without_compacting() {
        // Previously keyed off the substring "max_output_tokens"; now keyed
        // off the classification, so the recovery cannot be reached by an
        // error that merely mentions the phrase.
        let (attempts, max_tokens) =
            recovery_attempts_for(ApiFailure::output_limit_exceeded("output limit")).await;

        assert!(attempts > 1, "the turn should retry with a larger budget");
        assert!(
            max_tokens.windows(2).all(|pair| pair[1] > pair[0]),
            "max_tokens must escalate on each retry, got {max_tokens:?}"
        );
        assert_eq!(
            *max_tokens.last().unwrap(),
            64_000,
            "escalation stops at the ceiling"
        );
    }

    #[tokio::test]
    async fn an_unclassified_failure_triggers_no_recovery() {
        // The case the substring predicates got wrong: an error whose text
        // happens to contain "413" or "max_output_tokens" but which is
        // neither condition. It must fail fast, not burn retries.
        let (attempts, _) = recovery_attempts_for(ApiFailure::other(
            "internal error (request req_413_88): invalid max_tokens parameter",
        ))
        .await;

        assert_eq!(
            attempts, 1,
            "an unclassified failure must not trigger compaction or escalation"
        );
    }

    #[tokio::test]
    async fn a_context_overflow_attempts_compaction() {
        // The failing provider also serves the summarization request, so the
        // compact fails and ends the turn: attempt 1 is the turn, attempt 2 is
        // the compaction it triggered. What matters is that ContextExceeded
        // routes to compaction at all — an unclassified failure does not
        // (see `an_unclassified_failure_triggers_no_recovery`).
        let (attempts, _) =
            recovery_attempts_for(ApiFailure::new(ApiFailureKind::ContextExceeded, "too long"))
                .await;

        assert_eq!(
            attempts, 2,
            "a context overflow must trigger a compaction attempt"
        );
    }

    #[test]
    fn stream_errors_carry_their_classification_to_the_turn_loop() {
        // The turn loop downcasts `stream()` errors; a failure that loses its
        // type on the way through anyhow would silently stop being recoverable.
        let error = anyhow::Error::new(ApiFailure::malformed_tool_arguments("bad args"));
        assert_eq!(
            Engine::failure_kind(&error),
            ApiFailureKind::MalformedToolArguments
        );

        let untyped = anyhow::anyhow!("invalid arguments for tool call Read (call_3)");
        assert_eq!(
            Engine::failure_kind(&untyped),
            ApiFailureKind::Other,
            "prose alone must not be treated as a classification"
        );
    }

    #[tokio::test]
    async fn malformed_tool_arguments_retry_once_without_persisting_rejected_text() {
        let calls = Arc::new(AtomicUsize::new(0));
        let systems = Arc::new(Mutex::new(Vec::new()));
        let provider = Box::new(MalformedToolProvider {
            calls: calls.clone(),
            systems: systems.clone(),
            recover: true,
        });
        let mut engine =
            Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Default);
        engine.set_system_prompt("base system prompt".to_string());

        let response = engine
            .submit("hello", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(response, "recovered response");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let systems = systems.lock().unwrap();
        assert_eq!(systems[0], "base system prompt");
        assert!(systems[1].starts_with("base system prompt\n\n"));
        assert!(systems[1].contains("before any tools executed"));
        assert!(systems[1].contains("Reissue the entire intended tool-call batch"));

        assert_eq!(engine.messages().len(), 2);
        let MessageContent::Blocks(blocks) = &engine.messages()[1].content else {
            panic!("expected assistant blocks");
        };
        assert!(matches!(
            blocks.as_slice(),
            [ContentBlock::Text { text }] if text == "recovered response"
        ));
    }

    #[tokio::test]
    async fn malformed_tool_arguments_stop_after_one_retry() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(MalformedToolProvider {
            calls: calls.clone(),
            systems: Arc::new(Mutex::new(Vec::new())),
            recover: false,
        });
        let mut engine =
            Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Default);

        let error = engine
            .submit("hello", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap_err();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(error
            .to_string()
            .contains("invalid arguments for tool call"));
        assert_eq!(
            engine.messages().len(),
            1,
            "rejected assistant attempts must not enter conversation history"
        );
    }

    #[async_trait::async_trait]
    impl Provider for ResetTrackingProvider {
        fn name(&self) -> &str {
            "reset-tracking"
        }

        fn set_model(&mut self, _model: &str) {}

        fn reset_session(&mut self) {
            self.resets
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        async fn stream(
            &self,
            _messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _max_tokens: u32,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<ProviderStream> {
            let (_tx, rx) = mpsc::channel(1);
            Ok(ProviderStream::new(rx, cancel.child_token()))
        }
    }

    #[test]
    fn set_messages_resets_session_scoped_engine_state() {
        let resets = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let provider = Box::new(ResetTrackingProvider {
            resets: resets.clone(),
        });
        let mut engine = Engine::new(
            provider,
            ToolRegistry::without_agent_for_tests(),
            PermissionChecker::new(PermissionMode::Default),
            "private-model",
        );
        engine
            .cost
            .set_pricing_override(Some(crate::cost::ModelPricing {
                input: 2.0,
                output: 4.0,
                cache_read: 0.5,
                cache_write: 1.0,
            }));
        engine.cost.add_usage(&crate::api::types::Usage {
            input_tokens: 500,
            output_tokens: 200,
            cache_read_tokens: 100,
            cache_creation_tokens: 50,
            provider_cost_usd: None,
        });
        engine
            .steering_queue()
            .lock()
            .unwrap()
            .push_back("stale steering".to_string());
        engine.permissions.always_allow("Write");
        engine.permissions.always_allow_command("cargo test");

        engine.set_messages(vec![Message::user("loaded session")]);

        assert_eq!(resets.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(engine.message_count(), 1);
        assert!(engine.steering_queue().lock().unwrap().is_empty());
        assert_eq!(engine.cost.input_tokens, 0);
        assert_eq!(engine.cost.output_tokens, 0);
        assert!(matches!(
            engine.permissions.check(
                "Write",
                &serde_json::json!({"file_path": "/tmp/test"}),
                false
            ),
            PermissionResult::Ask { .. }
        ));
        assert!(matches!(
            engine
                .permissions
                .check("Bash", &serde_json::json!({"command": "cargo test"}), false),
            PermissionResult::Ask { .. }
        ));

        engine.cost.add_usage(&crate::api::types::Usage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: None,
        });
        assert_eq!(engine.cost.total_cost_usd(), 2.0);
    }

    #[test]
    fn resolved_model_metadata_configures_compaction_and_cost() {
        let mut engine = Engine::for_tests(
            Box::new(MockProvider),
            SteeringQueue::default(),
            PermissionMode::Default,
        );
        engine.set_model_metadata(crate::model::ModelMetadata {
            context_window: 64_000,
            pricing: Some(crate::cost::ModelPricing {
                input: 2.0,
                output: 4.0,
                cache_read: 0.5,
                cache_write: 1.0,
            }),
        });
        engine.cost.add_usage(&crate::api::types::Usage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: None,
        });

        assert_eq!(engine.context_window, 64_000);
        assert_eq!(engine.cost.total_cost_usd(), 2.0);
    }

    #[tokio::test]
    async fn test_parallel_tool_execution() {
        // Create a mock engine with read-only tools
        let provider = Box::new(MockProvider);
        let tools = ToolRegistry::without_agent_for_tests();
        let permissions = PermissionChecker::new(PermissionMode::Bypass);

        let mut engine = Engine {
            provider,
            tools,
            permissions,
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            model_binding: None,
            max_tokens: 1000,
            context_window: 128_000,
            auto_compact_threshold: 0.8,
            steering: SteeringQueue::default(),
            plugins: None,
            checkpoint_enabled: false,
            pending_checkpoint: None,
            last_checkpoint: None,
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
        let tools = ToolRegistry::without_agent_for_tests();
        let permissions = PermissionChecker::new(PermissionMode::Bypass);

        let mut engine = Engine {
            provider,
            tools,
            permissions,
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            model_binding: None,
            max_tokens: 1000,
            context_window: 128_000,
            auto_compact_threshold: 0.8,
            steering: SteeringQueue::default(),
            plugins: None,
            checkpoint_enabled: false,
            pending_checkpoint: None,
            last_checkpoint: None,
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

    struct CountingPlugin {
        trigger: HookTrigger,
        count: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Plugin for CountingPlugin {
        fn name(&self) -> &str {
            "counter"
        }

        fn trigger(&self) -> &HookTrigger {
            &self.trigger
        }

        async fn execute(
            &self,
            _env_vars: Option<&HashMap<String, String>>,
        ) -> Result<Option<String>> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(None)
        }
    }

    #[tokio::test]
    async fn one_shot_submit_fires_tool_and_turn_hooks() {
        let starts = Arc::new(AtomicUsize::new(0));
        let completes = Arc::new(AtomicUsize::new(0));
        let turns = Arc::new(AtomicUsize::new(0));
        let mut plugins = PluginRegistry::new();
        for (trigger, count) in [
            (HookTrigger::OnToolStart, starts.clone()),
            (HookTrigger::OnToolComplete, completes.clone()),
            (HookTrigger::OnTurnEnd, turns.clone()),
        ] {
            plugins.add(Box::new(CountingPlugin { trigger, count }));
        }

        let mut engine = steering_engine(
            vec![crate::test_support::tool_use(
                "read-1",
                "Read",
                serde_json::json!({"file_path": "/dev/null"}),
            )],
            None,
        );
        engine.set_plugins(Arc::new(plugins));
        engine
            .submit("read it", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(completes.load(Ordering::SeqCst), 1);
        assert_eq!(turns.load(Ordering::SeqCst), 1);
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
    async fn test_submit_rejects_stream_closed_without_done() {
        let mut engine = Engine::for_tests(
            Box::new(TruncatedProvider),
            SteeringQueue::default(),
            PermissionMode::Bypass,
        );

        let error = engine
            .submit("go", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap_err();

        assert!(error.to_string().contains("without completion"));
        assert_eq!(
            engine.messages().len(),
            1,
            "partial assistant content must not be committed to history"
        );
        assert_eq!(engine.messages()[0].role, "user");
    }

    #[tokio::test]
    async fn test_compact_rejects_stream_closed_without_done() {
        let mut engine = Engine::for_tests(
            Box::new(TruncatedProvider),
            SteeringQueue::default(),
            PermissionMode::Bypass,
        );
        engine
            .messages_mut()
            .push(Message::user("important context"));

        let error = engine.compact().await.unwrap_err();

        assert!(error.to_string().contains("without completion"));
        assert_eq!(
            engine.messages().len(),
            1,
            "failed compaction must preserve the original history"
        );
    }

    #[tokio::test]
    async fn snip_compaction_resets_provider_cursor_after_rewriting_history() {
        let resets = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(ResetTrackingProvider {
            resets: resets.clone(),
        });
        let mut engine =
            Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Bypass);

        let old_content = "old context ".repeat(1_000);
        engine.messages_mut().extend([
            Message::user(&old_content),
            Message::assistant_text(&old_content),
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({"file_path": "README.md"}),
            }]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "contents".to_string(),
                is_error: None,
            }]),
        ]);
        for index in 4..13 {
            engine
                .messages_mut()
                .push(Message::user(&format!("recent message {index}")));
        }

        engine.compact().await.unwrap();

        assert_eq!(
            engine.messages().len(),
            12,
            "tool-result boundary backoff reproduces the stale cursor index shape"
        );
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn summary_compaction_resets_provider_cursor_after_rewriting_history() {
        let resets = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(CompactionTrackingProvider {
            resets: resets.clone(),
            complete: true,
        });
        let mut engine =
            Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Bypass);
        let large_message = "context ".repeat(10_000);
        for index in 0..13 {
            engine
                .messages_mut()
                .push(Message::user(&format!("{index}: {large_message}")));
        }

        engine.compact().await.unwrap();

        assert_eq!(engine.messages().len(), 2);
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn snip_candidate_that_increases_tokens_is_not_committed() {
        let resets = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(CompactionTrackingProvider {
            resets: resets.clone(),
            complete: true,
        });
        let mut engine =
            Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Bypass);
        for index in 0..13 {
            engine
                .messages_mut()
                .push(Message::user(&index.to_string()));
        }

        engine.compact().await.unwrap();

        assert_eq!(
            engine.messages().len(),
            2,
            "a larger snip candidate should fall back to summary compaction"
        );
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_summary_preserves_history_after_snipping_candidate() {
        let resets = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(CompactionTrackingProvider {
            resets: resets.clone(),
            complete: false,
        });
        let mut engine =
            Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Bypass);
        let large_message = "context ".repeat(10_000);
        for index in 0..13 {
            engine
                .messages_mut()
                .push(Message::user(&format!("{index}: {large_message}")));
        }
        let original = serde_json::to_value(engine.messages()).unwrap();

        let error = engine.compact().await.unwrap_err();

        assert!(error.to_string().contains("without completion"));
        assert_eq!(
            serde_json::to_value(engine.messages()).unwrap(),
            original,
            "failed summarization must not commit the snipped candidate"
        );
        assert_eq!(
            resets.load(Ordering::SeqCst),
            0,
            "failed compaction must preserve provider continuation state"
        );
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
        let tools = ToolRegistry::without_agent_for_tests();
        let permissions = PermissionChecker::new(PermissionMode::Bypass);

        let mut engine = Engine {
            provider,
            tools,
            permissions,
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            model_binding: None,
            max_tokens: 1000,
            context_window: 128_000,
            auto_compact_threshold: 0.8,
            steering: SteeringQueue::default(),
            plugins: None,
            checkpoint_enabled: false,
            pending_checkpoint: None,
            last_checkpoint: None,
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
        let tools = ToolRegistry::without_agent_for_tests();
        let permissions = PermissionChecker::new(PermissionMode::Default);

        let mut engine = Engine {
            provider,
            tools,
            permissions,
            messages: vec![],
            system_prompt: String::new(),
            model: "test".to_string(),
            model_binding: None,
            max_tokens: 1000,
            context_window: 128_000,
            auto_compact_threshold: 0.8,
            steering: SteeringQueue::default(),
            plugins: None,
            checkpoint_enabled: false,
            pending_checkpoint: None,
            last_checkpoint: None,
            cost: CostTracker::new("test"),
        };

        // Under PermissionMode::Default, network reads ask for confirmation.
        let tool_uses = vec![(
            "test1".to_string(),
            "WebFetch".to_string(),
            serde_json::json!({"url": "https://example.com/private"}),
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
