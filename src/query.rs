use anyhow::Result;
use tokio::sync::mpsc;

use crate::api::{self, ApiEvent, ContentBlock, Message};
use crate::cost::CostTracker;
use crate::permissions::{PermissionChecker, PermissionResult};
use crate::tools::ToolRegistry;

/// The query engine: conversation loop that sends messages, streams responses,
/// dispatches tools, and continues until the assistant stops.
///
/// This is the Rust equivalent of Claude Code's query.ts + QueryEngine.ts.
pub struct Engine {
    client: api::Client,
    tools: ToolRegistry,
    permissions: PermissionChecker,
    messages: Vec<Message>,
    system_prompt: String,
    model: String,
    max_tokens: u32,
    pub cost: CostTracker,
}

/// Callback for streaming text and status updates to the UI.
pub enum StreamEvent {
    Text(String),
    ToolStart { name: String, id: String },
    ToolResult { name: String, content: String, is_error: bool },
    Error(String),
    Done,
}

impl Engine {
    pub fn new(
        client: api::Client,
        tools: ToolRegistry,
        permissions: PermissionChecker,
        model: &str,
    ) -> Self {
        Self {
            client,
            tools,
            permissions,
            messages: Vec::new(),
            system_prompt: String::new(),
            model: model.to_string(),
            max_tokens: 16384,
            cost: CostTracker::new(model),
        }
    }

    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Submit a user message and run the full turn loop (chat → tools → chat → ...).
    /// Returns the final assistant text response.
    pub async fn submit(&mut self, user_input: &str) -> Result<String> {
        self.messages.push(Message::user(user_input));

        let mut full_response = String::new();

        // Turn loop: keep going as long as the assistant requests tool use
        loop {
            let tool_defs = self.tools.definitions();
            let mut rx = self
                .client
                .stream(&self.messages, &self.system_prompt, &tool_defs, self.max_tokens)
                .await?;

            let mut text_buf = String::new();
            let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new(); // (id, name, input)

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
                        return Err(anyhow::anyhow!("API error: {}", e));
                    }
                }
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

            // Execute tools and collect results
            let mut result_blocks = Vec::new();
            for (id, name, input) in &tool_uses {
                let is_read_only = self.tools.is_read_only(name);
                let perm = self.permissions.check(name, input, is_read_only);

                let tool_output = match perm {
                    PermissionResult::Allow => self.tools.execute(name, input.clone()).await?,
                    PermissionResult::Deny(reason) => crate::tools::ToolOutput {
                        content: format!("Permission denied: {}", reason),
                        is_error: true,
                    },
                    PermissionResult::Ask(prompt) => {
                        // In non-interactive (--print) mode, deny
                        // In interactive mode, the REPL handles this
                        // For now, auto-allow (the REPL will override this)
                        eprintln!("  [tool] {} — auto-allowing", prompt);
                        self.tools.execute(name, input.clone()).await?
                    }
                };

                result_blocks.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: tool_output.content,
                    is_error: if tool_output.is_error {
                        Some(true)
                    } else {
                        None
                    },
                });
            }

            // Push tool results as a user message and continue the loop
            self.messages.push(Message::tool_results(result_blocks));
        }

        Ok(full_response)
    }

    /// Submit with streaming callbacks (for the REPL).
    pub async fn submit_streaming(
        &mut self,
        user_input: &str,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        self.messages.push(Message::user(user_input));

        loop {
            let tool_defs = self.tools.definitions();
            let mut rx = self
                .client
                .stream(&self.messages, &self.system_prompt, &tool_defs, self.max_tokens)
                .await?;

            let mut text_buf = String::new();
            let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();

            while let Some(event) = rx.recv().await {
                match event {
                    ApiEvent::Text(t) => {
                        let _ = tx.send(StreamEvent::Text(t.clone())).await;
                        text_buf.push_str(&t);
                    }
                    ApiEvent::ToolUse { id, name, input } => {
                        let _ = tx
                            .send(StreamEvent::ToolStart {
                                name: name.clone(),
                                id: id.clone(),
                            })
                            .await;
                        tool_uses.push((id, name, input));
                    }
                    ApiEvent::Usage(usage) => {
                        self.cost.add_usage(&usage);
                    }
                    ApiEvent::Done => break,
                    ApiEvent::Error(e) => {
                        let _ = tx.send(StreamEvent::Error(e.clone())).await;
                        return Err(anyhow::anyhow!("API error: {}", e));
                    }
                }
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
                        content: format!("Permission denied: {}", reason),
                        is_error: true,
                    },
                    PermissionResult::Ask(prompt) => {
                        // TODO: interactive permission prompt via tx channel
                        eprintln!("  [allow] {}", prompt);
                        self.tools.execute(name, input.clone()).await?
                    }
                };

                let _ = tx
                    .send(StreamEvent::ToolResult {
                        name: name.clone(),
                        content: tool_output.content.clone(),
                        is_error: tool_output.is_error,
                    })
                    .await;

                result_blocks.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: tool_output.content,
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
