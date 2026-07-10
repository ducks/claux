//! Shared test fixtures: a scripted provider and engine builder used by
//! the engine tests (src/query.rs) and the TUI interaction tests
//! (src/tui/chat.rs).

use anyhow::Result;
use tokio::sync::mpsc;

use crate::api::{ApiEvent, Message, Provider, ToolDefinition};
use crate::permissions::PermissionMode;
use crate::query::{Engine, SteeringQueue};

/// Provider that emits scripted text and tool_uses on its first call and
/// ends the turn on the second. Optionally pushes a steering message
/// during the first call, deterministically simulating a user typing
/// while the model streams.
pub struct ScriptedProvider {
    pub calls: std::sync::atomic::AtomicUsize,
    pub first_round_text: Option<String>,
    pub first_round: Vec<(String, String, serde_json::Value)>,
    pub push_on_first_call: Option<(SteeringQueue, String)>,
}

#[async_trait::async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted-mock"
    }

    fn model(&self) -> &str {
        "test-model"
    }

    fn set_model(&mut self, _model: &str) {}

    async fn stream(
        &self,
        _messages: &[Message],
        _system: &str,
        _tools: &[ToolDefinition],
        _max_tokens: u32,
    ) -> Result<mpsc::Receiver<ApiEvent>> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(10);
        if call == 0 {
            if let Some((queue, text)) = &self.push_on_first_call {
                queue.lock().unwrap().push_back(text.clone());
            }
            if let Some(text) = &self.first_round_text {
                let _ = tx.send(ApiEvent::Text(text.clone())).await;
            }
            for (id, name, input) in &self.first_round {
                let _ = tx
                    .send(ApiEvent::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    })
                    .await;
            }
        }
        let _ = tx.send(ApiEvent::Done).await;
        Ok(rx)
    }
}

/// Build an Engine over a ScriptedProvider. `push_on_first_call` queues a
/// steering message the moment the first stream starts.
pub fn scripted_engine(
    first_round: Vec<(String, String, serde_json::Value)>,
    push_on_first_call: Option<String>,
    mode: PermissionMode,
) -> Engine {
    let steering = SteeringQueue::default();
    let provider = Box::new(ScriptedProvider {
        calls: std::sync::atomic::AtomicUsize::new(0),
        first_round_text: Some("working on it".to_string()),
        first_round,
        push_on_first_call: push_on_first_call.map(|t| (steering.clone(), t)),
    });
    Engine::for_tests(provider, steering, mode)
}

/// One (id, name, input) tool_use tuple, briefly.
pub fn tool_use(
    id: &str,
    name: &str,
    input: serde_json::Value,
) -> (String, String, serde_json::Value) {
    (id.to_string(), name.to_string(), input)
}

/// Serve one HTTP response over loopback and return it as a reqwest response.
/// API stream parser tests use this to exercise clean EOF behavior.
pub async fn sse_response(body: &str) -> reqwest::Response {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let body = body.to_string();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 1024];
        let _ = socket.read(&mut request).await;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    reqwest::get(format!("http://{address}")).await.unwrap()
}
