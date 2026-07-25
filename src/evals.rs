//! Fixture-driven behavioral evaluations for the complete agent loop.
//!
//! These are intentionally deterministic: the provider is scripted, while
//! the real engine, permissions, tools, steering, and conversation history
//! execute unchanged.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use tokio::sync::mpsc;

use crate::api::{
    ApiEvent, ContentBlock, Message, MessageContent, Provider, ProviderStream, ToolDefinition,
};
use crate::permissions::PermissionMode;
use crate::query::{Engine, SteeringQueue};

#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    permission_mode: PermissionMode,
    prompt: String,
    #[serde(default)]
    initial_files: BTreeMap<String, String>,
    #[serde(default)]
    steering_on_first_call: Option<String>,
    rounds: Vec<Round>,
    expect: Expectations,
}

#[derive(Clone, Debug, Deserialize)]
struct Round {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    tools: Vec<ToolCall>,
    #[serde(default)]
    terminal: Terminal,
}

#[derive(Clone, Debug, Deserialize)]
struct ToolCall {
    id: String,
    name: String,
    input: serde_json::Value,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Terminal {
    #[default]
    Done,
    Close,
    Error,
}

#[derive(Debug, Deserialize)]
struct Expectations {
    #[serde(default)]
    assistant_contains: Option<String>,
    #[serde(default)]
    error_contains: Option<String>,
    tool_sequence: Vec<String>,
    tool_error_count: usize,
    #[serde(default)]
    tool_result_contains: Vec<String>,
    #[serde(default)]
    steering_delivered: bool,
    #[serde(default)]
    files: BTreeMap<String, Option<String>>,
}

struct FixtureProvider {
    rounds: Mutex<VecDeque<Round>>,
    steering: Option<(SteeringQueue, String)>,
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl Provider for FixtureProvider {
    fn name(&self) -> &str {
        "eval-fixture"
    }

    fn model(&self) -> &str {
        "eval-model"
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
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            if let Some((queue, message)) = &self.steering {
                queue.lock().unwrap().push_back(message.clone());
            }
        }

        let round = self.rounds.lock().unwrap().pop_front().unwrap_or(Round {
            text: None,
            tools: Vec::new(),
            terminal: Terminal::Done,
        });
        let (tx, rx) = mpsc::channel(32);
        if let Some(text) = round.text {
            let _ = tx.send(ApiEvent::Text(text)).await;
        }
        for tool in round.tools {
            let _ = tx
                .send(ApiEvent::ToolUse {
                    id: tool.id,
                    name: tool.name,
                    input: tool.input,
                })
                .await;
        }
        match round.terminal {
            Terminal::Done => {
                let _ = tx.send(ApiEvent::Done).await;
            }
            Terminal::Close => {}
            Terminal::Error => {
                let _ = tx
                    .send(ApiEvent::Error("scripted provider failure".to_string()))
                    .await;
            }
        }
        Ok(ProviderStream::new(rx, cancel.child_token()))
    }
}

#[tokio::test]
async fn deterministic_agent_contracts() {
    let scenarios: Vec<Scenario> =
        serde_json::from_str(include_str!("../evals/fixtures/agent_contracts.json"))
            .expect("evaluation fixtures must parse");
    let total = scenarios.len();

    for scenario in scenarios {
        let name = scenario.name.clone();
        if let Err(error) = run_scenario(scenario).await {
            panic!("agent behavior evaluation failed: {error:#}");
        }
        eprintln!("[eval ok] {name}");
    }
    eprintln!("[eval summary] {total}/{total} deterministic contracts passed");
}

async fn run_scenario(mut scenario: Scenario) -> Result<()> {
    let workspace = tempfile::tempdir().context("create evaluation workspace")?;
    for (relative, content) in &scenario.initial_files {
        let path = safe_join(workspace.path(), relative)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
    }

    let root = workspace.path().to_string_lossy();
    for round in &mut scenario.rounds {
        for tool in &mut round.tools {
            replace_root(&mut tool.input, &root);
        }
    }

    let steering = SteeringQueue::default();
    let provider = Box::new(FixtureProvider {
        rounds: Mutex::new(scenario.rounds.into()),
        steering: scenario
            .steering_on_first_call
            .clone()
            .map(|message| (steering.clone(), message)),
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let mut engine = Engine::for_tests(provider, steering, scenario.permission_mode);
    let result = engine
        .submit(&scenario.prompt, tokio_util::sync::CancellationToken::new())
        .await;

    match (&scenario.expect.error_contains, result) {
        (Some(expected), Err(error)) => {
            ensure(
                error.to_string().contains(expected),
                &scenario.name,
                format!("expected error containing {expected:?}, got {error:?}"),
            )?;
        }
        (Some(expected), Ok(text)) => {
            anyhow::bail!(
                "{}: expected error containing {expected:?}, got success {text:?}",
                scenario.name
            );
        }
        (None, Err(error)) => {
            return Err(error).with_context(|| format!("{}: turn failed", scenario.name));
        }
        (None, Ok(text)) => {
            if let Some(expected) = &scenario.expect.assistant_contains {
                ensure(
                    text.contains(expected),
                    &scenario.name,
                    format!("assistant output did not contain {expected:?}: {text:?}"),
                )?;
            }
        }
    }

    let observed = observed_contract(engine.messages());
    ensure(
        observed.tool_sequence == scenario.expect.tool_sequence,
        &scenario.name,
        format!(
            "tool sequence mismatch: expected {:?}, got {:?}",
            scenario.expect.tool_sequence, observed.tool_sequence
        ),
    )?;
    ensure(
        observed.tool_error_count == scenario.expect.tool_error_count,
        &scenario.name,
        format!(
            "tool error count mismatch: expected {}, got {}",
            scenario.expect.tool_error_count, observed.tool_error_count
        ),
    )?;
    for expected in &scenario.expect.tool_result_contains {
        ensure(
            observed
                .tool_results
                .iter()
                .any(|result| result.contains(expected)),
            &scenario.name,
            format!("no tool result contained {expected:?}"),
        )?;
    }
    if scenario.expect.steering_delivered {
        let steering = scenario
            .steering_on_first_call
            .as_deref()
            .unwrap_or_default();
        ensure(
            engine.messages().iter().any(
                |message| matches!(&message.content, MessageContent::Text(text) if text == steering),
            ),
            &scenario.name,
            "steering message was not delivered into conversation".to_string(),
        )?;
    }
    for (relative, expected) in &scenario.expect.files {
        let path = safe_join(workspace.path(), relative)?;
        match expected {
            Some(expected) => {
                let actual = std::fs::read_to_string(&path)
                    .with_context(|| format!("{}: expected file {}", scenario.name, relative))?;
                ensure(
                    &actual == expected,
                    &scenario.name,
                    format!("file {relative} mismatch: expected {expected:?}, got {actual:?}"),
                )?;
            }
            None => ensure(
                !path.exists(),
                &scenario.name,
                format!("file {relative} should not exist"),
            )?,
        }
    }
    Ok(())
}

struct ObservedContract {
    tool_sequence: Vec<String>,
    tool_error_count: usize,
    tool_results: Vec<String>,
}

fn observed_contract(messages: &[Message]) -> ObservedContract {
    let mut observed = ObservedContract {
        tool_sequence: Vec::new(),
        tool_error_count: 0,
        tool_results: Vec::new(),
    };
    for message in messages {
        let MessageContent::Blocks(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            match block {
                ContentBlock::ToolUse { name, .. } => observed.tool_sequence.push(name.clone()),
                ContentBlock::ToolResult {
                    content, is_error, ..
                } => {
                    observed.tool_error_count += usize::from(is_error == &Some(true));
                    observed.tool_results.push(content.clone());
                }
                ContentBlock::Text { .. } => {}
            }
        }
    }
    observed
}

fn replace_root(value: &mut serde_json::Value, root: &str) {
    match value {
        serde_json::Value::String(text) => *text = text.replace("$ROOT", root),
        serde_json::Value::Array(values) => {
            for value in values {
                replace_root(value, root);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                replace_root(value, root);
            }
        }
        _ => {}
    }
}

fn safe_join(root: &std::path::Path, relative: &str) -> Result<std::path::PathBuf> {
    let relative = std::path::Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        anyhow::bail!("unsafe evaluation fixture path: {}", relative.display());
    }
    Ok(root.join(relative))
}

fn ensure(condition: bool, scenario: &str, message: String) -> Result<()> {
    if !condition {
        anyhow::bail!("{scenario}: {message}");
    }
    Ok(())
}

/// Opt-in, paid smoke test. Run explicitly with:
/// `ANTHROPIC_API_KEY=... cargo test evals::live_anthropic_smoke -- --ignored`
#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a paid network request"]
async fn live_anthropic_smoke() {
    let key = std::env::var("ANTHROPIC_API_KEY")
        .expect("set ANTHROPIC_API_KEY to run the live provider smoke test");
    let model = std::env::var("CLAUX_EVAL_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
    let provider = Box::new(crate::api::AnthropicProvider::new(
        crate::config::AnthropicApiKey::new(key),
        &model,
    ));
    let mut engine = Engine::for_tests(provider, SteeringQueue::default(), PermissionMode::Plan);
    engine.set_system_prompt(
        "You are a provider smoke test. Follow the user's exact response format.".to_string(),
    );

    let response = engine
        .submit(
            "Reply with exactly CLAUX_EVAL_OK",
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("live provider request should succeed");
    assert!(
        response.contains("CLAUX_EVAL_OK"),
        "unexpected live response: {response:?}"
    );
}
