use crate::api::Message;
use crate::cost::{CostTracker, UsageSummary};
use crate::query::{ExecutionTiming, FailureRecord, ToolTraceEntry};
use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// One-shot `--output-format json` payload. Written on success and on
/// failure; `result` is null and `outcome` carries the classified failure
/// when the turn did not complete. Additive to schema version 1.
#[derive(Debug, Serialize)]
pub struct OneShotOutput<'a> {
    pub schema_version: u8,
    pub result: Option<&'a str>,
    pub model: &'a str,
    pub usage: UsageSummary,
    pub outcome: TranscriptOutcome<'a>,
}

impl<'a> OneShotOutput<'a> {
    pub fn new(result: &'a str, model: &'a str, cost: &CostTracker) -> Self {
        Self {
            schema_version: 1,
            result: Some(result),
            model,
            usage: cost.usage_summary(),
            outcome: TranscriptOutcome::Completed { result },
        }
    }

    pub fn failed(
        model: &'a str,
        cost: &CostTracker,
        message: &'a str,
        failure: Option<&'a FailureRecord>,
    ) -> Self {
        Self {
            schema_version: 1,
            result: None,
            model,
            usage: cost.usage_summary(),
            outcome: TranscriptOutcome::Error { message, failure },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct OneShotTranscript<'a> {
    pub schema_version: u8,
    pub model: &'a str,
    pub outcome: TranscriptOutcome<'a>,
    pub usage: UsageSummary,
    pub messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archive: Option<&'a [crate::session::ArchivedMessage]>,
    pub tool_trace: &'a [ToolTraceEntry],
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<&'a crate::tools::agent::SubAgentReport>,
    pub timing: ExecutionTiming,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TranscriptOutcome<'a> {
    Running,
    Completed {
        result: &'a str,
    },
    Error {
        message: &'a str,
        /// Classified failure, when the engine could determine one.
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<&'a FailureRecord>,
    },
}

impl<'a> OneShotTranscript<'a> {
    pub fn new(
        model: &'a str,
        cost: &CostTracker,
        messages: &'a [Message],
        tool_trace: &'a [ToolTraceEntry],
        timing: ExecutionTiming,
        outcome: TranscriptOutcome<'a>,
    ) -> Self {
        debug_assert!(!matches!(outcome, TranscriptOutcome::Running));
        Self {
            schema_version: 2,
            model,
            outcome,
            usage: cost.usage_summary(),
            messages,
            archive: None,
            tool_trace,
            sub_agents: tool_trace
                .iter()
                .filter_map(|entry| entry.sub_agent.as_deref())
                .collect(),
            timing,
        }
    }

    pub fn running(
        model: &'a str,
        cost: &CostTracker,
        messages: &'a [Message],
        tool_trace: &'a [ToolTraceEntry],
        timing: ExecutionTiming,
    ) -> Self {
        Self {
            schema_version: 2,
            model,
            outcome: TranscriptOutcome::Running,
            usage: cost.usage_summary(),
            messages,
            archive: None,
            tool_trace,
            sub_agents: tool_trace
                .iter()
                .filter_map(|entry| entry.sub_agent.as_deref())
                .collect(),
            timing,
        }
    }
    pub fn with_archive(mut self, archive: &'a [crate::session::ArchivedMessage]) -> Self {
        self.archive = Some(archive);
        self
    }
}

pub fn write_transcript(path: &Path, transcript: &OneShotTranscript<'_>) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("could not create transcript directory {}", parent.display())
        })?;
    }

    let partial_path = transcript_partial_path(path);
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&partial_path)
        .with_context(|| format!("could not create transcript {}", partial_path.display()))?;
    #[cfg(unix)]
    file.set_permissions({
        use std::os::unix::fs::PermissionsExt;
        std::fs::Permissions::from_mode(0o600)
    })
    .with_context(|| format!("could not secure transcript {}", path.display()))?;
    {
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, transcript)
            .with_context(|| format!("could not write transcript {}", path.display()))?;
        writer
            .flush()
            .with_context(|| format!("could not flush transcript {}", path.display()))?;
    }
    std::fs::rename(&partial_path, path).with_context(|| {
        format!(
            "could not publish transcript {} from {}",
            path.display(),
            partial_path.display()
        )
    })?;
    Ok(())
}

fn transcript_partial_path(path: &Path) -> PathBuf {
    let mut partial = path.as_os_str().to_os_string();
    partial.push(".partial");
    PathBuf::from(partial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::Usage;
    use crate::api::Message;
    use crate::query::{ModelRoundUsage, ModelTraceEntry, ToolTraceEntry};

    #[tokio::test]
    async fn contract_fixtures_from_a_real_engine_turn() {
        let mut engine = crate::test_support::scripted_engine(
            vec![],
            None,
            crate::permissions::PermissionMode::Default,
        );
        let result = engine
            .submit("hello", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();
        let output =
            serde_json::to_value(OneShotOutput::new(&result, engine.model(), &engine.cost))
                .unwrap();
        assert_eq!(
            output,
            serde_json::from_str::<serde_json::Value>(include_str!(
                "../docs/contracts/fixtures/result.json"
            ))
            .unwrap()
        );
        let mut timing = engine.execution_timing();
        timing.total_duration_ms = 0;
        for round in &mut timing.model_rounds {
            round.started_after_ms = 0;
            round.duration_ms = 0;
        }
        let transcript = OneShotTranscript::new(
            engine.model(),
            &engine.cost,
            engine.messages(),
            &[],
            timing,
            TranscriptOutcome::Completed { result: &result },
        );
        assert_eq!(
            serde_json::to_value(transcript).unwrap(),
            serde_json::from_str::<serde_json::Value>(include_str!(
                "../docs/contracts/fixtures/transcript.json"
            ))
            .unwrap()
        );
    }

    #[tokio::test]
    async fn usage_counts_include_failed_tools_and_reset() {
        let mut engine = crate::test_support::scripted_engine(
            vec![crate::test_support::tool_use(
                "call-1",
                "MissingTool",
                serde_json::json!({}),
            )],
            None,
            crate::permissions::PermissionMode::Bypass,
        );
        engine
            .submit("try a tool", tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();
        let usage = engine.cost.usage_summary();
        assert_eq!(usage.rounds, 2);
        assert_eq!(usage.tool_calls, 1);
        engine.cost.reset_usage();
        assert_eq!(engine.cost.usage_summary().rounds, 0);
        assert_eq!(engine.cost.usage_summary().tool_calls, 0);
    }

    #[test]
    fn serializes_stable_one_shot_contract() {
        let mut cost = CostTracker::new("unknown-model");
        cost.add_usage(&Usage {
            input_tokens: 12,
            output_tokens: 4,
            cache_read_tokens: 8,
            cache_creation_tokens: 2,
            provider_cost_usd: Some(0.00042),
        });

        let value = serde_json::to_value(OneShotOutput::new("done", "test/model", &cost)).unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "schema_version": 1,
                "result": "done",
                "model": "test/model",
                "usage": {
                    "rounds": 0,
                    "tool_calls": 0,
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "cache_read_tokens": 8,
                    "cache_creation_tokens": 2,
                    "cost_usd": 0.00042,
                    "cost_source": "provider"
                },
                // Additive: consumers pinned to schema 1 still find every
                // field they read.
                "outcome": { "status": "completed", "result": "done" }
            })
        );
    }

    #[test]
    fn writes_complete_tool_trace_without_changing_one_shot_contract() {
        let mut cost = CostTracker::new("test/model");
        cost.add_usage(&Usage {
            input_tokens: 3,
            output_tokens: 2,
            cache_read_tokens: 1,
            cache_creation_tokens: 0,
            provider_cost_usd: Some(0.0001),
        });
        let messages = vec![Message::user("diagnose the service")];
        let tool_trace = vec![ToolTraceEntry {
            sub_agent: None,
            id: "tool-1".to_string(),
            name: "Bash".to_string(),
            input: serde_json::json!({"command": "docker ps"}),
            output: "container-id\n".to_string(),
            is_error: false,
            read_only: true,
            started_after_ms: 120,
            duration_ms: 45,
        }];
        let transcript = OneShotTranscript::new(
            "test/model",
            &cost,
            &messages,
            &tool_trace,
            ExecutionTiming {
                total_duration_ms: 500,
                model_rounds: vec![ModelTraceEntry {
                    index: 1,
                    started_after_ms: 0,
                    duration_ms: 75,
                    failure: None,
                    status: "completed".to_string(),
                    usage: Some(ModelRoundUsage {
                        input_tokens: 3,
                        output_tokens: 2,
                        cache_read_tokens: 1,
                        cache_creation_tokens: 0,
                        cost_usd: Some(0.0001),
                    }),
                }],
            },
            TranscriptOutcome::Completed { result: "done" },
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/transcript.json");

        write_transcript(&path, &transcript).unwrap();

        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(value["schema_version"], 2);
        assert_eq!(value["outcome"]["status"], "completed");
        assert_eq!(value["outcome"]["result"], "done");
        assert_eq!(value["messages"][0]["content"], "diagnose the service");
        assert_eq!(value["tool_trace"][0]["input"]["command"], "docker ps");
        assert_eq!(value["tool_trace"][0]["output"], "container-id\n");
        assert_eq!(value["tool_trace"][0]["duration_ms"], 45);
        assert_eq!(value["timing"]["total_duration_ms"], 500);
        assert_eq!(value["timing"]["model_rounds"][0]["duration_ms"], 75);
        assert_eq!(
            value["timing"]["model_rounds"][0]["usage"]["input_tokens"],
            3
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(dir.path().join("nested/transcript.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn failed_one_shot_output_carries_outcome_and_null_result() {
        let cost = CostTracker::new("test/model");
        let failure = FailureRecord::cancelled(2);
        let value = serde_json::to_value(OneShotOutput::failed(
            "test/model",
            &cost,
            "Interrupted by shutdown signal.",
            Some(&failure),
        ))
        .unwrap();
        assert_eq!(value["schema_version"], 1);
        assert!(value["result"].is_null());
        assert_eq!(value["outcome"]["status"], "error");
        assert_eq!(value["outcome"]["failure"]["kind"], "cancelled");
        assert_eq!(value["outcome"]["failure"]["attempts"], 2);
        assert_eq!(value["outcome"]["failure"]["retryable"], false);
    }

    #[test]
    fn records_failed_outcome() {
        let cost = CostTracker::new("test/model");
        let failure = FailureRecord::unclassified();
        let transcript = OneShotTranscript::new(
            "test/model",
            &cost,
            &[],
            &[],
            ExecutionTiming {
                total_duration_ms: 0,
                model_rounds: vec![],
            },
            TranscriptOutcome::Error {
                message: "provider disconnected",
                failure: Some(&failure),
            },
        );

        let value = serde_json::to_value(transcript).unwrap();
        assert_eq!(value["outcome"]["status"], "error");
        assert_eq!(value["outcome"]["message"], "provider disconnected");
        assert_eq!(value["outcome"]["failure"]["kind"], "other");
    }

    #[test]
    fn atomically_replaces_running_checkpoint_with_final_outcome() {
        let cost = CostTracker::new("test/model");
        let messages = vec![Message::user("repair it")];
        let timing = ExecutionTiming {
            total_duration_ms: 25,
            model_rounds: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transcript.json");

        write_transcript(
            &path,
            &OneShotTranscript::running("test/model", &cost, &messages, &[], timing.clone()),
        )
        .unwrap();
        let running: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(running["outcome"]["status"], "running");

        write_transcript(
            &path,
            &OneShotTranscript::new(
                "test/model",
                &cost,
                &messages,
                &[],
                timing,
                TranscriptOutcome::Completed { result: "done" },
            ),
        )
        .unwrap();
        let completed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(completed["outcome"]["status"], "completed");
        assert!(!transcript_partial_path(&path).exists());
    }
}
