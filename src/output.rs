use crate::api::Message;
use crate::cost::{CostTracker, UsageSummary};
use crate::query::{ExecutionTiming, ToolTraceEntry};
use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct OneShotOutput<'a> {
    pub schema_version: u8,
    pub result: &'a str,
    pub model: &'a str,
    pub usage: UsageSummary,
}

impl<'a> OneShotOutput<'a> {
    pub fn new(result: &'a str, model: &'a str, cost: &CostTracker) -> Self {
        Self {
            schema_version: 1,
            result,
            model,
            usage: cost.usage_summary(),
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
    pub tool_trace: &'a [ToolTraceEntry],
    pub timing: ExecutionTiming,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TranscriptOutcome<'a> {
    Completed { result: &'a str },
    Error { message: &'a str },
}

impl<'a> OneShotTranscript<'a> {
    pub fn new(
        model: &'a str,
        cost: &CostTracker,
        messages: &'a [Message],
        tool_trace: &'a [ToolTraceEntry],
        timing: ExecutionTiming,
        result: Option<&'a str>,
        error: Option<&'a str>,
    ) -> Self {
        debug_assert!(result.is_some() ^ error.is_some());
        Self {
            schema_version: 2,
            model,
            outcome: match error {
                Some(message) => TranscriptOutcome::Error { message },
                None => TranscriptOutcome::Completed {
                    result: result.unwrap_or_default(),
                },
            },
            usage: cost.usage_summary(),
            messages,
            tool_trace,
            timing,
        }
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

    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(path)
        .with_context(|| format!("could not create transcript {}", path.display()))?;
    #[cfg(unix)]
    file.set_permissions({
        use std::os::unix::fs::PermissionsExt;
        std::fs::Permissions::from_mode(0o600)
    })
    .with_context(|| format!("could not secure transcript {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, transcript)
        .with_context(|| format!("could not write transcript {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::Usage;
    use crate::api::Message;
    use crate::query::{ModelRoundUsage, ModelTraceEntry, ToolTraceEntry};

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
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "cache_read_tokens": 8,
                    "cache_creation_tokens": 2,
                    "cost_usd": 0.00042
                }
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
            Some("done"),
            None,
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
    fn records_failed_outcome() {
        let cost = CostTracker::new("test/model");
        let transcript = OneShotTranscript::new(
            "test/model",
            &cost,
            &[],
            &[],
            ExecutionTiming {
                total_duration_ms: 0,
                model_rounds: vec![],
            },
            None,
            Some("provider disconnected"),
        );

        let value = serde_json::to_value(transcript).unwrap();
        assert_eq!(value["outcome"]["status"], "error");
        assert_eq!(value["outcome"]["message"], "provider disconnected");
    }
}
