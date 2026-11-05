//! Typed, turn-scoped worker handoff. Verification remains coordinator-owned.
use anyhow::{ensure, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum State {
    Waiting,
    NeedsInput,
    ReadyForReview,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Report {
    pub state: State,
    pub summary: String,
    #[serde(default)]
    pub question: Option<String>,
}

impl Report {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.summary.trim().is_empty() && self.summary.len() <= 8000,
            "summary must contain 1–8000 bytes"
        );
        ensure!(
            self.question.as_ref().is_none_or(|q| q.len() <= 2000),
            "question must not exceed 2000 bytes"
        );
        ensure!(
            self.state != State::NeedsInput
                || self.question.as_ref().is_some_and(|q| !q.trim().is_empty()),
            "needs_input requires a nonempty question"
        );
        Ok(())
    }

    pub(crate) fn handoff(&self) -> Value {
        json!({"state":self.state,"summary":self.summary,"question":self.question,"advisory":true})
    }
}

#[derive(Default)]
pub(crate) struct ReportSlot {
    report: Option<Report>,
    standalone: bool,
}

pub(crate) type SharedReport = Arc<Mutex<ReportSlot>>;

pub(crate) fn prepare(slot: &SharedReport, standalone: bool) {
    let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
    slot.report = None;
    slot.standalone = standalone;
}

pub(crate) fn snapshot(slot: &SharedReport) -> Option<Report> {
    slot.lock()
        .unwrap_or_else(|e| e.into_inner())
        .report
        .clone()
}

pub(crate) struct TaskReportTool(pub SharedReport);

#[async_trait]
impl Tool for TaskReportTool {
    fn name(&self) -> &str {
        "TaskReport"
    }
    fn description(&self) -> &str {
        "Finish this managed worker turn with a structured handoff. Call alone after all other tools finish. waiting ends this turn; needs_input requires a concrete question; ready_for_review requests review; failed reports a blocker. A successful call ends the turn. This is an advisory model report, not proof of verification, permission, or authority to publish."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{
            "state":{"type":"string","enum":["waiting","needs_input","ready_for_review","failed"]},
            "summary":{"type":"string","minLength":1,"maxLength":8000},
            "question":{"type":["string","null"],"maxLength":2000}
        },"required":["state","summary"],"additionalProperties":false})
    }
    fn is_read_only(&self) -> bool {
        true
    }
    fn summarize(&self, input: &Value) -> String {
        format!(
            "Worker handoff: {}",
            input["state"].as_str().unwrap_or("invalid")
        )
    }
    fn reset_session(&self) {
        prepare(&self.0, false);
    }
    async fn execute(&self, input: Value, cancel: CancellationToken) -> Result<ToolOutput> {
        if cancel.is_cancelled() {
            return Ok(super::interrupted_output());
        }
        let report: Report = serde_json::from_value(input)?;
        report.validate()?;
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        ensure!(
            slot.standalone,
            "TaskReport must be the only tool call in its batch; finish other work first"
        );
        if cancel.is_cancelled() {
            return Ok(super::interrupted_output());
        }
        slot.report = Some(report);
        Ok(ToolOutput { content: "Structured handoff recorded; this worker turn is ending. Verification and authorization remain coordinator-owned.".into(),is_error:false,sub_agent:None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn validates_fields_and_refuses_forged_verification() {
        let slot = Arc::new(Mutex::new(ReportSlot::default()));
        let tool = TaskReportTool(slot.clone());
        for input in [
            json!({"state":"success","summary":"done"}),
            json!({"state":"waiting","summary":" "}),
            json!({"state":"needs_input","summary":"choice"}),
            json!({"state":"needs_input","summary":"choice","question":" "}),
            json!({"state":"ready_for_review","summary":"done","acceptance_checks_passed":true}),
            json!({"state":"waiting","summary":"x".repeat(8001)}),
            json!({"state":"needs_input","summary":"choice","question":"x".repeat(2001)}),
        ] {
            prepare(&slot, true);
            assert!(tool.execute(input, CancellationToken::new()).await.is_err());
            assert!(snapshot(&slot).is_none());
        }
        prepare(&slot, true);
        tool.execute(
            json!({"state":"needs_input","summary":"Need a target","question":"Which target?"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let value = snapshot(&slot).unwrap().handoff();
        assert_eq!(value["state"], "needs_input");
        assert_eq!(value["advisory"], true);
        assert!(value.get("acceptance_checks_passed").is_none());
    }

    #[tokio::test]
    async fn cancelled_mixed_and_reset_reports_are_not_retained() {
        let slot = Arc::new(Mutex::new(ReportSlot::default()));
        let tool = TaskReportTool(slot.clone());
        let input = json!({"state":"waiting","summary":"done"});
        prepare(&slot, false);
        assert!(tool
            .execute(input.clone(), CancellationToken::new())
            .await
            .is_err());
        prepare(&slot, true);
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(tool.execute(input.clone(), cancel).await.unwrap().is_error);
        assert!(snapshot(&slot).is_none());
        tool.execute(input, CancellationToken::new()).await.unwrap();
        tool.reset_session();
        assert!(snapshot(&slot).is_none());
    }
}
