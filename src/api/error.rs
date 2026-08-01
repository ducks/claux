use reqwest::{Response, StatusCode};
use serde_json::Value;
use std::fmt;

/// What went wrong with a provider request, classified at the point where the
/// structured response is still available.
///
/// The turn loop drives recovery off this rather than off error prose:
/// substring matching against provider messages misfires (any error whose text
/// happens to contain "413" would trigger a compaction that discards history)
/// and depends on wording that OpenAI-compatible endpoints do not treat as
/// contractual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiFailureKind {
    /// The request exceeded the model's input context window.
    ContextExceeded,
    /// The response hit the output token limit.
    OutputLimitExceeded,
    /// The model emitted tool-call arguments that are not valid JSON.
    MalformedToolArguments,
    /// Anything not separately actionable.
    Other,
}

/// A provider failure: a classification the turn loop matches on, plus the
/// human-readable message shown to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiFailure {
    pub kind: ApiFailureKind,
    pub message: String,
}

impl ApiFailure {
    pub fn new(kind: ApiFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// An unclassified failure. Prefer a specific kind when one is known.
    pub fn other(message: impl Into<String>) -> Self {
        Self::new(ApiFailureKind::Other, message)
    }

    pub fn output_limit_exceeded(message: impl Into<String>) -> Self {
        Self::new(ApiFailureKind::OutputLimitExceeded, message)
    }

    pub fn malformed_tool_arguments(message: impl Into<String>) -> Self {
        Self::new(ApiFailureKind::MalformedToolArguments, message)
    }

    /// Prefix the message while preserving the classification.
    ///
    /// Providers wrap reader failures for display ("SSE stream error: ...");
    /// the wrapping must not erase what the failure was.
    pub fn prefixed(mut self, prefix: &str) -> Self {
        self.message = format!("{prefix}: {}", self.message);
        self
    }
}

impl fmt::Display for ApiFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ApiFailure {}

/// Classify a transport/protocol error raised while reading a stream.
///
/// Reader errors arrive as `anyhow::Error` from parsing code that already
/// names the condition precisely, so this recovers the classification from
/// the few markers the crate itself produces — not from provider prose.
pub(super) fn classify_reader_error(error: &anyhow::Error) -> ApiFailure {
    // `invalid arguments for tool call` / `invalid arguments for Anthropic
    // tool call` are emitted by this crate's own SSE parsers.
    let text = error.to_string();
    let kind = if text.contains("invalid arguments for") && text.contains("tool call") {
        ApiFailureKind::MalformedToolArguments
    } else {
        ApiFailureKind::Other
    };
    ApiFailure::new(kind, text)
}

/// Classify an HTTP status plus provider-declared error type.
fn classify(status: Option<StatusCode>, error_type: Option<&str>) -> ApiFailureKind {
    if status == Some(StatusCode::PAYLOAD_TOO_LARGE) {
        return ApiFailureKind::ContextExceeded;
    }
    match error_type {
        Some("context_length_exceeded") | Some("string_above_max_length") => {
            ApiFailureKind::ContextExceeded
        }
        Some("max_output_tokens") | Some("max_tokens_exceeded") => {
            ApiFailureKind::OutputLimitExceeded
        }
        _ => ApiFailureKind::Other,
    }
}

pub(super) async fn http_error(response: Response, provider: &str, model: &str) -> anyhow::Error {
    let status = response.status();
    let retry_after = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let body = response.text().await.unwrap_or_default();
    let value = serde_json::from_str::<Value>(&body).ok();
    let (error_type, message) = value
        .as_ref()
        .map(extract_details)
        .unwrap_or((None, nonempty(&body)));
    let message = message.map(|message| crate::utils::truncate_str(message, 2_048));

    let failure = ApiFailure::new(
        classify(Some(status), error_type),
        format_error(
            provider,
            model,
            Some(status),
            retry_after.as_deref(),
            error_type,
            message,
        ),
    );
    anyhow::Error::new(failure)
}

pub(super) fn stream_error(event: &Value, provider: &str, model: &str) -> ApiFailure {
    let (error_type, message) = extract_details(event);
    let status = extract_status(event).or_else(|| error_type.and_then(status_for_type));
    ApiFailure::new(
        classify(status, error_type),
        format_error(provider, model, status, None, error_type, message),
    )
}

fn extract_details(value: &Value) -> (Option<&str>, Option<&str>) {
    let response = &value["response"];
    let error = if value["error"].is_object() {
        &value["error"]
    } else if response["error"].is_object() {
        &response["error"]
    } else {
        value
    };
    let error_type = error["metadata"]["error_type"]
        .as_str()
        .or_else(|| error["error_type"].as_str())
        .or_else(|| value["error_type"].as_str())
        .or_else(|| response["error_type"].as_str())
        .or_else(|| error["type"].as_str())
        .or_else(|| error["code"].as_str());
    let message = error["message"]
        .as_str()
        .or_else(|| value["message"].as_str())
        .or_else(|| response["incomplete_details"]["reason"].as_str());
    (error_type, message)
}

fn extract_status(value: &Value) -> Option<StatusCode> {
    let response = &value["response"];
    let error = if value["error"].is_object() {
        &value["error"]
    } else if response["error"].is_object() {
        &response["error"]
    } else {
        value
    };
    error["code"]
        .as_u64()
        .or_else(|| value["code"].as_u64())
        .and_then(|code| u16::try_from(code).ok())
        .and_then(|code| StatusCode::from_u16(code).ok())
}

fn status_for_type(error_type: &str) -> Option<StatusCode> {
    match error_type {
        "authentication" => Some(StatusCode::UNAUTHORIZED),
        "payment_required" => Some(StatusCode::PAYMENT_REQUIRED),
        "permission_denied" => Some(StatusCode::FORBIDDEN),
        "rate_limit_exceeded" => Some(StatusCode::TOO_MANY_REQUESTS),
        "provider_unavailable" => Some(StatusCode::BAD_GATEWAY),
        "provider_overloaded" => Some(StatusCode::SERVICE_UNAVAILABLE),
        "timeout" => Some(StatusCode::GATEWAY_TIMEOUT),
        _ => None,
    }
}

fn format_error(
    provider: &str,
    model: &str,
    status: Option<StatusCode>,
    retry_after: Option<&str>,
    error_type: Option<&str>,
    message: Option<&str>,
) -> String {
    let mut output = format!("{provider} API error");
    if let Some(status) = status {
        output.push_str(&format!(" ({status})"));
    }
    output.push_str(&format!(" for model '{model}'"));
    if let Some(error_type) = error_type {
        output.push_str(&format!(" [{error_type}]"));
    }
    if let Some(message) = message {
        output.push_str(": ");
        output.push_str(message);
    }

    match status {
        Some(StatusCode::TOO_MANY_REQUESTS) => {
            if let Some(retry_after) = retry_after {
                output.push_str(&format!(". Retry after {retry_after}"));
            } else {
                output.push_str(". Retry later or choose another model/provider");
            }
        }
        Some(StatusCode::SERVICE_UNAVAILABLE) | Some(StatusCode::BAD_GATEWAY) => {
            output.push_str(". The provider may be unavailable; retry or choose another model")
        }
        _ => {}
    }
    output
}

fn nonempty(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_rate_limit_with_retry_after() {
        let message = format_error(
            "openrouter",
            "poolside/laguna",
            Some(StatusCode::TOO_MANY_REQUESTS),
            Some("60"),
            Some("rate_limit_exceeded"),
            Some("Rate limit exceeded"),
        );

        assert!(message.contains("429 Too Many Requests"));
        assert!(message.contains("poolside/laguna"));
        assert!(message.contains("rate_limit_exceeded"));
        assert!(message.contains("Retry after 60"));
    }

    #[test]
    fn extracts_openrouter_stream_error() {
        let event = serde_json::json!({
            "error": {
                "code": 429,
                "message": "upstream limit",
                "metadata": {"error_type": "rate_limit_exceeded"}
            }
        });

        let message = stream_error(&event, "openrouter", "model").message;

        assert!(message.contains("429 Too Many Requests"));
        assert!(message.contains("upstream limit"));
        assert!(message.contains("choose another model/provider"));
    }

    #[test]
    fn extracts_responses_error_type() {
        let event = serde_json::json!({
            "type": "response.failed",
            "response": {
                "error": {"code": "server_error", "message": "Rate limited"},
                "error_type": "rate_limit_exceeded"
            }
        });

        let message = stream_error(&event, "openrouter", "model").message;

        assert!(message.contains("429 Too Many Requests"));
        assert!(message.contains("rate_limit_exceeded"));
    }

    #[test]
    fn classifies_context_and_output_limits_from_structure() {
        assert_eq!(
            classify(Some(StatusCode::PAYLOAD_TOO_LARGE), None),
            ApiFailureKind::ContextExceeded
        );
        assert_eq!(
            classify(None, Some("context_length_exceeded")),
            ApiFailureKind::ContextExceeded
        );
        assert_eq!(
            classify(None, Some("max_tokens_exceeded")),
            ApiFailureKind::OutputLimitExceeded
        );
    }

    #[test]
    fn rate_limits_are_not_mistaken_for_context_or_output_limits() {
        assert_eq!(
            classify(
                Some(StatusCode::TOO_MANY_REQUESTS),
                Some("rate_limit_exceeded")
            ),
            ApiFailureKind::Other
        );
    }

    #[test]
    fn incidental_digits_in_a_message_do_not_trigger_compaction() {
        // The substring predicate this replaced matched "413" anywhere in the
        // error text, so a request id or a token count could trigger a
        // conversation-destroying compaction. Classification now comes from
        // the status code, so prose is inert.
        let event = serde_json::json!({
            "error": {
                "code": 500,
                "message": "internal error (request req_413_88 on model gpt-4-1300)"
            }
        });

        let failure = stream_error(&event, "openai", "model");

        assert!(failure.message.contains("req_413_88"));
        assert_eq!(failure.kind, ApiFailureKind::Other);
    }

    #[test]
    fn reader_errors_recover_the_malformed_tool_argument_classification() {
        // The SSE parsers name this condition precisely; the classification
        // must survive being wrapped for display.
        let error = anyhow::anyhow!(
            "invalid arguments for tool call Read (call_3): EOF while parsing a value"
        );

        let failure = classify_reader_error(&error).prefixed("OpenAI SSE stream error");

        assert_eq!(failure.kind, ApiFailureKind::MalformedToolArguments);
        assert!(failure.message.starts_with("OpenAI SSE stream error: "));
        assert!(failure.message.contains("invalid arguments for tool call"));
    }

    #[test]
    fn unrelated_reader_errors_stay_unclassified() {
        let error = anyhow::anyhow!("stream ended before message_stop");

        assert_eq!(classify_reader_error(&error).kind, ApiFailureKind::Other);
    }

    #[tokio::test]
    async fn http_errors_carry_their_classification_through_anyhow() {
        // The turn loop downcasts the boxed error, so the classification must
        // survive `anyhow::Error::new`.
        let response = crate::test_support::json_response(
            413,
            "{\"error\":{\"message\":\"prompt is too long\"}}",
        )
        .await;

        let error = http_error(response, "anthropic", "claude-test").await;
        let failure = error.downcast_ref::<ApiFailure>().expect("typed failure");

        assert_eq!(failure.kind, ApiFailureKind::ContextExceeded);
        assert!(failure.message.contains("prompt is too long"));
    }
}
