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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiFailureKind {
    /// The request exceeded the model's input context window.
    ContextExceeded,
    /// The response hit the output token limit.
    OutputLimitExceeded,
    /// The model emitted tool-call arguments that are not valid JSON.
    MalformedToolArguments,
    /// The provider throttled the request (HTTP 429). Retryable.
    RateLimited,
    /// The provider or its upstream is temporarily unable to serve the
    /// request (HTTP 5xx, overloaded). Retryable.
    Unavailable,
    /// The credential was rejected or the account cannot be billed.
    Authentication,
    /// The model or endpoint does not exist for this provider.
    ModelNotFound,
    /// The provider refused the content on policy grounds.
    PolicyRejection,
    /// The response violated the protocol claux expected: truncated
    /// stream, unparseable frames, an empty completion.
    ProtocolError,
    /// The request never completed at the transport level: connection
    /// refused or reset, DNS failure, timeout. Retryable.
    Network,
    /// The request was cancelled by the user or a shutdown signal.
    Cancelled,
    /// Anything not separately actionable.
    Other,
}

impl ApiFailureKind {
    /// Whether reissuing the same request may succeed without any change.
    pub fn retryable(self) -> bool {
        matches!(self, Self::RateLimited | Self::Unavailable | Self::Network)
    }

    /// The stable snake_case name used in JSON outputs and by consumers.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ContextExceeded => "context_exceeded",
            Self::OutputLimitExceeded => "output_limit_exceeded",
            Self::MalformedToolArguments => "malformed_tool_arguments",
            Self::RateLimited => "rate_limited",
            Self::Unavailable => "unavailable",
            Self::Authentication => "authentication",
            Self::ModelNotFound => "model_not_found",
            Self::PolicyRejection => "policy_rejection",
            Self::ProtocolError => "protocol_error",
            Self::Network => "network",
            Self::Cancelled => "cancelled",
            Self::Other => "other",
        }
    }

    /// Process exit code for a one-shot run that ended with this failure.
    /// 1 stays the generic failure; 2 is reserved for usage errors.
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Cancelled => 10,
            Self::RateLimited | Self::Unavailable => 11,
            Self::Authentication => 12,
            Self::ContextExceeded => 13,
            Self::PolicyRejection => 14,
            Self::ProtocolError | Self::MalformedToolArguments => 15,
            Self::ModelNotFound => 16,
            Self::Network => 17,
            Self::OutputLimitExceeded => 18,
            Self::Other => 1,
        }
    }
}

/// A provider failure: a classification the turn loop matches on, plus the
/// human-readable message shown to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiFailure {
    pub kind: ApiFailureKind,
    pub message: String,
    /// HTTP status of the failing response, when there was one.
    pub http_status: Option<u16>,
    /// Provider-requested wait before retrying, when it sent one.
    pub retry_after: Option<std::time::Duration>,
}

impl ApiFailure {
    pub fn new(kind: ApiFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            http_status: None,
            retry_after: None,
        }
    }

    pub fn with_status(mut self, status: Option<StatusCode>) -> Self {
        self.http_status = status.map(|status| status.as_u16());
        self
    }

    pub fn with_retry_after(mut self, retry_after: Option<std::time::Duration>) -> Self {
        self.retry_after = retry_after;
        self
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ApiFailureKind::Cancelled, message)
    }

    pub fn protocol_error(message: impl Into<String>) -> Self {
        Self::new(ApiFailureKind::ProtocolError, message)
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
    if let Some(failure) = error.downcast_ref::<ApiFailure>() {
        return failure.clone();
    }

    // A body read that fails at the transport layer surfaces as reqwest's
    // error type, not as provider prose.
    if let Some(transport) = error.downcast_ref::<reqwest::Error>() {
        return ApiFailure::new(
            ApiFailureKind::Network,
            format!("stream transport failed: {transport}"),
        );
    }

    // The remaining markers are all emitted by this crate's own SSE
    // readers: `invalid arguments for [Anthropic] tool call`, `stream
    // ended ...`, `invalid JSON in ... SSE event`.
    let text = error.to_string();
    let kind = if text.contains("invalid arguments for") && text.contains("tool call") {
        ApiFailureKind::MalformedToolArguments
    } else if text.contains("stream ended") || text.contains("SSE event") {
        ApiFailureKind::ProtocolError
    } else {
        ApiFailureKind::Other
    };
    ApiFailure::new(kind, text)
}

/// A request that failed before a response arrived.
pub(super) fn transport_error(error: reqwest::Error, provider: &str, model: &str) -> anyhow::Error {
    let what = if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "could not connect"
    } else {
        "failed"
    };
    anyhow::Error::new(ApiFailure::new(
        ApiFailureKind::Network,
        format!("{provider} API request {what} for model '{model}': {error}"),
    ))
}

/// The bounded wait for response headers elapsed.
pub(super) fn headers_timeout_error(provider: &str, model: &str) -> anyhow::Error {
    anyhow::Error::new(ApiFailure::new(
        ApiFailureKind::Network,
        format!(
            "{provider} API request timed out waiting for response headers for model '{model}'"
        ),
    ))
}

/// The request was cancelled before a response arrived.
pub(super) fn cancelled_error() -> anyhow::Error {
    anyhow::Error::new(ApiFailure::cancelled("API request cancelled"))
}

/// Classify an HTTP status plus provider-declared error type.
fn classify(status: Option<StatusCode>, error_type: Option<&str>) -> ApiFailureKind {
    if status == Some(StatusCode::PAYLOAD_TOO_LARGE) {
        return ApiFailureKind::ContextExceeded;
    }
    // Provider-declared types are more specific than the status and are
    // stable identifiers on the providers claux targets.
    match error_type {
        Some("context_length_exceeded") | Some("string_above_max_length") => {
            return ApiFailureKind::ContextExceeded
        }
        Some("max_output_tokens") | Some("max_tokens_exceeded") => {
            return ApiFailureKind::OutputLimitExceeded
        }
        Some("rate_limit_exceeded") | Some("rate_limit_error") | Some("insufficient_quota") => {
            return ApiFailureKind::RateLimited
        }
        Some("overloaded_error")
        | Some("provider_overloaded")
        | Some("provider_unavailable")
        | Some("api_error")
        | Some("server_error") => return ApiFailureKind::Unavailable,
        Some("authentication")
        | Some("authentication_error")
        | Some("invalid_api_key")
        | Some("permission_denied")
        | Some("permission_error")
        | Some("payment_required") => return ApiFailureKind::Authentication,
        Some("model_not_found") | Some("provider_model_not_found") | Some("not_found_error") => {
            return ApiFailureKind::ModelNotFound
        }
        Some("content_policy_violation")
        | Some("content_filter")
        | Some("prohibited_content")
        | Some("safety") => return ApiFailureKind::PolicyRejection,
        Some("timeout") => return ApiFailureKind::Network,
        _ => {}
    }
    match status {
        Some(StatusCode::TOO_MANY_REQUESTS) => ApiFailureKind::RateLimited,
        Some(StatusCode::UNAUTHORIZED)
        | Some(StatusCode::FORBIDDEN)
        | Some(StatusCode::PAYMENT_REQUIRED) => ApiFailureKind::Authentication,
        Some(StatusCode::NOT_FOUND) => ApiFailureKind::ModelNotFound,
        Some(StatusCode::REQUEST_TIMEOUT) | Some(StatusCode::GATEWAY_TIMEOUT) => {
            ApiFailureKind::Network
        }
        Some(status) if status.is_server_error() => ApiFailureKind::Unavailable,
        // Anthropic's overloaded status is not a registered code.
        Some(status) if status.as_u16() == 529 => ApiFailureKind::Unavailable,
        _ => ApiFailureKind::Other,
    }
}

/// Parse a Retry-After header: seconds, or an HTTP date in the future.
fn parse_retry_after(value: &str) -> Option<std::time::Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(std::time::Duration::from_secs(seconds));
    }
    let when = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    let delta = when.signed_duration_since(chrono::Utc::now());
    delta.to_std().ok()
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
        classify_details(provider, Some(status), error_type, message),
        format_error(
            provider,
            model,
            Some(status),
            retry_after.as_deref(),
            error_type,
            message,
        ),
    )
    .with_status(Some(status))
    .with_retry_after(retry_after.as_deref().and_then(parse_retry_after));
    anyhow::Error::new(failure)
}

pub(super) fn stream_error(event: &Value, provider: &str, model: &str) -> ApiFailure {
    let (error_type, message) = extract_details(event);
    let status = extract_status(event).or_else(|| error_type.and_then(status_for_type));
    ApiFailure::new(
        classify_details(provider, status, error_type, message),
        format_error(provider, model, status, None, error_type, message),
    )
    .with_status(status)
}

fn classify_details(
    provider: &str,
    status: Option<StatusCode>,
    error_type: Option<&str>,
    message: Option<&str>,
) -> ApiFailureKind {
    // Anthropic has no distinct context-overflow code. Restrict its message
    // fallback to that protocol's invalid-request envelope and exact prefix.
    if provider == "anthropic"
        && matches!(status, None | Some(StatusCode::BAD_REQUEST))
        && error_type == Some("invalid_request_error")
        && message.is_some_and(|text| {
            text == "prompt is too long" || text.starts_with("prompt is too long:")
        })
    {
        return ApiFailureKind::ContextExceeded;
    }
    classify(status, error_type)
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
        .or_else(|| {
            error["code"]
                .as_str()
                .filter(|code| classify(None, Some(code)) != ApiFailureKind::Other)
        })
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
    fn classifies_transient_auth_and_policy_failures() {
        use ApiFailureKind as K;
        let cases = [
            (Some(429), None, K::RateLimited),
            (Some(401), None, K::Authentication),
            (Some(403), None, K::Authentication),
            (Some(402), None, K::Authentication),
            (Some(404), None, K::ModelNotFound),
            (Some(500), None, K::Unavailable),
            (Some(502), None, K::Unavailable),
            (Some(503), None, K::Unavailable),
            (Some(504), None, K::Network),
            (Some(529), None, K::Unavailable),
            (Some(400), None, K::Other),
            (Some(400), Some("rate_limit_error"), K::RateLimited),
            (Some(500), Some("overloaded_error"), K::Unavailable),
            (
                Some(400),
                Some("content_policy_violation"),
                K::PolicyRejection,
            ),
            (Some(400), Some("model_not_found"), K::ModelNotFound),
            (
                Some(400),
                Some("context_length_exceeded"),
                K::ContextExceeded,
            ),
            (None, Some("provider_unavailable"), K::Unavailable),
        ];
        for (status, error_type, expected) in cases {
            let status = status.and_then(|code| StatusCode::from_u16(code).ok());
            assert_eq!(
                classify(status, error_type),
                expected,
                "status={status:?} type={error_type:?}"
            );
        }
    }

    #[test]
    fn only_transient_kinds_are_retryable() {
        use ApiFailureKind as K;
        for kind in [K::RateLimited, K::Unavailable, K::Network] {
            assert!(kind.retryable(), "{kind:?}");
        }
        for kind in [
            K::ContextExceeded,
            K::OutputLimitExceeded,
            K::MalformedToolArguments,
            K::Authentication,
            K::ModelNotFound,
            K::PolicyRejection,
            K::ProtocolError,
            K::Cancelled,
            K::Other,
        ] {
            assert!(!kind.retryable(), "{kind:?}");
        }
    }

    #[test]
    fn exit_codes_are_distinct_and_avoid_ssh_and_shell_codes() {
        use ApiFailureKind as K;
        let kinds = [
            K::Cancelled,
            K::RateLimited,
            K::Authentication,
            K::ContextExceeded,
            K::PolicyRejection,
            K::ProtocolError,
            K::ModelNotFound,
            K::Network,
            K::OutputLimitExceeded,
        ];
        let codes: std::collections::HashSet<u8> = kinds.iter().map(|k| k.exit_code()).collect();
        assert_eq!(
            codes.len(),
            kinds.len(),
            "each listed kind has its own code"
        );
        for code in codes {
            assert!((10..=18).contains(&code), "{code}");
        }
        assert_eq!(K::Other.exit_code(), 1);
        assert_eq!(K::Unavailable.exit_code(), K::RateLimited.exit_code());
    }

    #[test]
    fn retry_after_parses_seconds_and_http_dates() {
        assert_eq!(
            parse_retry_after("30"),
            Some(std::time::Duration::from_secs(30))
        );
        let future = (chrono::Utc::now() + chrono::Duration::seconds(90)).to_rfc2822();
        let parsed = parse_retry_after(&future).unwrap();
        assert!(
            parsed.as_secs() >= 85 && parsed.as_secs() <= 90,
            "{parsed:?}"
        );
        assert_eq!(parse_retry_after("soon"), None);
    }

    #[test]
    fn reader_errors_classify_transport_and_protocol_markers() {
        let protocol = classify_reader_error(&anyhow::anyhow!("stream ended before message_stop"));
        assert_eq!(protocol.kind, ApiFailureKind::ProtocolError);
        let json = classify_reader_error(&anyhow::anyhow!("invalid JSON in OpenAI SSE event: x"));
        assert_eq!(json.kind, ApiFailureKind::ProtocolError);
        let other = classify_reader_error(&anyhow::anyhow!("something else"));
        assert_eq!(other.kind, ApiFailureKind::Other);
    }

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
        let kind = classify(
            Some(StatusCode::TOO_MANY_REQUESTS),
            Some("rate_limit_exceeded"),
        );
        assert_eq!(kind, ApiFailureKind::RateLimited);
        assert!(!matches!(
            kind,
            ApiFailureKind::ContextExceeded | ApiFailureKind::OutputLimitExceeded
        ));
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
        assert_eq!(failure.kind, ApiFailureKind::Unavailable);
        assert_eq!(failure.http_status, Some(500));
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
        let error = anyhow::anyhow!("the provider said 429 things about 413 tokens");

        assert_eq!(classify_reader_error(&error).kind, ApiFailureKind::Other);
    }

    #[tokio::test]
    async fn http_errors_carry_their_classification_through_anyhow() {
        // The turn loop downcasts the boxed error, so the classification must
        // survive `anyhow::Error::new`.
        let response = crate::test_support::json_response(
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}"#,
        )
        .await;

        let error = http_error(response, "anthropic", "claude-test").await;
        let failure = error.downcast_ref::<ApiFailure>().expect("typed failure");

        assert_eq!(failure.kind, ApiFailureKind::ContextExceeded);
        assert!(failure.message.contains("prompt is too long"));
    }

    #[tokio::test]
    async fn context_overflow_envelopes_preserve_specific_codes() {
        for (provider, body) in [
            (
                "openai",
                serde_json::json!({"error": {"type": "invalid_request_error", "code": "context_length_exceeded", "message": "too many tokens"}}),
            ),
            (
                "openrouter",
                serde_json::json!({"error": {"code": 400, "metadata": {"error_type": "context_length_exceeded"}, "message": "too many tokens"}}),
            ),
        ] {
            let response = crate::test_support::json_response(400, &body.to_string()).await;
            let error = http_error(response, provider, "test").await;
            assert_eq!(
                error.downcast_ref::<ApiFailure>().unwrap().kind,
                ApiFailureKind::ContextExceeded
            );
            assert_eq!(
                stream_error(&body, provider, "test").kind,
                ApiFailureKind::ContextExceeded
            );
        }
    }

    #[test]
    fn anthropic_context_fallback_does_not_match_unrelated_errors() {
        for (provider, status, kind, message) in [
            ("openai", 400, "invalid_request_error", "prompt is too long"),
            (
                "anthropic",
                400,
                "invalid_request_error",
                "invalid field: prompt is too long",
            ),
            ("anthropic", 400, "other_error", "prompt is too long"),
            (
                "anthropic",
                401,
                "invalid_request_error",
                "prompt is too long",
            ),
        ] {
            assert_ne!(
                classify_details(
                    provider,
                    StatusCode::from_u16(status).ok(),
                    Some(kind),
                    Some(message)
                ),
                ApiFailureKind::ContextExceeded
            );
        }
        let body = serde_json::json!({"error": {"type":"authentication_error", "code":"unknown_new_code"}});
        assert_eq!(
            stream_error(&body, "anthropic", "test").kind,
            ApiFailureKind::Authentication
        );
    }
}
