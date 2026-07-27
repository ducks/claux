use reqwest::{Response, StatusCode};
use serde_json::Value;

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

    anyhow::anyhow!(format_error(
        provider,
        model,
        Some(status),
        retry_after.as_deref(),
        error_type,
        message,
    ))
}

pub(super) fn stream_error(event: &Value, provider: &str, model: &str) -> String {
    let (error_type, message) = extract_details(event);
    let status = extract_status(event).or_else(|| error_type.and_then(status_for_type));
    format_error(provider, model, status, None, error_type, message)
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

        let message = stream_error(&event, "openrouter", "model");

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

        let message = stream_error(&event, "openrouter", "model");

        assert!(message.contains("429 Too Many Requests"));
        assert!(message.contains("rate_limit_exceeded"));
    }
}
