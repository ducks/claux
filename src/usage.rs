//! Provider account and key usage status.
//!
//! Usage limits are provider-specific. This module only reports values from a
//! documented provider endpoint and never treats an unavailable value as zero.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const OPENROUTER_KEY_URL: &str = "https://openrouter.ai/api/v1/key";
const OPENCODE_GO_USAGE_URL: &str = "https://opencode.ai/zen/go/v1/usage";

#[derive(Debug, Serialize)]
struct UsageReport {
    schema_version: u8,
    provider: String,
    status: UsageStatus,
    source: Option<String>,
    details: Option<UsageDetails>,
    message: Option<String>,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum UsageStatus {
    Available,
    NotConfigured,
    Unavailable,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct OpenRouterKeyStatus {
    label: Option<String>,
    disabled: Option<bool>,
    is_free_tier: Option<bool>,
    limit: Option<f64>,
    limit_remaining: Option<f64>,
    limit_reset: Option<String>,
    usage: Option<f64>,
    usage_daily: Option<f64>,
    usage_weekly: Option<f64>,
    usage_monthly: Option<f64>,
    include_byok_in_limit: Option<bool>,
    byok_usage: Option<f64>,
    byok_usage_daily: Option<f64>,
    byok_usage_weekly: Option<f64>,
    byok_usage_monthly: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterKeyResponse {
    data: OpenRouterKeyStatus,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum UsageDetails {
    OpenRouter(OpenRouterKeyStatus),
    OpenCodeGo(OpenCodeGoUsage),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct OpenCodeGoUsage {
    rolling: OpenCodeGoWindow,
    weekly: OpenCodeGoWindow,
    monthly: OpenCodeGoWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct OpenCodeGoWindow {
    status: String,
    percent: f64,
    #[serde(rename = "resetsAt")]
    resets_at: String,
}

#[derive(Debug, Deserialize)]
struct OpenCodeGoUsageResponse {
    usage: OpenCodeGoUsage,
}

/// Print provider usage status in text or JSON form.
pub async fn status(provider: Option<&str>, json: bool) -> Result<()> {
    let provider = provider.unwrap_or("openrouter").trim().to_ascii_lowercase();
    let report = match provider.as_str() {
        "openrouter" => openrouter_status().await?,
        "opencode" | "opencode-go" => opencode_go_status().await?,
        other => unsupported_report(
            other,
            "This provider does not expose a usage-status integration in Claux yet.",
        ),
    };
    if json {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
        println!();
    } else {
        print_text(&report);
    }
    Ok(())
}

async fn openrouter_status() -> Result<UsageReport> {
    let key = crate::auth::read_openrouter_key()?.or_else(|| {
        std::env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
    });
    let Some(key) = key else {
        return Ok(UsageReport {
            schema_version: 1,
            provider: "openrouter".to_owned(),
            status: UsageStatus::NotConfigured,
            source: Some(OPENROUTER_KEY_URL.to_owned()),
            details: None,
            message: Some(
                "No OpenRouter credential found; run `claux auth login openrouter` or set OPENROUTER_API_KEY.".to_owned(),
            ),
            notes: Vec::new(),
        });
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent(concat!("claux/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("could not create the OpenRouter usage client")?;
    let response = client
        .get(OPENROUTER_KEY_URL)
        .bearer_auth(key)
        .send()
        .await
        .context("OpenRouter usage request failed")?;
    let status = response.status();
    if !status.is_success() {
        bail!("OpenRouter usage endpoint returned HTTP {status}");
    }
    let payload: OpenRouterKeyResponse = response
        .json()
        .await
        .context("OpenRouter returned an invalid usage response")?;

    Ok(UsageReport {
        schema_version: 1,
        provider: "openrouter".to_owned(),
        status: UsageStatus::Available,
        source: Some(OPENROUTER_KEY_URL.to_owned()),
        details: Some(UsageDetails::OpenRouter(payload.data)),
        message: None,
        notes: vec![
            "Values are for the authenticated API key; account-wide credits require a management key and were not queried.".to_owned(),
            "Remaining is unknown when the key has no configured spending limit.".to_owned(),
        ],
    })
}

async fn opencode_go_status() -> Result<UsageReport> {
    let key = std::env::var("OPENCODE_GO_API_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("OPENCODE_API_KEY")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });
    let Some(key) = key else {
        return Ok(UsageReport {
            schema_version: 1,
            provider: "opencode-go".to_owned(),
            status: UsageStatus::NotConfigured,
            source: Some(OPENCODE_GO_USAGE_URL.to_owned()),
            details: None,
            message: Some(
                "No OpenCode Go API key found; set OPENCODE_GO_API_KEY (or OPENCODE_API_KEY)."
                    .to_owned(),
            ),
            notes: vec![
                "OpenCode Go usage is account-wide and covers rolling 5-hour, weekly, and monthly limits."
                    .to_owned(),
                "Credit-wallet balance is separate and is not exposed by this endpoint.".to_owned(),
            ],
        });
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent(concat!("claux/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("could not create the OpenCode Go usage client")?;
    let response = client
        .get(OPENCODE_GO_USAGE_URL)
        .bearer_auth(key)
        .send()
        .await
        .context("OpenCode Go usage request failed")?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Ok(UsageReport {
            schema_version: 1,
            provider: "opencode-go".to_owned(),
            status: UsageStatus::Unavailable,
            source: Some(OPENCODE_GO_USAGE_URL.to_owned()),
            details: None,
            message: Some("OpenCode Go rejected the supplied API key.".to_owned()),
            notes: Vec::new(),
        });
    }
    if status == reqwest::StatusCode::FORBIDDEN {
        return Ok(UsageReport {
            schema_version: 1,
            provider: "opencode-go".to_owned(),
            status: UsageStatus::Unavailable,
            source: Some(OPENCODE_GO_USAGE_URL.to_owned()),
            details: None,
            message: Some("The supplied key does not have an OpenCode Go subscription.".to_owned()),
            notes: Vec::new(),
        });
    }
    if !status.is_success() {
        bail!("OpenCode Go usage endpoint returned HTTP {status}");
    }
    let payload: OpenCodeGoUsageResponse = response
        .json()
        .await
        .context("OpenCode Go returned an invalid usage response")?;

    Ok(UsageReport {
        schema_version: 1,
        provider: "opencode-go".to_owned(),
        status: UsageStatus::Available,
        source: Some(OPENCODE_GO_USAGE_URL.to_owned()),
        details: Some(UsageDetails::OpenCodeGo(payload.usage)),
        message: None,
        notes: vec![
            "Usage is account-wide and includes activity from other OpenCode Go clients."
                .to_owned(),
            "Credit-wallet balance is separate and is not exposed by this endpoint.".to_owned(),
        ],
    })
}

fn unsupported_report(provider: &str, message: &str) -> UsageReport {
    UsageReport {
        schema_version: 1,
        provider: provider.to_owned(),
        status: if provider == "opencode-go" {
            UsageStatus::Unavailable
        } else {
            UsageStatus::Unsupported
        },
        source: None,
        details: None,
        message: Some(message.to_owned()),
        notes: Vec::new(),
    }
}

fn print_text(report: &UsageReport) {
    println!("provider: {}", report.provider);
    println!("status: {}", status_label(&report.status));
    if let Some(source) = &report.source {
        println!("source: {source}");
    }
    if let Some(message) = &report.message {
        println!("message: {message}");
    }
    if let Some(details) = &report.details {
        match details {
            UsageDetails::OpenRouter(details) => {
                if let Some(label) = &details.label {
                    println!("key: {label}");
                }
                println!("key usage: {}", dollars(details.usage));
                println!(
                    "key limit: {}",
                    limit(details.limit, details.limit_reset.as_deref())
                );
                println!("remaining: {}", dollars(details.limit_remaining));
                println!("daily usage: {}", dollars(details.usage_daily));
                println!("weekly usage: {}", dollars(details.usage_weekly));
                println!("monthly usage: {}", dollars(details.usage_monthly));
                if let Some(free_tier) = details.is_free_tier {
                    println!("free tier: {}", if free_tier { "yes" } else { "no" });
                }
                if let Some(disabled) = details.disabled {
                    println!("key disabled: {}", if disabled { "yes" } else { "no" });
                }
            }
            UsageDetails::OpenCodeGo(details) => {
                print_opencode_window("rolling 5-hour", &details.rolling);
                print_opencode_window("weekly", &details.weekly);
                print_opencode_window("monthly", &details.monthly);
            }
        }
    }
    for note in &report.notes {
        println!("note: {note}");
    }
}

fn print_opencode_window(label: &str, window: &OpenCodeGoWindow) {
    println!(
        "{label}: {:.1}% used ({}) — resets at {}",
        window.percent, window.status, window.resets_at
    );
}

fn status_label(status: &UsageStatus) -> &'static str {
    match status {
        UsageStatus::Available => "available",
        UsageStatus::NotConfigured => "not_configured",
        UsageStatus::Unavailable => "unavailable",
        UsageStatus::Unsupported => "unsupported",
    }
}

fn dollars(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |value| format!("${value:.2}"))
}

fn limit(value: Option<f64>, reset: Option<&str>) -> String {
    match (value, reset) {
        (Some(value), Some(reset)) => format!("${value:.2} ({reset})"),
        (Some(value), None) => format!("${value:.2}"),
        (None, _) => "none configured".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_key_status_without_losing_optional_values() {
        let response: OpenRouterKeyResponse = serde_json::from_value(serde_json::json!({
            "data": {
                "label": "sk-or-v1-...890",
                "disabled": false,
                "is_free_tier": false,
                "limit": 100,
                "limit_remaining": 74.5,
                "limit_reset": "monthly",
                "usage": 25.5,
                "usage_daily": 2.0,
                "usage_weekly": 12.0,
                "usage_monthly": 25.5,
                "include_byok_in_limit": false,
                "byok_usage": 0,
                "byok_usage_daily": 0,
                "byok_usage_weekly": 0,
                "byok_usage_monthly": 0
            }
        }))
        .expect("key status");
        assert_eq!(response.data.limit_remaining, Some(74.5));
        assert_eq!(response.data.limit_reset.as_deref(), Some("monthly"));
    }

    #[test]
    fn unknown_limit_is_not_rendered_as_zero() {
        assert_eq!(dollars(None), "unavailable");
        assert_eq!(limit(None, Some("monthly")), "none configured");
        assert_eq!(status_label(&UsageStatus::Unavailable), "unavailable");
    }

    #[test]
    fn parses_opencode_go_usage_windows() {
        let response: OpenCodeGoUsageResponse = serde_json::from_value(serde_json::json!({
            "usage": {
                "rolling": {"status": "ok", "percent": 12.5, "resetsAt": "2026-08-27T19:00:00Z"},
                "weekly": {"status": "ok", "percent": 34, "resetsAt": "2026-08-31T00:00:00Z"},
                "monthly": {"status": "rate-limited", "percent": 100, "resetsAt": "2026-09-01T00:00:00Z"}
            }
        }))
        .expect("OpenCode Go usage");
        assert_eq!(response.usage.rolling.percent, 12.5);
        assert_eq!(response.usage.monthly.status, "rate-limited");
        assert_eq!(response.usage.weekly.resets_at, "2026-08-31T00:00:00Z");
    }
}
