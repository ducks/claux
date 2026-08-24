use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::time::Duration;

const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
const MAX_REQUEST_ATTEMPTS: usize = 6;
const WRAPPER_PREFIX: &str = "CLX_TOKENIZER_FINGERPRINT_BEGIN\n";
const WRAPPER_SUFFIX: &str = "\nCLX_TOKENIZER_FINGERPRINT_END";

const PROBES: &[(&str, &str)] = &[
    ("english-short", "hello infrastructure world"),
    (
        "english-prose",
        "The quick brown fox jumps over the lazy dog while the service recovers.",
    ),
    (
        "camel-snake",
        "tokenizerFingerprint native_tokens_prompt retry_after_ms",
    ),
    (
        "shell",
        "systemctl restart nginx && journalctl -u nginx --since '-5 min'",
    ),
    (
        "python",
        "def repair(node_id: str) -> bool:\n    return health[node_id] == 'ready'",
    ),
    (
        "rust",
        "let repaired: Result<Vec<_>, Error> = nodes.into_iter().map(repair).collect();",
    ),
    (
        "json",
        r#"{"service":"api","replicas":3,"healthy":true,"latency_ms":12.5}"#,
    ),
    ("punctuation", "!@#$%^&*()_+-=[]{}|;:',.<>/?`~\\\""),
    (
        "numbers",
        "000001 1234567890 3.141592653589793 2026-08-24T16:40:10Z",
    ),
    ("whitespace", "alpha  beta\t\tgamma\n\n    delta"),
    (
        "repetition",
        "abababababababababababababababab xyzxyzxyzxyzxyzxyz",
    ),
    (
        "urls",
        "https://例え.テスト/api/v1/健康?节点=主库&ready=true",
    ),
    ("chinese-common", "你好，世界。这个基础设施服务正在恢复。"),
    (
        "chinese-ops",
        "数据库连接池已耗尽，请检查主节点、只读副本和故障转移状态。",
    ),
    (
        "chinese-mixed",
        "部署 API gateway 到 us-west-2，然后验证 Redis 和 PostgreSQL。",
    ),
    (
        "japanese",
        "障害発生後にサービスを再起動し、データベース接続を確認します。",
    ),
    (
        "korean",
        "장애 조치 후 서비스와 데이터베이스 연결 상태를 확인합니다.",
    ),
    (
        "cyrillic",
        "После сбоя проверьте службу, базу данных и очередь заданий.",
    ),
    (
        "arabic",
        "بعد التعطل، تحقق من الخدمة وقاعدة البيانات وقائمة الانتظار.",
    ),
    ("emoji", "🧪🚀🛠️✅❌🔥🤠 infrastructure 👨‍💻👩🏽‍🔧"),
    (
        "combining",
        "cafe\u{301} nai\u{308}ve A\u{30a} re\u{301}sume\u{301}",
    ),
    ("rare-unicode", "𠮷野家 𓀀 ∑ ∆ ∞ → ⟶ ⊕ ⌘ ⚙︎"),
    (
        "zero-width",
        "token\u{200b}izer join\u{200d}ed soft\u{00ad}hyphen",
    ),
    (
        "long-identifiers",
        "HTTPRequestDurationMilliseconds database_connection_pool_exhausted",
    ),
];

#[derive(Debug, Clone, Serialize)]
struct Fingerprint {
    model: String,
    tokenizer_family: Option<String>,
    baseline_prompt_tokens: u64,
    probes: Vec<ProbeResult>,
    total_cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeResult {
    name: String,
    description: String,
    prompt_tokens: u64,
    delta_tokens: i64,
}

#[derive(Debug, Serialize)]
struct Report {
    method: &'static str,
    fingerprints: Vec<Fingerprint>,
    comparisons: Vec<Comparison>,
}

#[derive(Debug, Serialize)]
struct Comparison {
    left: String,
    right: String,
    matching_probes: usize,
    total_probes: usize,
    match_percent: f64,
    identical: bool,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: Option<u64>,
    input_tokens: Option<u64>,
    cost: Option<f64>,
}

pub async fn run(models: &[String], format: crate::cli::TokenizerOutputFormat) -> Result<()> {
    let api_key = openrouter_key()?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()
        .context("build OpenRouter HTTP client")?;
    let families = fetch_tokenizer_families(&client, &api_key).await;
    let mut fingerprints = Vec::with_capacity(models.len());

    for (index, model) in models.iter().enumerate() {
        eprintln!(
            "fingerprinting {} ({}/{}) with {} probes...",
            model,
            index + 1,
            models.len(),
            PROBES.len()
        );
        fingerprints.push(
            fingerprint_model(
                &client,
                &api_key,
                model,
                families.get(model).cloned().flatten(),
            )
            .await?,
        );
    }

    let report = Report {
        method: "native prompt-token deltas against a fixed chat wrapper",
        comparisons: compare_all(&fingerprints),
        fingerprints,
    };
    match format {
        crate::cli::TokenizerOutputFormat::Text => print_report(&report),
        crate::cli::TokenizerOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        crate::cli::TokenizerOutputFormat::Markdown => print!("{}", markdown_report(&report)),
    }
    Ok(())
}

fn openrouter_key() -> Result<String> {
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        if !key.trim().is_empty() {
            return Ok(key);
        }
    }
    crate::auth::read_openrouter_key()?.context(
        "OpenRouter authentication is not configured; set OPENROUTER_API_KEY or run `claux auth login openrouter`",
    )
}

async fn fingerprint_model(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    tokenizer_family: Option<String>,
) -> Result<Fingerprint> {
    let (baseline_prompt_tokens, baseline_cost) =
        count_prompt(client, api_key, model, &wrapped("")).await?;
    let mut probes = Vec::with_capacity(PROBES.len());
    let mut total_cost = baseline_cost;

    for (name, probe) in PROBES {
        let (prompt_tokens, cost) = count_prompt(client, api_key, model, &wrapped(probe)).await?;
        total_cost = add_optional(total_cost, cost);
        probes.push(ProbeResult {
            name: (*name).to_string(),
            description: probe_description(name).to_string(),
            prompt_tokens,
            delta_tokens: prompt_tokens as i64 - baseline_prompt_tokens as i64,
        });
    }

    Ok(Fingerprint {
        model: model.to_string(),
        tokenizer_family,
        baseline_prompt_tokens,
        probes,
        total_cost,
    })
}

async fn count_prompt(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    content: &str,
) -> Result<(u64, Option<f64>)> {
    for attempt in 0..MAX_REQUEST_ATTEMPTS {
        let response = client
            .post(format!("{OPENROUTER_API_BASE}/chat/completions"))
            .bearer_auth(api_key)
            .json(&json!({
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1,
                "stream": false
            }))
            .send()
            .await
            .with_context(|| format!("request tokenizer probe from {model}"))?;
        let status = response.status();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok());
        let body = response
            .text()
            .await
            .with_context(|| format!("read tokenizer probe response from {model}"))?;
        if status.is_success() {
            return decode_usage(model, &body);
        }
        if transient_status(status.as_u16()) && attempt + 1 < MAX_REQUEST_ATTEMPTS {
            let delay = retry_delay(attempt, retry_after);
            eprintln!(
                "{model} returned {status}; retrying in {}s ({}/{})...",
                delay.as_secs(),
                attempt + 2,
                MAX_REQUEST_ATTEMPTS
            );
            tokio::time::sleep(delay).await;
            continue;
        }
        bail!("OpenRouter request for {model} failed ({status}): {body}");
    }

    unreachable!("request loop always returns or fails")
}

fn decode_usage(model: &str, body: &str) -> Result<(u64, Option<f64>)> {
    let decoded: CompletionResponse = serde_json::from_str(body)
        .with_context(|| format!("decode tokenizer probe response from {model}"))?;
    let usage = decoded
        .usage
        .with_context(|| format!("OpenRouter response for {model} omitted usage"))?;
    let prompt_tokens = usage
        .prompt_tokens
        .or(usage.input_tokens)
        .with_context(|| format!("OpenRouter response for {model} omitted prompt token usage"))?;
    Ok((prompt_tokens, usage.cost))
}

fn transient_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

fn retry_delay(attempt: usize, retry_after: Option<u64>) -> Duration {
    Duration::from_secs(retry_after.unwrap_or_else(|| 2_u64.pow(attempt.min(4) as u32 + 1)))
}

async fn fetch_tokenizer_families(
    client: &reqwest::Client,
    api_key: &str,
) -> HashMap<String, Option<String>> {
    let response = match client
        .get(format!("{OPENROUTER_API_BASE}/models"))
        .bearer_auth(api_key)
        .send()
        .await
    {
        Ok(response) => response,
        Err(_) => return HashMap::new(),
    };
    let body: serde_json::Value = match response.json().await {
        Ok(body) => body,
        Err(_) => return HashMap::new(),
    };
    body.pointer("/data")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let id = entry.get("id")?.as_str()?.to_string();
            let tokenizer = entry
                .pointer("/architecture/tokenizer")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            Some((id, tokenizer))
        })
        .collect()
}

fn wrapped(probe: &str) -> String {
    format!("{WRAPPER_PREFIX}{probe}{WRAPPER_SUFFIX}")
}

fn probe_description(name: &str) -> &'static str {
    match name {
        "english-short" => "Common English words and spacing",
        "english-prose" => "English sentence-piece segmentation",
        "camel-snake" => "CamelCase, snake_case, and technical identifiers",
        "shell" => "Shell commands, flags, operators, and quoting",
        "python" => "Python syntax, indentation, and type annotations",
        "rust" => "Rust generics, paths, punctuation, and method chains",
        "json" => "Compact JSON keys, values, punctuation, and decimals",
        "punctuation" => "Dense ASCII symbols and escape-sensitive characters",
        "numbers" => "Leading zeros, long integers, decimals, and timestamps",
        "whitespace" => "Repeated spaces, tabs, newlines, and indentation",
        "repetition" => "Repeated substring merge behavior",
        "urls" => "Unicode domains, URL syntax, paths, and query parameters",
        "chinese-common" => "Common Simplified Chinese characters and punctuation",
        "chinese-ops" => "Chinese infrastructure vocabulary and longer compounds",
        "chinese-mixed" => "Chinese-English code switching and product names",
        "japanese" => "Japanese scripts and operational vocabulary",
        "korean" => "Korean Hangul and operational vocabulary",
        "cyrillic" => "Cyrillic segmentation and inflected words",
        "arabic" => "Arabic script, joining behavior, and punctuation",
        "emoji" => "Emoji sequences, variation selectors, and skin tones",
        "combining" => "Decomposed Latin characters with combining marks",
        "rare-unicode" => "Rare CJK, ancient symbols, math, and technical glyphs",
        "zero-width" => "Zero-width joiners, spaces, and soft hyphens",
        "long-identifiers" => "Long compound identifiers common in telemetry and code",
        _ => "Tokenizer segmentation behavior",
    }
}

fn add_optional(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn compare_all(fingerprints: &[Fingerprint]) -> Vec<Comparison> {
    let mut comparisons = Vec::new();
    for left_index in 0..fingerprints.len() {
        for right_index in (left_index + 1)..fingerprints.len() {
            let left = &fingerprints[left_index];
            let right = &fingerprints[right_index];
            let matching_probes = left
                .probes
                .iter()
                .zip(&right.probes)
                .filter(|(left, right)| left.delta_tokens == right.delta_tokens)
                .count();
            let total_probes = left.probes.len().min(right.probes.len());
            comparisons.push(Comparison {
                left: left.model.clone(),
                right: right.model.clone(),
                matching_probes,
                total_probes,
                match_percent: if total_probes == 0 {
                    0.0
                } else {
                    matching_probes as f64 * 100.0 / total_probes as f64
                },
                identical: matching_probes == total_probes,
            });
        }
    }
    comparisons
}

fn print_report(report: &Report) {
    println!("TOKENIZER FINGERPRINT");
    println!("method: {}", report.method);
    println!();
    println!("model                                      family      baseline   cost");
    println!("-----------------------------------------------------------------------");
    for fingerprint in &report.fingerprints {
        let cost = fingerprint
            .total_cost
            .map(|cost| format!("${cost:.6}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{:<42} {:<11} {:>8}   {:>9}",
            fingerprint.model,
            fingerprint.tokenizer_family.as_deref().unwrap_or("unknown"),
            fingerprint.baseline_prompt_tokens,
            cost
        );
    }
    println!();
    for comparison in &report.comparisons {
        println!(
            "{} vs {}: {}/{} deltas match ({:.1}%){}",
            comparison.left,
            comparison.right,
            comparison.matching_probes,
            comparison.total_probes,
            comparison.match_percent,
            if comparison.identical {
                " — identical fingerprint"
            } else {
                ""
            }
        );
    }
    println!();
    println!(
        "Matching deltas are evidence of shared tokenization behavior, not proof of model identity."
    );
}

fn markdown_report(report: &Report) -> String {
    let mut output = String::new();
    output.push_str("# Tokenizer Fingerprint\n\n");
    output.push_str(&format!("**Method:** {}\n\n", report.method));
    output.push_str("## Models\n\n");
    output.push_str("| Model | Metadata family | Baseline prompt tokens | Probe cost |\n");
    output.push_str("|---|---:|---:|---:|\n");
    for fingerprint in &report.fingerprints {
        let cost = fingerprint
            .total_cost
            .map(|cost| format!("${cost:.6}"))
            .unwrap_or_else(|| "n/a".to_string());
        output.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            markdown_escape(&fingerprint.model),
            fingerprint.tokenizer_family.as_deref().unwrap_or("unknown"),
            fingerprint.baseline_prompt_tokens,
            cost
        ));
    }

    output.push_str("\n## Comparisons\n\n");
    for comparison in &report.comparisons {
        output.push_str(&format!(
            "- `{}` vs `{}`: **{}/{} deltas match ({:.1}%)**{}\n",
            markdown_escape(&comparison.left),
            markdown_escape(&comparison.right),
            comparison.matching_probes,
            comparison.total_probes,
            comparison.match_percent,
            if comparison.identical {
                " — identical fingerprint"
            } else {
                ""
            }
        ));
    }
    output.push_str(
        "\n> Matching deltas are evidence of shared tokenization behavior, not proof of model identity.\n\n",
    );

    output.push_str("## Probe deltas\n\n");
    output.push_str("| Probe | What it checks |");
    for fingerprint in &report.fingerprints {
        output.push_str(&format!(" `{}` |", markdown_escape(&fingerprint.model)));
    }
    output.push_str("\n|---|---|");
    for _ in &report.fingerprints {
        output.push_str("---:|");
    }
    output.push('\n');
    if let Some(first) = report.fingerprints.first() {
        for (probe_index, probe) in first.probes.iter().enumerate() {
            output.push_str(&format!(
                "| {} | {} |",
                markdown_escape(&probe.name),
                markdown_escape(&probe.description)
            ));
            for fingerprint in &report.fingerprints {
                let delta = fingerprint
                    .probes
                    .get(probe_index)
                    .map(|probe| probe.delta_tokens.to_string())
                    .unwrap_or_else(|| "n/a".to_string());
                output.push_str(&format!(" {delta} |"));
            }
            output.push('\n');
        }
    }
    output
}

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|").replace('`', "\\`")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(model: &str, deltas: &[i64]) -> Fingerprint {
        Fingerprint {
            model: model.to_string(),
            tokenizer_family: Some("Other".to_string()),
            baseline_prompt_tokens: 10,
            probes: deltas
                .iter()
                .enumerate()
                .map(|(index, delta)| ProbeResult {
                    name: format!("probe-{index}"),
                    description: format!("description-{index}"),
                    prompt_tokens: (10 + delta) as u64,
                    delta_tokens: *delta,
                })
                .collect(),
            total_cost: None,
        }
    }

    #[test]
    fn compares_differential_fingerprints() {
        let report = compare_all(&[
            fingerprint("one", &[1, 2, 3, 4]),
            fingerprint("two", &[1, 8, 3, 4]),
        ]);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].matching_probes, 3);
        assert_eq!(report[0].total_probes, 4);
        assert_eq!(report[0].match_percent, 75.0);
        assert!(!report[0].identical);
    }

    #[test]
    fn wrapper_keeps_probe_boundaries_fixed() {
        assert_eq!(
            wrapped("hello"),
            "CLX_TOKENIZER_FINGERPRINT_BEGIN\nhello\nCLX_TOKENIZER_FINGERPRINT_END"
        );
    }

    #[test]
    fn sums_reported_cost_when_available() {
        assert!((add_optional(Some(0.1), Some(0.2)).unwrap() - 0.3).abs() < f64::EPSILON);
        assert_eq!(add_optional(None, Some(0.2)), Some(0.2));
        assert_eq!(add_optional(None, None), None);
    }

    #[test]
    fn retries_only_transient_provider_failures() {
        for status in [429, 500, 502, 503, 504] {
            assert!(transient_status(status));
        }
        for status in [400, 401, 402, 403, 404] {
            assert!(!transient_status(status));
        }
    }

    #[test]
    fn retry_delay_honors_header_and_bounds_backoff() {
        assert_eq!(retry_delay(0, Some(7)), Duration::from_secs(7));
        assert_eq!(retry_delay(0, None), Duration::from_secs(2));
        assert_eq!(retry_delay(8, None), Duration::from_secs(32));
    }

    #[test]
    fn markdown_contains_summary_and_probe_evidence() {
        let left = fingerprint("one", &[1, 2]);
        let right = fingerprint("two", &[1, 2]);
        let report = Report {
            method: "test method",
            comparisons: compare_all(&[left.clone(), right.clone()]),
            fingerprints: vec![left, right],
        };

        let markdown = markdown_report(&report);
        assert!(markdown.contains("# Tokenizer Fingerprint"));
        assert!(markdown.contains("**2/2 deltas match (100.0%)**"));
        assert!(markdown.contains("| probe-0 | description-0 | 1 | 1 |"));
        assert!(markdown.contains("not proof of model identity"));
    }

    #[test]
    fn every_probe_has_a_specific_description() {
        for (name, _) in PROBES {
            assert_ne!(probe_description(name), "Tokenizer segmentation behavior");
        }
    }
}
