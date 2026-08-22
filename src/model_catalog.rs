//! Best-effort provider model metadata discovery.
//!
//! Explicit configuration wins. Otherwise supported providers may refresh a
//! disk cache and fall back to stale or built-in metadata when offline.

use crate::config::ResolvedModel;
use crate::model::ModelMetadata;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CACHE_SCHEMA_VERSION: u32 = 1;
const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    schema_version: u32,
    model: String,
    context_window: usize,
    fetched_at_unix: u64,
}

pub async fn resolve(resolved: &ResolvedModel) -> ModelMetadata {
    resolve_with_cache_dir(resolved, &cache_dir()).await
}

async fn resolve_with_cache_dir(resolved: &ResolvedModel, cache_dir: &Path) -> ModelMetadata {
    if resolved.context_window_override.is_some() || !is_openrouter(resolved) {
        return resolved.metadata;
    }

    let cache_path = cache_path(cache_dir, &resolved.binding.model);
    let cached = read_cache(&cache_path, &resolved.binding.model);
    if let Some(entry) = cached.as_ref().filter(|entry| cache_is_fresh(entry)) {
        tracing::debug!(
            "using cached model metadata for {} ({} token context)",
            resolved.binding.model,
            entry.context_window
        );
        return with_context_window(resolved.metadata, entry.context_window);
    }

    match fetch_openrouter_context_window(resolved).await {
        Ok(context_window) => {
            tracing::debug!(
                "discovered model metadata for {} ({} token context)",
                resolved.binding.model,
                context_window
            );
            let entry = CacheEntry {
                schema_version: CACHE_SCHEMA_VERSION,
                model: resolved.binding.model.clone(),
                context_window,
                fetched_at_unix: now_unix(),
            };
            if let Err(error) = write_cache(&cache_path, &entry) {
                tracing::debug!("could not cache provider model metadata: {error}");
            }
            with_context_window(resolved.metadata, context_window)
        }
        Err(error) => {
            if let Some(cached) = cached {
                tracing::warn!(
                    "model metadata refresh failed for {}; using stale cache: {error}",
                    resolved.binding.model
                );
                with_context_window(resolved.metadata, cached.context_window)
            } else {
                tracing::warn!(
                    "model metadata discovery failed for {}; using built-in fallback: {error}",
                    resolved.binding.model
                );
                resolved.metadata
            }
        }
    }
}

fn with_context_window(mut metadata: ModelMetadata, context_window: usize) -> ModelMetadata {
    metadata.context_window = context_window;
    metadata
}

fn is_openrouter(resolved: &ResolvedModel) -> bool {
    resolved
        .binding
        .provider_name
        .eq_ignore_ascii_case("openrouter")
        || resolved
            .binding
            .base_url
            .as_deref()
            .and_then(|url| reqwest::Url::parse(url).ok())
            .and_then(|url| url.host_str().map(str::to_string))
            .is_some_and(|host| host == "openrouter.ai" || host.ends_with(".openrouter.ai"))
}

async fn fetch_openrouter_context_window(resolved: &ResolvedModel) -> anyhow::Result<usize> {
    let base_url = resolved
        .binding
        .base_url
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("OpenRouter profile has no base URL"))?;
    let url = format!(
        "{}/model/{}",
        base_url.trim_end_matches('/'),
        resolved.binding.model
    );
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;
    let mut request = client.get(url);
    if let Some(api_key) = resolved.resolve_api_key() {
        request = request.bearer_auth(api_key);
    }
    let body: serde_json::Value = request.send().await?.error_for_status()?.json().await?;
    body.pointer("/data/context_length")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow::anyhow!("response omitted a positive data.context_length"))
}

fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("claux")
        .join("model-metadata")
}

fn cache_path(cache_dir: &Path, model: &str) -> PathBuf {
    let encoded = model
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    cache_dir.join(format!("{encoded}.json"))
}

fn read_cache(path: &Path, model: &str) -> Option<CacheEntry> {
    let entry: CacheEntry = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    (entry.schema_version == CACHE_SCHEMA_VERSION
        && entry.model == model
        && entry.context_window > 0)
        .then_some(entry)
}

fn cache_is_fresh(entry: &CacheEntry) -> bool {
    now_unix().saturating_sub(entry.fetched_at_unix) <= CACHE_TTL.as_secs()
}

fn write_cache(path: &Path, entry: &CacheEntry) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(entry)?)?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn config(base_url: &str, context_window: Option<usize>) -> Config {
        let override_line = context_window
            .map(|value| format!("context_window = {value}"))
            .unwrap_or_default();
        toml::from_str(&format!(
            r#"
            [providers.openrouter]
            type = "openai"
            name = "openrouter"
            base_url = "{base_url}"
            api_key = "test-key"

            [model_profiles.test]
            provider = "openrouter"
            model = "vendor/new-model"
            {override_line}
            "#
        ))
        .unwrap()
    }

    fn serve_once(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let length = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..length]);
            assert!(request.starts_with("GET /model/vendor/new-model "));
            assert!(request.contains("authorization: Bearer test-key\r\n"));
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn explicit_override_bypasses_discovery() {
        let config = config("http://127.0.0.1:1", Some(64_000));
        let resolved = config.resolve_model("test").unwrap();
        let cache = tempfile::tempdir().unwrap();
        let metadata = resolve_with_cache_dir(&resolved, cache.path()).await;
        assert_eq!(metadata.context_window, 64_000);
        assert!(std::fs::read_dir(cache.path()).unwrap().next().is_none());
    }

    #[tokio::test]
    async fn discovers_and_caches_openrouter_context_window() {
        let base_url = serve_once(r#"{"data":{"context_length":1048576}}"#);
        let config = config(&base_url, None);
        let resolved = config.resolve_model("test").unwrap();
        let cache = tempfile::tempdir().unwrap();
        let discovered = resolve_with_cache_dir(&resolved, cache.path()).await;
        let cached = resolve_with_cache_dir(&resolved, cache.path()).await;
        assert_eq!(discovered.context_window, 1_048_576);
        assert_eq!(cached.context_window, 1_048_576);
    }

    #[tokio::test]
    async fn stale_cache_survives_provider_failure() {
        let config = config("http://127.0.0.1:1", None);
        let resolved = config.resolve_model("test").unwrap();
        let cache = tempfile::tempdir().unwrap();
        write_cache(
            &cache_path(cache.path(), &resolved.binding.model),
            &CacheEntry {
                schema_version: CACHE_SCHEMA_VERSION,
                model: resolved.binding.model.clone(),
                context_window: 333_000,
                fetched_at_unix: 0,
            },
        )
        .unwrap();
        let metadata = resolve_with_cache_dir(&resolved, cache.path()).await;
        assert_eq!(metadata.context_window, 333_000);
    }

    #[tokio::test]
    async fn malformed_discovery_falls_back_to_built_in_metadata() {
        let base_url = serve_once(r#"{"data":{}}"#);
        let config = config(&base_url, None);
        let resolved = config.resolve_model("test").unwrap();
        let cache = tempfile::tempdir().unwrap();
        let metadata = resolve_with_cache_dir(&resolved, cache.path()).await;
        assert_eq!(metadata, resolved.metadata);
    }

    #[tokio::test]
    async fn generic_compatible_provider_is_not_probed() {
        let config: Config = toml::from_str(
            r#"
            [providers.local]
            type = "openai"
            name = "local"
            base_url = "http://127.0.0.1:1"

            [model_profiles.test]
            provider = "local"
            model = "vendor/new-model"
            "#,
        )
        .unwrap();
        let resolved = config.resolve_model("test").unwrap();
        let cache = tempfile::tempdir().unwrap();
        let metadata = resolve_with_cache_dir(&resolved, cache.path()).await;
        assert_eq!(metadata, resolved.metadata);
        assert!(std::fs::read_dir(cache.path()).unwrap().next().is_none());
    }
}
