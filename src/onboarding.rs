//! First-run configuration and diagnostics.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;
use toml_edit::{value, DocumentMut, Item, Table};

use crate::cli::ConfigProvider;
use crate::config::{Config, ProviderKind, ResolvedModel};

pub struct DoctorReport {
    pub text: String,
    pub healthy: bool,
}

pub fn init_config(provider: ConfigProvider, model: Option<&str>, force: bool) -> Result<PathBuf> {
    init_config_at(&Config::global_path(), provider, model, force)
}

fn init_config_at(
    path: &Path,
    provider: ConfigProvider,
    model: Option<&str>,
    force: bool,
) -> Result<PathBuf> {
    if path
        .symlink_metadata()
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        anyhow::bail!("refusing to replace symlink {}", path.display());
    }

    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("configuration path has no parent"))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("could not create {}", parent.display()))?;
    set_private_directory(parent)?;

    let existing = if path.exists() && !force {
        std::fs::read_to_string(path)
            .with_context(|| format!("could not read {}", path.display()))?
    } else {
        String::new()
    };
    let template = add_model_profile(&existing, provider, model)?;
    toml::from_str::<Config>(&template).context("generated configuration was invalid")?;

    let temporary = parent.join(format!(".config.toml.{}.tmp", uuid::Uuid::new_v4()));
    let write_result = (|| -> Result<()> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temporary)
            .with_context(|| format!("could not create {}", temporary.display()))?;
        file.write_all(template.as_bytes())?;
        file.sync_all()?;
        #[cfg(windows)]
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        std::fs::rename(&temporary, path)?;
        set_private_file(path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    write_result?;
    Ok(path.to_path_buf())
}

fn config_template(provider: ConfigProvider, model: Option<&str>) -> String {
    add_model_profile("", provider, model).expect("built-in configuration must be valid")
}

fn add_model_profile(
    existing: &str,
    provider: ConfigProvider,
    model: Option<&str>,
) -> Result<String> {
    if !existing.trim().is_empty() {
        toml::from_str::<Config>(existing).context("existing claux configuration is invalid")?;
    }
    let mut document = if existing.trim().is_empty() {
        let mut document = "# claux configuration\n".parse::<DocumentMut>()?;
        document["permission_mode"] = value("default");
        document["max_tokens"] = value(16384);
        document["auto_compact_threshold"] = value(0.8);
        document
    } else {
        existing
            .parse::<DocumentMut>()
            .context("existing configuration is invalid TOML")?
    };
    migrate_legacy_model(&mut document);
    if document.get("providers").is_none() {
        document["providers"] = Item::Table(Table::new());
    }
    if document.get("model_profiles").is_none() {
        document["model_profiles"] = Item::Table(Table::new());
    }

    let specification = ProviderSpecification::new(provider, model);
    let provider_id = unique_table_name(&document, "providers", specification.id, |item| {
        item.get("type").and_then(Item::as_str) == Some(specification.kind)
            && item.get("base_url").and_then(Item::as_str) == specification.base_url
    });
    if document["providers"]
        .get(&provider_id)
        .and_then(Item::as_table_like)
        .is_none()
    {
        let mut table = Table::new();
        table["type"] = value(specification.kind);
        if let Some(base_url) = specification.base_url {
            table["base_url"] = value(base_url);
        }
        table["name"] = value(specification.id);
        table["protocol"] = value(specification.protocol);
        table["api_key_env"] = value(specification.api_key_env);
        document["providers"][&provider_id] = Item::Table(table);
    }

    let profile_base = format!("{}-{}", specification.id, slug(specification.model));
    let profile_id = unique_table_name(&document, "model_profiles", &profile_base, |item| {
        item.get("provider").and_then(Item::as_str) == Some(provider_id.as_str())
            && item.get("model").and_then(Item::as_str) == Some(specification.model)
    });
    if document["model_profiles"]
        .get(&profile_id)
        .and_then(Item::as_table_like)
        .is_none()
    {
        let mut table = Table::new();
        table["provider"] = value(provider_id);
        table["model"] = value(specification.model);
        table["display_name"] = value(specification.model);
        document["model_profiles"][&profile_id] = Item::Table(table);
    }
    if document.get("default_profile").is_none() {
        document["default_profile"] = value(profile_id);
    }
    Ok(document.to_string())
}

fn migrate_legacy_model(document: &mut DocumentMut) {
    let already_named = document
        .get("model_profiles")
        .and_then(Item::as_table_like)
        .is_some_and(|profiles| !profiles.is_empty());
    let Some(model) = document
        .get("model")
        .and_then(Item::as_str)
        .map(str::to_string)
    else {
        return;
    };
    if already_named || model.trim().is_empty() {
        return;
    }

    if document.get("providers").is_none() {
        document["providers"] = Item::Table(Table::new());
    }
    if document.get("model_profiles").is_none() {
        document["model_profiles"] = Item::Table(Table::new());
    }

    let openai_base_url = document
        .get("openai_base_url")
        .and_then(Item::as_str)
        .map(str::to_string);
    let provider_id = if openai_base_url.is_some() {
        document
            .get("openai_provider_name")
            .and_then(Item::as_str)
            .map(slug)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "openai".to_string())
    } else {
        "anthropic".to_string()
    };
    let mut provider = Table::new();
    provider["type"] = value(if openai_base_url.is_some() {
        "openai"
    } else {
        "anthropic"
    });
    if let Some(base_url) = openai_base_url {
        provider["base_url"] = value(base_url);
        provider["name"] = value(
            document
                .get("openai_provider_name")
                .and_then(Item::as_str)
                .unwrap_or("openai"),
        );
        provider["protocol"] = value(
            document
                .get("openai_protocol")
                .and_then(Item::as_str)
                .unwrap_or("chat_completions"),
        );
        copy_value(document, &mut provider, "openai_api_key", "api_key");
        copy_value(document, &mut provider, "openai_api_key_env", "api_key_env");
        copy_value(document, &mut provider, "openai_api_key_cmd", "api_key_cmd");
    } else {
        provider["name"] = value("anthropic");
        copy_value(document, &mut provider, "api_key", "api_key");
        copy_value(document, &mut provider, "api_key_env", "api_key_env");
        copy_value(document, &mut provider, "api_key_cmd", "api_key_cmd");
    }
    document["providers"][&provider_id] = Item::Table(provider);

    let profile_id = format!("{}-{}", provider_id, slug(&model));
    let mut profile = Table::new();
    profile["provider"] = value(provider_id);
    profile["model"] = value(&model);
    profile["display_name"] = value(model);
    if let Some(reasoning) = document.get("openai_reasoning_effort").cloned() {
        profile["reasoning_effort"] = reasoning;
    }
    document["model_profiles"][&profile_id] = Item::Table(profile);
    if document.get("default_profile").is_none() {
        document["default_profile"] = value(profile_id);
    }
}

fn copy_value(document: &DocumentMut, table: &mut Table, source: &str, target: &str) {
    if let Some(item) = document.get(source).cloned() {
        table[target] = item;
    }
}

fn unique_table_name(
    document: &DocumentMut,
    collection: &str,
    base: &str,
    matches: impl Fn(&Item) -> bool,
) -> String {
    for suffix in 1.. {
        let candidate = if suffix == 1 {
            base.to_string()
        } else {
            format!("{base}-{suffix}")
        };
        match document
            .get(collection)
            .and_then(|item| item.get(&candidate))
        {
            Some(item) if matches(item) => return candidate,
            Some(_) => continue,
            None => return candidate,
        }
    }
    unreachable!()
}

fn slug(value: &str) -> String {
    let slug = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    let slug = slug
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() {
        "model".to_string()
    } else {
        slug
    }
}

struct ProviderSpecification<'a> {
    id: &'static str,
    kind: &'static str,
    base_url: Option<&'static str>,
    protocol: &'static str,
    api_key_env: &'static str,
    model: &'a str,
}

impl<'a> ProviderSpecification<'a> {
    fn new(provider: ConfigProvider, model: Option<&'a str>) -> Self {
        match provider {
            ConfigProvider::Anthropic => Self {
                id: "anthropic",
                kind: "anthropic",
                base_url: None,
                protocol: "chat_completions",
                api_key_env: "ANTHROPIC_API_KEY",
                model: model.unwrap_or("claude-sonnet-5"),
            },
            ConfigProvider::Openai => Self {
                id: "openai",
                kind: "openai",
                base_url: Some("https://api.openai.com/v1"),
                protocol: "responses",
                api_key_env: "OPENAI_API_KEY",
                model: model.unwrap_or("gpt-5.6-sol"),
            },
            ConfigProvider::OpenRouter => Self {
                id: "openrouter",
                kind: "openai",
                base_url: Some("https://openrouter.ai/api/v1"),
                protocol: "chat_completions",
                api_key_env: "OPENROUTER_API_KEY",
                model: model.unwrap_or("anthropic/claude-sonnet-5"),
            },
            ConfigProvider::Ollama => Self {
                id: "ollama",
                kind: "openai",
                base_url: Some("http://localhost:11434/v1"),
                protocol: "chat_completions",
                api_key_env: "OLLAMA_API_KEY",
                model: model.unwrap_or("llama3"),
            },
        }
    }
}

pub async fn doctor(config: &Config, offline: bool) -> DoctorReport {
    let mut report = ReportBuilder::default();
    let config_path = Config::global_path();

    if config_path.exists() {
        report.ok(format!("configuration: {}", config_path.display()));
        if std::fs::metadata(&config_path)
            .map(|metadata| metadata.permissions().readonly())
            .unwrap_or(false)
        {
            report.warn("configuration file is read-only".to_string());
        }
    } else {
        report.warn(format!(
            "configuration: {} does not exist (run `claux config init`)",
            config_path.display()
        ));
    }

    let models = match config.selectable_models() {
        Ok(models) if !models.is_empty() => {
            report.ok(format!("models configured: {}", models.len()));
            models
        }
        Ok(_) => {
            report.fail("no models configured".to_string());
            Vec::new()
        }
        Err(error) => {
            report.fail(format!("model configuration: {error}"));
            Vec::new()
        }
    };
    if !(0.0..=1.0).contains(&config.auto_compact_threshold) {
        report.fail("auto_compact_threshold must be between 0 and 1".to_string());
    }

    match Command::available("git") {
        true => report.ok("git executable found".to_string()),
        false => {
            report.warn("git executable not found; context and checkpoints are limited".into())
        }
    }
    match Command::available("rg") {
        true => report.ok("ripgrep executable found".to_string()),
        false => report
            .warn("ripgrep executable not found; initial repository context is limited".into()),
    }

    let project_servers = config
        .project_trust
        .as_ref()
        .map(crate::config::load_mcp_json)
        .unwrap_or_default();
    for server in config.mcp_servers.iter().chain(&project_servers) {
        if Command::available(&server.command) {
            report.ok(format!(
                "MCP command found: {} ({})",
                server.name, server.command
            ));
        } else {
            report.fail(format!(
                "MCP command not found: {} ({})",
                server.name, server.command
            ));
        }
    }
    for plugin in &config.plugins {
        if Command::available(&plugin.command) {
            report.ok(format!(
                "hook command found: {} ({})",
                plugin.name, plugin.command
            ));
        } else {
            report.fail(format!(
                "hook command not found: {} ({})",
                plugin.name, plugin.command
            ));
        }
    }

    let mut ready_models = Vec::new();
    let mut checked_providers = HashSet::new();
    for model in models {
        if !checked_providers.insert(model.binding.provider.clone()) {
            continue;
        }
        let key = model.resolve_api_key();
        let label = format!(
            "{} ({})",
            model.binding.display_name, model.binding.provider_name
        );
        if model.requires_api_key() && key.is_none() {
            report.fail(format!(
                "authentication: {} requires {}",
                label, model.binding.api_key_env
            ));
        } else if key.is_some() {
            report.ok(format!("authentication: {label} key resolved"));
            ready_models.push((model, key));
        } else {
            report.ok(format!("authentication: {label} uses no API key"));
            ready_models.push((model, None));
        }
    }

    if offline {
        report.warn("provider connectivity: skipped (--offline)".to_string());
    } else {
        for (model, key) in ready_models {
            let label = format!(
                "{} ({})",
                model.binding.display_name, model.binding.provider_name
            );
            match check_provider(&model, key.as_deref()).await {
                Ok(status) => report.ok(format!("provider connectivity: {label}: HTTP {status}")),
                Err(error) => report.fail(format!("provider connectivity: {label}: {error}")),
            }
        }
    }

    let trust = config
        .project_trust
        .as_ref()
        .map(|trust| {
            if trust.is_trusted() {
                "trusted"
            } else {
                "untrusted (project commands and permission loosening are blocked)"
            }
        })
        .unwrap_or("not resolved");
    report.ok(format!("current project: {trust}"));

    report.finish()
}

async fn check_provider(
    model: &ResolvedModel,
    api_key: Option<&str>,
) -> Result<reqwest::StatusCode> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let request = match model.binding.provider_kind {
        ProviderKind::Anthropic => client
            .get("https://api.anthropic.com/v1/models?limit=1")
            .header("anthropic-version", "2023-06-01")
            .header(
                "x-api-key",
                api_key.ok_or_else(|| anyhow::anyhow!("API key is missing"))?,
            ),
        ProviderKind::Openai => {
            let base = model
                .binding
                .base_url
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("OpenAI-compatible base URL is missing"))?
                .trim_end_matches('/');
            let request = client.get(format!("{base}/models"));
            if let Some(key) = api_key {
                request.bearer_auth(key)
            } else {
                request
            }
        }
    };
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("server returned HTTP {status}");
    }
    Ok(status)
}

struct Command;

impl Command {
    fn available(command: &str) -> bool {
        let path = Path::new(command);
        if path.components().count() > 1 {
            return is_executable(path);
        }
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).any(|directory| {
            executable_candidates(&directory, command)
                .into_iter()
                .any(|candidate| is_executable(&candidate))
        })
    }
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

fn executable_candidates(directory: &Path, command: &str) -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        let extensions =
            std::env::var_os("PATHEXT").unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".into());
        return extensions
            .to_string_lossy()
            .split(';')
            .map(|extension| directory.join(format!("{command}{extension}")))
            .collect();
    }
    #[cfg(not(windows))]
    {
        vec![directory.join(command)]
    }
}

#[derive(Default)]
struct ReportBuilder {
    lines: Vec<String>,
    failures: usize,
}

impl ReportBuilder {
    fn ok(&mut self, message: String) {
        self.lines.push(format!("[ok]   {message}"));
    }

    fn warn(&mut self, message: String) {
        self.lines.push(format!("[warn] {message}"));
    }

    fn fail(&mut self, message: String) {
        self.failures += 1;
        self.lines.push(format!("[fail] {message}"));
    }

    fn finish(self) -> DoctorReport {
        let healthy = self.failures == 0;
        let summary = if healthy {
            "claux doctor: ready"
        } else {
            "claux doctor: action required"
        };
        DoctorReport {
            text: format!("{summary}\n\n{}\n", self.lines.join("\n")),
            healthy,
        }
    }
}

#[cfg(unix)]
fn set_private_directory(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_file(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_init_template_parses() {
        for provider in [
            ConfigProvider::Anthropic,
            ConfigProvider::Openai,
            ConfigProvider::OpenRouter,
            ConfigProvider::Ollama,
        ] {
            let parsed: Config = toml::from_str(&config_template(provider, None)).unwrap();
            assert_eq!(parsed.model_profiles.len(), 1);
            assert_eq!(parsed.providers.len(), 1);
            assert!(parsed.default_profile.is_some());
        }
    }

    #[test]
    fn openrouter_template_uses_compatible_endpoint_and_key_environment() {
        let parsed: Config =
            toml::from_str(&config_template(ConfigProvider::OpenRouter, None)).unwrap();
        let profile = &parsed.model_profiles["openrouter-anthropic-claude-sonnet-5"];
        let provider = &parsed.providers["openrouter"];
        assert_eq!(profile.model, "anthropic/claude-sonnet-5");
        assert_eq!(
            provider.base_url.as_deref(),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(
            provider.protocol,
            crate::config::OpenAIProtocol::ChatCompletions
        );
        assert_eq!(provider.api_key_env.as_deref(), Some("OPENROUTER_API_KEY"));
        assert!(provider.api_key.is_none());
    }

    #[test]
    fn init_adds_to_existing_config_without_overwriting_it() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("config.toml");
        std::fs::write(
            &path,
            "# keep this comment\nmax_tokens = 42\ncustom_setting = \"keep\"\n",
        )
        .unwrap();

        init_config_at(&path, ConfigProvider::Anthropic, None, false).unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("# keep this comment"));
        assert!(content.contains("max_tokens = 42"));
        assert!(content.contains("custom_setting = \"keep\""));
        assert!(content.contains("[providers.anthropic]"));
    }

    #[test]
    fn repeated_init_is_idempotent() {
        let first = config_template(ConfigProvider::OpenRouter, Some("test/model"));
        let second =
            add_model_profile(&first, ConfigProvider::OpenRouter, Some("test/model")).unwrap();
        let parsed: Config = toml::from_str(&second).unwrap();
        assert_eq!(parsed.providers.len(), 1);
        assert_eq!(parsed.model_profiles.len(), 1);
    }

    #[test]
    fn additive_init_migrates_an_existing_legacy_model() {
        let existing = r#"
            # legacy settings remain readable
            model = "claude-existing"
            api_key_env = "EXISTING_ANTHROPIC_KEY"
        "#;
        let updated =
            add_model_profile(existing, ConfigProvider::OpenRouter, Some("openai/new")).unwrap();
        let parsed: Config = toml::from_str(&updated).unwrap();

        assert_eq!(parsed.model_profiles.len(), 2);
        let legacy = parsed.resolve_model("anthropic-claude-existing").unwrap();
        assert_eq!(legacy.binding.model, "claude-existing");
        assert_eq!(legacy.binding.api_key_env, "EXISTING_ANTHROPIC_KEY");
        assert_eq!(
            parsed.default_profile.as_deref(),
            Some("anthropic-claude-existing")
        );
    }

    #[test]
    fn forced_init_replaces_config_without_embedding_secrets() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested/config.toml");
        let result =
            init_config_at(&path, ConfigProvider::Openai, Some("test-model"), true).unwrap();
        let content = std::fs::read_to_string(result).unwrap();
        assert!(content.contains("test-model"));
        assert!(content.contains("OPENAI_API_KEY"));
        assert!(!content.contains("sk-"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[tokio::test]
    async fn offline_doctor_skips_network() {
        let config = Config {
            api_key_env: "CLAUX_TEST_MISSING_AUTH".to_string(),
            ..Config::default()
        };
        let report = doctor(&config, true).await;
        assert!(report.text.contains("skipped (--offline)"));
    }

    #[tokio::test]
    async fn doctor_requires_openrouter_api_key() {
        let config = Config {
            model: "anthropic/claude-sonnet-5".to_string(),
            openai_base_url: Some("https://openrouter.ai/api/v1".to_string()),
            openai_provider_name: Some("openrouter".to_string()),
            openai_api_key_env: "CLAUX_TEST_MISSING_OPENROUTER_KEY".to_string(),
            ..Config::default()
        };
        let report = doctor(&config, true).await;
        assert!(!report.healthy);
        assert!(report
            .text
            .contains("requires CLAUX_TEST_MISSING_OPENROUTER_KEY"));
    }

    #[tokio::test]
    async fn doctor_marks_invalid_configuration_unhealthy() {
        let config = Config {
            model: String::new(),
            auto_compact_threshold: 2.0,
            ..Config::default()
        };
        let report = doctor(&config, true).await;
        assert!(!report.healthy);
        assert!(report.text.contains("[fail] no models configured"));
        assert!(report.text.contains("auto_compact_threshold"));
    }
}
