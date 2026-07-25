//! First-run configuration and diagnostics.

use anyhow::{Context, Result};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::cli::ConfigProvider;
use crate::config::{AnthropicApiKey, Config};

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
    if path.exists() && !force {
        anyhow::bail!(
            "{} already exists; inspect it or pass --force to replace it",
            path.display()
        );
    }
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

    let template = config_template(provider, model);
    // Validate our generated output before putting it on disk.
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
    let shared = "permission_mode = \"default\"\n\
                  max_tokens = 16384\n\
                  auto_compact_threshold = 0.8\n";
    match provider {
        ConfigProvider::Anthropic => format!(
            "# claux configuration\n\
             # Set ANTHROPIC_API_KEY in your environment; do not put secrets here.\n\
             model = {:?}\n\
             api_key_env = \"ANTHROPIC_API_KEY\"\n\
             {shared}",
            model.unwrap_or("claude-sonnet-5")
        ),
        ConfigProvider::Openai => format!(
            "# claux configuration\n\
             # Set OPENAI_API_KEY in your environment; do not put secrets here.\n\
             model = {:?}\n\
             openai_base_url = \"https://api.openai.com/v1\"\n\
             openai_provider_name = \"openai\"\n\
             openai_protocol = \"responses\"\n\
             openai_api_key_env = \"OPENAI_API_KEY\"\n\
             {shared}",
            model.unwrap_or("gpt-5.6-sol")
        ),
        ConfigProvider::Ollama => format!(
            "# claux configuration\n\
             # Change the model to one installed by `ollama list`.\n\
             model = {:?}\n\
             openai_base_url = \"http://localhost:11434/v1\"\n\
             openai_provider_name = \"ollama\"\n\
             openai_protocol = \"chat_completions\"\n\
             {shared}",
            model.unwrap_or("llama3")
        ),
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

    if config.model.trim().is_empty() {
        report.fail("model is empty".to_string());
    } else {
        report.ok(format!("model: {}", config.model));
    }
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

    let auth = if config.is_anthropic() {
        match config.resolve_auth() {
            Some(api_key) => {
                report.ok("authentication: Anthropic API key resolved".to_string());
                Some(ProviderAuth::Anthropic(api_key))
            }
            None => {
                report.fail(
                    "authentication: no Anthropic credentials; set ANTHROPIC_API_KEY".to_string(),
                );
                None
            }
        }
    } else {
        let key = config.resolve_openai_key();
        let official_openai = config
            .openai_base_url
            .as_deref()
            .is_some_and(|url| url.contains("api.openai.com"));
        if official_openai && key.is_none() {
            report
                .fail("authentication: OPENAI_API_KEY is required for api.openai.com".to_string());
        } else if key.is_some() {
            report.ok("authentication: OpenAI-compatible API key resolved".to_string());
        } else {
            report.ok("authentication: endpoint configured without an API key".to_string());
        }
        Some(ProviderAuth::OpenAi(key.unwrap_or_default()))
    };

    if offline {
        report.warn("provider connectivity: skipped (--offline)".to_string());
    } else if let Some(auth) = auth {
        match check_provider(config, auth).await {
            Ok(status) => report.ok(format!("provider connectivity: HTTP {status}")),
            Err(error) => report.fail(format!("provider connectivity: {error}")),
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

enum ProviderAuth {
    Anthropic(AnthropicApiKey),
    OpenAi(String),
}

async fn check_provider(config: &Config, auth: ProviderAuth) -> Result<reqwest::StatusCode> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let request = match auth {
        ProviderAuth::Anthropic(api_key) => client
            .get("https://api.anthropic.com/v1/models?limit=1")
            .header("anthropic-version", "2023-06-01")
            .header("x-api-key", api_key.expose()),
        ProviderAuth::OpenAi(key) => {
            let base = config
                .openai_base_url
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("OpenAI-compatible base URL is missing"))?
                .trim_end_matches('/');
            let request = client.get(format!("{base}/models"));
            if key.is_empty() {
                request
            } else {
                request.bearer_auth(key)
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
            ConfigProvider::Ollama,
        ] {
            let parsed: Config = toml::from_str(&config_template(provider, None)).unwrap();
            assert!(!parsed.model.is_empty());
        }
    }

    #[test]
    fn init_refuses_overwrite_without_force() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("config.toml");
        std::fs::write(&path, "original").unwrap();

        assert!(
            init_config_at(&path, ConfigProvider::Anthropic, None, false)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        assert_eq!(std::fs::read_to_string(path).unwrap(), "original");
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
    async fn doctor_marks_invalid_configuration_unhealthy() {
        let config = Config {
            model: String::new(),
            auto_compact_threshold: 2.0,
            ..Config::default()
        };
        let report = doctor(&config, true).await;
        assert!(!report.healthy);
        assert!(report.text.contains("[fail] model is empty"));
        assert!(report.text.contains("auto_compact_threshold"));
    }
}
