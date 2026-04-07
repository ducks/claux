use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::permissions::PermissionMode;

/// How we authenticate with the API.
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// Direct API key (x-api-key header)
    ApiKey(String),
    /// OAuth access token from `claude login` (Authorization: Bearer header)
    OAuthToken(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub model: String,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,

    #[serde(default)]
    pub api_key_cmd: Option<String>,

    #[serde(default)]
    pub permission_mode: PermissionMode,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Auto-compact threshold (0.0-1.0). If conversation exceeds this
    /// fraction of the context window, auto-compact before next request.
    /// Set to 0.0 to disable auto-compact.
    #[serde(default = "default_auto_compact_threshold")]
    pub auto_compact_threshold: f64,

    /// OpenAI-compatible endpoint (e.g. "http://localhost:11434/v1")
    #[serde(default)]
    pub openai_base_url: Option<String>,

    /// API key for the OpenAI-compatible endpoint
    #[serde(default)]
    pub openai_api_key: Option<String>,

    /// Shell command that returns the OpenAI-compatible API key
    #[serde(default)]
    pub openai_api_key_cmd: Option<String>,

    /// Display name for the provider (e.g. "ollama", "openai", "lmstudio")
    #[serde(default)]
    pub openai_provider_name: Option<String>,

    /// Plugin configuration
    #[serde(default)]
    pub plugins: Vec<PluginConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_trigger")]
    pub trigger: HookTrigger,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HookTrigger {
    #[default]
    OnContextBuild,
    OnToolStart,
    OnToolComplete,
    OnSessionStart,
}

fn default_trigger() -> HookTrigger {
    HookTrigger::OnContextBuild
}

fn default_model() -> String {
    "claude-sonnet-4-20250514".to_string()
}

fn default_api_key_env() -> String {
    "ANTHROPIC_API_KEY".to_string()
}

fn default_max_tokens() -> u32 {
    16384
}

fn default_auto_compact_threshold() -> f64 {
    0.8 // 80% of context window
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: default_model(),
            api_key: None,
            api_key_env: default_api_key_env(),
            api_key_cmd: None,
            permission_mode: PermissionMode::Default,
            max_tokens: default_max_tokens(),
            auto_compact_threshold: default_auto_compact_threshold(),
            openai_base_url: None,
            openai_api_key: None,
            openai_api_key_cmd: None,
            openai_provider_name: None,
            plugins: Vec::new(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let global_path = Self::global_path();
        let project_path = Self::project_path();

        let mut config = if global_path.exists() {
            let text = std::fs::read_to_string(&global_path)?;
            toml::from_str(&text)?
        } else {
            Self::default()
        };

        // Layer project config on top
        if let Some(ref path) = project_path {
            if path.exists() {
                let text = std::fs::read_to_string(path)?;
                let project: toml::Value = toml::from_str(&text)?;

                if let Some(model) = project.get("model").and_then(|v| v.as_str()) {
                    config.model = model.to_string();
                }
                if let Some(mode) = project.get("permission_mode").and_then(|v| v.as_str()) {
                    if let Ok(m) =
                        serde_json::from_value(serde_json::Value::String(mode.to_string()))
                    {
                        config.permission_mode = m;
                    }
                }
            }
        }

        Ok(config)
    }

    /// Resolve authentication. Priority:
    /// 1. Direct API key in config
    /// 2. API key from command
    /// 3. ANTHROPIC_API_KEY env var
    /// 4. OAuth token from ~/.claude/.credentials.json (claude login)
    pub fn resolve_auth(&self) -> Option<AuthMethod> {
        // Direct value
        if let Some(ref key) = self.api_key {
            if !key.is_empty() {
                return Some(AuthMethod::ApiKey(key.clone()));
            }
        }

        // Command
        if let Some(ref cmd) = self.api_key_cmd {
            if let Ok(output) = std::process::Command::new("sh").arg("-c").arg(cmd).output() {
                if output.status.success() {
                    let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !key.is_empty() {
                        return Some(AuthMethod::ApiKey(key));
                    }
                }
            }
        }

        // Environment variable
        if let Ok(key) = std::env::var(&self.api_key_env) {
            if !key.is_empty() {
                return Some(AuthMethod::ApiKey(key));
            }
        }

        // Fall back to Claude Code OAuth credentials
        if let Some(token) = Self::read_claude_oauth_token() {
            return Some(AuthMethod::OAuthToken(token));
        }

        None
    }

    /// Resolve the OpenAI API key: direct value, then command.
    pub fn resolve_openai_key(&self) -> Option<String> {
        if let Some(ref key) = self.openai_api_key {
            if !key.is_empty() {
                return Some(key.clone());
            }
        }

        if let Some(ref cmd) = self.openai_api_key_cmd {
            match std::process::Command::new("sh").arg("-c").arg(cmd).output() {
                Ok(output) => {
                    if output.status.success() {
                        let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        if !key.is_empty() {
                            tracing::debug!("openai_api_key_cmd succeeded, key len={}", key.len());
                            return Some(key);
                        }
                        tracing::warn!("openai_api_key_cmd returned empty output");
                    } else {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        tracing::warn!(
                            "openai_api_key_cmd failed ({}): {}",
                            output.status,
                            stderr.trim()
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("openai_api_key_cmd exec error: {}", e);
                }
            }
        }

        None
    }

    /// Read OAuth access token from ~/.claude/.credentials.json
    fn read_claude_oauth_token() -> Option<String> {
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join(".claude")
            .join(".credentials.json");

        let content = std::fs::read_to_string(&path).ok()?;
        let creds: serde_json::Value = serde_json::from_str(&content).ok()?;

        let oauth = creds.get("claudeAiOauth")?;

        // Check if token is expired (with 60s buffer)
        if let Some(expires_at) = oauth.get("expiresAt").and_then(|v| v.as_i64()) {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_millis() as i64;

            if now_ms > expires_at - 60_000 {
                tracing::warn!("Claude OAuth token is expired. Run `claude login` to refresh.");
                return None;
            }
        }

        oauth
            .get("accessToken")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn global_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("claux")
            .join("config.toml")
    }

    fn project_path() -> Option<PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let path = cwd.join(".claux.toml");
        Some(path)
    }
}
