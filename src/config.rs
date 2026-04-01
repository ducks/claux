use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::permissions::PermissionMode;

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

impl Default for Config {
    fn default() -> Self {
        Self {
            model: default_model(),
            api_key: None,
            api_key_env: default_api_key_env(),
            api_key_cmd: None,
            permission_mode: PermissionMode::Default,
            max_tokens: default_max_tokens(),
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
                    if let Ok(m) = serde_json::from_value(serde_json::Value::String(mode.to_string())) {
                        config.permission_mode = m;
                    }
                }
            }
        }

        Ok(config)
    }

    pub fn resolve_api_key(&self) -> Option<String> {
        // Direct value
        if let Some(ref key) = self.api_key {
            if !key.is_empty() {
                return Some(key.clone());
            }
        }

        // Command
        if let Some(ref cmd) = self.api_key_cmd {
            if let Ok(output) = std::process::Command::new("sh").arg("-c").arg(cmd).output() {
                if output.status.success() {
                    let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !key.is_empty() {
                        return Some(key);
                    }
                }
            }
        }

        // Environment variable
        if let Ok(key) = std::env::var(&self.api_key_env) {
            if !key.is_empty() {
                return Some(key);
            }
        }

        None
    }

    fn global_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("claude-rs")
            .join("config.toml")
    }

    fn project_path() -> Option<PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let path = cwd.join(".claude-rs.toml");
        Some(path)
    }
}
