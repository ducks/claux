use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::command_sandbox::BashFilesystemPolicy;
use crate::cost::ModelPricing;
use crate::permissions::PermissionMode;
use crate::sandbox::NativeToolFilesystemPolicy;

mod trust;

pub use trust::ProjectTrust;

/// An API key used to authenticate with Anthropic.
#[derive(Debug, Clone)]
pub struct AnthropicApiKey(String);

impl AnthropicApiKey {
    pub fn new(key: String) -> Self {
        Self(key)
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIProtocol {
    #[default]
    ChatCompletions,
    Responses,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    Openai,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub kind: ProviderKind,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub protocol: OpenAIProtocol,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub api_key_cmd: Option<String>,
    /// Ask compatible APIs to cache the stable prompt prefix across turns.
    #[serde(default)]
    pub prompt_caching: bool,
    /// Accept a clean SSE EOF even when a compatible endpoint omits both a
    /// finish reason and the `[DONE]` marker. Strict by default because EOF
    /// alone cannot prove that a response was complete.
    #[serde(default)]
    pub allow_eof_without_finish_reason: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Override the built-in context window for this provider/model profile.
    #[serde(default)]
    pub context_window: Option<usize>,
    /// Override built-in pricing in USD per million tokens.
    #[serde(default)]
    pub pricing: Option<ModelPricing>,
}

/// Credential-free provider/model identity persisted with each session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelBinding {
    pub profile: String,
    pub display_name: String,
    pub provider: String,
    pub provider_kind: ProviderKind,
    pub provider_name: String,
    pub model: String,
    pub base_url: Option<String>,
    pub protocol: OpenAIProtocol,
    pub api_key_env: String,
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub prompt_caching: bool,
    #[serde(default)]
    pub allow_eof_without_finish_reason: bool,
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub binding: ModelBinding,
    pub metadata: crate::model::ModelMetadata,
    pub api_key: Option<String>,
    pub api_key_cmd: Option<String>,
}

impl ResolvedModel {
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(key) = self.api_key.as_deref().filter(|key| !key.is_empty()) {
            return Some(key.to_string());
        }
        if let Some(command) = self.api_key_cmd.as_deref() {
            match std::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .output()
            {
                Ok(output) if output.status.success() => {
                    let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !key.is_empty() {
                        return Some(key);
                    }
                }
                Ok(output) => tracing::warn!(
                    "credential command failed ({}): {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
                Err(error) => tracing::warn!("credential command failed to start: {error}"),
            }
        }
        std::env::var(&self.binding.api_key_env)
            .ok()
            .filter(|key| !key.is_empty())
    }

    pub fn requires_api_key(&self) -> bool {
        match self.binding.provider_kind {
            ProviderKind::Anthropic => true,
            ProviderKind::Openai => {
                matches!(
                    self.binding.provider_name.to_ascii_lowercase().as_str(),
                    "openai" | "openrouter"
                ) || self.binding.base_url.as_deref().is_some_and(|url| {
                    url.contains("api.openai.com") || url.contains("openrouter.ai")
                })
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub model: String,

    /// Additional model IDs offered when creating a TUI session.
    #[serde(default)]
    pub models: Vec<String>,

    /// Named providers used by model profiles.
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,

    /// Named choices displayed by the TUI new-session picker.
    #[serde(default)]
    pub model_profiles: HashMap<String, ModelProfile>,

    /// Default named profile for new sessions.
    #[serde(default)]
    pub default_profile: Option<String>,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,

    #[serde(default)]
    pub api_key_cmd: Option<String>,

    #[serde(default)]
    pub permission_mode: PermissionMode,

    /// Filesystem containment for built-in Read/Write/Edit/Glob/Grep tools.
    /// Bash and MCP tools are governed by separate permission boundaries.
    #[serde(default)]
    pub native_tool_filesystem_policy: NativeToolFilesystemPolicy,

    /// Operating-system filesystem containment for Bash commands.
    #[serde(default)]
    pub bash_filesystem_policy: BashFilesystemPolicy,

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

    /// Environment variable containing the OpenAI-compatible API key
    #[serde(default = "default_openai_api_key_env")]
    pub openai_api_key_env: String,

    /// Shell command that returns the OpenAI-compatible API key
    #[serde(default)]
    pub openai_api_key_cmd: Option<String>,

    /// Wire protocol used by the OpenAI provider.
    #[serde(default)]
    pub openai_protocol: OpenAIProtocol,

    /// Optional reasoning effort for OpenAI-compatible APIs.
    #[serde(default)]
    pub openai_reasoning_effort: Option<String>,

    /// Legacy equivalent of providers.<name>.allow_eof_without_finish_reason.
    #[serde(default)]
    pub openai_allow_eof_without_finish_reason: bool,

    /// Per-model pricing overrides in USD per million tokens.
    #[serde(default)]
    pub model_pricing: std::collections::HashMap<String, ModelPricing>,

    /// Display name for the provider (e.g. "ollama", "openai", "lmstudio")
    #[serde(default)]
    pub openai_provider_name: Option<String>,

    /// Plugin configuration
    #[serde(default)]
    pub plugins: Vec<PluginConfig>,

    /// MCP server configuration
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,

    /// Project directories whose local configuration and MCP servers are
    /// explicitly trusted. This field is read only from the global config.
    #[serde(default)]
    pub trusted_projects: Vec<PathBuf>,

    /// Resolved trust for the current working directory. Runtime-only.
    #[serde(skip)]
    pub project_trust: Option<ProjectTrust>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
}

/// The .mcp.json format matching Claude Code's schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpJsonConfig {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: std::collections::HashMap<String, McpJsonServerEntry>,
}

/// A single server entry in .mcp.json (name comes from the key).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpJsonServerEntry {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
}

impl McpJsonServerEntry {
    pub fn into_server_config(self, name: String) -> McpServerConfig {
        McpServerConfig {
            name,
            command: self.command,
            args: self.args,
            env: self.env,
        }
    }
}

/// Load MCP servers from .mcp.json in the current directory (CC format).
pub fn load_mcp_json(trust: &ProjectTrust) -> Vec<McpServerConfig> {
    if !trust.is_trusted() {
        return Vec::new();
    }

    let path = trust.project_file(".mcp.json");

    if !path.exists() {
        return Vec::new();
    }

    match std::fs::read_to_string(&path) {
        Ok(content) => match serde_json::from_str::<McpJsonConfig>(&content) {
            Ok(config) => config
                .mcp_servers
                .into_iter()
                .map(|(name, entry)| entry.into_server_config(name))
                .collect(),
            Err(e) => {
                tracing::warn!("Failed to parse .mcp.json: {e}");
                Vec::new()
            }
        },
        Err(e) => {
            tracing::warn!("Failed to read .mcp.json: {e}");
            Vec::new()
        }
    }
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
    /// Fires when an agent turn completes and control returns to the user.
    OnTurnEnd,
    /// Fires when the agent blocks on a user decision (a permission prompt).
    OnPermissionRequest,
}

fn default_trigger() -> HookTrigger {
    HookTrigger::OnContextBuild
}

fn default_model() -> String {
    "claude-sonnet-5".to_string()
}

fn default_api_key_env() -> String {
    "ANTHROPIC_API_KEY".to_string()
}

fn default_openai_api_key_env() -> String {
    "OPENAI_API_KEY".to_string()
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
            models: Vec::new(),
            providers: HashMap::new(),
            model_profiles: HashMap::new(),
            default_profile: None,
            api_key: None,
            api_key_env: default_api_key_env(),
            api_key_cmd: None,
            permission_mode: PermissionMode::Default,
            native_tool_filesystem_policy: NativeToolFilesystemPolicy::default(),
            bash_filesystem_policy: BashFilesystemPolicy::default(),
            max_tokens: default_max_tokens(),
            auto_compact_threshold: default_auto_compact_threshold(),
            openai_base_url: None,
            openai_api_key: None,
            openai_api_key_env: default_openai_api_key_env(),
            openai_api_key_cmd: None,
            openai_protocol: OpenAIProtocol::default(),
            openai_reasoning_effort: None,
            openai_allow_eof_without_finish_reason: false,
            model_pricing: std::collections::HashMap::new(),
            openai_provider_name: None,
            plugins: Vec::new(),
            mcp_servers: Vec::new(),
            trusted_projects: Vec::new(),
            project_trust: None,
        }
    }
}

impl Config {
    /// Models available for new TUI sessions, with the default first.
    pub fn available_models(&self) -> Vec<String> {
        let mut models = Vec::with_capacity(self.models.len() + 1);
        for model in std::iter::once(&self.model).chain(&self.models) {
            let model = model.trim();
            if !model.is_empty() && !models.iter().any(|existing| existing == model) {
                models.push(model.to_string());
            }
        }
        models
    }

    /// Resolve every selectable model, putting the configured default first.
    pub fn selectable_models(&self) -> Result<Vec<ResolvedModel>> {
        if self.model_profiles.is_empty() {
            return self
                .available_models()
                .into_iter()
                .map(|model| self.resolve_legacy_model(&model))
                .collect();
        }

        let mut names = self.model_profiles.keys().cloned().collect::<Vec<_>>();
        names.sort();
        if let Some(default) = self.default_profile.as_deref() {
            if !self.model_profiles.contains_key(default) {
                anyhow::bail!("default_profile '{default}' is not defined in model_profiles");
            }
            names.retain(|name| name != default);
            names.insert(0, default.to_string());
        }
        names
            .into_iter()
            .map(|name| self.resolve_profile(&name))
            .collect()
    }

    pub fn default_resolved_model(&self) -> Result<ResolvedModel> {
        self.selectable_models()?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no models configured"))
    }

    /// Resolve a profile name first, then an exact configured model ID.
    pub fn resolve_model(&self, name_or_model: &str) -> Result<ResolvedModel> {
        if self.model_profiles.contains_key(name_or_model) {
            return self.resolve_profile(name_or_model);
        }
        let matches = self
            .selectable_models()?
            .into_iter()
            .filter(|resolved| resolved.binding.model == name_or_model)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [resolved] => Ok(resolved.clone()),
            [] if self.model_profiles.is_empty() => self.resolve_legacy_model(name_or_model),
            [] => anyhow::bail!(
                "model/profile '{name_or_model}' is not configured; choose one of: {}",
                self.model_profiles
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            _ => anyhow::bail!(
                "model ID '{name_or_model}' is configured by multiple profiles; use a profile name"
            ),
        }
    }

    pub fn resolve_binding(&self, binding: &ModelBinding) -> Result<ResolvedModel> {
        let provider = self
            .providers
            .get(&binding.provider)
            .filter(|provider| provider.kind == binding.provider_kind);
        let profile = self.model_profiles.get(&binding.profile).filter(|profile| {
            profile.provider == binding.provider && profile.model == binding.model
        });
        let (api_key, api_key_cmd) = provider
            .map(|provider| (provider.api_key.clone(), provider.api_key_cmd.clone()))
            .unwrap_or((None, None));
        Ok(ResolvedModel {
            binding: binding.clone(),
            metadata: self.resolve_metadata(profile, &binding.model),
            api_key,
            api_key_cmd,
        })
    }

    fn resolve_profile(&self, name: &str) -> Result<ResolvedModel> {
        let profile = self
            .model_profiles
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model profile '{name}' is not configured"))?;
        let provider = self.providers.get(&profile.provider).ok_or_else(|| {
            anyhow::anyhow!(
                "model profile '{name}' references missing provider '{}'",
                profile.provider
            )
        })?;
        let default_env = match provider.kind {
            ProviderKind::Anthropic => "ANTHROPIC_API_KEY",
            ProviderKind::Openai => "OPENAI_API_KEY",
        };
        let base_url = match provider.kind {
            // Anthropic-compatible gateways may expose the Messages API at a
            // custom root. A missing URL retains the official Anthropic API.
            ProviderKind::Anthropic => provider.base_url.clone(),
            ProviderKind::Openai => Some(provider.base_url.clone().ok_or_else(|| {
                anyhow::anyhow!("provider '{}' is missing base_url", profile.provider)
            })?),
        };
        Ok(ResolvedModel {
            binding: ModelBinding {
                profile: name.to_string(),
                display_name: profile
                    .display_name
                    .clone()
                    .unwrap_or_else(|| profile.model.clone()),
                provider: profile.provider.clone(),
                provider_kind: provider.kind,
                provider_name: provider
                    .name
                    .clone()
                    .unwrap_or_else(|| profile.provider.clone()),
                model: profile.model.clone(),
                base_url,
                protocol: provider.protocol,
                api_key_env: provider
                    .api_key_env
                    .clone()
                    .unwrap_or_else(|| default_env.to_string()),
                reasoning_effort: profile.reasoning_effort.clone(),
                prompt_caching: provider.prompt_caching,
                allow_eof_without_finish_reason: provider.allow_eof_without_finish_reason,
            },
            metadata: self.resolve_metadata(Some(profile), &profile.model),
            api_key: provider.api_key.clone(),
            api_key_cmd: provider.api_key_cmd.clone(),
        })
    }

    fn resolve_legacy_model(&self, model: &str) -> Result<ResolvedModel> {
        let (provider_kind, provider, provider_name, base_url, protocol, key_env, key, key_cmd) =
            if let Some(base_url) = self.openai_base_url.clone() {
                (
                    ProviderKind::Openai,
                    "legacy".to_string(),
                    self.openai_provider_name
                        .clone()
                        .unwrap_or_else(|| "openai".to_string()),
                    Some(base_url),
                    self.openai_protocol,
                    self.openai_api_key_env.clone(),
                    self.openai_api_key.clone(),
                    self.openai_api_key_cmd.clone(),
                )
            } else {
                (
                    ProviderKind::Anthropic,
                    "legacy".to_string(),
                    "anthropic".to_string(),
                    None,
                    OpenAIProtocol::ChatCompletions,
                    self.api_key_env.clone(),
                    self.api_key.clone(),
                    self.api_key_cmd.clone(),
                )
            };
        Ok(ResolvedModel {
            binding: ModelBinding {
                profile: format!("legacy:{model}"),
                display_name: model.to_string(),
                provider,
                provider_kind,
                provider_name,
                model: model.to_string(),
                base_url,
                protocol,
                api_key_env: key_env,
                reasoning_effort: self.openai_reasoning_effort.clone(),
                prompt_caching: false,
                allow_eof_without_finish_reason: self.openai_allow_eof_without_finish_reason,
            },
            metadata: self.resolve_metadata(None, model),
            api_key: key,
            api_key_cmd: key_cmd,
        })
    }

    fn resolve_metadata(
        &self,
        profile: Option<&ModelProfile>,
        model: &str,
    ) -> crate::model::ModelMetadata {
        let legacy_pricing = self.model_pricing.get(model).copied();
        crate::model::built_in_metadata(model).with_overrides(
            profile.and_then(|profile| profile.context_window),
            profile
                .and_then(|profile| profile.pricing)
                .or(legacy_pricing),
        )
    }

    /// Whether the configured OpenAI-compatible endpoint is a hosted
    /// provider that cannot be used anonymously.
    #[cfg(test)]
    pub fn openai_requires_api_key(&self) -> bool {
        self.openai_provider_name.as_deref().is_some_and(|name| {
            matches!(name.to_ascii_lowercase().as_str(), "openai" | "openrouter")
        }) || self
            .openai_base_url
            .as_deref()
            .is_some_and(|url| url.contains("api.openai.com") || url.contains("openrouter.ai"))
    }

    /// Whether the current project is trusted. Falls back to `false` (the
    /// fail-closed default) when trust has not been resolved, which only
    /// happens for configs built outside `Config::load`.
    pub fn is_project_trusted(&self) -> bool {
        self.project_trust
            .as_ref()
            .map(ProjectTrust::is_trusted)
            .unwrap_or(false)
    }

    pub fn load(force_project_trust: bool) -> Result<Self> {
        let global_path = Self::global_path();

        let mut config = if global_path.exists() {
            let text = std::fs::read_to_string(&global_path)?;
            toml::from_str(&text)?
        } else {
            Self::default()
        };

        let trust = ProjectTrust::resolve(force_project_trust, &config.trusted_projects);

        // Layer project config on top
        let project_path = trust.project_file(".claux.toml");
        if project_path.exists() {
            let text = std::fs::read_to_string(project_path)?;
            let project: toml::Value = toml::from_str(&text)?;
            apply_project_overrides(&mut config, &project, trust.is_trusted());
        }
        config.project_trust = Some(trust);

        Ok(config)
    }

    pub fn global_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("claux")
            .join("config.toml")
    }
}

fn apply_project_overrides(config: &mut Config, project: &toml::Value, trusted: bool) {
    if let Some(model) = project.get("model").and_then(|v| v.as_str()) {
        if config.model_profiles.is_empty() {
            config.model = model.to_string();
        } else if config.model_profiles.contains_key(model) {
            config.default_profile = Some(model.to_string());
        } else {
            let matches = config
                .model_profiles
                .iter()
                .filter(|(_, profile)| profile.model == model)
                .map(|(name, _)| name)
                .collect::<Vec<_>>();
            if let [name] = matches.as_slice() {
                config.default_profile = Some((*name).clone());
            } else {
                tracing::warn!(
                    "Ignoring project model={model:?}: use a unique configured model profile"
                );
            }
        }
    }
    if let Some(mode) = project.get("permission_mode").and_then(|v| v.as_str()) {
        if let Ok(requested) =
            serde_json::from_value::<PermissionMode>(serde_json::Value::String(mode.to_string()))
        {
            if trust::permits_permission_override(config.permission_mode, requested, trusted) {
                config.permission_mode = requested;
            } else {
                tracing::warn!(
                    "Ignoring project permission_mode={mode:?}: it would loosen the global policy; \
                     pass --trust-project or add this directory to trusted_projects"
                );
            }
        }
    }
    if let Some(policy) = project
        .get("native_tool_filesystem_policy")
        .and_then(|value| value.as_str())
    {
        if let Ok(requested) = serde_json::from_value::<NativeToolFilesystemPolicy>(
            serde_json::Value::String(policy.to_string()),
        ) {
            if config
                .native_tool_filesystem_policy
                .permits_project_override(requested, trusted)
            {
                config.native_tool_filesystem_policy = requested;
            } else {
                tracing::warn!(
                    "Ignoring project native_tool_filesystem_policy={policy:?}: it would loosen \
                     the global policy; pass --trust-project or add this directory to \
                     trusted_projects"
                );
            }
        }
    }
    if let Some(policy) = project
        .get("bash_filesystem_policy")
        .and_then(|value| value.as_str())
    {
        if let Ok(requested) = serde_json::from_value::<BashFilesystemPolicy>(
            serde_json::Value::String(policy.to_string()),
        ) {
            if config
                .bash_filesystem_policy
                .permits_project_override(requested, trusted)
            {
                config.bash_filesystem_policy = requested;
            } else {
                tracing::warn!(
                    "Ignoring project bash_filesystem_policy={policy:?}: it would loosen the \
                     global policy; pass --trust-project or add this directory to \
                     trusted_projects"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_defaults_to_standard_environment_and_legacy_protocol() {
        let config = Config::default();
        assert_eq!(config.openai_api_key_env, "OPENAI_API_KEY");
        assert_eq!(config.openai_protocol, OpenAIProtocol::ChatCompletions);
    }

    #[test]
    fn available_models_puts_default_first_and_deduplicates() {
        let config = Config {
            model: "primary".to_string(),
            models: vec![
                "fast".to_string(),
                "primary".to_string(),
                "  review  ".to_string(),
                String::new(),
            ],
            ..Config::default()
        };

        assert_eq!(config.available_models(), vec!["primary", "fast", "review"]);
    }

    #[test]
    fn named_profiles_resolve_across_providers_with_default_first() {
        let config: Config = toml::from_str(
            r#"
            default_profile = "sonnet"

            [providers.anthropic]
            type = "anthropic"
            base_url = "https://gateway.example/v1"
            api_key_env = "ANTHROPIC_TEST_KEY"

            [providers.openrouter]
            type = "openai"
            base_url = "https://openrouter.ai/api/v1"
            name = "openrouter"
            api_key_env = "OPENROUTER_TEST_KEY"
            prompt_caching = true
            allow_eof_without_finish_reason = true

            [model_profiles.sonnet]
            provider = "anthropic"
            model = "claude-sonnet"
            display_name = "Sonnet"

            [model_profiles.gpt]
            provider = "openrouter"
            model = "openai/gpt"
            display_name = "GPT via OpenRouter"
            "#,
        )
        .unwrap();

        let models = config.selectable_models().unwrap();
        assert_eq!(models[0].binding.profile, "sonnet");
        assert_eq!(models[0].binding.provider_kind, ProviderKind::Anthropic);
        assert_eq!(
            models[0].binding.base_url.as_deref(),
            Some("https://gateway.example/v1")
        );
        assert_eq!(models[1].binding.profile, "gpt");
        assert_eq!(
            models[1].binding.base_url.as_deref(),
            Some("https://openrouter.ai/api/v1")
        );
        assert!(models[1].binding.prompt_caching);
        assert!(models[1].binding.allow_eof_without_finish_reason);
    }

    #[test]
    fn saved_binding_keeps_transport_but_refreshes_credentials() {
        let mut config: Config = toml::from_str(
            r#"
            [providers.local]
            type = "openai"
            base_url = "http://old.invalid/v1"
            api_key = "new-secret"

            [model_profiles.local]
            provider = "local"
            model = "coder"
            context_window = 64000
            "#,
        )
        .unwrap();
        let mut binding = config.resolve_model("local").unwrap().binding;
        binding.base_url = Some("http://saved.invalid/v1".to_string());
        config.providers.get_mut("local").unwrap().base_url =
            Some("http://changed.invalid/v1".to_string());
        config
            .model_profiles
            .get_mut("local")
            .unwrap()
            .context_window = Some(32_000);

        let restored = config.resolve_binding(&binding).unwrap();
        assert_eq!(
            restored.binding.base_url.as_deref(),
            Some("http://saved.invalid/v1")
        );
        assert_eq!(restored.api_key.as_deref(), Some("new-secret"));
        assert_eq!(restored.metadata.context_window, 32_000);
        assert!(!serde_json::to_string(&binding)
            .unwrap()
            .contains("new-secret"));
    }

    #[test]
    fn duplicate_model_ids_require_profile_name() {
        let config: Config = toml::from_str(
            r#"
            [providers.a]
            type = "anthropic"
            [providers.b]
            type = "anthropic"
            [model_profiles.first]
            provider = "a"
            model = "same"
            [model_profiles.second]
            provider = "b"
            model = "same"
            "#,
        )
        .unwrap();

        assert!(config.resolve_model("same").is_err());
        assert_eq!(config.resolve_model("first").unwrap().binding.provider, "a");
    }

    #[test]
    fn hosted_openai_compatible_providers_require_keys() {
        for (name, url) in [
            ("openai", "https://api.openai.com/v1"),
            ("openrouter", "https://openrouter.ai/api/v1"),
        ] {
            let config = Config {
                openai_provider_name: Some(name.to_string()),
                openai_base_url: Some(url.to_string()),
                ..Config::default()
            };
            assert!(config.openai_requires_api_key());
        }

        let ollama = Config {
            openai_provider_name: Some("ollama".to_string()),
            openai_base_url: Some("http://localhost:11434/v1".to_string()),
            ..Config::default()
        };
        assert!(!ollama.openai_requires_api_key());
    }

    #[test]
    fn parses_model_pricing_override() {
        let config: Config = toml::from_str(
            r#"
            [model_pricing."private-coder"]
            input = 0.5
            output = 1.5
            "#,
        )
        .unwrap();
        let pricing = config.model_pricing["private-coder"];
        assert_eq!(pricing.input, 0.5);
        assert_eq!(pricing.output, 1.5);
        assert_eq!(pricing.cache_read, 0.0);
    }

    #[test]
    fn profile_metadata_overrides_built_ins_and_legacy_pricing() {
        let config: Config = toml::from_str(
            r#"
            [providers.openrouter]
            type = "openai"
            base_url = "https://openrouter.ai/api/v1"

            [model_pricing."claude-sonnet"]
            input = 0.8
            output = 1.8

            [model_profiles.custom]
            provider = "openrouter"
            model = "claude-sonnet"
            context_window = 64000

            [model_profiles.custom.pricing]
            input = 0.5
            output = 1.5
            cache_read = 0.05
            cache_write = 0.625
            "#,
        )
        .unwrap();

        let resolved = config.resolve_model("custom").unwrap();
        assert_eq!(resolved.metadata.context_window, 64_000);
        assert_eq!(
            resolved.metadata.pricing,
            Some(ModelPricing {
                input: 0.5,
                output: 1.5,
                cache_read: 0.05,
                cache_write: 0.625,
            })
        );
    }

    #[test]
    fn legacy_pricing_combines_with_built_in_context_window() {
        let config: Config = toml::from_str(
            r#"
            model = "claude-sonnet"

            [model_pricing."claude-sonnet"]
            input = 0.8
            output = 1.8
            "#,
        )
        .unwrap();

        let resolved = config.resolve_model("claude-sonnet").unwrap();
        assert_eq!(resolved.metadata.context_window, 200_000);
        assert_eq!(resolved.metadata.pricing.unwrap().input, 0.8);
    }

    #[test]
    fn untrusted_project_permission_can_only_tighten() {
        let mut config = Config {
            permission_mode: PermissionMode::Default,
            ..Config::default()
        };
        let project: toml::Value =
            toml::from_str("permission_mode = \"bypass\"\nmodel = \"project-model\"").unwrap();

        apply_project_overrides(&mut config, &project, false);

        assert_eq!(config.permission_mode, PermissionMode::Default);
        assert_eq!(config.model, "project-model");

        let project: toml::Value = toml::from_str("permission_mode = \"plan\"").unwrap();
        apply_project_overrides(&mut config, &project, false);
        assert_eq!(config.permission_mode, PermissionMode::Plan);
    }

    #[test]
    fn untrusted_project_native_tool_policy_can_only_tighten() {
        let mut config = Config {
            native_tool_filesystem_policy: NativeToolFilesystemPolicy::WorkspaceOnly,
            ..Config::default()
        };
        let project: toml::Value =
            toml::from_str("native_tool_filesystem_policy = \"unrestricted\"").unwrap();

        apply_project_overrides(&mut config, &project, false);
        assert_eq!(
            config.native_tool_filesystem_policy,
            NativeToolFilesystemPolicy::WorkspaceOnly
        );

        config.native_tool_filesystem_policy = NativeToolFilesystemPolicy::Unrestricted;
        let project: toml::Value =
            toml::from_str("native_tool_filesystem_policy = \"workspace_only\"").unwrap();
        apply_project_overrides(&mut config, &project, false);
        assert_eq!(
            config.native_tool_filesystem_policy,
            NativeToolFilesystemPolicy::WorkspaceOnly
        );
    }

    #[test]
    fn trusted_project_native_tool_policy_may_loosen() {
        let mut config = Config {
            native_tool_filesystem_policy: NativeToolFilesystemPolicy::WorkspaceOnly,
            ..Config::default()
        };
        let project: toml::Value =
            toml::from_str("native_tool_filesystem_policy = \"unrestricted\"").unwrap();

        apply_project_overrides(&mut config, &project, true);

        assert_eq!(
            config.native_tool_filesystem_policy,
            NativeToolFilesystemPolicy::Unrestricted
        );
    }

    #[test]
    fn untrusted_project_bash_policy_can_only_tighten() {
        let mut config = Config {
            bash_filesystem_policy: BashFilesystemPolicy::Auto,
            ..Config::default()
        };
        let project: toml::Value =
            toml::from_str("bash_filesystem_policy = \"unrestricted\"").unwrap();

        apply_project_overrides(&mut config, &project, false);
        assert_eq!(config.bash_filesystem_policy, BashFilesystemPolicy::Auto);

        let project: toml::Value =
            toml::from_str("bash_filesystem_policy = \"workspace_write\"").unwrap();
        apply_project_overrides(&mut config, &project, false);
        assert_eq!(
            config.bash_filesystem_policy,
            BashFilesystemPolicy::WorkspaceWrite
        );
    }

    #[test]
    fn trusted_project_bash_policy_may_loosen() {
        let mut config = Config {
            bash_filesystem_policy: BashFilesystemPolicy::WorkspaceWrite,
            ..Config::default()
        };
        let project: toml::Value =
            toml::from_str("bash_filesystem_policy = \"unrestricted\"").unwrap();

        apply_project_overrides(&mut config, &project, true);

        assert_eq!(
            config.bash_filesystem_policy,
            BashFilesystemPolicy::Unrestricted
        );
    }

    #[test]
    fn project_model_selects_a_named_profile() {
        let mut config: Config = toml::from_str(
            r#"
            [providers.p]
            type = "anthropic"
            [model_profiles.fast]
            provider = "p"
            model = "fast-id"
            [model_profiles.deep]
            provider = "p"
            model = "deep-id"
            "#,
        )
        .unwrap();
        let project: toml::Value = toml::from_str("model = \"deep\"").unwrap();

        apply_project_overrides(&mut config, &project, false);

        assert_eq!(config.default_profile.as_deref(), Some("deep"));
    }

    #[test]
    fn trusted_project_permission_may_loosen() {
        let mut config = Config {
            permission_mode: PermissionMode::Plan,
            ..Config::default()
        };
        let project: toml::Value = toml::from_str("permission_mode = \"bypass\"").unwrap();

        apply_project_overrides(&mut config, &project, true);

        assert_eq!(config.permission_mode, PermissionMode::Bypass);
    }

    #[test]
    fn untrusted_project_does_not_load_mcp_json() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join(".mcp.json"),
            r#"{"mcpServers":{"evil":{"command":"false"}}}"#,
        )
        .unwrap();
        let trust = ProjectTrust::for_test(temp.path().to_path_buf(), false);

        assert!(load_mcp_json(&trust).is_empty());
    }

    #[test]
    fn trusted_project_loads_mcp_json() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join(".mcp.json"),
            r#"{"mcpServers":{"safe":{"command":"true"}}}"#,
        )
        .unwrap();
        let trust = ProjectTrust::for_test(temp.path().to_path_buf(), true);

        let servers = load_mcp_json(&trust);
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "safe");
    }

    #[test]
    fn is_project_trusted_reflects_resolved_trust() {
        let mut config = Config::default();

        // Before trust is resolved, the fail-closed default is untrusted.
        assert!(!config.is_project_trusted());

        config.project_trust = Some(ProjectTrust::for_test(PathBuf::from("."), true));
        assert!(config.is_project_trusted());

        config.project_trust = Some(ProjectTrust::for_test(PathBuf::from("."), false));
        assert!(!config.is_project_trusted());
    }
}
