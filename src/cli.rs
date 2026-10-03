pub use crate::providers::{AuthProvider, BuiltinProvider as ConfigProvider};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "claux")]
#[command(about = "claux — an open, hackable terminal AI coding assistant in Rust")]
pub struct Cli {
    /// Alternate configuration file (also CLAUX_CONFIG)
    #[arg(long, global = true, value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// Override the selected provider endpoint for this invocation
    #[arg(long, global = true)]
    pub base_url: Option<String>,
    /// Override the selected provider protocol
    #[arg(long, global = true, value_parser = ["chat_completions", "responses", "anthropic"])]
    pub protocol: Option<String>,
    #[arg(long, global = true)]
    pub reasoning_effort: Option<String>,
    /// Native filesystem containment (loosening requires project trust)
    #[arg(long, global = true, value_parser = ["workspace_only", "unrestricted"])]
    pub native_fs_policy: Option<String>,
    /// Bash filesystem containment (loosening requires project trust)
    #[arg(long, global = true, value_parser = ["auto", "workspace_write", "unrestricted"])]
    pub bash_fs_policy: Option<String>,
    #[command(subcommand)]
    pub command: Option<CliCommand>,

    /// One-shot prompt (non-interactive)
    #[arg(short = 'p', long = "print")]
    pub prompt: Option<String>,

    /// Attach an image to the one-shot prompt (repeatable; PNG, JPEG, GIF, or WebP)
    #[arg(long, value_name = "FILE", requires = "prompt")]
    pub image: Vec<PathBuf>,

    /// Output format for one-shot mode
    #[arg(long, value_enum, requires = "prompt")]
    pub output_format: Option<OutputFormat>,

    /// Checkpoint and write the complete one-shot transcript and tool trace
    ///
    /// The artifact can contain sensitive tool inputs and outputs.
    #[arg(long, value_name = "FILE", requires = "prompt")]
    pub transcript: Option<PathBuf>,

    /// Model to use
    #[arg(long)]
    pub model: Option<String>,

    /// Resume a previous session
    #[arg(long, conflicts_with_all = ["prompt", "tui"])]
    pub resume: Option<String>,

    /// Permission mode (default, accept-edits, auto, bypass, plan)
    #[arg(long)]
    pub permission_mode: Option<String>,

    /// Trust project-local configuration and MCP servers for this invocation
    #[arg(long)]
    pub trust_project: bool,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Debug output
    #[arg(long)]
    pub debug: bool,

    /// Use full-screen TUI instead of inline REPL
    #[arg(long)]
    pub tui: bool,
}

impl Cli {
    pub fn config_path(&self) -> Option<PathBuf> {
        self.config
            .clone()
            .or_else(|| std::env::var_os("CLAUX_CONFIG").map(PathBuf::from))
    }

    pub fn load_config(&self) -> anyhow::Result<crate::config::Config> {
        let path = self.config_path();
        let mut config = crate::config::Config::load(self.trust_project, path.as_deref())?;
        self.apply_overrides(&mut config)?;
        Ok(config)
    }

    fn apply_overrides(&self, config: &mut crate::config::Config) -> anyhow::Result<()> {
        if let Some(url) = &self.base_url {
            let parsed = reqwest::Url::parse(url)?;
            anyhow::ensure!(
                matches!(parsed.scheme(), "http" | "https") && parsed.host_str().is_some(),
                "--base-url must be an HTTP(S) URL"
            );
            anyhow::ensure!(
                parsed.username().is_empty() && parsed.password().is_none(),
                "--base-url must not contain credentials"
            );
        }
        let trusted = config.is_project_trusted();
        if let Some(policy) = &self.native_fs_policy {
            let requested = serde_json::from_value(serde_json::json!(policy))?;
            anyhow::ensure!(
                config
                    .native_tool_filesystem_policy
                    .permits_project_override(requested, trusted),
                "--native-fs-policy would loosen containment; use --trust-project explicitly"
            );
            config.native_tool_filesystem_policy = requested;
        }
        if let Some(policy) = &self.bash_fs_policy {
            let requested = serde_json::from_value(serde_json::json!(policy))?;
            anyhow::ensure!(
                config
                    .bash_filesystem_policy
                    .permits_project_override(requested, trusted),
                "--bash-fs-policy would loosen containment; use --trust-project explicitly"
            );
            config.bash_filesystem_policy = requested;
        }
        config.transport_overrides = crate::config::TransportOverrides {
            base_url: self.base_url.clone(),
            protocol: self.protocol.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
        };
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Subcommand)]
pub enum CliCommand {
    /// Export the original conversation archive, independently of active context
    Archive {
        /// Session ID or unique prefix
        session: String,
    },
    /// Authenticate claux with a model provider
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// Diagnose configuration, authentication, tools, and provider connectivity
    Doctor {
        /// Skip the provider network check
        #[arg(long)]
        offline: bool,
    },
    /// Show provider-reported account and key usage
    Usage {
        #[command(subcommand)]
        command: UsageCommand,
    },
    /// Compare OpenRouter models using native prompt-token counts
    #[command(name = "tokenizer-fingerprint")]
    TokenizerFingerprint {
        /// OpenRouter model identifiers to compare
        #[arg(required = true, num_args = 2..)]
        models: Vec<String>,

        /// Report format
        #[arg(long, value_enum, conflicts_with = "json")]
        format: Option<TokenizerOutputFormat>,

        /// Emit JSON (legacy shorthand for --format json)
        #[arg(long, conflicts_with = "format")]
        json: bool,

        /// Write the report atomically instead of printing it to stdout
        #[arg(long = "output", value_name = "FILE")]
        report_output: Option<PathBuf>,

        /// Reuse completed models from this exact corpus and model list
        #[arg(long = "resume")]
        resume_fingerprint: bool,
    },
    /// Manage claux configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Internal entry point used to apply an operating-system sandbox.
    #[command(name = "__sandbox-exec", hide = true)]
    SandboxExec {
        #[arg(long)]
        workspace: PathBuf,
        #[arg(long)]
        command: String,
    },
    /// Internal entry point used to verify Landlock enforcement.
    #[command(name = "__sandbox-probe", hide = true)]
    SandboxProbe,
}

#[derive(Subcommand)]
pub enum AuthCommand {
    /// Authorize claux and save the resulting credential
    Login {
        #[arg(value_enum)]
        provider: AuthProvider,

        /// Display a code to copy and paste instead of using a localhost callback
        #[arg(long)]
        headless: bool,

        /// Print the authorization URL without opening a browser
        #[arg(long)]
        no_browser: bool,
    },
    /// Report whether a saved credential is available
    Status {
        #[arg(value_enum)]
        provider: AuthProvider,
    },
    /// Remove a saved credential
    Logout {
        #[arg(value_enum)]
        provider: AuthProvider,
    },
    /// Print a saved credential for integration with another local tool
    #[command(hide = true)]
    Token {
        #[arg(value_enum)]
        provider: AuthProvider,
    },
}

#[derive(Subcommand)]
pub enum UsageCommand {
    /// Query a provider's read-only usage status endpoint
    Status {
        /// Provider name (defaults to openrouter)
        provider: Option<String>,

        /// Emit machine-readable JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum TokenizerOutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Add a secure provider and model profile
    Init {
        /// Provider profile to add
        #[arg(long, value_enum, default_value_t = ConfigProvider::Anthropic)]
        provider: ConfigProvider,

        /// Model identifier for the new profile
        #[arg(long)]
        model: Option<String>,

        /// Replace the existing configuration instead of adding to it
        #[arg(long)]
        force: bool,
    },
}

#[cfg(test)]
mod tests {
    #[test]
    fn resume_cannot_be_silently_ignored_by_another_mode() {
        assert!(Cli::try_parse_from(["claux", "--resume", "id", "--print", "hello"]).is_err());
        assert!(Cli::try_parse_from(["claux", "--resume", "id", "--tui"]).is_err());
    }
    #[test]
    fn transport_flags_override_without_rewriting_config() {
        let args = Cli::try_parse_from([
            "claux",
            "--base-url",
            "http://localhost:9000/v1",
            "--protocol",
            "responses",
            "--reasoning-effort",
            "low",
        ])
        .unwrap();
        let mut config = crate::config::Config::default();
        args.apply_overrides(&mut config).unwrap();
        let resolved = config.resolve_model(&config.model).unwrap();
        assert_eq!(
            resolved.binding.base_url.as_deref(),
            Some("http://localhost:9000/v1")
        );
        assert_eq!(
            resolved.binding.protocol,
            crate::config::OpenAIProtocol::Responses
        );
        assert_eq!(
            resolved.binding.provider_kind,
            crate::config::ProviderKind::Openai
        );
        assert_eq!(resolved.binding.reasoning_effort.as_deref(), Some("low"));
        assert!(!toml::to_string(&config).unwrap().contains("localhost:9000"));
    }

    #[test]
    fn filesystem_flags_cannot_loosen_untrusted_policy() {
        for flag in ["--native-fs-policy", "--bash-fs-policy"] {
            let args = Cli::try_parse_from(["claux", flag, "unrestricted"]).unwrap();
            assert!(args
                .apply_overrides(&mut crate::config::Config::default())
                .is_err());
        }
        let args = Cli::try_parse_from([
            "claux",
            "--native-fs-policy",
            "workspace_only",
            "--bash-fs-policy",
            "workspace_write",
        ])
        .unwrap();
        args.apply_overrides(&mut crate::config::Config::default())
            .unwrap();
    }

    #[test]
    fn explicit_config_file_is_required_and_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alternate.toml");
        let args = Cli::try_parse_from(["claux", "--config", path.to_str().unwrap()]).unwrap();
        assert!(args.load_config().is_err());
        std::fs::write(&path, "max_rounds = 17").unwrap();
        assert_eq!(args.load_config().unwrap().max_rounds, 17);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "max_rounds = 17");
    }
    #[test]
    fn archive_accepts_a_session_prefix() {
        let cli = Cli::try_parse_from(["claux", "archive", "20261002"]).unwrap();
        assert!(
            matches!(cli.command, Some(CliCommand::Archive { session }) if session == "20261002")
        );
    }
    use super::*;

    #[test]
    fn parses_doctor_offline() {
        let cli = Cli::try_parse_from(["claux", "doctor", "--offline"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(CliCommand::Doctor { offline: true })
        ));
    }

    #[test]
    fn parses_usage_status() {
        let cli =
            Cli::try_parse_from(["claux", "usage", "status", "openrouter", "--json"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(CliCommand::Usage {
                command: UsageCommand::Status {
                    provider: Some(ref provider),
                    json: true,
                },
            }) if provider == "openrouter"
        ));
    }

    #[test]
    fn parses_tokenizer_fingerprint() {
        let cli = Cli::try_parse_from([
            "claux",
            "tokenizer-fingerprint",
            "stealth/ox-alpha",
            "z-ai/glm-5.3",
            "--json",
        ])
        .unwrap();

        assert!(matches!(
            cli.command,
            Some(CliCommand::TokenizerFingerprint {
                models,
                format: None,
                json: true,
                report_output: None,
                resume_fingerprint: false
            })
                if models == ["stealth/ox-alpha", "z-ai/glm-5.3"]
        ));
    }

    #[test]
    fn parses_markdown_tokenizer_fingerprint() {
        let cli = Cli::try_parse_from([
            "claux",
            "tokenizer-fingerprint",
            "stealth/ox-alpha",
            "z-ai/glm-5.3",
            "--format",
            "markdown",
            "--output",
            "/tmp/fingerprint.md",
            "--resume",
        ])
        .unwrap();

        assert!(matches!(
            cli.command,
            Some(CliCommand::TokenizerFingerprint {
                format: Some(TokenizerOutputFormat::Markdown),
                json: false,
                report_output: Some(ref output),
                resume_fingerprint: true,
                ..
            }) if output == &PathBuf::from("/tmp/fingerprint.md")
        ));
    }

    #[test]
    fn parses_headless_openrouter_login() {
        let cli = Cli::try_parse_from([
            "claux",
            "auth",
            "login",
            "openrouter",
            "--headless",
            "--no-browser",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(CliCommand::Auth {
                command: AuthCommand::Login {
                    provider: AuthProvider(ConfigProvider::OpenRouter),
                    headless: true,
                    no_browser: true,
                },
            })
        ));
    }

    #[test]
    fn parses_api_key_provider_logins() {
        let opencode = Cli::try_parse_from(["claux", "auth", "login", "opencode-go"]).unwrap();
        assert!(matches!(
            opencode.command,
            Some(CliCommand::Auth {
                command: AuthCommand::Login {
                    provider: AuthProvider(ConfigProvider::OpenCodeGo),
                    ..
                },
            })
        ));

        let vercel = Cli::try_parse_from(["claux", "auth", "login", "vercel"]).unwrap();
        assert!(matches!(
            vercel.command,
            Some(CliCommand::Auth {
                command: AuthCommand::Login {
                    provider: AuthProvider(ConfigProvider::Vercel),
                    ..
                },
            })
        ));

        let alias = Cli::try_parse_from(["claux", "auth", "status", "opencode"]).unwrap();
        assert!(matches!(
            alias.command,
            Some(CliCommand::Auth {
                command: AuthCommand::Status {
                    provider: AuthProvider(ConfigProvider::OpenCodeGo),
                },
            })
        ));
    }

    #[test]
    fn parses_json_output_for_one_shot_mode() {
        let cli =
            Cli::try_parse_from(["claux", "--print", "hello", "--output-format", "json"]).unwrap();

        assert_eq!(cli.output_format, Some(OutputFormat::Json));
    }

    #[test]
    fn parses_transcript_for_one_shot_mode() {
        let cli = Cli::try_parse_from([
            "claux",
            "--print",
            "hello",
            "--transcript",
            "/tmp/claux-transcript.json",
        ])
        .unwrap();

        assert_eq!(
            cli.transcript,
            Some(PathBuf::from("/tmp/claux-transcript.json"))
        );
    }

    #[test]
    fn parses_repeated_images_for_one_shot_mode() {
        let cli = Cli::try_parse_from([
            "claux",
            "--print",
            "describe these",
            "--image",
            "one.png",
            "--image",
            "two.jpg",
        ])
        .unwrap();

        assert_eq!(
            cli.image,
            vec![PathBuf::from("one.png"), PathBuf::from("two.jpg")]
        );
    }

    #[test]
    fn image_requires_one_shot_mode() {
        let error = match Cli::try_parse_from(["claux", "--image", "one.png"]) {
            Ok(_) => panic!("image should require one-shot mode"),
            Err(error) => error,
        };
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn transcript_requires_one_shot_mode() {
        let error =
            match Cli::try_parse_from(["claux", "--transcript", "/tmp/claux-transcript.json"]) {
                Ok(_) => panic!("transcript should require one-shot mode"),
                Err(error) => error,
            };

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn output_format_requires_one_shot_mode() {
        let error = match Cli::try_parse_from(["claux", "--output-format", "json"]) {
            Ok(_) => panic!("output format should require one-shot mode"),
            Err(error) => error,
        };

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn parses_config_init_provider() {
        let cli = Cli::try_parse_from([
            "claux",
            "config",
            "init",
            "--provider",
            "ollama",
            "--model",
            "local-coder",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(CliCommand::Config {
                command: ConfigCommand::Init {
                    provider: ConfigProvider::Ollama,
                    model: Some(ref model),
                    force: false,
                },
            }) if model == "local-coder"
        ));
    }

    #[test]
    fn parses_openrouter_config_init() {
        let cli = Cli::try_parse_from([
            "claux",
            "config",
            "init",
            "--provider",
            "openrouter",
            "--model",
            "anthropic/claude-sonnet-5",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(CliCommand::Config {
                command: ConfigCommand::Init {
                    provider: ConfigProvider::OpenRouter,
                    model: Some(ref model),
                    force: false,
                },
            }) if model == "anthropic/claude-sonnet-5"
        ));
    }
}
