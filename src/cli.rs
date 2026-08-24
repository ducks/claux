use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "claux")]
#[command(about = "claux — an open, hackable terminal AI coding assistant in Rust")]
pub struct Cli {
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
    #[arg(long)]
    pub resume: Option<String>,

    /// Permission mode (default, accept-edits, bypass, plan)
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Subcommand)]
pub enum CliCommand {
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
    /// Compare OpenRouter models using native prompt-token counts
    #[command(name = "tokenizer-fingerprint")]
    TokenizerFingerprint {
        /// OpenRouter model identifiers to compare
        #[arg(required = true, num_args = 2..)]
        models: Vec<String>,

        /// Emit the complete machine-readable fingerprint
        #[arg(long)]
        json: bool,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum AuthProvider {
    #[value(name = "openrouter")]
    OpenRouter,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ConfigProvider {
    Anthropic,
    Openai,
    #[value(name = "openrouter")]
    OpenRouter,
    Ollama,
}

#[cfg(test)]
mod tests {
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
            Some(CliCommand::TokenizerFingerprint { models, json: true })
                if models == ["stealth/ox-alpha", "z-ai/glm-5.3"]
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
                    provider: AuthProvider::OpenRouter,
                    headless: true,
                    no_browser: true,
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
