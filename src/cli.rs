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

    /// Output format for one-shot mode
    #[arg(long, value_enum, requires = "prompt")]
    pub output_format: Option<OutputFormat>,

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
    /// Diagnose configuration, authentication, tools, and provider connectivity
    Doctor {
        /// Skip the provider network check
        #[arg(long)]
        offline: bool,
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
    fn parses_json_output_for_one_shot_mode() {
        let cli =
            Cli::try_parse_from(["claux", "--print", "hello", "--output-format", "json"]).unwrap();

        assert_eq!(cli.output_format, Some(OutputFormat::Json));
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
