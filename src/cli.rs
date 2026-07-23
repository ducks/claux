use clap::Parser;

#[derive(Parser)]
#[command(name = "claux")]
#[command(about = "claux — an open, hackable terminal AI coding assistant in Rust")]
pub struct Cli {
    /// One-shot prompt (non-interactive)
    #[arg(short = 'p', long = "print")]
    pub prompt: Option<String>,

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
