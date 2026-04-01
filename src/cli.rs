use clap::Parser;

#[derive(Parser)]
#[command(name = "claude-rs")]
#[command(about = "Claude Code — rewritten in Rust")]
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

    /// Permission mode (default, accept-edits, bypass)
    #[arg(long, default_value = "default")]
    pub permission_mode: String,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Debug output
    #[arg(long)]
    pub debug: bool,
}
