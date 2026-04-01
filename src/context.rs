use anyhow::Result;

/// Build the system prompt from environment context.
/// Mirrors Claude Code's context.ts: git status, CLAUDE.md, date, env info.
pub async fn build_system_prompt() -> Result<String> {
    let mut parts: Vec<String> = Vec::new();

    parts.push(base_system_prompt());

    // Environment
    parts.push(format!("# Environment"));
    parts.push(format!("- Platform: {}", std::env::consts::OS));
    parts.push(format!("- Shell: {}", std::env::var("SHELL").unwrap_or_else(|_| "sh".into())));

    if let Ok(cwd) = std::env::current_dir() {
        parts.push(format!("- Working directory: {}", cwd.display()));
    }

    // Current date
    parts.push(format!(
        "- Date: {}",
        chrono::Local::now().format("%Y-%m-%d")
    ));

    // Git status
    if let Some(git_info) = git_status().await {
        parts.push(format!("\n# Git Status\n{}", git_info));
    }

    // CLAUDE.md / project context
    if let Some(claude_md) = read_claude_md().await {
        parts.push(format!("\n# Project Context (CLAUDE.md)\n{}", claude_md));
    }

    Ok(parts.join("\n"))
}

fn base_system_prompt() -> String {
    r#"You are Claude, an AI assistant by Anthropic. You are running inside claude-rs, a Rust rewrite of Claude Code.

You are an interactive agent that helps users with software engineering tasks. Use the tools available to you to assist the user.

# Tool usage
- Use Read to read files, not Bash with cat
- Use Edit for surgical file changes
- Use Write to create new files
- Use Glob to find files by pattern
- Use Grep to search file contents
- Use Bash for shell commands, builds, git operations

# Style
- Be concise and direct
- Lead with the answer, not the reasoning
- When referencing code, include file_path:line_number
"#
    .to_string()
}

async fn git_status() -> Option<String> {
    let branch = run_cmd("git", &["branch", "--show-current"]).await?;
    let status = run_cmd("git", &["status", "--short"]).await.unwrap_or_default();
    let log = run_cmd("git", &["log", "--oneline", "-n", "5"]).await.unwrap_or_default();

    let mut info = format!("Branch: {}", branch.trim());
    if !status.is_empty() {
        let status = if status.len() > 2000 {
            format!("{}... (truncated)", &status[..2000])
        } else {
            status
        };
        info.push_str(&format!("\nStatus:\n{}", status));
    }
    if !log.is_empty() {
        info.push_str(&format!("\nRecent commits:\n{}", log));
    }

    Some(info)
}

async fn read_claude_md() -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    let cwd = std::env::current_dir().ok()?;

    // 1. Check cwd and .claude/ subdir
    for name in &["CLAUDE.md", ".claude/CLAUDE.md"] {
        let path = cwd.join(name);
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                parts.push(format!("# {} ({})\n{}", name, cwd.display(), content));
            }
        }
    }

    // 2. Walk up parent directories
    let mut dir = cwd.as_path();
    while let Some(parent) = dir.parent() {
        // Don't re-read cwd
        if parent == cwd {
            dir = parent;
            continue;
        }
        for name in &["CLAUDE.md", ".claude/CLAUDE.md"] {
            let path = parent.join(name);
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    parts.push(format!("# {} ({})\n{}", name, parent.display(), content));
                }
            }
        }
        dir = parent;
    }

    // 3. Check ~/.claude/CLAUDE.md (user-global)
    if let Ok(home) = std::env::var("HOME") {
        let path = std::path::PathBuf::from(&home)
            .join(".claude")
            .join("CLAUDE.md");
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                parts.push(format!("# ~/.claude/CLAUDE.md\n{}", content));
            }
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

async fn run_cmd(program: &str, args: &[&str]) -> Option<String> {
    let output = tokio::process::Command::new(program)
        .args(args)
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    Some(String::from_utf8_lossy(&output.stdout).to_string())
}
