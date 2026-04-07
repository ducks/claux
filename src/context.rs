use anyhow::Result;
use crate::plugin::PluginRegistry;
use crate::config::HookTrigger;

/// Build the system prompt from environment context.
/// Mirrors Claude Code's context.ts: git status, CLAUDE.md, date, env info.
pub async fn build_system_prompt() -> Result<String> {
    build_system_prompt_for_model("an AI assistant", None, &HookTrigger::OnContextBuild).await
}

pub async fn build_system_prompt_for_model(model: &str, plugins: Option<&PluginRegistry>, trigger: &HookTrigger) -> Result<String> {
    let mut parts: Vec<String> = Vec::new();

    parts.push(base_system_prompt(model));

    // Environment
    parts.push("# Environment".to_string());
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
        parts.push(format!("\n# Git Status\n{git_info}"));
    }

    // Project map (smart context)
    if let Some(project_map) = build_project_map().await {
        parts.push(format!("\n# Project Structure\n{project_map}"));
    }

    // CLAUDE.md / project context
    if let Some(claude_md) = read_claude_md().await {
        parts.push(format!("\n# Project Context (CLAUDE.md)\n{claude_md}"));
    }

    // Plugin context
    if let Some(registry) = plugins {
        if let Ok(plugin_context) = registry.execute_all(trigger, None) {
            if !plugin_context.is_empty() {
                parts.push(format!("\n# Plugin Context\n{plugin_context}"));
            }
        }
    }

    Ok(parts.join("\n"))
}

fn base_system_prompt(model: &str) -> String {
    format!(r#"You are {model}, running inside claux, a terminal AI coding assistant.

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
"#)
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
        info.push_str(&format!("\nStatus:\n{status}"));
    }
    if !log.is_empty() {
        info.push_str(&format!("\nRecent commits:\n{log}"));
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
                parts.push(format!("# ~/.claude/CLAUDE.md\n{content}"));
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

/// Build a lightweight project map: file structure + top-level symbols.
/// Runs `rg --files` and `rg --symbols` once at session start.
/// This gives the LLM "memory" of the codebase without heavy indexing.
async fn build_project_map() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    
    // Check if we're in a git repo (skip if not)
    if !cwd.join(".git").exists() {
        return None;
    }

    let mut parts: Vec<String> = Vec::new();

    // 1. Project type detection
    let project_type = detect_project_type(&cwd);
    parts.push(format!("**Project Type:** {project_type}"));

    // 2. File structure (top 100 files, sorted by relevance)
    let files = run_cmd("rg", &["--files", "--max-depth", "5", "--hidden", "-g", "!.git"]).await?;
    let file_count = files.lines().count();
    let files = if file_count > 100 {
        format!("{}... ({} total files, showing top 100)", 
            files.lines().take(100).collect::<Vec<_>>().join("\n"),
            file_count
        )
    } else {
        files
    };
    parts.push(format!("\n**File Structure** ({file_count} files):\n{files}"));

    // 3. Top-level symbols (if ripgrep supports it)
    // Note: --symbols is experimental, fallback to just files if it fails
    if let Some(symbols) = run_cmd("rg", &["--symbols", "--max-depth", "3"]).await {
        if !symbols.trim().is_empty() {
            let symbols = if symbols.lines().count() > 50 {
                format!("{}... (truncated)", symbols.lines().take(50).collect::<Vec<_>>().join("\n"))
            } else {
                symbols
            };
            parts.push(format!("\n**Top-Level Symbols**:\n{symbols}"));
        }
    }

    Some(parts.join("\n"))
}

/// Detect project type from manifest files
fn detect_project_type(cwd: &std::path::Path) -> &'static str {
    if cwd.join("Cargo.toml").exists() {
        "Rust"
    } else if cwd.join("Gemfile").exists() {
        "Ruby"
    } else if cwd.join("package.json").exists() {
        "Node.js/TypeScript"
    } else if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
        "Python"
    } else if cwd.join("go.mod").exists() {
        "Go"
    } else if cwd.join("Cargo.lock").exists() {
        "Rust (locked)"
    } else {
        "Unknown"
    }
}
