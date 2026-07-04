use crate::config::HookTrigger;
use crate::plugin::PluginRegistry;
use anyhow::Result;

/// Separator between system prompt blocks.
/// The Anthropic provider splits on this to send an array of text blocks:
/// one static instruction block (identical across sessions, cache-friendly)
/// and one runtime block (environment, git status, project context).
/// Other providers join the blocks into a single string.
pub const SYSTEM_PROMPT_BLOCK_SEPARATOR: &str = "\n__CLAUX_BLOCK__\n";

/// Build the system prompt from environment context.
/// Used by sub-agents.
pub async fn build_system_prompt() -> Result<String> {
    build_system_prompt_for_model("an AI assistant", None, &HookTrigger::OnContextBuild, true).await
}

pub async fn build_system_prompt_for_model(
    model: &str,
    plugins: Option<&PluginRegistry>,
    trigger: &HookTrigger,
    is_anthropic: bool,
) -> Result<String> {
    // Block 0: static instructions — claux's own prompt, same for every
    // provider. What you read here is exactly what the model gets.
    let instructions = claux_system_prompt(model);

    // Block 1: runtime (environment, git status, CLAUDE.md, memory, plugins)
    let runtime = build_runtime_section(model, plugins, trigger).await;

    if is_anthropic {
        Ok(format!(
            "{instructions}{SYSTEM_PROMPT_BLOCK_SEPARATOR}{runtime}"
        ))
    } else {
        Ok(format!("{instructions}\n\n{runtime}"))
    }
}

/// Build the runtime portion of the system prompt matching CC's dynamic sections.
async fn build_runtime_section(
    model: &str,
    plugins: Option<&PluginRegistry>,
    trigger: &HookTrigger,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Environment section matching CC's computeSimpleEnvInfo format
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let is_git = std::path::Path::new(".git").exists();
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".into());
    let shell_name = if shell.contains("zsh") {
        "zsh"
    } else if shell.contains("bash") {
        "bash"
    } else {
        &shell
    };
    let os_version = run_cmd("uname", &["-sr"])
        .await
        .unwrap_or_else(|| format!("{} unknown", std::env::consts::OS));

    let env_items = [
        format!(" - Primary working directory: {cwd}"),
        format!(" - Is a git repository: {is_git}"),
        format!(" - Platform: {}", std::env::consts::OS),
        format!(" - Shell: {shell_name}"),
        format!(" - OS Version: {}", os_version.trim()),
        format!(" - You are powered by the model {model}."),
    ];

    parts.push(format!(
        "# Environment\nYou have been invoked in the following environment: \n{}",
        env_items.join("\n")
    ));

    // Git status matching CC's gitStatus format
    if is_git {
        if let Some(git_info) = git_status().await {
            parts.push(format!("\ngitStatus: {git_info}"));
        }
    }

    // CLAUDE.md / project context
    if let Some(claude_md) = read_claude_md().await {
        parts.push(format!("\n{claude_md}"));
    }

    // Ensure memory directory exists and load MEMORY.md if present
    let memory_dir = build_memory_dir_path();
    let _ = std::fs::create_dir_all(&memory_dir);
    let memory_index = std::path::Path::new(&memory_dir).join("MEMORY.md");
    if memory_index.exists() {
        if let Ok(content) = std::fs::read_to_string(&memory_index) {
            if !content.trim().is_empty() {
                // Truncate to 200 lines matching CC behavior
                let truncated: String = content.lines().take(200).collect::<Vec<_>>().join("\n");
                parts.push(format!("\n# Memory Index (MEMORY.md)\n{truncated}"));
            }
        }
    }

    // Plugin context
    if let Some(registry) = plugins {
        if let Ok(plugin_context) = registry.execute_all(trigger, None) {
            if !plugin_context.is_empty() {
                parts.push(format!("\n# Plugin Context\n{plugin_context}"));
            }
        }
    }

    parts.join("\n")
}

/// claux's system prompt. One prompt for every provider: what you read
/// here is exactly what the model gets, plus the runtime section built in
/// build_runtime_section.
fn claux_system_prompt(model: &str) -> String {
    let memory_dir = build_memory_dir_path();

    format!(
        r#"You are claux, an open-source terminal coding assistant, currently powered by the model {model}. You help users with software engineering tasks in the working directory: fixing bugs, adding features, refactoring, explaining code, and running project tooling.

# Communication
- Text you output outside of tool calls is shown to the user, rendered as markdown in a terminal.
- Be concise and direct. Lead with the answer or the action, not the reasoning that led there. Skip preamble, filler, and restating what the user said.
- When referencing code, use the pattern file_path:line_number so the user can jump to it.
- Only use emojis if the user asks for them.

# Using tools
- Prefer the dedicated tools over shell equivalents: Read (not cat/head/tail), Edit (not sed/awk), Write (not echo/heredoc redirection), Glob (not find), Grep (not grep/rg). Reserve Bash for things that need a shell: builds, tests, git, package managers, project scripts.
- Read a file before you propose changes to it. Do not speculate about code you have not opened.
- Independent tool calls can be issued together and run in parallel; dependent calls must run one at a time.
- Use the Agent tool to delegate self-contained subtasks (research, broad searches, multi-step side quests) when doing them inline would flood the conversation with output. Sub-agents cannot spawn further agents.
- Use TodoWrite to plan multi-step work and mark items done as you finish them, so the user can follow progress.
- Use WebFetch to retrieve a URL when the task needs it. Never invent URLs; use ones from the user or the code.
- MCP tools may be available beyond the built-in set; treat them like any other tool.
- Tools run behind the user's permission mode. If the user denies a tool call, do not retry it verbatim: reconsider, adjust, or ask why.

# Doing tasks
- Make the change the user asked for and stop. No drive-by refactors, no extra configurability, no comments or docs on code you did not touch.
- Match the existing style of the file you are editing: naming, formatting, idiom, comment density.
- Prefer editing existing files over creating new ones. Only create files that the task genuinely requires.
- After a nontrivial change, verify it with the project's own tooling when available: run the tests, the linter, the build. Report results honestly, including failures.
- If an approach fails, read the error and diagnose before switching tactics. Do not retry the identical action blindly, and do not abandon a viable approach after one failure.
- Validate at system boundaries (user input, external APIs); trust internal code and framework guarantees. Do not add error handling for situations that cannot happen.

# Acting with care
- Local, reversible actions (editing files, running tests) are yours to take freely within the permission mode.
- For destructive or hard-to-reverse actions - deleting files or branches, rm -rf, force-pushing, git reset --hard, dropping data, killing processes - and for anything visible to others (pushing, opening PRs, posting to external services), confirm with the user first unless they have explicitly told you to proceed.
- When you hit an obstacle, fix the cause instead of bypassing the safeguard. Never skip hooks or checks to make an error go away.
- If you find unexpected state (unfamiliar files, lock files, merge conflicts), investigate before deleting or overwriting; it may be someone's in-progress work.

# Git
- Never commit unless the user asks. When they do: review the diff and recent commit messages first, follow the repository's message style, and stage specific files rather than git add -A.
- Never update git config, amend published commits, or run destructive git commands without an explicit request.
- If a pre-commit hook fails, fix the issue properly and create a new commit; do not amend and do not use --no-verify.

# Memory
You have a persistent memory directory at `{memory_dir}`. Its index, MEMORY.md, is loaded into your context each session.

- To save something durable (who the user is, feedback on how to work, project context, pointers to external resources), write a small markdown file in that directory, then add a one-line entry to MEMORY.md linking it: `- [Title](file.md) - hook`.
- Save when the user corrects you, confirms an unusual approach, or asks you to remember something. Do not save what the code, git history, or CLAUDE.md already records.
- Update or delete memories that turn out to be wrong. Check for an existing file before creating a duplicate.
- Memories reflect what was true when written. Verify against the current code before acting on one.
"#
    )
}

/// Build claux's per-project memory directory:
/// <data dir>/claux/projects/<sanitized-cwd>/memory/
/// (e.g. ~/.local/share/claux/projects/home-ducks-dev/memory/ on Linux).
/// claux keeps its own memory root rather than sharing Claude Code's
/// ~/.claude tree, so the two tools never write over each other's memories.
fn build_memory_dir_path() -> String {
    let base = dirs::data_local_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "/tmp".to_string());
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Sanitize the cwd into a single path component
    let sanitized = cwd.trim_start_matches('/').replace('/', "-");

    format!("{base}/claux/projects/{sanitized}/memory/")
}

/// Build git status matching CC's exact format from context.ts
async fn git_status() -> Option<String> {
    let branch = run_cmd("git", &["branch", "--show-current"]).await?;
    let branch = branch.trim();

    // Determine main/default branch (same priority as CC's getCachedDefaultBranch)
    let main_branch = detect_default_branch().await;

    // Git user name
    let user_name = run_cmd("git", &["config", "user.name"]).await;

    let status = run_cmd("git", &["--no-optional-locks", "status", "--short"])
        .await
        .unwrap_or_default();
    let log = run_cmd(
        "git",
        &["--no-optional-locks", "log", "--oneline", "-n", "5"],
    )
    .await
    .unwrap_or_default();

    let truncated_status = if status.len() > 2000 {
        format!(
            "{}... (truncated because it exceeds 2k characters. If you need more information, run \"git status\" using Bash)",
            crate::utils::truncate_str(&status, 2000)
        )
    } else if status.trim().is_empty() {
        "(clean)".to_string()
    } else {
        status
    };

    let mut parts = vec![
        "This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation.".to_string(),
        format!("Current branch: {branch}"),
        format!("Main branch (you will usually use this for PRs): {main_branch}"),
    ];

    if let Some(ref name) = user_name {
        let name = name.trim();
        if !name.is_empty() {
            parts.push(format!("Git user: {name}"));
        }
    }

    parts.push(format!("Status:\n{truncated_status}"));
    parts.push(format!("Recent commits:\n{log}"));

    Some(parts.join("\n\n"))
}

/// Detect the default branch matching CC's getCachedDefaultBranch logic:
/// 1. Check refs/remotes/origin/HEAD symref
/// 2. Fall back to refs/remotes/origin/main
/// 3. Fall back to refs/remotes/origin/master
/// 4. Default to "main"
async fn detect_default_branch() -> String {
    // Try symbolic-ref
    if let Some(head_ref) = run_cmd("git", &["symbolic-ref", "refs/remotes/origin/HEAD"]).await {
        let head_ref = head_ref.trim();
        if let Some(branch) = head_ref.strip_prefix("refs/remotes/origin/") {
            if !branch.is_empty() {
                return branch.to_string();
            }
        }
    }

    // Check if origin/main exists
    if run_cmd(
        "git",
        &["rev-parse", "--verify", "refs/remotes/origin/main"],
    )
    .await
    .is_some()
    {
        return "main".to_string();
    }

    // Check if origin/master exists
    if run_cmd(
        "git",
        &["rev-parse", "--verify", "refs/remotes/origin/master"],
    )
    .await
    .is_some()
    {
        return "master".to_string();
    }

    "main".to_string()
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
    let files = run_cmd(
        "rg",
        &["--files", "--max-depth", "5", "--hidden", "-g", "!.git"],
    )
    .await?;
    let file_count = files.lines().count();
    let files = if file_count > 100 {
        format!(
            "{}... ({} total files, showing top 100)",
            files.lines().take(100).collect::<Vec<_>>().join("\n"),
            file_count
        )
    } else {
        files
    };
    parts.push(format!(
        "\n**File Structure** ({file_count} files):\n{files}"
    ));

    // 3. Top-level symbols (if ripgrep supports it)
    // Note: --symbols is experimental, fallback to just files if it fails
    if let Some(symbols) = run_cmd("rg", &["--symbols", "--max-depth", "3"]).await {
        if !symbols.trim().is_empty() {
            let symbols = if symbols.lines().count() > 50 {
                format!(
                    "{}... (truncated)",
                    symbols.lines().take(50).collect::<Vec<_>>().join("\n")
                )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_native_claux() {
        let p = claux_system_prompt("test-model");
        assert!(p.starts_with("You are claux"));
        assert!(p.contains("test-model"));
        assert!(
            !p.contains("Claude Code"),
            "claux must not identify as Claude Code"
        );
        // Every tool the prompt tells the model about must exist in the
        // registry; a prompt promising phantom tools causes hallucinated
        // tool calls. ToolRegistry::new() covers everything except Agent.
        let registry = crate::tools::ToolRegistry::new();
        for def in registry.definitions() {
            assert!(
                p.contains(&def.name),
                "prompt should mention the {} tool",
                def.name
            );
        }
        assert!(p.contains("Agent tool"));
    }

    #[test]
    fn memory_dir_is_claux_owned() {
        let dir = build_memory_dir_path();
        assert!(dir.contains("claux"));
        assert!(
            !dir.contains(".claude"),
            "claux memory must not share Claude Code's ~/.claude tree"
        );
    }
}
