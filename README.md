# claux

A terminal-based AI coding assistant written in Rust. Streams responses, executes tools, manages sessions, and stays out of your way.

## Features

- **Streaming chat** with tool execution (Read, Write, Edit, Glob, Grep, Bash, Agent)
- **Interactive permissions** — prompts before writes, `y/n/a` (always allow per-session)
- **Session persistence** — JSONL-based, resume with `/resume` or `--resume`
- **Compaction** — `/compact` summarizes conversation to free context
- **Model switching** — `/model <name>` mid-conversation
- **Sub-agents** — Agent tool spawns scoped sub-conversations
- **Auto-compact** — triggers when conversation gets large
- **Cost tracking** — per-model token usage and USD estimates
- **Context assembly** — git status, CLAUDE.md, environment info in system prompt
- **TUI mode** — full-screen ratatui interface with `--tui`
- **OAuth support** — works with existing `claude login` credentials

## Install

```bash
cargo install --path .
```

Requires Rust 1.88+. A `shell.nix` is included.

## Auth

claux resolves authentication in order:

1. `api_key` in `~/.config/claux/config.toml`
2. `api_key_cmd` (shell command that returns a key)
3. `ANTHROPIC_API_KEY` environment variable
4. OAuth token from `~/.claude/.credentials.json`

If you've already run `claude login`, claux picks up those credentials automatically.

## Usage

```bash
# Interactive REPL (default)
claux

# Full-screen TUI
claux --tui

# One-shot
claux -p "explain this error"

# Resume a session
claux --resume 20260401-143022
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/cost` | Token usage and estimated cost |
| `/compact` | Summarize conversation to free context |
| `/model [name]` | Show or switch model |
| `/resume [id]` | List or resume past sessions |
| `/clear` | Clear screen |
| `/exit` | Exit |

## Config

Global: `~/.config/claux/config.toml`

```toml
model = "claude-sonnet-4-20250514"
permission_mode = "default"  # default | accept-edits | bypass | plan
```

Per-project: `.claux.toml` in the project root (overrides global).

## Permission Modes

| Mode | Reads | File edits | Bash |
|------|-------|------------|------|
| `default` | auto | prompt | prompt |
| `accept-edits` | auto | auto | prompt |
| `bypass` | auto | auto | auto |
| `plan` | auto | denied | denied |

## License

MIT
