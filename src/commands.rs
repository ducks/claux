use anyhow::Result;

use crate::config::{ModelBinding, ResolvedModel};
use crate::query::Engine;
use crate::session;
use crate::theme::ThemeName;
use std::path::PathBuf;

/// One slash command, described once.
///
/// The parser, the help text and the completer all read this table. They used
/// to be independent: a `match` on string literals plus a hand-maintained help
/// string, which had already drifted (`/quit` parsed but was undocumented).
pub struct CommandSpec {
    /// Canonical name, with the leading slash.
    pub name: &'static str,
    /// Accepted alternatives. Not offered by the completer - one obvious
    /// spelling per command keeps the menu short.
    pub aliases: &'static [&'static str],
    pub summary: &'static str,
    /// Placeholder for the argument, when the command takes one. Drives both
    /// the help column and whether accepting a completion leaves a trailing
    /// space ready for that argument.
    pub arg: Option<&'static str>,
    /// True for commands only the full-screen TUI can service. `/home` returns
    /// to the session picker, which the line-based REPL does not have, so it is
    /// listed and completed there but not in the REPL.
    pub tui_only: bool,
}

impl CommandSpec {
    /// Display form: `/model [profile]`.
    pub fn usage(&self) -> String {
        match self.arg {
            Some(arg) => format!("{} [{arg}]", self.name),
            None => self.name.to_string(),
        }
    }

    /// True if `token` is a prefix of this command's canonical name. Aliases
    /// are matched by the parser but deliberately not completed.
    fn completes(&self, token: &str) -> bool {
        self.name.starts_with(token)
    }

    fn matches(&self, token: &str) -> bool {
        self.name == token || self.aliases.contains(&token)
    }
}

/// Every slash command, in the order the help and completion menu show them.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/help",
        aliases: &[],
        summary: "Show this help",
        arg: None,
        tui_only: false,
    },
    CommandSpec {
        name: "/cost",
        aliases: &[],
        summary: "Show token usage and cost",
        arg: None,
        tui_only: false,
    },
    CommandSpec {
        name: "/context",
        aliases: &[],
        summary: "Show context use and compaction threshold",
        arg: None,
        tui_only: false,
    },
    CommandSpec {
        name: "/compact",
        aliases: &[],
        summary: "Summarize conversation to free context",
        arg: None,
        tui_only: false,
    },
    CommandSpec {
        name: "/diff",
        aliases: &[],
        summary: "Show the last turn's file changes",
        arg: None,
        tui_only: false,
    },
    CommandSpec {
        name: "/undo-turn",
        aliases: &[],
        summary: "Safely undo the last turn's file changes",
        arg: None,
        tui_only: false,
    },
    CommandSpec {
        name: "/image",
        aliases: &[],
        summary: "Attach an image to the next prompt",
        arg: Some("path"),
        tui_only: false,
    },
    CommandSpec {
        name: "/model",
        aliases: &[],
        summary: "Show configured models or switch profile",
        arg: Some("profile"),
        tui_only: false,
    },
    CommandSpec {
        name: "/theme",
        aliases: &[],
        summary: "Show or switch theme (dark, light, ansi)",
        arg: Some("name"),
        tui_only: false,
    },
    CommandSpec {
        name: "/resume",
        aliases: &[],
        summary: "List or resume past sessions",
        arg: Some("id"),
        tui_only: false,
    },
    CommandSpec {
        name: "/clear",
        aliases: &[],
        summary: "Clear screen",
        arg: None,
        tui_only: false,
    },
    CommandSpec {
        name: "/home",
        aliases: &[],
        summary: "Return to the session picker",
        arg: None,
        // Intercepted by the TUI event loop before parse_command, because
        // returning home is a screen transition the engine knows nothing
        // about. Listed here so it is discoverable and completable.
        tui_only: true,
    },
    CommandSpec {
        name: "/exit",
        aliases: &["/quit"],
        summary: "Exit claux",
        arg: None,
        tui_only: false,
    },
];

/// Which frontend is asking. Some commands only exist in one of them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    Tui,
    Repl,
}

impl Surface {
    fn shows(self, spec: &CommandSpec) -> bool {
        !spec.tui_only || self == Surface::Tui
    }
}

/// Commands whose canonical name starts with `token`, in table order.
///
/// `token` is the raw first word including the leading slash. A bare `/`
/// matches everything, which is what makes the menu a discovery surface.
pub fn complete(token: &str, surface: Surface) -> Vec<&'static CommandSpec> {
    if !token.starts_with('/') {
        return Vec::new();
    }
    COMMANDS
        .iter()
        .filter(|c| surface.shows(c) && c.completes(token))
        .collect()
}

pub enum CommandResult {
    /// Print text to the user
    Text(String),
    /// Exit the REPL
    Exit,
    /// Async command that needs engine access (handled by caller)
    Async(AsyncCommand),
}

pub enum AsyncCommand {
    Compact,
    Diff,
    UndoTurn,
    Image(PathBuf),
    Resume(Option<String>),
    Model(Option<String>),
    Theme(Option<String>),
}

/// Parse a slash command. Returns None if input isn't a command.
///
/// `surface` decides what `/help` advertises and how a TUI-only command is
/// answered when it reaches the REPL.
pub fn parse_command(input: &str, surface: Surface) -> Option<CommandResult> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let (cmd, args) = match trimmed.split_once(' ') {
        Some((c, a)) => (c, a.trim()),
        None => (trimmed, ""),
    };

    // Resolve aliases to the canonical name once, so the dispatch below only
    // ever sees canonical spellings and cannot fall out of step with the table.
    let cmd = COMMANDS
        .iter()
        .find(|spec| spec.matches(cmd))
        .map(|spec| spec.name)
        .unwrap_or(cmd);

    match cmd {
        // /home is a screen transition, handled by the TUI event loop before
        // this runs. Reaching it here means the REPL, which has no home screen.
        "/home" => Some(CommandResult::Text(
            "/home is only available in the full-screen TUI (claux --tui).".to_string(),
        )),
        "/help" => Some(CommandResult::Text(help_text(surface))),
        "/exit" => Some(CommandResult::Exit),
        "/clear" => Some(CommandResult::Text("\x1b[2J\x1b[H".to_string())),
        "/compact" => Some(CommandResult::Async(AsyncCommand::Compact)),
        "/context" => Some(CommandResult::Text("__context__".to_string())),
        "/diff" => Some(CommandResult::Async(AsyncCommand::Diff)),
        "/undo-turn" => Some(CommandResult::Async(AsyncCommand::UndoTurn)),
        "/image" => {
            if args.is_empty() {
                Some(CommandResult::Text("Usage: /image <path>".to_string()))
            } else {
                Some(CommandResult::Async(AsyncCommand::Image(PathBuf::from(
                    args,
                ))))
            }
        }
        "/resume" => {
            let id = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            Some(CommandResult::Async(AsyncCommand::Resume(id)))
        }
        "/model" => {
            let model = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            Some(CommandResult::Async(AsyncCommand::Model(model)))
        }
        "/theme" => {
            let theme = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            Some(CommandResult::Async(AsyncCommand::Theme(theme)))
        }
        "/cost" => Some(CommandResult::Text("__cost__".to_string())),
        _ => Some(CommandResult::Text(format!(
            "Unknown command: {cmd}. Type /help for available commands."
        ))),
    }
}

/// Execute an async command that needs engine access.
pub async fn execute_async(cmd: AsyncCommand, engine: &mut Engine) -> Result<String> {
    match cmd {
        AsyncCommand::Compact => engine.compact().await,
        AsyncCommand::Diff => Ok(engine.last_turn_diff()),
        AsyncCommand::UndoTurn => engine.undo_last_turn(),
        AsyncCommand::Image(path) => {
            let image = crate::image_input::load_image(&path)?;
            let count = engine.queue_image(image);
            Ok(format!(
                "Attached {} for the next prompt ({count} queued).",
                path.display()
            ))
        }
        AsyncCommand::Resume(id) => execute_resume(id, engine),
        AsyncCommand::Model(_) => {
            anyhow::bail!("model switching must be resolved through a configured model profile")
        }
        AsyncCommand::Theme(theme_name) => execute_theme(theme_name, engine).await,
    }
}

/// Show cost info (separate since it only needs read access).
pub fn format_cost(engine: &Engine) -> String {
    engine.cost.format_summary()
}

pub fn format_context(engine: &Engine) -> String {
    engine.context_report()
}

fn execute_resume(id: Option<String>, engine: &mut Engine) -> Result<String> {
    match id {
        Some(session_id) => {
            // Session switching is owned by the REPL and TUI surfaces.  Do
            // not load another session into this engine: callers that route
            // through execute_async may still be writing the current session
            // immediately afterwards.
            let _ = engine;
            anyhow::bail!(
                "session switching for /resume {session_id} must be handled by the session browser"
            )
        }
        None => {
            // List recent sessions
            let sessions = session::list_sessions()?;
            if sessions.is_empty() {
                return Ok("No sessions found.".to_string());
            }

            let mut output = String::from("Recent sessions:\n");
            for (i, (id, path)) in sessions.iter().take(10).enumerate() {
                let meta_line = match session::load_session(path) {
                    Ok((meta, msgs)) => format!(
                        "  \x1b[33m{}\x1b[0m  {}  {} msgs  {}",
                        meta.id,
                        meta.model,
                        msgs.len(),
                        meta.cwd
                    ),
                    Err(_) => format!("  \x1b[33m{id}\x1b[0m  (error reading)"),
                };
                output.push_str(&meta_line);
                if i < sessions.len().min(10) - 1 {
                    output.push('\n');
                }
            }
            output.push_str("\n\nUse /resume <id> to resume a session.");
            Ok(output)
        }
    }
}

/// Format configured model profiles for frontends that handle provider-safe
/// switching themselves.
pub fn format_model_choices(current: Option<&ModelBinding>, models: &[ResolvedModel]) -> String {
    let mut models = models.iter().collect::<Vec<_>>();
    models.sort_by_key(|resolved| {
        (
            resolved.binding.provider_name.to_ascii_lowercase(),
            resolved.binding.display_name.to_ascii_lowercase(),
        )
    });

    let mut output = String::from("Configured models:");
    let mut provider = None::<&str>;
    for resolved in models {
        let binding = &resolved.binding;
        if provider != Some(binding.provider_name.as_str()) {
            provider = Some(binding.provider_name.as_str());
            output.push_str(&format!("\n\n{}:", binding.provider_name));
        }
        let marker = if current == Some(binding) { " *" } else { "" };
        output.push_str(&format!(
            "\n  {}{marker} — {} ({})",
            binding.profile, binding.display_name, binding.model
        ));
    }
    output.push_str("\n\nUse /model <profile> to switch.");
    output
}

async fn execute_theme(theme_name: Option<String>, engine: &mut Engine) -> Result<String> {
    match theme_name {
        Some(name) => {
            let _theme = match name.to_lowercase().as_str() {
                "dark" => ThemeName::Dark,
                "light" => ThemeName::Light,
                "ansi" => ThemeName::Ansi,
                "dracula" => ThemeName::Dracula,
                "nord" => ThemeName::Nord,
                "catppuccin" => ThemeName::Catppuccin,
                _ => {
                    return Ok(format!(
                        "Unknown theme: {name}\n\n\
                         Available themes:\n\
                         - dark: gruvbox-inspired (default)\n\
                         - light: high-contrast for bright terminals\n\
                         - ansi: 16-color fallback\n\
                         - dracula: dark purple/violet theme\n\
                         - nord: arctic blue-gray theme\n\
                         - catppuccin: pastel mocha theme"
                    ));
                }
            };
            engine.set_theme(_theme);
            Ok(format!("Theme set to: {name}"))
        }
        None => Ok("Current theme: dark\n\n\
                 Available themes:\n\
                 - dark: gruvbox-inspired (default)\n\
                 - light: high-contrast for bright terminals\n\
                 - ansi: 16-color fallback\n\
                 - dracula: dark purple/violet theme\n\
                 - nord: arctic blue-gray theme\n\
                 - catppuccin: pastel mocha theme\n\n\
                 Use /theme <name> to switch."
            .to_string()),
    }
}

fn help_text(surface: Surface) -> String {
    // Rendered from COMMANDS so it cannot drift from what the parser accepts,
    // and filtered by surface so the REPL is not told about /home.
    let visible: Vec<_> = COMMANDS.iter().filter(|c| surface.shows(c)).collect();
    let width = visible
        .iter()
        .map(|c| c.usage().chars().count())
        .max()
        .unwrap_or(0);

    let mut out = String::from("Available commands:\n");
    for spec in visible {
        out.push_str(&format!(
            "  {:<width$}  {}\n",
            spec.usage(),
            spec.summary,
            width = width
        ));
    }
    out.push_str(
        "\nKeyboard:\n  Tab       Accept completion\n  Ctrl+C    Cancel current request\n  Ctrl+D    Exit",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_command_returns_none() {
        assert!(parse_command("hello world", Surface::Tui).is_none());
    }

    #[test]
    fn home_is_completable_in_the_tui() {
        // Regression: /home was special-cased in the TUI event loop and absent
        // from the table, so typing / never offered it and /help never listed
        // it - exactly the drift the table exists to prevent.
        let names: Vec<_> = complete("/h", Surface::Tui)
            .iter()
            .map(|c| c.name)
            .collect();
        assert!(names.contains(&"/home"), "got {names:?}");
    }

    #[test]
    fn home_is_hidden_from_the_repl() {
        let names: Vec<_> = complete("/h", Surface::Repl)
            .iter()
            .map(|c| c.name)
            .collect();
        assert!(!names.contains(&"/home"), "REPL has no home screen");
        assert!(names.contains(&"/help"), "other matches still offered");
    }

    #[test]
    fn home_in_the_repl_explains_itself() {
        // Reaching parse_command with /home means the REPL: the TUI intercepts
        // it earlier. Say so rather than falling through to "Unknown command".
        let Some(CommandResult::Text(text)) = parse_command("/home", Surface::Tui) else {
            panic!("expected explanatory text");
        };
        assert!(text.contains("full-screen TUI"), "got {text}");
    }

    #[test]
    fn tui_help_lists_every_command_in_the_table() {
        let Some(CommandResult::Text(text)) = parse_command("/help", Surface::Tui) else {
            panic!("expected help text");
        };
        for spec in COMMANDS {
            assert!(text.contains(spec.name), "{} missing from /help", spec.name);
        }
    }

    #[test]
    fn repl_help_omits_tui_only_commands() {
        let Some(CommandResult::Text(text)) = parse_command("/help", Surface::Repl) else {
            panic!("expected help text");
        };
        assert!(!text.contains("/home"), "REPL has no home screen to offer");
        for spec in COMMANDS.iter().filter(|c| !c.tui_only) {
            assert!(text.contains(spec.name), "{} missing from /help", spec.name);
        }
    }

    #[test]
    fn help_returns_text() {
        let result = parse_command("/help", Surface::Tui);
        assert!(matches!(result, Some(CommandResult::Text(_))));
    }

    #[test]
    fn exit_returns_exit() {
        assert!(matches!(
            parse_command("/exit", Surface::Tui),
            Some(CommandResult::Exit)
        ));
        assert!(matches!(
            parse_command("/quit", Surface::Tui),
            Some(CommandResult::Exit)
        ));
    }

    #[test]
    fn cost_returns_sentinel() {
        if let Some(CommandResult::Text(text)) = parse_command("/cost", Surface::Tui) {
            assert_eq!(text, "__cost__");
        } else {
            panic!("expected Text");
        }
    }

    #[test]
    fn context_returns_sentinel() {
        if let Some(CommandResult::Text(text)) = parse_command("/context", Surface::Tui) {
            assert_eq!(text, "__context__");
        } else {
            panic!("expected Text");
        }
    }

    #[test]
    fn compact_returns_async() {
        assert!(matches!(
            parse_command("/compact", Surface::Tui),
            Some(CommandResult::Async(AsyncCommand::Compact))
        ));
    }

    #[test]
    fn checkpoint_commands_return_async_actions() {
        assert!(matches!(
            parse_command("/diff", Surface::Tui),
            Some(CommandResult::Async(AsyncCommand::Diff))
        ));
        assert!(matches!(
            parse_command("/undo-turn", Surface::Tui),
            Some(CommandResult::Async(AsyncCommand::UndoTurn))
        ));
    }

    #[test]
    fn image_command_parses_path() {
        assert!(matches!(
            parse_command("/image /tmp/my picture.png", Surface::Tui),
            Some(CommandResult::Async(AsyncCommand::Image(path)))
                if path.as_path() == std::path::Path::new("/tmp/my picture.png")
        ));
    }

    #[test]
    fn image_command_requires_path() {
        assert!(matches!(
            parse_command("/image", Surface::Repl),
            Some(CommandResult::Text(text)) if text.contains("Usage:")
        ));
    }

    #[test]
    fn model_no_args_returns_none_model() {
        if let Some(CommandResult::Async(AsyncCommand::Model(m))) =
            parse_command("/model", Surface::Tui)
        {
            assert!(m.is_none());
        } else {
            panic!("expected Model(None)");
        }
    }

    #[test]
    fn model_with_profile() {
        if let Some(CommandResult::Async(AsyncCommand::Model(Some(m)))) =
            parse_command("/model kimi", Surface::Tui)
        {
            assert_eq!(m, "kimi");
        } else {
            panic!("expected Model(Some)");
        }
    }

    #[test]
    fn model_choices_are_grouped_and_mark_current_profile() {
        let config: crate::config::Config = toml::from_str(
            r#"
[providers.a]
type = "anthropic"
name = "Anthropic"

[providers.b]
type = "openai"
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"

[model_profiles.sonnet]
provider = "a"
model = "claude-sonnet"

[model_profiles.kimi]
provider = "b"
model = "moonshotai/kimi-k3"
"#,
        )
        .unwrap();
        let models = config.selectable_models().unwrap();
        let current = &config.resolve_model("kimi").unwrap().binding;

        let output = format_model_choices(Some(current), &models);

        assert!(output.contains("Anthropic:\n  sonnet"));
        assert!(output.contains("OpenRouter:\n  kimi *"));
        assert!(output.contains("Use /model <profile> to switch."));
    }

    #[test]
    fn resume_no_args() {
        if let Some(CommandResult::Async(AsyncCommand::Resume(id))) =
            parse_command("/resume", Surface::Tui)
        {
            assert!(id.is_none());
        } else {
            panic!("expected Resume(None)");
        }
    }

    #[test]
    fn resume_with_id() {
        if let Some(CommandResult::Async(AsyncCommand::Resume(Some(id)))) =
            parse_command("/resume 20260401-143022", Surface::Tui)
        {
            assert_eq!(id, "20260401-143022");
        } else {
            panic!("expected Resume(Some)");
        }
    }

    #[test]
    fn unknown_command_returns_error_text() {
        if let Some(CommandResult::Text(text)) = parse_command("/bogus", Surface::Tui) {
            assert!(text.contains("Unknown command"));
        } else {
            panic!("expected Text");
        }
    }
}
