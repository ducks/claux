//! Slash-command completion for the TUI input line.
//!
//! The menu is *derived*, not stored: given the input buffer and cursor, this
//! computes whether a completion is showing and what is in it. A `popup_open`
//! flag would desync the moment the user pastes, backspaces past the slash, or
//! moves the cursor - there is nothing to desync here because nothing is kept.
//!
//! The only retained state is the selected index and an Esc suppression, both
//! held by the caller and both reconciled against the derived candidate list
//! every keystroke.

use crate::commands::{self, CommandSpec};

/// What the completion menu should show right now.
pub struct Completion {
    pub matches: Vec<&'static CommandSpec>,
    /// Index into `matches`, always in range.
    pub selected: usize,
    /// The token being completed, e.g. `/co`.
    pub token: String,
}

impl Completion {
    pub fn selected_spec(&self) -> &'static CommandSpec {
        self.matches[self.selected]
    }
}

/// Caller-held completion state.
#[derive(Default)]
pub struct CompletionState {
    selected: usize,
    /// Token the user dismissed with Esc. Suppression lasts while the token is
    /// unchanged, so Esc then typing another letter brings the menu back.
    dismissed: Option<String>,
}

impl CompletionState {
    /// The first word of the line, if the cursor is still inside it.
    ///
    /// Completion applies only to a command at the very start of the input and
    /// only before any argument: `/mo` completes, `/model dev` does not, and a
    /// slash mid-sentence is just a character.
    fn token(input: &str, cursor: usize) -> Option<String> {
        let prefix: String = input.chars().take(cursor).collect();
        if !prefix.starts_with('/') || prefix.contains(char::is_whitespace) {
            return None;
        }
        // The cursor may sit inside a longer line; only complete when the rest
        // of the line is also part of this token.
        let rest: String = input.chars().skip(cursor).collect();
        if rest.contains(char::is_whitespace) {
            return None;
        }
        Some(prefix)
    }

    /// Compute the menu for the current input, or None when it should be
    /// hidden. Clamps the stored selection to the candidate list.
    pub fn active(&mut self, input: &str, cursor: usize) -> Option<Completion> {
        let token = Self::token(input, cursor)?;

        if self.dismissed.as_deref() == Some(token.as_str()) {
            return None;
        }
        // Typing past a dismissal re-arms the menu.
        self.dismissed = None;

        let matches = commands::complete(&token, commands::Surface::Tui);
        if matches.is_empty() {
            return None;
        }
        // An exact, unambiguous match is not worth a menu - the user has
        // already typed the whole command.
        if matches.len() == 1 && matches[0].name == token {
            return None;
        }

        if self.selected >= matches.len() {
            self.selected = 0;
        }
        Some(Completion {
            matches,
            selected: self.selected,
            token,
        })
    }

    pub fn move_selection(&mut self, delta: isize, len: usize) {
        if len == 0 {
            return;
        }
        let len = len as isize;
        // Wrap, so Up from the first entry lands on the last.
        self.selected = (((self.selected as isize + delta) % len + len) % len) as usize;
    }

    pub fn dismiss(&mut self, token: &str) {
        self.dismissed = Some(token.to_string());
        self.selected = 0;
    }

    /// Forget everything. Called when the line is submitted or cleared, so a
    /// stale selection cannot leak into the next command.
    pub fn reset(&mut self) {
        self.selected = 0;
        self.dismissed = None;
    }
}

/// Replace the token under the cursor with the chosen command.
///
/// Commands taking an argument get a trailing space so the user can type it
/// immediately; commands without one do not, so Enter submits straight away.
pub fn apply(input: &str, cursor: usize, spec: &CommandSpec) -> (String, usize) {
    let rest: String = input.chars().skip(cursor).collect();
    let mut completed = spec.name.to_string();
    // Only add the argument space when there is not already whitespace waiting;
    // otherwise completing `/th dark` would yield `/theme  dark`.
    if spec.arg.is_some() && !rest.starts_with(char::is_whitespace) {
        completed.push(' ');
    }
    let new_cursor = completed.chars().count();
    (format!("{completed}{rest}"), new_cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> CompletionState {
        CompletionState::default()
    }

    #[test]
    fn a_bare_slash_offers_every_command() {
        let mut s = state();
        let c = s.active("/", 1).expect("menu");
        assert_eq!(
            c.matches.len(),
            commands::COMMANDS.len(),
            "TUI sees every command"
        );
    }

    #[test]
    fn typing_filters_by_prefix() {
        let mut s = state();
        let c = s.active("/co", 3).expect("menu");
        let names: Vec<_> = c.matches.iter().map(|m| m.name).collect();
        assert_eq!(names, vec!["/cost", "/context", "/compact"]);
    }

    #[test]
    fn no_menu_without_a_leading_slash() {
        let mut s = state();
        assert!(s.active("hello", 5).is_none());
        assert!(s.active("say /help", 9).is_none(), "slash mid-line is text");
    }

    #[test]
    fn no_menu_once_an_argument_is_being_typed() {
        let mut s = state();
        assert!(s.active("/model dev", 10).is_none());
    }

    #[test]
    fn no_menu_for_an_exactly_typed_command() {
        // /compact is unambiguous and complete; a one-item menu is just noise.
        let mut s = state();
        assert!(s.active("/compact", 8).is_none());
    }

    #[test]
    fn an_ambiguous_prefix_offers_every_candidate() {
        let mut s = state();
        let c = s.active("/c", 2).expect("menu");
        assert!(c.matches.len() > 1, "/c matches cost, compact, clear");
    }

    #[test]
    fn unknown_prefix_shows_nothing() {
        let mut s = state();
        assert!(s.active("/zzz", 4).is_none());
    }

    #[test]
    fn selection_wraps_in_both_directions() {
        let mut s = state();
        s.move_selection(-1, 3);
        assert_eq!(s.selected, 2, "up from the top wraps to the bottom");
        s.move_selection(1, 3);
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn selection_is_clamped_when_filtering_shrinks_the_list() {
        // Arrow down the full list, then type a letter that leaves fewer
        // candidates than the stored index.
        let mut s = state();
        let all = s.active("/", 1).expect("menu").matches.len();
        s.move_selection((all - 1) as isize, all);
        assert_eq!(s.selected, all - 1);

        let c = s.active("/co", 3).expect("menu");
        assert!(c.selected < c.matches.len(), "index must be in range");
    }

    #[test]
    fn esc_suppresses_until_the_token_changes() {
        let mut s = state();
        assert!(s.active("/co", 3).is_some());
        s.dismiss("/co");
        assert!(
            s.active("/co", 3).is_none(),
            "stays dismissed while unchanged"
        );
        assert!(s.active("/com", 4).is_some(), "typing re-arms the menu");
    }

    #[test]
    fn accepting_a_command_with_an_argument_leaves_a_trailing_space() {
        let spec = commands::COMMANDS
            .iter()
            .find(|c| c.name == "/model")
            .unwrap();
        let (text, cursor) = apply("/mo", 3, spec);
        assert_eq!(text, "/model ");
        assert_eq!(cursor, 7, "cursor sits ready for the argument");
    }

    #[test]
    fn accepting_a_command_without_an_argument_does_not() {
        let spec = commands::COMMANDS
            .iter()
            .find(|c| c.name == "/compact")
            .unwrap();
        let (text, cursor) = apply("/comp", 5, spec);
        assert_eq!(text, "/compact");
        assert_eq!(cursor, 8, "Enter submits immediately");
    }

    #[test]
    fn accepting_preserves_text_after_the_cursor_without_doubling_the_space() {
        let spec = commands::COMMANDS
            .iter()
            .find(|c| c.name == "/theme")
            .unwrap();
        let (text, cursor) = apply("/th dark", 3, spec);
        assert_eq!(text, "/theme dark");
        assert_eq!(cursor, 6, "cursor lands before the existing space");
    }
}
