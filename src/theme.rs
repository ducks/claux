//! Theme system for claux.
//!
//! Provides RGB color definitions for the TUI with support for multiple themes
//! (dark, light, ansi) and runtime switching.

use ratatui::style::Color;

/// Available theme names.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ThemeName {
    /// Dark theme (default) - gruvbox-inspired
    #[default]
    Dark,
    /// Light theme
    Light,
    /// ANSI 16-color fallback
    Ansi,
}

/// Color palette for a theme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Theme {
    // UI elements
    pub fg: Color,
    pub bg: Color,
    pub dim: Color,
    pub bold: Color,

    // Semantic colors
    pub success: Color,
    pub error: Color,
    pub warning: Color,
    pub info: Color,

    // Assistant/user message colors
    pub assistant: Color,
    pub assistant_bold: Color,
    pub user: Color,
    pub user_bold: Color,

    // Tool colors
    pub tool_name: Color,
    pub tool_summary: Color,
    pub tool_success: Color,
    pub tool_error: Color,

    // Diff colors
    pub diff_added: Color,
    pub diff_removed: Color,
    pub diff_added_dim: Color,
    pub diff_removed_dim: Color,

    // Status bar
    pub status_bg: Color,
    pub status_fg: Color,
    pub thinking: Color,

    // Borders and dividers
    pub border: Color,
    pub border_bold: Color,
    pub divider: Color,
}

impl Theme {
    /// Get the default dark theme.
    pub fn dark() -> Self {
        Self {
            fg: Color::Rgb(213, 196, 161),   // gruvbox fg2
            bg: Color::Rgb(40, 40, 40),       // gruvbox bg
            dim: Color::Rgb(146, 131, 116),   // gruvbox gray
            bold: Color::Rgb(250, 233, 213),  // gruvbox fg1

            success: Color::Rgb(184, 187, 38), // gruvbox green
            error: Color::Rgb(251, 73, 52),    // gruvbox red
            warning: Color::Rgb(250, 189, 47), // gruvbox yellow
            info: Color::Rgb(131, 165, 152),   // gruvbox blue

            assistant: Color::Rgb(131, 165, 152), // gruvbox blue (purple alternative)
            assistant_bold: Color::Rgb(211, 134, 155), // gruvbox purple
            user: Color::Rgb(184, 187, 38),    // gruvbox green
            user_bold: Color::Rgb(250, 233, 213), // gruvbox fg1

            tool_name: Color::Rgb(250, 189, 47), // gruvbox yellow
            tool_summary: Color::Rgb(146, 131, 116), // gruvbox gray
            tool_success: Color::Rgb(184, 187, 38), // gruvbox green
            tool_error: Color::Rgb(251, 73, 52),    // gruvbox red

            diff_added: Color::Rgb(152, 195, 121), // lighter green
            diff_removed: Color::Rgb(224, 108, 117), // lighter red
            diff_added_dim: Color::Rgb(99, 166, 71), // darker green
            diff_removed_dim: Color::Rgb(184, 68, 77), // darker red

            status_bg: Color::Rgb(60, 60, 60),
            status_fg: Color::Rgb(213, 196, 161),
            thinking: Color::Rgb(184, 187, 38), // gruvbox green

            border: Color::Rgb(100, 100, 100),
            border_bold: Color::Rgb(180, 180, 180),
            divider: Color::Rgb(80, 80, 80),
        }
    }

    /// Get the light theme.
    pub fn light() -> Self {
        Self {
            fg: Color::Rgb(60, 60, 60),       // dark gray
            bg: Color::Rgb(250, 250, 250),    // off-white
            dim: Color::Rgb(140, 140, 140),   // medium gray
            bold: Color::Rgb(30, 30, 30),     // near black

            success: Color::Rgb(84, 128, 0),  // dark green
            error: Color::Rgb(200, 60, 60),   // red
            warning: Color::Rgb(200, 150, 0), // amber
            info: Color::Rgb(40, 80, 180),    // blue

            assistant: Color::Rgb(80, 120, 180), // blue
            assistant_bold: Color::Rgb(140, 80, 160), // purple
            user: Color::Rgb(84, 128, 0),     // green
            user_bold: Color::Rgb(30, 30, 30), // near black

            tool_name: Color::Rgb(200, 150, 0), // amber
            tool_summary: Color::Rgb(140, 140, 140), // gray
            tool_success: Color::Rgb(84, 128, 0), // green
            tool_error: Color::Rgb(200, 60, 60), // red

            diff_added: Color::Rgb(120, 180, 60),
            diff_removed: Color::Rgb(200, 100, 100),
            diff_added_dim: Color::Rgb(180, 220, 150),
            diff_removed_dim: Color::Rgb(240, 200, 200),

            status_bg: Color::Rgb(220, 220, 220),
            status_fg: Color::Rgb(60, 60, 60),
            thinking: Color::Rgb(84, 128, 0), // green

            border: Color::Rgb(180, 180, 180),
            border_bold: Color::Rgb(100, 100, 100),
            divider: Color::Rgb(200, 200, 200),
        }
    }

    /// Get the ANSI 16-color theme (fallback for limited terminals).
    pub fn ansi() -> Self {
        Self {
            fg: Color::White,
            bg: Color::Black,
            dim: Color::DarkGray,
            bold: Color::White,

            success: Color::Green,
            error: Color::Red,
            warning: Color::Yellow,
            info: Color::Blue,

            assistant: Color::Blue,
            assistant_bold: Color::Magenta,
            user: Color::Green,
            user_bold: Color::White,

            tool_name: Color::Yellow,
            tool_summary: Color::DarkGray,
            tool_success: Color::Green,
            tool_error: Color::Red,

            diff_added: Color::Green,
            diff_removed: Color::Red,
            diff_added_dim: Color::Rgb(99, 166, 71), // darker green fallback
            diff_removed_dim: Color::Rgb(184, 68, 77), // darker red fallback

            status_bg: Color::DarkGray,
            status_fg: Color::White,
            thinking: Color::Green,

            border: Color::White,
            border_bold: Color::White,
            divider: Color::DarkGray,
        }
    }

    /// Get the theme by name.
    pub fn from_name(name: ThemeName) -> Self {
        match name {
            ThemeName::Dark => Self::dark(),
            ThemeName::Light => Self::light(),
            ThemeName::Ansi => Self::ansi(),
        }
    }
}

/// Get the default theme.
pub fn default_theme() -> Theme {
    Theme::dark()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dark_theme_colors() {
        let theme = Theme::dark();
        assert_eq!(theme.fg, Color::Rgb(213, 196, 161));
        assert_eq!(theme.bg, Color::Rgb(40, 40, 40));
        assert_eq!(theme.success, Color::Rgb(184, 187, 38));
    }

    #[test]
    fn test_light_theme_colors() {
        let theme = Theme::light();
        assert_eq!(theme.fg, Color::Rgb(60, 60, 60));
        assert_eq!(theme.bg, Color::Rgb(250, 250, 250));
        assert_eq!(theme.success, Color::Rgb(84, 128, 0));
    }

    #[test]
    fn test_theme_from_name() {
        assert_eq!(Theme::from_name(ThemeName::Dark), Theme::dark());
        assert_eq!(Theme::from_name(ThemeName::Light), Theme::light());
        assert_eq!(Theme::from_name(ThemeName::Ansi), Theme::ansi());
    }
}
