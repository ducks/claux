//! Home screen: session browser and project navigator.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::collections::HashMap;
use std::io::Stdout;

use crate::db::{Db, SessionInfo};
use crate::theme::Theme;

use super::screen::Action;

/// What the home screen is doing.
#[derive(Debug, Clone, PartialEq)]
enum Mode {
    /// Browsing projects/sessions
    Browse,
    /// Typing a name for a new session
    NewSession,
    /// Typing a name for a new project
    NewProject,
}

/// An item in the tree view -- either a project header or a session.
#[derive(Debug, Clone)]
enum TreeItem {
    Project {
        name: String,
        expanded: bool,
        session_count: usize,
    },
    Session(SessionInfo),
}

pub struct HomeScreen {
    db: Db,
    tree: Vec<TreeItem>,
    selected: usize,
    mode: Mode,
    input: String,
    cursor: usize,
    theme: Theme,
    model: String,
    /// Empty projects created by the user (no sessions yet)
    empty_projects: Vec<String>,
}

impl HomeScreen {
    pub fn new(db: Db, theme: Theme, model: &str) -> Self {
        let mut screen = Self {
            db,
            tree: Vec::new(),
            selected: 0,
            mode: Mode::Browse,
            input: String::new(),
            cursor: 0,
            theme,
            model: model.to_string(),
            empty_projects: Vec::new(),
        };
        let _ = screen.reload();
        screen
    }

    /// Reload sessions from DB and rebuild the tree.
    fn reload(&mut self) -> Result<()> {
        let sessions = self.db.list_sessions()?;

        // Group by project
        let mut by_project: HashMap<String, Vec<SessionInfo>> = HashMap::new();
        for session in sessions {
            by_project
                .entry(session.project.clone())
                .or_default()
                .push(session);
        }

        // Preserve expanded state from old tree
        let was_expanded: HashMap<String, bool> = self
            .tree
            .iter()
            .filter_map(|item| {
                if let TreeItem::Project { name, expanded, .. } = item {
                    Some((name.clone(), *expanded))
                } else {
                    None
                }
            })
            .collect();

        // Build tree: sorted projects, including empty ones
        let mut projects: Vec<String> = by_project.keys().cloned().collect();
        for ep in &self.empty_projects {
            if !projects.contains(ep) {
                projects.push(ep.clone());
            }
        }
        projects.sort();

        self.tree.clear();
        for project in projects {
            let sessions = by_project.get(&project).cloned().unwrap_or_default();
            let expanded = was_expanded.get(&project).copied().unwrap_or(true);
            let session_count = sessions.len();

            self.tree.push(TreeItem::Project {
                name: project,
                expanded,
                session_count,
            });

            if expanded {
                for session in sessions {
                    self.tree.push(TreeItem::Session(session));
                }
            }
        }

        // Clamp selection
        if !self.tree.is_empty() && self.selected >= self.tree.len() {
            self.selected = self.tree.len() - 1;
        }

        Ok(())
    }

    /// Get the project that the currently selected item belongs to.
    fn selected_project(&self) -> Option<String> {
        // Walk backwards from selected to find the project header
        for i in (0..=self.selected).rev() {
            if let Some(TreeItem::Project { name, .. }) = self.tree.get(i) {
                return Some(name.clone());
            }
        }
        None
    }

    /// Run the home screen event loop. Returns an Action.
    pub fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<Action> {
        loop {
            terminal.draw(|f| self.draw(f))?;

            if event::poll(std::time::Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match self.mode {
                        Mode::Browse => {
                            if let Some(action) = self.handle_browse_key(key)? {
                                return Ok(action);
                            }
                        }
                        Mode::NewSession | Mode::NewProject => {
                            if let Some(action) = self.handle_prompt_key(key)? {
                                return Ok(action);
                            }
                        }
                    }
                }
            }
        }
    }

    fn handle_browse_key(&mut self, key: KeyEvent) -> Result<Option<Action>> {
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c'))
            | (KeyModifiers::CONTROL, KeyCode::Char('d'))
            | (_, KeyCode::Char('q')) => {
                return Ok(Some(Action::Quit));
            }

            (_, KeyCode::Up) | (_, KeyCode::Char('k')) => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
            }

            (_, KeyCode::Down) | (_, KeyCode::Char('j')) => {
                if self.selected + 1 < self.tree.len() {
                    self.selected += 1;
                }
            }

            (_, KeyCode::Enter) => {
                if let Some(item) = self.tree.get(self.selected).cloned() {
                    match item {
                        TreeItem::Project { expanded, .. } => {
                            // Toggle expand/collapse
                            if let Some(TreeItem::Project {
                                expanded: ref mut exp,
                                ..
                            }) = self.tree.get_mut(self.selected)
                            {
                                *exp = !expanded;
                            }
                            self.reload()?;
                        }
                        TreeItem::Session(session) => {
                            return Ok(Some(Action::Chat {
                                session_id: session.id,
                            }));
                        }
                    }
                }
            }

            (_, KeyCode::Char('n')) => {
                self.mode = Mode::NewSession;
                self.input.clear();
                self.cursor = 0;
            }

            (_, KeyCode::Char('p')) => {
                self.mode = Mode::NewProject;
                self.input.clear();
                self.cursor = 0;
            }

            (_, KeyCode::Char('d')) => {
                if let Some(TreeItem::Session(session)) = self.tree.get(self.selected).cloned() {
                    self.db.delete_session(&session.id)?;
                    self.reload()?;
                }
            }

            _ => {}
        }

        Ok(None)
    }

    fn handle_prompt_key(&mut self, key: KeyEvent) -> Result<Option<Action>> {
        match key.code {
            KeyCode::Esc => {
                self.mode = Mode::Browse;
                self.input.clear();
            }
            KeyCode::Enter => {
                let name = if self.input.is_empty() {
                    chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string()
                } else {
                    self.input.clone()
                };

                match self.mode {
                    Mode::NewSession => {
                        let project = self
                            .selected_project()
                            .unwrap_or_else(|| "uncategorized".to_string());
                        let session_id = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
                        self.db.create_session(
                            &session_id,
                            &self.model,
                            Some(&name),
                            Some(&project),
                        )?;
                        self.mode = Mode::Browse;
                        self.input.clear();
                        self.reload()?;
                        // Jump into the new session
                        return Ok(Some(Action::Chat { session_id }));
                    }
                    Mode::NewProject => {
                        if !name.is_empty() {
                            self.empty_projects.push(name);
                        }
                        self.mode = Mode::Browse;
                        self.input.clear();
                        self.reload()?;
                    }
                    _ => {}
                }
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.input.remove(self.cursor);
                }
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
            }
            _ => {}
        }

        Ok(None)
    }

    fn draw(&mut self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(1),    // Tree view
                Constraint::Length(3), // Input / help
                Constraint::Length(1), // Status
            ])
            .split(f.area());

        // Header
        let session_count: usize = self
            .tree
            .iter()
            .filter(|i| matches!(i, TreeItem::Session(_)))
            .count();
        let project_count: usize = self
            .tree
            .iter()
            .filter(|i| matches!(i, TreeItem::Project { .. }))
            .count();

        let header = Paragraph::new(Line::from(vec![
            Span::styled(
                " claux ",
                Style::default()
                    .fg(self.theme.assistant_bold)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {project_count} projects, {session_count} sessions"),
                Style::default().fg(self.theme.dim),
            ),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(self.theme.border)),
        );
        f.render_widget(header, chunks[0]);

        // Tree view
        let mut lines: Vec<Line> = Vec::new();
        for (i, item) in self.tree.iter().enumerate() {
            let selected = i == self.selected;
            let highlight = if selected {
                Style::default()
                    .fg(self.theme.fg)
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else {
                Style::default()
            };

            match item {
                TreeItem::Project {
                    name,
                    expanded,
                    session_count,
                } => {
                    let icon = if *expanded { "▼" } else { "▶" };
                    let line = Line::from(vec![
                        Span::styled(
                            format!(" {icon} {name} "),
                            if selected {
                                highlight
                            } else {
                                Style::default()
                                    .fg(self.theme.info)
                                    .add_modifier(Modifier::BOLD)
                            },
                        ),
                        Span::styled(
                            format!("({session_count})"),
                            Style::default().fg(self.theme.dim),
                        ),
                    ]);
                    lines.push(line);
                }
                TreeItem::Session(session) => {
                    let display_name = session
                        .name
                        .as_deref()
                        .filter(|n| !n.is_empty())
                        .unwrap_or(&session.id);
                    let model_short = if session.model.len() > 15 {
                        &session.model[..15]
                    } else {
                        &session.model
                    };

                    let line = Line::from(vec![
                        Span::styled(
                            format!("     {display_name} "),
                            if selected {
                                highlight
                            } else {
                                Style::default().fg(self.theme.fg)
                            },
                        ),
                        Span::styled(
                            format!(" {model_short} ",),
                            Style::default().fg(self.theme.dim),
                        ),
                        Span::styled(
                            format!("{}msgs", session.message_count),
                            Style::default().fg(self.theme.dim),
                        ),
                    ]);
                    lines.push(line);
                }
            }
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No sessions yet. Press n to create one.",
                Style::default().fg(self.theme.dim),
            )));
        }

        let tree_widget = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(self.theme.border))
                .title(" Sessions "),
        );
        f.render_widget(tree_widget, chunks[1]);

        // Input / help area
        match self.mode {
            Mode::NewSession => {
                let prompt = format!("Session name (Enter for timestamp): {}", self.input);
                let input_widget = Paragraph::new(prompt)
                    .style(Style::default().fg(self.theme.fg))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(self.theme.info))
                            .title(" New Session "),
                    );
                f.render_widget(input_widget, chunks[2]);
                f.set_cursor_position((chunks[2].x + 39 + self.cursor as u16, chunks[2].y + 1));
            }
            Mode::NewProject => {
                let prompt = format!("Project name: {}", self.input);
                let input_widget = Paragraph::new(prompt)
                    .style(Style::default().fg(self.theme.fg))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(self.theme.info))
                            .title(" New Project "),
                    );
                f.render_widget(input_widget, chunks[2]);
                f.set_cursor_position((chunks[2].x + 15 + self.cursor as u16, chunks[2].y + 1));
            }
            Mode::Browse => {
                let help = Paragraph::new(Line::from(vec![
                    Span::styled(" n", Style::default().fg(self.theme.info)),
                    Span::styled(":new  ", Style::default().fg(self.theme.dim)),
                    Span::styled("p", Style::default().fg(self.theme.info)),
                    Span::styled(":project  ", Style::default().fg(self.theme.dim)),
                    Span::styled("d", Style::default().fg(self.theme.info)),
                    Span::styled(":delete  ", Style::default().fg(self.theme.dim)),
                    Span::styled("Enter", Style::default().fg(self.theme.info)),
                    Span::styled(":open  ", Style::default().fg(self.theme.dim)),
                    Span::styled("q", Style::default().fg(self.theme.info)),
                    Span::styled(":quit", Style::default().fg(self.theme.dim)),
                ]))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(self.theme.border)),
                );
                f.render_widget(help, chunks[2]);
            }
        }

        // Status
        let status = Paragraph::new(Line::from(vec![Span::styled(
            format!(" {} ", self.model),
            Style::default().fg(self.theme.dim),
        )]));
        f.render_widget(status, chunks[3]);
    }
}

// ---------------------------------------------------------------------------
// tuishot integration: capture canonical HomeScreen states as SVG for docs.
//
// Run `cargo test --test tuishot_capture` (or the module tests) to verify that
// committed screenshots still match. Set `TUISHOT_UPDATE=1` to accept drift.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tuishot_shots {
    use super::*;
    use tuishot::Tuishot;

    /// Build a temp-file-backed Db seeded with a known set of projects/sessions.
    fn seeded_db() -> (Db, tempfile::NamedTempFile) {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let db = Db::open(&tmp.path().to_path_buf()).expect("open db");
        db.create_session("20260101-120000", "claude-sonnet-4-20250514", Some("auth refactor"), Some("claux"))
            .unwrap();
        db.create_session("20260102-093000", "claude-sonnet-4-20250514", Some("tui polish"), Some("claux"))
            .unwrap();
        db.create_session("20260103-160000", "gpt-4o", Some("ssac brainstorm"), Some("tuishot"))
            .unwrap();
        (db, tmp)
    }

    #[derive(Tuishot)]
    enum HomeShot {
        #[tuishot(name = "home-empty", description = "First launch: no sessions yet")]
        Empty,

        #[tuishot(name = "home-populated", description = "Session browser with two projects")]
        Populated,

        #[tuishot(name = "home-new-session", description = "Creating a new session in the selected project")]
        NewSession,

        #[tuishot(name = "home-new-project", description = "Creating a new project")]
        NewProject,
    }

    impl HomeShotRender for HomeShot {
        fn render(&self, buf: &mut ratatui::buffer::Buffer, area: ratatui::layout::Rect) {
            let theme = Theme::dark();
            let (mut screen, _keepalive) = match self {
                HomeShot::Empty => {
                    let tmp = tempfile::NamedTempFile::new().unwrap();
                    let db = Db::open(&tmp.path().to_path_buf()).unwrap();
                    (HomeScreen::new(db, theme, "claude-sonnet-4-20250514"), Some(tmp))
                }
                _ => {
                    let (db, tmp) = seeded_db();
                    (HomeScreen::new(db, theme, "claude-sonnet-4-20250514"), Some(tmp))
                }
            };
            match self {
                HomeShot::NewSession => {
                    screen.mode = Mode::NewSession;
                    screen.input = String::from("refactor queue");
                    screen.cursor = screen.input.len();
                }
                HomeShot::NewProject => {
                    screen.mode = Mode::NewProject;
                    screen.input = String::from("hosted-resumes");
                    screen.cursor = screen.input.len();
                }
                _ => {}
            }
            // Render through a TestBackend so the same draw() powers the capture.
            let rendered = tuishot::render_to_buffer(area.width, area.height, |f| {
                screen.draw(f);
            });
            buf.clone_from(&rendered);
        }
    }

    #[test]
    fn capture_home_screens() {
        HomeShot::check_all().expect("home screen capture");
    }
}
