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

use crate::config::ResolvedModel;
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
    /// Waiting for explicit confirmation before deleting a session
    ConfirmDelete {
        session_id: String,
        display_name: String,
    },
}

/// An item in the tree view -- either a project header or a session.
#[derive(Debug, Clone)]
enum TreeItem {
    Project {
        name: String,
        expanded: bool,
        session_count: usize,
    },
    Session(Box<SessionInfo>),
}

pub struct HomeScreen {
    db: Db,
    tree: Vec<TreeItem>,
    selected: usize,
    mode: Mode,
    input: String,
    cursor: usize,
    theme: Theme,
    models: Vec<ResolvedModel>,
    selected_model: usize,
    notice: Option<String>,
    /// Empty projects created by the user (no sessions yet)
    empty_projects: Vec<String>,
}

impl HomeScreen {
    pub fn new(db: Db, theme: Theme, models: Vec<ResolvedModel>) -> Self {
        debug_assert!(!models.is_empty());
        let mut screen = Self {
            db,
            tree: Vec::new(),
            selected: 0,
            mode: Mode::Browse,
            input: String::new(),
            cursor: 0,
            theme,
            models,
            selected_model: 0,
            notice: None,
            empty_projects: Vec::new(),
        };
        let _ = screen.reload();
        screen
    }

    fn selected_model(&self) -> &ResolvedModel {
        &self.models[self.selected_model]
    }

    pub fn set_notice(&mut self, notice: impl Into<String>) {
        self.notice = Some(notice.into());
    }

    fn select_next_model(&mut self) {
        self.selected_model = (self.selected_model + 1) % self.models.len();
    }

    fn select_previous_model(&mut self) {
        self.selected_model = self
            .selected_model
            .checked_sub(1)
            .unwrap_or(self.models.len() - 1);
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
                    self.tree.push(TreeItem::Session(Box::new(session)));
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
                    match self.mode.clone() {
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
                        Mode::ConfirmDelete { .. } => {
                            self.handle_delete_confirmation(key)?;
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

            (_, KeyCode::Up) | (_, KeyCode::Char('k')) if self.selected > 0 => {
                self.selected -= 1;
            }

            (_, KeyCode::Down) | (_, KeyCode::Char('j')) if self.selected + 1 < self.tree.len() => {
                self.selected += 1;
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
                    let display_name = session
                        .name
                        .filter(|name| !name.is_empty())
                        .unwrap_or_else(|| session.id.clone());
                    self.mode = Mode::ConfirmDelete {
                        session_id: session.id,
                        display_name,
                    };
                }
            }

            _ => {}
        }

        Ok(None)
    }

    fn handle_delete_confirmation(&mut self, key: KeyEvent) -> Result<()> {
        let Mode::ConfirmDelete { session_id, .. } = self.mode.clone() else {
            return Ok(());
        };

        match key.code {
            KeyCode::Char('y') => {
                self.db.delete_session(&session_id)?;
                self.mode = Mode::Browse;
                self.reload()?;
            }
            KeyCode::Char('n') | KeyCode::Esc => {
                self.mode = Mode::Browse;
            }
            _ => {}
        }

        Ok(())
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
                        let session_id = crate::session::new_session_id();
                        self.db.create_session_with_binding(
                            &session_id,
                            &self.selected_model().binding,
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
            KeyCode::Backspace if self.cursor > 0 => {
                super::input::backspace(&mut self.input, &mut self.cursor);
            }
            KeyCode::Tab | KeyCode::Right if self.mode == Mode::NewSession => {
                self.select_next_model();
            }
            KeyCode::BackTab | KeyCode::Left if self.mode == Mode::NewSession => {
                self.select_previous_model();
            }
            KeyCode::Char(c) => {
                super::input::insert(&mut self.input, &mut self.cursor, c);
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
                    let model_label = session
                        .model_binding
                        .as_ref()
                        .map(|binding| {
                            format!("{} · {}", binding.display_name, binding.provider_name)
                        })
                        .unwrap_or_else(|| session.model.clone());
                    let model_short = model_label.chars().take(24).collect::<String>();

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
                            format!(" {model_short} "),
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
        match &self.mode {
            Mode::NewSession => {
                let prompt = format!(
                    "Name: {}  Model: {} (Tab to change)",
                    self.input,
                    self.selected_model().binding.display_name
                );
                let input_widget = Paragraph::new(prompt)
                    .style(Style::default().fg(self.theme.fg))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(self.theme.info))
                            .title(" New Session "),
                    );
                f.render_widget(input_widget, chunks[2]);
                let cursor_width = super::input::display_width_before(&self.input, self.cursor);
                f.set_cursor_position((chunks[2].x + 7 + cursor_width as u16, chunks[2].y + 1));
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
                let cursor_width = super::input::display_width_before(&self.input, self.cursor);
                f.set_cursor_position((chunks[2].x + 15 + cursor_width as u16, chunks[2].y + 1));
            }
            Mode::ConfirmDelete { display_name, .. } => {
                let prompt = Paragraph::new(Line::from(vec![
                    Span::styled(
                        format!(" Delete \"{display_name}\"? "),
                        Style::default()
                            .fg(self.theme.error)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("(y)es / (n)o", Style::default().fg(self.theme.dim)),
                ]))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(self.theme.error))
                        .title(" Confirm Delete "),
                );
                f.render_widget(prompt, chunks[2]);
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
        let status_text = self.notice.as_deref().map_or_else(
            || {
                format!(
                    " {} · {} ",
                    self.selected_model().binding.display_name,
                    self.selected_model().binding.provider_name
                )
            },
            |notice| format!(" {notice} "),
        );
        let status = Paragraph::new(Line::from(vec![Span::styled(
            status_text,
            Style::default().fg(if self.notice.is_some() {
                self.theme.error
            } else {
                self.theme.dim
            }),
        )]));
        f.render_widget(status, chunks[3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models(names: &[&str]) -> Vec<ResolvedModel> {
        let config = crate::config::Config::default();
        names
            .iter()
            .map(|name| config.resolve_model(name).unwrap())
            .collect()
    }

    fn screen_with_session() -> (HomeScreen, tempfile::TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let db = Db::open(&temp.path().join("test.db")).unwrap();
        db.create_session(
            "session-1",
            "model",
            Some("Important work"),
            Some("project"),
        )
        .unwrap();
        let mut screen = HomeScreen::new(db, Theme::dark(), models(&["model"]));
        screen.selected = screen
            .tree
            .iter()
            .position(|item| matches!(item, TreeItem::Session(_)))
            .unwrap();
        (screen, temp)
    }

    fn key(character: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE)
    }

    #[test]
    fn delete_requires_explicit_confirmation() {
        let (mut screen, _temp) = screen_with_session();

        screen.handle_browse_key(key('d')).unwrap();

        assert!(matches!(screen.mode, Mode::ConfirmDelete { .. }));
        assert!(screen.db.get_session("session-1").unwrap().is_some());

        screen.handle_delete_confirmation(key('y')).unwrap();

        assert_eq!(screen.mode, Mode::Browse);
        assert!(screen.db.get_session("session-1").unwrap().is_none());
    }

    #[test]
    fn delete_confirmation_can_be_cancelled() {
        let (mut screen, _temp) = screen_with_session();
        screen.handle_browse_key(key('d')).unwrap();

        screen.handle_delete_confirmation(key('n')).unwrap();

        assert_eq!(screen.mode, Mode::Browse);
        assert!(screen.db.get_session("session-1").unwrap().is_some());
    }

    #[test]
    fn new_session_uses_selected_model() {
        let temp = tempfile::tempdir().unwrap();
        let db = Db::open(&temp.path().join("test.db")).unwrap();
        let mut screen = HomeScreen::new(db, Theme::dark(), models(&["model-a", "model-b"]));
        screen.mode = Mode::NewSession;
        screen.select_next_model();

        let action = screen
            .handle_prompt_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .unwrap()
            .unwrap();
        let Action::Chat { session_id } = action else {
            panic!("expected chat action");
        };
        assert_eq!(
            screen.db.get_session(&session_id).unwrap().unwrap().model,
            "model-b"
        );
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

    fn models() -> Vec<ResolvedModel> {
        let config = crate::config::Config::default();
        ["claude-sonnet-4-20250514", "gpt-4o"]
            .into_iter()
            .map(|model| config.resolve_model(model).unwrap())
            .collect()
    }

    /// Build a temp-file-backed Db seeded with a known set of projects/sessions.
    fn seeded_db() -> (Db, tempfile::NamedTempFile) {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let db = Db::open(&tmp.path().to_path_buf()).expect("open db");
        db.create_session(
            "20260101-120000",
            "claude-sonnet-4-20250514",
            Some("auth refactor"),
            Some("claux"),
        )
        .unwrap();
        db.create_session(
            "20260102-093000",
            "claude-sonnet-4-20250514",
            Some("tui polish"),
            Some("claux"),
        )
        .unwrap();
        db.create_session(
            "20260103-160000",
            "gpt-4o",
            Some("ssac brainstorm"),
            Some("tuishot"),
        )
        .unwrap();
        (db, tmp)
    }

    #[derive(Tuishot)]
    enum HomeShot {
        #[tuishot(name = "home-empty", description = "First launch: no sessions yet")]
        Empty,

        #[tuishot(
            name = "home-populated",
            description = "Session browser with two projects"
        )]
        Populated,

        #[tuishot(
            name = "home-new-session",
            description = "Creating a new session in the selected project"
        )]
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
                    (HomeScreen::new(db, theme, models()), Some(tmp))
                }
                _ => {
                    let (db, tmp) = seeded_db();
                    (HomeScreen::new(db, theme, models()), Some(tmp))
                }
            };
            match self {
                HomeShot::NewSession => {
                    screen.mode = Mode::NewSession;
                    screen.input = String::from("refactor queue");
                    screen.cursor = crate::tui::input::char_count(&screen.input);
                }
                HomeShot::NewProject => {
                    screen.mode = Mode::NewProject;
                    screen.input = String::from("hosted-resumes");
                    screen.cursor = crate::tui::input::char_count(&screen.input);
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
