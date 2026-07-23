use std::path::{Path, PathBuf};

use crate::permissions::PermissionMode;

/// Trust state for configuration sourced from the current project.
///
/// Project files are untrusted by default because merely opening a checkout
/// must not grant it broader tool permissions or execute its MCP commands.
#[derive(Debug, Clone)]
pub struct ProjectTrust {
    cwd: PathBuf,
    trusted: bool,
}

impl ProjectTrust {
    pub fn resolve(force_trust: bool, trusted_projects: &[PathBuf]) -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let trusted = force_trust
            || trusted_projects
                .iter()
                .any(|project| paths_match(&cwd, project));
        Self { cwd, trusted }
    }

    #[cfg(test)]
    pub(crate) fn for_test(cwd: PathBuf, trusted: bool) -> Self {
        Self { cwd, trusted }
    }

    pub fn is_trusted(&self) -> bool {
        self.trusted
    }

    pub fn project_file(&self, name: &str) -> PathBuf {
        self.cwd.join(name)
    }
}

/// A project may make its permission policy stricter without trust. Loosening
/// the global policy requires an explicit trust decision.
pub fn permits_permission_override(
    current: PermissionMode,
    requested: PermissionMode,
    trusted: bool,
) -> bool {
    trusted || permission_rank(requested) <= permission_rank(current)
}

fn permission_rank(mode: PermissionMode) -> u8 {
    match mode {
        PermissionMode::Plan => 0,
        PermissionMode::Default => 1,
        PermissionMode::AcceptEdits => 2,
        PermissionMode::Bypass => 3,
    }
}

fn paths_match(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrusted_project_can_tighten_permissions() {
        assert!(permits_permission_override(
            PermissionMode::Bypass,
            PermissionMode::Default,
            false
        ));
        assert!(permits_permission_override(
            PermissionMode::Default,
            PermissionMode::Plan,
            false
        ));
    }

    #[test]
    fn untrusted_project_cannot_loosen_permissions() {
        assert!(!permits_permission_override(
            PermissionMode::Default,
            PermissionMode::AcceptEdits,
            false
        ));
        assert!(!permits_permission_override(
            PermissionMode::Plan,
            PermissionMode::Bypass,
            false
        ));
    }

    #[test]
    fn trusted_project_can_choose_any_mode() {
        assert!(permits_permission_override(
            PermissionMode::Plan,
            PermissionMode::Bypass,
            true
        ));
    }
}
