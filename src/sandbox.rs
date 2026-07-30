//! Application-level filesystem containment for native tools.
//!
//! This policy does not sandbox child processes. It constrains filesystem
//! operations performed directly by Claux; process isolation is a separate
//! layer.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Component, Path, PathBuf};

/// Filesystem containment applied to Claux's built-in file tools.
///
/// This does not apply to Bash commands or MCP tools. Those are separate
/// execution boundaries and continue to use the permission system.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeToolFilesystemPolicy {
    #[default]
    WorkspaceOnly,
    Unrestricted,
}

impl NativeToolFilesystemPolicy {
    pub fn permits_project_override(self, requested: Self, trusted: bool) -> bool {
        trusted
            || self == requested
            || matches!(
                (self, requested),
                (
                    NativeToolFilesystemPolicy::Unrestricted,
                    Self::WorkspaceOnly
                )
            )
    }
}

impl fmt::Display for NativeToolFilesystemPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceOnly => formatter.write_str("workspace_only"),
            Self::Unrestricted => formatter.write_str("unrestricted"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SandboxPolicy {
    workspace_root: PathBuf,
    read_roots: Vec<PathBuf>,
    write_roots: Vec<PathBuf>,
    unrestricted: bool,
}

impl SandboxPolicy {
    pub fn from_native_tool_policy(
        policy: NativeToolFilesystemPolicy,
        workspace_root: impl AsRef<Path>,
    ) -> Result<Self> {
        match policy {
            NativeToolFilesystemPolicy::WorkspaceOnly => Self::workspace_only(workspace_root),
            NativeToolFilesystemPolicy::Unrestricted => Self::unrestricted(workspace_root),
        }
    }

    pub fn workspace_only(workspace_root: impl AsRef<Path>) -> Result<Self> {
        let workspace_root = canonical_workspace_root(workspace_root)?;

        Ok(Self {
            read_roots: vec![workspace_root.clone()],
            write_roots: vec![workspace_root.clone()],
            workspace_root,
            unrestricted: false,
        })
    }

    pub fn unrestricted(workspace_root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            workspace_root: canonical_workspace_root(workspace_root)?,
            read_roots: Vec::new(),
            write_roots: Vec::new(),
            unrestricted: true,
        })
    }

    #[cfg(test)]
    pub(crate) fn unrestricted_for_tests() -> Self {
        Self::unrestricted(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
            .expect("test working directory must resolve")
    }

    pub fn authorize_read(&self, requested: impl AsRef<Path>) -> Result<PathBuf> {
        let requested = requested.as_ref();
        let resolved = self.resolve_existing(requested)?;
        self.ensure_allowed(requested, &resolved, &self.read_roots, "read")?;
        Ok(resolved)
    }

    pub fn authorize_write(&self, requested: impl AsRef<Path>) -> Result<PathBuf> {
        let requested = requested.as_ref();
        let resolved = self.resolve_for_write(requested)?;
        self.ensure_allowed(requested, &resolved, &self.write_roots, "write")?;
        Ok(resolved)
    }

    pub fn authorize_search_pattern(&self, pattern: &str) -> Result<()> {
        if self.unrestricted {
            return Ok(());
        }
        if Path::new(pattern).components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            bail!("search pattern escapes the authorized base directory: {pattern}");
        }
        Ok(())
    }

    fn resolve_existing(&self, requested: &Path) -> Result<PathBuf> {
        let absolute = self.absolute_path(requested);
        absolute
            .canonicalize()
            .with_context(|| format!("could not resolve path {}", requested.display()))
    }

    fn resolve_for_write(&self, requested: &Path) -> Result<PathBuf> {
        let absolute = self.absolute_path(requested);
        if absolute.exists() {
            return absolute
                .canonicalize()
                .with_context(|| format!("could not resolve path {}", requested.display()));
        }

        let mut ancestor = absolute.as_path();
        let mut missing_components = Vec::new();
        while !ancestor.exists() {
            let name = ancestor.file_name().ok_or_else(|| {
                anyhow::anyhow!("could not resolve a parent for {}", requested.display())
            })?;
            missing_components.push(name.to_os_string());
            ancestor = ancestor.parent().ok_or_else(|| {
                anyhow::anyhow!("could not resolve a parent for {}", requested.display())
            })?;
        }

        let mut resolved = ancestor
            .canonicalize()
            .with_context(|| format!("could not resolve path {}", ancestor.display()))?;
        for component in missing_components.into_iter().rev() {
            resolved.push(component);
        }
        Ok(resolved)
    }

    fn absolute_path(&self, requested: &Path) -> PathBuf {
        let expanded = expand_tilde(requested);
        if expanded.is_absolute() {
            expanded
        } else {
            self.workspace_root.join(expanded)
        }
    }

    fn ensure_allowed(
        &self,
        requested: &Path,
        resolved: &Path,
        roots: &[PathBuf],
        operation: &str,
    ) -> Result<()> {
        if self.unrestricted || roots.iter().any(|root| resolved.starts_with(root)) {
            return Ok(());
        }

        bail!(
            "sandbox denied {operation} access to {} (resolved to {})",
            requested.display(),
            resolved.display()
        )
    }
}

fn canonical_workspace_root(workspace_root: impl AsRef<Path>) -> Result<PathBuf> {
    let workspace_root = workspace_root.as_ref().canonicalize().with_context(|| {
        format!(
            "could not resolve workspace root {}",
            workspace_root.as_ref().display()
        )
    })?;
    if !workspace_root.is_dir() {
        bail!(
            "workspace root is not a directory: {}",
            workspace_root.display()
        );
    }
    Ok(workspace_root)
}

fn expand_tilde(path: &Path) -> PathBuf {
    let Some(path) = path.to_str() else {
        return path.to_path_buf();
    };
    let Some(stripped) = path.strip_prefix("~/") else {
        return PathBuf::from(path);
    };
    dirs::home_dir()
        .map(|home| home.join(stripped))
        .unwrap_or_else(|| PathBuf::from(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_policy_allows_paths_beneath_the_root() {
        let workspace = tempfile::tempdir().unwrap();
        let file = workspace.path().join("src/lib.rs");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "contents").unwrap();
        let policy = SandboxPolicy::workspace_only(workspace.path()).unwrap();

        assert_eq!(policy.authorize_read(&file).unwrap(), file);
        assert_eq!(
            policy
                .authorize_write(workspace.path().join("new/file.rs"))
                .unwrap(),
            workspace.path().join("new/file.rs")
        );
    }

    #[test]
    fn workspace_policy_rejects_parent_traversal() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let outside = parent.path().join("secret.txt");
        std::fs::write(&outside, "secret").unwrap();
        let policy = SandboxPolicy::workspace_only(&workspace).unwrap();

        assert!(policy
            .authorize_read(workspace.join("../secret.txt"))
            .unwrap_err()
            .to_string()
            .contains("sandbox denied read"));
        assert!(policy
            .authorize_write(workspace.join("../new-secret.txt"))
            .unwrap_err()
            .to_string()
            .contains("sandbox denied write"));
    }

    #[cfg(unix)]
    #[test]
    fn workspace_policy_rejects_symlinks_that_escape() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let outside = parent.path().join("outside");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, workspace.join("escape")).unwrap();
        let policy = SandboxPolicy::workspace_only(&workspace).unwrap();

        assert!(policy
            .authorize_read(workspace.join("escape/secret.txt"))
            .is_err());
        assert!(policy
            .authorize_write(workspace.join("escape/new.txt"))
            .is_err());
    }

    #[test]
    fn search_patterns_cannot_select_parent_or_absolute_paths() {
        let workspace = tempfile::tempdir().unwrap();
        let policy = SandboxPolicy::workspace_only(workspace.path()).unwrap();

        assert!(policy.authorize_search_pattern("src/**/*.rs").is_ok());
        assert!(policy.authorize_search_pattern("../**/*").is_err());
        assert!(policy.authorize_search_pattern("/etc/*").is_err());
    }

    #[test]
    fn unrestricted_policy_allows_paths_and_searches_outside_the_workspace() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let outside = parent.path().join("outside.txt");
        std::fs::write(&outside, "outside").unwrap();
        let policy = SandboxPolicy::unrestricted(&workspace).unwrap();

        assert_eq!(policy.authorize_read(&outside).unwrap(), outside);
        assert!(policy
            .authorize_write(parent.path().join("created-outside.txt"))
            .is_ok());
        assert!(policy.authorize_search_pattern("../**/*").is_ok());
    }

    #[test]
    fn untrusted_projects_can_only_tighten_native_tool_policy() {
        let workspace_only = NativeToolFilesystemPolicy::WorkspaceOnly;
        let unrestricted = NativeToolFilesystemPolicy::Unrestricted;

        assert!(unrestricted.permits_project_override(workspace_only, false));
        assert!(!workspace_only.permits_project_override(unrestricted, false));
        assert!(workspace_only.permits_project_override(unrestricted, true));
    }
}
