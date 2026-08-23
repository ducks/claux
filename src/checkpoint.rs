//! Conservative, per-turn worktree checkpoints.
//!
//! Checkpoints cover Git-tracked files and non-ignored untracked files. They
//! intentionally exclude ignored files and anything outside the repository.
//! Undo verifies the complete post-turn state before writing a byte, so later
//! human edits are never silently overwritten.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_FILES: usize = 10_000;
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIFF_BYTES: usize = 256 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
enum FileState {
    Missing,
    File {
        contents: Vec<u8>,
        readonly: bool,
        #[cfg(unix)]
        mode: u32,
    },
    Symlink(PathBuf),
}

#[derive(Debug)]
pub struct PendingCheckpoint {
    root: PathBuf,
    before: BTreeMap<PathBuf, FileState>,
}

#[derive(Debug)]
pub struct TurnCheckpoint {
    root: PathBuf,
    changes: BTreeMap<PathBuf, (FileState, FileState)>,
}

impl PendingCheckpoint {
    pub fn capture() -> Result<Self> {
        let cwd = std::env::current_dir().context("could not determine current directory")?;
        Self::capture_from(&cwd)
    }

    fn capture_from(cwd: &Path) -> Result<Self> {
        let root = git_root(cwd)?;
        let paths = git_paths(&root)?;
        let before = snapshot_paths(&root, paths)?;
        Ok(Self { root, before })
    }

    pub fn finish(self) -> Result<TurnCheckpoint> {
        let mut paths: BTreeSet<PathBuf> = self.before.keys().cloned().collect();
        paths.extend(git_paths(&self.root)?);
        let after = snapshot_paths(&self.root, paths)?;
        let mut changes = BTreeMap::new();

        for path in self.before.keys().chain(after.keys()) {
            if changes.contains_key(path) {
                continue;
            }
            let before = self.before.get(path).cloned().unwrap_or(FileState::Missing);
            let after = after.get(path).cloned().unwrap_or(FileState::Missing);
            if before != after {
                changes.insert(path.clone(), (before, after));
            }
        }

        Ok(TurnCheckpoint {
            root: self.root,
            changes,
        })
    }
}

impl TurnCheckpoint {
    pub fn diff(&self) -> String {
        if self.changes.is_empty() {
            return "The last turn made no checkpointed file changes.".to_string();
        }

        let mut output = String::new();
        for (path, (before, after)) in &self.changes {
            let display = crate::utils::sanitize_terminal_text(path.to_string_lossy().as_ref());
            match (text_contents(before), text_contents(after)) {
                (Some(before), Some(after)) => {
                    let before = crate::utils::sanitize_terminal_text(before);
                    let after = crate::utils::sanitize_terminal_text(after);
                    output.push_str(
                        &similar::TextDiff::from_lines(&before, &after)
                            .unified_diff()
                            .header(&format!("a/{display}"), &format!("b/{display}"))
                            .to_string(),
                    );
                }
                _ => {
                    output.push_str(&format!(
                        "Binary or non-UTF-8 file changed: {display} ({} -> {})\n",
                        state_size(before),
                        state_size(after)
                    ));
                }
            }
            if output.len() > MAX_DIFF_BYTES {
                output.truncate(floor_char_boundary(&output, MAX_DIFF_BYTES));
                output.push_str("\n[diff truncated]\n");
                break;
            }
        }
        output
    }

    pub fn undo(&self) -> Result<String> {
        self.undo_with_before_restore(|_| {})
    }

    fn undo_with_before_restore(&self, mut before_restore: impl FnMut(&Path)) -> Result<String> {
        if self.changes.is_empty() {
            return Ok("The last turn made no checkpointed file changes.".to_string());
        }

        let mut conflicts = Vec::new();
        for (path, (_, expected_after)) in &self.changes {
            if !path_matches(&self.root, path, expected_after)? {
                conflicts.push(crate::utils::sanitize_terminal_text(
                    path.to_string_lossy().as_ref(),
                ));
            }
        }
        if !conflicts.is_empty() {
            anyhow::bail!(
                "Undo refused because these files changed after the turn:\n  {}",
                conflicts.join("\n  ")
            );
        }

        for (path, (before, expected_after)) in &self.changes {
            // Verification and restoration cannot be one filesystem-atomic
            // operation. Re-read immediately before every write to close the
            // much larger window created by the complete verification pass.
            before_restore(path);
            if !path_matches(&self.root, path, expected_after)? {
                anyhow::bail!(
                    "Undo stopped because {} changed while undo was in progress",
                    crate::utils::sanitize_terminal_text(path.to_string_lossy().as_ref())
                );
            }
            restore_path(&self.root.join(path), before)
                .with_context(|| format!("failed to restore {}", path.display()))?;
        }

        Ok(format!(
            "Undid the last turn's changes to {} file(s).",
            self.changes.len()
        ))
    }
}

fn path_matches(root: &Path, relative: &Path, expected: &FileState) -> Result<bool> {
    Ok(snapshot_path(&root.join(relative))? == *expected)
}

fn git_root(cwd: &Path) -> Result<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("could not run git")?;
    if !output.status.success() {
        anyhow::bail!("turn checkpoints require a Git worktree");
    }
    let root = String::from_utf8(output.stdout).context("Git worktree path was not UTF-8")?;
    Ok(PathBuf::from(root.trim()))
}

fn git_paths(root: &Path) -> Result<BTreeSet<PathBuf>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ])
        .output()
        .context("could not list Git worktree files")?;
    if !output.status.success() {
        anyhow::bail!("git ls-files failed");
    }

    let mut paths = BTreeSet::new();
    for raw in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|p| !p.is_empty())
    {
        let path = bytes_to_path(raw)?;
        if path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            anyhow::bail!("Git returned an unsafe worktree path");
        }
        paths.insert(path);
        if paths.len() > MAX_FILES {
            anyhow::bail!("worktree has more than {MAX_FILES} checkpointable files");
        }
    }
    Ok(paths)
}

fn snapshot_paths(
    root: &Path,
    paths: impl IntoIterator<Item = PathBuf>,
) -> Result<BTreeMap<PathBuf, FileState>> {
    let mut snapshots = BTreeMap::new();
    let mut total = 0_u64;
    for relative in paths {
        let state = snapshot_path(&root.join(&relative))?;
        total = total.saturating_add(state_size_bytes(&state));
        if total > MAX_TOTAL_BYTES {
            anyhow::bail!("checkpoint exceeds the {MAX_TOTAL_BYTES}-byte safety limit");
        }
        snapshots.insert(relative, state);
    }
    Ok(snapshots)
}

fn snapshot_path(path: &Path) -> Result<FileState> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(FileState::Missing)
        }
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        return Ok(FileState::Symlink(std::fs::read_link(path)?));
    }
    if !metadata.is_file() {
        anyhow::bail!("unsupported checkpoint path type: {}", path.display());
    }
    if metadata.len() > MAX_FILE_BYTES {
        anyhow::bail!(
            "{} exceeds the per-file checkpoint limit of {MAX_FILE_BYTES} bytes",
            path.display()
        );
    }
    Ok(FileState::File {
        contents: std::fs::read(path)?,
        readonly: metadata.permissions().readonly(),
        #[cfg(unix)]
        mode: {
            use std::os::unix::fs::MetadataExt as _;
            metadata.mode()
        },
    })
}

fn restore_path(path: &Path, state: &FileState) -> Result<()> {
    match state {
        FileState::Missing => match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        },
        FileState::Symlink(target) => {
            remove_existing(path)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            create_symlink(target, path)?;
        }
        FileState::File {
            contents,
            readonly,
            #[cfg(unix)]
            mode,
        } => {
            remove_symlink(path)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, contents)?;
            let mut permissions = std::fs::metadata(path)?.permissions();
            permissions.set_readonly(*readonly);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                permissions.set_mode(*mode);
            }
            std::fs::set_permissions(path, permissions)?;
        }
    }
    Ok(())
}

fn remove_existing(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            anyhow::bail!("refusing to replace directory {}", path.display())
        }
        Ok(_) => std::fs::remove_file(path)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn remove_symlink(path: &Path) -> Result<()> {
    if std::fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(unix)]
fn create_symlink(target: &Path, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, link)?;
    Ok(())
}

#[cfg(windows)]
fn create_symlink(target: &Path, link: &Path) -> Result<()> {
    if target.is_dir() {
        std::os::windows::fs::symlink_dir(target, link)?;
    } else {
        std::os::windows::fs::symlink_file(target, link)?;
    }
    Ok(())
}

fn text_contents(state: &FileState) -> Option<&str> {
    match state {
        FileState::Missing => Some(""),
        FileState::File { contents, .. } => std::str::from_utf8(contents).ok(),
        FileState::Symlink(_) => None,
    }
}

fn state_size(state: &FileState) -> String {
    match state {
        FileState::Missing => "missing".to_string(),
        FileState::File { contents, .. } => format!("{} bytes", contents.len()),
        FileState::Symlink(target) => format!(
            "symlink to {}",
            crate::utils::sanitize_terminal_text(target.to_string_lossy().as_ref())
        ),
    }
}

fn state_size_bytes(state: &FileState) -> u64 {
    match state {
        FileState::File { contents, .. } => contents.len() as u64,
        FileState::Symlink(target) => target.to_string_lossy().len() as u64,
        FileState::Missing => 0,
    }
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(unix)]
fn bytes_to_path(raw: &[u8]) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt as _;
    Ok(PathBuf::from(OsString::from_vec(raw.to_vec())))
}

#[cfg(windows)]
fn bytes_to_path(raw: &[u8]) -> Result<PathBuf> {
    Ok(PathBuf::from(
        String::from_utf8(raw.to_vec()).context("Git path was not UTF-8")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_repo() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        let status = Command::new("git")
            .arg("init")
            .arg("-q")
            .arg(temp.path())
            .status()
            .unwrap();
        assert!(status.success());
        temp
    }

    #[test]
    fn diff_and_undo_restore_modified_and_created_files() {
        let repo = init_repo();
        std::fs::write(repo.path().join("existing.txt"), "before\n").unwrap();
        let pending = PendingCheckpoint::capture_from(repo.path()).unwrap();

        std::fs::write(repo.path().join("existing.txt"), "after\n").unwrap();
        std::fs::write(repo.path().join("created.txt"), "new\n").unwrap();
        let checkpoint = pending.finish().unwrap();

        let diff = checkpoint.diff();
        assert!(diff.contains("-before"));
        assert!(diff.contains("+after"));
        assert!(diff.contains("created.txt"));
        assert!(checkpoint.undo().unwrap().contains("2 file(s)"));
        assert_eq!(
            std::fs::read_to_string(repo.path().join("existing.txt")).unwrap(),
            "before\n"
        );
        assert!(!repo.path().join("created.txt").exists());
    }

    #[test]
    fn undo_refuses_to_overwrite_later_edits() {
        let repo = init_repo();
        let path = repo.path().join("file.txt");
        std::fs::write(&path, "before").unwrap();
        let pending = PendingCheckpoint::capture_from(repo.path()).unwrap();
        std::fs::write(&path, "agent").unwrap();
        let checkpoint = pending.finish().unwrap();
        std::fs::write(&path, "human").unwrap();

        let error = checkpoint.undo().unwrap_err().to_string();
        assert!(error.contains("Undo refused"));
        assert_eq!(std::fs::read_to_string(path).unwrap(), "human");
    }

    #[test]
    fn undo_rechecks_a_file_immediately_before_restoring_it() {
        let repo = init_repo();
        let path = repo.path().join("file.txt");
        std::fs::write(&path, "before").unwrap();
        let pending = PendingCheckpoint::capture_from(repo.path()).unwrap();
        std::fs::write(&path, "agent").unwrap();
        let checkpoint = pending.finish().unwrap();

        let mut changed = false;
        let error = checkpoint
            .undo_with_before_restore(|_| {
                if !changed {
                    std::fs::write(&path, "human during undo").unwrap();
                    changed = true;
                }
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("changed while undo was in progress"));
        assert_eq!(std::fs::read_to_string(path).unwrap(), "human during undo");
    }

    #[test]
    fn ignored_files_are_not_checkpointed() {
        let repo = init_repo();
        std::fs::write(repo.path().join(".gitignore"), "secret.txt\n").unwrap();
        std::fs::write(repo.path().join("secret.txt"), "before").unwrap();
        let pending = PendingCheckpoint::capture_from(repo.path()).unwrap();
        std::fs::write(repo.path().join("secret.txt"), "after").unwrap();
        let checkpoint = pending.finish().unwrap();

        assert_eq!(
            checkpoint.diff(),
            "The last turn made no checkpointed file changes."
        );
    }
}
