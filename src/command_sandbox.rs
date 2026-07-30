//! Operating-system containment for commands spawned by the Bash tool.
//!
//! Linux workspace-write mode self-invokes Claux as a small helper. The helper
//! applies Landlock before replacing itself with the requested shell command,
//! which avoids running non-async-signal-safe setup in a `pre_exec` callback.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::process::Stdio;
use tokio::process::Command;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BashFilesystemPolicy {
    /// Use workspace-write containment on Linux and preserve existing
    /// unrestricted behavior on platforms without a backend.
    #[default]
    Auto,
    WorkspaceWrite,
    Unrestricted,
}

impl BashFilesystemPolicy {
    pub fn permits_project_override(self, requested: Self, trusted: bool) -> bool {
        trusted || policy_rank(requested) <= policy_rank(self)
    }

    pub fn diagnose(self) -> BashPolicyDiagnostic {
        diagnose_with_availability(self, landlock_availability())
    }
}

impl fmt::Display for BashFilesystemPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::WorkspaceWrite => formatter.write_str("workspace_write"),
            Self::Unrestricted => formatter.write_str("unrestricted"),
        }
    }
}

fn policy_rank(policy: BashFilesystemPolicy) -> u8 {
    match policy {
        BashFilesystemPolicy::WorkspaceWrite => 0,
        BashFilesystemPolicy::Auto => 1,
        BashFilesystemPolicy::Unrestricted => 2,
    }
}

#[derive(Clone, Debug)]
pub struct CommandSandbox {
    effective_policy: BashFilesystemPolicy,
    workspace_root: PathBuf,
}

impl CommandSandbox {
    pub fn new(policy: BashFilesystemPolicy, workspace_root: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_availability(policy, workspace_root, landlock_availability())
    }

    fn new_with_availability(
        policy: BashFilesystemPolicy,
        workspace_root: impl AsRef<Path>,
        availability: LandlockAvailability,
    ) -> Result<Self> {
        let workspace_root = workspace_root.as_ref().canonicalize().with_context(|| {
            format!(
                "could not resolve command sandbox workspace {}",
                workspace_root.as_ref().display()
            )
        })?;
        if !workspace_root.is_dir() {
            bail!(
                "command sandbox workspace is not a directory: {}",
                workspace_root.display()
            );
        }
        let diagnostic = diagnose_with_availability(policy, availability);
        if !diagnostic.healthy {
            bail!(
                "{}",
                diagnostic
                    .issue
                    .expect("unhealthy diagnostics have an issue")
            );
        }
        if let Some(issue) = diagnostic.issue {
            tracing::warn!("{issue}");
        }
        Ok(Self {
            effective_policy: diagnostic.effective_policy,
            workspace_root,
        })
    }

    #[cfg(test)]
    pub(crate) fn unrestricted_for_tests() -> Self {
        Self::new(
            BashFilesystemPolicy::Unrestricted,
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        )
        .expect("test working directory must resolve")
    }

    pub fn command(&self, shell_command: &str) -> Result<Command> {
        if self.uses_workspace_write() {
            let executable =
                std::env::current_exe().context("could not locate the claux executable")?;
            let mut command = Command::new(executable);
            command
                .arg("__sandbox-exec")
                .arg("--workspace")
                .arg(&self.workspace_root)
                .arg("--command")
                .arg(shell_command);
            return Ok(command);
        }

        let mut command = Command::new("sh");
        command.arg("-c").arg(shell_command);
        Ok(command)
    }

    fn uses_workspace_write(&self) -> bool {
        match self.effective_policy {
            BashFilesystemPolicy::WorkspaceWrite => true,
            BashFilesystemPolicy::Auto => {
                unreachable!("auto is resolved when the command sandbox is created")
            }
            BashFilesystemPolicy::Unrestricted => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
// Availability variants are platform-specific in production but all are used
// by the platform-independent policy tests.
#[allow(dead_code)]
enum LandlockAvailability {
    Available,
    UnsupportedPlatform,
    Unavailable(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BashPolicyDiagnostic {
    pub summary: String,
    pub issue: Option<String>,
    pub healthy: bool,
    effective_policy: BashFilesystemPolicy,
}

fn diagnose_with_availability(
    policy: BashFilesystemPolicy,
    availability: LandlockAvailability,
) -> BashPolicyDiagnostic {
    const WORKSPACE_WRITE_SUMMARY: &str =
        "workspace_write (Landlock; temporary and Git metadata paths remain writable)";

    match (policy, availability) {
        (BashFilesystemPolicy::Unrestricted, _) => BashPolicyDiagnostic {
            summary: "unrestricted".to_string(),
            issue: None,
            healthy: true,
            effective_policy: BashFilesystemPolicy::Unrestricted,
        },
        (BashFilesystemPolicy::Auto, LandlockAvailability::Available)
        | (BashFilesystemPolicy::WorkspaceWrite, LandlockAvailability::Available) => {
            BashPolicyDiagnostic {
                summary: WORKSPACE_WRITE_SUMMARY.to_string(),
                issue: None,
                healthy: true,
                effective_policy: BashFilesystemPolicy::WorkspaceWrite,
            }
        }
        (BashFilesystemPolicy::Auto, LandlockAvailability::UnsupportedPlatform) => {
            BashPolicyDiagnostic {
                summary: "unrestricted (no sandbox backend for this platform)".to_string(),
                issue: None,
                healthy: true,
                effective_policy: BashFilesystemPolicy::Unrestricted,
            }
        }
        (BashFilesystemPolicy::Auto, LandlockAvailability::Unavailable(reason)) => {
            BashPolicyDiagnostic {
                summary: "unrestricted (automatic Landlock fallback)".to_string(),
                issue: Some(format!(
                    "Landlock workspace-write sandbox unavailable; Bash will run unrestricted: \
                     {reason}"
                )),
                healthy: true,
                effective_policy: BashFilesystemPolicy::Unrestricted,
            }
        }
        (BashFilesystemPolicy::WorkspaceWrite, LandlockAvailability::UnsupportedPlatform) => {
            BashPolicyDiagnostic {
                summary: "workspace_write (unavailable)".to_string(),
                issue: Some(
                    "bash_filesystem_policy=workspace_write is currently only supported on Linux"
                        .to_string(),
                ),
                healthy: false,
                effective_policy: BashFilesystemPolicy::WorkspaceWrite,
            }
        }
        (BashFilesystemPolicy::WorkspaceWrite, LandlockAvailability::Unavailable(reason)) => {
            BashPolicyDiagnostic {
                summary: "workspace_write (unavailable)".to_string(),
                issue: Some(format!(
                    "bash_filesystem_policy=workspace_write requires complete Landlock ABI V3 \
                     enforcement: {reason}"
                )),
                healthy: false,
                effective_policy: BashFilesystemPolicy::WorkspaceWrite,
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn landlock_availability() -> LandlockAvailability {
    let executable = match std::env::current_exe() {
        Ok(executable) => executable,
        Err(error) => {
            return LandlockAvailability::Unavailable(format!(
                "could not locate the claux executable: {error}"
            ));
        }
    };
    match std::process::Command::new(executable)
        .arg("__sandbox-probe")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) if output.status.success() => LandlockAvailability::Available,
        Ok(output) => {
            let reason = String::from_utf8_lossy(&output.stderr);
            let reason = reason
                .trim()
                .strip_prefix("Error: ")
                .unwrap_or(reason.trim());
            LandlockAvailability::Unavailable(if reason.is_empty() {
                format!("probe exited with {}", output.status)
            } else {
                reason.to_string()
            })
        }
        Err(error) => {
            LandlockAvailability::Unavailable(format!("could not run Landlock probe: {error}"))
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn landlock_availability() -> LandlockAvailability {
    LandlockAvailability::UnsupportedPlatform
}

#[cfg(target_os = "linux")]
pub fn run_helper(workspace_root: &Path, shell_command: &str) -> Result<()> {
    apply_landlock(workspace_root)?;

    use std::os::unix::process::CommandExt;
    let error = std::process::Command::new("sh")
        .arg("-c")
        .arg(shell_command)
        .exec();
    Err(error).context("could not execute sandboxed shell command")
}

#[cfg(target_os = "linux")]
pub fn run_probe() -> Result<()> {
    apply_landlock(&std::env::current_dir()?)
}

#[cfg(not(target_os = "linux"))]
pub fn run_helper(_workspace_root: &Path, _shell_command: &str) -> Result<()> {
    bail!("bash workspace-write sandboxing is currently only supported on Linux")
}

#[cfg(not(target_os = "linux"))]
pub fn run_probe() -> Result<()> {
    bail!("Landlock is only available on Linux")
}

#[cfg(target_os = "linux")]
fn apply_landlock(workspace_root: &Path) -> Result<()> {
    use landlock::{
        Access, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetCreatedAttr, ABI,
    };

    let workspace_root = workspace_root.canonicalize().with_context(|| {
        format!(
            "could not resolve command sandbox workspace {}",
            workspace_root.display()
        )
    })?;
    // ABI V3 is the minimum complete write boundary because it added
    // TRUNCATE. Older ABIs could prevent creating an outside file while still
    // allowing an existing outside file to be truncated.
    let access_rw = AccessFs::from_all(ABI::V3);
    let access_ro = AccessFs::from_read(ABI::V3);

    let mut ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::BestEffort)
        .handle_access(access_rw)?
        .create()?
        .add_rules(landlock::path_beneath_rules(&["/"], access_ro))?
        .no_new_privs(true);

    let writable_roots = writable_roots(&workspace_root);
    ruleset = ruleset.add_rules(landlock::path_beneath_rules(&writable_roots, access_rw))?;

    let status = ruleset.restrict_self()?;
    if status.ruleset != landlock::RulesetStatus::FullyEnforced {
        bail!(
            "Landlock ABI V3 is unavailable; refusing to run Bash without complete filesystem \
             containment"
        );
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn writable_roots(workspace_root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![workspace_root.to_path_buf()];
    for path in [
        Some(std::env::temp_dir()),
        Some(PathBuf::from("/tmp")),
        Some(PathBuf::from("/var/tmp")),
        Some(PathBuf::from("/dev/null")),
    ]
    .into_iter()
    .flatten()
    {
        push_canonical_if_present(&mut roots, &path);
    }
    add_git_metadata_roots(&mut roots, workspace_root);
    roots.sort();
    roots.dedup();
    roots
}

#[cfg(target_os = "linux")]
fn add_git_metadata_roots(roots: &mut Vec<PathBuf>, workspace_root: &Path) {
    let Some(dot_git) = workspace_root
        .ancestors()
        .map(|directory| directory.join(".git"))
        .find(|path| path.exists())
    else {
        return;
    };
    if dot_git
        .symlink_metadata()
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return;
    }
    if dot_git.is_dir() {
        push_canonical_if_present(roots, &dot_git);
        return;
    }

    let Ok(contents) = std::fs::read_to_string(&dot_git) else {
        return;
    };
    let Some(git_dir) = contents.trim().strip_prefix("gitdir:") else {
        return;
    };
    let Some(worktree_root) = dot_git.parent() else {
        return;
    };
    let git_dir = resolve_relative(worktree_root, Path::new(git_dir.trim()));
    let Ok(git_dir) = git_dir.canonicalize() else {
        return;
    };
    let Ok(backlink) = std::fs::read_to_string(git_dir.join("gitdir")) else {
        return;
    };
    let backlink = resolve_relative(&git_dir, Path::new(backlink.trim()));
    let (Ok(backlink), Ok(dot_git)) = (backlink.canonicalize(), dot_git.canonicalize()) else {
        return;
    };
    if backlink != dot_git {
        return;
    }
    roots.push(git_dir.clone());

    let Ok(common_dir) = std::fs::read_to_string(git_dir.join("commondir")) else {
        return;
    };
    let common_dir = resolve_relative(&git_dir, Path::new(common_dir.trim()));
    let Ok(common_dir) = common_dir.canonicalize() else {
        return;
    };
    if common_dir.file_name().is_some_and(|name| name == ".git") && git_dir.starts_with(&common_dir)
    {
        roots.push(common_dir);
    }
}

#[cfg(target_os = "linux")]
fn resolve_relative(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[cfg(target_os = "linux")]
fn push_canonical_if_present(roots: &mut Vec<PathBuf>, path: &Path) {
    if let Ok(path) = path.canonicalize() {
        roots.push(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrusted_projects_can_only_tighten_bash_policy() {
        assert!(BashFilesystemPolicy::Auto
            .permits_project_override(BashFilesystemPolicy::WorkspaceWrite, false));
        assert!(!BashFilesystemPolicy::Auto
            .permits_project_override(BashFilesystemPolicy::Unrestricted, false));
        assert!(BashFilesystemPolicy::WorkspaceWrite
            .permits_project_override(BashFilesystemPolicy::Unrestricted, true));
    }

    #[test]
    fn unrestricted_commands_run_the_shell_directly() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox =
            CommandSandbox::new(BashFilesystemPolicy::Unrestricted, workspace.path()).unwrap();
        let command = sandbox.command("echo hello").unwrap();

        assert_eq!(command.as_std().get_program(), "sh");
        assert_eq!(
            command
                .as_std()
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            ["-c", "echo hello"]
        );
    }

    #[test]
    fn auto_falls_back_with_a_warning_when_landlock_is_unavailable() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox = CommandSandbox::new_with_availability(
            BashFilesystemPolicy::Auto,
            workspace.path(),
            LandlockAvailability::Unavailable("kernel is too old".to_string()),
        )
        .unwrap();

        assert_eq!(
            sandbox
                .command("echo hello")
                .unwrap()
                .as_std()
                .get_program(),
            "sh"
        );
        assert_eq!(
            diagnose_with_availability(
                BashFilesystemPolicy::Auto,
                LandlockAvailability::Unavailable("kernel is too old".to_string())
            ),
            BashPolicyDiagnostic {
                summary: "unrestricted (automatic Landlock fallback)".to_string(),
                issue: Some(
                    "Landlock workspace-write sandbox unavailable; Bash will run unrestricted: \
                     kernel is too old"
                        .to_string()
                ),
                healthy: true,
                effective_policy: BashFilesystemPolicy::Unrestricted,
            }
        );
    }

    #[test]
    fn explicit_workspace_write_fails_closed_when_landlock_is_unavailable() {
        let workspace = tempfile::tempdir().unwrap();
        let error = CommandSandbox::new_with_availability(
            BashFilesystemPolicy::WorkspaceWrite,
            workspace.path(),
            LandlockAvailability::Unavailable("kernel is too old".to_string()),
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("requires complete Landlock ABI V3 enforcement"));
    }

    #[test]
    fn auto_is_quietly_unrestricted_on_platforms_without_a_backend() {
        assert_eq!(
            diagnose_with_availability(
                BashFilesystemPolicy::Auto,
                LandlockAvailability::UnsupportedPlatform
            ),
            BashPolicyDiagnostic {
                summary: "unrestricted (no sandbox backend for this platform)".to_string(),
                issue: None,
                healthy: true,
                effective_policy: BashFilesystemPolicy::Unrestricted,
            }
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn workspace_write_commands_use_the_helper() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox = CommandSandbox::new_with_availability(
            BashFilesystemPolicy::WorkspaceWrite,
            workspace.path(),
            LandlockAvailability::Available,
        )
        .unwrap();
        let command = sandbox.command("echo hello").unwrap();
        let arguments = command
            .as_std()
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert_eq!(arguments[0], "__sandbox-exec");
        assert_eq!(arguments[1], "--workspace");
        assert_eq!(arguments[2], workspace.path().to_string_lossy());
        assert_eq!(arguments[3], "--command");
        assert_eq!(arguments[4], "echo hello");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn git_metadata_outside_a_worktree_is_writable() {
        let parent = tempfile::tempdir().unwrap();
        let common = parent.path().join("repository/.git");
        let git_dir = common.join("worktrees/workspace");
        let workspace = parent.path().join("workspace/subdirectory");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(
            parent.path().join("workspace/.git"),
            "gitdir: ../repository/.git/worktrees/workspace\n",
        )
        .unwrap();
        std::fs::write(
            git_dir.join("gitdir"),
            format!("{}\n", parent.path().join("workspace/.git").display()),
        )
        .unwrap();
        std::fs::write(git_dir.join("commondir"), "../..\n").unwrap();

        let roots = writable_roots(&workspace);

        assert!(roots.contains(&git_dir.canonicalize().unwrap()));
        assert!(roots.contains(&common.canonicalize().unwrap()));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn forged_git_metadata_cannot_add_an_arbitrary_writable_root() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let outside = parent.path().join("outside");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(
            workspace.join(".git"),
            format!("gitdir: {}\n", outside.display()),
        )
        .unwrap();

        let roots = writable_roots(&workspace);

        assert!(!roots.contains(&outside.canonicalize().unwrap()));
    }
}
