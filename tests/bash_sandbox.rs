#![cfg(target_os = "linux")]

use std::process::Command;

fn run_sandboxed(workspace: &std::path::Path, command: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_claux"))
        .arg("__sandbox-exec")
        .arg("--workspace")
        .arg(workspace)
        .arg("--command")
        .arg(command)
        .output()
        .expect("sandbox helper should start")
}

#[test]
fn landlock_probe_verifies_real_enforcement() {
    let output = Command::new(env!("CARGO_BIN_EXE_claux"))
        .arg("__sandbox-probe")
        .output()
        .expect("sandbox probe should start");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn workspace_write_allows_writes_inside_the_workspace() {
    let workspace = tempfile::tempdir().unwrap();
    let target = workspace.path().join("inside.txt");
    let output = run_sandboxed(
        workspace.path(),
        &format!("printf allowed > '{}'", target.display()),
    );

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read_to_string(target).unwrap(), "allowed");
}

#[test]
fn workspace_write_denies_writes_outside_the_workspace() {
    let workspace = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
    let target = outside.path().join("outside.txt");
    let output = run_sandboxed(
        workspace.path(),
        &format!("printf denied > '{}'", target.display()),
    );

    assert!(!output.status.success());
    assert!(!target.exists());
}

#[test]
fn workspace_write_denies_truncating_an_existing_outside_file() {
    let workspace = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
    let target = outside.path().join("outside.txt");
    std::fs::write(&target, "preserved").unwrap();
    let output = run_sandboxed(
        workspace.path(),
        &format!("printf denied > '{}'", target.display()),
    );

    assert!(!output.status.success());
    assert_eq!(std::fs::read_to_string(target).unwrap(), "preserved");
}

#[test]
fn workspace_write_keeps_the_filesystem_readable() {
    let parent = tempfile::tempdir().unwrap();
    let workspace = parent.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let source = parent.path().join("outside.txt");
    std::fs::write(&source, "readable").unwrap();
    let output = run_sandboxed(&workspace, &format!("cat '{}'", source.display()));

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "readable");
}

#[test]
fn workspace_write_restrictions_are_inherited_by_child_processes() {
    let workspace = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
    let target = outside.path().join("child-outside.txt");
    let output = run_sandboxed(
        workspace.path(),
        &format!("sh -c \"printf denied > '{}'\"", target.display()),
    );

    assert!(!output.status.success());
    assert!(!target.exists());
}

#[cfg(unix)]
#[test]
fn workspace_write_denies_symlink_escapes() {
    use std::os::unix::fs::symlink;

    let workspace = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
    symlink(outside.path(), workspace.path().join("escape")).unwrap();
    let target = outside.path().join("symlink-outside.txt");
    let output = run_sandboxed(
        workspace.path(),
        &format!(
            "printf denied > '{}'",
            workspace
                .path()
                .join("escape/symlink-outside.txt")
                .display()
        ),
    );

    assert!(!output.status.success());
    assert!(!target.exists());
}

#[test]
fn workspace_write_allows_system_temporary_files() {
    let workspace = tempfile::tempdir().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let target = temporary.path().join("tool-output.txt");
    let output = run_sandboxed(
        workspace.path(),
        &format!("printf temporary > '{}'", target.display()),
    );

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read_to_string(target).unwrap(), "temporary");
}

#[test]
fn workspace_write_allows_linked_worktree_git_metadata() {
    let parent = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
    let common = parent.path().join("repository/.git");
    let git_dir = common.join("worktrees/workspace");
    let workspace = parent.path().join("workspace");
    std::fs::create_dir_all(&git_dir).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(
        workspace.join(".git"),
        "gitdir: ../repository/.git/worktrees/workspace\n",
    )
    .unwrap();
    std::fs::write(
        git_dir.join("gitdir"),
        format!("{}\n", workspace.join(".git").display()),
    )
    .unwrap();
    std::fs::write(git_dir.join("commondir"), "../..\n").unwrap();
    let target = common.join("sandbox-test.lock");
    let output = run_sandboxed(&workspace, &format!("printf git > '{}'", target.display()));

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read_to_string(target).unwrap(), "git");
}

#[test]
fn forged_git_metadata_cannot_expand_the_writable_roots() {
    let parent = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
    let workspace = parent.path().join("workspace");
    let outside = parent.path().join("outside");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(
        workspace.join(".git"),
        format!("gitdir: {}\n", outside.display()),
    )
    .unwrap();
    let target = outside.join("must-not-exist.txt");
    let output = run_sandboxed(
        &workspace,
        &format!("printf unsafe > '{}'", target.display()),
    );

    assert!(!output.status.success());
    assert!(!target.exists());
}

#[test]
fn sandbox_setup_errors_do_not_fall_back_to_unrestricted_execution() {
    let missing_workspace = tempfile::tempdir().unwrap().path().join("removed");
    let temporary = tempfile::tempdir().unwrap();
    let target = temporary.path().join("must-not-exist.txt");
    let output = run_sandboxed(
        &missing_workspace,
        &format!("printf unsafe > '{}'", target.display()),
    );

    assert!(!output.status.success());
    assert!(!target.exists());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("could not resolve command sandbox workspace"));
}
