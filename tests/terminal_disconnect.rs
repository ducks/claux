#![cfg(unix)]

#[allow(dead_code)]
#[path = "../src/diagnostics.rs"]
mod diagnostics;

#[path = "../src/terminal_input.rs"]
mod terminal_input;

use nix::{
    pty::{openpty, Winsize},
    sys::signal::{kill, Signal},
    unistd::{setsid, Pid},
};
use std::{
    fs::File,
    io::{self, Read, Write},
    os::{fd::AsRawFd, unix::process::CommandExt},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct TuiProcess {
    child: Child,
    master: Option<File>,
    _storage: tempfile::TempDir,
}

impl Drop for TuiProcess {
    fn drop(&mut self) {
        // Always reap the reproduction, including when an assertion fails.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl TuiProcess {
    fn start() -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_claux"));
        command.args(["--config", "/dev/null", "--tui"]);
        let mut process = Self::spawn(command);
        process.wait_for_output(b"\x1b[?1049h");
        process.master.as_mut().unwrap().write_all(b"p").unwrap();
        process.wait_for_output(b"Project name:");
        process
    }

    fn spawn(mut command: Command) -> Self {
        let storage = tempfile::tempdir().unwrap();
        let pty = openpty(
            Some(&Winsize {
                ws_row: 24,
                ws_col: 100,
                ws_xpixel: 0,
                ws_ypixel: 0,
            }),
            None,
        )
        .unwrap();
        let slave = File::from(pty.slave);
        let master = File::from(pty.master);
        // SAFETY: master is an owned live descriptor; only its status flags change.
        unsafe {
            assert_ne!(
                nix::libc::fcntl(
                    master.as_raw_fd(),
                    nix::libc::F_SETFD,
                    nix::libc::FD_CLOEXEC
                ),
                -1
            );
            let flags = nix::libc::fcntl(master.as_raw_fd(), nix::libc::F_GETFL);
            assert_ne!(flags, -1);
            assert_ne!(
                nix::libc::fcntl(
                    master.as_raw_fd(),
                    nix::libc::F_SETFL,
                    flags | nix::libc::O_NONBLOCK
                ),
                -1
            );
        }
        command
            .current_dir(storage.path())
            .env("XDG_CONFIG_HOME", storage.path().join("config"))
            .env("XDG_DATA_HOME", storage.path().join("data"))
            .env("TERM", "xterm-256color")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave.try_clone().unwrap()));
        // SAFETY: only async-signal-safe syscalls run between fork and exec.
        unsafe {
            command.pre_exec(|| {
                setsid().map_err(io::Error::from)?;
                if nix::libc::ioctl(0, nix::libc::TIOCSCTTY as _, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        drop(command);
        drop(slave);
        Self {
            child,
            master: Some(master),
            _storage: storage,
        }
    }

    fn wait_for_output(&mut self, expected: &[u8]) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut output = Vec::new();
        loop {
            let mut buffer = [0; 65536];
            match self.master.as_mut().unwrap().read(&mut buffer) {
                Ok(0) => panic!("terminal closed before expected output"),
                Ok(count) => output.extend_from_slice(&buffer[..count]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("PTY read failed: {e}"),
            }
            if output
                .windows(expected.len())
                .any(|bytes| bytes == expected)
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "missing terminal output: {}",
                String::from_utf8_lossy(&output)
            );
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "TUI exited during startup"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn assert_exits(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.child.try_wait().unwrap().is_none() {
            assert!(
                Instant::now() < deadline,
                "TUI did not exit after terminal shutdown"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[test]
fn tui_exits_when_its_terminal_is_closed() {
    let mut process = TuiProcess::start();
    drop(process.master.take());
    process.assert_exits();
    assert!(matches!(process.child.wait().unwrap().code(), Some(0 | 1)));
}

#[test]
fn tui_disconnect_with_hidden_cursor_does_not_panic_during_cleanup() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_claux"));
    command.args(["--config", "/dev/null", "--tui"]);
    let mut process = TuiProcess::spawn(command);
    process.wait_for_output(b"\x1b[?25l");
    drop(process.master.take());
    process.assert_exits();
    assert!(matches!(process.child.wait().unwrap().code(), Some(0 | 1)));
}

#[test]
fn tui_still_exits_on_sigterm_with_a_connected_terminal() {
    let mut process = TuiProcess::start();
    kill(Pid::from_raw(process.child.id() as i32), Signal::SIGTERM).unwrap();
    process.assert_exits();
    assert!(process.child.wait().unwrap().success());
    let directory = process._storage.path().join("data/claux/diagnostics");
    let paths: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(paths.len(), 1);
    let text = std::fs::read_to_string(&paths[0]).unwrap();
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records[0]["event"], "process_start");
    assert!(records[0]["executable_bytes"].as_u64().unwrap() > 0);
    assert!(records[0]["executable_inode"].is_number());
    assert!(records.iter().any(|record| record["event"] == "sigterm"));
    assert!(records
        .iter()
        .any(|record| record["event"] == "terminal_restored"));
    assert_eq!(records.last().unwrap()["event"], "process_exit");
}

#[test]
fn short_poll_child() {
    if std::env::var_os("CLAUX_SHORT_POLL_CHILD").is_none() {
        return;
    }
    crossterm::terminal::enable_raw_mode().unwrap();
    // Keep SIGHUP from terminating the probe before it exercises EOF handling.
    // SAFETY: SIG_IGN is a valid disposition, installed only in this child.
    unsafe {
        nix::libc::signal(nix::libc::SIGHUP, nix::libc::SIG_IGN);
    }
    assert!(!terminal_input::poll(Duration::ZERO).unwrap());
    print!("READY");
    io::stdout().flush().unwrap();
    loop {
        if terminal_input::poll(Duration::ZERO).unwrap() {
            assert!(matches!(terminal_input::read().unwrap(),
                crossterm::event::Event::Key(key) if key.code == crossterm::event::KeyCode::Char('x')));
            break;
        }
    }
    print!("KEY RECEIVED");
    io::stdout().flush().unwrap();
    while terminal_input::poll(Duration::ZERO).is_ok() {}
    // libtest cannot print its summary to a disconnected terminal.
    std::process::exit(0);
}

#[test]
fn short_polls_read_keys_and_report_disconnect_without_a_signal() {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "short_poll_child", "--nocapture"])
        .env("CLAUX_SHORT_POLL_CHILD", "1");
    let mut process = TuiProcess::spawn(command);
    process.wait_for_output(b"READY");
    process.master.as_mut().unwrap().write_all(b"x").unwrap();
    process.wait_for_output(b"KEY RECEIVED");
    drop(process.master.take());
    process.assert_exits();
    assert!(process.child.wait().unwrap().success());
}
