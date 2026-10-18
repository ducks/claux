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
    process::{Child, Command, ExitStatus, Stdio},
    time::{Duration, Instant},
};

struct TuiProcess {
    child: Child,
    master: Option<File>,
    _storage: tempfile::TempDir,
    output: Vec<u8>,
}

impl Drop for TuiProcess {
    fn drop(&mut self) {
        // Disconnect before reaping so terminal cleanup cannot wait for a reader.
        let _ = self.child.kill();
        drop(self.master.take());
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                result => {
                    eprintln!(
                        "PTY child {} could not be reaped: {result:?}",
                        self.child.id()
                    );
                    break;
                }
            }
        }
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
            .env("HOME", storage.path())
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
            output: Vec::new(),
        }
    }

    fn drain_output(&mut self) -> bool {
        let Some(master) = self.master.as_mut() else {
            return false;
        };
        let mut progressed = false;
        // Bound each drain so a noisy child cannot prevent deadline checks.
        for _ in 0..16 {
            let mut buffer = [0; 8192];
            match master.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    progressed = true;
                    self.output.extend_from_slice(&buffer[..count]);
                    if self.output.len() > 65536 {
                        self.output.drain(..self.output.len() - 65536);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                // Linux reports EIO after the slave closes; Darwin can revoke it.
                Err(error)
                    if matches!(
                        error.raw_os_error(),
                        Some(nix::libc::EIO | nix::libc::ENXIO)
                    ) =>
                {
                    break
                }
                Err(error) => panic!("PTY read failed: {error}"),
            }
        }
        progressed
    }

    fn wait_for_output(&mut self, expected: &[u8]) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let progressed = self.drain_output();
            if self
                .output
                .windows(expected.len())
                .any(|bytes| bytes == expected)
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "missing terminal output: {}",
                String::from_utf8_lossy(&self.output)
            );
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "TUI exited during startup: {}",
                String::from_utf8_lossy(&self.output)
            );
            if !progressed {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    fn wait_for_exit(&mut self, timeout: Duration) -> io::Result<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            // Keep consuming output while the child restores and flushes its PTY.
            let progressed = self.drain_output();
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "PTY child did not exit",
                ));
            }
            if !progressed {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    fn assert_exits(&mut self) -> ExitStatus {
        self.wait_for_exit(Duration::from_secs(10))
            .unwrap_or_else(|error| {
                panic!(
                    "{error}; terminal output: {}",
                    String::from_utf8_lossy(&self.output)
                )
            })
    }

    fn diagnostics_dir(&self) -> std::path::PathBuf {
        #[cfg(target_os = "macos")]
        let base = self._storage.path().join("Library/Application Support");
        #[cfg(not(target_os = "macos"))]
        let base = self._storage.path().join("data");
        base.join("claux/diagnostics")
    }
}

#[test]
fn tui_exits_when_its_terminal_is_closed() {
    let mut process = TuiProcess::start();
    drop(process.master.take());
    let status = process.assert_exits();
    assert!(
        matches!(status.code(), Some(0 | 1)),
        "{status}; output: {}",
        String::from_utf8_lossy(&process.output)
    );
}

#[test]
fn tui_disconnect_with_hidden_cursor_does_not_panic_during_cleanup() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_claux"));
    command.args(["--config", "/dev/null", "--tui"]);
    let mut process = TuiProcess::spawn(command);
    process.wait_for_output(b"\x1b[?25l");
    drop(process.master.take());
    let status = process.assert_exits();
    assert!(
        matches!(status.code(), Some(0 | 1)),
        "{status}; output: {}",
        String::from_utf8_lossy(&process.output)
    );
}

#[test]
fn tui_still_exits_on_sigterm_with_a_connected_terminal() {
    let mut process = TuiProcess::start();
    kill(Pid::from_raw(process.child.id() as i32), Signal::SIGTERM).unwrap();
    let status = process.assert_exits();
    assert!(
        status.success(),
        "{status}; output: {}",
        String::from_utf8_lossy(&process.output)
    );
    let directory = process.diagnostics_dir();
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
fn lifecycle_probe_child() {
    let Ok(mode) = std::env::var("CLAUX_PTY_LIFECYCLE_PROBE") else {
        return;
    };
    println!("PROBE READY");
    io::stdout().flush().unwrap();
    match mode.as_str() {
        "output" => {
            for _ in 0..256 {
                io::stdout().write_all(&[b'x'; 8192]).unwrap();
            }
            io::stdout().flush().unwrap();
        }
        "stopped" => {
            kill(Pid::this(), Signal::SIGSTOP).unwrap();
            panic!("probe unexpectedly resumed");
        }
        _ => panic!("unknown probe mode"),
    }
}

fn lifecycle_probe(mode: &str) -> TuiProcess {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "lifecycle_probe_child", "--nocapture"])
        .env("CLAUX_PTY_LIFECYCLE_PROBE", mode);
    TuiProcess::spawn(command)
}

#[test]
fn exit_wait_drains_output_larger_than_the_pty_buffer() {
    let mut process = lifecycle_probe("output");
    assert!(process.assert_exits().success());
    assert!(process.output.len() <= 65536);
}

#[test]
fn stopped_child_times_out_and_is_reaped_by_cleanup() {
    let mut process = lifecycle_probe("stopped");
    process.wait_for_output(b"PROBE READY");
    assert_eq!(
        process
            .wait_for_exit(Duration::from_millis(100))
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
    let pid = Pid::from_raw(process.child.id() as i32);
    drop(process);
    assert_eq!(kill(pid, None), Err(nix::errno::Errno::ESRCH));
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
    assert!(process.assert_exits().success());
}
