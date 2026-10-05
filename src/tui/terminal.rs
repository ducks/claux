use anyhow::Result;
use crossterm::{
    cursor::Show,
    event::{DisableBracketedPaste, EnableBracketedPaste},
    execute,
    style::{Attribute, ResetColor, SetAttribute},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, stdout, Stdout, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub type AppTerminal = Terminal<CrosstermBackend<TerminalOutput>>;

pub struct TerminalOutput {
    stdout: Stdout,
    closing: Arc<AtomicBool>,
}

impl Write for TerminalOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self.stdout.write(bytes) {
            Err(_) if self.closing.load(Ordering::Relaxed) => Ok(bytes.len()),
            result => result,
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.stdout.flush() {
            Err(_) if self.closing.load(Ordering::Relaxed) => Ok(()),
            result => result,
        }
    }
}

/// Owns the process-wide terminal modes used by the TUI.
///
/// Cleanup is attempted both explicitly and from Drop, so errors, early
/// returns, and panics cannot normally leave the shell in raw/alternate mode.
pub struct TerminalGuard {
    terminal: AppTerminal,
    modes: TerminalModes,
    closing: Arc<AtomicBool>,
}

struct TerminalModes {
    active: bool,
}

impl TerminalGuard {
    pub fn enter() -> Result<Self> {
        Self::enter_with(Terminal::new)
    }

    fn enter_with(
        create: impl FnOnce(CrosstermBackend<TerminalOutput>) -> io::Result<AppTerminal>,
    ) -> Result<Self> {
        // Arm cleanup before the first change: an escape-sequence write can
        // fail after the terminal has already accepted part of it.
        let modes = TerminalModes { active: true };
        enable_raw_mode()?;
        execute!(stdout(), EnterAlternateScreen, EnableBracketedPaste)?;

        let closing = Arc::new(AtomicBool::new(false));
        let backend = CrosstermBackend::new(TerminalOutput {
            stdout: stdout(),
            closing: closing.clone(),
        });
        let terminal = create(backend)?;

        Ok(Self {
            terminal,
            modes,
            closing,
        })
    }

    pub fn terminal_mut(&mut self) -> &mut AppTerminal {
        &mut self.terminal
    }

    pub fn restore(&mut self) -> Result<()> {
        self.modes.restore(&mut stdout(), disable_raw_mode)?;
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        // Ratatui retries cursor cleanup in Drop and reports failures using
        // eprintln!, which panics if stderr's PTY is gone too. Explicit restore
        // still reports errors; only the destructor's writes are best-effort.
        self.closing.store(true, Ordering::Relaxed);
    }
}

impl TerminalModes {
    fn restore(
        &mut self,
        writer: &mut impl Write,
        restore_raw: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }

        // Keep each escape operation independent so a failed write does not
        // skip the others. Show the cursor after restoring the main screen.
        let results = [
            restore_raw(),
            execute!(writer, DisableBracketedPaste),
            execute!(writer, LeaveAlternateScreen),
            execute!(writer, SetAttribute(Attribute::Reset)),
            execute!(writer, ResetColor),
            execute!(writer, Show),
        ];
        let result = results
            .into_iter()
            .find_map(Result::err)
            .map_or(Ok(()), Err);
        self.active = result.is_err();
        result
    }
}

impl Drop for TerminalModes {
    fn drop(&mut self) {
        let _ = self.restore(&mut stdout(), disable_raw_mode);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    struct FailOnce {
        bytes: Vec<u8>,
        fail: bool,
    }

    impl Write for FailOnce {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if std::mem::take(&mut self.fail) {
                return Err(io::Error::other("temporary output failure"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn failed_cleanup_attempts_every_reset_and_can_be_retried() {
        let mut modes = TerminalModes { active: true };
        let mut writer = FailOnce {
            bytes: Vec::new(),
            fail: true,
        };
        assert!(modes
            .restore(&mut writer, || Err(io::Error::other("raw mode failure")))
            .is_err());
        assert!(modes.active);
        let output = String::from_utf8_lossy(&writer.bytes);
        assert!(output.contains("\x1b[?1049l"));
        assert!(output.contains("\x1b[0m"));
        assert!(output.contains("\x1b[?25h"));

        modes.restore(&mut writer, || Ok(())).unwrap();
        assert!(!modes.active);
        assert!(String::from_utf8_lossy(&writer.bytes).contains("\x1b[?2004l"));
        let count = writer.bytes.len();
        modes
            .restore(&mut writer, || panic!("already restored"))
            .unwrap();
        assert_eq!(writer.bytes.len(), count);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pty_child() {
        use futures_util::FutureExt;
        use std::{panic::AssertUnwindSafe, time::Duration};

        let Ok(scenario) = std::env::var("CLAUX_TERMINAL_TEST_SCENARIO") else {
            return;
        };
        let ready = std::env::var("CLAUX_TERMINAL_TEST_READY").unwrap();
        let release = std::env::var("CLAUX_TERMINAL_TEST_RELEASE").unwrap();
        let restored = std::env::var("CLAUX_TERMINAL_TEST_RESTORED").unwrap();
        let checked = std::env::var("CLAUX_TERMINAL_TEST_CHECKED").unwrap();
        let handshake = |ready: &str, release: &str| {
            std::fs::write(ready, "ready").unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !std::path::Path::new(release).exists() {
                assert!(std::time::Instant::now() < deadline, "PTY parent timed out");
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        let result = AssertUnwindSafe(async {
            if scenario == "startup_error" {
                assert!(TerminalGuard::enter_with(|_| {
                    handshake(&ready, &release);
                    Err(io::Error::other("terminal initialization failed"))
                })
                .is_err());
                return;
            }
            let shutdown = crate::shutdown::TuiShutdown::listen().unwrap();
            let mut guard = TerminalGuard::enter().unwrap();
            handshake(&ready, &release);
            match scenario.as_str() {
                "drop" => {}
                "explicit" => guard.restore().unwrap(),
                "panic" => panic!("terminal unwind regression"),
                "signal" => shutdown.token.cancelled().await,
                _ => panic!("unknown scenario"),
            }
        })
        .catch_unwind()
        .await;
        // macOS revokes the controlling PTY when its session leader exits.
        // Check after guard cleanup (including unwinding), but before exit.
        handshake(&restored, &checked);
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    #[cfg(unix)]
    fn check_pty_restore(scenario: &str, signal: Option<nix::sys::signal::Signal>) {
        use nix::{
            pty::{openpty, Winsize},
            sys::{
                signal::kill,
                termios::{tcgetattr, LocalFlags},
            },
            unistd::{setsid, Pid},
        };
        use std::{
            fs::File,
            io::Read,
            os::unix::process::CommandExt,
            process::{Command, Stdio},
            time::{Duration, Instant},
        };

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
        let before = tcgetattr(&slave).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ready = dir.path().join("ready");
        let release = dir.path().join("release");
        let restored = dir.path().join("restored");
        let checked = dir.path().join("checked");
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "tui::terminal::tests::pty_child", "--nocapture"])
            .env("CLAUX_TERMINAL_TEST_SCENARIO", scenario)
            .env("CLAUX_TERMINAL_TEST_READY", &ready)
            .env("CLAUX_TERMINAL_TEST_RELEASE", &release)
            .env("CLAUX_TERMINAL_TEST_RESTORED", &restored)
            .env("CLAUX_TERMINAL_TEST_CHECKED", &checked)
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave.try_clone().unwrap()));
        // Give the child its own controlling terminal; never alter the
        // developer's terminal when crossterm opens /dev/tty.
        unsafe {
            command.pre_exec(|| {
                setsid().map_err(io::Error::from)?;
                if nix::libc::ioctl(0, nix::libc::TIOCSCTTY as _, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn().unwrap();
        let reader = std::thread::spawn(move || {
            let mut master = File::from(pty.master);
            let mut output = Vec::new();
            let _ = master.read_to_end(&mut output); // Linux PTYs end with EIO.
            output
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !ready.exists() {
            if Instant::now() > deadline || child.try_wait().unwrap().is_some() {
                let _ = child.kill();
                let _ = child.wait();
                panic!("PTY child failed to enter raw mode");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let during = tcgetattr(&slave).unwrap();
        assert!(!during
            .local_flags
            .intersects(LocalFlags::ICANON | LocalFlags::ECHO));
        std::fs::write(&release, "continue").unwrap();
        if let Some(signal) = signal {
            kill(Pid::from_raw(child.id() as i32), signal).unwrap();
        }
        while !restored.exists() {
            if Instant::now() > deadline || child.try_wait().unwrap().is_some() {
                let _ = child.kill();
                let _ = child.wait();
                panic!("PTY child failed to finish terminal cleanup");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let after = tcgetattr(&slave);
        // Let the child exit even if the restoration assertion will fail.
        std::fs::write(&checked, "continue").unwrap();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("PTY child failed to exit");
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(status.success(), scenario != "panic");
        assert_eq!(
            comparable_terminal_settings(after.unwrap()),
            comparable_terminal_settings(before),
            "terminal modes were not restored"
        );
        drop(command);
        drop(slave);
        let output = reader.join().unwrap();
        let output = String::from_utf8_lossy(&output);
        for reset in ["\x1b[?2004l", "\x1b[?1049l", "\x1b[0m", "\x1b[?25h"] {
            assert!(output.contains(reset), "missing terminal reset {reset:?}");
        }
    }

    fn comparable_terminal_settings(
        settings: nix::sys::termios::Termios,
    ) -> nix::sys::termios::Termios {
        // Normalize the underlying struct too: nix's equality includes both
        // its cached libc termios and the public flag fields.
        let settings: nix::libc::termios = settings.into();
        #[cfg(target_os = "macos")]
        let settings = {
            let mut settings = settings;
            // Darwin sets this transient input-reprocessing flag when ICANON
            // is restored, even when the saved settings did not contain it.
            settings.c_lflag &= !nix::libc::PENDIN;
            settings
        };
        settings.into()
    }

    #[test]
    fn terminal_comparison_ignores_only_macos_pending_input() {
        use nix::{pty::openpty, sys::termios::tcgetattr};

        let pty = openpty(None, None).unwrap();
        let mut original: nix::libc::termios = tcgetattr(&pty.slave).unwrap().into();
        original.c_lflag &= !nix::libc::PENDIN;
        let expected = comparable_terminal_settings(original.into());
        let mut pending = original;
        pending.c_lflag |= nix::libc::PENDIN;
        assert_eq!(
            comparable_terminal_settings(pending.into()) == expected,
            cfg!(target_os = "macos")
        );
        for flag in [nix::libc::ECHO, nix::libc::ICANON, nix::libc::ISIG] {
            let mut changed = original;
            changed.c_lflag ^= flag;
            assert_ne!(comparable_terminal_settings(changed.into()), expected);
        }
        let mut changed = original;
        changed.c_cc[nix::libc::VMIN] ^= 1;
        assert_ne!(comparable_terminal_settings(changed.into()), expected);
    }

    #[cfg(unix)]
    #[test]
    fn restores_real_terminal_on_exit_startup_failure_and_panic() {
        for scenario in ["drop", "explicit", "startup_error", "panic"] {
            check_pty_restore(scenario, None);
        }
    }

    #[cfg(unix)]
    #[test]
    fn restores_real_terminal_after_shutdown_signals() {
        use nix::sys::signal::Signal;
        for signal in [Signal::SIGTERM, Signal::SIGHUP, Signal::SIGINT] {
            check_pty_restore("signal", Some(signal));
        }
    }
}
