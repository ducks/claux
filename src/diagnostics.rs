//! Content-free runtime breadcrumbs, written independently of the async runtime.
use serde_json::{json, Value};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        mpsc, OnceLock,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_BYTES: u64 = 1024 * 1024;
const HEARTBEAT: Duration = Duration::from_secs(30);
static EVENTS: OnceLock<mpsc::SyncSender<Value>> = OnceLock::new();
static PHASE: AtomicU8 = AtomicU8::new(0);
static INPUT: AtomicU8 = AtomicU8::new(0);
static PROGRESS: AtomicU64 = AtomicU64::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static INPUT_ERROR: AtomicBool = AtomicBool::new(false);

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Phase {
    Startup,
    Home,
    Chat,
    Turn,
    Shutdown,
    Restore,
}

pub fn phase(value: Phase) {
    if PHASE.swap(value as u8, Ordering::Relaxed) != value as u8 {
        event("phase_changed");
    }
}

pub fn event(name: &'static str) {
    if let Some(sender) = EVENTS.get() {
        if sender.try_send(snapshot(name)).is_err() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub struct InputOperation;
impl InputOperation {
    pub fn enter(read: bool) -> Self {
        INPUT.store(if read { 2 } else { 1 }, Ordering::Relaxed);
        PROGRESS.fetch_add(1, Ordering::Relaxed);
        Self
    }
}
impl Drop for InputOperation {
    fn drop(&mut self) {
        INPUT.store(0, Ordering::Relaxed);
        PROGRESS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn input_error() {
    if !INPUT_ERROR.swap(true, Ordering::Relaxed) {
        event("terminal_input_error");
    }
}

pub struct Guard {
    sender: mpsc::SyncSender<Value>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Drop for Guard {
    fn drop(&mut self) {
        // Diagnostics must not turn a stalled disk or full queue into a hung exit.
        if self.sender.try_send(snapshot("process_exit")).is_ok() {
            if let Some(worker) = self.thread.take() {
                let deadline = Instant::now() + Duration::from_secs(5);
                while !worker.is_finished() {
                    if Instant::now() >= deadline {
                        tracing::warn!("Runtime diagnostics did not finish within five seconds");
                        return;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                if worker.join().is_err() {
                    tracing::warn!("Runtime diagnostics worker panicked");
                }
            }
        } else {
            tracing::warn!("Runtime diagnostics exit record could not be queued");
        }
    }
}

pub fn start() -> io::Result<Guard> {
    phase(Phase::Startup);
    let directory = dirs::data_local_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no data directory"))?
        .join("claux/diagnostics");
    fs::create_dir_all(&directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
    }
    cleanup(&directory)?;
    let path = directory.join(format!(
        "{}-{}.jsonl",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let mut log = Log::create(path, MAX_BYTES)?;
    let (sender, receiver) = mpsc::sync_channel(128);
    let worker = thread::Builder::new()
        .name("claux-diagnostics".into())
        .spawn(move || {
            let result = (|| -> io::Result<()> {
                log.startup = Some(identity()?);
                log.write(log.startup.clone().unwrap())?;
                write_events(&mut log, receiver, HEARTBEAT)?;
                Ok(())
            })();
            if let Err(error) = result {
                tracing::warn!(kind = ?error.kind(), "Runtime diagnostics stopped writing");
            }
        })?;
    let _ = EVENTS.set(sender.clone());
    Ok(Guard {
        sender,
        thread: Some(worker),
    })
}

fn write_events(
    log: &mut Log,
    receiver: mpsc::Receiver<Value>,
    interval: Duration,
) -> io::Result<()> {
    let mut heartbeat = Instant::now();
    loop {
        let value = match receiver.recv_timeout(interval.saturating_sub(heartbeat.elapsed())) {
            Ok(value) => value,
            Err(mpsc::RecvTimeoutError::Timeout) => snapshot("heartbeat"),
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let exiting = value["event"] == "process_exit";
        let is_heartbeat = value["event"] == "heartbeat";
        log.write(value)?;
        if exiting {
            break;
        }
        if heartbeat.elapsed() >= interval {
            if !is_heartbeat {
                log.write(snapshot("heartbeat"))?;
            }
            heartbeat = Instant::now();
        }
    }
    Ok(())
}

fn timestamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn identity() -> io::Result<Value> {
    let executable = std::env::current_exe()?;
    // /proc/self/exe still identifies this running image after a rebuild/reinstall.
    #[cfg(target_os = "linux")]
    let file = File::open("/proc/self/exe")?;
    #[cfg(not(target_os = "linux"))]
    let file = File::open(&executable)?;
    let metadata = file.metadata()?;
    #[allow(unused_mut)]
    let mut value = json!({"event": "process_start", "timestamp_ms": timestamp(),
        "pid": std::process::id(), "version": env!("CARGO_PKG_VERSION"),
        "executable": executable, "executable_bytes": metadata.len(),
        "executable_modified_ms": metadata.modified().ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis())});
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        value["executable_device"] = json!(metadata.dev());
        value["executable_inode"] = json!(metadata.ino());
    }
    Ok(value)
}

fn snapshot(name: &str) -> Value {
    let phase = match PHASE.load(Ordering::Relaxed) {
        1 => "home",
        2 => "chat",
        3 => "turn",
        4 => "shutdown",
        5 => "terminal_restore",
        _ => "startup",
    };
    let input = match INPUT.load(Ordering::Relaxed) {
        1 => "poll",
        2 => "read",
        _ => "idle",
    };
    json!({"event": name, "timestamp_ms": timestamp(), "pid": std::process::id(),
        "phase": phase, "terminal_operation": input,
        "terminal_progress": PROGRESS.load(Ordering::Relaxed),
        "cpu_seconds": cpu_seconds(), "dropped_events": DROPPED.load(Ordering::Relaxed)})
}

fn cpu_seconds() -> Option<f64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<nix::libc::rusage>::uninit();
        // SAFETY: getrusage initializes usage on success.
        if unsafe { nix::libc::getrusage(nix::libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            let usage = unsafe { usage.assume_init() };
            return Some(
                usage.ru_utime.tv_sec as f64
                    + usage.ru_stime.tv_sec as f64
                    + (usage.ru_utime.tv_usec as f64 + usage.ru_stime.tv_usec as f64) / 1_000_000.0,
            );
        }
    }
    None
}

struct Log {
    path: PathBuf,
    file: File,
    size: u64,
    limit: u64,
    startup: Option<Value>,
}
impl Log {
    fn create(path: PathBuf, limit: u64) -> io::Result<Self> {
        let file = private_file(&path)?;
        Ok(Self {
            path,
            file,
            size: 0,
            limit,
            startup: None,
        })
    }
    fn write(&mut self, value: Value) -> io::Result<()> {
        let line = serde_json::to_vec(&value)?;
        if self.size + line.len() as u64 + 1 > self.limit {
            let previous = self.path.with_extension("previous.jsonl");
            // Copy instead of renaming an open file for Windows compatibility.
            fs::copy(&self.path, previous)?;
            self.file.set_len(0)?;
            use std::io::{Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(0))?;
            self.size = 0;
            if let Some(startup) = self.startup.clone() {
                self.append(&serde_json::to_vec(&startup)?)?;
            }
        }
        self.append(&line)
    }
    fn append(&mut self, line: &[u8]) -> io::Result<()> {
        self.file.write_all(line)?;
        self.file.write_all(b"\n")?;
        self.size += line.len() as u64 + 1;
        Ok(())
    }
}

fn private_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn cleanup(directory: &Path) -> io::Result<()> {
    let mut inactive = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".jsonl") {
            continue;
        }
        let Some(pid) = name
            .split('-')
            .next()
            .and_then(|pid| pid.parse::<u32>().ok())
        else {
            continue;
        };
        if process_alive(pid) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        if metadata.is_file() {
            inactive.push((metadata.modified()?, entry.path()));
        }
    }
    inactive.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    for (index, (modified, path)) in inactive.into_iter().enumerate() {
        if index >= 40 || modified.elapsed().unwrap_or_default() > Duration::from_secs(7 * 86400) {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

fn process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let Ok(pid) = i32::try_from(pid) else {
            return true;
        };
        if pid <= 0 {
            return true;
        }
        // SAFETY: signal 0 checks existence without delivering a signal.
        (unsafe { nix::libc::kill(pid, 0) } == 0
            || io::Error::last_os_error().raw_os_error() != Some(nix::libc::ESRCH))
    }
    // Conservatively retain other runs when process liveness is unavailable.
    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_keeps_identity_and_recent_events_in_private_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("probe.jsonl");
        let mut log = Log::create(path.clone(), 256).unwrap();
        log.startup = Some(json!({"event": "process_start", "pid": 1}));
        for n in 0..100 {
            log.write(json!({"event": "heartbeat", "sequence": n}))
                .unwrap();
        }
        let current = fs::read_to_string(&path).unwrap();
        assert!(current.contains("process_start"));
        assert!(current.contains("99"));
        assert!(fs::metadata(&path).unwrap().len() <= 256);
        assert!(
            fs::metadata(path.with_extension("previous.jsonl"))
                .unwrap()
                .len()
                <= 256
        );
        for line in current.lines() {
            serde_json::from_str::<Value>(line).unwrap();
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
    #[test]
    fn independent_writer_emits_heartbeats_and_finishes_on_exit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("watchdog.jsonl");
        let mut log = Log::create(path.clone(), MAX_BYTES).unwrap();
        let (sender, receiver) = mpsc::sync_channel(4);
        let worker =
            thread::spawn(move || write_events(&mut log, receiver, Duration::from_millis(10)));
        let deadline = Instant::now() + Duration::from_secs(2);
        while !fs::read_to_string(&path).unwrap().contains("heartbeat") {
            if Instant::now() > deadline {
                drop(sender);
                worker.join().unwrap().unwrap();
                panic!("watchdog did not emit a heartbeat");
            }
            thread::sleep(Duration::from_millis(10));
        }
        sender.send(snapshot("process_exit")).unwrap();
        worker.join().unwrap().unwrap();
        let records: Vec<Value> = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.last().unwrap()["event"], "process_exit");
        assert!(records[0].get("terminal_progress").is_some());
        #[cfg(unix)]
        assert!(records[0]["cpu_seconds"].is_number());
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_bounds_inactive_runs_and_removes_old_files() {
        let directory = tempfile::tempdir().unwrap();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--list")
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id();
        child.wait().unwrap();
        assert!(!process_alive(pid));
        for n in 0..45 {
            fs::write(directory.path().join(format!("{pid}-{n}.jsonl")), b"{}").unwrap();
        }
        cleanup(directory.path()).unwrap();
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 40);
        for entry in fs::read_dir(directory.path()).unwrap() {
            File::options()
                .write(true)
                .open(entry.unwrap().path())
                .unwrap()
                .set_times(
                    std::fs::FileTimes::new()
                        .set_modified(SystemTime::now() - Duration::from_secs(8 * 86400)),
                )
                .unwrap();
        }
        cleanup(directory.path()).unwrap();
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn live_process_logs_survive_cleanup() {
        let directory = tempfile::tempdir().unwrap();
        for n in 0..45 {
            fs::write(
                directory
                    .path()
                    .join(format!("{}-{n}.jsonl", std::process::id())),
                b"{}",
            )
            .unwrap();
        }
        cleanup(directory.path()).unwrap();
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 45);
    }
}
