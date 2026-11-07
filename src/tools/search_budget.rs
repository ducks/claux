//! Cooperative bounds shared by filesystem searches. OS I/O itself is not preemptible.
use std::cell::Cell;
use std::io::{self, Read};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::ToolOutput;

pub(super) const TRUNCATED_NOTICE: &str = "[results truncated: refine the pattern or path]";

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub timeout: Duration,
    pub entries: usize,
    pub results: usize,
    pub output_bytes: usize,
    pub depth: usize,
    pub file_bytes: u64,
    pub read_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            entries: 20_000,
            results: 1_000,
            output_bytes: 256 * 1024,
            depth: 64,
            file_bytes: 8 * 1024 * 1024,
            read_bytes: 64 * 1024 * 1024,
        }
    }
}

pub(super) struct Budget {
    pub limits: Limits,
    cancel: CancellationToken,
    start: Instant,
    visited: Cell<usize>,
    read: Cell<usize>,
    reason: Cell<Option<&'static str>>,
    pub omitted: Cell<bool>,
}
impl Budget {
    pub fn new(limits: Limits, cancel: CancellationToken) -> Self {
        Self {
            limits,
            cancel,
            start: Instant::now(),
            visited: Cell::new(0),
            read: Cell::new(0),
            reason: Cell::new(None),
            omitted: Cell::new(false),
        }
    }
    pub fn stop(&self, reason: &'static str) {
        if self.reason.get().is_none() {
            self.reason.set(Some(reason));
        }
    }
    pub fn check(&self) -> bool {
        if self.cancel.is_cancelled() {
            self.reason.set(Some("cancelled by user"));
        } else if self.start.elapsed() >= self.limits.timeout {
            self.stop("time limit reached");
        }
        self.reason.get().is_none()
    }
    pub fn visit(&self) -> bool {
        if !self.check() {
            return false;
        }
        if self.visited.get() >= self.limits.entries {
            self.stop("entry limit reached");
            return false;
        }
        self.visited.set(self.visited.get() + 1);
        true
    }
    pub fn remaining_read(&self) -> usize {
        self.limits.read_bytes.saturating_sub(self.read.get())
    }
    pub fn finish(&self, output: Output) -> ToolOutput {
        self.check();
        let mut content = output.lines.join("\n");
        let partial = self.reason.get().is_some() || self.omitted.get();
        if content.is_empty() {
            content = if partial {
                "Search incomplete; no matches found in the searched entries."
            } else {
                "No matches found"
            }
            .into();
        }
        if let Some(reason) = self.reason.get() {
            content.push('\n');
            if reason == "output limit reached" {
                content.push_str(TRUNCATED_NOTICE);
            } else {
                content.push_str(&format!("[Search incomplete: {reason}. Partial results only; narrow the path or pattern.]"));
            }
        }
        if self.omitted.get() {
            content.push_str("\n[Search incomplete: some entries were skipped due to depth, file size, or read errors. Narrow the path or inspect specific files.]");
        }
        ToolOutput {
            content,
            is_error: self.cancel.is_cancelled(),
            sub_agent: None,
        }
    }
}

#[derive(Default)]
pub(super) struct Output {
    pub lines: Vec<String>,
    bytes: usize,
}
impl Output {
    pub fn remaining_bytes(&self, budget: &Budget) -> usize {
        budget
            .limits
            .output_bytes
            .saturating_sub(self.bytes + usize::from(!self.lines.is_empty()))
    }
    pub fn push(&mut self, line: String, budget: &Budget) -> bool {
        if self.lines.len() >= budget.limits.results || line.len() > self.remaining_bytes(budget) {
            budget.stop("output limit reached");
            return false;
        }
        self.bytes += line.len() + usize::from(!self.lines.is_empty());
        self.lines.push(line);
        true
    }
}

/// Checking each read also bounds long nonmatching files and unterminated lines.
pub(super) struct BoundedReader<'a, R> {
    pub inner: R,
    pub budget: &'a Budget,
    pub file_left: u64,
}
impl<R: Read> Read for BoundedReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if !self.budget.check() {
            return Err(io::Error::other("search stopped"));
        }
        let available = buffer
            .len()
            .min(self.budget.remaining_read())
            .min(self.file_left as usize);
        if available == 0 {
            if buffer.is_empty() {
                return Ok(0);
            }
            if self.budget.remaining_read() == 0 {
                self.budget.stop("read byte limit reached");
            } else {
                self.budget.omitted.set(true);
            }
            return Err(io::Error::other("search read limit reached"));
        }
        let size = self.inner.read(&mut buffer[..available])?;
        self.file_left -= size as u64;
        self.budget.read.set(self.budget.read.get() + size);
        Ok(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_during_file_reads_stops_without_retryable_interrupted_error() {
        struct Cancels(CancellationToken);
        impl Read for Cancels {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                self.0.cancel();
                buffer[0] = b'x';
                Ok(1)
            }
        }
        let cancel = CancellationToken::new();
        let budget = Budget::new(Limits::default(), cancel.clone());
        let mut reader = BoundedReader {
            inner: Cancels(cancel),
            budget: &budget,
            file_left: 100,
        };
        let mut buffer = [0; 10];
        assert_eq!(reader.read(&mut buffer).unwrap(), 1);
        assert_ne!(
            reader.read(&mut buffer).unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        let result = budget.finish(Output::default());
        assert!(result.is_error);
        assert!(result.content.contains("cancelled"));
    }

    #[test]
    fn budget_counts_actual_read_bytes_and_output_separators() {
        let budget = Budget::new(
            Limits {
                read_bytes: 3,
                output_bytes: 3,
                ..Limits::default()
            },
            CancellationToken::new(),
        );
        let mut reader = BoundedReader {
            inner: io::Cursor::new(b"abcdefgh"),
            budget: &budget,
            file_left: 100,
        };
        let mut buffer = [0; 20];
        assert_eq!(reader.read(&mut buffer).unwrap(), 3);
        assert!(reader.read(&mut buffer).is_err());
        assert!(budget
            .finish(Output::default())
            .content
            .contains("read byte limit"));
        let budget = Budget::new(
            Limits {
                output_bytes: 3,
                ..Limits::default()
            },
            CancellationToken::new(),
        );
        let mut output = Output::default();
        assert!(output.push("a".into(), &budget));
        assert!(output.push("b".into(), &budget));
        assert!(!output.push("c".into(), &budget));
    }
}
