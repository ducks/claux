use std::{io, time::Duration};

/// Keep terminal loss out of Crossterm's event parser. Its alternate Unix
/// backend bounds the read even if the terminal disconnects after this check.
pub fn poll(timeout: Duration) -> io::Result<bool> {
    let _operation = crate::diagnostics::InputOperation::enter(false);
    let result = poll_inner(timeout);
    if result.is_err() {
        crate::diagnostics::input_error();
    }
    result
}

fn poll_inner(timeout: Duration) -> io::Result<bool> {
    #[cfg(unix)]
    {
        use std::{fs::File, io::IsTerminal, os::fd::AsRawFd};

        // Match Crossterm: stdin may be a PTY without a controlling terminal.
        let stdin = io::stdin();
        let tty = if stdin.is_terminal() {
            None
        } else {
            Some(File::open("/dev/tty")?)
        };
        let mut descriptor = nix::libc::pollfd {
            fd: tty
                .as_ref()
                .map_or_else(|| stdin.as_raw_fd(), AsRawFd::as_raw_fd),
            events: nix::libc::POLLIN,
            revents: 0,
        };
        // SAFETY: descriptor points to one initialized pollfd for a live file.
        if unsafe { nix::libc::poll(&mut descriptor, 1, 0) } == -1 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                return Ok(false);
            }
            return Err(error);
        }
        if descriptor.revents & (nix::libc::POLLHUP | nix::libc::POLLERR | nix::libc::POLLNVAL) != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "terminal disconnected",
            ));
        }
    }

    // The use-dev-tty backend skips input entirely for a zero timeout.
    #[cfg(unix)]
    let timeout = timeout.max(Duration::from_millis(1));
    crossterm::event::poll(timeout)
}

pub fn read() -> io::Result<crossterm::event::Event> {
    loop {
        if poll(Duration::from_millis(50))? {
            return read_ready();
        }
    }
}

pub fn read_ready() -> io::Result<crossterm::event::Event> {
    let _operation = crate::diagnostics::InputOperation::enter(true);
    let result = crossterm::event::read();
    if result.is_err() {
        crate::diagnostics::input_error();
    }
    result
}
