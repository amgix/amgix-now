use crate::common::{host_os, HostOs};

pub fn init() {
    if matches!(host_os(), HostOs::Windows) {
        platform_windows::enable_console_ansi();
    }
}

#[cfg(windows)]
mod platform_windows {
    pub fn enable_console_ansi() {
        let _ = nu_ansi_term::enable_ansi_support();
    }
}

#[cfg(not(windows))]
mod platform_windows {
    pub fn enable_console_ansi() {}
}
