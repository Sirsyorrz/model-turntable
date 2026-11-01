#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod cli;
mod export;
mod lighting;
mod model;
mod render;
mod session;
mod settings;
mod theme;
mod update;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    if std::env::args_os().len() > 1 {
        attach_console();
    }
    let args = cli::Args::parse();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    match args.command {
        Some(cmd) => cli::run(cmd),
        None => app::run(args.model),
    }
}

/// Release builds on Windows use the GUI subsystem, so reattach to the
/// launching terminal for CLI output. Done before parsing so `--help` and
/// argument errors are visible too.
fn attach_console() {
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Console::AttachConsole(
            windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS,
        );
    }
}
