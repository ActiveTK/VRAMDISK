//! Windows pagefile diagnostics and explicit swapadd integration.
//!
//! A successful `swapadd` operation cannot establish that paging over a
//! WinSpd user-mode storage service is deadlock-safe. Until that I/O path has
//! been validated in a recoverable Windows test environment, the GUI never
//! changes paging automatically. An explicit user command can invoke swapadd
//! to try adding a pagefile while preserving existing Windows pagefiles.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingCompatibility {
    ExperimentalUnverified,
    Unsupported,
}

impl PagingCompatibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExperimentalUnverified => "EXPERIMENTAL_UNVERIFIED",
            Self::Unsupported => "UNSUPPORTED",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PagefileReport {
    pub compatibility: PagingCompatibility,
    pub auto_enable: bool,
    pub swapadd_path: Option<PathBuf>,
    pub active_pagefiles: Option<String>,
    pub commit_status: Option<String>,
    pub diagnostics: Vec<String>,
}

pub struct PagefileManager;

impl PagefileManager {
    /// Read-only health and compatibility check. Does not run `swapadd`, alter
    /// registry settings, create files, or change the running paging state.
    pub fn check() -> PagefileReport {
        let mut diagnostics = vec![
            "WinSpd is an optional feature; this status check does not query active WinSpd devices.".to_string(),
            "WinSpd paging I/O safety has no validation evidence; automatic pagefile enable is disabled.".to_string(),
            "Keep the Windows-managed pagefile on a persistent system volume.".to_string(),
        ];
        let swapadd_path = discover_swapadd();
        if swapadd_path.is_some() {
            diagnostics.push("swapadd was found; it will not be executed by this check.".into());
        } else {
            diagnostics.push(
                "swapadd.exe was not found beside vramdisk.exe, in C:\\ProgramData, or on PATH (set VRAMDISK_SWAPADD to its full path to identify it).".into(),
            );
        }

        let (active_pagefiles, commit_status) = query_windows_memory_status();
        if active_pagefiles.is_none() {
            diagnostics.push("Active pagefile details could not be queried.".into());
        }

        PagefileReport {
            compatibility: PagingCompatibility::ExperimentalUnverified,
            auto_enable: false,
            swapadd_path,
            active_pagefiles,
            commit_status,
            diagnostics,
        }
    }

    pub fn status() -> PagefileReport {
        Self::check()
    }

    /// Explicitly ask swapadd to add a pagefile. This does not remove or change
    /// existing pagefiles, and it does not establish that paging over WinSpd is
    /// safe. The caller must have administrative privileges.
    pub fn enable(path: &Path, min_size: u64, max_size: u64) -> Result<()> {
        ensure!(min_size > 0, "minimum pagefile size must be greater than zero");
        ensure!(max_size >= min_size, "maximum pagefile size must be at least the minimum size");
        let path_text = path.to_string_lossy();
        ensure!(
            path_text.len() >= 14
                && path_text.as_bytes()[1] == b':'
                && path_text.as_bytes()[2] == b'\\'
                && path_text.to_ascii_lowercase().ends_with("\\pagefile.sys"),
            "pagefile path must be an absolute drive path ending in \\pagefile.sys"
        );
        ensure!(path.parent().is_some_and(Path::is_dir), "pagefile volume is not mounted or accessible");
        ensure!(!path.exists(), "target already exists; choose a new pagefile.sys path so existing pagefiles are preserved");
        #[cfg(windows)]
        ensure!(available_space(path)? >= max_size, "the target volume does not have enough free space for the requested maximum pagefile size");

        let swapadd = discover_swapadd().context("swapadd.exe was not found")?;
        let output = Command::new(&swapadd)
            .arg(path)
            .arg(format_size(min_size))
            .arg(format_size(max_size))
            .output()
            .with_context(|| format!("could not launch {}", swapadd.display()))?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        ensure!(
            output.status.success(),
            "swapadd failed ({}): {}{}",
            output.status,
            stdout,
            if stderr.is_empty() { String::new() } else { format!("\n{stderr}") }
        );
        Ok(())
    }

    pub fn disable() -> Result<()> {
        bail!("pagefile disable is blocked: VRAMDISK does not manage or remove active Windows pagefiles")
    }

    pub fn verify() -> Result<()> {
        let report = Self::status();
        println!("Compatibility: {}", report.compatibility.as_str());
        println!("Automatic enable: {}", report.auto_enable);
        println!("Active pagefiles: {}", report.active_pagefiles.as_deref().unwrap_or("unavailable"));
        println!("Commit status: {}", report.commit_status.as_deref().unwrap_or("unavailable"));
        bail!("paging validation is not available; status above is observational only")
    }
}

fn discover_swapadd() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("VRAMDISK_SWAPADD") {
        candidates.push(PathBuf::from(path));
    }

    if let Some(executable_dir) = env::current_exe().ok().and_then(|path| path.parent().map(Path::to_path_buf)) {
        candidates.extend([executable_dir.join("swapadd64.exe"), executable_dir.join("swapadd.exe")]);
    }

    candidates.extend([
        PathBuf::from(r"C:\ProgramData\swapadd64.exe"),
        PathBuf::from(r"C:\ProgramData\swapadd.exe"),
    ]);

    candidates.extend(
        env::split_paths(&env::var_os("PATH").unwrap_or_default())
            .flat_map(|dir| [dir.join("swapadd64.exe"), dir.join("swapadd.exe")]),
    );

    candidates.into_iter().find(|path| path.is_file())
}

#[cfg(windows)]
fn query_windows_memory_status() -> (Option<String>, Option<String>) {
    const SCRIPT: &str = r#"$p=@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage); $m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | Select-Object CommittedBytes,CommitLimit; [pscustomobject]@{Pagefiles=$p;Memory=$m} | ConvertTo-Json -Compress -Depth 4"#;
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", SCRIPT])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let json = String::from_utf8_lossy(&out.stdout);
            let Ok(value) = serde_json::from_str::<serde_json::Value>(json.trim()) else {
                return (None, Some("Windows returned malformed CIM JSON".into()));
            };
            let pagefiles = value
                .get("Pagefiles")
                .map(ToString::to_string)
                .unwrap_or_else(|| "[]".into());
            let memory = value.get("Memory");
            let commit = match memory {
                Some(memory) => format!(
                    "CommittedBytes={}, CommitLimit={}",
                    memory.get("CommittedBytes").map_or("unavailable".into(), ToString::to_string),
                    memory.get("CommitLimit").map_or("unavailable".into(), ToString::to_string)
                ),
                None => "unavailable".into(),
            };
            (Some(pagefiles), Some(commit))
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let err = if err.is_empty() { format!("PowerShell exited with {}", out.status) } else { err };
            (None, Some(err))
        }
        Err(err) => (None, Some(format!("could not launch PowerShell: {err}"))),
    }
}

#[cfg(not(windows))]
fn query_windows_memory_status() -> (Option<String>, Option<String>) {
    (None, Some("Windows-only query".into()))
}

#[derive(Parser)]
#[command(name = "vramdisk pagefile", no_binary_name = true)]
struct PagefileCli {
    #[command(subcommand)]
    action: PagefileAction,
}

#[derive(Subcommand)]
enum PagefileAction {
    /// Display current pagefile and compatibility status (read-only).
    Status,
    /// Check dependencies and paging compatibility (read-only).
    Check,
    /// Explicitly add a pagefile with swapadd; existing pagefiles are preserved.
    Enable {
        #[arg(long)] path: PathBuf,
        #[arg(long, value_parser = crate::cli::parse_size)] min_size: u64,
        #[arg(long, value_parser = crate::cli::parse_size)] max_size: u64,
    },
    /// Report whether paging validation has been completed.
    Verify,
    /// Refuse to remove an active Windows pagefile.
    Disable,
}

/// Run the dedicated `pagefile` command path before normal mount argument
/// parsing. The enable action invokes swapadd only when explicitly requested.
pub fn run_cli(args: &[String]) -> Result<()> {
    let cli = PagefileCli::try_parse_from(args).context("parse pagefile command")?;
    match cli.action {
        PagefileAction::Status | PagefileAction::Check => {
            let report = PagefileManager::check();
            println!("Paging compatibility: {}", report.compatibility.as_str());
            println!("Automatic pagefile enable: {}", report.auto_enable);
            println!("swapadd: {}", report.swapadd_path.as_deref().map_or_else(|| "not found".into(), |p| p.display().to_string()));
            println!("Active pagefiles: {}", report.active_pagefiles.as_deref().unwrap_or("unavailable"));
            println!("Commit status: {}", report.commit_status.as_deref().unwrap_or("unavailable"));
            for diagnostic in report.diagnostics {
                println!("- {diagnostic}");
            }
            Ok(())
        }
        PagefileAction::Enable { path, min_size, max_size } => PagefileManager::enable(&path, min_size, max_size),
        PagefileAction::Verify => PagefileManager::verify(),
        PagefileAction::Disable => PagefileManager::disable(),
    }
}

/// Check whether a candidate path names an actual file, without running it.
pub fn is_swapadd_file(path: &Path) -> bool {
    path.is_file()
}

fn format_size(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    if bytes % MIB == 0 {
        format!("{}M", bytes / MIB)
    } else {
        bytes.to_string()
    }
}

#[cfg(windows)]
fn available_space(path: &Path) -> Result<u64> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            free_to_caller: *mut u64,
            total_bytes: *mut u64,
            total_free: *mut u64,
        ) -> i32;
    }

    let volume_root = path.parent().context("pagefile path has no volume root")?;
    let wide_path: Vec<u16> = volume_root.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut free = 0u64;
    let result = unsafe {
        GetDiskFreeSpaceExW(
            wide_path.as_ptr(),
            &mut free,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    ensure!(result != 0, "could not query free space on the pagefile volume");
    Ok(free)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paging_is_unverified_and_never_auto_enabled() {
        let report = PagefileManager::check();
        assert_eq!(report.compatibility, PagingCompatibility::ExperimentalUnverified);
        assert!(!report.auto_enable);
        assert!(report.diagnostics.iter().any(|line| line.contains("does not query active WinSpd devices")));
    }

    #[test]
    fn disabling_is_guarded() {
        assert!(PagefileManager::disable().is_err());
    }
}
