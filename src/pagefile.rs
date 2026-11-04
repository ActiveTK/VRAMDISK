//! Read-only diagnostics and guarded CLI surface for Windows page files.
//!
//! A successful `swapadd` operation cannot establish that paging over a
//! WinSpd user-mode storage service is deadlock-safe. Until that I/O path has
//! been validated in a recoverable Windows test environment, enable/disable
//! remain blocked and this module never changes Windows paging configuration.

use std::env;
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::process::Command;

use anyhow::{bail, Context, Result};
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

    /// Changes to paging are intentionally impossible while compatibility is
    /// unverified. This is a policy guard, not a missing `swapadd` invocation.
    pub fn enable(_size_bytes: u64) -> Result<()> {
        bail!("pagefile enable is blocked: WinSpd paging I/O is EXPERIMENTAL_UNVERIFIED")
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
    /// Refuse to enable until paging compatibility is validated.
    Enable { #[arg(long, value_parser = crate::cli::parse_size)] size: u64 },
    /// Report whether paging validation has been completed.
    Verify,
    /// Refuse to remove an active Windows pagefile.
    Disable,
}

/// Run the dedicated `pagefile` command path before normal mount argument
/// parsing. No action in this parser can modify system paging configuration.
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
        PagefileAction::Enable { size } => PagefileManager::enable(size),
        PagefileAction::Verify => PagefileManager::verify(),
        PagefileAction::Disable => PagefileManager::disable(),
    }
}

/// Check whether a candidate path names an actual file, without running it.
pub fn is_swapadd_file(path: &Path) -> bool {
    path.is_file()
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
    fn enabling_and_disabling_are_guarded() {
        assert!(PagefileManager::enable(2 * 1024 * 1024 * 1024).is_err());
        assert!(PagefileManager::disable().is_err());
    }
}
