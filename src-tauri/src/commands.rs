//! Tauri command surface. Each command is a thin adapter over the mount
//! [`Manager`], converting errors to strings for the JS bridge.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};
use tauri_plugin_dialog::DialogExt;

use crate::manager::{GpuDto, Manager, MountConfig, MountStatus};
use crate::CliSeed;

#[tauri::command]
pub fn list_gpus(manager: State<Manager>) -> Vec<GpuDto> {
    manager.list_gpus()
}

#[tauri::command]
pub fn list_free_drives(manager: State<Manager>) -> Vec<String> {
    manager.list_free_drives()
}

#[tauri::command]
pub fn mount_status(manager: State<Manager>) -> Option<MountStatus> {
    manager.status()
}

/// Whether nvCOMP is available for GPU compression. The GUI uses this to gray
/// out the "圧縮" mount option and the archive/compression panel instead of
/// silently falling back to CPU zstd, which is what the CLI does.
#[tauri::command]
pub fn nvcomp_available() -> bool {
    vramdisk::nvcomp::nvcomp_available()
}

/// CLI-flag-derived defaults for the setup screen, e.g. a shortcut launching
/// `vramdisk.exe --mount R: --compress` pre-fills those fields instead of
/// mounting automatically. Every field is `null` unless that specific flag
/// was present on this process's argv (see `vramdisk::cli::scan_overrides`),
/// so the frontend only overrides what was actually asked for.
#[derive(Debug, Default, Serialize)]
pub struct InitialOverrides {
    pub mount: Option<String>,
    pub size_bytes: Option<u64>,
    pub compress: Option<bool>,
    pub dedup: Option<bool>,
    pub device: Option<usize>,
}

#[tauri::command]
pub fn initial_overrides(seed: State<CliSeed>) -> InitialOverrides {
    match &seed.0 {
        Some(o) => InitialOverrides {
            mount: o.mount.clone(),
            size_bytes: o.size,
            compress: o.compress,
            dedup: o.dedup,
            device: o.device,
        },
        None => InitialOverrides::default(),
    }
}

#[tauri::command]
pub fn stats(manager: State<Manager>) -> Result<serde_json::Value, String> {
    manager.stats()
}

/// Open a native "select folder" dialog and return the chosen path, or
/// `None` if the user cancelled. Used by the "フォルダ" mount mode's
/// "参照..." button and the archive panel's folder targets. When a disk is
/// mounted, the dialog opens inside it so archive paths land on the volume.
#[tauri::command]
pub fn browse_folder(app: AppHandle, manager: State<Manager>) -> Option<String> {
    let mut dlg = app.dialog().file();
    if let Some(mp) = manager.mount_point() {
        dlg = dlg.set_directory(mp);
    }
    dlg.blocking_pick_folder()
        .and_then(|fp| fp.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

/// Open a native "select file" dialog (for the archive to extract), rooted at
/// the mounted volume when one exists.
#[tauri::command]
pub fn browse_file(app: AppHandle, manager: State<Manager>) -> Option<String> {
    let mut dlg = app.dialog().file();
    if let Some(mp) = manager.mount_point() {
        dlg = dlg.set_directory(mp);
    }
    dlg.blocking_pick_file()
        .and_then(|fp| fp.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

/// Open a native "save file" dialog (for the archive to write), rooted at the
/// mounted volume when one exists.
#[tauri::command]
pub fn browse_save(app: AppHandle, manager: State<Manager>) -> Option<String> {
    let mut dlg = app.dialog().file();
    if let Some(mp) = manager.mount_point() {
        dlg = dlg.set_directory(mp);
    }
    dlg.blocking_save_file()
        .and_then(|fp| fp.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

/// Mount the single VRAMDISK. On success: open the mount point in Explorer,
/// hide the main window (the tray keeps the mount alive and reachable), and
/// show a one-time confirmation.
#[tauri::command]
pub fn mount(cfg: MountConfig, app: AppHandle, manager: State<Manager>) -> Result<(), String> {
    if cfg.compress && !vramdisk::nvcomp::nvcomp_available() {
        return Err(
            "nvCOMP が見つからないため、この GUI では圧縮を利用できません。".to_string(),
        );
    }
    let mount_point = cfg.mount_point.clone();
    manager.mount(cfg)?;
    crate::on_mount_state_changed(&app, &manager);
    crate::after_mount_success(&app, &mount_point);
    Ok(())
}

#[tauri::command]
pub fn unmount(app: AppHandle, manager: State<Manager>) -> Result<(), String> {
    manager.unmount()?;
    crate::on_mount_state_changed(&app, &manager);
    Ok(())
}

/// Run a GPU batch-hash job over paths on the mounted volume via its
/// `$VRAMDISK\jobs` API, and return the parsed `result.json`.
#[tauri::command]
pub fn hash_job(
    paths: Vec<String>,
    algorithm: String,
    recursive: bool,
    manager: State<Manager>,
) -> Result<serde_json::Value, String> {
    if paths.is_empty() {
        return Err("no paths given".to_string());
    }
    let mount_point = manager
        .mount_point()
        .ok_or_else(|| "nothing is mounted".to_string())?;
    let norm = paths
        .iter()
        .map(|p| normalize_path(&mount_point, p))
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor = serde_json::json!({
        "op": "hash",
        "algorithm": algorithm,
        "paths": norm,
        "recursive": recursive,
    });
    submit_job(&mount_point, descriptor).map_err(|e| e.to_string())
}

#[derive(Debug, Deserialize)]
pub struct ArchiveCompressRequest {
    pub format: String,
    pub paths: Vec<String>,
    pub output: String,
}

#[derive(Debug, Deserialize)]
pub struct ArchiveExtractRequest {
    pub format: String,
    pub archive: String,
    pub output_dir: String,
}

/// GPU archive compression (`tar.zst` / `tar.lz4` / `tar.gz` / `zip`) over
/// paths on the mounted volume, via `$VRAMDISK\jobs`. Always recursive: a
/// non-recursive compress isn't a meaningful option for this tool.
#[tauri::command]
pub fn archive_compress_job(
    req: ArchiveCompressRequest,
    manager: State<Manager>,
) -> Result<serde_json::Value, String> {
    if req.paths.is_empty() {
        return Err("no paths given".to_string());
    }
    let mount_point = manager
        .mount_point()
        .ok_or_else(|| "nothing is mounted".to_string())?;
    let norm = req
        .paths
        .iter()
        .map(|p| normalize_path(&mount_point, p))
        .collect::<Result<Vec<_>, _>>()?;
    let output = normalize_path(&mount_point, &req.output)?;
    let descriptor = serde_json::json!({
        "op": "archive.compress",
        "format": req.format,
        "paths": norm,
        "output": output,
        "recursive": true,
    });
    submit_job(&mount_point, descriptor).map_err(|e| e.to_string())
}

/// GPU archive extraction over an archive file on the mounted volume.
#[tauri::command]
pub fn archive_extract_job(
    req: ArchiveExtractRequest,
    manager: State<Manager>,
) -> Result<serde_json::Value, String> {
    let mount_point = manager
        .mount_point()
        .ok_or_else(|| "nothing is mounted".to_string())?;
    let archive = normalize_path(&mount_point, &req.archive)?;
    let output_dir = normalize_path(&mount_point, &req.output_dir)?;
    let descriptor = serde_json::json!({
        "op": "archive.extract",
        "format": req.format,
        "archive": archive,
        "output_dir": output_dir,
    });
    submit_job(&mount_point, descriptor).map_err(|e| e.to_string())
}

/// Normalize a user-entered path to a drive-relative "\..." form.
///
/// Accepts drive-relative input ("\data", "data") as well as an absolute path
/// that lives under the mount point itself — whether the mount point is a
/// drive letter ("R:\data", "R:data") or a directory
/// ("C:\vramdisk\data", matching a directory mount point). An absolute path
/// that clearly points somewhere else (another drive, or a UNC path) is
/// rejected with a clear error instead of being silently mangled into a
/// bogus drive-relative path.
fn normalize_path(mount_point: &str, p: &str) -> Result<String, String> {
    let input = p.trim().replace('/', "\\");
    let mount_trimmed = mount_point.trim_end_matches('\\');

    if let Some(prefix) = input.get(..mount_trimmed.len()) {
        if prefix.eq_ignore_ascii_case(mount_trimmed) {
            let rest = &input[mount_trimmed.len()..];
            let rest = rest.strip_prefix('\\').unwrap_or(rest);
            return Ok(if rest.is_empty() {
                "\\".to_string()
            } else {
                format!("\\{rest}")
            });
        }
    }

    let looks_like_elsewhere =
        (input.len() >= 2 && input.as_bytes()[1] == b':') || input.starts_with("\\\\");
    if looks_like_elsewhere {
        return Err(format!(
            "パス \"{p}\" はマウント先 {mount_point} 上のパスではありません"
        ));
    }

    Ok(if input.starts_with('\\') {
        input
    } else {
        format!("\\{input}")
    })
}

/// Submit a job descriptor to the volume's `$VRAMDISK\jobs` API and block
/// until it reaches a terminal state, then return the parsed result document.
/// Shared by hash and archive jobs; both are simple submit-then-poll flows.
fn submit_job(
    mount_point: &str,
    descriptor: serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    use std::io::Write;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    let job_id = format!(
        "gui{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    // Submit: CREATE_NEW the pending descriptor, write, close.
    let pending = format!("{mount_point}\\$VRAMDISK\\jobs\\pending\\{job_id}.json");
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)
            .map_err(|e| anyhow::anyhow!("submit {pending}: {e}"))?;
        f.write_all(descriptor.to_string().as_bytes())?;
    }

    // Poll status.json until terminal, then read result.json.
    let status_path = format!("{mount_point}\\$VRAMDISK\\jobs\\{job_id}\\status.json");
    let result_path = format!("{mount_point}\\$VRAMDISK\\jobs\\{job_id}\\result.json");
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Ok(text) = std::fs::read_to_string(&status_path) {
            if let Ok(status) = serde_json::from_str::<serde_json::Value>(&text) {
                if status["terminal"].as_bool().unwrap_or(false) {
                    break;
                }
            }
        }
        if Instant::now() > deadline {
            anyhow::bail!("job timed out");
        }
        std::thread::sleep(Duration::from_millis(80));
    }

    let text = std::fs::read_to_string(&result_path)
        .map_err(|e| anyhow::anyhow!("read {result_path}: {e}"))?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_path_relative_unchanged() {
        assert_eq!(normalize_path("R:", "\\data").unwrap(), "\\data");
        assert_eq!(normalize_path("R:", "data").unwrap(), "\\data");
        assert_eq!(normalize_path(r"C:\vramdisk", "\\data").unwrap(), "\\data");
    }

    #[test]
    fn normalize_path_absolute_on_drive_letter_mount() {
        assert_eq!(normalize_path("R:", "R:\\data").unwrap(), "\\data");
        assert_eq!(normalize_path("R:", "r:data").unwrap(), "\\data");
        assert_eq!(normalize_path("R:", "R:").unwrap(), "\\");
        assert!(normalize_path("R:", "C:\\Windows").is_err());
    }

    #[test]
    fn normalize_path_absolute_on_directory_mount() {
        let mount = r"C:\vramdisk";
        assert_eq!(
            normalize_path(mount, r"C:\vramdisk\data").unwrap(),
            "\\data"
        );
        assert_eq!(
            normalize_path(mount, r"c:\VRAMDISK\data\a.txt").unwrap(),
            "\\data\\a.txt"
        );
        assert_eq!(normalize_path(mount, mount).unwrap(), "\\");
        assert!(normalize_path(mount, r"C:\other\data").is_err());
        assert!(normalize_path(mount, r"D:\vramdisk\data").is_err());
    }
}
