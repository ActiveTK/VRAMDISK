//! Mount manager.
//!
//! A single dedicated thread owns the one live [`MountedVramDisk`] (which
//! wraps a WinFsp host and the CUDA context), if any. All GUI commands talk to
//! it over a channel, so the non-`Send` mount handle never crosses a thread
//! boundary and CUDA stays affine to one owning thread. Only plain `Send` data
//! (configs, JSON) travels over the channel.
//!
//! Only one disk can be mounted at a time: this app manages a single VRAMDISK,
//! not a fleet across multiple GPUs. The mount point itself can be either a
//! drive letter (`R:`) or an existing empty directory (WinFsp supports both
//! transparently via `FspFileSystemSetMountPoint`).

use std::sync::mpsc::{channel, Sender};
use std::thread;

use serde::{Deserialize, Serialize};
#[cfg(all(windows, feature = "winspd"))]
use anyhow::Context;

use vramdisk::cuda::Vram;
use vramdisk::engine::StorageEngine;
use vramdisk::{default_size, round_up_to_chunk};
#[cfg(all(windows, feature = "winspd"))]
use vramdisk::block::{BlockDeviceConfig, GpuBlockStorage};
#[cfg(all(windows, feature = "winspd"))]
use vramdisk::winspd::WinSpdDevice;

/// A CUDA device, as sent to the UI.
#[derive(Debug, Clone, Serialize)]
pub struct GpuDto {
    pub ordinal: usize,
    pub name: String,
    pub total_vram: u64,
    pub default_size: u64,
}

/// Mount request coming from the UI. `size` is in bytes; `None` means "use the
/// device default (max(0.8*VRAM, 2GiB))". `mount_point` is either a drive
/// letter ("R:") or a path to an existing empty directory.
#[derive(Debug, Clone, Deserialize)]
pub struct MountConfig {
    #[serde(default)]
    pub size: Option<u64>,
    pub mount_point: String,
    #[serde(default)]
    pub device: usize,
    #[serde(default)]
    pub compress: bool,
    #[serde(default)]
    pub dedup: bool,
}

/// The (single) live mount, as reported to the UI.
#[derive(Debug, Clone, Serialize)]
pub struct MountStatus {
    pub mount_point: String,
    pub device: usize,
    pub size: u64,
    pub compress: bool,
    pub dedup: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PagefileDiskConfig {
    pub pagefile_size: u64,
    #[serde(default)]
    pub device: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PagefileDiskStatus {
    pub drive: String,
    pub pagefile_path: String,
    pub device: usize,
    pub pagefile_size: u64,
    pub disk_size: u64,
    pub disable_pending_restart: bool,
}

/// Reply channel for a request.
type Reply<T> = Sender<T>;

/// The manager thread is spawned before Tauri exists, so it cannot reach
/// `UiLang` app state the way a command can. Rather than handing it a shared
/// handle to mutable state it would have to lock on every failure, the three
/// commands that can produce a user-visible message carry the caller's
/// language with the request: the caller (a Tauri command, or the tray path in
/// `main.rs`) always knows it, and the message is then rendered in whatever
/// language was selected at the moment the request was made.
enum Cmd {
    ListGpus(Reply<Vec<GpuDto>>),
    ListFreeDrives(Reply<Vec<String>>),
    Mount(MountConfig, String, Reply<Result<(), String>>),
    Unmount(String, Reply<Result<(), String>>),
    Status(Reply<Option<MountStatus>>),
    Stats(String, Reply<Result<serde_json::Value, String>>),
    MountPoint(Reply<Option<String>>),
    StartPagefileDisk(PagefileDiskConfig, String, Reply<Result<PagefileDiskStatus, String>>),
    DisablePagefileDisk(String, Reply<Result<PagefileDiskStatus, String>>),
    PagefileDiskStatus(Reply<Option<PagefileDiskStatus>>),
    Shutdown(Reply<()>),
}

/// Handle to the manager thread. Cloneable, `Send + Sync`.
#[derive(Clone)]
pub struct Manager {
    tx: Sender<Cmd>,
}

impl Manager {
    /// Spawn the owning thread and return a handle.
    pub fn spawn() -> Self {
        let (tx, rx) = channel::<Cmd>();
        thread::Builder::new()
            .name("vramdisk-mount-manager".into())
            .spawn(move || {
                let mut state = ManagerState::default();
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Cmd::ListGpus(reply) => {
                            let _ = reply.send(state.list_gpus());
                        }
                        Cmd::ListFreeDrives(reply) => {
                            let _ = reply.send(list_free_drives());
                        }
                        Cmd::Mount(cfg, lang, reply) => {
                            let _ = reply.send(state.mount(cfg, &lang).map_err(|e| e.to_string()));
                        }
                        Cmd::Unmount(lang, reply) => {
                            let _ = reply.send(state.unmount(&lang).map_err(|e| e.to_string()));
                        }
                        Cmd::Status(reply) => {
                            let _ = reply.send(state.status());
                        }
                        Cmd::Stats(lang, reply) => {
                            let _ = reply.send(state.stats(&lang).map_err(|e| e.to_string()));
                        }
                        Cmd::MountPoint(reply) => {
                            let _ = reply.send(state.mount_point());
                        }
                        Cmd::StartPagefileDisk(cfg, lang, reply) => {
                            let result = state
                                .start_pagefile_disk(cfg, &lang)
                                .map_err(|e| format!("{e:#}"));
                            let _ = reply.send(result);
                        }
                        Cmd::DisablePagefileDisk(lang, reply) => {
                            let result = state
                                .disable_pagefile_disk(&lang)
                                .map_err(|e| format!("{e:#}"));
                            let _ = reply.send(result);
                        }
                        Cmd::PagefileDiskStatus(reply) => {
                            let _ = reply.send(state.pagefile_disk_status());
                        }
                        Cmd::Shutdown(reply) => {
                            state.shutdown();
                            let _ = reply.send(());
                            break;
                        }
                    }
                }
            })
            .expect("spawn mount manager thread");
        Manager { tx }
    }

    fn request<T, F>(&self, make: F) -> T
    where
        F: FnOnce(Reply<T>) -> Cmd,
    {
        let (tx, rx) = channel::<T>();
        // If the manager thread is gone the app is shutting down; the recv error
        // is surfaced by the caller-visible panic only in that already-fatal case.
        self.tx
            .send(make(tx))
            .expect("mount manager thread not running");
        rx.recv().expect("mount manager dropped reply")
    }

    pub fn list_gpus(&self) -> Vec<GpuDto> {
        self.request(Cmd::ListGpus)
    }

    pub fn list_free_drives(&self) -> Vec<String> {
        self.request(Cmd::ListFreeDrives)
    }

    /// `lang` is the UI language ("ja" / "en") the failure message should be
    /// rendered in; see [`Cmd`].
    pub fn mount(&self, cfg: MountConfig, lang: &str) -> Result<(), String> {
        let lang = lang.to_string();
        self.request(|r| Cmd::Mount(cfg, lang, r))
    }

    pub fn unmount(&self, lang: &str) -> Result<(), String> {
        let lang = lang.to_string();
        self.request(|r| Cmd::Unmount(lang, r))
    }

    /// The current mount, if any.
    pub fn status(&self) -> Option<MountStatus> {
        self.request(Cmd::Status)
    }

    pub fn stats(&self, lang: &str) -> Result<serde_json::Value, String> {
        let lang = lang.to_string();
        self.request(|r| Cmd::Stats(lang, r))
    }

    /// The mount point (drive letter or directory) of the current mount, if
    /// any. Used by job commands to address the mounted volume without
    /// holding the manager thread for the duration of a (possibly slow) GPU
    /// job.
    pub fn mount_point(&self) -> Option<String> {
        self.request(Cmd::MountPoint)
    }

    pub fn start_pagefile_disk(
        &self,
        cfg: PagefileDiskConfig,
        lang: &str,
    ) -> Result<PagefileDiskStatus, String> {
        let lang = lang.to_string();
        self.request(|reply| Cmd::StartPagefileDisk(cfg, lang, reply))
    }

    pub fn pagefile_disk_status(&self) -> Option<PagefileDiskStatus> {
        self.request(Cmd::PagefileDiskStatus)
    }

    pub fn disable_pagefile_disk(&self, lang: &str) -> Result<PagefileDiskStatus, String> {
        let lang = lang.to_string();
        self.request(|reply| Cmd::DisablePagefileDisk(lang, reply))
    }

    /// Unmount (if mounted) and stop the thread. Called on real app exit.
    /// Idempotent: safe to call again after the thread has already stopped.
    pub fn shutdown(&self) {
        let (tx, rx) = channel::<()>();
        if self.tx.send(Cmd::Shutdown(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

struct MountRecord {
    status: MountStatus,
    #[cfg(windows)]
    mounted: vramdisk::fs::MountedVramDisk,
}

#[derive(Default)]
struct ManagerState {
    mount: Option<MountRecord>,
    pagefile_disk: Option<PagefileDiskRecord>,
}

struct PagefileDiskRecord {
    status: PagefileDiskStatus,
    pagefile_added: bool,
    #[cfg(all(windows, feature = "winspd"))]
    _device: WinSpdDevice,
}

impl ManagerState {
    fn list_gpus(&self) -> Vec<GpuDto> {
        vramdisk::list_gpus()
            .into_iter()
            .map(|g| GpuDto {
                ordinal: g.ordinal,
                name: g.name,
                total_vram: g.total_vram,
                default_size: g.default_size,
            })
            .collect()
    }

    fn status(&self) -> Option<MountStatus> {
        self.mount.as_ref().map(|r| r.status.clone())
    }

    fn mount_point(&self) -> Option<String> {
        self.mount.as_ref().map(|r| r.status.mount_point.clone())
    }

    fn pagefile_disk_status(&self) -> Option<PagefileDiskStatus> {
        self.pagefile_disk
            .as_ref()
            .filter(|record| record.pagefile_added)
            .map(|record| record.status.clone())
    }

    fn start_pagefile_disk(
        &mut self,
        cfg: PagefileDiskConfig,
        _lang: &str,
    ) -> anyhow::Result<PagefileDiskStatus> {
        anyhow::ensure!(self.mount.is_none(), "Unmount the file RAM disk before starting the WinSpd pagefile disk");
        anyhow::ensure!(self.pagefile_disk.is_none(), "A WinSpd pagefile disk is already running");
        anyhow::ensure!(cfg.pagefile_size > 0, "pagefile size must be greater than zero");
        anyhow::ensure!(
            list_free_drives().iter().any(|drive| drive.eq_ignore_ascii_case("M:")),
            "Drive M: is already assigned; free it before creating the WinSpd pagefile disk"
        );

        #[cfg(all(windows, feature = "winspd"))]
        {
            start_winspd_pagefile_disk(self, cfg)
        }
        #[cfg(not(all(windows, feature = "winspd")))]
        {
            let _ = cfg;
            anyhow::bail!("This build does not include the Windows WinSpd feature")
        }
    }

    fn disable_pagefile_disk(&mut self, _lang: &str) -> anyhow::Result<PagefileDiskStatus> {
        let record = self
            .pagefile_disk
            .as_mut()
            .filter(|record| record.pagefile_added)
            .context("No VRAMDISK pagefile is active in this session")?;
        if record.status.disable_pending_restart {
            return Ok(record.status.clone());
        }
        vramdisk::pagefile::PagefileManager::disable_specific(
            std::path::Path::new(&record.status.pagefile_path),
        )
        .context("could not remove the M: pagefile startup setting")?;
        record.status.disable_pending_restart = true;
        Ok(record.status.clone())
    }

    #[cfg(windows)]
    fn mount(&mut self, cfg: MountConfig, lang: &str) -> anyhow::Result<()> {
        if self.mount.is_some() {
            anyhow::bail!("{}", crate::tr(lang, "err_already_mounted"));
        }
        if self.pagefile_disk.is_some() {
            anyhow::bail!("Stop the WinSpd pagefile workflow before mounting the file RAM disk");
        }
        let mount_point = normalize_mount_point(&cfg.mount_point);
        if mount_point.is_empty() {
            anyhow::bail!("{}", crate::tr(lang, "err_mount_point_required"));
        }
        let directory_mode = !is_drive_letter_form(&mount_point);
        if directory_mode {
            validate_mount_point(&mount_point, lang)?;
        }

        let total = Vram::device_total_mem(cfg.device)?;
        let size = match cfg.size {
            Some(s) => round_up_to_chunk(s),
            None => default_size(total),
        };
        if size == 0 {
            anyhow::bail!("{}", crate::tr(lang, "err_size_zero"));
        }
        if size > total {
            anyhow::bail!(
                "{}",
                crate::trf(
                    lang,
                    "err_size_exceeds_vram",
                    &[&size.to_string(), &total.to_string()],
                )
            );
        }

        let vram = Vram::new(cfg.device, size)?;
        let engine = StorageEngine::new(vram, cfg.compress, cfg.dedup)?;

        // WinFsp creates the mount-point directory itself and fails with
        // STATUS_OBJECT_NAME_COLLISION if anything already occupies that
        // path — even an already-verified-empty directory. If the user
        // picked an existing empty folder (the natural "参照..." UX), remove
        // it right before mounting and put it back if the mount fails, so a
        // failed attempt never leaves the user's folder deleted.
        let removed_existing = directory_mode && std::path::Path::new(&mount_point).exists();
        if removed_existing {
            std::fs::remove_dir(&mount_point).map_err(|e| {
                anyhow::anyhow!(
                    "{}",
                    crate::trf(
                        lang,
                        "err_mount_prepare_folder",
                        &[&mount_point, &e.to_string()],
                    )
                )
            })?;
        }

        let mounted = match vramdisk::fs::mount(engine, &mount_point, "VRAMDISK") {
            Ok(m) => m,
            Err(e) => {
                if removed_existing {
                    if let Err(restore) = std::fs::create_dir(&mount_point) {
                        // Both the mount and the restore failed: say so
                        // instead of silently leaving the user's folder gone.
                        return Err(anyhow::anyhow!(
                            "{}",
                            crate::trf(
                                lang,
                                "err_mount_restore_folder",
                                &[&format!("{e:#}"), &restore.to_string()],
                            )
                        ));
                    }
                }
                // The engine has no language, so it reports this one in English
                // by a known constant; swap in the localised wording here.
                if e.to_string() == vramdisk::fs::WINFSP_DRIVER_NOT_LOADED {
                    return Err(anyhow::anyhow!(
                        "{}",
                        crate::tr(lang, "err_winfsp_driver_not_loaded")
                    ));
                }
                return Err(e);
            }
        };

        self.mount = Some(MountRecord {
            status: MountStatus {
                mount_point,
                device: cfg.device,
                size,
                compress: cfg.compress,
                dedup: cfg.dedup,
            },
            mounted,
        });
        Ok(())
    }

    #[cfg(not(windows))]
    fn mount(&mut self, _cfg: MountConfig, lang: &str) -> anyhow::Result<()> {
        anyhow::bail!("{}", crate::tr(lang, "err_mount_windows_only"))
    }

    fn unmount(&mut self, lang: &str) -> anyhow::Result<()> {
        match self.mount.take() {
            Some(_record) => {
                #[cfg(windows)]
                _record.mounted.unmount();
                Ok(())
            }
            None => anyhow::bail!("{}", crate::tr(lang, "err_not_mounted")),
        }
    }

    /// Read the mounted volume's `$VRAMDISK\stats.json` and return it parsed.
    fn stats(&self, lang: &str) -> anyhow::Result<serde_json::Value> {
        let Some(record) = self.mount.as_ref() else {
            anyhow::bail!("{}", crate::tr(lang, "err_not_mounted"));
        };
        let path = format!("{}\\$VRAMDISK\\stats.json", record.status.mount_point);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            anyhow::anyhow!(
                "{}",
                crate::trf(lang, "err_read_failed", &[&path, &e.to_string()])
            )
        })?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            anyhow::anyhow!(
                "{}",
                crate::trf(lang, "err_parse_failed", &[&path, &e.to_string()])
            )
        })?;
        Ok(value)
    }

    fn shutdown(&mut self) {
        if let Some(_record) = self.mount.take() {
            #[cfg(windows)]
            _record.mounted.unmount();
        }
    }
}

#[cfg(all(windows, feature = "winspd"))]
fn start_winspd_pagefile_disk(
    state: &mut ManagerState,
    cfg: PagefileDiskConfig,
) -> anyhow::Result<PagefileDiskStatus> {
    use std::path::Path;
    const FILESYSTEM_OVERHEAD: u64 = 32 * 1024 * 1024;
    let disk_size = cfg
        .pagefile_size
        .checked_add(FILESYSTEM_OVERHEAD)
        .context("pagefile disk size overflow")?;
    let total_vram = Vram::device_total_mem(cfg.device)?;
    anyhow::ensure!(
        disk_size <= total_vram,
        "pagefile plus filesystem overhead ({disk_size} bytes) exceeds GPU VRAM ({total_vram} bytes)"
    );
    let disks_before = list_windows_disk_numbers()?;

    let storage_cfg = BlockDeviceConfig::new(disk_size, vramdisk::block::DEFAULT_LOGICAL_SECTOR_SIZE)?;
    let storage = GpuBlockStorage::new(cfg.device, storage_cfg)?;
    let device = WinSpdDevice::start(Box::new(storage), None)
        .context("could not publish the GPU-backed WinSpd disk")?;
    let status = PagefileDiskStatus {
        drive: "M:".into(),
        pagefile_path: r"M:\pagefile.sys".into(),
        device: cfg.device,
        pagefile_size: cfg.pagefile_size,
        disk_size,
        disable_pending_restart: false,
    };
    // Retain the device even if partitioning or swapadd fails; dropping it
    // while Windows may have opened the new disk would make cleanup unsafe.
    state.pagefile_disk = Some(PagefileDiskRecord {
        status: status.clone(),
        pagefile_added: false,
        _device: device,
    });

    prepare_new_pagefile_volume(&disks_before, disk_size)
        .context("WinSpd disk is running, but automatic M: partitioning/formatting failed")?;
    vramdisk::pagefile::PagefileManager::enable(
        Path::new(&status.pagefile_path),
        cfg.pagefile_size,
        cfg.pagefile_size,
    )
    .context("M: was created and formatted, but swapadd could not add the pagefile")?;
    if let Some(record) = state.pagefile_disk.as_mut() {
        record.pagefile_added = true;
    }
    Ok(status)
}

#[cfg(all(windows, feature = "winspd"))]
fn list_windows_disk_numbers() -> anyhow::Result<Vec<u32>> {
    use std::process::Command;

    let script = "$ErrorActionPreference='Stop'; $d=@(Get-Disk | Select-Object -ExpandProperty Number); ConvertTo-Json -Compress -InputObject $d";
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
        .context("could not enumerate Windows disks")?;
    anyhow::ensure!(
        output.status.success(),
        "Get-Disk failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .context("Get-Disk returned invalid JSON")?;
    let values = value.as_array().cloned().unwrap_or_else(|| vec![value]);
    values
        .into_iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .context("Get-Disk returned an invalid disk number")
        })
        .collect()
}

#[cfg(all(windows, feature = "winspd"))]
fn prepare_new_pagefile_volume(disks_before: &[u32], disk_size: u64) -> anyhow::Result<()> {
    use std::process::Command;

    let before = disks_before
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        r#"$ErrorActionPreference='Stop'
$before=@({before})
$expected=[uint64]{disk_size}
$disk=$null
for($i=0; $i -lt 120 -and $null -eq $disk; $i++) {{
  $matches=@(Get-Disk | Where-Object {{ $_.Number -notin $before -and $_.FriendlyName -match 'VRAMDISK' -and [Math]::Abs([double]$_.Size-[double]$expected) -lt 16777216 }})
  if($matches.Count -gt 1) {{ throw 'More than one new VRAMDISK disk appeared; refusing to format an ambiguous target.' }}
  if($matches.Count -eq 1) {{ $disk=$matches[0] }} else {{ Start-Sleep -Milliseconds 500 }}
}}
if($null -eq $disk) {{ throw 'The new WinSpd VRAMDISK did not appear in Get-Disk.' }}
if($disk.PartitionStyle -ne 'RAW') {{ throw 'The new VRAMDISK is not RAW; refusing to overwrite a partitioned disk.' }}
if($disk.IsReadOnly) {{ Set-Disk -Number $disk.Number -IsReadOnly $false }}
if($disk.IsOffline) {{ Set-Disk -Number $disk.Number -IsOffline $false }}
Initialize-Disk -Number $disk.Number -PartitionStyle GPT | Out-Null
$partition=New-Partition -DiskNumber $disk.Number -UseMaximumSize -DriveLetter 'M'
Format-Volume -Partition $partition -FileSystem NTFS -NewFileSystemLabel 'VRAMDISK' -Confirm:$false -Force | Out-Null
$volume=Get-Volume -DriveLetter M
[pscustomobject]@{{DiskNumber=$disk.Number; DriveLetter='M:'; FreeBytes=[uint64]$volume.SizeRemaining}} | ConvertTo-Json -Compress"#
    );
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .context("could not run automatic partition and format commands")?;
    anyhow::ensure!(
        output.status.success(),
        "automatic M: setup failed: {}{}",
        String::from_utf8_lossy(&output.stdout).trim(),
        if output.stderr.is_empty() {
            String::new()
        } else {
            format!("\n{}", String::from_utf8_lossy(&output.stderr).trim())
        }
    );
    let volume: serde_json::Value = serde_json::from_slice(&output.stdout)
        .context("automatic M: setup returned invalid status JSON")?;
    anyhow::ensure!(
        volume.get("DriveLetter").and_then(|value| value.as_str()) == Some("M:"),
        "automatic setup did not assign the requested M: drive"
    );
    Ok(())
}

/// Drive letters (`C:`..`Z:`) not currently assigned to any volume. Uses the
/// Win32 GetLogicalDrives bitmask (bit 0 = A:), which — unlike probing the
/// root path — reliably reports assigned-but-inaccessible letters too.
#[cfg(windows)]
pub fn list_free_drives() -> Vec<String> {
    extern "system" {
        fn GetLogicalDrives() -> u32;
    }
    let mask = unsafe { GetLogicalDrives() };
    (b'C'..=b'Z')
        .filter(|&c| mask & (1 << (c - b'A') as u32) == 0)
        .map(|c| format!("{}:", c as char))
        .collect()
}

#[cfg(not(windows))]
pub fn list_free_drives() -> Vec<String> {
    Vec::new()
}

/// True if `s` is exactly a bare drive letter spec ("R", "r:", "R:").
fn is_drive_letter_form(s: &str) -> bool {
    let t = s.trim().trim_end_matches('\\');
    let letters: Vec<char> = t.chars().collect();
    letters.len() == 1 && letters[0].is_ascii_alphabetic()
        || (letters.len() == 2 && letters[0].is_ascii_alphabetic() && letters[1] == ':')
}

/// Normalize a mount point spec: a bare drive letter ("R", "r:") becomes the
/// canonical `X:` form; anything else (a directory path) is passed through
/// unchanged apart from trimming and dropping a trailing backslash.
fn normalize_mount_point(input: &str) -> String {
    let trimmed = input.trim().trim_end_matches('\\');
    if is_drive_letter_form(trimmed) {
        let letter = trimmed.chars().next().unwrap().to_ascii_uppercase();
        return format!("{letter}:");
    }
    trimmed.to_string()
}

/// For a directory mount point, check it is *either* a path that doesn't
/// exist yet (WinFsp will create it fresh as part of mounting — the natural
/// outcome when the user types a brand-new folder name) *or* an existing,
/// empty directory (the natural outcome when the user browses to a folder
/// they made for this; `mount()` removes it right before mounting since
/// WinFsp insists on creating the mount directory itself). Anything else —
/// an existing non-directory, or a non-empty directory — is rejected.
/// Drive-letter mount points are left to WinFsp itself to validate (it
/// already knows which letters are free).
fn validate_mount_point(mount_point: &str, lang: &str) -> anyhow::Result<()> {
    if is_drive_letter_form(mount_point) {
        return Ok(());
    }
    let path = std::path::Path::new(mount_point);
    if !path.exists() {
        return Ok(());
    }
    if !path.is_dir() {
        anyhow::bail!(
            "{}",
            crate::trf(lang, "err_mount_point_not_folder", &[mount_point])
        );
    }
    let mut entries = std::fs::read_dir(path).map_err(|e| {
        anyhow::anyhow!(
            "{}",
            crate::trf(
                lang,
                "err_mount_folder_open_failed",
                &[mount_point, &e.to_string()],
            )
        )
    })?;
    if entries.next().is_some() {
        anyhow::bail!(
            "{}",
            crate::trf(lang, "err_mount_folder_not_empty", &[mount_point])
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_point_normalization() {
        assert_eq!(normalize_mount_point("R"), "R:");
        assert_eq!(normalize_mount_point("r"), "R:");
        assert_eq!(normalize_mount_point("R:"), "R:");
        assert_eq!(normalize_mount_point("r:"), "R:");
        assert_eq!(normalize_mount_point("R:\\"), "R:");
        assert_eq!(normalize_mount_point("  g:  "), "G:");
        // Directory paths pass through unchanged (case preserved), just
        // trimmed of a trailing backslash.
        assert_eq!(normalize_mount_point(r"C:\vramdisk\"), r"C:\vramdisk");
        assert_eq!(normalize_mount_point(r"C:\vramdisk"), r"C:\vramdisk");
    }

    #[test]
    fn drive_letter_form_detection() {
        assert!(is_drive_letter_form("R"));
        assert!(is_drive_letter_form("r:"));
        assert!(is_drive_letter_form("R:\\"));
        assert!(!is_drive_letter_form(r"C:\vramdisk"));
        assert!(!is_drive_letter_form(r"\\server\share"));
    }

    #[cfg(windows)]
    #[test]
    fn free_drives_excludes_assigned_letters() {
        let free = list_free_drives();
        // The drive this repo lives on must be assigned, hence absent.
        let current = std::env::current_dir()
            .ok()
            .and_then(|p| p.components().next().map(|c| c.as_os_str().to_owned()))
            .map(|s| s.to_string_lossy().to_ascii_uppercase());
        if let Some(letter) = current {
            assert!(!free.iter().any(|d| letter.starts_with(d.trim_end_matches(':'))));
        }
    }

    /// Full end-to-end through the Manager: mount a real VRAMDISK, read stats,
    /// round-trip a file through the OS filesystem, then unmount.
    ///
    /// Requires a CUDA GPU and WinFsp, so it is ignored by default:
    ///   cargo test --manifest-path src-tauri/Cargo.toml -- --ignored e2e
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn e2e_mount_roundtrip_unmount() {
        let drive = list_free_drives()
            .into_iter()
            .rev()
            .find(|d| d.as_str() >= "F:")
            .expect("a free drive letter F..Z");
        let mgr = Manager::spawn();

        mgr.mount(
            MountConfig {
                size: Some(128 * 1024 * 1024),
                mount_point: drive.clone(),
                device: 0,
                compress: false,
                dedup: false,
            },
            "ja",
        )
        .expect("mount");

        // Volume should be visible and report a sane total.
        let stats = mgr.stats("ja").expect("stats");
        assert!(stats["volume"]["total_bytes"].as_u64().unwrap() >= 128 * 1024 * 1024);
        assert!(mgr.status().is_some());

        // Round-trip a file through the mounted filesystem.
        let path = format!("{drive}\\hello.txt");
        let payload = b"vramdisk e2e";
        std::fs::write(&path, payload).expect("write to mounted drive");
        let back = std::fs::read(&path).expect("read from mounted drive");
        assert_eq!(back, payload);

        mgr.unmount("ja").expect("unmount");
        assert!(mgr.status().is_none());
        assert!(!std::path::Path::new(&format!("{drive}\\")).exists());

        mgr.shutdown();
    }

    fn scratch_dir_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vramdisk-gui-test-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    /// Same E2E flow but mounting to an *existing, empty* directory (the
    /// natural outcome of the UI's "参照..." folder browser) instead of a
    /// drive letter, proving WinFsp's directory mount-point support works
    /// end-to-end through this manager. WinFsp actually requires the mount
    /// directory to not exist at mount time (it creates it itself), so
    /// `mount()` transparently removes the pre-existing empty directory
    /// right before mounting; this test exercises exactly that path.
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn e2e_mount_to_existing_empty_directory_roundtrip_unmount() {
        let dir = scratch_dir_path("existing");
        std::fs::create_dir_all(&dir).expect("create empty test dir");
        let mount_point = dir.to_string_lossy().to_string();

        let mgr = Manager::spawn();
        mgr.mount(
            MountConfig {
                size: Some(128 * 1024 * 1024),
                mount_point: mount_point.clone(),
                device: 0,
                compress: false,
                dedup: false,
            },
            "ja",
        )
        .expect("mount to pre-existing empty directory");

        let stats = mgr.stats("ja").expect("stats");
        assert!(stats["volume"]["total_bytes"].as_u64().unwrap() >= 128 * 1024 * 1024);

        let path = format!("{mount_point}\\hello.txt");
        let payload = b"vramdisk directory mount e2e";
        std::fs::write(&path, payload).expect("write to mounted directory");
        let back = std::fs::read(&path).expect("read from mounted directory");
        assert_eq!(back, payload);

        mgr.unmount("ja").expect("unmount");
        mgr.shutdown();

        // WinFsp owns the directory's lifecycle in directory-mount mode: it
        // creates it on mount and removes it entirely on unmount.
        assert!(!dir.exists());
    }

    /// Mounting to a path that doesn't exist yet at all (the natural outcome
    /// of the user typing a brand-new folder name) must also work, without
    /// any pre-creation on our part.
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn e2e_mount_to_nonexistent_directory_roundtrip_unmount() {
        let dir = scratch_dir_path("fresh");
        assert!(!dir.exists());
        let mount_point = dir.to_string_lossy().to_string();

        let mgr = Manager::spawn();
        mgr.mount(
            MountConfig {
                size: Some(128 * 1024 * 1024),
                mount_point: mount_point.clone(),
                device: 0,
                compress: false,
                dedup: false,
            },
            "ja",
        )
        .expect("mount to not-yet-existing directory");

        assert!(dir.exists());
        mgr.unmount("ja").expect("unmount");
        mgr.shutdown();
        assert!(!dir.exists());
    }

    /// A mount failure that happens before the directory-removal step (here,
    /// simulated via a bogus device ordinal so `Vram::device_total_mem`
    /// fails) must never touch the user's folder. The harder case — a
    /// failure from `fs::mount` itself *after* we've removed the directory,
    /// which `mount()` handles by recreating it — is exercised by code
    /// review rather than a deterministic test, since forcing WinFsp's own
    /// mount call to fail on demand isn't reliable to set up here.
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn e2e_failed_directory_mount_before_removal_leaves_folder_untouched() {
        let dir = scratch_dir_path("restore");
        std::fs::create_dir_all(&dir).expect("create empty test dir");
        let mount_point = dir.to_string_lossy().to_string();

        let mgr = Manager::spawn();
        let err = mgr
            .mount(
                MountConfig {
                    size: Some(128 * 1024 * 1024),
                    mount_point: mount_point.clone(),
                    device: 9999, // no such CUDA device -> Vram::device_total_mem fails
                    compress: false,
                    dedup: false,
                },
                "ja",
            )
            .expect_err("mount with a bogus device ordinal must fail");
        assert!(!err.is_empty());

        // The failure happened before we ever touched the directory (device
        // lookup fails first), so it must still be there, untouched.
        assert!(dir.exists());
        mgr.shutdown();
        std::fs::remove_dir_all(&dir).ok();
    }
}
