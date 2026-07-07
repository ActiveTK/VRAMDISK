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

use vramdisk::cuda::Vram;
use vramdisk::engine::StorageEngine;
use vramdisk::{default_size, round_up_to_chunk};

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

/// Reply channel for a request.
type Reply<T> = Sender<T>;

enum Cmd {
    ListGpus(Reply<Vec<GpuDto>>),
    ListFreeDrives(Reply<Vec<String>>),
    Mount(MountConfig, Reply<Result<(), String>>),
    Unmount(Reply<Result<(), String>>),
    Status(Reply<Option<MountStatus>>),
    Stats(Reply<Result<serde_json::Value, String>>),
    MountPoint(Reply<Option<String>>),
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
                        Cmd::Mount(cfg, reply) => {
                            let _ = reply.send(state.mount(cfg).map_err(|e| e.to_string()));
                        }
                        Cmd::Unmount(reply) => {
                            let _ = reply.send(state.unmount().map_err(|e| e.to_string()));
                        }
                        Cmd::Status(reply) => {
                            let _ = reply.send(state.status());
                        }
                        Cmd::Stats(reply) => {
                            let _ = reply.send(state.stats().map_err(|e| e.to_string()));
                        }
                        Cmd::MountPoint(reply) => {
                            let _ = reply.send(state.mount_point());
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

    pub fn mount(&self, cfg: MountConfig) -> Result<(), String> {
        self.request(|r| Cmd::Mount(cfg, r))
    }

    pub fn unmount(&self) -> Result<(), String> {
        self.request(Cmd::Unmount)
    }

    /// The current mount, if any.
    pub fn status(&self) -> Option<MountStatus> {
        self.request(Cmd::Status)
    }

    pub fn stats(&self) -> Result<serde_json::Value, String> {
        self.request(Cmd::Stats)
    }

    /// The mount point (drive letter or directory) of the current mount, if
    /// any. Used by job commands to address the mounted volume without
    /// holding the manager thread for the duration of a (possibly slow) GPU
    /// job.
    pub fn mount_point(&self) -> Option<String> {
        self.request(Cmd::MountPoint)
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

    #[cfg(windows)]
    fn mount(&mut self, cfg: MountConfig) -> anyhow::Result<()> {
        if self.mount.is_some() {
            anyhow::bail!("a disk is already mounted; unmount it first");
        }
        let mount_point = normalize_mount_point(&cfg.mount_point);
        if mount_point.is_empty() {
            anyhow::bail!("マウント先を指定してください");
        }
        let directory_mode = !is_drive_letter_form(&mount_point);
        if directory_mode {
            validate_mount_point(&mount_point)?;
        }

        let total = Vram::device_total_mem(cfg.device)?;
        let size = match cfg.size {
            Some(s) => round_up_to_chunk(s),
            None => default_size(total),
        };
        if size == 0 {
            anyhow::bail!("disk size must be greater than 0");
        }
        if size > total {
            anyhow::bail!("requested {size} bytes exceeds device VRAM ({total} bytes)");
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
                anyhow::anyhow!("マウント先フォルダを準備できません: {mount_point}: {e}")
            })?;
        }

        let mounted = match vramdisk::fs::mount(engine, &mount_point, "VRAMDISK") {
            Ok(m) => m,
            Err(e) => {
                if removed_existing {
                    let _ = std::fs::create_dir(&mount_point);
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
    fn mount(&mut self, _cfg: MountConfig) -> anyhow::Result<()> {
        anyhow::bail!("mounting is only supported on Windows (WinFsp)")
    }

    fn unmount(&mut self) -> anyhow::Result<()> {
        match self.mount.take() {
            Some(_record) => {
                #[cfg(windows)]
                _record.mounted.unmount();
                Ok(())
            }
            None => anyhow::bail!("nothing is mounted"),
        }
    }

    /// Read the mounted volume's `$VRAMDISK\stats.json` and return it parsed.
    fn stats(&self) -> anyhow::Result<serde_json::Value> {
        let Some(record) = self.mount.as_ref() else {
            anyhow::bail!("nothing is mounted");
        };
        let path = format!("{}\\$VRAMDISK\\stats.json", record.status.mount_point);
        let text =
            std::fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("read {path}: {e}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("parse {path}: {e}"))?;
        Ok(value)
    }

    fn shutdown(&mut self) {
        if let Some(_record) = self.mount.take() {
            #[cfg(windows)]
            _record.mounted.unmount();
        }
    }
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
fn validate_mount_point(mount_point: &str) -> anyhow::Result<()> {
    if is_drive_letter_form(mount_point) {
        return Ok(());
    }
    let path = std::path::Path::new(mount_point);
    if !path.exists() {
        return Ok(());
    }
    if !path.is_dir() {
        anyhow::bail!("マウント先はフォルダではありません: {mount_point}");
    }
    let mut entries = std::fs::read_dir(path)
        .map_err(|e| anyhow::anyhow!("マウント先フォルダを開けません: {mount_point}: {e}"))?;
    if entries.next().is_some() {
        anyhow::bail!("マウント先フォルダが空ではありません: {mount_point}");
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

        mgr.mount(MountConfig {
            size: Some(128 * 1024 * 1024),
            mount_point: drive.clone(),
            device: 0,
            compress: false,
            dedup: false,
        })
        .expect("mount");

        // Volume should be visible and report a sane total.
        let stats = mgr.stats().expect("stats");
        assert!(stats["volume"]["total_bytes"].as_u64().unwrap() >= 128 * 1024 * 1024);
        assert!(mgr.status().is_some());

        // Round-trip a file through the mounted filesystem.
        let path = format!("{drive}\\hello.txt");
        let payload = b"vramdisk e2e";
        std::fs::write(&path, payload).expect("write to mounted drive");
        let back = std::fs::read(&path).expect("read from mounted drive");
        assert_eq!(back, payload);

        mgr.unmount().expect("unmount");
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
        mgr.mount(MountConfig {
            size: Some(128 * 1024 * 1024),
            mount_point: mount_point.clone(),
            device: 0,
            compress: false,
            dedup: false,
        })
        .expect("mount to pre-existing empty directory");

        let stats = mgr.stats().expect("stats");
        assert!(stats["volume"]["total_bytes"].as_u64().unwrap() >= 128 * 1024 * 1024);

        let path = format!("{mount_point}\\hello.txt");
        let payload = b"vramdisk directory mount e2e";
        std::fs::write(&path, payload).expect("write to mounted directory");
        let back = std::fs::read(&path).expect("read from mounted directory");
        assert_eq!(back, payload);

        mgr.unmount().expect("unmount");
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
        mgr.mount(MountConfig {
            size: Some(128 * 1024 * 1024),
            mount_point: mount_point.clone(),
            device: 0,
            compress: false,
            dedup: false,
        })
        .expect("mount to not-yet-existing directory");

        assert!(dir.exists());
        mgr.unmount().expect("unmount");
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
            .mount(MountConfig {
                size: Some(128 * 1024 * 1024),
                mount_point: mount_point.clone(),
                device: 9999, // no such CUDA device -> Vram::device_total_mem fails
                compress: false,
                dedup: false,
            })
            .expect_err("mount with a bogus device ordinal must fail");
        assert!(!err.is_empty());

        // The failure happened before we ever touched the directory (device
        // lookup fails first), so it must still be there, untouched.
        assert!(dir.exists());
        mgr.shutdown();
        std::fs::remove_dir_all(&dir).ok();
    }
}
