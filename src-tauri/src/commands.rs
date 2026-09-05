//! Tauri command surface. Each command is a thin adapter over the mount
//! [`Manager`], converting errors to strings for the JS bridge.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter as _, Manager as _, State};
use tauri_plugin_dialog::DialogExt;

use crate::manager::{GpuDto, Manager, MountConfig, MountStatus};
use crate::{tr, trf, CliSeed, ExportControl, UiLang};

/// The UI language a command's messages must be rendered in.
///
/// Commands that do not otherwise need an `AppHandle` take `State<UiLang>`
/// (Tauri injects it like any other state) and read it through here, so no
/// command has to grow an `AppHandle` parameter just to translate an error.
fn lang_of(lang: &State<UiLang>) -> String {
    lang.0.lock().unwrap().clone()
}

/// Current UI language ("ja" or "en"), resolved at startup from the registry
/// or the Windows display language (see `main::detect_language`).
#[tauri::command]
pub fn get_ui_language(lang: State<UiLang>) -> String {
    lang.0.lock().unwrap().clone()
}

/// Persist the user's language choice to HKCU\Software\VRAMDISK and rebuild
/// the tray menu so it switches language immediately.
#[tauri::command]
pub fn set_ui_language(
    language: String,
    app: AppHandle,
    lang: State<UiLang>,
    manager: State<Manager>,
) -> Result<(), String> {
    if language != "ja" && language != "en" {
        // Reported in the language still in effect — the rejected one is by
        // definition not one we have strings for.
        return Err(trf(
            &lang_of(&lang),
            "err_unsupported_language",
            &[&language],
        ));
    }
    crate::persist_language(&language)?;
    *lang.0.lock().unwrap() = language;
    crate::refresh_tray_menu(&app, manager.status().is_some());
    Ok(())
}

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
pub fn stats(manager: State<Manager>, lang: State<UiLang>) -> Result<serde_json::Value, String> {
    manager.stats(&lang_of(&lang))
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

/// Open a native "save file" dialog for the teardown ZIP export.
///
/// Deliberately *not* `browse_save`: that one roots the dialog inside the
/// mounted volume, which is exactly the wrong default here — the archive
/// exists to survive the volume, so writing it onto the disk we are about to
/// destroy would be self-defeating (and `export_zip` rejects it anyway). This
/// starts in Documents instead and pre-fills a timestamped name.
#[tauri::command]
pub fn browse_export_zip(app: AppHandle) -> Option<String> {
    let now = time::OffsetDateTime::now_local().unwrap_or_else(|_| time::OffsetDateTime::now_utc());
    let suggested = format!(
        "VRAMDISK-{:04}{:02}{:02}-{:02}{:02}{:02}.zip",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    );
    let mut dlg = app.dialog().file().add_filter("ZIP", &["zip"]);
    if let Ok(dir) = app.path().document_dir() {
        dlg = dlg.set_directory(dir);
    }
    dlg.set_file_name(suggested)
        .blocking_save_file()
        .and_then(|fp| fp.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

/// Mount the single VRAMDISK. On success: open the mount point in Explorer,
/// hide the main window (the tray keeps the mount alive and reachable), and
/// show a one-time confirmation.
#[tauri::command]
pub fn mount(cfg: MountConfig, app: AppHandle, manager: State<Manager>) -> Result<(), String> {
    let lang = crate::app_language(&app);
    if cfg.compress && !vramdisk::nvcomp::nvcomp_available() {
        return Err(tr(&lang, "err_nvcomp_unavailable").to_string());
    }
    let mount_point = cfg.mount_point.clone();
    manager.mount(cfg, &lang)?;
    crate::on_mount_state_changed(&app, &manager);
    crate::after_mount_success(&app, &mount_point);
    Ok(())
}

#[tauri::command]
pub fn unmount(app: AppHandle, manager: State<Manager>) -> Result<(), String> {
    manager.unmount(&crate::app_language(&app))?;
    crate::on_mount_state_changed(&app, &manager);
    Ok(())
}

/// Quit the whole app (unmounting first), from the window UI.
///
/// The tray's "終了" while mounted hands the three-way teardown choice to the
/// window (a native dialog can only offer two buttons), so the window needs a
/// way to finish the exit the tray started.
#[tauri::command]
pub fn quit_app(app: AppHandle) {
    crate::quit(&app);
}

/// Submit a GPU batch-hash job over paths on the mounted volume via its
/// `$VRAMDISK\jobs` API. Returns the job id immediately; the frontend polls
/// `job_status` and reads `job_result` when the job turns terminal, so a long
/// hash never blocks an invoke round-trip and stays cancellable.
#[tauri::command]
pub fn hash_job(
    paths: Vec<String>,
    algorithm: String,
    recursive: bool,
    manager: State<Manager>,
    lang: State<UiLang>,
) -> Result<String, String> {
    let lang = lang_of(&lang);
    if paths.is_empty() {
        return Err(tr(&lang, "err_no_paths").to_string());
    }
    let mount_point = mounted_point(&manager, &lang)?;
    let norm = paths
        .iter()
        .map(|p| normalize_path(&mount_point, p, &lang))
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor = serde_json::json!({
        "op": "hash",
        "algorithm": algorithm,
        "paths": norm,
        "recursive": recursive,
    });
    submit_job_async(&mount_point, descriptor, &lang)
}

/// The mount point of the live mount, or the localized "nothing is mounted"
/// error every job command needs to return when there isn't one.
fn mounted_point(manager: &State<Manager>, lang: &str) -> Result<String, String> {
    manager
        .mount_point()
        .ok_or_else(|| tr(lang, "err_not_mounted").to_string())
}

/// Read and parse one of a job's JSON documents from `$VRAMDISK\jobs\<id>`.
/// Shared by `job_status` and `job_result`, which differ only in the filename.
fn read_job_json(
    manager: &State<Manager>,
    job_id: &str,
    file: &str,
    lang: &str,
) -> Result<serde_json::Value, String> {
    let mount_point = mounted_point(manager, lang)?;
    validate_job_id(job_id, lang)?;
    let path = format!("{mount_point}\\$VRAMDISK\\jobs\\{job_id}\\{file}");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| trf(lang, "err_read_failed", &[&path, &e.to_string()]))?;
    serde_json::from_str(&text).map_err(|e| trf(lang, "err_parse_failed", &[&path, &e.to_string()]))
}

/// Poll a submitted job's `status.json`. Returns the parsed document
/// (`state`, `terminal`, ...).
#[tauri::command]
pub fn job_status(
    job_id: String,
    manager: State<Manager>,
    lang: State<UiLang>,
) -> Result<serde_json::Value, String> {
    read_job_json(&manager, &job_id, "status.json", &lang_of(&lang))
}

/// Read a terminal job's `result.json`.
#[tauri::command]
pub fn job_result(
    job_id: String,
    manager: State<Manager>,
    lang: State<UiLang>,
) -> Result<serde_json::Value, String> {
    read_job_json(&manager, &job_id, "result.json", &lang_of(&lang))
}

/// Request cancellation of a running job (reading the virtual `cancel` file
/// performs the cancellation).
#[tauri::command]
pub fn job_cancel(
    job_id: String,
    manager: State<Manager>,
    lang: State<UiLang>,
) -> Result<(), String> {
    let lang = lang_of(&lang);
    let mount_point = mounted_point(&manager, &lang)?;
    validate_job_id(&job_id, &lang)?;
    let path = format!("{mount_point}\\$VRAMDISK\\jobs\\{job_id}\\cancel");
    std::fs::read(&path)
        .map(|_| ())
        .map_err(|e| trf(&lang, "err_job_cancel_failed", &[&path, &e.to_string()]))
}

/// Reject anything that could escape `$VRAMDISK\jobs\<id>` before the id is
/// spliced into a filesystem path. Mirrors the volume-side job-id rules.
fn validate_job_id(id: &str, lang: &str) -> Result<(), String> {
    let ok = !id.is_empty()
        && id.len() <= 128
        && id != "."
        && id != ".."
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
    if ok {
        Ok(())
    } else {
        Err(trf(lang, "err_invalid_job_id", &[id]))
    }
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

/// Submit a GPU archive compression job (`tar.zst` / `tar.lz4` / `tar.gz` /
/// `zip`) over paths on the mounted volume, via `$VRAMDISK\jobs`. Always
/// recursive: a non-recursive compress isn't a meaningful option for this
/// tool. Returns the job id (see `hash_job` for the poll/cancel flow).
#[tauri::command]
pub fn archive_compress_job(
    req: ArchiveCompressRequest,
    manager: State<Manager>,
    lang: State<UiLang>,
) -> Result<String, String> {
    let lang = lang_of(&lang);
    if req.paths.is_empty() {
        return Err(tr(&lang, "err_no_paths").to_string());
    }
    let mount_point = mounted_point(&manager, &lang)?;
    let norm = req
        .paths
        .iter()
        .map(|p| normalize_path(&mount_point, p, &lang))
        .collect::<Result<Vec<_>, _>>()?;
    let output = normalize_path(&mount_point, &req.output, &lang)?;
    let descriptor = serde_json::json!({
        "op": "archive.compress",
        "format": req.format,
        "paths": norm,
        "output": output,
        "recursive": true,
    });
    submit_job_async(&mount_point, descriptor, &lang)
}

/// Submit a GPU archive extraction job over an archive file on the mounted
/// volume. Returns the job id (see `hash_job` for the poll/cancel flow).
#[tauri::command]
pub fn archive_extract_job(
    req: ArchiveExtractRequest,
    manager: State<Manager>,
    lang: State<UiLang>,
) -> Result<String, String> {
    let lang = lang_of(&lang);
    let mount_point = mounted_point(&manager, &lang)?;
    let archive = normalize_path(&mount_point, &req.archive, &lang)?;
    let output_dir = normalize_path(&mount_point, &req.output_dir, &lang)?;
    let descriptor = serde_json::json!({
        "op": "archive.extract",
        "format": req.format,
        "archive": archive,
        "output_dir": output_dir,
    });
    submit_job_async(&mount_point, descriptor, &lang)
}

/// Submit a GPU encode/decode job (Base64 / hex) over one file on the mounted
/// volume. Returns the job id (see `hash_job` for the poll/cancel flow).
#[tauri::command]
pub fn encode_job(
    req: EncodeRequest,
    manager: State<Manager>,
    lang: State<UiLang>,
) -> Result<String, String> {
    let lang = lang_of(&lang);
    let mount_point = mounted_point(&manager, &lang)?;
    let input = normalize_path(&mount_point, &req.input, &lang)?;
    let output = normalize_path(&mount_point, &req.output, &lang)?;
    let descriptor = serde_json::json!({
        "op": "encode",
        "codec": req.codec,
        "direction": req.direction,
        "input": input,
        "output": output,
    });
    submit_job_async(&mount_point, descriptor, &lang)
}

#[derive(Debug, Deserialize)]
pub struct EncodeRequest {
    /// "base64" or "hex".
    pub codec: String,
    /// "encode" or "decode".
    pub direction: String,
    pub input: String,
    pub output: String,
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
fn normalize_path(mount_point: &str, p: &str, lang: &str) -> Result<String, String> {
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
        return Err(trf(lang, "err_path_not_on_volume", &[p, mount_point]));
    }

    Ok(if input.starts_with('\\') {
        input
    } else {
        format!("\\{input}")
    })
}

/// Submit a job descriptor to the volume's `$VRAMDISK\jobs` API and return
/// the generated job id immediately. The frontend drives the rest through
/// `job_status` / `job_result` / `job_cancel`, so an hours-long archive job
/// neither blocks an invoke round-trip nor becomes uncancellable.
fn submit_job_async(
    mount_point: &str,
    descriptor: serde_json::Value,
    lang: &str,
) -> Result<String, String> {
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    let job_id = format!(
        "gui{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    // Submit: CREATE_NEW the pending descriptor, write, close (closing the
    // handle is what queues the job).
    let pending = format!("{mount_point}\\$VRAMDISK\\jobs\\pending\\{job_id}.json");
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)
            .map_err(|e| trf(lang, "err_job_submit_failed", &[&pending, &e.to_string()]))?;
        f.write_all(descriptor.to_string().as_bytes())
            .map_err(|e| trf(lang, "err_job_submit_failed", &[&pending, &e.to_string()]))?;
    }
    Ok(job_id)
}

// --- ZIP export (rescue the volume before it is torn down) -------------------
//
// VRAMDISK is volatile: unmounting or exiting throws the data away. This is
// the escape hatch offered at teardown time — the whole volume, streamed into
// an ordinary ZIP on the host filesystem.
//
// Why not the engine's existing GPU ZIP writer (`archive.compress` jobs)? That
// one writes its output *onto the VRAM disk*, so it needs free VRAM roughly
// the size of the archive. An export whose entire purpose is "don't lose my
// data" must not fail because the disk it is rescuing is nearly full — which
// is precisely when people reach for it. Reading the volume through ordinary
// Win32 file APIs (the process can open its own mounted volume; `job_status`
// already does this for `$VRAMDISK\jobs\...`) and streaming straight into a
// host-side file needs no VRAM at all, and cannot be defeated by a full disk.

/// One entry that could not be archived. Collected rather than fatal: a single
/// locked or vanished file must not throw away the rest of the rescue.
#[derive(Debug, Serialize)]
pub struct ExportFailure {
    /// Path relative to the mount root, as it would have appeared in the ZIP.
    pub path: String,
    pub error: String,
}

/// Outcome of a completed (or cancelled) export.
#[derive(Debug, Serialize)]
pub struct ExportReport {
    pub destination: String,
    pub file_count: u64,
    pub dir_count: u64,
    /// Uncompressed bytes actually read out of the volume.
    pub bytes_written: u64,
    /// Size of the finished ZIP on the host filesystem.
    pub archive_bytes: u64,
    pub cancelled: bool,
    pub failures: Vec<ExportFailure>,
}

/// One filesystem entry found under the mount root, in the shape the ZIP
/// writer wants it.
struct ExportEntry {
    abs: PathBuf,
    /// Path relative to the mount root, `/`-separated (the ZIP convention).
    rel: String,
    is_dir: bool,
    len: u64,
    modified: Option<SystemTime>,
}

/// Read granularity for a single file. Big enough that the syscall overhead
/// disappears against a RAM-speed volume, small enough that a multi-gigabyte
/// file still moves the progress bar and stays cancellable.
const EXPORT_CHUNK: usize = 1024 * 1024;

/// Minimum gap between progress events, so a volume full of tiny files does
/// not drown the webview in IPC messages.
const EXPORT_EMIT_INTERVAL_MS: u128 = 80;

/// Write the entire mounted volume to `destination` as a ZIP file on the host
/// filesystem, reporting progress through `EVENT_EXPORT_PROGRESS`.
///
/// `#[tauri::command(async)]` (rather than a plain command) because this can
/// run for minutes: a synchronous command would block the main thread and
/// freeze the very progress bar it is feeding.
#[tauri::command(async)]
pub fn export_zip(destination: String, app: AppHandle) -> Result<ExportReport, String> {
    let lang = crate::app_language(&app);
    let mount_point = app
        .state::<Manager>()
        .mount_point()
        .ok_or_else(|| tr(&lang, "err_not_mounted").to_string())?;

    let destination = destination.trim().to_string();
    if destination.is_empty() {
        return Err(tr(&lang, "err_export_destination_required").to_string());
    }
    // Writing the rescue archive onto the volume being rescued would both eat
    // the VRAM we are trying to drain and vanish on unmount.
    if is_under_mount(&mount_point, &destination) {
        return Err(trf(
            &lang,
            "err_export_destination_on_volume",
            &[&destination],
        ));
    }

    // One export at a time: two concurrent walks would interleave their
    // progress events into one nonsensical bar.
    let cancel = {
        let control = app.state::<ExportControl>();
        if control
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(tr(&lang, "err_export_already_running").to_string());
        }
        control.cancel.store(false, Ordering::SeqCst);
        control.cancel.clone()
    };

    let result = run_export(
        &mount_point,
        &destination,
        &lang,
        &cancel,
        &mut |done, total| emit_export_progress(&app, done, total),
    );
    app.state::<ExportControl>()
        .running
        .store(false, Ordering::SeqCst);
    result
}

/// Ask the in-flight export to stop. Harmless when nothing is running.
#[tauri::command]
pub fn export_cancel(app: AppHandle) {
    app.state::<ExportControl>()
        .cancel
        .store(true, Ordering::SeqCst);
}

/// The export itself. Takes a plain progress sink rather than an `AppHandle`
/// so the walk-and-archive logic can be tested against an ordinary directory,
/// with no Tauri app and no mounted volume in sight.
fn run_export(
    mount_point: &str,
    destination: &str,
    lang: &str,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<ExportReport, String> {
    let root = PathBuf::from(format!("{}\\", mount_point.trim_end_matches('\\')));

    // Pre-pass: total the bytes so the bar is determinate rather than a
    // spinner. On a RAM-speed volume this walk costs milliseconds.
    let mut total_bytes = 0u64;
    // Anything unreadable here is reported by the writing pass below, so these
    // are dropped rather than reported twice.
    let mut scan_failures = Vec::new();
    walk_volume(
        &root,
        "",
        0,
        lang,
        cancel,
        &mut scan_failures,
        &mut |entry| {
            if !entry.is_dir {
                total_bytes = total_bytes.saturating_add(entry.len);
            }
            Ok(None)
        },
    )?;
    on_progress(0, total_bytes);

    let file = std::fs::File::create(destination).map_err(|e| {
        trf(
            lang,
            "err_export_create_failed",
            &[destination, &e.to_string()],
        )
    })?;
    let mut writer = zip::ZipWriter::new(std::io::BufWriter::new(file));

    let mut failures = Vec::new();
    let mut file_count = 0u64;
    let mut dir_count = 0u64;
    let mut done_bytes = 0u64;
    let mut buf = vec![0u8; EXPORT_CHUNK];
    let started = Instant::now();
    let mut last_emit = 0u128;

    let write_result = walk_volume(&root, "", 0, lang, cancel, &mut failures, &mut |entry| {
        if entry.is_dir {
            writer
                .add_directory(entry.rel.as_str(), zip_options(entry.modified, 0))
                .map_err(|e| {
                    trf(
                        lang,
                        "err_export_add_dir_failed",
                        &[&entry.rel, &e.to_string()],
                    )
                })?;
            dir_count += 1;
            return Ok(None);
        }

        writer
            .start_file(entry.rel.as_str(), zip_options(entry.modified, entry.len))
            .map_err(|e| {
                trf(
                    lang,
                    "err_export_add_file_failed",
                    &[&entry.rel, &e.to_string()],
                )
            })?;
        let mut src = match std::fs::File::open(&entry.abs) {
            Ok(f) => f,
            Err(e) => {
                let _ = writer.abort_file();
                return Ok(Some(ExportFailure {
                    path: entry.rel.clone(),
                    error: e.to_string(),
                }));
            }
        };
        loop {
            if cancel.load(Ordering::Relaxed) {
                // Drop the half-written entry rather than leaving a truncated
                // file that looks complete to an unzip tool.
                let _ = writer.abort_file();
                return Ok(None);
            }
            let n = match src.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    let _ = writer.abort_file();
                    return Ok(Some(ExportFailure {
                        path: entry.rel.clone(),
                        error: e.to_string(),
                    }));
                }
            };
            writer.write_all(&buf[..n]).map_err(|e| {
                trf(
                    lang,
                    "err_export_write_failed",
                    &[&entry.rel, &e.to_string()],
                )
            })?;
            done_bytes = done_bytes.saturating_add(n as u64);
            let elapsed = started.elapsed().as_millis();
            if elapsed.saturating_sub(last_emit) >= EXPORT_EMIT_INTERVAL_MS {
                last_emit = elapsed;
                on_progress(done_bytes, total_bytes);
            }
        }
        file_count += 1;
        Ok(None)
    });

    if let Err(e) = write_result {
        // The archive is unusable; leaving a plausible-looking .zip behind
        // would be worse than leaving nothing.
        drop(writer);
        let _ = std::fs::remove_file(destination);
        return Err(e);
    }

    // Even a cancelled export is finished properly: what was copied so far is
    // then a valid archive the user can actually open, instead of a truncated
    // file. The caller still sees `cancelled` and must not unmount.
    let mut inner = writer.finish().map_err(|e| {
        trf(
            lang,
            "err_export_finish_failed",
            &[destination, &e.to_string()],
        )
    })?;
    inner.flush().map_err(|e| {
        trf(
            lang,
            "err_export_flush_failed",
            &[destination, &e.to_string()],
        )
    })?;
    drop(inner);

    let archive_bytes = std::fs::metadata(destination).map(|m| m.len()).unwrap_or(0);
    on_progress(done_bytes, total_bytes);

    Ok(ExportReport {
        destination: destination.to_string(),
        file_count,
        dir_count,
        bytes_written: done_bytes,
        archive_bytes,
        cancelled: cancel.load(Ordering::SeqCst),
        failures,
    })
}

/// ZIP entry options: Deflate (universally readable), the source file's mtime
/// where we could read it, and ZIP64 only for entries that actually need it.
fn zip_options(modified: Option<SystemTime>, len: u64) -> zip::write::SimpleFileOptions {
    let mut opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .large_file(len > u32::MAX as u64);
    if let Some(dt) = modified.and_then(zip_datetime) {
        opts = opts.last_modified_time(dt);
    }
    opts
}

/// A ZIP entry's timestamp is an MS-DOS date/time with no timezone, which by
/// convention means *local* time — storing UTC would show every file shifted
/// by the local offset in Explorer. Falls back to UTC if the OS timezone
/// cannot be read, which is still better than the 1980-01-01 default.
fn zip_datetime(t: SystemTime) -> Option<zip::DateTime> {
    let utc = time::OffsetDateTime::from(t);
    let local = time::UtcOffset::local_offset_at(utc)
        .map(|off| utc.to_offset(off))
        .unwrap_or(utc);
    zip::DateTime::try_from(time::PrimitiveDateTime::new(local.date(), local.time())).ok()
}

fn emit_export_progress(app: &AppHandle, done_bytes: u64, total_bytes: u64) {
    // Same shape as the job system's `status.json` progress object (see
    // `vramdisk::jobs::JobProgress`), so the frontend's existing progress-bar
    // helpers render it without a second code path.
    let _ = app.emit(
        crate::EVENT_EXPORT_PROGRESS,
        serde_json::json!({ "done_bytes": done_bytes, "total_bytes": total_bytes }),
    );
}

/// Depth-first walk of the mounted volume, handing every directory and regular
/// file to `visit` (directories before their contents, so the ZIP contains
/// empty directories too).
///
/// I/O errors are per-entry, never fatal: an unreadable directory or a file
/// that vanished mid-walk is recorded in `failures` and the walk continues.
/// `visit` returns `Ok(Some(failure))` to record one of its own, and `Err`
/// only for a failure of the *archive* (which really must abort everything).
fn walk_volume(
    dir: &Path,
    rel_prefix: &str,
    depth: usize,
    lang: &str,
    cancel: &AtomicBool,
    failures: &mut Vec<ExportFailure>,
    visit: &mut dyn FnMut(&ExportEntry) -> Result<Option<ExportFailure>, String>,
) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        return Ok(());
    }
    let iter = match std::fs::read_dir(dir) {
        Ok(iter) => iter,
        Err(e) => {
            failures.push(ExportFailure {
                path: rel_display(rel_prefix),
                error: e.to_string(),
            });
            return Ok(());
        }
    };

    for item in iter {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        let item = match item {
            Ok(item) => item,
            Err(e) => {
                failures.push(ExportFailure {
                    path: rel_display(rel_prefix),
                    error: e.to_string(),
                });
                continue;
            }
        };
        let name = item.file_name().to_string_lossy().to_string();

        // `$VRAMDISK` is the volume's synthetic control directory, and it
        // exists only at the root (see `vramdisk::internal_api`, and the
        // root-only branch in the WinFsp ReadDirectory handler). It holds no
        // user data and must never be walked into: `jobs\<id>\wait` blocks
        // until the job finishes and `chunks.json\<path>` is an unbounded
        // synthetic namespace, so an ordinary recursive copy would hang or
        // never terminate.
        if depth == 0 && name.eq_ignore_ascii_case(vramdisk::internal_api::DISPLAY_ROOT) {
            continue;
        }

        let rel = if rel_prefix.is_empty() {
            name
        } else {
            format!("{rel_prefix}/{name}")
        };
        let file_type = match item.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                failures.push(ExportFailure {
                    path: rel,
                    error: e.to_string(),
                });
                continue;
            }
        };
        let abs = item.path();

        if file_type.is_dir() {
            let entry = ExportEntry {
                abs: abs.clone(),
                rel: rel.clone(),
                is_dir: true,
                len: 0,
                modified: item.metadata().ok().and_then(|m| m.modified().ok()),
            };
            if let Some(failure) = visit(&entry)? {
                failures.push(failure);
                continue;
            }
            walk_volume(&abs, &rel, depth + 1, lang, cancel, failures, visit)?;
        } else if file_type.is_file() {
            let meta = match item.metadata() {
                Ok(meta) => meta,
                Err(e) => {
                    failures.push(ExportFailure {
                        path: rel,
                        error: e.to_string(),
                    });
                    continue;
                }
            };
            let entry = ExportEntry {
                abs,
                rel,
                is_dir: false,
                len: meta.len(),
                modified: meta.modified().ok(),
            };
            if let Some(failure) = visit(&entry)? {
                failures.push(failure);
            }
        } else {
            // Reparse points and the like: `file_type()` does not follow them,
            // so copying one would need semantics this rescue archive has no
            // way to express. Report it rather than pretending it was saved.
            failures.push(ExportFailure {
                path: rel,
                error: tr(lang, "err_export_unsupported_entry").to_string(),
            });
        }
    }
    Ok(())
}

/// How a relative path is named in a failure report; the mount root itself has
/// an empty relative path.
fn rel_display(rel: &str) -> String {
    if rel.is_empty() {
        "\\".to_string()
    } else {
        rel.to_string()
    }
}

/// Whether `path` points somewhere inside `mount_point`. Compared
/// case-insensitively on the mount-point prefix only, which is enough here:
/// this is a "don't shoot yourself in the foot" guard, not a security
/// boundary.
fn is_under_mount(mount_point: &str, path: &str) -> bool {
    let mount = mount_point.trim_end_matches('\\');
    let candidate = path.replace('/', "\\");
    let Some(prefix) = candidate.get(..mount.len()) else {
        return false;
    };
    if !prefix.eq_ignore_ascii_case(mount) {
        return false;
    }
    matches!(candidate.as_bytes().get(mount.len()), None | Some(b'\\'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_path_relative_unchanged() {
        assert_eq!(normalize_path("R:", "\\data", "ja").unwrap(), "\\data");
        assert_eq!(normalize_path("R:", "data", "ja").unwrap(), "\\data");
        assert_eq!(
            normalize_path(r"C:\vramdisk", "\\data", "ja").unwrap(),
            "\\data"
        );
    }

    #[test]
    fn normalize_path_absolute_on_drive_letter_mount() {
        assert_eq!(normalize_path("R:", "R:\\data", "ja").unwrap(), "\\data");
        assert_eq!(normalize_path("R:", "r:data", "ja").unwrap(), "\\data");
        assert_eq!(normalize_path("R:", "R:", "ja").unwrap(), "\\");
        assert!(normalize_path("R:", "C:\\Windows", "ja").is_err());
    }

    #[test]
    fn export_destination_on_the_volume_is_detected() {
        assert!(is_under_mount("R:", r"R:\rescue.zip"));
        assert!(is_under_mount("R:", r"r:\sub\rescue.zip"));
        assert!(is_under_mount("R:", "R:"));
        assert!(is_under_mount(r"C:\vramdisk", r"C:\VRAMDISK\rescue.zip"));
        // Forward slashes are the same path to Windows.
        assert!(is_under_mount("R:", "R:/rescue.zip"));

        assert!(!is_under_mount("R:", r"C:\Users\me\rescue.zip"));
        assert!(!is_under_mount(r"C:\vramdisk", r"C:\vramdisk-backup\rescue.zip"));
        assert!(!is_under_mount(r"C:\vramdisk", r"D:\vramdisk\rescue.zip"));
        assert!(!is_under_mount("R:", ""));
    }

    fn export_scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vramdisk-export-test-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The archiving half of the export, exercised against an ordinary
    /// directory: a real mount needs a GPU and WinFsp, but the walk, the
    /// `$VRAMDISK` exclusion and the ZIP layout are plain filesystem logic.
    #[test]
    fn export_writes_the_whole_tree_and_skips_the_control_directory() {
        let root = export_scratch_dir("tree");
        std::fs::write(root.join("a.txt"), b"hello").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub").join("b.bin"), vec![7u8; 4096]).unwrap();
        std::fs::create_dir(root.join("empty")).unwrap();
        // The synthetic control directory lives only at the volume root and
        // must never be descended into...
        std::fs::create_dir(root.join("$VRAMDISK")).unwrap();
        std::fs::write(root.join("$VRAMDISK").join("stats.json"), b"{}").unwrap();
        // ...whereas a user directory that merely shares the name deeper in the
        // tree is ordinary data and must be archived.
        std::fs::create_dir(root.join("sub").join("$VRAMDISK")).unwrap();
        std::fs::write(root.join("sub").join("$VRAMDISK").join("mine.txt"), b"keep").unwrap();

        let zip_path = export_scratch_dir("out").join("rescue.zip");
        let cancel = AtomicBool::new(false);
        let mut seen_progress = Vec::new();
        let report = run_export(
            &root.to_string_lossy(),
            &zip_path.to_string_lossy(),
            "ja",
            &cancel,
            &mut |done, total| seen_progress.push((done, total)),
        )
        .expect("export");

        assert!(!report.cancelled);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert_eq!(report.file_count, 3);
        assert_eq!(report.dir_count, 3); // sub, empty, sub/$VRAMDISK
        assert_eq!(report.bytes_written, 5 + 4096 + 4);
        assert!(report.archive_bytes > 0);
        // The pre-pass makes the bar determinate from the very first event.
        assert_eq!(seen_progress.first().unwrap().1, 5 + 4096 + 4);

        let mut archive =
            zip::ZipArchive::new(std::fs::File::open(&zip_path).unwrap()).expect("read back");
        let mut names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "a.txt",
                "empty/",
                "sub/",
                "sub/$VRAMDISK/",
                "sub/$VRAMDISK/mine.txt",
                "sub/b.bin",
            ]
        );
        let mut body = String::new();
        archive
            .by_name("a.txt")
            .unwrap()
            .read_to_string(&mut body)
            .unwrap();
        assert_eq!(body, "hello");

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(zip_path.parent().unwrap()).ok();
    }

    /// A pre-set cancel flag must stop before any entry is archived, and still
    /// leave a well-formed (if empty) ZIP behind rather than a truncated file.
    #[test]
    fn export_honours_cancellation() {
        let root = export_scratch_dir("cancel");
        std::fs::write(root.join("a.txt"), b"hello").unwrap();

        let zip_path = export_scratch_dir("cancel-out").join("rescue.zip");
        let cancel = AtomicBool::new(true);
        let report = run_export(
            &root.to_string_lossy(),
            &zip_path.to_string_lossy(),
            "ja",
            &cancel,
            &mut |_, _| {},
        )
        .expect("export");

        assert!(report.cancelled);
        assert_eq!(report.file_count, 0);
        assert!(zip::ZipArchive::new(std::fs::File::open(&zip_path).unwrap()).is_ok());

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(zip_path.parent().unwrap()).ok();
    }

    #[test]
    fn normalize_path_absolute_on_directory_mount() {
        let mount = r"C:\vramdisk";
        assert_eq!(
            normalize_path(mount, r"C:\vramdisk\data", "ja").unwrap(),
            "\\data"
        );
        assert_eq!(
            normalize_path(mount, r"c:\VRAMDISK\data\a.txt", "ja").unwrap(),
            "\\data\\a.txt"
        );
        assert_eq!(normalize_path(mount, mount, "ja").unwrap(), "\\");
        assert!(normalize_path(mount, r"C:\other\data", "ja").is_err());
        assert!(normalize_path(mount, r"D:\vramdisk\data", "ja").is_err());
    }
}
