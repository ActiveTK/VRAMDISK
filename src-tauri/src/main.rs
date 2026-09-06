// Prevent an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod manager;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tauri::menu::{Menu, MenuBuilder, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager as _, Runtime, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use manager::Manager;

/// CLI-flag-derived defaults for the setup screen (see `commands::initial_overrides`).
/// `None` when the process was launched with no argv at all, in which case the
/// frontend falls back entirely to its saved `localStorage` config.
pub(crate) struct CliSeed(pub Option<vramdisk::cli::SeedOverrides>);

/// UI language ("ja" or "en"). Chosen from, in order: the value persisted
/// under HKCU\Software\VRAMDISK, else the Windows display language. The
/// frontend reads/writes it via the `get_ui_language` / `set_ui_language`
/// commands; the tray menu and native dialogs read it through this state.
pub(crate) struct UiLang(pub Mutex<String>);

/// Shared control block for the teardown ZIP export (`commands::export_zip`).
///
/// `running` keeps two exports from interleaving their progress events;
/// `cancel` is the flag the export's walk polls, set by `export_cancel`. Both
/// live in app state rather than the export itself because the cancel request
/// arrives on a different thread than the one doing the copying.
#[derive(Default)]
pub(crate) struct ExportControl {
    pub cancel: Arc<AtomicBool>,
    pub running: AtomicBool,
}

const LANG_REG_PATH: &str = r"Software\VRAMDISK";
const LANG_REG_VALUE: &str = "UiLanguage";

fn stored_language() -> Option<String> {
    let key = winreg::RegKey::predef(winreg::enums::HKEY_CURRENT_USER)
        .open_subkey(LANG_REG_PATH)
        .ok()?;
    let v: String = key.get_value(LANG_REG_VALUE).ok()?;
    matches!(v.as_str(), "ja" | "en").then_some(v)
}

fn detect_language() -> String {
    stored_language().unwrap_or_else(|| {
        let sys = sys_locale::get_locale().unwrap_or_default();
        if sys.to_ascii_lowercase().starts_with("ja") {
            "ja"
        } else {
            "en"
        }
        .to_string()
    })
}

pub(crate) fn persist_language(lang: &str) -> Result<(), String> {
    let (key, _) = winreg::RegKey::predef(winreg::enums::HKEY_CURRENT_USER)
        .create_subkey(LANG_REG_PATH)
        .map_err(|e| e.to_string())?;
    key.set_value(LANG_REG_VALUE, &lang).map_err(|e| e.to_string())
}

/// The language every backend-rendered string should be produced in. Commands
/// that already receive an `AppHandle` read it through here; the rest take
/// `State<UiLang>` directly.
pub(crate) fn app_language(app: &AppHandle) -> String {
    app.try_state::<UiLang>()
        .map(|s| s.0.lock().unwrap().clone())
        .unwrap_or_else(|| "ja".to_string())
}

/// All backend-rendered strings in both languages: the tray menu and native
/// dialogs rendered here, plus every message a Tauri command or the mount
/// manager can hand back to the frontend for display.
///
/// `key` is `&'static str` (every call site passes a literal) purely so the
/// fallback arm can hand the key back with the required lifetime.
///
/// Messages that embed a value use `{0}`, `{1}`, ... placeholders and are
/// rendered through [`trf`]; the numbering mirrors the frontend's `t()` in
/// `ui/app.js` so both halves of the UI read the same way.
pub(crate) fn tr(lang: &str, key: &'static str) -> &'static str {
    let ja = lang == "ja";
    match key {
        "show" => if ja { "ウィンドウを開く" } else { "Open window" },
        "unmount" => if ja { "アンマウント" } else { "Unmount" },
        "hash" => if ja { "ファイルのハッシュ計算" } else { "Hash files" },
        "archive" => if ja { "ファイルの圧縮・展開" } else { "Compress / extract files" },
        "search" => if ja { "ファイルの全文検索" } else { "Search file contents" },
        "encode" => if ja { "ファイルのエンコード（Base64 / hex）" } else { "Encode files (Base64 / hex)" },
        "quit" => if ja { "終了" } else { "Exit" },
        "close_hint" => if ja { "VRAMDISKはタスクトレイからアンマウントできます" } else { "VRAMDISK stays available in the system tray." },
        "unmount_warning" => if ja { "本当にアンマウントしますか？\nドライブ上のデータは全て失われます。" } else { "Unmount now?\nAll data on the drive will be lost." },
        "quit_warning" => if ja { "本当に終了しますか？\nドライブ上のデータは全て失われます。" } else { "Exit now?\nAll data on the drive will be lost." },
        "continue" => if ja { "続行" } else { "Continue" },
        "cancel" => if ja { "キャンセル" } else { "Cancel" },
        "mounted_msg" => if ja { "マウントしました。\nタスクトレイから操作できます。" } else { "Mounted.\nUse the tray icon to manage the drive." },

        // --- Errors returned by Tauri commands (commands.rs) ---
        "err_unsupported_language" => if ja { "対応していない言語です: {0}" } else { "Unsupported language: {0}" },
        "err_nvcomp_unavailable" => if ja { "nvCOMP が見つからないため、この GUI では圧縮を利用できません。" } else { "nvCOMP was not found, so compression is unavailable in this GUI." },
        "err_no_paths" => if ja { "対象のパスが指定されていません" } else { "No paths given" },
        "err_not_mounted" => if ja { "マウントされていません" } else { "Nothing is mounted" },
        "err_read_failed" => if ja { "{0} を読み取れません: {1}" } else { "Could not read {0}: {1}" },
        "err_parse_failed" => if ja { "{0} を解析できません: {1}" } else { "Could not parse {0}: {1}" },
        "err_invalid_job_id" => if ja { "ジョブ ID が不正です: {0}" } else { "Invalid job id: {0}" },
        "err_job_cancel_failed" => if ja { "ジョブを中止できません {0}: {1}" } else { "Could not cancel the job {0}: {1}" },
        "err_job_submit_failed" => if ja { "ジョブを投入できません {0}: {1}" } else { "Could not submit the job {0}: {1}" },
        "err_path_not_on_volume" => if ja { "パス \"{0}\" はマウント先 {1} 上のパスではありません" } else { "The path \"{0}\" is not on the mount point {1}" },

        // --- Teardown ZIP export (commands.rs) ---
        "err_export_destination_required" => if ja { "保存先を指定してください" } else { "Choose where to save" },
        "err_export_destination_on_volume" => if ja { "保存先 \"{0}\" はマウント中のドライブ上です。別のドライブを指定してください" } else { "The destination \"{0}\" is on the mounted drive. Choose another drive." },
        "err_export_already_running" => if ja { "保存は既に実行中です" } else { "A save is already in progress" },
        "err_export_create_failed" => if ja { "保存先を作成できません {0}: {1}" } else { "Could not create the destination {0}: {1}" },
        "err_export_add_dir_failed" => if ja { "ZIP にフォルダを追加できません {0}: {1}" } else { "Could not add the folder {0} to the ZIP: {1}" },
        "err_export_add_file_failed" => if ja { "ZIP にファイルを追加できません {0}: {1}" } else { "Could not add the file {0} to the ZIP: {1}" },
        "err_export_write_failed" => if ja { "ZIP に書き込めません {0}: {1}" } else { "Could not write {0} to the ZIP: {1}" },
        "err_export_finish_failed" => if ja { "ZIP を閉じられません {0}: {1}" } else { "Could not close the ZIP {0}: {1}" },
        "err_export_flush_failed" => if ja { "ZIP を書き出せません {0}: {1}" } else { "Could not flush the ZIP {0}: {1}" },
        "err_export_unsupported_entry" => if ja { "通常のファイルでもフォルダでもないため保存できません" } else { "Unsupported entry type (not a regular file or directory)" },

        // --- Mount manager (manager.rs) ---
        "err_already_mounted" => if ja { "既にマウントされています。先にアンマウントしてください" } else { "A disk is already mounted; unmount it first" },
        "err_mount_point_required" => if ja { "マウント先を指定してください" } else { "Choose a mount point" },
        "err_size_zero" => if ja { "容量は 0 より大きい値を指定してください" } else { "Disk size must be greater than 0" },
        "err_size_exceeds_vram" => if ja { "指定した容量 {0} バイトは GPU の VRAM 容量 {1} バイトを超えています" } else { "The requested {0} bytes exceeds the device VRAM ({1} bytes)" },
        "err_mount_prepare_folder" => if ja { "マウント先フォルダを準備できません: {0}: {1}" } else { "Could not prepare the mount folder {0}: {1}" },
        "err_mount_restore_folder" => if ja { "{0}（さらに、マウント先フォルダの復元にも失敗しました: {1}）" } else { "{0} (and restoring the mount folder failed as well: {1})" },
        "err_winfsp_driver_not_loaded" => if ja { "WinFsp のカーネルドライバが読み込まれていません。\n\nWinFsp 自体はインストールされていますが、ファイルシステムドライバが動作していないため、\nボリュームを接続するデバイスがありません。`fsptool unload` を実行するとこの状態になり、\nドライバが自動で読み込み直されることはありません。\n\n管理者権限のコマンドプロンプトで次を実行してください（再起動は不要です）:\n\n    \"C:\\Program Files (x86)\\WinFsp\\bin\\fsptool-x64.exe\" load" } else { "The WinFsp kernel driver is not loaded.\n\nWinFsp itself is installed, but its file system driver is not running, so there is\nno device to attach a volume to. `fsptool unload` leaves the machine in this state,\nand nothing reloads the driver on its own.\n\nFrom an elevated prompt (no reboot needed):\n\n    \"C:\\Program Files (x86)\\WinFsp\\bin\\fsptool-x64.exe\" load" },
        "err_mount_point_not_folder" => if ja { "マウント先はフォルダではありません: {0}" } else { "The mount point is not a folder: {0}" },
        "err_mount_folder_open_failed" => if ja { "マウント先フォルダを開けません: {0}: {1}" } else { "Could not open the mount folder {0}: {1}" },
        "err_mount_folder_not_empty" => if ja { "マウント先フォルダが空ではありません: {0}" } else { "The mount folder is not empty: {0}" },
        "err_mount_windows_only" => if ja { "マウントは Windows（WinFsp）でのみ利用できます" } else { "Mounting is only supported on Windows (WinFsp)" },

        // Fall back to the key itself, never to "": a typo'd or not-yet-added
        // key would otherwise render as a blank tray entry or an empty dialog,
        // which looks like a broken app rather than a missing translation.
        _ => key,
    }
}

/// [`tr`] for messages that embed values — a path, an OS error string — as
/// `{0}`, `{1}`, ... placeholders. Separate from `tr` rather than folded into
/// it because the tray/dialog callers need a `&'static str` and must not pay
/// for an allocation, while these necessarily build a new `String`.
///
/// Inherits `tr`'s fallback: an unknown key yields the key text itself, which
/// contains no placeholders and so comes back verbatim — never blank. A
/// placeholder with no matching argument is likewise left as written instead
/// of silently swallowing part of the message.
///
/// Substitution is single-pass, so an argument that happens to contain
/// something like `{0}` (a path can) is never re-substituted.
pub(crate) fn trf(lang: &str, key: &'static str, args: &[&str]) -> String {
    let template = tr(lang, key);
    let mut out = String::with_capacity(template.len() + 32);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open..].find('}').map(|i| open + i) else {
            break;
        };
        match rest[open + 1..close]
            .parse::<usize>()
            .ok()
            .and_then(|i| args.get(i))
        {
            Some(value) => {
                out.push_str(&rest[..open]);
                out.push_str(value);
            }
            None => out.push_str(&rest[..=close]),
        }
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

/// Emitted to the main window whenever the mount state changes (mount,
/// unmount, whether triggered from the UI or the tray), carrying the current
/// `Option<MountStatus>` so the frontend can switch screens without polling.
const EVENT_MOUNT_CHANGED: &str = "mount-changed";

/// Emitted when the user asks (via tray or in-app button) to open the GPU
/// archive compression panel.
const EVENT_OPEN_ARCHIVE_PANEL: &str = "open-archive-panel";

/// Emitted when the user asks (via tray) to open the GPU hash panel.
const EVENT_OPEN_HASH_PANEL: &str = "open-hash-panel";

/// Emitted when the user asks (via tray) to open the GPU encode panel.
const EVENT_OPEN_ENCODE_PANEL: &str = "open-encode-panel";

/// Emitted when the user asks (via tray) to open the GPU search panel.
const EVENT_OPEN_SEARCH_PANEL: &str = "open-search-panel";

/// Emitted when the tray asks the window to run the teardown prompt. The
/// payload is the intent — "unmount" or "quit" — because the two differ only
/// in what happens after the (optional) ZIP export succeeds.
///
/// This exists because the choice is three-way (save then tear down / tear
/// down anyway / cancel) and a native message dialog tops out at two buttons.
const EVENT_REQUEST_TEARDOWN: &str = "request-teardown";

/// Progress of the teardown ZIP export, emitted while `commands::export_zip`
/// runs. The payload deliberately mirrors the job system's `status.json`
/// progress object (`{"done_bytes": N, "total_bytes": M}`, see
/// `vramdisk::jobs::JobProgress`) so the frontend's existing progress-bar
/// helpers can render it unchanged.
pub(crate) const EVENT_EXPORT_PROGRESS: &str = "export-progress";

/// Emitted whenever the main window is hidden rather than closed. The window
/// keeps running while hidden, so anything modal it was showing — in practice
/// the teardown prompt — has to be dismissed; otherwise reopening from the tray
/// lands the user back on a stale "about to unmount" dialog they never answered.
const EVENT_WINDOW_HIDDEN: &str = "window-hidden";

/// Hide the main window and tell the frontend it happened.
fn hide_main_window(app: &AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.hide();
    }
    let _ = app.emit(EVENT_WINDOW_HIDDEN, ());
}

fn show_main_window(app: &AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.show();
        let _ = win.unminimize();
        let _ = win.set_focus();
    }
}

pub(crate) fn quit(app: &AppHandle) {
    if let Some(manager) = app.try_state::<Manager>() {
        manager.shutdown();
    }
    app.exit(0);
}

/// Tear down from the tray ("アンマウント" / mounted "終了").
///
/// The user must be offered three outcomes — save the volume to a ZIP first,
/// tear down anyway, or cancel — and `MessageDialogButtons` has no three-button
/// form, so the prompt lives in the window: show it (it is usually hidden while
/// mounted) and let the frontend run the choice. The old two-button native
/// confirm survives only for the case where there is genuinely no window to
/// show, where "discard or cancel" is the best that can be offered.
fn request_teardown(app: &AppHandle, intent: &'static str) {
    if app.get_webview_window("main").is_some() {
        show_main_window(app);
        let _ = app.emit(EVENT_REQUEST_TEARDOWN, intent);
        return;
    }
    let warning = if intent == "quit" {
        "quit_warning"
    } else {
        "unmount_warning"
    };
    confirm_discard(app, warning, move |app| {
        let lang = app_language(&app);
        let manager = app.state::<Manager>();
        if manager.unmount(&lang).is_ok() {
            on_mount_state_changed(&app, &manager);
        }
        if intent == "quit" {
            quit(&app);
        }
    });
}

/// The two-button "you are about to lose the data" native confirm. Only the
/// no-window fallback of [`request_teardown`] still uses it; the real prompt
/// is the window's three-way one.
fn confirm_discard(
    app: &AppHandle,
    warning: &'static str,
    on_confirm: impl FnOnce(AppHandle) + Send + 'static,
) {
    let app = app.clone();
    let lang = app_language(&app);
    app.dialog()
        .message(tr(&lang, warning))
        .kind(MessageDialogKind::Warning)
        .title("VRAMDISK")
        .buttons(MessageDialogButtons::OkCancelCustom(
            tr(&lang, "continue").into(),
            tr(&lang, "cancel").into(),
        ))
        .show(move |confirmed| {
            if confirmed {
                on_confirm(app);
            }
        });
}

/// Build the tray menu in the given language, enabling the unmount / GPU tool
/// items only while a disk is actually mounted.
fn build_tray_menu<R: Runtime>(
    app: &AppHandle<R>,
    mounted: bool,
    lang: &str,
) -> tauri::Result<Menu<R>> {
    let show_item = MenuItemBuilder::with_id("show", tr(lang, "show")).build(app)?;
    let unmount_item = MenuItemBuilder::with_id("unmount", tr(lang, "unmount"))
        .enabled(mounted)
        .build(app)?;
    let hash_item = MenuItemBuilder::with_id("hash", tr(lang, "hash"))
        .enabled(mounted)
        .build(app)?;
    let archive_item = MenuItemBuilder::with_id("archive", tr(lang, "archive"))
        .enabled(mounted && vramdisk::nvcomp::nvcomp_available())
        .build(app)?;
    let search_item = MenuItemBuilder::with_id("search", tr(lang, "search"))
        .enabled(mounted)
        .build(app)?;
    let encode_item = MenuItemBuilder::with_id("encode", tr(lang, "encode"))
        .enabled(mounted)
        .build(app)?;
    let quit_item = MenuItemBuilder::with_id("quit", tr(lang, "quit")).build(app)?;
    MenuBuilder::new(app)
        .items(&[
            &show_item,
            &unmount_item,
            &hash_item,
            &archive_item,
            &search_item,
            &encode_item,
        ])
        .separator()
        .item(&quit_item)
        .build()
}

/// Rebuild the tray menu to match the current mount state and notify the
/// frontend. Call this after any mount/unmount, regardless of whether it was
/// triggered from the UI or the tray.
pub(crate) fn on_mount_state_changed(app: &AppHandle, manager: &Manager) {
    let status = manager.status();
    refresh_tray_menu(app, status.is_some());
    let _ = app.emit(EVENT_MOUNT_CHANGED, status);
}

/// Rebuild the tray menu for the current language and the given mount state.
/// Also called by `set_ui_language` when the user switches languages.
pub(crate) fn refresh_tray_menu(app: &AppHandle, mounted: bool) {
    let lang = app_language(app);
    if let Some(tray) = app.tray_by_id("vramdisk-tray") {
        if let Ok(menu) = build_tray_menu(app, mounted, &lang) {
            let _ = tray.set_menu(Some(menu));
        }
    }
}

/// After a successful mount: hide the main window (the tray keeps it
/// reachable), show a one-time confirmation dialog, and open the drive in
/// Explorer only once the user dismisses that dialog.
pub(crate) fn after_mount_success(app: &AppHandle, mount_point: &str) {
    hide_main_window(app);
    let path = format!("{mount_point}\\");
    let lang = app_language(app);
    app.dialog()
        .message(tr(&lang, "mounted_msg"))
        .kind(MessageDialogKind::Info)
        .title("VRAMDISK")
        .show(move |_| {
            let _ = std::process::Command::new("explorer.exe")
                .arg(&path)
                .spawn();
        });
}

/// vramdisk.exe is both the GUI and (via these two dispatch tokens) the CLI
/// that used to be a separate `vramdisk-cli.exe`. `cli`/`benchmark` run the
/// old CLI synchronously in the console and exit — no window is ever
/// created. Any other argv (e.g. a shortcut with `--mount R: --compress`) is
/// left for the normal GUI startup below, which uses it only to seed the
/// setup screen's initial field values (see `commands::initial_overrides`),
/// never to mount automatically.
fn dispatch_cli_mode(argv: &[String]) {
    let Some(first) = argv.first() else {
        return;
    };
    let mode = first.to_ascii_lowercase();
    if mode != "cli" && mode != "benchmark" {
        return;
    }
    let mut rest = argv[1..].to_vec();
    if mode == "benchmark" && !rest.iter().any(|a| a == "--bench" || a == "--bench-io") {
        rest.insert(0, "--bench".to_string());
    }
    std::process::exit(vramdisk::cli_run::run(rest));
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    dispatch_cli_mode(&argv);
    let cli_seed = CliSeed(if argv.is_empty() {
        None
    } else {
        Some(vramdisk::cli::scan_overrides(&argv))
    });

    let manager = Manager::spawn();

    let app = tauri::Builder::default()
        // Must be the first plugin registered: it lets an already-running
        // instance intercept a second launch (instead of allocating a second
        // VRAM buffer / WinFsp host) and just refocus its window instead.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app);
        }))
        .plugin(tauri_plugin_dialog::init())
        .manage(manager)
        .manage(cli_seed)
        .manage(UiLang(Mutex::new(detect_language())))
        .manage(ExportControl::default())
        .invoke_handler(tauri::generate_handler![
            commands::get_ui_language,
            commands::set_ui_language,
            commands::list_gpus,
            commands::list_free_drives,
            commands::browse_folder,
            commands::browse_file,
            commands::browse_save,
            commands::browse_export_zip,
            commands::mount_status,
            commands::mount,
            commands::unmount,
            commands::quit_app,
            commands::export_zip,
            commands::export_cancel,
            commands::stats,
            commands::nvcomp_available,
            commands::initial_overrides,
            commands::hash_job,
            commands::archive_compress_job,
            commands::archive_extract_job,
            commands::encode_job,
            commands::search_job,
            commands::job_status,
            commands::job_result,
            commands::job_cancel,
        ])
        .setup(|app| {
            // --- System tray (starts unmounted: a fresh process owns no mount yet) ---
            let lang = app_language(app.handle());
            let menu = build_tray_menu(app.handle(), false, &lang)?;

            TrayIconBuilder::with_id("vramdisk-tray")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("VRAMDISK")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => show_main_window(app),
                    "unmount" => request_teardown(app, "unmount"),
                    "hash" => {
                        show_main_window(app);
                        let _ = app.emit(EVENT_OPEN_HASH_PANEL, ());
                    }
                    "archive" => {
                        show_main_window(app);
                        let _ = app.emit(EVENT_OPEN_ARCHIVE_PANEL, ());
                    }
                    "search" => {
                        show_main_window(app);
                        let _ = app.emit(EVENT_OPEN_SEARCH_PANEL, ());
                    }
                    "encode" => {
                        show_main_window(app);
                        let _ = app.emit(EVENT_OPEN_ENCODE_PANEL, ());
                    }
                    "quit" => {
                        let mounted = app
                            .try_state::<Manager>()
                            .map(|m| m.status().is_some())
                            .unwrap_or(false);
                        if mounted {
                            // Quitting with a disk mounted discards it just as
                            // surely as unmounting does, so it gets the same
                            // save-first offer.
                            request_teardown(app, "quit");
                        } else {
                            quit(app);
                        }
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main_window(tray.app_handle());
                    }
                })
                .build(app)?;

            // --- Hide-on-close: keep the mount alive, hint the user once ---
            if let Some(win) = app.get_webview_window("main") {
                let handle = app.handle().clone();
                let hinted = Arc::new(AtomicBool::new(false));
                win.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        // Only hide-to-tray when a disk is actually mounted;
                        // otherwise closing the window really quits the app.
                        let mounted = handle
                            .try_state::<Manager>()
                            .map(|m| m.status().is_some())
                            .unwrap_or(false);
                        if !mounted {
                            quit(&handle);
                            return;
                        }
                        api.prevent_close();
                        hide_main_window(&handle);
                        if !hinted.swap(true, Ordering::SeqCst) {
                            let lang = app_language(&handle);
                            handle
                                .dialog()
                                .message(tr(&lang, "close_hint"))
                                .kind(MessageDialogKind::Info)
                                .title("VRAMDISK")
                                .show(|_| {});
                        }
                    }
                });
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building VRAMDISK GUI");

    // Safety net: unmount on any exit path.
    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(manager) = app_handle.try_state::<Manager>() {
                manager.shutdown();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the fallback arm: a key that isn't in the catalog
    /// must still render as *something*. A blank tray entry or an empty error
    /// dialog reads as a broken app; the key name reads as a missing string.
    #[test]
    fn unknown_keys_fall_back_to_the_key_itself() {
        assert_eq!(tr("ja", "no_such_key"), "no_such_key");
        assert_eq!(tr("en", "no_such_key"), "no_such_key");
        assert_eq!(trf("ja", "no_such_key", &["x"]), "no_such_key");
        assert_eq!(trf("en", "no_such_key", &[]), "no_such_key");
    }

    #[test]
    fn interpolated_messages_carry_their_values_in_both_languages() {
        let ja = trf("ja", "err_read_failed", &[r"R:\a.json", "denied"]);
        let en = trf("en", "err_read_failed", &[r"R:\a.json", "denied"]);
        assert_ne!(ja, en);
        for rendered in [&ja, &en] {
            assert!(rendered.contains(r"R:\a.json"), "{rendered}");
            assert!(rendered.contains("denied"), "{rendered}");
            assert!(!rendered.contains('{'), "{rendered}");
        }
    }

    /// A path really can contain braces, so substitution must not rescan what
    /// it just wrote; and a placeholder with no argument stays visible rather
    /// than silently eating part of the sentence.
    #[test]
    fn substitution_is_single_pass_and_keeps_unfilled_placeholders() {
        assert_eq!(
            trf("en", "err_read_failed", &["{1}", "real"]),
            "Could not read {1}: real"
        );
        assert_eq!(
            trf("en", "err_read_failed", &["only"]),
            "Could not read only: {1}"
        );
    }
}
