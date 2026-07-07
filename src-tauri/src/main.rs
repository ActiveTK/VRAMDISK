// Prevent an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod manager;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tauri::menu::{Menu, MenuBuilder, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager as _, Runtime, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use manager::Manager;

/// CLI-flag-derived defaults for the setup screen (see `commands::initial_overrides`).
/// `None` when the process was launched with no argv at all, in which case the
/// frontend falls back entirely to its saved `localStorage` config.
pub(crate) struct CliSeed(pub Option<vramdisk::cli::SeedOverrides>);

/// Shown once, when the user closes the window while a mount may be live.
const CLOSE_HINT: &str = "VRAMDISKはタスクトレイからアンマウントできます";

const UNMOUNT_WARNING: &str = "本当にアンマウントしますか？\nドライブ上のデータは全て失われます。";

/// Emitted to the main window whenever the mount state changes (mount,
/// unmount, whether triggered from the UI or the tray), carrying the current
/// `Option<MountStatus>` so the frontend can switch screens without polling.
const EVENT_MOUNT_CHANGED: &str = "mount-changed";

/// Emitted when the user asks (via tray or in-app button) to open the GPU
/// archive compression panel.
const EVENT_OPEN_ARCHIVE_PANEL: &str = "open-archive-panel";

/// Emitted when the user asks (via tray) to open the GPU hash panel.
const EVENT_OPEN_HASH_PANEL: &str = "open-hash-panel";

fn show_main_window(app: &AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.show();
        let _ = win.unminimize();
        let _ = win.set_focus();
    }
}

fn quit(app: &AppHandle) {
    if let Some(manager) = app.try_state::<Manager>() {
        manager.shutdown();
    }
    app.exit(0);
}

fn confirm_unmount(app: &AppHandle, on_confirm: impl FnOnce(AppHandle) + Send + 'static) {
    let app = app.clone();
    app.dialog()
        .message(UNMOUNT_WARNING)
        .kind(MessageDialogKind::Warning)
        .title("VRAMDISK")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "続行".into(),
            "キャンセル".into(),
        ))
        .show(move |confirmed| {
            if confirmed {
                on_confirm(app);
            }
        });
}

/// Build the tray menu, enabling "アンマウント" / "ファイルのハッシュ計算" /
/// "ファイルをGPU上で圧縮" only while a disk is actually mounted.
fn build_tray_menu<R: Runtime>(app: &AppHandle<R>, mounted: bool) -> tauri::Result<Menu<R>> {
    let show_item = MenuItemBuilder::with_id("show", "ウィンドウを開く").build(app)?;
    let unmount_item = MenuItemBuilder::with_id("unmount", "アンマウント")
        .enabled(mounted)
        .build(app)?;
    let hash_item = MenuItemBuilder::with_id("hash", "ファイルのハッシュ計算")
        .enabled(mounted)
        .build(app)?;
    let archive_item = MenuItemBuilder::with_id("archive", "ファイル圧縮（nvCOMP）")
        .enabled(mounted && vramdisk::nvcomp::nvcomp_available())
        .build(app)?;
    let quit_item = MenuItemBuilder::with_id("quit", "終了").build(app)?;
    MenuBuilder::new(app)
        .items(&[&show_item, &unmount_item, &hash_item, &archive_item])
        .separator()
        .item(&quit_item)
        .build()
}

/// Rebuild the tray menu to match the current mount state and notify the
/// frontend. Call this after any mount/unmount, regardless of whether it was
/// triggered from the UI or the tray.
pub(crate) fn on_mount_state_changed(app: &AppHandle, manager: &Manager) {
    let status = manager.status();
    if let Some(tray) = app.tray_by_id("vramdisk-tray") {
        if let Ok(menu) = build_tray_menu(app, status.is_some()) {
            let _ = tray.set_menu(Some(menu));
        }
    }
    let _ = app.emit(EVENT_MOUNT_CHANGED, status);
}

/// After a successful mount: hide the main window (the tray keeps it
/// reachable), show a one-time confirmation dialog, and open the drive in
/// Explorer only once the user dismisses that dialog.
pub(crate) fn after_mount_success(app: &AppHandle, mount_point: &str) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.hide();
    }
    let path = format!("{mount_point}\\");
    app.dialog()
        .message("マウントしました。\nタスクトレイから操作できます。")
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
        .invoke_handler(tauri::generate_handler![
            commands::list_gpus,
            commands::list_free_drives,
            commands::browse_folder,
            commands::browse_file,
            commands::browse_save,
            commands::mount_status,
            commands::mount,
            commands::unmount,
            commands::stats,
            commands::nvcomp_available,
            commands::initial_overrides,
            commands::hash_job,
            commands::archive_compress_job,
            commands::archive_extract_job,
        ])
        .setup(|app| {
            // --- System tray (starts unmounted: a fresh process owns no mount yet) ---
            let menu = build_tray_menu(app.handle(), false)?;

            TrayIconBuilder::with_id("vramdisk-tray")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("VRAMDISK")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => show_main_window(app),
                    "unmount" => {
                        confirm_unmount(app, |app| {
                            let manager = app.state::<Manager>();
                            if manager.unmount().is_ok() {
                                on_mount_state_changed(&app, &manager);
                            }
                        });
                    }
                    "hash" => {
                        show_main_window(app);
                        let _ = app.emit(EVENT_OPEN_HASH_PANEL, ());
                    }
                    "archive" => {
                        show_main_window(app);
                        let _ = app.emit(EVENT_OPEN_ARCHIVE_PANEL, ());
                    }
                    "quit" => {
                        let mounted = app
                            .try_state::<Manager>()
                            .map(|m| m.status().is_some())
                            .unwrap_or(false);
                        if mounted {
                            confirm_unmount(app, |app| quit(&app));
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
                        if let Some(win) = handle.get_webview_window("main") {
                            let _ = win.hide();
                        }
                        if !hinted.swap(true, Ordering::SeqCst) {
                            handle
                                .dialog()
                                .message(CLOSE_HINT)
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
