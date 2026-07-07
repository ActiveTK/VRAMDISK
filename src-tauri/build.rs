fn main() {
    // DELAYLOAD the system WinFsp DLL (resolved at runtime by
    // preload_winfsp_dll()) so the GUI loads without WinFsp's bin dir on PATH.
    #[cfg(windows)]
    winfsp::build::winfsp_link_delayload();

    tauri_build::build();
}
