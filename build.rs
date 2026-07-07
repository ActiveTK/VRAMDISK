//! Build script: enable DELAYLOAD linking to the system WinFsp DLL so the
//! binary loads even though `winfsp-x64.dll` lives in WinFsp's own bin dir
//! (resolved at runtime by `winfsp::winfsp_init`).

fn main() {
    #[cfg(windows)]
    winfsp::build::winfsp_link_delayload();
}
