//! Build script: delay-load the system WinFsp DLL. The optional WinSpd feature
//! builds a small C ABI shim against the WinSpd Developer SDK headers.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn run(program: &str, args: &[String]) {
    let status = Command::new(program)
        .args(args)
        .status()
        .unwrap_or_else(|error| panic!("could not run {program}: {error}"));
    assert!(status.success(), "{program} failed with {status}");
}

fn build_winspd_shim() {
    println!("cargo:rerun-if-env-changed=WINSPD_SDK_DIR");
    println!("cargo:rerun-if-changed=native/winspd_shim.c");

    let sdk = env::var_os("WINSPD_SDK_DIR")
        .map(PathBuf::from)
        .expect("the `winspd` feature requires WINSPD_SDK_DIR pointing to the WinSpd Developer SDK");
    let include = sdk.join("inc");
    assert!(include.join("winspd/winspd.h").is_file(), "WinSpd headers not found under {}", include.display());

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let object = out_dir.join("winspd_shim.obj");
    let library = out_dir.join("winspd_shim.lib");
    let source = PathBuf::from("native/winspd_shim.c");

    run(
        "cl.exe",
        &[
            "/nologo".into(),
            "/c".into(),
            "/O2".into(),
            "/W3".into(),
            format!("/I{}", include.display()),
            format!("/Fo{}", object.display()),
            source.display().to_string(),
        ],
    );
    run(
        "lib.exe",
        &[
            "/nologo".into(),
            format!("/OUT:{}", library.display()),
            object.display().to_string(),
        ],
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=winspd_shim");
    println!("cargo:rustc-link-lib=rpcrt4");
}

fn main() {
    #[cfg(windows)]
    winfsp::build::winfsp_link_delayload();

    if env::var_os("CARGO_FEATURE_WINSPD").is_some() {
        if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
            panic!("the `winspd` feature is available only for Windows targets");
        }
        if env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
            panic!("the `winspd` feature requires the MSVC target");
        }
        build_winspd_shim();
    }
}
