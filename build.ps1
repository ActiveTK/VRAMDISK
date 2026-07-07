# Build the `vramdisk` library crate (core engine only, no binary — the CLI
# was merged into the GUI, see build-gui.ps1) inside a Visual Studio Dev Shell
# so winfsp-sys' bindgen can find the MSVC CRT headers, with LIBCLANG_PATH
# pointed at the LLVM install. Useful for a fast compile/test check of the
# engine without touching the Tauri/Node toolchain.
#
#   .\build.ps1            # debug build
#   .\build.ps1 --release  # release build
#
# Requirements: Visual Studio 2022 (with C++ tools), LLVM (libclang), CUDA
# Toolkit, and WinFsp installed.

param([Parameter(ValueFromRemainingArguments = $true)] $CargoArgs)

$ErrorActionPreference = "Stop"

function Find-VsInstallPath {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($p) { return $p }
    }
    foreach ($ed in @("Community", "Professional", "Enterprise", "BuildTools")) {
        $cand = "$env:ProgramFiles\Microsoft Visual Studio\2022\$ed"
        if (Test-Path $cand) { return $cand }
    }
    throw "Could not locate a Visual Studio 2022 installation with C++ tools."
}

$vsPath = Find-VsInstallPath
Import-Module "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments '-arch=x64 -host_arch=x64' -SkipAutomaticLocation | Out-Null

if (-not $env:LIBCLANG_PATH) {
    foreach ($cand in @("$env:ProgramFiles\LLVM\bin", "${env:ProgramFiles(x86)}\LLVM\bin")) {
        if (Test-Path "$cand\libclang.dll") { $env:LIBCLANG_PATH = $cand; break }
    }
}
if (-not $env:LIBCLANG_PATH) {
    throw "libclang.dll not found. Install LLVM or set LIBCLANG_PATH."
}

Set-Location $PSScriptRoot
Write-Host "Building with VS at $vsPath, LIBCLANG_PATH=$env:LIBCLANG_PATH" -ForegroundColor Cyan
cargo build @CargoArgs
exit $LASTEXITCODE
