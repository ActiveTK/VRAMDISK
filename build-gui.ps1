# Build (or run) VRAMDISK inside a Visual Studio Dev Shell.
#
# This produces the single vramdisk.exe binary: launched with no args (or by
# double-click) it's the GUI; launched as `vramdisk.exe cli ...` or
# `vramdisk.exe benchmark ...` it runs the old standalone CLI's logic instead
# (no window). Any other argv just seeds the GUI's setup screen defaults.
#
# The Tauri backend (src-tauri) path-depends on the root `vramdisk` crate, which
# transitively compiles winfsp-sys via bindgen. bindgen needs the MSVC CRT
# headers and libclang, exactly like build.ps1 — so the GUI must be built from a
# VS Dev Shell with LIBCLANG_PATH set.
#
#   .\build-gui.ps1                 # release build -> src-tauri\target\release\vramdisk.exe
#   .\build-gui.ps1 dev             # tauri dev (run with devtools)
#   .\build-gui.ps1 build --debug   # debug build
#
# No installer is produced (bundle.active=false in tauri.conf.json):
# vramdisk.exe is a standalone executable, just copy and run it.
#
# Requirements: Visual Studio 2022 (C++ tools), LLVM (libclang), CUDA Toolkit,
# WinFsp, Node/npm with `npm install` already run in this directory.

param([Parameter(ValueFromRemainingArguments = $true)] $TauriArgs)

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

# Default subcommand is `build` when only flags (or nothing) are given.
# Note: with no args, $TauriArgs is $null and @($null) is a 1-element array
# (Count=1) holding a null, so filter nulls out before the count check.
$argsList = @($TauriArgs | Where-Object { $null -ne $_ })
if ($argsList.Count -eq 0 -or $argsList[0] -like '-*') {
    $argsList = @('build') + $argsList
}

Write-Host "GUI $($argsList -join ' ') via VS at $vsPath, LIBCLANG_PATH=$env:LIBCLANG_PATH" -ForegroundColor Cyan
& npx tauri @argsList
exit $LASTEXITCODE
