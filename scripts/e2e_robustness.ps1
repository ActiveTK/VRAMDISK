# Live mount robustness E2E for VRAMDISK.
# Mounts the release binary on a free drive letter, exercises file/dir CRUD,
# rename, delete, large-file integrity, malformed/unexpected operations, and
# concurrent access, then unmounts. Exits non-zero on the first failure.
#
# Usage:  pwsh -File scripts\e2e_robustness.ps1 [-Drive T] [-Exe path\to\vramdisk.exe]
#
# $Exe is the merged vramdisk.exe (GUI binary); this script drives its `cli`
# subcommand, not the GUI.

param(
    [string]$Drive = 'T',
    [string]$Exe = "$PSScriptRoot\..\src-tauri\target\release\vramdisk.exe"
)

$ErrorActionPreference = 'Stop'
$root = "${Drive}:"
$fail = 0
function Check($cond, $msg) {
    if ($cond) { Write-Host "  [ok]   $msg" -ForegroundColor Green }
    else       { Write-Host "  [FAIL] $msg" -ForegroundColor Red; $script:fail++ }
}
function ExpectThrows($script, $msg) {
    $threw = $false
    try { & $script | Out-Null } catch { $threw = $true }
    Check $threw $msg
}

Write-Host "Mounting $root via $Exe ..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $Exe -ArgumentList @('cli', '--mount', "$root", '--size', '512MiB') -PassThru -WindowStyle Hidden

try {
    # Wait for the volume to appear.
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Path "$root\") { $ready = $true; break }
        if ($proc.HasExited) { throw "mount process exited early (code $($proc.ExitCode))" }
    }
    if (-not $ready) { throw "volume did not appear at $root" }
    Write-Host "Mounted.`n" -ForegroundColor Cyan

    Write-Host "[1] Basic file/dir CRUD"
    New-Item -ItemType Directory "$root\docs" | Out-Null
    Check (Test-Path "$root\docs") "create directory"
    Set-Content "$root\docs\a.txt" "hello vramdisk"
    Check ((Get-Content "$root\docs\a.txt" -Raw).Trim() -eq "hello vramdisk") "write + read file"
    Add-Content "$root\docs\a.txt" "more"
    Check ((Get-Content "$root\docs\a.txt").Count -eq 2) "append grows file"
    Rename-Item "$root\docs\a.txt" "b.txt"
    Check ((Test-Path "$root\docs\b.txt") -and -not (Test-Path "$root\docs\a.txt")) "rename file"

    Write-Host "[2] Large binary file integrity (8 MiB)"
    $bytes = New-Object byte[] (8 * 1024 * 1024)
    (New-Object Random 12345).NextBytes($bytes)
    [IO.File]::WriteAllBytes("$root\big.bin", $bytes)
    $back = [IO.File]::ReadAllBytes("$root\big.bin")
    $h1 = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($bytes))
    $h2 = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($back))
    Check ($h1 -eq $h2) "8 MiB round-trip SHA-256 matches"

    Write-Host "[3] Nested dirs, subtree rename, recursive delete"
    New-Item -ItemType Directory "$root\src\sub" -Force | Out-Null
    Set-Content "$root\src\sub\deep.txt" "deep"
    Move-Item "$root\src" "$root\moved"
    Check (Test-Path "$root\moved\sub\deep.txt") "rename dir moves whole subtree"
    Remove-Item "$root\moved" -Recurse -Force
    Check (-not (Test-Path "$root\moved")) "recursive delete"

    Write-Host "[4] Unexpected / malformed operations must fail gracefully (mount stays up)"
    New-Item -ItemType Directory "$root\nonempty" | Out-Null
    Set-Content "$root\nonempty\x" "x"
    # Use .NET calls for the "must fail" cases so PowerShell never raises an
    # interactive confirmation prompt (which would hang a headless run).
    ExpectThrows { [IO.Directory]::Delete("$root\nonempty", $false) } "delete non-empty dir refused"
    Check (Test-Path "$root\nonempty\x") "...and its contents survive"
    ExpectThrows { [IO.Directory]::Move("$root\nonempty", "$root\nonempty\inner") } "rename dir into own subtree refused"
    Check (Test-Path "$root\nonempty\x") "...namespace intact after refused rename"
    New-Item -ItemType File "$root\dup" | Out-Null
    ExpectThrows { [IO.Directory]::CreateDirectory("$root\dup") } "name collision (file vs dir) refused"
    # Sparse write far into a file (within volume capacity) then read back.
    $fs = [IO.File]::Open("$root\sparse.bin", 'Create', 'ReadWrite')
    $fs.Seek(4MB, 'Begin') | Out-Null
    $fs.Write([byte[]]@(1,2,3,4), 0, 4)
    $fs.Close()
    Check ((Get-Item "$root\sparse.bin").Length -eq (4MB + 4)) "sparse write extends file"
    $sb = [IO.File]::ReadAllBytes("$root\sparse.bin")
    Check ($sb[0] -eq 0 -and $sb[4MB] -eq 1 -and $sb[4MB+3] -eq 4) "sparse hole reads zero, data intact"

    Write-Host "[5] Concurrent access (4 workers x 40 files each)"
    $jobs = 1..4 | ForEach-Object {
        Start-Job -ArgumentList $root, $_ -ScriptBlock {
            param($root, $id)
            $dir = "$root\work$id"
            New-Item -ItemType Directory $dir -Force | Out-Null
            for ($i = 0; $i -lt 40; $i++) {
                $f = "$dir\f$i.txt"
                Set-Content $f "worker $id file $i"
                $null = Get-Content $f
                if ($i % 3 -eq 0) { Remove-Item $f }
            }
            (Get-ChildItem $dir).Count
        }
    }
    $counts = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    # Each worker deletes indices 0,3,6,... (14 of 40) -> 26 survive.
    Check (($counts | Measure-Object -Sum).Sum -eq (4 * 26)) "concurrent workers left expected file count ($($counts -join ','))"

    Write-Host "[6] Volume still healthy after the storm"
    Set-Content "$root\final.txt" "still alive"
    Check ((Get-Content "$root\final.txt" -Raw).Trim() -eq "still alive") "filesystem responsive post-concurrency"
    Check ((Get-ChildItem "$root\").Count -ge 4) "root listing works"
}
finally {
    Write-Host "`nUnmounting (stop pid $($proc.Id)) ..." -ForegroundColor Cyan
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Milliseconds 800
}

if ($fail -eq 0) { Write-Host "`nALL E2E CHECKS PASSED" -ForegroundColor Green; exit 0 }
else            { Write-Host "`n$fail E2E CHECK(S) FAILED" -ForegroundColor Red; exit 1 }
