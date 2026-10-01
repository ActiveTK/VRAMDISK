# Head-to-head benchmark: VRAMDISK vs an ordinary RAM disk.
#
# VRAMDISK's data lives behind PCIe, a RAM disk's does not, so the interesting
# question is not "is it fast" but "what does that cost, and where does it stop
# mattering". This script answers that with numbers rather than intuition:
# sequential throughput at several block sizes, small random-read latency, and
# the metadata/small-file workload a RAM disk is usually bought for.
#
# Needs an NVIDIA GPU, CUDA and WinFsp (so it cannot run on CI), plus a RAM disk
# already mounted somewhere for the baseline -- ImDisk, OSFMount, or any other.
# Creating one usually needs elevation, which this script deliberately does not
# ask for; mount it yourself and pass its drive letter.
#
#   pwsh -File scripts\bench_vs_ramdisk.ps1 -RamDisk R:\ [-Drive V:] [-Size 4GiB]
#
# Both targets are driven through the identical harness. I/O is measured both
# unbuffered (FILE_FLAG_NO_BUFFERING, i.e. the storage path itself) and buffered
# (what an ordinary application does, with the Windows cache manager in front),
# because both drives sit behind the cache manager and measuring only one mode
# would say more about the cache than about the device.
param(
    [Parameter(Mandatory = $true)][string]$RamDisk,
    [string]$Drive = "V:",
    [string]$Size = "4GiB",
    [long]$FileBytes = 2GB,
    [int]$SmallFiles = 4000,
    [string]$Exe = "$PSScriptRoot\..\src-tauri\target\release\vramdisk.exe"
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Exe)) { throw "release binary not found at $Exe -- build it first" }
if (-not (Test-Path $RamDisk)) { throw "RAM disk not found at $RamDisk" }

Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class Bench {
    const uint GENERIC_READ = 0x80000000, GENERIC_WRITE = 0x40000000;
    const uint CREATE_ALWAYS = 2, OPEN_EXISTING = 3;
    const uint NO_BUFFERING = 0x20000000, WRITE_THROUGH = 0x80000000, SEQUENTIAL = 0x08000000;
    const uint MEM_COMMIT = 0x1000, MEM_RESERVE = 0x2000, MEM_RELEASE = 0x8000, PAGE_RW = 0x04;

    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern SafeFileHandle CreateFileW(string n, uint a, uint s, IntPtr sec, uint d, uint f, IntPtr t);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool WriteFile(SafeFileHandle h, IntPtr b, uint n, out uint w, IntPtr o);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool ReadFile(SafeFileHandle h, IntPtr b, uint n, out uint r, IntPtr o);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool SetFilePointerEx(SafeFileHandle h, long d, IntPtr p, uint o);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool FlushFileBuffers(SafeFileHandle h);
    [DllImport("kernel32", SetLastError = true)]
    static extern IntPtr VirtualAlloc(IntPtr a, UIntPtr s, uint t, uint p);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool VirtualFree(IntPtr a, UIntPtr s, uint t);

    // Unbuffered I/O requires sector-aligned buffers, which a plain byte[] does
    // not guarantee; VirtualAlloc always returns page-aligned memory.
    static IntPtr Alloc(int n) {
        var p = VirtualAlloc(IntPtr.Zero, (UIntPtr)(ulong)n, MEM_COMMIT | MEM_RESERVE, PAGE_RW);
        if (p == IntPtr.Zero) throw new OutOfMemoryException("VirtualAlloc");
        return p;
    }
    static SafeFileHandle Open(string path, bool write, bool unbuffered) {
        uint f = SEQUENTIAL;
        if (unbuffered) f |= NO_BUFFERING | WRITE_THROUGH;
        var h = CreateFileW(path, write ? GENERIC_WRITE : GENERIC_READ, 0, IntPtr.Zero,
                            write ? CREATE_ALWAYS : OPEN_EXISTING, f, IntPtr.Zero);
        if (h.IsInvalid) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), path);
        return h;
    }

    public static double Write(string path, long total, int block, bool unbuffered) {
        IntPtr buf = Alloc(block);
        try {
            var tmp = new byte[block]; new Random(1).NextBytes(tmp); Marshal.Copy(tmp, 0, buf, block);
            using (var h = Open(path, true, unbuffered)) {
                var sw = Stopwatch.StartNew();
                for (long done = 0; done < total; ) {
                    uint take = (uint)Math.Min(block, total - done); uint w;
                    if (!WriteFile(h, buf, take, out w, IntPtr.Zero) || w != take)
                        throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "WriteFile");
                    done += w;
                }
                FlushFileBuffers(h); sw.Stop(); return sw.Elapsed.TotalSeconds;
            }
        } finally { VirtualFree(buf, UIntPtr.Zero, MEM_RELEASE); }
    }

    public static double Read(string path, long total, int block, bool unbuffered) {
        IntPtr buf = Alloc(block);
        try {
            using (var h = Open(path, false, unbuffered)) {
                var sw = Stopwatch.StartNew();
                for (long done = 0; done < total; ) {
                    uint take = (uint)Math.Min(block, total - done); uint r;
                    if (!ReadFile(h, buf, take, out r, IntPtr.Zero))
                        throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "ReadFile");
                    if (r == 0) break;
                    done += r;
                }
                sw.Stop(); return sw.Elapsed.TotalSeconds;
            }
        } finally { VirtualFree(buf, UIntPtr.Zero, MEM_RELEASE); }
    }

    public static double RandomReadUs(string path, long total, int block, int count) {
        IntPtr buf = Alloc(block);
        try {
            var rnd = new Random(7); long blocks = total / block;
            using (var h = Open(path, false, true)) {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < count; i++) {
                    SetFilePointerEx(h, (long)(rnd.NextDouble() * (blocks - 1)) * block, IntPtr.Zero, 0);
                    uint r;
                    if (!ReadFile(h, buf, (uint)block, out r, IntPtr.Zero))
                        throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "ReadFile");
                }
                sw.Stop(); return sw.Elapsed.TotalSeconds / count * 1e6;
            }
        } finally { VirtualFree(buf, UIntPtr.Zero, MEM_RELEASE); }
    }
}
'@

function Sec([scriptblock]$b) { $sw = [Diagnostics.Stopwatch]::StartNew(); & $b; $sw.Stop(); $sw.Elapsed.TotalSeconds }
function BestGBps([string]$path, [long]$bytes, [int]$block, [bool]$unbuf, [string]$op, [int]$runs = 3) {
    $best = 0.0
    for ($i = 0; $i -lt $runs; $i++) {
        $s = if ($op -eq 'w') { [Bench]::Write($path, $bytes, $block, $unbuf) }
             else { [Bench]::Read($path, $bytes, $block, $unbuf) }
        $g = $bytes / $s / 1GB
        if ($g -gt $best) { $best = $g }
    }
    $best
}

function Measure-Target([string]$root, [string]$name) {
    $dir = Join-Path $root "vdbench"
    if ([IO.Directory]::Exists($dir)) { [IO.Directory]::Delete($dir, $true) }
    [IO.Directory]::CreateDirectory($dir) | Out-Null
    $file = Join-Path $dir "seq.dat"

    $io = @()
    foreach ($unbuf in @($true, $false)) {
        foreach ($b in @(4KB, 64KB, 1MB, 16MB)) {
            $io += [pscustomobject]@{
                Target = $name
                Mode   = if ($unbuf) { "unbuffered" } else { "buffered" }
                Block  = if ($b -ge 1MB) { "$($b/1MB) MiB" } else { "$($b/1KB) KiB" }
                WriteGBps = [math]::Round((BestGBps $file $FileBytes ([int]$b) $unbuf 'w'), 2)
                ReadGBps  = [math]::Round((BestGBps $file $FileBytes ([int]$b) $unbuf 'r'), 2)
            }
        }
    }
    $randUs = [math]::Round([Bench]::RandomReadUs($file, $FileBytes, 4096, 20000), 1)
    [IO.File]::Delete($file)

    # Small-file / metadata workload.
    $payload = New-Object byte[] 4096
    [Random]::new(5).NextBytes($payload)
    $create = Sec {
        for ($i = 0; $i -lt $SmallFiles; $i++) {
            $d = Join-Path $dir ("d{0}" -f ($i % 20))
            if ($i -lt 20) { [IO.Directory]::CreateDirectory($d) | Out-Null }
            [IO.File]::WriteAllBytes((Join-Path $d "f$i.bin"), $payload)
        }
    }
    $files = [IO.Directory]::GetFiles($dir, "*", [IO.SearchOption]::AllDirectories)
    $stat = Sec { foreach ($f in $files) { $null = (New-Object IO.FileInfo $f).Length } }
    $read = Sec { foreach ($f in $files) { $null = [IO.File]::ReadAllBytes($f) } }
    $del  = Sec { [IO.Directory]::Delete($dir, $true) }

    [pscustomobject]@{
        Name = $name; Io = $io; RandUs = $randUs
        Meta = [pscustomobject]@{
            Target = $name
            "create/s" = [math]::Round($SmallFiles / $create)
            "stat/s"   = [math]::Round($SmallFiles / $stat)
            "read/s"   = [math]::Round($SmallFiles / $read)
            "delete/s" = [math]::Round($SmallFiles / $del)
        }
    }
}

Write-Host "Measuring RAM disk at $RamDisk ..." -ForegroundColor Cyan
$ram = Measure-Target $RamDisk "RAM disk"

Write-Host "Mounting VRAMDISK ($Size) at $Drive ..." -ForegroundColor Cyan
# Start the mount with stdin redirected so it can be stopped *gracefully*.
# `Stop-Process -Force` is TerminateProcess: no cleanup runs, and a terminated
# WinFsp host can leave its volume device behind -- the drive letter then
# answers no I/O and cannot be reused until the machine reboots.
function Start-Mount([string]$exe, [string[]]$mountArgs) {
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    foreach ($a in $mountArgs) { [void]$psi.ArgumentList.Add($a) }
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    [Diagnostics.Process]::Start($psi)
}

# Ask the mount to unmount (the CLI stops on Enter), and only fall back to a
# hard kill if it will not go.
function Stop-Mount($proc) {
    if ($null -eq $proc -or $proc.HasExited) { return }
    try { $proc.StandardInput.WriteLine(); $proc.StandardInput.Flush() } catch { }
    if (-not $proc.WaitForExit(20000)) {
        Write-Host "  mount did not stop on request; forcing" -ForegroundColor Yellow
        try { $proc.Kill() } catch { }
    }
}
$proc = Start-Mount $Exe @('cli', '--mount', $Drive, '--size', $Size)
try {
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Path "$Drive\") { $ready = $true; break }
        if ($proc.HasExited) { throw "mount process exited early (code $($proc.ExitCode))" }
    }
    if (-not $ready) { throw "volume did not appear at $Drive" }
    Write-Host "Measuring VRAMDISK at $Drive ..." -ForegroundColor Cyan
    $vram = Measure-Target "$Drive\" "VRAMDISK"
} finally {
    Stop-Mount $proc
}

Write-Host "`n=== Sequential throughput (best of 3, $([math]::Round($FileBytes/1GB,1)) GiB file) ===" -ForegroundColor Cyan
($ram.Io + $vram.Io) | Sort-Object Mode, Block, Target | Format-Table -AutoSize

Write-Host "=== 4 KiB random read, unbuffered ===" -ForegroundColor Cyan
"  RAM disk {0,7:N1} us/op      VRAMDISK {1,7:N1} us/op" -f $ram.RandUs, $vram.RandUs

Write-Host "`n=== Metadata / small files ($SmallFiles x 4 KiB) ===" -ForegroundColor Cyan
@($ram.Meta, $vram.Meta) | Format-Table -AutoSize
