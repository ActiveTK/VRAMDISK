# Three-way benchmark: VRAMDISK vs GpuRamDrive vs an ordinary RAM disk.
#
# All three put a drive letter in front of memory, but they are three different
# designs and the numbers only mean something once that is spelled out:
#
#   RAM disk      system RAM behind a kernel block device (ImDisk), NTFS on top.
#   GpuRamDrive   GPU VRAM behind the *same* kernel block device, reached over
#                 PCIe by a user-mode proxy, NTFS on top.
#   VRAMDISK      GPU VRAM behind a user-mode file system (WinFsp). No NTFS --
#                 the file system itself lives in this process, so the GPU can
#                 also compute over the data in place.
#
# Every target is driven through the identical harness, so the columns are
# comparable. I/O is measured unbuffered (FILE_FLAG_NO_BUFFERING, i.e. the
# storage path itself) and buffered (what an ordinary application does, with the
# Windows cache manager in front), because all three sit behind the cache
# manager and measuring only one mode would say more about the cache than about
# the device.
#
# Needs an NVIDIA GPU, CUDA and WinFsp, plus the other two drives already
# mounted (both need elevation to create, which this script deliberately does
# not ask for).
#
#   pwsh -File scripts\bench_three_way.ps1 -RamDisk R:\ -GpuRamDrive P:\ [-Drive V:]
#
param(
    [Parameter(Mandatory = $true)][string]$RamDisk,
    [Parameter(Mandatory = $true)][string]$GpuRamDrive,
    [string]$Drive = "V:",
    [string]$Size = "4GiB",
    [long]$FileBytes = 2GB,
    [long]$ZipBytes = 512MB,
    [int]$SmallFiles = 4000,
    [string]$Report = "$PSScriptRoot\..\hikaku.md",
    [string]$Exe = "$PSScriptRoot\..\src-tauri\target\release\vramdisk.exe"
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Exe)) { throw "release binary not found at $Exe -- build it first" }
if (-not (Test-Path $RamDisk)) { throw "RAM disk not found at $RamDisk" }
if (-not (Test-Path $GpuRamDrive)) { throw "GpuRamDrive not found at $GpuRamDrive" }

Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading;
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
        var h = CreateFileW(path, write ? GENERIC_WRITE : GENERIC_READ, 0x3, IntPtr.Zero,
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

    // Each thread reads its own contiguous slice through its own handle, so this
    // measures how the device behaves when several readers are in flight -- the
    // case a single-threaded loop cannot see.
    public static double ParallelRead(string path, long total, int block, int threads) {
        long per = (total / threads) / block * block;
        var ts = new Thread[threads];
        Exception failure = null;
        var start = new ManualResetEventSlim(false);
        for (int t = 0; t < threads; t++) {
            long offset = (long)t * per;
            ts[t] = new Thread(() => {
                IntPtr buf = Alloc(block);
                try {
                    using (var h = Open(path, false, true)) {
                        SetFilePointerEx(h, offset, IntPtr.Zero, 0);
                        start.Wait();
                        for (long done = 0; done < per; ) {
                            uint take = (uint)Math.Min(block, per - done); uint r;
                            if (!ReadFile(h, buf, take, out r, IntPtr.Zero))
                                throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "ReadFile");
                            if (r == 0) break;
                            done += r;
                        }
                    }
                } catch (Exception e) { Interlocked.CompareExchange(ref failure, e, null); }
                finally { VirtualFree(buf, UIntPtr.Zero, MEM_RELEASE); }
            });
            ts[t].Start();
        }
        Thread.Sleep(50);
        var sw = Stopwatch.StartNew();
        start.Set();
        foreach (var th in ts) th.Join();
        sw.Stop();
        if (failure != null) throw failure;
        return sw.Elapsed.TotalSeconds;
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

    // The CPU-side equivalents of what VRAMDISK can do on the GPU, so the
    // "GPU-native operations" table compares like with like. Span<byte>.IndexOf
    // is vectorized by .NET -- roughly what a good grep does, not a naive loop.
    public static double ScanSecs(string path, byte[] needle, out long count) {
        var buf = new byte[8 << 20];
        long hits = 0; int overlap = needle.Length - 1;
        var sw = Stopwatch.StartNew();
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20, FileOptions.SequentialScan)) {
            int carry = 0;
            while (true) {
                int n = fs.Read(buf, carry, buf.Length - carry);
                if (n <= 0) break;
                int have = carry + n;
                var span = new ReadOnlySpan<byte>(buf, 0, have);
                int pos = 0;
                while (pos <= have - needle.Length) {
                    int at = span.Slice(pos).IndexOf(new ReadOnlySpan<byte>(needle));
                    if (at < 0) break;
                    hits++; pos += at + 1;
                }
                if (have >= overlap) { Array.Copy(buf, have - overlap, buf, 0, overlap); carry = overlap; }
                else carry = have;
            }
        }
        sw.Stop(); count = hits; return sw.Elapsed.TotalSeconds;
    }

    public static double HashSecs(string path, string alg) {
        var sw = Stopwatch.StartNew();
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20, FileOptions.SequentialScan))
        using (HashAlgorithm h = alg == "md5" ? (HashAlgorithm)MD5.Create() : SHA256.Create()) { h.ComputeHash(fs); }
        sw.Stop(); return sw.Elapsed.TotalSeconds;
    }
}
'@

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Sec([scriptblock]$b) { $sw = [Diagnostics.Stopwatch]::StartNew(); & $b; $sw.Stop(); $sw.Elapsed.TotalSeconds }

# Log-like ASCII: compressible, and realistic for both the search and the archive
# tests. Random bytes would make the first trivially fast and the second
# meaningless.
function New-LogCorpus([string]$path, [long]$bytes) {
    $sb = New-Object Text.StringBuilder
    for ($i = 0; $i -lt 4096; $i++) {
        [void]$sb.AppendLine("2026-09-06T00:00:00Z INFO  request id=$i path=/api/v1/items status=200 dur=12ms")
    }
    $block = [Text.Encoding]::ASCII.GetBytes($sb.ToString())
    $fs = [IO.File]::Create($path)
    try {
        $done = 0L
        while ($done -lt $bytes) {
            $take = [int][Math]::Min([long]$block.Length, $bytes - $done)
            $fs.Write($block, 0, $take); $done += $take
        }
    } finally { $fs.Close() }
}

function Measure-Zip([string]$src, [string]$dst) {
    if (Test-Path $dst) { [IO.File]::Delete($dst) }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $za = [IO.Compression.ZipFile]::Open($dst, [IO.Compression.ZipArchiveMode]::Create)
    try {
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $za, $src, (Split-Path $src -Leaf), [IO.Compression.CompressionLevel]::Optimal)
    } finally { $za.Dispose() }
    $sw.Stop(); $sw.Elapsed.TotalSeconds
}

function VramUsedMiB {
    [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim())
}
function HostUsedMiB {
    $os = Get-CimInstance Win32_OperatingSystem
    [int](($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1KB)
}

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

# --- VRAMDISK job control plane ------------------------------------------------
$script:jobSeq = 0
function Invoke-VramJob([string]$drive, [hashtable]$desc) {
    $script:jobSeq++
    $id = "bench{0}{1}" -f $PID, $script:jobSeq
    $pending = Join-Path $drive "`$VRAMDISK\jobs\pending\$id.json"
    $fs = [IO.File]::Open($pending, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes(($desc | ConvertTo-Json -Compress))
        $fs.Write($bytes, 0, $bytes.Length)
    } finally { $fs.Close() }

    $statusPath = Join-Path $drive "`$VRAMDISK\jobs\$id\status.json"
    $deadline = (Get-Date).AddSeconds(900)
    do {
        Start-Sleep -Milliseconds 10
        $status = [IO.File]::ReadAllText($statusPath) | ConvertFrom-Json
    } while (-not $status.terminal -and (Get-Date) -lt $deadline)
    $result = [IO.File]::ReadAllText((Join-Path $drive "`$VRAMDISK\jobs\$id\result.json")) | ConvertFrom-Json
    if ($status.state -ne 'succeeded') { throw "job $($status.state): $($result.error)" }
    $result
}

# Wall-clock, from submitting the descriptor to the terminal status -- i.e. what
# the user waits, queue latency included. Not every job kind reports its own
# elapsed_ms, and mixing the two would not be comparable.
function Time-VramJob([string]$drive, [hashtable]$desc) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-VramJob $drive $desc
    $sw.Stop()
    [pscustomobject]@{ Secs = $sw.Elapsed.TotalSeconds; Result = $r }
}

# --- one target ----------------------------------------------------------------
function Measure-Target([string]$root, [string]$name, [bool]$gpuNative) {
    $dir = Join-Path $root "vdbench"
    if ([IO.Directory]::Exists($dir)) { [IO.Directory]::Delete($dir, $true) }
    [IO.Directory]::CreateDirectory($dir) | Out-Null
    $file = Join-Path $dir "seq.dat"

    # Where does the payload actually land? Sample both memories around a write
    # of a known size; whichever one grows by ~FileBytes is holding the data.
    # Settle first: a RAM disk hands memory back when a file is deleted, but not
    # instantly, and a baseline taken while the previous target's bytes are still
    # held reads as "this drive costs nothing".
    Start-Sleep -Seconds 5
    $vram0 = VramUsedMiB; $host0 = HostUsedMiB
    [void][Bench]::Write($file, $FileBytes, 16MB, $true)
    Start-Sleep -Seconds 2
    $footprint = [pscustomobject]@{
        Target = $name
        "VRAM MiB" = (VramUsedMiB) - $vram0
        "host RAM MiB" = (HostUsedMiB) - $host0
    }

    $io = @()
    foreach ($unbuf in @($true, $false)) {
        foreach ($b in @(4KB, 64KB, 1MB, 16MB)) {
            $io += [pscustomobject]@{
                Target = $name
                Mode   = if ($unbuf) { "unbuffered" } else { "buffered" }
                Block  = if ($b -ge 1MB) { "{0} MiB" -f ($b / 1MB) } else { "{0} KiB" -f ($b / 1KB) }
                WriteGBps = [math]::Round((BestGBps $file $FileBytes ([int]$b) $unbuf 'w'), 2)
                ReadGBps  = [math]::Round((BestGBps $file $FileBytes ([int]$b) $unbuf 'r'), 2)
            }
        }
    }

    $lat = @()
    foreach ($b in @(4KB, 64KB, 1MB)) {
        $lat += [pscustomobject]@{
            Target = $name
            Block  = if ($b -ge 1MB) { "{0} MiB" -f ($b / 1MB) } else { "{0} KiB" -f ($b / 1KB) }
            "us/op" = [math]::Round([Bench]::RandomReadUs($file, $FileBytes, [int]$b, $(if ($b -ge 1MB) { 2000 } else { 20000 })), 1)
        }
    }

    $conc = @()
    foreach ($t in @(1, 2, 4, 8)) {
        # Each thread reads a whole number of 1 MiB blocks, so the bytes actually
        # moved are slightly under FileBytes; divide by what was really read.
        $moved = [long]([math]::Floor($FileBytes / $t / 1MB)) * 1MB * $t
        $best = 0.0
        for ($i = 0; $i -lt 3; $i++) {
            $s = [Bench]::ParallelRead($file, $FileBytes, 1MB, $t)
            $g = $moved / $s / 1GB
            if ($g -gt $best) { $best = $g }
        }
        $conc += [pscustomobject]@{ Target = $name; Threads = $t; ReadGBps = [math]::Round($best, 2) }
    }

    # Data-processing operations. VRAMDISK computes over the bytes where they
    # already live; the other two are plain block devices, so the same work means
    # pulling every byte across PCIe (GpuRamDrive) or out of RAM (RAM disk) into
    # the CPU first. Their column is therefore the best CPU implementation
    # available, not a strawman.
    $needle = [Text.Encoding]::ASCII.GetBytes("XXRAREMARKERXX")
    $fsW = [IO.File]::Open($file, [IO.FileMode]::Open, [IO.FileAccess]::Write)
    try { $fsW.Position = $fsW.Length - 4096; $fsW.Write($needle, 0, $needle.Length) } finally { $fsW.Close() }

    # A separate, compressible corpus for the archive test: zipping incompressible
    # random bytes measures nothing but the entropy coder's give-up path.
    $logFile = Join-Path $dir "log.txt"
    New-LogCorpus $logFile $ZipBytes
    $zipOut = Join-Path $dir "log.zip"

    $ops = @()
    if ($gpuNative) {
        $rel = "\vdbench\seq.dat"
        $best = [double]::MaxValue; $matches = 0; $engineMs = 0
        for ($i = 0; $i -lt 3; $i++) {
            $t = Time-VramJob $root @{ op = "search"; pattern = "XXRAREMARKERXX"; paths = @($rel); max_offsets = 4 }
            if ($t.Secs -lt $best) { $best = $t.Secs; $engineMs = $t.Result.elapsed_ms }
            $matches = $t.Result.total_matches
        }
        $ops += [pscustomobject]@{ Target = $name; Operation = "full-text search"; Bytes = $FileBytes; Seconds = [math]::Round($best, 3); GBps = [math]::Round($FileBytes / $best / 1GB, 2); Detail = "$matches hit(s); GPU kernel, engine $engineMs ms" }

        foreach ($alg in @("md5", "sha256")) {
            $best = [double]::MaxValue
            for ($i = 0; $i -lt 3; $i++) {
                $t = Time-VramJob $root @{ op = "hash"; algorithm = $alg; paths = @($rel) }
                if ($t.Secs -lt $best) { $best = $t.Secs }
            }
            $ops += [pscustomobject]@{ Target = $name; Operation = $alg.ToUpper(); Bytes = $FileBytes; Seconds = [math]::Round($best, 3); GBps = [math]::Round($FileBytes / $best / 1GB, 2); Detail = "job; calibration picks GPU or CPU" }
        }

        $best = [double]::MaxValue; $zlen = 0
        for ($i = 0; $i -lt 3; $i++) {
            if (Test-Path $zipOut) { [IO.File]::Delete($zipOut) }
            $t = Time-VramJob $root @{ op = "archive.compress"; format = "zip"; paths = @("\vdbench\log.txt"); output = "\vdbench\log.zip" }
            if ($t.Secs -lt $best) { $best = $t.Secs }
            $zlen = (New-Object IO.FileInfo $zipOut).Length
        }
        $ops += [pscustomobject]@{ Target = $name; Operation = "zip compress"; Bytes = $ZipBytes; Seconds = [math]::Round($best, 3); GBps = [math]::Round($ZipBytes / $best / 1GB, 2); Detail = "nvCOMP deflate on GPU, out {0:N0} MiB" -f ($zlen / 1MB) }
    } else {
        $best = [double]::MaxValue; $count = 0
        for ($i = 0; $i -lt 3; $i++) {
            $c = [ref][long]0
            $s = [Bench]::ScanSecs($file, $needle, $c)
            if ($s -lt $best) { $best = $s }
            $count = $c.Value
        }
        $ops += [pscustomobject]@{ Target = $name; Operation = "full-text search"; Bytes = $FileBytes; Seconds = [math]::Round($best, 3); GBps = [math]::Round($FileBytes / $best / 1GB, 2); Detail = "$count hit(s); vectorized CPU scan" }

        foreach ($alg in @("md5", "sha256")) {
            $best = [double]::MaxValue
            for ($i = 0; $i -lt 3; $i++) {
                $s = [Bench]::HashSecs($file, $alg)
                if ($s -lt $best) { $best = $s }
            }
            $ops += [pscustomobject]@{ Target = $name; Operation = $alg.ToUpper(); Bytes = $FileBytes; Seconds = [math]::Round($best, 3); GBps = [math]::Round($FileBytes / $best / 1GB, 2); Detail = ".NET, CPU" }
        }

        $best = [double]::MaxValue; $zlen = 0
        for ($i = 0; $i -lt 3; $i++) {
            $s = Measure-Zip $logFile $zipOut
            if ($s -lt $best) { $best = $s }
            $zlen = (New-Object IO.FileInfo $zipOut).Length
        }
        $ops += [pscustomobject]@{ Target = $name; Operation = "zip compress"; Bytes = $ZipBytes; Seconds = [math]::Round($best, 3); GBps = [math]::Round($ZipBytes / $best / 1GB, 2); Detail = ".NET deflate (Optimal) on CPU, out {0:N0} MiB" -f ($zlen / 1MB) }
    }
    [IO.File]::Delete($logFile)
    if (Test-Path $zipOut) { [IO.File]::Delete($zipOut) }

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
        Name = $name
        Io = $io
        Latency = $lat
        Concurrency = $conc
        Ops = $ops
        Footprint = $footprint
        Meta = [pscustomobject]@{
            Target = $name
            "create/s" = [math]::Round($SmallFiles / $create)
            "stat/s"   = [math]::Round($SmallFiles / $stat)
            "read/s"   = [math]::Round($SmallFiles / $read)
            "delete/s" = [math]::Round($SmallFiles / $del)
        }
    }
}

# --- mount control -------------------------------------------------------------
# Start the mount with stdin redirected so it can be stopped *gracefully*.
# `Stop-Process -Force` is TerminateProcess: no cleanup runs, and a terminated
# WinFsp host can leave its volume device behind -- the drive letter then answers
# no I/O and cannot be reused until the WinFsp driver is reloaded.
function Start-Mount([string]$exe, [string[]]$mountArgs) {
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    foreach ($a in $mountArgs) { [void]$psi.ArgumentList.Add($a) }
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    [Diagnostics.Process]::Start($psi)
}
function Stop-Mount($proc) {
    if ($null -eq $proc -or $proc.HasExited) { return }
    try { $proc.StandardInput.WriteLine(); $proc.StandardInput.Flush() } catch { }
    if (-not $proc.WaitForExit(20000)) {
        Write-Host "  mount did not stop on request; forcing" -ForegroundColor Yellow
        try { $proc.Kill() } catch { }
    }
}

# --- run -----------------------------------------------------------------------
Write-Host "Measuring RAM disk at $RamDisk ..." -ForegroundColor Cyan
$ram = Measure-Target $RamDisk "RAM disk" $false

Write-Host "Measuring GpuRamDrive at $GpuRamDrive ..." -ForegroundColor Cyan
$grd = Measure-Target $GpuRamDrive "GpuRamDrive" $false

Write-Host "Mounting VRAMDISK ($Size) at $Drive ..." -ForegroundColor Cyan
$mountVram0 = VramUsedMiB; $mountHost0 = HostUsedMiB
$proc = Start-Mount $Exe @('cli', '--mount', $Drive, '--size', $Size)
try {
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Path "$Drive\") { $ready = $true; break }
        if ($proc.HasExited) { throw "mount process exited early (code $($proc.ExitCode))" }
    }
    if (-not $ready) { throw "volume did not appear at $Drive" }
    Start-Sleep -Seconds 3
    $mountCost = [pscustomobject]@{
        Target = "VRAMDISK"
        "VRAM MiB at mount" = (VramUsedMiB) - $mountVram0
        "host RAM MiB at mount" = (HostUsedMiB) - $mountHost0
    }
    Write-Host "Measuring VRAMDISK at $Drive ..." -ForegroundColor Cyan
    $vram = Measure-Target "$Drive\" "VRAMDISK" $true
} finally {
    Stop-Mount $proc
}

$all = @($ram, $grd, $vram)
$order = @{ "RAM disk" = 0; "GpuRamDrive" = 1; "VRAMDISK" = 2 }
function ByTarget($rows) { $rows | Sort-Object { $order[$_.Target] } }

Write-Host "`n=== Sequential throughput (best of 3, $([math]::Round($FileBytes/1GB,1)) GiB file) ===" -ForegroundColor Cyan
($all.Io) | Sort-Object Mode, Block, { $order[$_.Target] } | Format-Table -AutoSize

Write-Host "=== Random read latency, unbuffered ===" -ForegroundColor Cyan
($all.Latency) | Sort-Object Block, { $order[$_.Target] } | Format-Table -AutoSize

Write-Host "=== Concurrent sequential read, 1 MiB blocks, unbuffered ===" -ForegroundColor Cyan
($all.Concurrency) | Sort-Object Threads, { $order[$_.Target] } | Format-Table -AutoSize

Write-Host "=== Data-processing operations (search/hash on $([math]::Round($FileBytes/1GB,1)) GiB, zip on $([math]::Round($ZipBytes/1MB)) MiB) ===" -ForegroundColor Cyan
($all.Ops) | Sort-Object Operation, { $order[$_.Target] } | Format-Table -AutoSize

Write-Host "=== Where the bytes live (delta while holding the file) ===" -ForegroundColor Cyan
(ByTarget $all.Footprint) | Format-Table -AutoSize
Write-Host "=== Reserved up front, before a single byte is written ===" -ForegroundColor Cyan
$mountCost | Format-Table -AutoSize

Write-Host "=== Metadata / small files ($SmallFiles x 4 KiB) ===" -ForegroundColor Cyan
(ByTarget $all.Meta) | Format-Table -AutoSize

# --- machine-readable dump, so the report can be written from real numbers -----
$dump = [pscustomobject]@{
    fileBytes = $FileBytes
    zipBytes = $ZipBytes
    smallFiles = $SmallFiles
    io = $all.Io
    latency = $all.Latency
    concurrency = $all.Concurrency
    ops = $all.Ops
    footprint = $all.Footprint
    mountCost = $mountCost
    meta = $all.Meta
}
$json = Join-Path (Split-Path $Report -Parent) ".bench_three_way.json"
$dump | ConvertTo-Json -Depth 6 | Out-File -FilePath $json -Encoding utf8
Write-Host "raw results written to $json" -ForegroundColor DarkGray
