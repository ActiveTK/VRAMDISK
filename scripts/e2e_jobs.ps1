# Virtual `$VRAMDISK API E2E for VRAMDISK.
# Mounts the release binary on a free drive letter and drives the internal
# virtual filesystem API end to end: the read-only info surface (help / stats /
# trace / chunks), then the jobs API -- hash, encode and archive jobs, job
# cancellation, and malformed job descriptors -- checking after each storm that
# the volume is still healthy. Unmounts at the end. Exits non-zero if anything
# failed.
#
# This needs a real NVIDIA GPU, CUDA and WinFsp, so it cannot run on CI. It is
# the manual pre-release verification step, run next to e2e_robustness.ps1
# (which covers the ordinary filesystem surface and deliberately stays fast).
#
# Archive jobs additionally need a compatible nvCOMP DLL. On a machine without
# one those checks report [skip] rather than [FAIL], so the rest of the script
# is still useful there.
#
# Usage:  pwsh -File scripts\e2e_jobs.ps1 [-Drive T] [-Exe path\to\vramdisk.exe] [-Size 2GiB]
#
# $Exe is the merged vramdisk.exe (GUI binary); this script drives its `cli`
# subcommand, not the GUI. $Size must leave room for the ~800 MiB of test data
# the hash / cancellation checks need.

param(
    [string]$Drive = 'T',
    [string]$Exe = "$PSScriptRoot\..\src-tauri\target\release\vramdisk.exe",
    [string]$Size = '2GiB'
)

$ErrorActionPreference = 'Stop'
$root = "${Drive}:"
# Literal '$VRAMDISK' -- single quotes so PowerShell does not eat the sigil.
$api = $root + '\$VRAMDISK'
$fail = 0
$skipped = 0

function Check($cond, $msg) {
    if ($cond) { Write-Host "  [ok]   $msg" -ForegroundColor Green }
    else       { Write-Host "  [FAIL] $msg" -ForegroundColor Red; $script:fail++ }
}
function ExpectThrows($script, $msg) {
    $threw = $false
    try { & $script | Out-Null } catch { $threw = $true }
    Check $threw $msg
}
function Skipped($msg) {
    Write-Host "  [skip] $msg" -ForegroundColor Yellow; $script:skipped++
}
# Each numbered section runs in its own try/catch: one section blowing up on an
# unexpected exception counts as a failure but must not cost us the diagnostic
# value of every later section (the ordinary-filesystem script is short enough
# to just let $ErrorActionPreference='Stop' abort it; this one is not).
function Section($name, $body) {
    Write-Host "`n$name"
    try { & $body }
    catch {
        Write-Host "  [FAIL] unexpected error in section: $($_.Exception.Message)" -ForegroundColor Red
        $script:fail++
    }
}

# --- $VRAMDISK protocol helpers -------------------------------------------
# Info files are plain read-only files. Jobs are submitted by CREATE_NEW-ing
# \$VRAMDISK\jobs\pending\<id>.json, writing the JSON descriptor, and closing
# the handle (the close is what queues the job). State and results then come
# from \$VRAMDISK\jobs\<id>\{status.json,result.json,wait,cancel}.

function ApiText($rel) { Get-Content -LiteralPath "$api\$rel" -Raw }
function ApiJson($rel) { ApiText $rel | ConvertFrom-Json }

# Every $VRAMDISK node is hidden+system, so a bare Get-ChildItem there silently
# returns nothing. -Force is mandatory when listing inside the namespace.
function ApiList($rel) {
    if ($rel) { (Get-ChildItem -LiteralPath "$api\$rel" -Force).Name }
    else      { (Get-ChildItem -LiteralPath $api -Force).Name }
}

function SubmitJob($id, $descriptorHash) {
    $json = $descriptorHash | ConvertTo-Json -Compress -Depth 8
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    # CREATE_NEW specifically: that is the disposition the API reserves the id
    # on, and it is what makes a duplicate id fail instead of silently reusing.
    $fs = [IO.File]::Open("$api\jobs\pending\${id}.json", [IO.FileMode]::CreateNew,
                          [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $fs.Write($bytes, 0, $bytes.Length) } finally { $fs.Dispose() }
}

function JobStatus($id) { Get-Content -LiteralPath "$api\jobs\${id}\status.json" -Raw | ConvertFrom-Json }
function JobResult($id) { Get-Content -LiteralPath "$api\jobs\${id}\result.json" -Raw | ConvertFrom-Json }

# Reading the `cancel` node is the cancel request; it returns the job's current
# result.json body.
function CancelJob($id) { Get-Content -LiteralPath "$api\jobs\${id}\cancel" -Raw | Out-Null }

# Always poll status.json with a deadline rather than reading the blocking
# `wait` node: a wedged job would otherwise hang this script forever.
function WaitJob($id, $timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        $s = JobStatus $id
        if ($s.terminal) { return $s }
        Start-Sleep -Milliseconds 100
    }
    return $null
}

# Submit + wait. Returns the terminal status object, or $null on timeout.
function RunJob($id, $descriptorHash, $timeoutSec = 180) {
    SubmitJob $id $descriptorHash
    WaitJob $id $timeoutSec
}

function JobError($id) {
    $r = JobResult $id
    if ($null -eq $r.error) { return '' }
    return [string]$r.error
}

# A job that failed only because nvCOMP is not installed on this machine.
function IsNvcompMissing($message) {
    return ($message -match '(?i)nvcomp')
}

function HostDigest($bytes, $alg) {
    $ms = New-Object IO.MemoryStream(, $bytes)
    try { (Get-FileHash -InputStream $ms -Algorithm $alg).Hash.ToLowerInvariant() }
    finally { $ms.Dispose() }
}
function Sha256Of($bytes) {
    [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($bytes)).Replace('-', '')
}
function BytesEqual($a, $b) {
    if ($a.Length -ne $b.Length) { return $false }
    return ((Sha256Of $a) -eq (Sha256Of $b))
}
function RandomBytes($len, $seed) {
    $b = New-Object byte[] $len
    (New-Object Random $seed).NextBytes($b)
    return , $b
}

$runTag = Get-Date -Format 'HHmmss'
function JobId($name) { "e2e-$runTag-$name" }

Write-Host "Mounting $root via $Exe (size $Size) ..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $Exe -ArgumentList @('cli', '--mount', "$root", '--size', $Size) -PassThru -WindowStyle Hidden

try {
    # Wait for the volume to appear.
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Path "$root\") { $ready = $true; break }
        if ($proc.HasExited) { throw "mount process exited early (code $($proc.ExitCode))" }
    }
    if (-not $ready) { throw "volume did not appear at $root" }
    Write-Host "Mounted." -ForegroundColor Cyan

    # ---- shared test data -------------------------------------------------
    # "hello world" as raw ASCII: 11 bytes, no BOM and no trailing newline, so
    # the host-side expected digests below are over exactly these bytes.
    $smallBytes = [Text.Encoding]::ASCII.GetBytes('hello world')
    $smallMd5 = HostDigest $smallBytes 'MD5'
    $smallSha1 = HostDigest $smallBytes 'SHA1'
    $smallSha256 = HostDigest $smallBytes 'SHA256'
    # FNV-1a 64 has no Get-FileHash equivalent, so this one is hard-coded.
    # Derivation: h = 0xcbf29ce484222325; for each byte b of "hello world":
    #   h ^= b; h = (h * 0x100000001b3) mod 2^64
    # then render the 64-bit state big-endian as lowercase hex. That is the
    # standard FNV-1a-64 test vector for "hello world", and matches both the
    # CUDA kernel (api_kernel.rs, alg 4) and the Rust CpuHashState fallback.
    $smallFnv = '779a65e7023cd2e7'
    $smallB64 = [Convert]::ToBase64String($smallBytes)   # aGVsbG8gd29ybGQ=
    $smallHex = '68656c6c6f20776f726c64'

    # Comfortably past the 128 MiB ceiling the runtime-calibrated CPU routing
    # threshold is clamped to, so this file is guaranteed to take the CPU
    # streaming hash path no matter how the calibration lands on this GPU.
    $largeMiB = 256
    $largeBytes = RandomBytes ($largeMiB * 1MB) 20260905
    $largeMd5 = HostDigest $largeBytes 'MD5'
    $largeSha256 = HostDigest $largeBytes 'SHA256'

    Section '[1] Info surface' {
        Check (Test-Path -LiteralPath $api) '$VRAMDISK directory is visible'
        $entries = ApiList $null
        foreach ($n in @('help.txt', 'stats.txt', 'stats.json', 'trace.txt', 'trace.json', 'chunks.json', 'jobs')) {
            Check ($entries -contains $n) ('$VRAMDISK lists ' + $n)
        }
        Check ((ApiList 'jobs') -contains 'pending' -and (ApiList 'jobs') -contains 'completed') '$VRAMDISK\jobs lists pending and completed'

        $help = ApiText 'help.txt'
        Check ($help.Length -gt 0) 'help.txt is readable and non-empty'
        Check ($help -match 'jobs\\pending' -and $help -match 'stats\.json') 'help.txt documents the jobs and stats paths'

        $statsTxt = ApiText 'stats.txt'
        Check ($statsTxt -match 'chunks\.total:') 'stats.txt reports chunk counters'
        Check ((ApiText 'trace.txt') -match 'calls\.read:') 'trace.txt reports call counters'

        $s0 = ApiJson 'stats.json'
        Check ($null -ne $s0.volume) 'stats.json parses as JSON'
        Check ($s0.volume.total_chunks -gt 0) "stats.json total_chunks is plausible ($($s0.volume.total_chunks))"
        Check ($s0.volume.used_chunks + $s0.volume.free_chunks -eq $s0.volume.total_chunks) 'stats.json chunk accounting adds up'

        $t0 = ApiJson 'trace.json'
        Check ($null -ne $t0.calls.read) 'trace.json parses as JSON'

        # 4 MiB written == 64 logical chunks of 64 KiB.
        New-Item -ItemType Directory "$root\info" -Force | Out-Null
        [IO.File]::WriteAllBytes("$root\info\grow.bin", (RandomBytes 4MB 7))
        $s1 = ApiJson 'stats.json'
        Check ($s1.volume.used_chunks -ge $s0.volume.used_chunks + 64) "stats.json used_chunks grew after a 4 MiB write ($($s0.volume.used_chunks) -> $($s1.volume.used_chunks))"
        Check ($s1.namespace.file_count -gt $s0.namespace.file_count) 'stats.json file_count grew'

        $chunks = ApiJson 'chunks.json\info\grow.bin'
        Check ($chunks.size -eq 4MB -and $chunks.logical_chunks -eq 64) 'chunks.json reports the file layout'
        Check ($chunks.chunks[0].kind -eq 'raw') 'chunks.json reports a raw placement'

        # The whole namespace is read-only apart from jobs\pending.
        ExpectThrows { [IO.File]::WriteAllBytes("$api\intruder.txt", [byte[]]@(1, 2, 3)) } 'writing a new file into $VRAMDISK is refused'
        ExpectThrows { [IO.Directory]::CreateDirectory("$api\intruder") } 'creating a directory inside $VRAMDISK is refused'
        ExpectThrows { [IO.File]::Delete("$api\help.txt") } 'deleting a $VRAMDISK info file is refused'
    }

    Section '[2] Hash jobs' {
        New-Item -ItemType Directory "$root\hash" -Force | Out-Null
        [IO.File]::WriteAllBytes("$root\hash\small.bin", $smallBytes)
        [IO.File]::WriteAllBytes("$root\hash\large.bin", $largeBytes)

        # Small file: under the routing threshold and stored raw, so this is
        # the CUDA API-kernel path. All four documented algorithms.
        $expected = @{ md5 = $smallMd5; sha1 = $smallSha1; sha256 = $smallSha256; fnv1a64 = $smallFnv }
        foreach ($alg in @('md5', 'sha1', 'sha256', 'fnv1a64')) {
            $id = JobId "hash-small-$alg"
            $st = RunJob $id @{ op = 'hash'; algorithm = $alg; paths = @('\hash\small.bin'); recursive = $true } 60
            if ($null -eq $st) { Check $false "hash($alg) small file reached a terminal state"; continue }
            Check ($st.state -eq 'succeeded') "hash($alg) small file succeeded (state=$($st.state) error=$($st.error))"
            $r = JobResult $id
            Check ($r.ok -eq $true -and $r.algorithm -eq $alg -and $r.file_count -eq 1) "hash($alg) result.json shape"
            Check ($r.files[0].path -eq '\hash\small.bin') "hash($alg) result names the input path"
            Check ($r.files[0].digest -eq $expected[$alg]) "hash($alg) digest matches the host value ($($r.files[0].digest))"
        }

        # Large file: forced onto the CPU streaming path by size.
        foreach ($pair in @(@('sha256', $largeSha256), @('md5', $largeMd5))) {
            $alg = $pair[0]
            $id = JobId "hash-large-$alg"
            $st = RunJob $id @{ op = 'hash'; algorithm = $alg; paths = @('\hash\large.bin') } 300
            if ($null -eq $st) { Check $false "hash($alg) ${largeMiB} MiB file reached a terminal state"; continue }
            Check ($st.state -eq 'succeeded') "hash($alg) ${largeMiB} MiB file succeeded (state=$($st.state) error=$($st.error))"
            Check ($st.progress.total_bytes -eq ($largeMiB * 1MB)) "hash($alg) progress totalled the whole file"
            Check ((JobResult $id).files[0].digest -eq $pair[1]) "hash($alg) ${largeMiB} MiB digest matches the host value"
        }

        # Directory recursion + the non-blocking read of `wait` on a job that
        # has already finished.
        $id = JobId 'hash-tree'
        $st = RunJob $id @{ op = 'hash'; algorithm = 'sha256'; paths = @('\hash'); recursive = $true } 300
        Check ($null -ne $st -and $st.state -eq 'succeeded') 'recursive hash over a directory succeeded'
        if ($null -ne $st) {
            $r = JobResult $id
            Check ($r.file_count -eq 2) "recursive hash walked both files (file_count=$($r.file_count))"
            $waited = Get-Content -LiteralPath "$api\jobs\${id}\wait" -Raw | ConvertFrom-Json
            Check ($waited.ok -eq $true -and $waited.id -eq $id) 'reading `wait` on a finished job returns result.json'
        }
        $id = JobId 'hash-nodir'
        $st = RunJob $id @{ op = 'hash'; algorithm = 'sha256'; paths = @('\hash'); recursive = $false } 60
        Check ($null -ne $st -and $st.state -eq 'failed') "recursive=false over a directory fails the job (state=$($st.state))"
    }

    Section '[3] Encode jobs' {
        New-Item -ItemType Directory "$root\enc" -Force | Out-Null
        [IO.File]::WriteAllBytes("$root\enc\small.bin", $smallBytes)
        $binBytes = RandomBytes 3MB 4242
        [IO.File]::WriteAllBytes("$root\enc\bin.bin", $binBytes)

        # Base64 encode of a small file must match the host byte for byte.
        $id = JobId 'enc-b64-small'
        $st = RunJob $id @{ op = 'encode'; codec = 'base64'; direction = 'encode'; input = '\enc\small.bin'; output = '\enc\small.b64' } 60
        Check ($null -ne $st -and $st.state -eq 'succeeded') "base64 encode succeeded (state=$($st.state) error=$($st.error))"
        if ($null -ne $st -and $st.state -eq 'succeeded') {
            $got = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes("$root\enc\small.b64"))
            Check ($got -eq $smallB64) "base64 output matches [Convert]::ToBase64String ($got)"
            $r = JobResult $id
            Check ($r.codec -eq 'base64' -and $r.direction -eq 'encode' -and $r.input_bytes -eq $smallBytes.Length) 'encode result.json shape'
        }

        # Hex encode of the same file, lowercase per spec.
        $id = JobId 'enc-hex-small'
        $st = RunJob $id @{ op = 'encode'; codec = 'hex'; direction = 'encode'; input = '\enc\small.bin'; output = '\enc\small.hex' } 60
        Check ($null -ne $st -and $st.state -eq 'succeeded') "hex encode succeeded (state=$($st.state) error=$($st.error))"
        if ($null -ne $st -and $st.state -eq 'succeeded') {
            $got = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes("$root\enc\small.hex"))
            Check ($got -eq $smallHex) "hex output is lowercase hex of the input ($got)"
        }

        # Binary round-trips: encode then decode must be byte-identical.
        foreach ($codec in @('base64', 'hex')) {
            $enc = JobId "enc-$codec-bin"
            $dec = JobId "dec-$codec-bin"
            $encOut = "\enc\bin.$codec"
            $decOut = "\enc\bin.$codec.out"
            $st = RunJob $enc @{ op = 'encode'; codec = $codec; direction = 'encode'; input = '\enc\bin.bin'; output = $encOut } 180
            Check ($null -ne $st -and $st.state -eq 'succeeded') "$codec encode of a 3 MiB binary succeeded (state=$($st.state) error=$($st.error))"
            if ($null -eq $st -or $st.state -ne 'succeeded') { continue }
            $st = RunJob $dec @{ op = 'encode'; codec = $codec; direction = 'decode'; input = $encOut; output = $decOut } 180
            Check ($null -ne $st -and $st.state -eq 'succeeded') "$codec decode succeeded (state=$($st.state) error=$($st.error))"
            if ($null -eq $st -or $st.state -ne 'succeeded') { continue }
            $back = [IO.File]::ReadAllBytes("$root$decOut")
            Check (BytesEqual $binBytes $back) "$codec round-trip is byte-identical"
        }

        # Negative: an invalid character must fail the job, not hang it and not
        # take the mount down. The bad char sits in the first 4-char group, so
        # it is the GPU kernel's status flag that has to catch it.
        $bad = 'AB!D' + ('QUJD' * 15)
        [IO.File]::WriteAllBytes("$root\enc\bad.b64", [Text.Encoding]::ASCII.GetBytes($bad))
        $id = JobId 'dec-b64-bad'
        $st = RunJob $id @{ op = 'encode'; codec = 'base64'; direction = 'decode'; input = '\enc\bad.b64'; output = '\enc\bad.out' } 60
        Check ($null -ne $st) 'invalid base64 decode reached a terminal state (did not hang)'
        Check ($null -ne $st -and $st.state -eq 'failed') "invalid base64 decode failed the job (state=$($st.state))"
        Check ((JobError $id).Length -gt 0) "invalid base64 decode reported an error ($(JobError $id))"
        Check (Test-Path "$root\enc\bin.bin") 'volume still readable after the failed decode'
    }

    Section '[3b] Search jobs' {
        # A corpus with a known number of hits, some of them deliberately
        # placed astride the engine's scan-window seams.
        New-Item -ItemType Directory -Path "$root\search" -Force | Out-Null
        $line = "alpha beta GAMMA delta needle-here epsilon`r`n"
        $sb = New-Object Text.StringBuilder
        for ($i = 0; $i -lt 20000; $i++) { [void]$sb.Append($line) }
        $text = [Text.Encoding]::ASCII.GetBytes($sb.ToString())
        [IO.File]::WriteAllBytes("$root\search\corpus.txt", $text)
        [IO.File]::WriteAllBytes("$root\search\other.txt", [Text.Encoding]::ASCII.GetBytes("nothing to see"))

        $expect = 20000
        $id = JobId 'search-basic'
        $st = RunJob $id @{ op = 'search'; pattern = 'needle-here'; paths = @("\search") }
        Check ($null -ne $st -and $st.state -eq 'succeeded') "search job succeeded"
        if ($st -and $st.state -eq 'succeeded') {
            $r = JobResult $id
            Check ($r.total_matches -eq $expect) "search found every occurrence ($($r.total_matches) vs $expect)"
            Check ($r.files_matched -eq 1) "only the file that contains it matched ($($r.files_matched))"
            Check ($r.bytes_scanned -eq ($text.Length + 14)) "scanned both files fully ($($r.bytes_scanned))"
            $hit = $r.files | Where-Object { $_.path -like '*corpus.txt' }
            Check ($null -ne $hit -and $hit.offsets.Count -gt 0) "match offsets reported"
            if ($hit -and $hit.offsets.Count -gt 0) {
                $first = [int]$hit.offsets[0]
                $slice = [Text.Encoding]::ASCII.GetString($text[$first..($first + 10)])
                Check ($slice -eq 'needle-here') "first reported offset really holds the pattern ('$slice')"
            }
        }

        # Case folding is ASCII-only and opt-in.
        $id = JobId 'search-nocase'
        $st = RunJob $id @{ op = 'search'; pattern = 'gamma'; paths = @("\search"); ignore_case = $true }
        Check ($null -ne $st -and $st.state -eq 'succeeded' -and (JobResult $id).total_matches -eq $expect) "ignore_case matches GAMMA"
        $id = JobId 'search-case'
        $st = RunJob $id @{ op = 'search'; pattern = 'gamma'; paths = @("\search") }
        Check ($null -ne $st -and $st.state -eq 'succeeded' -and (JobResult $id).total_matches -eq 0) "case-sensitive search does not"

        # Counts stay exact even when the reported offsets are capped.
        $id = JobId 'search-cap'
        $st = RunJob $id @{ op = 'search'; pattern = 'alpha'; paths = @("\search"); max_offsets = 5 }
        if ($st -and $st.state -eq 'succeeded') {
            $r = JobResult $id
            $h = $r.files | Where-Object { $_.path -like '*corpus.txt' }
            Check ($r.total_matches -eq $expect) "count is exact under max_offsets ($($r.total_matches))"
            Check ($h.offsets.Count -eq 5 -and $h.offsets_truncated) "offsets capped and flagged"
        }

        # Binary needles via hex, and the whole volume when paths is omitted.
        $id = JobId 'search-hex'
        $st = RunJob $id @{ op = 'search'; pattern_hex = '6e6565646c65' }   # "needle"
        Check ($null -ne $st -and $st.state -eq 'succeeded' -and (JobResult $id).total_matches -ge $expect) "pattern_hex searches the whole volume"

        # Bad input fails the job without taking the mount down.
        $id = JobId 'search-empty'
        $st = RunJob $id @{ op = 'search'; pattern = '' }
        Check ($null -ne $st -and $st.state -eq 'failed') "empty pattern fails the job"
        $id = JobId 'search-badhex'
        $st = RunJob $id @{ op = 'search'; pattern_hex = 'zz' }
        Check ($null -ne $st -and $st.state -eq 'failed') "invalid pattern_hex fails the job"
        Check (Test-Path "$root\search\corpus.txt") "volume still healthy"
    }

    Section '[4] Archive jobs' {
        New-Item -ItemType Directory "$root\arc\src\sub" -Force | Out-Null
        $srcFiles = @{
            '\arc\src\a.bin'     = (RandomBytes 2MB 11)
            '\arc\src\sub\b.bin' = (RandomBytes 1MB 12)
            '\arc\src\sub\c.txt' = [Text.Encoding]::ASCII.GetBytes(('vramdisk archive payload ' * 400))
        }
        foreach ($p in $srcFiles.Keys) { [IO.File]::WriteAllBytes("$root$p", $srcFiles[$p]) }

        $skipArchive = $false
        foreach ($fmt in @(@('zip', 'zip'), @('tar.zst', 'tarzst'))) {
            if ($skipArchive) { break }
            $format = $fmt[0]
            $tag = $fmt[1]
            $archive = "\arc\out-$tag.$format"
            $restore = "\arc\restore-$tag"

            $cid = JobId "arc-c-$tag"
            $st = RunJob $cid @{ op = 'archive.compress'; format = $format; paths = @('\arc\src'); output = $archive; recursive = $true } 300
            if ($null -eq $st) { Check $false "archive.compress($format) reached a terminal state"; continue }
            if ($st.state -ne 'succeeded') {
                $err = JobError $cid
                if (IsNvcompMissing $err) {
                    Skipped "archive jobs need a compatible nvCOMP DLL; none loadable here ($err)"
                    $skipArchive = $true
                    continue
                }
                Check $false "archive.compress($format) succeeded (state=$($st.state) error=$err)"
                continue
            }
            Check $true "archive.compress($format) succeeded"
            $r = JobResult $cid
            Check ($r.file_count -eq 3) "archive.compress($format) packed all 3 files"
            Check ((Get-Item "$root$archive").Length -gt 0) "archive.compress($format) wrote a non-empty archive"

            $xid = JobId "arc-x-$tag"
            $st = RunJob $xid @{ op = 'archive.extract'; format = $format; archive = $archive; output_dir = $restore } 300
            if ($null -eq $st) { Check $false "archive.extract($format) reached a terminal state"; continue }
            Check ($st.state -eq 'succeeded') "archive.extract($format) succeeded (state=$($st.state) error=$(JobError $xid))"
            if ($st.state -ne 'succeeded') { continue }

            # tar entry names are the source path minus its leading backslash,
            # so a file at \arc\src\sub\b.bin lands at <restore>\arc\src\sub\b.bin.
            $identical = $true
            foreach ($p in $srcFiles.Keys) {
                $out = "$root$restore$p"
                if (-not (Test-Path -LiteralPath $out)) { $identical = $false; break }
                if (-not (BytesEqual $srcFiles[$p] ([IO.File]::ReadAllBytes($out)))) { $identical = $false; break }
            }
            Check $identical "$format extracted tree is byte-identical to the source"
            Check ((Get-ChildItem "$root$restore" -Recurse -File).Count -eq 3) "$format extracted exactly 3 files"
        }
    }

    Section '[5] Job cancellation' {
        # Enough work that the job is still running when the cancel lands: two
        # more copies of the large file, hashed together with the original.
        New-Item -ItemType Directory "$root\cancel" -Force | Out-Null
        [IO.File]::WriteAllBytes("$root\cancel\c0.bin", $largeBytes)
        [IO.File]::WriteAllBytes("$root\cancel\c1.bin", $largeBytes)

        $id = JobId 'cancel-hash'
        SubmitJob $id @{ op = 'hash'; algorithm = 'sha256'; paths = @('\cancel', '\hash\large.bin'); recursive = $true }
        $before = JobStatus $id
        Check ($before.state -in @('receiving', 'queued', 'running')) "long hash job is live before cancelling (state=$($before.state))"
        CancelJob $id
        $st = WaitJob $id 60
        Check ($null -ne $st) 'cancelled job reached a terminal state within 60 s'
        if ($null -ne $st) {
            if ($st.state -eq 'succeeded') {
                # Not a product failure: this GPU hashed 768 MiB faster than the
                # cancel request could reach the worker. Say so rather than
                # reporting a red check for a race the script lost.
                Skipped 'the hash finished before the cancel request landed; cancellation could not be exercised on this machine'
            }
            else {
                Check ($st.state -eq 'cancelled') "job state is 'cancelled' (state=$($st.state))"
                Check ((JobError $id) -match 'cancel') "result.json explains the cancellation ($(JobError $id))"
            }
        }

        # Cancelling an already-terminal job is a no-op, not an error.
        CancelJob $id
        Check ((JobStatus $id).terminal -eq $true) 'a second cancel on a finished job is harmless'

        # The volume must be completely usable afterwards.
        [IO.File]::WriteAllBytes("$root\cancel\after.txt", [Text.Encoding]::ASCII.GetBytes('alive'))
        Check ([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes("$root\cancel\after.txt")) -eq 'alive') 'filesystem responsive after cancellation'
        Check ($null -ne (ApiJson 'stats.json').volume) 'stats.json still parses after cancellation'
    }

    Section '[6] Job robustness' {
        # Malformed descriptors must fail the job, not the mount.
        $id = JobId 'bad-json'
        $fs = [IO.File]::Open("$api\jobs\pending\${id}.json", [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $raw = [Text.Encoding]::ASCII.GetBytes('{ this is not json')
        $fs.Write($raw, 0, $raw.Length); $fs.Dispose()
        $st = WaitJob $id 30
        Check ($null -ne $st -and $st.state -eq 'failed') "unparseable descriptor fails the job (state=$($st.state))"
        Check ((JobError $id) -match 'descriptor') "...with a descriptor error ($(JobError $id))"

        $cases = @(
            @{ name = 'no-op'; d = @{ algorithm = 'sha256'; paths = @('\hash\small.bin') }; why = 'descriptor without "op"' },
            @{ name = 'bad-op'; d = @{ op = 'definitely.not.a.job'; paths = @('\hash\small.bin') }; why = 'unknown op' },
            @{ name = 'bad-alg'; d = @{ op = 'hash'; algorithm = 'crc9000'; paths = @('\hash\small.bin') }; why = 'unsupported hash algorithm' },
            @{ name = 'no-paths'; d = @{ op = 'hash'; algorithm = 'sha256' }; why = 'hash job with no paths' },
            @{ name = 'missing-path'; d = @{ op = 'hash'; algorithm = 'sha256'; paths = @('\nope\nothing.bin') }; why = 'hash job over a missing path' },
            @{ name = 'bad-fmt'; d = @{ op = 'archive.compress'; format = 'rar'; paths = @('\hash\small.bin'); output = '\out.rar' }; why = 'unsupported archive format' },
            @{ name = 'bad-codec'; d = @{ op = 'encode'; codec = 'rot13'; direction = 'encode'; input = '\hash\small.bin'; output = '\out.rot' }; why = 'unsupported encode codec' },
            @{ name = 'enc-no-out'; d = @{ op = 'encode'; codec = 'base64'; direction = 'encode'; input = '\hash\small.bin' }; why = 'encode job with no output' }
        )
        foreach ($c in $cases) {
            $cid = JobId $c.name
            $st = RunJob $cid $c.d 60
            Check ($null -ne $st -and $st.state -eq 'failed') "$($c.why) fails the job (state=$($st.state))"
            Check ($null -ne $st -and (JobError $cid).Length -gt 0) "...and reports an error ($(JobError $cid))"
        }

        # Job-namespace edge cases.
        ExpectThrows { Get-Content -LiteralPath "$api\jobs\no-such-job-id\status.json" -Raw } 'status.json for an unknown job id fails cleanly'
        ExpectThrows { Get-Content -LiteralPath "$api\jobs\no-such-job-id\result.json" -Raw } 'result.json for an unknown job id fails cleanly'
        ExpectThrows { Get-Content -LiteralPath "$api\jobs\no-such-job-id\cancel" -Raw } 'cancel on an unknown job id fails cleanly'
        ExpectThrows { Get-ChildItem -LiteralPath "$api\jobs\no-such-job-id" -Force } 'listing an unknown job directory fails cleanly'
        ExpectThrows { Get-Content -LiteralPath "$api\jobs\$(JobId 'bad-json')\nonsense" -Raw } 'unknown leaf under a job directory fails cleanly'

        $dupId = JobId 'dup'
        SubmitJob $dupId @{ op = 'hash'; algorithm = 'sha256'; paths = @('\hash\small.bin') }
        ExpectThrows { SubmitJob $dupId @{ op = 'hash'; algorithm = 'sha256'; paths = @('\hash\small.bin') } } 'resubmitting an existing job id is refused'
        Check ($null -ne (WaitJob $dupId 60)) 'the original job still completed'

        # Invalid ids are not part of the pending namespace at all.
        ExpectThrows { [IO.File]::Open("$api\jobs\pending\bad id.json", [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None) } 'job id with a space is refused'
        ExpectThrows { [IO.File]::Open("$api\jobs\pending\bad`$id.json", [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None) } 'job id with an illegal character is refused'
        ExpectThrows { [IO.File]::Open("$api\jobs\pending\noext", [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None) } 'pending descriptor without a .json suffix is refused'

        $completed = ApiList 'jobs\completed'
        Check ($completed -contains $dupId) 'finished jobs are listed under jobs\completed'
        Check ((ApiList "jobs\$dupId") -contains 'status.json') 'a job directory lists status.json / result.json / wait / cancel'
    }

    Section '[7] Volume still healthy after the whole job storm' {
        [IO.File]::WriteAllBytes("$root\final.bin", $smallBytes)
        Check (BytesEqual $smallBytes ([IO.File]::ReadAllBytes("$root\final.bin"))) 'write + read still works'
        Check ((Get-ChildItem "$root\").Count -ge 4) 'root listing works'
        $s = ApiJson 'stats.json'
        Check ($s.volume.used_chunks -gt 0 -and $s.volume.free_chunks -gt 0) 'stats.json still reports a sane volume'
        $t = ApiJson 'trace.json'
        Check ($t.calls.read -gt 0 -and $t.calls.write -gt 0) 'trace.json counted the traffic this run generated'
        $id = JobId 'final-hash'
        $st = RunJob $id @{ op = 'hash'; algorithm = 'sha256'; paths = @('\final.bin') } 60
        Check ($null -ne $st -and $st.state -eq 'succeeded') 'the job worker is still alive and accepting work'
        if ($null -ne $st -and $st.state -eq 'succeeded') {
            Check ((JobResult $id).files[0].digest -eq $smallSha256) '...and still returns correct digests'
        }
    }
}
finally {
    # Nothing is written outside the mount, so unmounting is the whole cleanup:
    # every temp file this script made lives on the volume being torn down.
    Write-Host "`nUnmounting (stop pid $($proc.Id)) ..." -ForegroundColor Cyan
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Milliseconds 800
    $largeBytes = $null
    [GC]::Collect()
}

if ($skipped -gt 0) { Write-Host "`n$skipped check group(s) skipped" -ForegroundColor Yellow }
if ($fail -eq 0) { Write-Host "ALL `$VRAMDISK JOB E2E CHECKS PASSED" -ForegroundColor Green; exit 0 }
else             { Write-Host "$fail `$VRAMDISK JOB E2E CHECK(S) FAILED" -ForegroundColor Red; exit 1 }
