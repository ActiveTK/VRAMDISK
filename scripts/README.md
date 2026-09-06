# scripts/

Manual end-to-end verification and benchmarking for VRAMDISK. Every script here
mounts the real release binary on a real drive letter and drives it through the
Windows filesystem API, so **they all need an NVIDIA GPU, CUDA and WinFsp on the
machine running them** — they cannot run on CI, and they are not part of
`cargo test`.
Run them by hand before a release, after the release build succeeds.

Build first (these scripts never build anything themselves):

```powershell
cargo build --release --manifest-path src-tauri\Cargo.toml
```

Then, from the repository root:

```powershell
pwsh -File scripts\e2e_robustness.ps1            # ~1 minute
pwsh -File scripts\e2e_jobs.ps1                  # several minutes
```

Each script exits `0` only if every check passed, and prints one green `[ok]`
or red `[FAIL]` line per check. Neither writes anything outside the volume it
mounts; unmounting is the whole cleanup.

## e2e_robustness.ps1

The ordinary filesystem surface, kept short and fast: file and directory CRUD,
append, rename, subtree rename, recursive delete, 8 MiB binary round-trip,
sparse writes, malformed and refused operations (delete a non-empty directory,
rename a directory into its own subtree, file/directory name collisions), and
four concurrent workers hammering the volume at once.

```powershell
pwsh -File scripts\e2e_robustness.ps1 [-Drive T] [-Exe path\to\vramdisk.exe]
```

Mounts a 512 MiB volume.

## bench_vs_ramdisk.ps1

Head-to-head performance against an ordinary RAM disk, which is the thing
VRAMDISK is really competing with. Measures sequential throughput at 4 KiB /
64 KiB / 1 MiB / 16 MiB blocks in both unbuffered and buffered mode, 4 KiB
random-read latency, and a small-file/metadata workload (create, stat, read,
delete).

```powershell
pwsh -File scripts\bench_vs_ramdisk.ps1 -RamDisk R:\ [-Drive V:] [-Size 4GiB]
```

You need a RAM disk mounted already -- ImDisk, OSFMount, whatever -- and its
drive letter passed in. Creating one normally needs elevation, which this
script deliberately does not ask for. It mounts and unmounts VRAMDISK itself.

Both drives go through the identical harness, using raw
`CreateFile`/`ReadFile`/`WriteFile` with page-aligned buffers so unbuffered
mode really bypasses the Windows cache manager. Both modes are reported
because both drives sit behind that cache; measuring only one would say more
about the cache than about the device.

## bench_three_way.ps1

The same head-to-head widened to three targets: VRAMDISK, GpuRamDrive (GPU VRAM
behind ImDisk's block device, with NTFS on top) and an ordinary RAM disk. On top
of what `bench_vs_ramdisk.ps1` measures it adds random-read latency at three
block sizes, concurrent sequential reads at 1/2/4/8 threads, the memory each
design reserves, and a data-processing table — full-text search, MD5, SHA-256
and zip — where VRAMDISK computes on the GPU and the other two must pull every
byte into the CPU first.

```powershell
pwsh -File scripts\bench_three_way.ps1 -RamDisk R:\ -GpuRamDrive P:\ [-Drive V:]
```

Both of the other drives must already be mounted (each needs elevation to
create, which this script deliberately does not ask for). Raw results are
written to `.bench_three_way.json` at the repository root, which is git-ignored;
the write-up lives in [`hikaku.md`](../hikaku.md).

## e2e_jobs.ps1

The `$VRAMDISK` internal virtual API (DEV.md section 11), which the robustness
script does not touch at all:

- **Info surface** — `help.txt`, `stats.txt`, `stats.json`, `trace.txt`,
  `trace.json`, `chunks.json\<path>`; that the JSON parses, that the chunk
  counters are self-consistent and grow after a write, and that the namespace
  refuses writes, directory creation and deletes.
- **Hash jobs** — `md5` / `sha1` / `sha256` / `fnv1a64` against host-computed
  digests, for both a tiny file (CUDA API-kernel path) and a 256 MiB file
  (which calibration routes to the CPU streaming path on any machine whose CPU
  hashes faster than its GPU, i.e. every machine measured so far), plus
  recursive directory hashing.
- **Encode jobs** — Base64 and hex, encode and decode, a byte-identical binary
  round-trip, the Base64 output checked against `[Convert]::ToBase64String`,
  and an invalid-character decode that must fail the job rather than hang.
- **Archive jobs** — `zip` and `tar.zst` compress, then extract elsewhere and
  compare the tree byte for byte. These need a compatible nvCOMP DLL; without
  one they report `[skip]`, not `[FAIL]`.
- **Cancellation** — cancel a long hash job through
  `jobs\<id>\cancel` and check it reaches `cancelled` promptly.
- **Job robustness** — unparseable descriptors, unknown ops, unsupported
  algorithms / formats / codecs, missing fields, unknown job ids, duplicate job
  ids and illegal job ids must all fail cleanly with the mount still up.

```powershell
pwsh -File scripts\e2e_jobs.ps1 [-Drive T] [-Exe path\to\vramdisk.exe] [-Size 2GiB]
```

`-Size` must leave room for the roughly 800 MiB of test data the hash and
cancellation checks write; 2 GiB is the default and needs a GPU with enough
free VRAM for it.
