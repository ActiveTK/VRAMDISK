# WinSpd / VRAM pagefile audit

**Audit date:** 2026-11-04

## Current status

| Area | Status | Evidence |
|---|---|---|
| Existing WinFsp filesystem | `RUNTIME_VERIFIED` by the existing project | Existing `src/fs.rs` and `src/engine.rs` path is unchanged. |
| Generic block-storage contract | `IMPLEMENTED` | `src/block.rs`; checked LBA ranges, 512/4096-byte sectors, CUDA backing, and a test-only in-memory backend. |
| WinSpd SCSI adapter | `IMPLEMENTED` (optional build feature) | `src/winspd.rs` and `native/winspd_shim.c` map Read/Write/Flush/Unmap requests to `BlockStorage`. |
| Pagefile diagnostics | `IMPLEMENTED` (read-only) | `vramdisk.exe pagefile status` / `check` query Windows CIM data and locate `swapadd.exe` / `swapadd64.exe` beside the app, in `C:\ProgramData`, or on `PATH`; they never execute it. |
| Dynamic pagefile add/remove | `UNSUPPORTED / EXPERIMENTAL_UNVERIFIED` | `enable`, `disable`, and `verify` are guarded and cannot modify Windows paging configuration. |
| GPU block-storage runtime | `NOT RUNTIME_VERIFIED` | GitHub Actions builds the Windows Release executable and runs GPU-free tests; no VRAM disk runtime test has been performed on the Tesla M40. |
| Windows paging over WinSpd | `NOT PAGING_VALIDATED` | No WinSpd paging-path support statement or system-level paging test was found. |

## Architecture findings

- `src/cuda.rs` owns a contiguous CUDA device allocation through `Vram`. `Vram::new` uses a zero-initializing allocation; `read_at`, `write_at`, `zero_at`, and `sync` provide byte-addressed transfers.
- `src/engine.rs`, `src/chunk.rs`, `src/arena.rs`, and `src/lookup.rs` form the existing file-storage model. The 64 KiB allocator, sparse file map, compression, and deduplication are file semantics and must not be used as an NTFS LBA translation layer.
- `src/fs.rs` exposes the file engine through WinFsp. The optional WinSpd API currently runs through CLI mode; GUI lifecycle controls are not added yet.
- `BlockStorage` is the boundary below the WinSpd adapter. `GpuBlockStorage` uses only VRAM and serializes CUDA access; the RAM-backed implementation exists only in tests.
- The local machine has a Tesla M40 24 GB in WDDM mode. WinFsp exists under `C:\Program Files (x86)\WinFsp`; WinSpd SDK/driver paths and services were not found. Rust/Cargo and npm are unavailable in this shell.

## WinSpd and paging-path assessment

The [WinSpd project](https://github.com/winfsp/winspd) describes a kernel StorPort virtual miniport plus a user-mode DLL and launcher service. It exposes SCSI read, write, flush, and unmap operations and presents a SCSI disk that Windows can partition and format. Its [raw-disk tutorial](https://github.com/winfsp/winspd/blob/master/doc/WinSpd-Tutorial.asciidoc) is a user-mode storage-service example. The tutorial warns that such a service is an important system component and must service requests promptly; it does not document paging-file support.

Microsoft's [storage-driver paging-path guidance](https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/restrictions-on-pageable-code-in-storage-drivers) describes deadlock constraints for drivers that service paging I/O. That guidance does not establish that a user-mode WinSpd storage process, its dispatcher, CUDA context, CUDA driver calls, and their memory dependencies are safe in the paging path. A page fault could wait for a user-mode service whose working set or dependencies themselves require paging. This circular-wait scenario is a technical risk assessment, not a claim that every WinSpd device will deadlock.

Therefore NTFS visibility, successful `swapadd`, or an increased commit limit would not prove paging I/O correctness. Paging remains guarded until the WinSpd maintainers/documentation explicitly support it and a recoverable test environment validates the complete in-paging path, including low-memory behavior and service/CUDA dependencies.

## License and redistribution

WinSpd states that it is GPLv3 with a special exception for Free/Libre and Open Source Software and offers a commercial license. VRAMDISK is MIT-licensed. No WinSpd code, DLL, driver, or SDK binary is redistributed by this change. Before bundling or linking WinSpd artifacts, retain the exact upstream license and exception text and review the obligations for the intended distribution; do not infer permission from the repository summary alone.

## Implemented commands

```text
vramdisk.exe pagefile status
vramdisk.exe pagefile check
vramdisk.exe pagefile verify
vramdisk.exe pagefile enable --size 2048M   # intentionally blocked
vramdisk.exe pagefile disable               # intentionally blocked
```

`status` and `check` are observational. `VRAMDISK_SWAPADD` can point to a local `swapadd.exe` / `swapadd64.exe`; diagnostics also look beside `vramdisk.exe`, in `C:\ProgramData`, and on `PATH`. They identify the executable without running it. The official download/source listing is maintained by [LTRData](https://ltr-data.se/opencode.html); VRAMDISK does not download or redistribute the executable.

The optional WinSpd build adds these Windows CLI modes:

```text
build-gui.ps1 build --features winspd
vramdisk.exe winspd --mode pipe --size 1GiB
vramdisk.exe winspd --mode disk --size 1GiB
```

Set `WINSPD_SDK_DIR` to the WinSpd Developer SDK root when building. Pipe mode is for `stgtest-x64.exe` at `\\.\pipe\vramdisk\0`; it does not expose a Windows disk. Disk mode requires the installed WinSpd driver and an elevated terminal; it exposes a new VRAM disk but does not initialize or format it. Release builds open a dedicated console window for the interactive WinSpd CLI so Enter can be read even though the main executable uses the Windows GUI subsystem. GUI controls and lifecycle handling are still pending.

## Deferred work and safe test order

1. Run `vramdisk.exe winspd --mode pipe --size 1GiB` and use `stgtest-x64.exe` against `\\.\pipe\vramdisk\0`; this requires the WinSpd user-mode DLL but not its kernel driver.
2. Test `vramdisk.exe winspd --mode disk --size 1GiB` only after installing WinSpd and identifying the newly created VRAMDISK device; never initialize or format another disk.
3. Add GUI controls and manager-thread lifecycle handling after the CLI pipe path passes the WinSpd protocol test.
4. Run GPU tests on the Tesla M40 using 1–4 GiB only after WinSpd protocol tests pass.
5. Keep pagefile enable disabled unless the paging-path safety case and recoverable-system experiment are approved and pass. Never remove or replace `C:\pagefile.sys` automatically.

No driver installation, disk initialization/formatting, system paging configuration change, GPU reset, forced service termination, low-memory stress, or crash test was performed.
