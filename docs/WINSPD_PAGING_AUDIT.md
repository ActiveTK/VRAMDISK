# WinSpd / VRAM pagefile audit

**Audit date:** 2026-11-05

## Current status

| Area | Status | Evidence |
|---|---|---|
| Existing WinFsp filesystem | `RUNTIME_VERIFIED` by the existing project | Existing `src/fs.rs` and `src/engine.rs` path is unchanged. |
| Generic block-storage contract | `IMPLEMENTED` | `src/block.rs`; checked LBA ranges, 512/4096-byte sectors, CUDA backing, and a test-only in-memory backend. |
| WinSpd SCSI adapter | `IMPLEMENTED` (optional build feature) | `src/winspd.rs` and `native/winspd_shim.c` map Read/Write/Flush/Unmap requests to `BlockStorage`. |
| Pagefile diagnostics | `IMPLEMENTED` (read-only) | `vramdisk.exe pagefile status` / `check` query Windows CIM data and locate `swapadd.exe` / `swapadd64.exe` beside the app, in `C:\ProgramData`, or on `PATH`. |
| Explicit pagefile add | `IMPLEMENTED / EXPERIMENTAL_UNVERIFIED` | The GUI creates a GPU-backed WinSpd disk, partitions and formats the new disk as M:, then asks `swapadd` to add a fixed-size pagefile. A configurable extra reserve (default 128 MiB) is added beyond the requested pagefile size. Existing pagefiles are preserved. This does not validate paging safety. |
| Pagefile remove | `UNSUPPORTED` | VRAMDISK does not remove or reconfigure active Windows pagefiles. |
| GPU block-storage runtime | `BASIC I/O VERIFIED BY USER` | User reported `stgtest` returned `OK`, the WinSpd disk came online, and ordinary file reads/writes succeeded. Low-memory and paging-path behavior remain untested. |
| Windows paging over WinSpd | `NOT PAGING_VALIDATED` | No WinSpd paging-path support statement or system-level paging test was found. |

## Architecture findings

- `src/cuda.rs` owns a contiguous CUDA device allocation through `Vram`. `Vram::new` uses a zero-initializing allocation; `read_at`, `write_at`, `zero_at`, and `sync` provide byte-addressed transfers.
- `src/engine.rs`, `src/chunk.rs`, `src/arena.rs`, and `src/lookup.rs` form the existing file-storage model. The 64 KiB allocator, sparse file map, compression, and deduplication are file semantics and must not be used as an NTFS LBA translation layer.
- `src/fs.rs` exposes the file engine through WinFsp. The GUI has separate tabs for the file-based WinFsp RAM disk and the GPU-backed WinSpd pagefile disk. The pagefile workflow owns the WinSpd device on the manager thread and only formats a newly appeared, size-matched VRAMDISK that is still RAW.
- `BlockStorage` is the boundary below the WinSpd adapter. `GpuBlockStorage` uses only VRAM and serializes CUDA access; the RAM-backed implementation exists only in tests.
- The local machine has a Tesla M40 24 GB in WDDM mode. WinFsp exists under `C:\Program Files (x86)\WinFsp`; WinSpd SDK/driver paths and services were not found. Rust/Cargo and npm are unavailable in this shell.

## WinSpd and paging-path assessment

The [WinSpd project](https://github.com/winfsp/winspd) describes a kernel StorPort virtual miniport plus a user-mode DLL and launcher service. It exposes SCSI read, write, flush, and unmap operations and presents a SCSI disk that Windows can partition and format. Its [raw-disk tutorial](https://github.com/winfsp/winspd/blob/master/doc/WinSpd-Tutorial.asciidoc) is a user-mode storage-service example. The tutorial warns that such a service is an important system component and must service requests promptly; it does not document paging-file support.

Microsoft's [storage-driver paging-path guidance](https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/restrictions-on-pageable-code-in-storage-drivers) describes deadlock constraints for drivers that service paging I/O. That guidance does not establish that a user-mode WinSpd storage process, its dispatcher, CUDA context, CUDA driver calls, and their memory dependencies are safe in the paging path. A page fault could wait for a user-mode service whose working set or dependencies themselves require paging. This circular-wait scenario is a technical risk assessment, not a claim that every WinSpd device will deadlock.

Therefore NTFS visibility, successful `swapadd`, or an increased commit limit do not prove paging I/O correctness. The GUI exposes only a manual, user-initiated creation and swapadd sequence; it does not include startup integration to recreate the GPU disk at boot. Once the pagefile is active, shut down or restart Windows to release the GPU allocation and close the disk. Treat a pagefile on this disk as experimental until WinSpd explicitly supports paging and a recoverable test environment validates the complete paging path, including low-memory behavior and service/CUDA dependencies.

## License and redistribution

WinSpd states that it is GPLv3 with a special exception for Free/Libre and Open Source Software and offers a commercial license. VRAMDISK is MIT-licensed. No WinSpd code, DLL, driver, or SDK binary is redistributed by this change. Before bundling or linking WinSpd artifacts, retain the exact upstream license and exception text and review the obligations for the intended distribution; do not infer permission from the repository summary alone.

## Implemented commands

```text
vramdisk.exe pagefile status
vramdisk.exe pagefile check
vramdisk.exe pagefile verify
vramdisk.exe pagefile enable --path R:\pagefile.sys --min-size 256M --max-size 512M
vramdisk.exe pagefile disable               # intentionally blocked
```

`status` and `check` are observational. `VRAMDISK_SWAPADD` can point to a local `swapadd.exe` / `swapadd64.exe`; diagnostics also look beside `vramdisk.exe`, in `C:\ProgramData`, and on `PATH`. The enable action runs it only after the user explicitly supplies a mounted volume path and size. Run VRAMDISK elevated if swapadd requires administrator privileges. The official download/source listing is maintained by [LTRData](https://ltr-data.se/opencode.html); VRAMDISK does not download or redistribute the executable.

The optional WinSpd build adds these Windows CLI modes:

```text
build-gui.ps1 build --features winspd
vramdisk.exe winspd --mode pipe --size 1GiB
vramdisk.exe winspd --mode disk --size 1GiB
```

Set `WINSPD_SDK_DIR` to the WinSpd Developer SDK root when building. Pipe mode is for `stgtest-x64.exe` at `\\.\pipe\vramdisk\0`; it does not expose a Windows disk. Disk mode requires the installed WinSpd driver and an elevated terminal; it exposes a new VRAM disk but does not initialize or format it. The GUI pagefile tab runs the integrated disk creation and pagefile sequence and also requires elevation. Release builds open a dedicated console window for the interactive WinSpd CLI so Enter can be read even though the main executable uses the Windows GUI subsystem.

## Deferred work and safe test order

1. Run `vramdisk.exe winspd --mode pipe --size 1GiB` and use `stgtest-x64.exe` against `\\.\pipe\vramdisk\0`; this requires the WinSpd user-mode DLL but not its kernel driver.
2. Test `vramdisk.exe winspd --mode disk --size 1GiB` only after installing WinSpd and identifying the newly created VRAMDISK device; never initialize or format another disk.
3. Validate clean manager-thread shutdown and M: disk removal behavior in release builds; the GUI stays in the tray after a pagefile is added, and a reboot ends that session.
4. Run GPU tests on the Tesla M40 using small pagefile sizes only in a recoverable environment after WinSpd protocol tests pass.
5. Keep `C:\pagefile.sys` enabled. The experimental swapadd action only adds the requested file; do not use it for workloads that cannot tolerate a WinSpd/GPU service interruption. A successful swapadd call or ordinary file I/O is not evidence that Windows paging is safe on this disk.

No driver installation, disk initialization/formatting, system paging configuration change, GPU reset, forced service termination, low-memory stress, or crash test was performed.
