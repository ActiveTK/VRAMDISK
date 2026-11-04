# WinSpd / VRAM pagefile audit

**Audit date:** 2026-11-04

## Current status

| Area | Status | Evidence |
|---|---|---|
| Existing WinFsp filesystem | `RUNTIME_VERIFIED` by the existing project | Existing `src/fs.rs` and `src/engine.rs` path is unchanged. |
| Generic block-storage contract | `IMPLEMENTED` | `src/block.rs`; checked LBA ranges, 512/4096-byte sectors, CUDA backing, and a test-only in-memory backend. |
| WinSpd SCSI device and Windows disk integration | `NOT IMPLEMENTED` | WinSpd SDK/driver is not installed in the current environment, and no WinSpd adapter is present. |
| Pagefile diagnostics | `IMPLEMENTED` (read-only) | `vramdisk.exe pagefile status` / `check` query Windows CIM data and locate `swapadd.exe` / `swapadd64.exe` beside the app, in `C:\ProgramData`, or on `PATH`; they never execute it. |
| Dynamic pagefile add/remove | `UNSUPPORTED / EXPERIMENTAL_UNVERIFIED` | `enable`, `disable`, and `verify` are guarded and cannot modify Windows paging configuration. |
| GPU block-storage runtime | `NOT RUNTIME_VERIFIED` | GitHub Actions builds the Windows Release executable and runs GPU-free tests; no VRAM disk runtime test has been performed on the Tesla M40. |
| Windows paging over WinSpd | `NOT PAGING_VALIDATED` | No WinSpd paging-path support statement or system-level paging test was found. |

## Architecture findings

- `src/cuda.rs` owns a contiguous CUDA device allocation through `Vram`. `Vram::new` uses a zero-initializing allocation; `read_at`, `write_at`, `zero_at`, and `sync` provide byte-addressed transfers.
- `src/engine.rs`, `src/chunk.rs`, `src/arena.rs`, and `src/lookup.rs` form the existing file-storage model. The 64 KiB allocator, sparse file map, compression, and deduplication are file semantics and must not be used as an NTFS LBA translation layer.
- `src/fs.rs` exposes that file engine through WinFsp callbacks. `src-tauri/src/manager.rs` owns the live mount on a dedicated manager thread. There is no current block-device or service lifecycle abstraction.
- `BlockStorage` is the new boundary below any future WinSpd adapter. `GpuBlockStorage` uses only VRAM and serializes access to its CUDA stream. The RAM-backed implementation is compiled only in tests; production code has no RAM fallback.
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

## Deferred work and safe test order

1. Install/build against a pinned WinSpd SDK in a disposable development environment; inspect the exact public header and rawdisk sample before writing an adapter.
2. Add a narrowly scoped Windows FFI adapter with panic containment at every callback boundary, unique device identity, explicit SCSI status mapping, and verified teardown.
3. Run CPU `BlockStorage` tests, then WinSpd `stgtest` and file-integrity checks against an explicitly identified disposable disk. Never initialize or format an existing physical disk.
4. Add GPU tests on the Tesla M40 using 1–4 GiB only after the block adapter builds and passes CPU/WinSpd tests.
5. Keep pagefile enable disabled unless the paging-path safety case and recoverable-system experiment are approved and pass. Never remove or replace `C:\pagefile.sys` automatically.

No driver installation, disk initialization/formatting, system paging configuration change, GPU reset, forced service termination, low-memory stress, or crash test was performed.
