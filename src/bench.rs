//! Comprehensive speed benchmarks: raw VRAM bandwidth, storage engine
//! throughput, compression codecs, and GPU hashing.
//!
//! Each measurement is repeated [`RUNS`] times; the reported figure is the
//! arithmetic mean of those runs.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};

use crate::cli::format_size;
use crate::cuda::Vram;
use crate::engine::StorageEngine;
use crate::gpu_hash::GpuHasher;
use crate::nvcomp::Lz4Codec;
use crate::CHUNK_SIZE;

/// Number of timed iterations per measurement.
const RUNS: usize = 3;

/// Default VRAM to allocate when `--size` is not given.
pub const DEFAULT_BENCH_SIZE: u64 = 512 * 1024 * 1024; // 512 MiB
const IO_BENCH_BLOCK_SIZE: usize = 8 * 1024 * 1024;
const IO_BENCH_BASE_FILE_SIZES: &[u64] = &[
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
];
const IO_BENCH_FIRST_LARGE_FILE_SIZE: u64 = 1024 * 1024 * 1024;
const IO_BENCH_MAX_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024;

pub fn run(device: usize, vram_size: u64) -> Result<()> {
    let dev_name = Vram::device_name(device).unwrap_or_else(|_| "?".into());
    let total_vram = Vram::device_total_mem(device)?;

    println!("=== VRAMDISK Speed Benchmark ===");
    println!(
        "  device : CUDA[{}] {} ({})",
        device,
        dev_name,
        format_size(total_vram)
    );
    println!("  vram   : {}", format_size(vram_size));
    println!("  runs   : {} per measurement\n", RUNS);

    bench_vram(device, vram_size)?;
    bench_engine(device, vram_size)?;
    bench_engine_dedup(device)?;
    bench_engine_compress(device)?;
    bench_compression(device)?;
    bench_hash(device)?;

    println!("Benchmark complete.");
    Ok(())
}

#[cfg(windows)]
pub fn run_io(device: usize) -> Result<()> {
    let dev_name = Vram::device_name(device).unwrap_or_else(|_| "?".into());
    let total_vram = Vram::device_total_mem(device)?;
    let disk_size = io_bench_disk_size(total_vram);
    let file_sizes = io_bench_file_sizes(disk_size);
    anyhow::ensure!(
        !file_sizes.is_empty(),
        "GPU VRAM is too small for the smallest --bench-io file size"
    );
    let mount = choose_bench_mount()?;
    let root = mount_root(&mount);

    println!("=== VRAMDISK Filesystem I/O Benchmark ===");
    println!(
        "  device : CUDA[{}] {} ({})",
        device,
        dev_name,
        format_size(total_vram)
    );
    println!("  mount  : {mount} (temporary)");
    println!("  disk   : {}", format_size(disk_size));
    println!("  block  : {}", format_size(IO_BENCH_BLOCK_SIZE as u64));
    println!(
        "  runs   : {} mode(s) x {} file size(s)\n",
        4,
        file_sizes.len()
    );

    for mode in IoBenchMode::all() {
        println!(
            "[{}] compress={}, dedup={}",
            mode.label(),
            if mode.compress { "on" } else { "off" },
            if mode.dedup { "on" } else { "off" }
        );

        let vram = Vram::new(device, disk_size)
            .with_context(|| format!("failed to allocate {}", format_size(disk_size)))?;
        let engine = StorageEngine::new(vram, mode.compress, mode.dedup)?;
        let mounted = crate::fs::mount(engine, &mount, "VRAMDISK Bench")?;
        wait_for_mount(&root)?;

        println!(
            "    {:>12} {:>14} {:>14} {:>14} {:>14}",
            "File", "Write", "Read", "Write wall", "Read wall"
        );
        println!("    {}", "─".repeat(76));

        for &size in &file_sizes {
            let result = run_io_file_size(&root, size, IO_BENCH_BLOCK_SIZE)
                .with_context(|| format!("I/O benchmark failed at {}", format_size(size)))?;
            println!(
                "    {:>12} {:>14} {:>14} {:>14.3?} {:>14.3?}",
                format_size(size),
                throughput(size, result.write.wall),
                throughput(size, result.read.wall),
                result.write.wall,
                result.read.wall
            );
        }

        mounted.unmount();
        println!();
    }

    println!("Filesystem I/O benchmark complete.");
    Ok(())
}

#[cfg(not(windows))]
pub fn run_io(_device: usize) -> Result<()> {
    anyhow::bail!("--bench-io requires Windows/WinFsp")
}

fn io_bench_disk_size(total_vram: u64) -> u64 {
    let target = (total_vram as f64 * 0.8) as u64;
    (target / CHUNK_SIZE) * CHUNK_SIZE
}

fn io_bench_file_sizes(disk_size: u64) -> Vec<u64> {
    let mut sizes: Vec<u64> = IO_BENCH_BASE_FILE_SIZES
        .iter()
        .copied()
        .filter(|&size| size <= disk_size)
        .collect();

    let mut size = IO_BENCH_FIRST_LARGE_FILE_SIZE;
    while size <= disk_size && size <= IO_BENCH_MAX_FILE_SIZE {
        sizes.push(size);
        let Some(next) = size.checked_mul(2) else {
            break;
        };
        size = next;
    }
    sizes
}

struct IoBenchMode {
    compress: bool,
    dedup: bool,
}

impl IoBenchMode {
    fn all() -> [Self; 4] {
        [
            Self {
                compress: false,
                dedup: false,
            },
            Self {
                compress: true,
                dedup: false,
            },
            Self {
                compress: false,
                dedup: true,
            },
            Self {
                compress: true,
                dedup: true,
            },
        ]
    }

    fn label(&self) -> &'static str {
        match (self.compress, self.dedup) {
            (false, false) => "raw",
            (true, false) => "compress",
            (false, true) => "dedup",
            (true, true) => "compress+dedup",
        }
    }
}

struct IoFileResult {
    write: IoPhaseResult,
    read: IoPhaseResult,
}

fn run_io_file_size(dir: &Path, size: u64, block_size: usize) -> Result<IoFileResult> {
    anyhow::ensure!(
        dir.is_dir(),
        "benchmark target is not a directory: {}",
        dir.display()
    );
    let path = dir.join("vramdisk-io-bench.bin");
    let _ = std::fs::remove_file(&path);
    let paths = vec![path.clone()];
    let write = run_io_phase(paths.clone(), size, block_size, IoPhase::Write)?;
    let read = run_io_phase(paths, size, block_size, IoPhase::Read)?;
    let _ = std::fs::remove_file(path);
    anyhow::ensure!(
        write.checksum == read.checksum,
        "checksum mismatch after filesystem readback"
    );
    Ok(IoFileResult { write, read })
}

#[cfg(windows)]
fn choose_bench_mount() -> Result<String> {
    for letter in ['R', 'T', 'S', 'Q', 'U', 'V', 'W', 'X', 'Y', 'Z'] {
        let root = format!("{letter}:\\");
        if !Path::new(&root).exists() {
            return Ok(format!("{letter}:"));
        }
    }
    anyhow::bail!("could not find a free drive letter for --bench-io")
}

#[cfg(windows)]
fn mount_root(mount: &str) -> PathBuf {
    if mount.as_bytes().get(1) == Some(&b':') && mount.len() == 2 {
        PathBuf::from(format!("{mount}\\"))
    } else {
        PathBuf::from(mount)
    }
}

#[cfg(windows)]
fn wait_for_mount(root: &Path) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if root.is_dir() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(25));
    }
    anyhow::bail!("mount did not become visible: {}", root.display())
}

#[derive(Clone, Copy)]
enum IoPhase {
    Write,
    Read,
}

struct IoPhaseResult {
    wall: Duration,
    checksum: u64,
}

fn run_io_phase(
    paths: Vec<PathBuf>,
    size: u64,
    block_size: usize,
    phase: IoPhase,
) -> Result<IoPhaseResult> {
    let barrier = Arc::new(Barrier::new(paths.len() + 1));
    let mut handles = Vec::with_capacity(paths.len());
    for (worker, path) in paths.into_iter().enumerate() {
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || -> Result<(Duration, u64)> {
            let block = make_io_block(block_size, worker as u8);
            barrier.wait();
            let start = Instant::now();
            let checksum = match phase {
                IoPhase::Write => write_test_file(&path, size, &block)?,
                IoPhase::Read => read_test_file(&path, size, block.len())?,
            };
            Ok((start.elapsed(), checksum))
        }));
    }

    barrier.wait();
    let wall_start = Instant::now();
    let mut checksum = 0u64;
    for h in handles {
        let (_elapsed, csum) = h
            .join()
            .map_err(|_| anyhow::anyhow!("I/O benchmark worker panicked"))??;
        checksum = checksum.wrapping_add(csum);
    }
    Ok(IoPhaseResult {
        wall: wall_start.elapsed(),
        checksum,
    })
}

fn write_test_file(path: &Path, size: u64, block: &[u8]) -> Result<u64> {
    let mut checksum = 0u64;
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut written = 0u64;
    while written < size {
        let take = (size - written).min(block.len() as u64) as usize;
        file.write_all(&block[..take])?;
        checksum = checksum.wrapping_add(sample_checksum(&block[..take]));
        written += take as u64;
    }
    file.sync_all()?;
    Ok(checksum)
}

fn read_test_file(path: &Path, size: u64, block_size: usize) -> Result<u64> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    let mut buf = vec![0u8; block_size];
    let mut checksum = 0u64;
    let mut read_total = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        read_total += n as u64;
        checksum = checksum.wrapping_add(sample_checksum(&buf[..n]));
    }
    anyhow::ensure!(
        read_total == size,
        "short read from {}: expected {} bytes, got {}",
        path.display(),
        size,
        read_total
    );
    Ok(checksum)
}

fn make_io_block(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7 ^ seed))
        .collect()
}

fn sample_checksum(buf: &[u8]) -> u64 {
    if buf.is_empty() {
        0
    } else {
        buf[0] as u64 + ((buf[buf.len() - 1] as u64) << 8) + buf.len() as u64
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn avg(times: &[Duration]) -> Duration {
    times.iter().sum::<Duration>() / times.len() as u32
}

fn throughput(bytes: u64, elapsed: Duration) -> String {
    let bps = bytes as f64 / elapsed.as_secs_f64();
    let gib = 1024.0f64.powi(3);
    let mib = 1024.0f64.powi(2);
    if bps >= gib {
        format!("{:.2} GB/s", bps / gib)
    } else if bps >= mib {
        format!("{:.1} MB/s", bps / mib)
    } else {
        format!("{:.0} KB/s", bps / 1024.0)
    }
}

// ─── [1] Raw VRAM bandwidth ───────────────────────────────────────────────────

fn bench_vram(device: usize, vram_size: u64) -> Result<()> {
    println!(
        "[1] Raw VRAM Bandwidth  (host↔device memcpy, avg of {} runs)",
        RUNS
    );
    println!(
        "    {:<12} {:>16} {:>16}",
        "Size", "Write (H→D)", "Read (D→H)"
    );
    println!("    {}", "─".repeat(48));

    let test_sizes: &[u64] = &[4 * 1024, 256 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024];

    for &size in test_sizes {
        if size > vram_size / 2 {
            break;
        }
        let mut vram = Vram::new(device, size)?;
        let src = vec![0xAAu8; size as usize];
        let mut dst = vec![0u8; size as usize];

        let mut wt = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            vram.write_at(0, &src)?;
            wt.push(t.elapsed());
        }
        let mut rt = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            vram.read_at(0, &mut dst)?;
            rt.push(t.elapsed());
        }

        println!(
            "    {:<12} {:>16} {:>16}",
            format_size(size),
            throughput(size, avg(&wt)),
            throughput(size, avg(&rt)),
        );
    }
    println!();
    Ok(())
}

// ─── [2] Storage engine throughput ───────────────────────────────────────────

fn bench_engine(device: usize, vram_size: u64) -> Result<()> {
    println!(
        "[2] Storage Engine Throughput  (no compress, no dedup, avg of {} runs)",
        RUNS
    );
    println!("    {:<12} {:>16} {:>16}", "File size", "Write", "Read");
    println!("    {}", "─".repeat(48));

    let test_sizes: &[u64] = &[
        1 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    ];

    for &size in test_sizes {
        // Reserve roughly half the VRAM for engine metadata + chunk headers.
        if size > vram_size / 2 {
            break;
        }
        let vram = Vram::new(device, size + 4 * CHUNK_SIZE)?;
        let mut engine = StorageEngine::new(vram, false, false)?;
        engine.table_mut().create_file("\\bench", 0).unwrap();

        // Data: cycling byte values (non-trivial so the compiler won't elide it).
        let data: Vec<u8> = (0..size as usize).map(|i| i as u8).collect();

        // Warm-up: prime CUDA JIT and VRAM page mapping.
        engine
            .write("\\bench", 0, &data)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let mut wt = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            // Truncate to 0 to free all chunks before the next write.
            engine
                .set_size("\\bench", 0)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let t = Instant::now();
            engine
                .write("\\bench", 0, &data)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            wt.push(t.elapsed());
        }

        let mut rt = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            engine
                .read("\\bench", 0, size as usize)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            rt.push(t.elapsed());
        }

        println!(
            "    {:<12} {:>16} {:>16}",
            format_size(size),
            throughput(size, avg(&wt)),
            throughput(size, avg(&rt)),
        );
    }
    println!();
    Ok(())
}

// ─── [2a] Deduplicated storage engine throughput ────────────────────────────

fn bench_engine_dedup(device: usize) -> Result<()> {
    println!(
        "[2a] Dedup Engine Throughput  (--dedup, avg of {} runs)",
        RUNS
    );

    let size = 64 * 1024 * 1024u64;
    let vram = match Vram::new(device, size + 4 * CHUNK_SIZE) {
        Ok(v) => v,
        Err(_) => {
            println!("    (skipped: cannot allocate VRAM)\n");
            return Ok(());
        }
    };
    let mut engine = StorageEngine::new(vram, false, true)?;
    engine.table_mut().create_file("\\unique", 0).unwrap();
    engine.table_mut().create_file("\\dupe", 0).unwrap();
    let unique: Vec<u8> = (0..size as usize).map(|i| (i % 251) as u8).collect();

    println!("    {:<18} {:>16}", "Case", "Write");
    println!("    {}", "─".repeat(38));

    let mut unique_times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        engine
            .set_size("\\unique", 0)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let t = Instant::now();
        engine
            .write("\\unique", 0, &unique)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        unique_times.push(t.elapsed());
    }

    let mut dupe_times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        engine
            .set_size("\\dupe", 0)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let t = Instant::now();
        engine
            .write("\\dupe", 0, &unique)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        dupe_times.push(t.elapsed());
    }

    println!(
        "    {:<18} {:>16}",
        "unique write",
        throughput(size, avg(&unique_times))
    );
    println!(
        "    {:<18} {:>16}",
        "duplicate write",
        throughput(size, avg(&dupe_times))
    );
    println!();
    Ok(())
}

// ─── [2b] Compressed storage engine throughput ───────────────────────────────

fn bench_engine_compress(device: usize) -> Result<()> {
    println!(
        "[2b] Compressed Engine Throughput  (--compress, compressible data, avg of {} runs)",
        RUNS
    );

    // Allocate a context buffer; compressible data uses little of it, and the
    // nvCOMP codec keeps its own batch scratch separately.
    let vram = match Vram::new(device, 64 * 1024 * 1024) {
        Ok(v) => v,
        Err(_) => {
            println!("    (skipped: cannot allocate VRAM)\n");
            return Ok(());
        }
    };
    let mut engine = match StorageEngine::new(vram, true, false) {
        Ok(e) => e,
        Err(_) => {
            println!("    (skipped: engine init failed)\n");
            return Ok(());
        }
    };
    engine.table_mut().create_file("\\bench", 0).unwrap();

    println!("    {:<12} {:>16} {:>16}", "Data", "Write", "Read");
    println!("    {}", "─".repeat(48));

    let size = 64 * 1024 * 1024u64;
    for (label, data) in [
        (
            "compressible",
            (0u8..64).cycle().take(size as usize).collect::<Vec<_>>(),
        ),
        ("incompressible", {
            let mut s = 0x9e37_79b9u32;
            (0..size as usize)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (s >> 24) as u8
                })
                .collect::<Vec<_>>()
        }),
    ] {
        // Warm-up (primes nvCOMP + arena).
        engine
            .set_size("\\bench", 0)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        engine
            .write("\\bench", 0, &data)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let mut wt = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            engine
                .set_size("\\bench", 0)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let t = Instant::now();
            engine
                .write("\\bench", 0, &data)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            wt.push(t.elapsed());
        }
        let mut rt = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            engine
                .read("\\bench", 0, size as usize)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            rt.push(t.elapsed());
        }

        println!(
            "    {:<12} {:>16} {:>16}",
            label,
            throughput(size, avg(&wt)),
            throughput(size, avg(&rt)),
        );
    }
    println!();
    Ok(())
}

// ─── [3] Compression throughput ───────────────────────────────────────────────

fn bench_compression(device: usize) -> Result<()> {
    const CHUNKS: usize = 256; // 256 × 64 KiB = 16 MiB per run
    let total_bytes = CHUNK_SIZE * CHUNKS as u64;
    // Extra scratch for nvCOMP internal buffers.
    let scratch_vram = (CHUNKS as u64 + 16) * CHUNK_SIZE;

    println!(
        "[3] Compression  ({} × 64 KiB chunks per run, avg of {} runs)",
        CHUNKS, RUNS
    );

    for (label, data) in &[
        (
            "compressible (64-byte repeating pattern)",
            (0u8..64)
                .cycle()
                .take(CHUNK_SIZE as usize)
                .collect::<Vec<_>>(),
        ),
        ("incompressible (pseudo-random)", {
            let mut s = 0x9e3779b9u32;
            (0..CHUNK_SIZE as usize)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (s >> 24) as u8
                })
                .collect::<Vec<_>>()
        }),
    ] {
        println!("\n    Data: {label}");
        println!(
            "    {:<14} {:>14} {:>14} {:>8}",
            "Codec", "Compress", "Decompress", "Ratio"
        );
        println!("    {}", "─".repeat(54));

        // ── LZ4 (GPU nvCOMP), whole batch per launch ──────────────────────
        if let Ok(vram) = Vram::new(device, scratch_vram) {
            if let Ok(mut codec) = Lz4Codec::load(&vram) {
                // Contiguous buffer of CHUNKS chunks fed to nvCOMP in one call.
                let mut batch_in = Vec::with_capacity(total_bytes as usize);
                for _ in 0..CHUNKS {
                    batch_in.extend_from_slice(data);
                }

                let mut comps: Vec<Option<Vec<u8>>> = Vec::new();
                let mut ct = Vec::with_capacity(RUNS);
                for _ in 0..RUNS {
                    let t = Instant::now();
                    comps = codec
                        .compress_batch(&batch_in)
                        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                    ct.push(t.elapsed());
                }

                let shrank = comps[0].is_some();
                let (ratio_str, decomp_str) = if shrank {
                    let csize = comps[0].as_ref().unwrap().len();
                    let ratio = format!("{:.1}:1", data.len() as f64 / csize as f64);
                    // Raw-store any chunk that didn't shrink so every slot has a blob.
                    let owned: Vec<Vec<u8>> = comps
                        .iter()
                        .map(|c| c.clone().unwrap_or_else(|| data.clone()))
                        .collect();
                    let blobs: Vec<&[u8]> = owned.iter().map(|b| b.as_slice()).collect();
                    let mut dt = Vec::with_capacity(RUNS);
                    for _ in 0..RUNS {
                        let t = Instant::now();
                        codec
                            .decompress_batch(&blobs)
                            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                        dt.push(t.elapsed());
                    }
                    (ratio, throughput(total_bytes, avg(&dt)))
                } else {
                    ("1.0:1".into(), "N/A (raw)".into())
                };

                println!(
                    "    {:<14} {:>14} {:>14} {:>8}",
                    "LZ4 (GPU)",
                    throughput(total_bytes, avg(&ct)),
                    decomp_str,
                    ratio_str,
                );
            } else {
                println!("    LZ4 (GPU)     : nvCOMP not available");
            }
        }

        // ── zstd level 3 (CPU) ────────────────────────────────────────────
        {
            let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(CHUNKS);

            let mut ct = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                chunks.clear();
                let t = Instant::now();
                for _ in 0..CHUNKS {
                    chunks.push(
                        zstd::encode_all(data.as_slice(), 3).map_err(|e| anyhow::anyhow!("{e}"))?,
                    );
                }
                ct.push(t.elapsed());
            }

            let ratio_str = if chunks[0].len() < data.len() {
                format!("{:.1}:1", data.len() as f64 / chunks[0].len() as f64)
            } else {
                "1.0:1".into()
            };

            let mut dt = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let t = Instant::now();
                for c in &chunks {
                    zstd::decode_all(c.as_slice()).map_err(|e| anyhow::anyhow!("{e}"))?;
                }
                dt.push(t.elapsed());
            }

            println!(
                "    {:<14} {:>14} {:>14} {:>8}",
                "zstd-3 (CPU)",
                throughput(total_bytes, avg(&ct)),
                throughput(total_bytes, avg(&dt)),
                ratio_str,
            );
        }
    }
    println!();
    Ok(())
}

// ─── [4] GPU FNV-1a hash (dedup path) ────────────────────────────────────────

fn bench_hash(device: usize) -> Result<()> {
    const CHUNKS: usize = 256;
    let total_bytes = CHUNK_SIZE * CHUNKS as u64;

    println!(
        "[4] GPU FNV-1a Hash  (dedup path, {} × 64 KiB chunks per run, avg of {} runs)",
        CHUNKS, RUNS
    );

    let vram_size = total_bytes + CHUNK_SIZE;
    let mut vram = Vram::new(device, vram_size)?;
    let vram_base = vram.buf_device_ptr();
    let mut hasher = GpuHasher::new(&vram)?;

    // Fill the buffer with recognizable data.
    let fill = vec![0x5Au8; vram_size as usize];
    vram.write_at(0, &fill)?;

    // All chunk offsets, hashed in one batched launch (one block per chunk).
    let offsets: Vec<u64> = (0..CHUNKS as u64).map(|i| i * CHUNK_SIZE).collect();
    let mut out = vec![0u64; CHUNKS];

    // Warm-up.
    hasher
        .hash_chunks(vram_base, &offsets, &mut out)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        hasher
            .hash_chunks(vram_base, &offsets, &mut out)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        times.push(t.elapsed());
    }

    println!(
        "    batched ({} chunks/launch) : {}",
        CHUNKS,
        throughput(total_bytes, avg(&times))
    );

    // Single-chunk latency (the dedup verify path hashes one candidate/write).
    let mut single = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        hasher
            .hash_chunk(vram_base, 0)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        single.push(t.elapsed());
    }
    let one = avg(&single);
    println!(
        "    single chunk             : {} ({:.1} µs/chunk)\n",
        throughput(CHUNK_SIZE, one),
        one.as_secs_f64() * 1e6
    );
    Ok(())
}
