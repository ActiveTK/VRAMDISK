//! CrystalDiskMark back end that measures GPU transfers instead of file I/O.
//!
//! CrystalDiskMark does no I/O itself: it is a GUI that shells out to a
//! DiskSpd build and reads the result back through two channels. This module
//! speaks that same protocol, so a CDM patched to launch
//! `vramdisk.exe cdm-bench ...` drives this code and renders the numbers in its
//! own UI.
//!
//! # Why this exists
//!
//! No file benchmark can measure a transfer that stays on the GPU. DiskSpd,
//! CrystalDiskMark and every other tool built on `ReadFile`/`WriteFile` read
//! into, and write out of, a buffer in their own address space — system memory.
//! Pointed at VRAMDISK they therefore measure VRAM ↔ system RAM across PCIe,
//! while the same run against a RAM disk is system RAM ↔ system RAM. That is a
//! real and fair thing to measure, but it is not the GPU's transfer rate, and
//! no amount of patching the *tool* changes it: the data has to reach the
//! benchmarking process.
//!
//! So this back end drops the file layer entirely. The "disk" is a VRAM region,
//! a "read" copies out of it and a "write" copies into it, and both are
//! device-to-device copies that never touch the bus.
//!
//! # Protocol (from CrystalDiskMark's `DiskBench.cpp`)
//!
//! Command line, of which the flags below are honoured and the rest ignored:
//!
//! ```text
//! -b<N>K  block size          -o<N>  outstanding ops per thread
//! -t<N>   threads             -w<N>  write percentage (0 = read, 100 = write)
//! -r      random access       -d<N>  duration in seconds
//! -A<pid> CDM's process id, which names the shared memory for the latency
//! ```
//!
//! Results go back two ways:
//!
//! * **Throughput** as the process exit code: CDM computes `code / 10 / 1000.0`
//!   and displays it as MB/s, so the code is `MB/s * 10000`.
//! * **Latency** as an `f64` of milliseconds written into the named shared
//!   section `CrystalDiskMark<pid:08X>`, which CDM creates before launching us
//!   and scales to microseconds.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};

use crate::cuda::Vram;

/// VRAM per thread for the region being copied out of / into. Comfortably past
/// the L2 on current parts, so this measures memory rather than cache, while
/// still leaving room for CDM's 16-thread profile.
const REGION_BYTES: u64 = 128 * 1024 * 1024;

/// Ceiling on the total allocation, so a high thread count degrades the region
/// size instead of failing to allocate.
const TOTAL_VRAM_BUDGET: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
struct Params {
    block: u64,
    queue: u32,
    threads: u32,
    write_pct: u32,
    random: bool,
    duration: Duration,
    cdm_pid: Option<u32>,
    device: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            block: 1024 * 1024,
            queue: 1,
            threads: 1,
            write_pct: 0,
            random: false,
            duration: Duration::from_secs(5),
            cdm_pid: None,
            device: 0,
        }
    }
}

/// Parse a `-bNNNK` style suffixed size. DiskSpd accepts `K`, `M` and `G`;
/// CrystalDiskMark only ever emits `K`, but the others cost nothing.
fn parse_suffixed(v: &str) -> Option<u64> {
    let (digits, mul) = match v.as_bytes().last() {
        Some(b'K') | Some(b'k') => (&v[..v.len() - 1], 1024),
        Some(b'M') | Some(b'm') => (&v[..v.len() - 1], 1024 * 1024),
        Some(b'G') | Some(b'g') => (&v[..v.len() - 1], 1024 * 1024 * 1024),
        _ => (v, 1),
    };
    digits.parse::<u64>().ok()?.checked_mul(mul)
}

fn parse_args(args: &[String]) -> Params {
    let mut p = Params::default();
    for a in args {
        let Some(rest) = a.strip_prefix('-') else {
            // A bare argument is the target path CDM passes last. There is no
            // file in this back end, so it is ignored on purpose.
            continue;
        };
        let (flag, value) = rest.split_at(1);
        match flag {
            "b" => {
                if let Some(v) = parse_suffixed(value) {
                    p.block = v.max(1);
                }
            }
            "o" => {
                if let Ok(v) = value.parse::<u32>() {
                    p.queue = v.max(1);
                }
            }
            "t" => {
                if let Ok(v) = value.parse::<u32>() {
                    p.threads = v.max(1);
                }
            }
            "w" => {
                if let Ok(v) = value.parse::<u32>() {
                    p.write_pct = v.min(100);
                }
            }
            "d" => {
                if let Ok(v) = value.parse::<u64>() {
                    p.duration = Duration::from_secs(v.max(1));
                }
            }
            "A" => p.cdm_pid = value.parse::<u32>().ok(),
            "r" => p.random = true,
            // -W (warm-up), -S (no buffering), -Z (write buffer), -L (latency),
            // -a/-ag (affinity), -si (interlocked sequential): all describe the
            // file path this back end does not have.
            _ => {}
        }
    }
    p
}

struct ThreadResult {
    bytes: u64,
    ops: u64,
    busy: Duration,
}

/// One worker: its own VRAM allocation, its own stream, its own slice of the
/// work. Per-thread allocations keep `Vram` off the thread boundary and make
/// the copies independent, which is what a queue-depth test wants anyway.
fn run_thread(p: Params, region: u64, seed: u64, stop: Arc<AtomicBool>) -> Result<ThreadResult> {
    let vram = Vram::new(p.device, region * 2)
        .with_context(|| format!("failed to allocate {} of VRAM", region * 2))?;
    let base = vram.buf_device_ptr();
    // Low half is the "disk", high half the buffer the copies land in or come
    // from. A read copies disk -> buffer, a write copies buffer -> disk; both
    // are device-to-device.
    let scratch = region;
    let blocks = (region / p.block).max(1);

    let mut rng = seed | 1;
    let mut next_block = 0u64;
    let mut bytes = 0u64;
    let mut ops = 0u64;
    let mut busy = Duration::ZERO;

    while !stop.load(Ordering::Relaxed) {
        let start = Instant::now();
        for _ in 0..p.queue {
            let idx = if p.random {
                // xorshift64*: enough for scattering block offsets, and no
                // dependency on how a general-purpose RNG is seeded per thread.
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng % blocks
            } else {
                let i = next_block;
                next_block = (next_block + 1) % blocks;
                i
            };
            let off = idx * p.block;
            if p.write_pct >= 50 {
                vram.copy_dev_into(off, base + scratch + off, p.block)?;
            } else {
                vram.copy_dev_into(scratch + off, base + off, p.block)?;
            }
        }
        vram.sync()?;
        busy += start.elapsed();
        bytes += p.block * p.queue as u64;
        ops += p.queue as u64;
    }

    Ok(ThreadResult { bytes, ops, busy })
}

/// Hand the mean latency back to CrystalDiskMark.
///
/// CDM creates the section before launching us and reads it after we exit; if
/// it is not there (this back end run by hand, say) there is simply nobody to
/// report to, so a failure here is not worth failing the run over.
#[cfg(windows)]
fn publish_latency(pid: u32, millis: f64) {
    use windows::core::HSTRING;
    use windows::Win32::System::Memory::{
        MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
    };
    let name = HSTRING::from(format!("CrystalDiskMark{pid:08X}"));
    unsafe {
        let Ok(handle) = OpenFileMappingW(FILE_MAP_ALL_ACCESS.0, false, &name) else {
            return;
        };
        let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 8);
        if !view.Value.is_null() {
            std::ptr::write_unaligned(view.Value as *mut f64, millis);
            let _ = UnmapViewOfFile(view);
        }
        let _ = windows::Win32::Foundation::CloseHandle(handle);
    }
}

#[cfg(not(windows))]
fn publish_latency(_pid: u32, _millis: f64) {}

/// Entry point for `vramdisk.exe cdm-bench ...`. Returns the process exit code
/// CrystalDiskMark reads the throughput out of.
pub fn run(args: Vec<String>) -> i32 {
    let p = parse_args(&args);

    // Shrink the per-thread region rather than fail when a profile asks for
    // many threads (CDM's RND4K Q32T16 asks for sixteen).
    let region = (TOTAL_VRAM_BUDGET / (2 * p.threads as u64))
        .min(REGION_BYTES)
        .max(p.block * 4);

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::with_capacity(p.threads as usize);
    for t in 0..p.threads {
        let stop = stop.clone();
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t as u64 + 1);
        handles.push(std::thread::spawn(move || {
            run_thread(p, region, seed, stop)
        }));
    }

    std::thread::sleep(p.duration);
    stop.store(true, Ordering::Relaxed);

    let mut bytes = 0u64;
    let mut ops = 0u64;
    let mut busy = Duration::ZERO;
    let mut failure = None;
    for h in handles {
        match h.join() {
            Ok(Ok(r)) => {
                bytes += r.bytes;
                ops += r.ops;
                busy += r.busy;
            }
            Ok(Err(e)) => {
                failure.get_or_insert(format!("{e:#}"));
            }
            Err(_) => {
                failure.get_or_insert_with(|| "benchmark thread panicked".to_string());
            }
        }
    }
    if let Some(e) = failure {
        eprintln!("Error: {e}");
        return 0;
    }
    if ops == 0 {
        eprintln!("Error: no transfers completed");
        return 0;
    }

    // `busy` is summed across threads, so dividing by the total op count gives
    // the mean time one operation was outstanding — the same quantity DiskSpd
    // reports, not the wall clock divided by work done.
    let latency_ms = busy.as_secs_f64() * 1000.0 / ops as f64;
    let mb_per_sec = bytes as f64 / p.duration.as_secs_f64() / (1000.0 * 1000.0);

    if let Some(pid) = p.cdm_pid {
        publish_latency(pid, latency_ms);
    }
    println!(
        "{:.2} MB/s, {:.1} us/op, {} ops, block {} KiB, queue {}, threads {}, {}{}",
        mb_per_sec,
        latency_ms * 1000.0,
        ops,
        p.block / 1024,
        p.queue,
        p.threads,
        if p.random { "random " } else { "sequential " },
        if p.write_pct >= 50 { "write" } else { "read" },
    );

    // CrystalDiskMark divides the exit code by 10_000 to get MB/s.
    let code = (mb_per_sec * 10_000.0).round();
    code.clamp(0.0, i32::MAX as f64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_a_crystaldiskmark_command_line() {
        // Exactly what DiskBench.cpp emits for SEQ1M Q8T1 read.
        let p = parse_args(&args(
            "-b1024K -o8 -t1 -W0 -S -w0 -ag -d5 -A00001A2B -L C:\\test.dat",
        ));
        assert_eq!(p.block, 1024 * 1024);
        assert_eq!(p.queue, 8);
        assert_eq!(p.threads, 1);
        assert_eq!(p.write_pct, 0);
        assert!(!p.random);
        assert_eq!(p.duration, Duration::from_secs(5));
        // -A is decimal in CDM's `%d`; a hex-looking value must not silently
        // become a different pid.
        assert_eq!(p.cdm_pid, None);

        let p = parse_args(&args("-b4K -o32 -t16 -W0 -S -w100 -r -ag -d1 -A4321"));
        assert_eq!(p.block, 4096);
        assert_eq!(p.queue, 32);
        assert_eq!(p.threads, 16);
        assert_eq!(p.write_pct, 100);
        assert!(p.random);
        assert_eq!(p.cdm_pid, Some(4321));
    }

    #[test]
    fn unknown_flags_and_missing_values_keep_the_defaults() {
        let d = Params::default();
        let p = parse_args(&args("-si -Z1024K -junk -b -o"));
        assert_eq!(p.block, d.block);
        assert_eq!(p.queue, d.queue);
        assert_eq!(p.threads, d.threads);
    }

    #[test]
    fn suffixes_scale() {
        assert_eq!(parse_suffixed("4K"), Some(4096));
        assert_eq!(parse_suffixed("1M"), Some(1024 * 1024));
        assert_eq!(parse_suffixed("512"), Some(512));
        assert_eq!(parse_suffixed(""), None);
        assert_eq!(parse_suffixed("K"), None);
    }
}
