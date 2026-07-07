//! CLI-mode entry point, invoked as `vramdisk.exe cli ...` / `vramdisk.exe
//! benchmark ...` from the merged GUI binary (see `src-tauri/src/main.rs`).
//!
//! This used to be its own `vramdisk-cli.exe` binary's `main()`; it now lives
//! here as a plain library function so the single GUI executable can run it
//! directly instead of shelling out to a second binary.

use anyhow::Result;
use clap::Parser;

use crate::cli::{format_size, Cli};
use crate::cuda::Vram;
use crate::{bench, default_size, engine, round_up_to_chunk, CHUNK_SIZE};

/// Run in CLI mode. `args` is the argv tail *after* the `cli`/`benchmark`
/// dispatch token (see the GUI's `main()`), i.e. exactly what a standalone
/// `vramdisk-cli.exe` used to receive. Returns a process exit code.
pub fn run(args: Vec<String>) -> i32 {
    // The GUI binary has the Windows subsystem (no console auto-allocated),
    // so when launched from an existing terminal for CLI use, make sure our
    // stdout/stdin are actually connected to it before printing anything.
    #[cfg(windows)]
    {
        use windows::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
        unsafe {
            let _ = AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }

    let cli_args = std::iter::once("vramdisk".to_string()).chain(args);
    let args = Cli::parse_from(cli_args);

    match run_inner(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {e:?}");
            1
        }
    }
}

fn run_inner(args: Cli) -> Result<()> {
    if args.bench_io {
        bench::run_io(args.device)?;
        return Ok(());
    }

    let total_vram = Vram::device_total_mem(args.device)?;
    let dev_name = Vram::device_name(args.device).unwrap_or_else(|_| "?".into());

    let size = match args.size {
        Some(s) => round_up_to_chunk(s),
        None => default_size(total_vram),
    };
    anyhow::ensure!(size > 0, "disk size must be greater than 0");
    anyhow::ensure!(
        size <= total_vram,
        "requested size {} exceeds device {} VRAM ({})",
        format_size(size),
        args.device,
        format_size(total_vram)
    );
    let chunks = size / CHUNK_SIZE;

    // --bench: run benchmarks and exit; does not mount a drive.
    if args.bench {
        let size = match args.size {
            Some(s) => round_up_to_chunk(s),
            None => default_size(total_vram).min(bench::DEFAULT_BENCH_SIZE),
        };
        bench::run(args.device, size)?;
        return Ok(());
    }

    println!("VRAMDISK starting");
    println!(
        "  device      : CUDA[{}] {} ({})",
        args.device,
        dev_name,
        format_size(total_vram)
    );
    println!("  mount       : {}", args.mount);
    println!(
        "  compression : {}",
        if args.compress { "on" } else { "off" }
    );
    println!("  dedup       : {}", if args.dedup { "on" } else { "off" });
    println!(
        "  disk size   : {} ({} chunks x {})",
        format_size(size),
        chunks,
        format_size(CHUNK_SIZE)
    );

    print!("  allocating VRAM... ");
    let mut vram = Vram::new(args.device, size)?;
    println!("ok");

    self_test(&mut vram)?;

    let engine = engine::StorageEngine::new(vram, args.compress, args.dedup)?;

    #[cfg(windows)]
    {
        crate::fs::run(engine, &args.mount, "VRAMDISK")?;
    }
    #[cfg(not(windows))]
    {
        let _ = engine;
        println!("\n(WinFsp mount is only available on Windows)");
    }
    Ok(())
}

/// Round-trip a small pattern through VRAM to prove the buffer is usable.
fn self_test(vram: &mut Vram) -> Result<()> {
    print!("  VRAM self-test... ");
    let pattern: Vec<u8> = (0..4096u32)
        .map(|i| i.wrapping_mul(2654435761) as u8)
        .collect();
    // Write near the start and near the end to exercise offset math.
    let tail = vram.size() - pattern.len() as u64;
    vram.write_at(0, &pattern)?;
    vram.write_at(tail, &pattern)?;

    let mut out = vec![0u8; pattern.len()];
    vram.read_at(tail, &mut out)?;
    anyhow::ensure!(out == pattern, "VRAM round-trip mismatch at tail");

    vram.zero_at(0, pattern.len() as u64)?;
    vram.read_at(0, &mut out)?;
    anyhow::ensure!(out.iter().all(|&b| b == 0), "VRAM zero_at failed");

    println!("ok");
    Ok(())
}
