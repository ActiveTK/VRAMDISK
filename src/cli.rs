//! Command-line interface for VRAMDISK.

use anyhow::{bail, Result};
use clap::Parser;

/// VRAMDISK: GPU as a Storage. Exposes GPU VRAM as a Windows drive.
#[derive(Parser, Debug)]
#[command(name = "vramdisk", version, about)]
pub struct Cli {
    /// Total size of the VRAM-backed disk, e.g. `2GB`, `512MB`, `4GiB`.
    ///
    /// Rounded up to a multiple of the 64KiB chunk size. When omitted the
    /// default is `max(0.8 * GPU[0] VRAM, 2GiB)`.
    #[arg(short, long, value_parser = parse_size)]
    pub size: Option<u64>,

    /// Enable per-chunk GPU compression (LZ4/zstd/none chosen per chunk).
    #[arg(short, long, default_value_t = false)]
    pub compress: bool,

    /// Enable chunk-level deduplication (identical 64KiB chunks share storage).
    #[arg(short, long, default_value_t = false)]
    pub dedup: bool,

    /// Drive letter / mount point to expose the disk on (Windows).
    #[arg(short, long, default_value = "R:")]
    pub mount: String,

    /// CUDA device ordinal to allocate the buffer on.
    #[arg(long, default_value_t = 0)]
    pub device: usize,

    /// Run a comprehensive speed benchmark (VRAM bandwidth, engine throughput,
    /// compression, and GPU hashing) then exit.
    ///
    /// Cannot be combined with --compress, --dedup, or --mount.
    /// --size and --device are still respected.
    #[arg(long, conflicts_with_all = ["compress", "dedup", "mount"])]
    pub bench: bool,

    /// Mount VRAMDISK internally and run filesystem I/O benchmarks for all
    /// compression/deduplication combinations, then exit.
    #[arg(long, conflicts_with_all = ["bench", "compress", "dedup", "size", "mount"])]
    pub bench_io: bool,
}

/// Parse a human-friendly byte size such as `2GB`, `512MiB`, `1048576`.
///
/// Both decimal (KB/MB/GB = 1000^n) and binary (KiB/MiB/GiB = 1024^n)
/// suffixes are accepted, case-insensitively. A bare number is bytes.
pub fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty size");
    }

    let split = s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let num: f64 = num
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid number in size: {s:?}"))?;
    if num < 0.0 {
        bail!("size must be non-negative: {s:?}");
    }

    let mult: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" => 1e3,
        "kib" => 1024.0,
        "m" | "mb" => 1e6,
        "mib" => 1024f64.powi(2),
        "g" | "gb" => 1e9,
        "gib" => 1024f64.powi(3),
        "t" | "tb" => 1e12,
        "tib" => 1024f64.powi(4),
        other => bail!("unknown size unit: {other:?}"),
    };

    Ok((num * mult) as u64)
}

/// Flags recognized from a raw argv, used to seed the GUI's setup screen
/// (see the GUI's `main()` and the `initial_overrides` Tauri command) rather
/// than to actually run anything. Every field is `None`/absent unless that
/// specific flag was found, so a bare `--compress` doesn't also reset
/// `mount`/`device` back to their clap defaults — only what was actually
/// typed should override the GUI's saved/default values.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SeedOverrides {
    pub mount: Option<String>,
    pub size: Option<u64>,
    pub compress: Option<bool>,
    pub dedup: Option<bool>,
    pub device: Option<usize>,
}

/// Scan a raw argv (no program name) for the flags [`Cli`] also understands,
/// ignoring anything unrecognized or malformed instead of erroring — this is
/// a best-effort seed, not validation, so a typo should never stop the GUI
/// from opening.
pub fn scan_overrides(args: &[String]) -> SeedOverrides {
    let mut out = SeedOverrides::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--mount" => {
                if let Some(v) = args.get(i + 1) {
                    out.mount = Some(v.clone());
                    i += 1;
                }
            }
            "-s" | "--size" => {
                if let Some(v) = args.get(i + 1) {
                    if let Ok(bytes) = parse_size(v) {
                        out.size = Some(bytes);
                    }
                    i += 1;
                }
            }
            "-c" | "--compress" => out.compress = Some(true),
            "-d" | "--dedup" => out.dedup = Some(true),
            "--device" => {
                if let Some(v) = args.get(i + 1) {
                    if let Ok(n) = v.parse::<usize>() {
                        out.device = Some(n);
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// Format a byte count for human-readable logs. Uses binary (1024-based)
/// math but Windows-style unit labels (B/KB/MB/GB/TB/PB), matching the GUI
/// and the size-picker's MB/GB dropdown — never IEC KiB/MiB/GiB labels.
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{bytes} {}", UNITS[0])
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_units() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_size("2GiB").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("1GB").unwrap(), 1_000_000_000);
        assert_eq!(
            parse_size("1.5MiB").unwrap(),
            (1.5 * 1024.0 * 1024.0) as u64
        );
        assert_eq!(parse_size("64kib").unwrap(), 65536);
    }

    #[test]
    fn rejects_bad() {
        assert!(parse_size("").is_err());
        assert!(parse_size("abc").is_err());
        assert!(parse_size("-5MB").is_err());
        assert!(parse_size("10QB").is_err());
    }

    #[test]
    fn scan_overrides_only_sets_whats_present() {
        let out = scan_overrides(&["--compress".to_string()]);
        assert_eq!(
            out,
            SeedOverrides {
                compress: Some(true),
                ..Default::default()
            }
        );
    }

    #[test]
    fn scan_overrides_reads_values() {
        let args = [
            "--mount", "G:", "--size", "4GiB", "--dedup", "--device", "1",
        ]
        .map(String::from);
        let out = scan_overrides(&args);
        assert_eq!(
            out,
            SeedOverrides {
                mount: Some("G:".to_string()),
                size: Some(4 * 1024 * 1024 * 1024),
                compress: None,
                dedup: Some(true),
                device: Some(1),
            }
        );
    }

    #[test]
    fn scan_overrides_ignores_garbage() {
        let out = scan_overrides(&["cli".to_string(), "--nonsense".to_string()]);
        assert_eq!(out, SeedOverrides::default());
    }

    #[test]
    fn formats() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }
}
