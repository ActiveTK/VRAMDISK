//! VRAMDISK: GPU as a Storage.
//!
//! Allocates a large contiguous buffer in GPU VRAM and exposes it as a
//! Windows drive via WinFsp, with optional per-chunk GPU compression and
//! chunk-level deduplication.
//!
//! This is a library-only crate: the single distributed binary is the GUI
//! (`src-tauri/`), which links this crate directly. `cli_run` holds the
//! former standalone CLI's logic, invoked by the GUI when launched as
//! `vramdisk.exe cli ...` / `vramdisk.exe benchmark ...` instead of shelling
//! out to a second binary.

pub mod api_kernel;
pub mod arena;
pub mod bench;
pub mod chunk;
pub mod cli;
pub mod cli_run;
pub mod cuda;
pub mod engine;
#[cfg(windows)]
pub mod fs;
pub mod gpu_hash;
pub mod internal_api;
pub mod jobs;
pub mod lookup;
pub mod nvcomp;

/// Logical chunk size: 64 KiB. The VRAM buffer is always a multiple of this.
pub const CHUNK_SIZE: u64 = 64 * 1024;

/// Default disk size when `--size` is omitted: max(0.8 * VRAM, 2 GiB).
pub const MIN_DEFAULT_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Round `bytes` up to the next multiple of [`CHUNK_SIZE`].
pub fn round_up_to_chunk(bytes: u64) -> u64 {
    bytes.div_ceil(CHUNK_SIZE) * CHUNK_SIZE
}

/// Default disk size: `max(0.8 * total VRAM, 2 GiB)`, capped at the device's
/// total VRAM and chunk-aligned.
///
/// The 2 GiB floor exists to give small requests room on typical desktop
/// GPUs, but on a device with less VRAM than that (older/entry GPUs, laptop
/// dGPUs, MIG/vGPU slices) it must not push the default above what the
/// device actually has. So the usual `round_up_to_chunk` result is clamped
/// against a chunk-*floor* of `total_vram` (rounding down, never up), which
/// guarantees the final value can never exceed the device's real capacity.
pub fn default_size(total_vram: u64) -> u64 {
    let target = ((total_vram as f64 * 0.8) as u64).max(MIN_DEFAULT_SIZE);
    let cap = (total_vram / CHUNK_SIZE) * CHUNK_SIZE;
    round_up_to_chunk(target).min(cap)
}

/// A CUDA device the disk can be allocated on, as surfaced to the GUI.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// CUDA device ordinal (pass as `--device`).
    pub ordinal: usize,
    /// Human-readable device name.
    pub name: String,
    /// Total physical VRAM in bytes.
    pub total_vram: u64,
    /// Default disk size this device would use when `--size` is omitted.
    pub default_size: u64,
}

/// Enumerate all visible CUDA devices. Devices that fail to report their
/// capacity are skipped rather than failing the whole query, so a single bad
/// device never hides the working ones from the UI.
pub fn list_gpus() -> Vec<GpuInfo> {
    use crate::cuda::Vram;
    let count = Vram::device_count().unwrap_or(0);
    (0..count)
        .filter_map(|ordinal| {
            let total_vram = Vram::device_total_mem(ordinal).ok()?;
            let name = Vram::device_name(ordinal).unwrap_or_else(|_| format!("CUDA[{ordinal}]"));
            Some(GpuInfo {
                ordinal,
                name,
                total_vram,
                default_size: default_size(total_vram),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_rounding() {
        assert_eq!(round_up_to_chunk(0), 0);
        assert_eq!(round_up_to_chunk(1), CHUNK_SIZE);
        assert_eq!(round_up_to_chunk(CHUNK_SIZE), CHUNK_SIZE);
        assert_eq!(round_up_to_chunk(CHUNK_SIZE + 1), 2 * CHUNK_SIZE);
    }

    #[test]
    fn default_size_floor_is_2gib() {
        // GPU with more than 2 GiB but where 0.8x would undercut the floor
        // -> floored at 2 GiB.
        let vram = 2200 * 1024 * 1024;
        assert_eq!(default_size(vram), MIN_DEFAULT_SIZE);
        // Large GPU -> 0.8 * VRAM, chunk aligned.
        let vram = 24u64 * 1024 * 1024 * 1024;
        assert_eq!(
            default_size(vram),
            round_up_to_chunk((vram as f64 * 0.8) as u64)
        );
    }

    #[test]
    fn default_size_never_exceeds_total_vram() {
        // Old/entry/MIG-slice GPUs with less than the 2 GiB floor must not
        // get a default that's bigger than the device actually has.
        for vram in [0u64, 1, 4096, 256 * 1024 * 1024, 1024 * 1024 * 1024] {
            let size = default_size(vram);
            assert!(size <= vram, "default_size({vram}) = {size} exceeds VRAM");
            assert_eq!(size % CHUNK_SIZE, 0, "default_size({vram}) not chunk-aligned");
        }
    }
}
