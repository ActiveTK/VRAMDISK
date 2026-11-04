//! Raw block storage primitives for a future storage-device backend.
//!
//! This module is deliberately independent of the file-oriented
//! [`crate::engine::StorageEngine`]. A block device exposes one fixed LBA
//! address space; Windows owns its partition table and filesystem metadata.
//!
//! `GpuBlockStorage` is a real CUDA allocation with no RAM fallback. The
//! in-memory implementation exists only in tests and is never selected by a
//! production backend.

use std::sync::Mutex;

use anyhow::{ensure, Context, Result};

use crate::cuda::Vram;
use crate::CHUNK_SIZE;

pub const DEFAULT_LOGICAL_SECTOR_SIZE: u32 = 4096;
pub const SUPPORTED_LOGICAL_SECTOR_SIZES: [u32; 2] = [512, 4096];

/// Geometry for a fixed-capacity raw block device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDeviceConfig {
    pub capacity_bytes: u64,
    pub logical_sector_size: u32,
}

impl BlockDeviceConfig {
    pub fn new(capacity_bytes: u64, logical_sector_size: u32) -> Result<Self> {
        ensure!(capacity_bytes > 0, "block-device capacity must be non-zero");
        ensure!(
            SUPPORTED_LOGICAL_SECTOR_SIZES.contains(&logical_sector_size),
            "logical sector size must be 512 or 4096 bytes"
        );
        ensure!(
            capacity_bytes.is_multiple_of(logical_sector_size as u64),
            "capacity must be aligned to the logical sector size"
        );
        ensure!(
            capacity_bytes.is_multiple_of(CHUNK_SIZE),
            "capacity must be aligned to the 64 KiB VRAM management unit"
        );
        Ok(Self {
            capacity_bytes,
            logical_sector_size,
        })
    }

    pub fn block_count(self) -> u64 {
        self.capacity_bytes / self.logical_sector_size as u64
    }
}

/// Storage operations required by a raw disk backend.
///
/// `buffer.len()` must be exactly `block_count * logical_sector_size`.
/// Implementations must reject overflow and any range outside the configured
/// capacity. `unmap_blocks` must make subsequent reads return zeroes for the
/// unmapped range; implementations without a safe deallocation primitive may
/// implement it by writing zeroes.
pub trait BlockStorage: Send + Sync {
    fn config(&self) -> BlockDeviceConfig;

    fn read_blocks(&self, lba: u64, block_count: u32, buffer: &mut [u8]) -> Result<()>;

    fn write_blocks(&self, lba: u64, block_count: u32, buffer: &[u8]) -> Result<()>;

    fn flush_blocks(&self, lba: u64, block_count: u32) -> Result<()>;

    fn unmap_blocks(&self, lba: u64, block_count: u32) -> Result<()>;

    fn get_capacity(&self) -> u64 {
        self.config().capacity_bytes
    }
}

fn checked_byte_range(
    config: BlockDeviceConfig,
    lba: u64,
    block_count: u32,
    buffer_len: Option<usize>,
) -> Result<(u64, u64)> {
    let sector = config.logical_sector_size as u64;
    let byte_offset = lba
        .checked_mul(sector)
        .context("LBA-to-byte offset overflow")?;
    let byte_len = (block_count as u64)
        .checked_mul(sector)
        .context("block-count-to-byte length overflow")?;
    let end = byte_offset
        .checked_add(byte_len)
        .context("block range overflow")?;
    ensure!(end <= config.capacity_bytes, "block range exceeds capacity");
    if let Some(actual_len) = buffer_len {
        let expected = usize::try_from(byte_len).context("I/O length does not fit usize")?;
        ensure!(
            actual_len == expected,
            "I/O buffer length {actual_len} does not match expected length {expected}"
        );
    }
    Ok((byte_offset, byte_len))
}

/// GPU-backed raw byte-addressed block storage.
///
/// The CUDA allocation is zero-initialized by `Vram::new`. A mutex serializes
/// transfers through the allocation because `Vram` exposes mutable write
/// methods and one CUDA context/stream. There is no compression, deduplication,
/// or system-RAM fallback in this path.
pub struct GpuBlockStorage {
    config: BlockDeviceConfig,
    device_ordinal: usize,
    vram: Mutex<Vram>,
}

impl GpuBlockStorage {
    pub fn new(device_ordinal: usize, config: BlockDeviceConfig) -> Result<Self> {
        let vram = Vram::new(device_ordinal, config.capacity_bytes)
            .with_context(|| format!("allocate {} bytes on CUDA device {device_ordinal}", config.capacity_bytes))?;
        ensure!(
            vram.size() == config.capacity_bytes,
            "CUDA allocation size did not match block-device capacity"
        );
        Ok(Self {
            config,
            device_ordinal,
            vram: Mutex::new(vram),
        })
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}

impl BlockStorage for GpuBlockStorage {
    fn config(&self) -> BlockDeviceConfig {
        self.config
    }

    fn read_blocks(&self, lba: u64, block_count: u32, buffer: &mut [u8]) -> Result<()> {
        let (offset, _) = checked_byte_range(self.config, lba, block_count, Some(buffer.len()))?;
        let vram = self.vram.lock().map_err(|_| anyhow::anyhow!("GPU block storage lock poisoned"))?;
        vram.read_at(offset, buffer).context("read block data from VRAM")
    }

    fn write_blocks(&self, lba: u64, block_count: u32, buffer: &[u8]) -> Result<()> {
        let (offset, _) = checked_byte_range(self.config, lba, block_count, Some(buffer.len()))?;
        let mut vram = self.vram.lock().map_err(|_| anyhow::anyhow!("GPU block storage lock poisoned"))?;
        vram.write_at(offset, buffer).context("write block data to VRAM")
    }

    fn flush_blocks(&self, lba: u64, block_count: u32) -> Result<()> {
        checked_byte_range(self.config, lba, block_count, None)?;
        let vram = self.vram.lock().map_err(|_| anyhow::anyhow!("GPU block storage lock poisoned"))?;
        vram.sync().context("flush VRAM block writes")
    }

    fn unmap_blocks(&self, lba: u64, block_count: u32) -> Result<()> {
        let (offset, length) = checked_byte_range(self.config, lba, block_count, None)?;
        let mut vram = self.vram.lock().map_err(|_| anyhow::anyhow!("GPU block storage lock poisoned"))?;
        vram.zero_at(offset, length).context("zero unmapped VRAM blocks")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Test-only backend; production code cannot silently substitute system
    /// RAM for an unavailable GPU allocation.
    struct MemoryBlockStorage {
        config: BlockDeviceConfig,
        data: Mutex<Vec<u8>>,
    }

    impl MemoryBlockStorage {
        fn new(config: BlockDeviceConfig) -> Self {
            Self {
                config,
                data: Mutex::new(vec![0; config.capacity_bytes as usize]),
            }
        }
    }

    impl BlockStorage for MemoryBlockStorage {
        fn config(&self) -> BlockDeviceConfig {
            self.config
        }

        fn read_blocks(&self, lba: u64, block_count: u32, buffer: &mut [u8]) -> Result<()> {
            let (offset, length) =
                checked_byte_range(self.config, lba, block_count, Some(buffer.len()))?;
            let data = self.data.lock().map_err(|_| anyhow::anyhow!("test backend lock poisoned"))?;
            let start = offset as usize;
            let end = start + length as usize;
            buffer.copy_from_slice(&data[start..end]);
            Ok(())
        }

        fn write_blocks(&self, lba: u64, block_count: u32, buffer: &[u8]) -> Result<()> {
            let (offset, length) =
                checked_byte_range(self.config, lba, block_count, Some(buffer.len()))?;
            let mut data = self.data.lock().map_err(|_| anyhow::anyhow!("test backend lock poisoned"))?;
            let start = offset as usize;
            let end = start + length as usize;
            data[start..end].copy_from_slice(buffer);
            Ok(())
        }

        fn flush_blocks(&self, lba: u64, block_count: u32) -> Result<()> {
            checked_byte_range(self.config, lba, block_count, None)?;
            Ok(())
        }

        fn unmap_blocks(&self, lba: u64, block_count: u32) -> Result<()> {
            let (offset, length) = checked_byte_range(self.config, lba, block_count, None)?;
            let mut data = self.data.lock().map_err(|_| anyhow::anyhow!("test backend lock poisoned"))?;
            let start = offset as usize;
            let end = start + length as usize;
            data[start..end].fill(0);
            Ok(())
        }
    }

    fn test_config() -> BlockDeviceConfig {
        BlockDeviceConfig::new(256 * 1024, 4096).unwrap()
    }

    #[test]
    fn capacity_and_sector_geometry_are_validated() {
        assert_eq!(test_config().block_count(), 64);
        assert!(BlockDeviceConfig::new(0, 4096).is_err());
        assert!(BlockDeviceConfig::new(128 * 1024, 1024).is_err());
        assert!(BlockDeviceConfig::new(256 * 1024 + 512, 512).is_err());
        assert!(BlockDeviceConfig::new(64 * 1024, 512).is_ok());
    }

    #[test]
    fn zero_initialization_and_sequential_round_trip() {
        let disk = MemoryBlockStorage::new(test_config());
        let mut initial = vec![0xff; 3 * 4096];
        disk.read_blocks(0, 3, &mut initial).unwrap();
        assert!(initial.iter().all(|&b| b == 0));

        let write: Vec<u8> = (0..3 * 4096).map(|i| (i * 31) as u8).collect();
        disk.write_blocks(2, 3, &write).unwrap();
        disk.flush_blocks(2, 3).unwrap();
        let mut read = vec![0; write.len()];
        disk.read_blocks(2, 3, &mut read).unwrap();
        assert_eq!(read, write);
    }

    #[test]
    fn io_crosses_64k_management_boundary_and_supports_both_sector_sizes() {
        for sector in SUPPORTED_LOGICAL_SECTOR_SIZES {
            let disk = MemoryBlockStorage::new(
                BlockDeviceConfig::new(128 * 1024, sector).unwrap(),
            );
            let lba = (64 * 1024 / sector) as u64 - 1;
            let input = vec![0x5a; sector as usize * 2];
            disk.write_blocks(lba, 2, &input).unwrap();
            let mut output = vec![0; input.len()];
            disk.read_blocks(lba, 2, &mut output).unwrap();
            assert_eq!(output, input);
        }
    }

    #[test]
    fn rejects_capacity_overrun_and_wrong_buffer_lengths() {
        let disk = MemoryBlockStorage::new(test_config());
        assert!(disk.read_blocks(63, 2, &mut vec![0; 8192]).is_err());
        assert!(disk.write_blocks(0, 2, &[0; 4096]).is_err());
        assert!(disk.flush_blocks(u64::MAX, 1).is_err());
        assert!(disk.unmap_blocks(64, 1).is_err());
    }

    #[test]
    fn unmap_reads_as_zero() {
        let disk = MemoryBlockStorage::new(test_config());
        disk.write_blocks(7, 2, &[0xa5; 8192]).unwrap();
        disk.unmap_blocks(7, 2).unwrap();
        let mut read = vec![1; 8192];
        disk.read_blocks(7, 2, &mut read).unwrap();
        assert!(read.iter().all(|&b| b == 0));
    }

    #[test]
    fn concurrent_disjoint_writes_remain_consistent() {
        let disk = Arc::new(MemoryBlockStorage::new(test_config()));
        let workers: Vec<_> = (0..8u64)
            .map(|worker| {
                let disk = Arc::clone(&disk);
                thread::spawn(move || {
                    let data = vec![worker as u8 + 1; 4096];
                    disk.write_blocks(worker, 1, &data).unwrap();
                    data
                })
            })
            .collect();
        for (worker, handle) in workers.into_iter().enumerate() {
            let expected = handle.join().unwrap();
            let mut actual = vec![0; 4096];
            disk.read_blocks(worker as u64, 1, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }
}
