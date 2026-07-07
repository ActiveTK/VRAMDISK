//! Storage engine: binds the namespace, the chunk allocator and VRAM into
//! byte-range file I/O.
//!
//! Each logical 64KiB chunk maps to one physical chunk (`Placement::Raw`) or
//! to a sparse hole (`None`) that reads as zeros. With dedup enabled, identical
//! full chunks share one physical chunk via a content-hash reverse index and
//! reference counts; any partial modification copies-on-write so sharers are
//! never disturbed. Compression hooks in later by producing other `Placement`
//! variants.

use std::collections::{BTreeSet, HashMap};
use std::thread;
use std::time::Instant;

use crate::api_kernel::{ApiKernel, HashAlgorithm, HashSegment};
use crate::arena::CompressedAllocator;
use crate::chunk::{ChunkAllocator, ChunkId};
use crate::cuda::Vram;
use crate::gpu_hash::GpuHasher;
use crate::lookup::{Codec, LookupError, LookupTable, Node, Placement};
use crate::nvcomp::{Lz4Codec, NvcompBatchedCodec, NvcompFrameCodec};
use crate::CHUNK_SIZE;

const ZIP_DEFLATE_CHUNK: u64 = 1024 * 1024;

/// Minimum number of non-zero bytes in a 64 KiB chunk before compression is
/// attempted. Below this the payload is so sparse that the per-call overhead
/// of launching a GPU kernel or zstd isn't worth it.
const MIN_COMPRESS_NONZERO: usize = 1024;

/// Shannon entropy threshold (bits/byte) above which a chunk is considered
/// already compressed or effectively random. Re-compressing it would expand the
/// data. Truly random data reaches 8.0; practical compressed payloads sit in
/// the 7.4–7.9 range.
const ENTROPY_SKIP_THRESHOLD: f64 = 7.2;

/// Number of 256-byte windows sampled evenly across the chunk for the entropy
/// estimate. More windows → more accurate but slightly more CPU time.
const ENTROPY_WINDOWS: usize = 8;

/// Compute the Shannon entropy (bits/byte) of a 256-byte window.
fn window_entropy(window: &[u8]) -> f64 {
    let mut freq = [0u32; 256];
    for &b in window {
        freq[b as usize] += 1;
    }
    let n = window.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Return `true` when compressing `data` is unlikely to be beneficial:
/// (i) the chunk has too little actual content, or
/// (ii) sampled entropy suggests the data is already compressed / random.
fn should_skip_compression(data: &[u8]) -> bool {
    let non_zero = data.iter().filter(|&&b| b != 0).count();
    if non_zero < MIN_COMPRESS_NONZERO {
        return true;
    }
    let win = 256usize;
    let step = (data.len().saturating_sub(win)) / ENTROPY_WINDOWS;
    let avg_entropy: f64 = (0..ENTROPY_WINDOWS)
        .map(|i| window_entropy(&data[i * step..i * step + win]))
        .sum::<f64>()
        / ENTROPY_WINDOWS as f64;
    avg_entropy >= ENTROPY_SKIP_THRESHOLD
}

fn should_skip_batch_compression(data: &[u8]) -> bool {
    let win = 256usize;
    let step = (data.len().saturating_sub(win)) / ENTROPY_WINDOWS;
    let avg_entropy: f64 = (0..ENTROPY_WINDOWS)
        .map(|i| window_entropy(&data[i * step..i * step + win]))
        .sum::<f64>()
        / ENTROPY_WINDOWS as f64;
    avg_entropy >= ENTROPY_SKIP_THRESHOLD
}

fn group_looks_incompressible(group: &[u8], chunks: usize) -> bool {
    if chunks == 0 {
        return false;
    }
    let probes = [0usize, chunks / 3, (chunks * 2) / 3, chunks - 1];
    let mut seen = Vec::new();
    let mut high_entropy = 0usize;
    for &chunk in &probes {
        if seen.contains(&chunk) {
            continue;
        }
        seen.push(chunk);
        let s = chunk * CHUNK_SIZE as usize;
        if should_skip_batch_compression(&group[s..s + CHUNK_SIZE as usize]) {
            high_entropy += 1;
        }
    }
    high_entropy == seen.len()
}

#[derive(Debug)]
pub enum EngineError {
    Lookup(LookupError),
    /// No free chunk available.
    NoSpace,
    /// Operation expected a file but found a directory.
    NotAFile,
    /// Underlying CUDA failure.
    Cuda(#[allow(dead_code)] String),
}

impl From<LookupError> for EngineError {
    fn from(e: LookupError) -> Self {
        EngineError::Lookup(e)
    }
}

pub type EResult<T> = Result<T, EngineError>;

fn cuda<T>(r: anyhow::Result<T>) -> EResult<T> {
    r.map_err(|e| EngineError::Cuda(format!("{e:#}")))
}

/// Number of logical chunks needed to hold `size` bytes.
fn logical_chunks(size: u64) -> usize {
    size.div_ceil(CHUNK_SIZE) as usize
}

fn ranges_overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b1 && b0 < a1
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub total_chunks: u32,
    pub used_chunks: u32,
    pub free_chunks: u32,
    pub total_bytes: u64,
    pub used_physical_bytes: u64,
    pub free_physical_bytes: u64,
    pub file_count: u64,
    pub dir_count: u64,
    pub logical_file_bytes: u64,
    pub logical_allocated_bytes: u64,
    pub raw_unique_chunks: u64,
    pub raw_logical_chunks: u64,
    pub compressed_logical_chunks: u64,
    pub compressed_payload_bytes: u64,
    pub sparse_logical_chunks: u64,
    pub dedup_shared_logical_chunks: u64,
    pub dedup_saved_bytes: u64,
    pub compression_saved_bytes: u64,
    pub compress_enabled: bool,
    pub dedup_enabled: bool,
    pub nvcomp_lz4_available: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineTrace {
    pub read_calls: u64,
    pub write_calls: u64,
    pub logical_read_bytes: u64,
    pub logical_write_bytes: u64,
    pub raw_read_ops: u64,
    pub raw_read_bytes: u64,
    pub raw_write_ops: u64,
    pub raw_write_bytes: u64,
    pub compressed_read_chunks: u64,
    pub compressed_read_requested_bytes: u64,
    pub compressed_read_full_bytes: u64,
    pub compress_batches: u64,
    pub compress_chunks: u64,
    pub compress_raw_fallback_chunks: u64,
    pub dedup_hash_chunks: u64,
    pub dedup_candidate_chunks: u64,
    pub dedup_shared_chunks: u64,
    pub dedup_unique_chunks: u64,
    pub gpu_hash_chunks: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileChunkReport {
    pub path: String,
    pub size: u64,
    pub chunk_size: u64,
    pub logical_chunks: u64,
    pub chunks: Vec<ChunkReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkReport {
    pub logical_chunk: u64,
    pub logical_offset: u64,
    pub logical_len: u64,
    pub placement: ChunkPlacementReport,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArchiveJobStats {
    pub format: String,
    pub output: String,
    pub file_count: usize,
    pub input_bytes: u64,
    pub archive_bytes: u64,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArchiveExtractStats {
    pub format: String,
    pub archive: String,
    pub output_dir: String,
    pub file_count: usize,
    pub archive_bytes: u64,
    pub output_bytes: u64,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkPlacementReport {
    Sparse,
    Raw {
        physical_chunk: ChunkId,
        physical_offset: u64,
        refcount: u32,
        content_hash: Option<u64>,
    },
    Compressed {
        offset: u64,
        len: u32,
        codec: Codec,
        refcount: u32,
        content_hash: Option<u64>,
    },
}

/// Two-level parallel FNV-1a 64-bit hash of a full 64 KiB chunk.
///
/// Mirrors the GPU kernel in `hash_kernel.ptx` exactly:
///   Phase 1 – 256 independent FNV-1a passes, each over a 256-byte segment.
///   Phase 2 – FNV-1a over the 256 per-segment u64 hashes (as little-endian bytes).
///
/// Identical data ⇒ identical hash on both CPU and GPU, so the CPU hash of
/// incoming host data can be compared directly with the GPU hash of an
/// existing VRAM chunk.
fn fnv1a(data: &[u8]) -> u64 {
    debug_assert_eq!(data.len(), CHUNK_SIZE as usize);
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;

    // Phase 1: hash each 256-byte segment independently.
    let mut seg_hashes = [0u64; 256];
    for t in 0..256 {
        let mut h = BASIS;
        for &b in &data[t * 256..(t + 1) * 256] {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        seg_hashes[t] = h;
    }

    // Phase 2: FNV-1a over the segment hashes (8 LE bytes each).
    let mut h = BASIS;
    for sh in seg_hashes {
        for byte_idx in 0..8u64 {
            h ^= (sh >> (byte_idx * 8)) & 0xff;
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

fn fnv1a_chunks(data: &[u8]) -> Vec<u64> {
    debug_assert_eq!(data.len() as u64 % CHUNK_SIZE, 0);
    let n = data.len() / CHUNK_SIZE as usize;
    if n <= 16 {
        return (0..n)
            .map(|i| {
                let s = i * CHUNK_SIZE as usize;
                fnv1a(&data[s..s + CHUNK_SIZE as usize])
            })
            .collect();
    }

    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n);
    let per_worker = n.div_ceil(workers);
    let mut parts = Vec::with_capacity(workers);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let start_chunk = worker * per_worker;
            if start_chunk >= n {
                break;
            }
            let end_chunk = ((worker + 1) * per_worker).min(n);
            let start = start_chunk * CHUNK_SIZE as usize;
            let end = end_chunk * CHUNK_SIZE as usize;
            handles.push(scope.spawn(move || {
                let slice = &data[start..end];
                let hashes: Vec<u64> = (0..end_chunk - start_chunk)
                    .map(|i| {
                        let s = i * CHUNK_SIZE as usize;
                        fnv1a(&slice[s..s + CHUNK_SIZE as usize])
                    })
                    .collect();
                (start_chunk, hashes)
            }));
        }
        for h in handles {
            parts.push(h.join().expect("hash worker panicked"));
        }
    });
    parts.sort_by_key(|(start, _)| *start);
    let mut out = Vec::with_capacity(n);
    for (_, hashes) in parts {
        out.extend(hashes);
    }
    out
}

pub struct StorageEngine {
    vram: Vram,
    alloc: ChunkAllocator,
    table: LookupTable,
    compress: bool,
    dedup: bool,

    // Dedup state (only populated when `dedup` is true).
    /// Reference count per physical chunk.
    refcount: Vec<u32>,
    /// Hash currently indexed for a raw physical chunk (its reverse-map key), if any.
    chunk_hash: Vec<Option<u64>>,
    /// Content hash -> placement holding that content.
    hash_index: HashMap<u64, Placement>,
    /// Reference count per shared compressed blob, keyed by absolute blob offset.
    compressed_refcount: HashMap<u64, u32>,
    /// Hash currently indexed for a compressed blob, keyed by absolute blob offset.
    compressed_hash: HashMap<u64, u64>,

    // Compression state (only populated when `compress` is true).
    /// Sub-allocator packing variable-length compressed blobs into chunks.
    carena: CompressedAllocator,
    /// Loaded nvCOMP LZ4 codec.
    codec: Option<Lz4Codec>,

    // GPU hash (populated when `dedup` is true).
    /// Raw base address of the VRAM buffer (device pointer), cached at
    /// construction. Used by the GPU hash kernel and by the compressed read path
    /// to address packed arena blobs directly (device-to-device).
    vram_base: u64,
    /// GPU kernel that hashes a 64 KiB VRAM chunk without a host round-trip.
    gpu_hasher: Option<GpuHasher>,
    /// Generic CUDA kernels used by `$VRAMDISK` virtual APIs.
    api_kernel: Option<ApiKernel>,
    trace: EngineTrace,
}

impl StorageEngine {
    pub fn new(vram: Vram, compress: bool, dedup: bool) -> anyhow::Result<Self> {
        let total = (vram.size() / CHUNK_SIZE) as u32;
        let (refcount, chunk_hash) = if dedup {
            (vec![0u32; total as usize], vec![None; total as usize])
        } else {
            (Vec::new(), Vec::new())
        };
        let codec = if compress {
            match Lz4Codec::load(&vram) {
                Ok(c) => Some(c),
                Err(e) => {
                    eprintln!("nvCOMP LZ4 unavailable ({e:#}); falling back to CPU zstd");
                    None
                }
            }
        } else {
            None
        };
        // Device base address of the VRAM buffer. Used by the GPU hasher (dedup)
        // and by the compressed read path to decompress straight from the arena.
        let vram_base = vram.buf_device_ptr();
        let gpu_hasher = if dedup {
            Some(GpuHasher::new(&vram)?)
        } else {
            None
        };
        Ok(StorageEngine {
            vram,
            alloc: ChunkAllocator::new(total),
            table: LookupTable::new(),
            compress,
            dedup,
            refcount,
            chunk_hash,
            hash_index: HashMap::new(),
            compressed_refcount: HashMap::new(),
            compressed_hash: HashMap::new(),
            carena: CompressedAllocator::new(),
            codec,
            vram_base,
            gpu_hasher,
            api_kernel: None,
            trace: EngineTrace::default(),
        })
    }

    pub fn table(&self) -> &LookupTable {
        &self.table
    }

    pub fn table_mut(&mut self) -> &mut LookupTable {
        &mut self.table
    }

    pub fn total_chunks(&self) -> u32 {
        self.alloc.total()
    }

    pub fn used_chunks(&self) -> u32 {
        self.alloc.used()
    }

    pub fn stats(&self) -> EngineStats {
        let total_chunks = self.total_chunks();
        let used_chunks = self.used_chunks();
        let mut stats = EngineStats {
            total_chunks,
            used_chunks,
            free_chunks: total_chunks - used_chunks,
            total_bytes: total_chunks as u64 * CHUNK_SIZE,
            used_physical_bytes: used_chunks as u64 * CHUNK_SIZE,
            free_physical_bytes: (total_chunks - used_chunks) as u64 * CHUNK_SIZE,
            compress_enabled: self.compress,
            dedup_enabled: self.dedup,
            nvcomp_lz4_available: self.codec.is_some(),
            ..EngineStats::default()
        };

        let mut raw_unique = BTreeSet::new();
        let mut compressed_unique = BTreeSet::new();
        for node in self.table.nodes() {
            if node.is_dir {
                stats.dir_count += 1;
                continue;
            }
            stats.file_count += 1;
            stats.logical_file_bytes += node.size;
            stats.logical_allocated_bytes += crate::round_up_to_chunk(node.size);
            for placement in &node.coords {
                match placement {
                    Some(Placement::Raw { chunk }) => {
                        stats.raw_logical_chunks += 1;
                        raw_unique.insert(*chunk);
                    }
                    Some(Placement::Compressed { offset, len, .. }) => {
                        stats.compressed_logical_chunks += 1;
                        if compressed_unique.insert(*offset) {
                            stats.compressed_payload_bytes += *len as u64;
                        }
                    }
                    None => stats.sparse_logical_chunks += 1,
                }
            }
        }

        stats.raw_unique_chunks = raw_unique.len() as u64;
        let compressed_shared = stats
            .compressed_logical_chunks
            .saturating_sub(compressed_unique.len() as u64);
        stats.dedup_shared_logical_chunks = stats
            .raw_logical_chunks
            .saturating_sub(stats.raw_unique_chunks)
            + compressed_shared;
        stats.dedup_saved_bytes = stats.dedup_shared_logical_chunks * CHUNK_SIZE;
        stats.compression_saved_bytes = (stats.compressed_logical_chunks * CHUNK_SIZE)
            .saturating_sub(stats.compressed_payload_bytes);
        stats
    }

    pub fn trace_snapshot(&self) -> EngineTrace {
        self.trace.clone()
    }

    #[allow(dead_code)]
    pub fn reset_trace(&mut self) {
        self.trace = EngineTrace::default();
    }

    pub fn get(&self, path: &str) -> Option<&Node> {
        self.table.get(path)
    }

    pub fn file_chunks(&self, path: &str) -> EResult<FileChunkReport> {
        let node = self
            .table
            .get(path)
            .ok_or(EngineError::Lookup(LookupError::NotFound))?;
        if node.is_dir {
            return Err(EngineError::NotAFile);
        }

        let logical_chunks = logical_chunks(node.size);
        let mut chunks = Vec::with_capacity(logical_chunks);
        for lc in 0..logical_chunks {
            let logical_offset = lc as u64 * CHUNK_SIZE;
            let logical_len = (node.size - logical_offset).min(CHUNK_SIZE);
            let placement = match node.coords.get(lc).copied().flatten() {
                None => ChunkPlacementReport::Sparse,
                Some(Placement::Raw { chunk }) => ChunkPlacementReport::Raw {
                    physical_chunk: chunk,
                    physical_offset: chunk as u64 * CHUNK_SIZE,
                    refcount: if self.dedup {
                        self.refcount[chunk as usize]
                    } else {
                        1
                    },
                    content_hash: if self.dedup {
                        self.chunk_hash[chunk as usize]
                    } else {
                        None
                    },
                },
                Some(Placement::Compressed { offset, len, codec }) => {
                    ChunkPlacementReport::Compressed {
                        offset,
                        len,
                        codec,
                        refcount: if self.dedup {
                            self.compressed_refcount.get(&offset).copied().unwrap_or(1)
                        } else {
                            1
                        },
                        content_hash: if self.dedup {
                            self.compressed_hash.get(&offset).copied()
                        } else {
                            None
                        },
                    }
                }
            };
            chunks.push(ChunkReport {
                logical_chunk: lc as u64,
                logical_offset,
                logical_len,
                placement,
            });
        }

        Ok(FileChunkReport {
            path: crate::lookup::normalize(path),
            size: node.size,
            chunk_size: CHUNK_SIZE,
            logical_chunks: logical_chunks as u64,
            chunks,
        })
    }

    // ---- coordinate helpers -------------------------------------------------

    fn coord(&self, path: &str, lc: usize) -> Option<Placement> {
        self.table
            .get(path)
            .and_then(|n| n.coords.get(lc).copied().flatten())
    }

    fn set_coord(&mut self, path: &str, lc: usize, p: Option<Placement>) {
        if let Some(n) = self.table.get_mut(path) {
            n.coords[lc] = p;
        }
    }

    // ---- physical chunk lifecycle ------------------------------------------

    fn alloc_chunk(&mut self) -> EResult<ChunkId> {
        let c = self.alloc.alloc_one().ok_or(EngineError::NoSpace)?;
        if self.dedup {
            self.refcount[c as usize] = 1;
        }
        Ok(c)
    }

    fn ref_inc(&mut self, c: ChunkId) {
        if self.dedup {
            self.refcount[c as usize] += 1;
        }
    }

    fn ref_inc_placement(&mut self, p: Placement) {
        match p {
            Placement::Raw { chunk } => self.ref_inc(chunk),
            Placement::Compressed { offset, .. } if self.dedup => {
                *self.compressed_refcount.entry(offset).or_insert(0) += 1;
            }
            Placement::Compressed { .. } => {}
        }
    }

    /// Drop one reference to a physical chunk, freeing it at zero.
    fn release_chunk(&mut self, c: ChunkId) {
        if self.dedup {
            self.refcount[c as usize] -= 1;
            if self.refcount[c as usize] == 0 {
                self.index_remove(c);
                self.alloc.free_one(c);
            }
        } else {
            self.alloc.free_one(c);
        }
    }

    fn index_remove(&mut self, c: ChunkId) {
        if let Some(h) = self.chunk_hash[c as usize].take() {
            if self.hash_index.get(&h) == Some(&Placement::Raw { chunk: c }) {
                self.hash_index.remove(&h);
            }
        }
    }

    fn index_insert(&mut self, c: ChunkId, h: u64) {
        self.chunk_hash[c as usize] = Some(h);
        self.hash_index.insert(h, Placement::Raw { chunk: c });
    }

    fn compressed_index_insert(&mut self, p: Placement, h: u64) {
        if let Placement::Compressed { offset, .. } = p {
            self.compressed_refcount.insert(offset, 1);
            self.compressed_hash.insert(offset, h);
            self.hash_index.insert(h, p);
        }
    }

    fn compressed_index_remove(&mut self, p: Placement) {
        if let Placement::Compressed { offset, .. } = p {
            if let Some(h) = self.compressed_hash.remove(&offset) {
                if self.hash_index.get(&h) == Some(&p) {
                    self.hash_index.remove(&h);
                }
            }
        }
    }

    /// Verify a dedup candidate by comparing the GPU hash of the stored chunk
    /// with the CPU hash of the incoming data. Both use the same two-level
    /// FNV-1a algorithm, so they produce identical values for identical content.
    ///
    /// Replaces the previous 64 KiB D2H transfer with a GPU kernel (produces
    /// 8 bytes instead of 65536 bytes of host traffic per dedup hit).
    fn verify_chunk(&mut self, c: ChunkId, expected_hash: u64) -> EResult<bool> {
        let hasher = self
            .gpu_hasher
            .as_mut()
            .expect("gpu_hasher present when dedup");
        let gpu_hash = cuda(hasher.hash_chunk(self.vram_base, c as u64 * CHUNK_SIZE))?;
        Ok(gpu_hash == expected_hash)
    }

    fn verify_placement(&mut self, p: Placement, expected_hash: u64) -> EResult<bool> {
        match p {
            Placement::Raw { chunk } => self.verify_chunk(chunk, expected_hash),
            Placement::Compressed { offset, .. } => {
                Ok(self.compressed_hash.get(&offset) == Some(&expected_hash))
            }
        }
    }

    fn try_share_hashed(&mut self, path: &str, lc: usize, h: u64) -> EResult<bool> {
        let Some(cand) = self.hash_index.get(&h).copied() else {
            return Ok(false);
        };
        if !self.verify_placement(cand, h)? {
            return Ok(false);
        }
        self.ref_inc_placement(cand);
        if let Some(p) = self.coord(path, lc) {
            self.free_placement(p);
        }
        self.set_coord(path, lc, Some(cand));
        Ok(true)
    }

    /// Ensure logical chunk `lc` of `path` is backed by a physical chunk that
    /// this file exclusively owns, copying-on-write if it is currently shared.
    /// Returns that chunk id. The chunk is removed from the dedup index since
    /// it is about to be mutated in place.
    fn make_exclusive(&mut self, path: &str, lc: usize) -> EResult<ChunkId> {
        match self.coord(path, lc) {
            Some(Placement::Raw { chunk }) => {
                if self.dedup && self.refcount[chunk as usize] > 1 {
                    let nc = self.alloc_chunk()?;
                    cuda(self.vram.copy_within(
                        chunk as u64 * CHUNK_SIZE,
                        nc as u64 * CHUNK_SIZE,
                        CHUNK_SIZE,
                    ))?;
                    self.release_chunk(chunk);
                    self.set_coord(path, lc, Some(Placement::Raw { chunk: nc }));
                    Ok(nc)
                } else {
                    if self.dedup {
                        self.index_remove(chunk);
                    }
                    Ok(chunk)
                }
            }
            None => {
                let nc = self.alloc_chunk()?;
                cuda(self.vram.zero_at(nc as u64 * CHUNK_SIZE, CHUNK_SIZE))?;
                self.set_coord(path, lc, Some(Placement::Raw { chunk: nc }));
                Ok(nc)
            }
            Some(Placement::Compressed { offset, len, codec }) => {
                // Decompress the blob into a fresh exclusively-owned Raw chunk so
                // the caller can modify it in place.
                let data = self.decompress_blob(offset, len, codec)?;
                let nc = self.alloc_chunk()?;
                cuda(self.vram.write_at(nc as u64 * CHUNK_SIZE, &data))?;
                self.free_placement(Placement::Compressed { offset, len, codec });
                self.set_coord(path, lc, Some(Placement::Raw { chunk: nc }));
                Ok(nc)
            }
        }
    }

    fn free_coords(&mut self, coords: &[Option<Placement>]) {
        for p in coords.iter().flatten() {
            self.free_placement(*p);
        }
    }

    /// Release the storage backing one placement (a physical chunk, or a
    /// packed compressed region whose arena may then be reclaimed).
    fn free_placement(&mut self, p: Placement) {
        match p {
            Placement::Raw { chunk } => self.release_chunk(chunk),
            Placement::Compressed { offset, len, codec } => {
                if self.dedup {
                    if let Some(rc) = self.compressed_refcount.get_mut(&offset) {
                        *rc -= 1;
                        if *rc > 0 {
                            return;
                        }
                        self.compressed_refcount.remove(&offset);
                        self.compressed_index_remove(Placement::Compressed { offset, len, codec });
                    }
                }
                if let Some(freed_chunk) = self.carena.free(offset, len) {
                    self.alloc.free_one(freed_chunk);
                }
            }
        }
    }

    /// Allocate `len` bytes of packed storage for a compressed blob, grabbing a
    /// fresh chunk from the bitmap when no arena has room.
    fn carena_alloc(&mut self, len: u32) -> EResult<u64> {
        if let Some(off) = self.carena.try_alloc(len) {
            return Ok(off);
        }
        let chunk = self.alloc.alloc_one().ok_or(EngineError::NoSpace)?;
        Ok(self.carena.add_arena(chunk, len))
    }

    /// Decompress a packed blob at `offset`/`len` back to a full 64KiB chunk.
    fn decompress_blob(&mut self, offset: u64, len: u32, codec: Codec) -> EResult<Vec<u8>> {
        let mut buf = vec![0u8; len as usize];
        cuda(self.vram.read_at(offset, &mut buf))?;
        match codec {
            Codec::Lz4 => {
                let lz4 = self
                    .codec
                    .as_mut()
                    .expect("nvCOMP codec present when compress=true");
                cuda(lz4.decompress(&buf, CHUNK_SIZE as usize))
            }
            Codec::Zstd => {
                zstd::decode_all(buf.as_slice()).map_err(|e| EngineError::Cuda(e.to_string()))
            }
        }
    }

    /// Materialize the full 64KiB content of logical chunk `lc` (zeros for a
    /// hole, raw read, or decompressed blob).
    fn read_logical_chunk(&mut self, path: &str, lc: usize) -> EResult<Vec<u8>> {
        match self.coord(path, lc) {
            None => Ok(vec![0u8; CHUNK_SIZE as usize]),
            Some(Placement::Raw { chunk }) => {
                let mut b = vec![0u8; CHUNK_SIZE as usize];
                cuda(self.vram.read_at(chunk as u64 * CHUNK_SIZE, &mut b))?;
                Ok(b)
            }
            Some(Placement::Compressed { offset, len, codec }) => {
                self.decompress_blob(offset, len, codec)
            }
        }
    }

    fn api_kernel(&mut self) -> EResult<&mut ApiKernel> {
        if self.api_kernel.is_none() {
            self.api_kernel = Some(cuda(ApiKernel::new(&self.vram))?);
        }
        Ok(self.api_kernel.as_mut().unwrap())
    }

    fn update_api_hash(&mut self, segments: &[HashSegment]) -> EResult<()> {
        cuda(self.api_kernel()?.update(segments))
    }

    /// Hash a file using CUDA-resident data only. Raw chunks are read in place
    /// from the VRAM buffer; LZ4-compressed chunks are decompressed by nvCOMP
    /// into device scratch and immediately consumed by the API hash kernel.
    /// Sparse holes are represented as zero segments and generated on GPU.
    ///
    /// CPU zstd fallback chunks cannot satisfy the GPU-only contract and are
    /// rejected instead of silently pulling file bytes through host memory.
    pub fn hash_file_gpu(&mut self, path: &str, alg: HashAlgorithm) -> EResult<Vec<u8>> {
        let (size, is_dir) = {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            (node.size, node.is_dir)
        };
        if is_dir {
            return Err(EngineError::NotAFile);
        }

        cuda(self.api_kernel()?.begin(alg))?;
        if size == 0 {
            return cuda(self.api_kernel()?.finish(alg));
        }

        let max_segments = self.api_kernel()?.max_segments().min(crate::nvcomp::BATCH);
        let logical = logical_chunks(size);
        let mut lc = 0usize;
        while lc < logical {
            let mut segs = Vec::with_capacity(max_segments);

            while lc < logical && segs.len() < max_segments {
                let in_file_off = lc as u64 * CHUNK_SIZE;
                let take = (size - in_file_off).min(CHUNK_SIZE) as u32;
                match self.coord(path, lc) {
                    None => segs.push(HashSegment {
                        ptr: 0,
                        len: take,
                        kind: 1,
                    }),
                    Some(Placement::Raw { chunk }) => segs.push(HashSegment {
                        ptr: self.vram_base + chunk as u64 * CHUNK_SIZE,
                        len: take,
                        kind: 0,
                    }),
                    Some(Placement::Compressed {
                        codec: Codec::Lz4, ..
                    }) => break,
                    Some(Placement::Compressed {
                        codec: Codec::Zstd, ..
                    }) => {
                        return Err(EngineError::Cuda(
                            "GPU-only hash is unavailable for CPU zstd fallback chunks".into(),
                        ));
                    }
                }
                lc += 1;
            }

            if !segs.is_empty() {
                self.update_api_hash(&segs)?;
                continue;
            }

            let mut blobs = Vec::with_capacity(max_segments);
            let start_lc = lc;
            while lc < logical && blobs.len() < max_segments {
                match self.coord(path, lc) {
                    Some(Placement::Compressed {
                        offset,
                        len,
                        codec: Codec::Lz4,
                    }) => {
                        blobs.push((offset, len));
                        lc += 1;
                    }
                    Some(Placement::Compressed {
                        codec: Codec::Zstd, ..
                    }) => {
                        return Err(EngineError::Cuda(
                            "GPU-only hash is unavailable for CPU zstd fallback chunks".into(),
                        ));
                    }
                    _ => break,
                }
            }

            if blobs.is_empty() {
                continue;
            }
            let vram_base = self.vram_base;
            let codec = self
                .codec
                .as_mut()
                .ok_or_else(|| EngineError::Cuda("LZ4 chunk without nvCOMP codec".into()))?;
            cuda(codec.decompress_from_arena_dev(vram_base, &blobs))?;

            let mut comp_segs = Vec::with_capacity(blobs.len());
            for i in 0..blobs.len() {
                let chunk_lc = start_lc + i;
                let in_file_off = chunk_lc as u64 * CHUNK_SIZE;
                let take = (size - in_file_off).min(CHUNK_SIZE) as u32;
                comp_segs.push(HashSegment {
                    ptr: codec.uncomp_slot_ptr(i),
                    len: take,
                    kind: 0,
                });
            }
            self.update_api_hash(&comp_segs)?;
        }

        cuda(self.api_kernel()?.finish(alg))
    }

    pub fn hash_files_gpu_many(
        &mut self,
        paths: &[String],
        alg: HashAlgorithm,
    ) -> EResult<Vec<Vec<u8>>> {
        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            let (size, is_dir) = {
                let node = self.table.get(path).ok_or(LookupError::NotFound)?;
                (node.size, node.is_dir)
            };
            if is_dir {
                return Err(EngineError::NotAFile);
            }

            let logical = logical_chunks(size);
            let mut segs = Vec::with_capacity(logical);
            for lc in 0..logical {
                let in_file_off = lc as u64 * CHUNK_SIZE;
                let take = (size - in_file_off).min(CHUNK_SIZE) as u32;
                match self.coord(path, lc) {
                    None => segs.push(HashSegment {
                        ptr: 0,
                        len: take,
                        kind: 1,
                    }),
                    Some(Placement::Raw { chunk }) => segs.push(HashSegment {
                        ptr: self.vram_base + chunk as u64 * CHUNK_SIZE,
                        len: take,
                        kind: 0,
                    }),
                    Some(Placement::Compressed { .. }) => {
                        return Err(EngineError::Cuda(
                            "batched GPU hash currently requires raw/sparse file placements".into(),
                        ));
                    }
                }
            }
            files.push(segs);
        }
        cuda(self.api_kernel()?.hash_many(alg, &files))
    }

    /// Store a full 64KiB chunk for logical position `lc`, compressing it when
    /// beneficial.
    ///
    /// Skips compression when the chunk has little content or high entropy
    /// (already-compressed / random data). Otherwise tries LZ4 via GPU nvCOMP;
    /// if nvCOMP is unavailable falls back to CPU zstd. Stores raw when the
    /// compressed result is not smaller than the input.
    fn store_compressed(
        &mut self,
        path: &str,
        lc: usize,
        full: &[u8],
        content_hash: Option<u64>,
    ) -> EResult<()> {
        debug_assert_eq!(full.len(), CHUNK_SIZE as usize);
        let old = self.coord(path, lc);

        // Attempt compression; returns `Some((bytes, codec))` only when the
        // result is strictly smaller than the input.
        let comp: Option<(Vec<u8>, Codec)> = if should_skip_compression(full) {
            None
        } else if let Some(lz4) = self.codec.as_mut() {
            // Primary path: GPU LZ4 via nvCOMP.
            cuda(lz4.compress(full))?.map(|b| (b, Codec::Lz4))
        } else {
            // Fallback: CPU zstd (nvCOMP not available on this machine).
            zstd::encode_all(full, 3)
                .ok()
                .filter(|b| b.len() < full.len())
                .map(|b| (b, Codec::Zstd))
        };

        self.place_chunk(path, lc, full, old, comp, content_hash)
    }

    /// Commit the result of compressing logical chunk `lc` to storage: pack the
    /// blob into the arena when it compressed, otherwise store `full` raw. Frees
    /// whatever `old` placement the chunk previously had.
    fn place_chunk(
        &mut self,
        path: &str,
        lc: usize,
        full: &[u8],
        old: Option<Placement>,
        comp: Option<(Vec<u8>, Codec)>,
        content_hash: Option<u64>,
    ) -> EResult<()> {
        match comp {
            Some((bytes, codec)) => {
                let off = self.carena_alloc(bytes.len() as u32)?;
                cuda(self.vram.write_at(off, &bytes))?;
                self.trace.compress_batches += 1;
                self.trace.compress_chunks += 1;
                if let Some(p) = old {
                    self.free_placement(p);
                }
                let p = Placement::Compressed {
                    offset: off,
                    len: bytes.len() as u32,
                    codec,
                };
                self.set_coord(path, lc, Some(p));
                if self.dedup {
                    if let Some(h) = content_hash {
                        self.compressed_index_insert(p, h);
                    }
                }
            }
            None => {
                // Low-content, high-entropy, or incompressible: store raw.
                // When dedup is also active, go through alloc_chunk() so the
                // refcount slot is initialised (release_chunk expects it at 1).
                let chunk = if self.dedup {
                    self.alloc_chunk()?
                } else {
                    self.alloc.alloc_one().ok_or(EngineError::NoSpace)?
                };
                cuda(self.vram.write_at(chunk as u64 * CHUNK_SIZE, full))?;
                self.trace.raw_write_ops += 1;
                self.trace.raw_write_bytes += CHUNK_SIZE;
                if self.compress {
                    self.trace.compress_raw_fallback_chunks += 1;
                }
                if let Some(p) = old {
                    self.free_placement(p);
                }
                self.set_coord(path, lc, Some(Placement::Raw { chunk }));
                if self.dedup {
                    if let Some(h) = content_hash {
                        self.index_insert(chunk, h);
                    }
                }
            }
        }
        Ok(())
    }

    // ---- public file operations --------------------------------------------

    /// Remove a file/empty dir and free any chunks it owned.
    pub fn remove(&mut self, path: &str) -> EResult<()> {
        let node = self.table.remove(path)?;
        self.free_coords(&node.coords);
        Ok(())
    }

    /// Read up to `len` bytes from `path` starting at `offset`.
    ///
    /// Takes `&mut self` because decompression mutates the codec's device
    /// scratch; WinFsp serialises callbacks so this never aliases.
    pub fn read(&mut self, path: &str, offset: u64, len: usize) -> EResult<Vec<u8>> {
        let (size, is_dir) = {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            (node.size, node.is_dir)
        };
        if is_dir {
            return Err(EngineError::NotAFile);
        }
        if offset >= size || len == 0 {
            return Ok(Vec::new());
        }
        let n = ((size - offset).min(len as u64)) as usize;
        let mut out = vec![0u8; n];
        let got = self.read_into(path, offset, &mut out)?;
        out.truncate(got);
        Ok(out)
    }

    /// Read up to `buf.len()` bytes at `offset` directly into `buf`, returning
    /// the number of bytes written (`min(buf.len(), size - offset)`).
    ///
    /// Unlike [`read`](Self::read), this fills a caller-provided buffer instead
    /// of allocating and returning a fresh `Vec`. The WinFsp read callback
    /// passes its own output buffer straight through, so a read no longer pays
    /// for a zero-initialized allocation *and* a second full-size `memcpy` back
    /// into WinFsp's buffer — the device-to-host transfer lands in the final
    /// destination directly. Sparse holes are zeroed explicitly here because a
    /// caller buffer isn't pre-zeroed like the `Vec` path's was.
    pub fn read_into(&mut self, path: &str, offset: u64, buf: &mut [u8]) -> EResult<usize> {
        let (size, is_dir) = {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            (node.size, node.is_dir)
        };
        if is_dir {
            return Err(EngineError::NotAFile);
        }
        if offset >= size || buf.is_empty() {
            return Ok(0);
        }
        let n = ((size - offset).min(buf.len() as u64)) as usize;
        let out = &mut buf[..n];
        self.trace.read_calls += 1;
        self.trace.logical_read_bytes += n as u64;

        // LZ4-compressed chunks touched by this read are gathered and decompressed
        // in batched nvCOMP calls reading the packed blobs straight from the VRAM
        // arena (device-to-device, no host upload), rather than one launch per
        // chunk. Each entry records the arena location and where the decompressed
        // bytes land. (Only Lz4 lands here; Zstd — the nvCOMP-absent fallback — is
        // rare and decoded inline on the CPU below.)
        struct Pending {
            out_pos: usize,
            take: usize,
            in_off: usize,
            off: u64,
            len: u32,
        }
        let mut pending: Vec<Pending> = Vec::new();

        let mut done = 0usize;
        let mut pos = offset;
        while done < n {
            let lc = (pos / CHUNK_SIZE) as usize;
            let in_off = pos % CHUNK_SIZE;
            let take = ((CHUNK_SIZE - in_off) as usize).min(n - done);
            match self.coord(path, lc) {
                None => {
                    // Sparse hole. The caller buffer isn't pre-zeroed (unlike
                    // the old `vec![0u8; n]`), so zero this span explicitly.
                    out[done..done + take].fill(0);
                }
                Some(Placement::Raw { chunk }) => {
                    let phys = chunk as u64 * CHUNK_SIZE + in_off;
                    let mut run_take = take;
                    let mut prev_chunk = chunk;
                    while done + run_take < n {
                        let next_pos = pos + run_take as u64;
                        if next_pos % CHUNK_SIZE != 0 {
                            break;
                        }
                        let next_lc = (next_pos / CHUNK_SIZE) as usize;
                        let next_take = (CHUNK_SIZE as usize).min(n - done - run_take);
                        match self.coord(path, next_lc) {
                            Some(Placement::Raw { chunk }) if chunk == prev_chunk + 1 => {
                                run_take += next_take;
                                prev_chunk = chunk;
                            }
                            _ => break,
                        }
                    }
                    self.trace.raw_read_ops += 1;
                    self.trace.raw_read_bytes += run_take as u64;
                    cuda(self.vram.read_at(phys, &mut out[done..done + run_take]))?;
                    done += run_take;
                    pos += run_take as u64;
                    continue;
                }
                Some(Placement::Compressed {
                    offset: off,
                    len: clen,
                    codec: Codec::Lz4,
                }) => {
                    // Record the arena location; decompress in bulk straight from VRAM.
                    pending.push(Pending {
                        out_pos: done,
                        take,
                        in_off: in_off as usize,
                        off,
                        len: clen,
                    });
                    self.trace.compressed_read_chunks += 1;
                    self.trace.compressed_read_requested_bytes += take as u64;
                    self.trace.compressed_read_full_bytes += CHUNK_SIZE;
                }
                Some(Placement::Compressed {
                    offset: off,
                    len: clen,
                    codec: Codec::Zstd,
                }) => {
                    let full = self.decompress_blob(off, clen, Codec::Zstd)?;
                    let s = in_off as usize;
                    out[done..done + take].copy_from_slice(&full[s..s + take]);
                }
            }
            done += take;
            pos += take as u64;
        }

        // Bulk-decompress the gathered LZ4 blobs (read device-to-device from the
        // arena) and scatter into the output.
        if !pending.is_empty() {
            let requests: Vec<(u64, u32, usize, usize)> = pending
                .iter()
                .map(|p| (p.off, p.len, p.in_off, p.take))
                .collect();
            let base = self.vram_base;
            let codec = self
                .codec
                .as_mut()
                .expect("nvCOMP codec present for Lz4 placements");
            let pieces = cuda(codec.decompress_from_arena_slices(base, &requests))?;
            for (p, piece) in pending.iter().zip(pieces.iter()) {
                out[p.out_pos..p.out_pos + p.take].copy_from_slice(piece);
            }
        }
        Ok(n)
    }

    /// Write `data` to `path` at `offset`, growing the file as needed.
    pub fn write(&mut self, path: &str, offset: u64, data: &[u8]) -> EResult<u64> {
        if data.is_empty() {
            return Ok(0);
        }
        {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            if node.is_dir {
                return Err(EngineError::NotAFile);
            }
        }
        // Guard against arithmetic overflow from a pathological offset.
        let end = offset
            .checked_add(data.len() as u64)
            .ok_or(EngineError::NoSpace)?;
        self.ensure_logical_len(path, end)?;
        self.trace.write_calls += 1;
        self.trace.logical_write_bytes += data.len() as u64;

        // Compress-only mode with nvCOMP available: compress the run of full
        // chunks in this write as one batched GPU call instead of one launch per
        // chunk. Leading/trailing partial chunks fall back to per-chunk RMW.
        if self.dedup && !self.compress {
            self.write_dedup_uncompressed(path, offset, data)?;
        } else if !self.compress && !self.dedup {
            self.write_raw_unshared(path, offset, data)?;
        } else if self.compress && !self.dedup && self.codec.is_some() {
            self.write_compressed(path, offset, data)?;
        } else if self.compress && self.dedup && self.codec.is_some() {
            self.write_compress_dedup(path, offset, data)?;
        } else {
            let mut done = 0usize;
            let mut pos = offset;
            while done < data.len() {
                let lc = (pos / CHUNK_SIZE) as usize;
                let in_off = pos % CHUNK_SIZE;
                let take = ((CHUNK_SIZE - in_off) as usize).min(data.len() - done);
                self.write_chunk(path, lc, in_off, &data[done..done + take])?;
                done += take;
                pos += take as u64;
            }
        }

        let node = self.table.get_mut(path).unwrap();
        node.size = node.size.max(end);
        let now = crate::lookup::now_filetime();
        node.modified = now;
        node.changed = now;
        Ok(data.len() as u64)
    }

    pub fn clone_range(
        &mut self,
        src_path: &str,
        dst_path: &str,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    ) -> EResult<u64> {
        if len == 0 {
            return Ok(0);
        }
        let (src_size, src_is_dir, dst_is_dir) = {
            let src = self.table.get(src_path).ok_or(LookupError::NotFound)?;
            let dst = self.table.get(dst_path).ok_or(LookupError::NotFound)?;
            (src.size, src.is_dir, dst.is_dir)
        };
        if src_is_dir || dst_is_dir {
            return Err(EngineError::NotAFile);
        }
        if src_offset >= src_size {
            return Ok(0);
        }
        let n = len.min(src_size - src_offset);
        if !self.dedup {
            let data = self.read(src_path, src_offset, n as usize)?;
            return self.write(dst_path, dst_offset, &data);
        }
        if src_path.eq_ignore_ascii_case(dst_path)
            && ranges_overlap(src_offset, src_offset + n, dst_offset, dst_offset + n)
        {
            let data = self.read(src_path, src_offset, n as usize)?;
            return self.write(dst_path, dst_offset, &data);
        }

        self.ensure_logical_len(dst_path, dst_offset + n)?;
        let mut done = 0u64;
        while done < n
            && ((src_offset + done) % CHUNK_SIZE != 0 || (dst_offset + done) % CHUNK_SIZE != 0)
        {
            let src_next = CHUNK_SIZE - ((src_offset + done) % CHUNK_SIZE);
            let dst_next = CHUNK_SIZE - ((dst_offset + done) % CHUNK_SIZE);
            let take = (n - done).min(src_next.min(dst_next));
            let data = self.read(src_path, src_offset + done, take as usize)?;
            self.write(dst_path, dst_offset + done, &data)?;
            done += take;
        }

        let full_chunks = ((n - done) / CHUNK_SIZE) as usize;
        if full_chunks > 0 {
            let src_lc0 = ((src_offset + done) / CHUNK_SIZE) as usize;
            let dst_lc0 = ((dst_offset + done) / CHUNK_SIZE) as usize;
            let placements: Vec<Option<Placement>> = (0..full_chunks)
                .map(|i| self.coord(src_path, src_lc0 + i))
                .collect();
            for p in placements.iter().flatten() {
                self.ref_inc_placement(*p);
            }
            for (i, p) in placements.into_iter().enumerate() {
                let dst_lc = dst_lc0 + i;
                if let Some(old) = self.coord(dst_path, dst_lc) {
                    self.free_placement(old);
                }
                self.set_coord(dst_path, dst_lc, p);
            }
            done += full_chunks as u64 * CHUNK_SIZE;
            self.trace.dedup_shared_chunks += full_chunks as u64;
        }

        if done < n {
            let take = n - done;
            let data = self.read(src_path, src_offset + done, take as usize)?;
            self.write(dst_path, dst_offset + done, &data)?;
        }

        let node = self.table.get_mut(dst_path).unwrap();
        node.size = node.size.max(dst_offset + n);
        let now = crate::lookup::now_filetime();
        node.modified = now;
        node.changed = now;
        Ok(n)
    }

    /// Dedup-only write path. Batches the common full-chunk case so a large
    /// copy does one GPU verification launch and one transfer sync per request
    /// instead of one of each per 64 KiB chunk.
    fn write_dedup_uncompressed(&mut self, path: &str, offset: u64, data: &[u8]) -> EResult<()> {
        let cs = CHUNK_SIZE;
        let end = offset + data.len() as u64;
        let full_start = offset.div_ceil(cs) * cs;
        let full_end = (end / cs) * cs;

        if full_end <= full_start {
            let mut done = 0usize;
            let mut pos = offset;
            while done < data.len() {
                let lc = (pos / cs) as usize;
                let in_off = pos % cs;
                let take = ((cs - in_off) as usize).min(data.len() - done);
                self.write_chunk(path, lc, in_off, &data[done..done + take])?;
                done += take;
                pos += take as u64;
            }
            return Ok(());
        }

        if full_start > offset {
            let lc = (offset / cs) as usize;
            let in_off = offset % cs;
            let take = (full_start - offset) as usize;
            self.write_chunk(path, lc, in_off, &data[..take])?;
        }

        let mid_off = (full_start - offset) as usize;
        let lc0 = (full_start / cs) as usize;
        let n = ((full_end - full_start) / cs) as usize;
        self.write_dedup_full_chunks(path, lc0, &data[mid_off..mid_off + n * cs as usize])?;

        if end > full_end {
            let lc = (full_end / cs) as usize;
            let take = (end - full_end) as usize;
            let start = (full_end - offset) as usize;
            self.write_chunk(path, lc, 0, &data[start..start + take])?;
        }
        Ok(())
    }

    fn write_dedup_full_chunks(&mut self, path: &str, lc0: usize, data: &[u8]) -> EResult<()> {
        debug_assert_eq!(data.len() as u64 % CHUNK_SIZE, 0);
        let n = data.len() / CHUNK_SIZE as usize;
        if self.hash_index.is_empty() && (0..n).all(|i| self.coord(path, lc0 + i).is_none()) {
            return self.write_dedup_full_chunks_gpu_staged(path, lc0, data);
        }
        let hashes = fnv1a_chunks(data);
        self.trace.dedup_hash_chunks += n as u64;

        let candidates: Vec<Option<Placement>> = hashes
            .iter()
            .map(|h| self.hash_index.get(h).copied())
            .collect();
        self.trace.dedup_candidate_chunks +=
            candidates.iter().filter(|c| c.is_some()).count() as u64;

        let mut raw_verify_offsets = Vec::new();
        let mut raw_verify_items = Vec::new();
        for (i, cand) in candidates.iter().enumerate() {
            if let Some(Placement::Raw { chunk }) = cand {
                raw_verify_offsets.push(*chunk as u64 * CHUNK_SIZE);
                raw_verify_items.push(i);
            }
        }

        let mut raw_verified = vec![false; n];
        if !raw_verify_offsets.is_empty() {
            let mut out = vec![0u64; raw_verify_offsets.len()];
            let base = self.vram_base;
            let hasher = self
                .gpu_hasher
                .as_mut()
                .expect("gpu_hasher present when dedup");
            cuda(hasher.hash_chunks(base, &raw_verify_offsets, &mut out))?;
            for (slot, &i) in raw_verify_items.iter().enumerate() {
                raw_verified[i] = out[slot] == hashes[i];
            }
        }

        let mut indexed_writes = Vec::new();
        let mut wrote = false;
        for i in 0..n {
            let lc = lc0 + i;
            let old = self.coord(path, lc);
            let candidate_ok = match candidates[i] {
                Some(Placement::Raw { .. }) => raw_verified[i],
                Some(Placement::Compressed { offset, .. }) => {
                    self.compressed_hash.get(&offset) == Some(&hashes[i])
                }
                None => false,
            };
            if candidate_ok {
                let p = candidates[i].unwrap();
                self.ref_inc_placement(p);
                if let Some(old) = old {
                    self.free_placement(old);
                }
                self.set_coord(path, lc, Some(p));
                self.trace.dedup_shared_chunks += 1;
                continue;
            }

            let chunk = match old {
                Some(Placement::Raw { chunk }) if self.refcount[chunk as usize] == 1 => {
                    self.index_remove(chunk);
                    chunk
                }
                _ => self.alloc_chunk()?,
            };
            let s = i * CHUNK_SIZE as usize;
            cuda(
                self.vram
                    .write_at_async(chunk as u64 * CHUNK_SIZE, &data[s..s + CHUNK_SIZE as usize]),
            )?;
            self.trace.raw_write_ops += 1;
            self.trace.raw_write_bytes += CHUNK_SIZE;
            wrote = true;
            if !matches!(old, Some(Placement::Raw { chunk: c }) if c == chunk) {
                if let Some(old) = old {
                    self.free_placement(old);
                }
            }
            self.set_coord(path, lc, Some(Placement::Raw { chunk }));
            indexed_writes.push((chunk, hashes[i]));
            self.trace.dedup_unique_chunks += 1;
        }

        if wrote {
            cuda(self.vram.sync())?;
        }
        for (chunk, h) in indexed_writes {
            self.index_insert(chunk, h);
        }
        Ok(())
    }

    fn write_dedup_full_chunks_gpu_staged(
        &mut self,
        path: &str,
        lc0: usize,
        data: &[u8],
    ) -> EResult<()> {
        let cs = CHUNK_SIZE as usize;
        let n = data.len() / cs;
        let mut chunks = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n);
        let mut j = 0usize;
        while j < n {
            let remaining = (n - j) as u32;
            let (start, got) = self.alloc_contiguous_run(remaining)?;
            let got = got as usize;
            let bytes = got * cs;
            let data_off = j * cs;
            cuda(
                self.vram
                    .write_at(start as u64 * CHUNK_SIZE, &data[data_off..data_off + bytes]),
            )?;
            self.trace.raw_write_ops += 1;
            self.trace.raw_write_bytes += bytes as u64;
            for k in 0..got {
                let chunk = start + k as u32;
                self.refcount[chunk as usize] = 1;
                self.set_coord(path, lc0 + j + k, Some(Placement::Raw { chunk }));
                chunks.push(chunk);
                offsets.push(chunk as u64 * CHUNK_SIZE);
            }
            j += got;
        }

        let mut hashes = vec![0u64; offsets.len()];
        let base = self.vram_base;
        let hasher = self
            .gpu_hasher
            .as_mut()
            .expect("gpu_hasher present when dedup");
        cuda(hasher.hash_chunks(base, &offsets, &mut hashes))?;
        self.trace.gpu_hash_chunks += hashes.len() as u64;
        self.trace.dedup_unique_chunks += hashes.len() as u64;
        for (chunk, h) in chunks.into_iter().zip(hashes.into_iter()) {
            self.index_insert(chunk, h);
        }
        Ok(())
    }

    /// Raw non-dedup write path. Batches contiguous full-chunk regions into
    /// large H→D transfers instead of synchronising once per 64 KiB chunk.
    fn write_raw_unshared(&mut self, path: &str, offset: u64, data: &[u8]) -> EResult<()> {
        let cs = CHUNK_SIZE;
        let end = offset + data.len() as u64;
        let full_start = offset.div_ceil(cs) * cs;
        let full_end = (end / cs) * cs;

        if full_end <= full_start {
            let mut done = 0usize;
            let mut pos = offset;
            while done < data.len() {
                let lc = (pos / cs) as usize;
                let in_off = pos % cs;
                let take = ((cs - in_off) as usize).min(data.len() - done);
                self.write_chunk(path, lc, in_off, &data[done..done + take])?;
                done += take;
                pos += take as u64;
            }
            return Ok(());
        }

        if full_start > offset {
            let lc = (offset / cs) as usize;
            let in_off = offset % cs;
            let take = (full_start - offset) as usize;
            self.write_chunk(path, lc, in_off, &data[..take])?;
        }

        let mid_off = (full_start - offset) as usize;
        let lc0 = (full_start / cs) as usize;
        let n = ((full_end - full_start) / cs) as usize;
        let mut j = 0usize;
        while j < n {
            let lc = lc0 + j;
            let data_off = mid_off + j * cs as usize;
            match self.coord(path, lc) {
                None => {
                    let mut count = 1usize;
                    while j + count < n && self.coord(path, lc + count).is_none() {
                        count += 1;
                    }
                    let (start, got) = self.alloc_contiguous_run(count as u32)?;
                    let count = got as usize;
                    let bytes = count * cs as usize;
                    cuda(
                        self.vram
                            .write_at(start as u64 * cs, &data[data_off..data_off + bytes]),
                    )?;
                    self.trace.raw_write_ops += 1;
                    self.trace.raw_write_bytes += bytes as u64;
                    for k in 0..count {
                        self.set_coord(
                            path,
                            lc + k,
                            Some(Placement::Raw {
                                chunk: start + k as u32,
                            }),
                        );
                    }
                    j += count;
                }
                Some(Placement::Raw { chunk }) => {
                    let mut count = 1usize;
                    let mut prev = chunk;
                    while j + count < n {
                        match self.coord(path, lc + count) {
                            Some(Placement::Raw { chunk }) if chunk == prev + 1 => {
                                count += 1;
                                prev = chunk;
                            }
                            _ => break,
                        }
                    }
                    let bytes = count * cs as usize;
                    cuda(
                        self.vram
                            .write_at(chunk as u64 * cs, &data[data_off..data_off + bytes]),
                    )?;
                    self.trace.raw_write_ops += 1;
                    self.trace.raw_write_bytes += bytes as u64;
                    j += count;
                }
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Cuda(
                        "compressed placement in raw writer".into(),
                    ));
                }
            }
        }

        if end > full_end {
            let lc = (full_end / cs) as usize;
            let take = (end - full_end) as usize;
            let start = (full_end - offset) as usize;
            self.write_chunk(path, lc, 0, &data[start..start + take])?;
        }
        Ok(())
    }

    fn alloc_contiguous_run(&mut self, count: u32) -> EResult<(ChunkId, u32)> {
        for n in (1..=count).rev() {
            if let Some(start) = self.alloc.alloc_contiguous(n) {
                return Ok((start, n));
            }
        }
        Err(EngineError::NoSpace)
    }

    /// Compress-mode write (no dedup, nvCOMP present). Splits the write into an
    /// optional leading partial chunk, a run of whole chunks compressed in one
    /// batched nvCOMP call, and an optional trailing partial chunk.
    fn write_compressed(&mut self, path: &str, offset: u64, data: &[u8]) -> EResult<()> {
        let cs = CHUNK_SIZE;
        let end = offset + data.len() as u64;
        // Byte range covered by whole, chunk-aligned chunks fully inside the write.
        let full_start = offset.div_ceil(cs) * cs;
        let full_end = (end / cs) * cs;

        if full_end <= full_start {
            // No whole chunk: the entire write lands in partial chunk(s).
            let mut done = 0usize;
            let mut pos = offset;
            while done < data.len() {
                let lc = (pos / cs) as usize;
                let in_off = pos % cs;
                let take = ((cs - in_off) as usize).min(data.len() - done);
                self.write_chunk(path, lc, in_off, &data[done..done + take])?;
                done += take;
                pos += take as u64;
            }
            return Ok(());
        }

        // Leading partial chunk (offset not chunk-aligned).
        if full_start > offset {
            let lc = (offset / cs) as usize;
            let in_off = offset % cs;
            let take = (full_start - offset) as usize;
            self.write_chunk(path, lc, in_off, &data[..take])?;
        }

        // Whole-chunk run: compress group-by-group, moving each compressed blob
        // straight from the codec scratch into the packed VRAM arena with a
        // device-to-device copy (no host bounce). Incompressible chunks fall
        // back to a raw host write via place_chunk.
        let mid_off = (full_start - offset) as usize;
        let mid_len = (full_end - full_start) as usize;
        let lc0 = (full_start / cs) as usize;
        let n = mid_len / cs as usize;
        let batch = Lz4Codec::max_batch();

        let mut j = 0usize;
        while j < n {
            let m = (n - j).min(batch);
            let g0 = mid_off + j * cs as usize;
            let group = &data[g0..g0 + m * cs as usize];

            if group_looks_incompressible(group, m) {
                for k in 0..m {
                    let lc = lc0 + j + k;
                    let full = &group[k * cs as usize..(k + 1) * cs as usize];
                    let old = self.coord(path, lc);
                    self.write_raw_compress_fallback(path, lc, full, old, None)?;
                    self.trace.compress_raw_fallback_chunks += 1;
                }
                j += m;
                continue;
            }

            // Compress the group; blobs stay resident in the codec scratch.
            let (sizes, slots) = {
                let codec = self.codec.as_mut().expect("nvCOMP codec present");
                let sizes = cuda(codec.compress_group_dev(group, m))?;
                let slots: Vec<u64> = (0..m).map(|k| codec.comp_slot_ptr(k)).collect();
                (sizes, slots)
            };
            self.trace.compress_batches += 1;
            self.trace.compress_chunks += m as u64;

            for k in 0..m {
                let lc = lc0 + j + k;
                let full = &group[k * cs as usize..(k + 1) * cs as usize];
                let old = self.coord(path, lc);
                match sizes[k] {
                    Some(len) => {
                        let off = self.carena_alloc(len as u32)?;
                        cuda(self.vram.copy_dev_into(off, slots[k], len as u64))?;
                        if let Some(p) = old {
                            self.free_placement(p);
                        }
                        self.set_coord(
                            path,
                            lc,
                            Some(Placement::Compressed {
                                offset: off,
                                len: len as u32,
                                codec: Codec::Lz4,
                            }),
                        );
                    }
                    None => {
                        self.write_raw_compress_fallback(path, lc, full, old, None)?;
                        self.trace.compress_raw_fallback_chunks += 1;
                    }
                }
            }
            j += m;
        }
        // Flush the enqueued device-to-device arena copies and raw fallbacks.
        cuda(self.vram.sync())?;

        // Trailing partial chunk.
        if end > full_end {
            let lc = (full_end / cs) as usize;
            let take = (end - full_end) as usize;
            let start = (full_end - offset) as usize;
            self.write_chunk(path, lc, 0, &data[start..start + take])?;
        }
        Ok(())
    }

    fn write_compress_dedup(&mut self, path: &str, offset: u64, data: &[u8]) -> EResult<()> {
        let cs = CHUNK_SIZE;
        let end = offset + data.len() as u64;
        let full_start = offset.div_ceil(cs) * cs;
        let full_end = (end / cs) * cs;

        if full_end <= full_start {
            let mut done = 0usize;
            let mut pos = offset;
            while done < data.len() {
                let lc = (pos / cs) as usize;
                let in_off = pos % cs;
                let take = ((cs - in_off) as usize).min(data.len() - done);
                self.write_chunk(path, lc, in_off, &data[done..done + take])?;
                done += take;
                pos += take as u64;
            }
            return Ok(());
        }

        if full_start > offset {
            let lc = (offset / cs) as usize;
            let in_off = offset % cs;
            let take = (full_start - offset) as usize;
            self.write_chunk(path, lc, in_off, &data[..take])?;
        }

        let mid_off = (full_start - offset) as usize;
        let mid_len = (full_end - full_start) as usize;
        let lc0 = (full_start / cs) as usize;
        let n = mid_len / cs as usize;
        let batch = Lz4Codec::max_batch();

        let mut j = 0usize;
        while j < n {
            let m = (n - j).min(batch);
            let g0 = mid_off + j * cs as usize;
            let group = &data[g0..g0 + m * cs as usize];
            let hashes = fnv1a_chunks(group);
            self.trace.dedup_hash_chunks += m as u64;
            let candidates: Vec<Option<Placement>> = hashes
                .iter()
                .map(|h| self.hash_index.get(h).copied())
                .collect();
            self.trace.dedup_candidate_chunks +=
                candidates.iter().filter(|c| c.is_some()).count() as u64;

            let mut raw_verify_offsets = Vec::new();
            let mut raw_verify_items = Vec::new();
            for (i, cand) in candidates.iter().enumerate() {
                if let Some(Placement::Raw { chunk }) = cand {
                    raw_verify_offsets.push(*chunk as u64 * CHUNK_SIZE);
                    raw_verify_items.push(i);
                }
            }
            let mut raw_verified = vec![false; m];
            if !raw_verify_offsets.is_empty() {
                let mut out = vec![0u64; raw_verify_offsets.len()];
                let base = self.vram_base;
                let hasher = self
                    .gpu_hasher
                    .as_mut()
                    .expect("gpu_hasher present when dedup");
                cuda(hasher.hash_chunks(base, &raw_verify_offsets, &mut out))?;
                self.trace.gpu_hash_chunks += out.len() as u64;
                for (slot, &i) in raw_verify_items.iter().enumerate() {
                    raw_verified[i] = out[slot] == hashes[i];
                }
            }

            let mut misses = Vec::new();
            let mut miss_buf = Vec::new();
            for i in 0..m {
                let lc = lc0 + j + i;
                let candidate_ok = match candidates[i] {
                    Some(Placement::Raw { .. }) => raw_verified[i],
                    Some(Placement::Compressed { offset, .. }) => {
                        self.compressed_hash.get(&offset) == Some(&hashes[i])
                    }
                    None => false,
                };
                if candidate_ok {
                    let p = candidates[i].unwrap();
                    self.ref_inc_placement(p);
                    if let Some(old) = self.coord(path, lc) {
                        self.free_placement(old);
                    }
                    self.set_coord(path, lc, Some(p));
                    self.trace.dedup_shared_chunks += 1;
                } else {
                    let s = i * cs as usize;
                    misses.push((i, lc, self.coord(path, lc), hashes[i]));
                    miss_buf.extend_from_slice(&group[s..s + cs as usize]);
                }
            }

            if misses.is_empty() {
                j += m;
                continue;
            }

            if group_looks_incompressible(&miss_buf, misses.len()) {
                for (slot, &(_i, lc, old, h)) in misses.iter().enumerate() {
                    let s = slot * cs as usize;
                    self.write_raw_compress_fallback(
                        path,
                        lc,
                        &miss_buf[s..s + cs as usize],
                        old,
                        Some(h),
                    )?;
                    self.trace.compress_raw_fallback_chunks += 1;
                    self.trace.dedup_unique_chunks += 1;
                }
                j += m;
                continue;
            }

            let miss_count = misses.len();
            let (sizes, slots) = {
                let codec = self.codec.as_mut().expect("nvCOMP codec present");
                let sizes = cuda(codec.compress_group_dev(&miss_buf, miss_count))?;
                let slots: Vec<u64> = (0..miss_count).map(|k| codec.comp_slot_ptr(k)).collect();
                (sizes, slots)
            };
            self.trace.compress_batches += 1;
            self.trace.compress_chunks += miss_count as u64;

            for (slot, &(_i, lc, old, h)) in misses.iter().enumerate() {
                let s = slot * cs as usize;
                let full = &miss_buf[s..s + cs as usize];
                match sizes[slot] {
                    Some(len) => {
                        let off = self.carena_alloc(len as u32)?;
                        cuda(self.vram.copy_dev_into(off, slots[slot], len as u64))?;
                        if let Some(p) = old {
                            self.free_placement(p);
                        }
                        let p = Placement::Compressed {
                            offset: off,
                            len: len as u32,
                            codec: Codec::Lz4,
                        };
                        self.set_coord(path, lc, Some(p));
                        self.compressed_index_insert(p, h);
                    }
                    None => {
                        self.write_raw_compress_fallback(path, lc, full, old, Some(h))?;
                        self.trace.compress_raw_fallback_chunks += 1;
                    }
                }
                self.trace.dedup_unique_chunks += 1;
            }
            j += m;
        }

        cuda(self.vram.sync())?;

        if end > full_end {
            let lc = (full_end / cs) as usize;
            let take = (end - full_end) as usize;
            let start = (full_end - offset) as usize;
            self.write_chunk(path, lc, 0, &data[start..start + take])?;
        }
        Ok(())
    }

    fn write_raw_compress_fallback(
        &mut self,
        path: &str,
        lc: usize,
        full: &[u8],
        old: Option<Placement>,
        content_hash: Option<u64>,
    ) -> EResult<()> {
        let chunk = match old {
            Some(Placement::Raw { chunk }) if !self.dedup || self.refcount[chunk as usize] == 1 => {
                if self.dedup {
                    self.index_remove(chunk);
                }
                chunk
            }
            _ if self.dedup => self.alloc_chunk()?,
            _ => self.alloc.alloc_one().ok_or(EngineError::NoSpace)?,
        };
        cuda(self.vram.write_at_async(chunk as u64 * CHUNK_SIZE, full))?;
        self.trace.raw_write_ops += 1;
        self.trace.raw_write_bytes += CHUNK_SIZE;
        if !matches!(old, Some(Placement::Raw { chunk: c }) if c == chunk) {
            if let Some(p) = old {
                self.free_placement(p);
            }
        }
        self.set_coord(path, lc, Some(Placement::Raw { chunk }));
        if self.dedup {
            if let Some(h) = content_hash {
                self.index_insert(chunk, h);
            }
        }
        Ok(())
    }

    /// Write `sub` into logical chunk `lc` at byte `in_off` within the chunk.
    fn write_chunk(&mut self, path: &str, lc: usize, in_off: u64, sub: &[u8]) -> EResult<()> {
        let full = in_off == 0 && sub.len() as u64 == CHUNK_SIZE;

        // Dedup path for full-chunk writes: try to share an identical chunk.
        if self.dedup && full {
            let h = fnv1a(sub);
            if self.try_share_hashed(path, lc, h)? {
                return Ok(());
            }
            // No dedup match.
            if self.compress {
                // Unique content in compress+dedup mode: store compressed (or raw
                // if incompressible) and enter the resulting placement into the
                // dedup index using the uncompressed content hash.
                return self.store_compressed(path, lc, sub, Some(h));
            }
            // Dedup-only: place in an exclusively-owned Raw chunk.
            let old = self.coord(path, lc);
            match old {
                Some(Placement::Raw { chunk }) if self.refcount[chunk as usize] == 1 => {
                    self.index_remove(chunk);
                    cuda(self.vram.write_at(chunk as u64 * CHUNK_SIZE, sub))?;
                    self.index_insert(chunk, h);
                }
                _ => {
                    let c = self.alloc_chunk()?;
                    cuda(self.vram.write_at(c as u64 * CHUNK_SIZE, sub))?;
                    self.index_insert(c, h);
                    if let Some(p) = old {
                        self.free_placement(p);
                    }
                    self.set_coord(path, lc, Some(Placement::Raw { chunk: c }));
                }
            }
            return Ok(());
        }

        // Compression path (mutually exclusive with dedup).
        if self.compress {
            if full {
                return self.store_compressed(path, lc, sub, None);
            }
            // Partial write: read-modify-write the whole logical chunk.
            let mut whole = self.read_logical_chunk(path, lc)?;
            let s = in_off as usize;
            whole[s..s + sub.len()].copy_from_slice(sub);
            if self.dedup {
                let h = fnv1a(&whole);
                if self.try_share_hashed(path, lc, h)? {
                    return Ok(());
                }
                return self.store_compressed(path, lc, &whole, Some(h));
            }
            return self.store_compressed(path, lc, &whole, None);
        }

        // Non-dedup fast path.
        if !self.dedup {
            let existing = self.coord(path, lc);
            let (chunk, fresh) = match existing {
                Some(Placement::Raw { chunk }) => (chunk, false),
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Cuda("compressed write not implemented".into()));
                }
                None => (self.alloc_chunk()?, true),
            };
            let base = chunk as u64 * CHUNK_SIZE;
            if fresh && !full {
                cuda(self.vram.zero_at_async(base, CHUNK_SIZE))?;
                cuda(self.vram.write_at_async(base + in_off, sub))?;
                cuda(self.vram.sync())?;
            } else {
                cuda(self.vram.write_at(base + in_off, sub))?;
            }
            self.trace.raw_write_ops += 1;
            self.trace.raw_write_bytes += sub.len() as u64;
            if fresh {
                self.set_coord(path, lc, Some(Placement::Raw { chunk }));
            }
            return Ok(());
        }

        // Dedup + partial write: copy-on-write, then modify in place. The
        // chunk is left out of the dedup index until it is next fully written.
        let chunk = self.make_exclusive(path, lc)?;
        cuda(self.vram.write_at(chunk as u64 * CHUNK_SIZE + in_off, sub))?;
        self.trace.raw_write_ops += 1;
        self.trace.raw_write_bytes += sub.len() as u64;
        Ok(())
    }

    /// Largest number of logical chunks any single file may address. Bounds the
    /// per-file coordinate vector so a write or truncate at an enormous offset
    /// can't exhaust host memory — it is also the most chunks the whole volume
    /// could ever physically hold, so a larger logical size is never useful.
    fn max_logical_chunks(&self) -> usize {
        self.alloc.total() as usize
    }

    /// Grow a file's coordinate array so it can address `byte_len` bytes,
    /// padding with sparse holes. Does not shrink or change `size`.
    fn ensure_logical_len(&mut self, path: &str, byte_len: u64) -> EResult<()> {
        let need = logical_chunks(byte_len);
        if need > self.max_logical_chunks() {
            return Err(EngineError::NoSpace);
        }
        let node = self.table.get_mut(path).ok_or(LookupError::NotFound)?;
        if node.coords.len() < need {
            node.coords.resize(need, None);
        }
        Ok(())
    }

    /// Set a file's logical size (WinFsp SetFileSize / truncate).
    pub fn set_size(&mut self, path: &str, new_size: u64) -> EResult<()> {
        let (old_size, old_chunks) = {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            if node.is_dir {
                return Err(EngineError::NotAFile);
            }
            (node.size, node.coords.len())
        };

        let new_chunks = logical_chunks(new_size);
        // Reject sizes that would require an unbounded coordinate vector.
        if new_chunks > self.max_logical_chunks() {
            return Err(EngineError::NoSpace);
        }

        if new_size < old_size {
            let freed: Vec<Option<Placement>> = {
                let node = self.table.get_mut(path).unwrap();
                node.coords
                    .drain(new_chunks..old_chunks.max(new_chunks))
                    .collect()
            };
            self.free_coords(&freed);

            // Zero the tail of the last surviving chunk so a later regrow reads
            // zeros.
            let tail = new_size % CHUNK_SIZE;
            if tail != 0 {
                let last = new_chunks - 1;
                match self.coord(path, last) {
                    Some(Placement::Compressed { .. }) => {
                        // Read-modify-write under compression.
                        let mut whole = self.read_logical_chunk(path, last)?;
                        for b in &mut whole[tail as usize..] {
                            *b = 0;
                        }
                        let h = self.dedup.then(|| fnv1a(&whole));
                        let shared = if let Some(h) = h {
                            self.try_share_hashed(path, last, h)?
                        } else {
                            false
                        };
                        if !shared {
                            self.store_compressed(path, last, &whole, h)?;
                        }
                    }
                    Some(Placement::Raw { .. }) => {
                        let c = self.make_exclusive(path, last)?;
                        cuda(
                            self.vram
                                .zero_at(c as u64 * CHUNK_SIZE + tail, CHUNK_SIZE - tail),
                        )?;
                    }
                    None => {}
                }
            }
        } else if new_chunks > old_chunks {
            let node = self.table.get_mut(path).unwrap();
            node.coords.resize(new_chunks, None);
        }

        let node = self.table.get_mut(path).unwrap();
        node.size = new_size;
        let now = crate::lookup::now_filetime();
        node.modified = now;
        node.changed = now;
        Ok(())
    }

    pub fn archive_compress_gpu(
        &mut self,
        format: NvcompFrameCodec,
        files: &[String],
        output: &str,
    ) -> EResult<ArchiveJobStats> {
        let start = Instant::now();
        let output = crate::lookup::normalize(output);
        if format == NvcompFrameCodec::Deflate {
            let stats = self.archive_zip_compress_gpu(files, &output, start)?;
            return Ok(stats);
        }
        let tmp = format!("\\.__vramdisk_archive_tmp_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&tmp)?;
        let mut planned = Vec::with_capacity(files.len());
        let mut tar_total = 1024u64;
        for file in files {
            let path = crate::lookup::normalize(file);
            let (size, is_dir) = {
                let node = self.table.get(&path).ok_or(LookupError::NotFound)?;
                (node.size, node.is_dir)
            };
            if is_dir {
                return Err(EngineError::NotAFile);
            }
            tar_total += 512 + size + pad512(size);
            planned.push((path, size));
        }
        self.allocate_raw_file(&tmp, tar_total)?;
        let mut tar_pos = 0u64;
        let mut input_bytes = 0u64;
        for (path, size) in planned {
            let name = path.trim_start_matches('\\').replace('\\', "/");
            let header = tar_header(&name, size)?;
            self.write(&tmp, tar_pos, &header)?;
            tar_pos += 512;
            self.copy_file_payload_raw(&path, 0, &tmp, tar_pos, size)?;
            tar_pos += size;
            input_bytes += size;
            let pad = pad512(size);
            if pad != 0 {
                self.write(&tmp, tar_pos, &vec![0u8; pad as usize])?;
                tar_pos += pad;
            }
        }
        self.write(&tmp, tar_pos, &[0u8; 1024])?;
        tar_pos += 1024;

        self.create_or_truncate_file(&output)?;
        let comp_len = if format == NvcompFrameCodec::Gzip {
            self.write_gzip_deflate_members(&tmp, tar_pos, &output)?
        } else if format == NvcompFrameCodec::Lz4 {
            self.write_lz4_frame(&tmp, tar_pos, &output)?
        } else {
            let mut codec = cuda(NvcompBatchedCodec::load(&self.vram, format))?;
            let src_ptr = self.contiguous_file_ptr(&tmp, tar_pos)?;
            let sizes = cuda(codec.compress_device(&[src_ptr], &[tar_pos]))?;
            let comp_len = sizes[0];
            self.write_device_bytes(&output, 0, codec.compressed_slot_ptr(0), comp_len)?;
            comp_len
        };
        self.set_size(&output, comp_len)?;
        let _ = self.remove(&tmp);
        Ok(ArchiveJobStats {
            format: match format {
                NvcompFrameCodec::Zstd => "tar.zst".to_string(),
                NvcompFrameCodec::Lz4 => "tar.lz4".to_string(),
                NvcompFrameCodec::Gzip => "tar.gz".to_string(),
                NvcompFrameCodec::Deflate => "zip".to_string(),
            },
            output,
            file_count: files.len(),
            input_bytes,
            archive_bytes: comp_len,
            elapsed_ms: start.elapsed().as_millis(),
        })
    }

    pub fn archive_extract_gpu(
        &mut self,
        format: NvcompFrameCodec,
        archive: &str,
        output_dir: &str,
    ) -> EResult<ArchiveExtractStats> {
        let start = Instant::now();
        let archive = crate::lookup::normalize(archive);
        if format == NvcompFrameCodec::Deflate {
            return self.archive_zip_extract_gpu(&archive, output_dir, start);
        }
        let archive_size = {
            let node = self.table.get(&archive).ok_or(LookupError::NotFound)?;
            if node.is_dir {
                return Err(EngineError::NotAFile);
            }
            node.size
        };
        let tmp = format!("\\.__vramdisk_extract_tmp_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&tmp)?;
        let (tar_size, packed_archive) = if format == NvcompFrameCodec::Gzip {
            let tar_size = self.extract_gzip_deflate_members(&archive, archive_size, &tmp)?;
            (tar_size, None)
        } else if format == NvcompFrameCodec::Lz4 {
            let tar_size = self.extract_lz4_frame(&archive, archive_size, &tmp)?;
            (tar_size, None)
        } else {
            let packed_archive = format!("\\.__vramdisk_extract_src_{}", crate::lookup::now_filetime());
            self.create_or_truncate_file(&packed_archive)?;
            self.allocate_raw_file(&packed_archive, archive_size)?;
            self.copy_file_payload_raw(&archive, 0, &packed_archive, 0, archive_size)?;
            let archive_ptr = self.contiguous_file_ptr(&packed_archive, archive_size)?;
            let mut codec = cuda(NvcompBatchedCodec::load(&self.vram, format))?;
            let tar_size = cuda(codec.decompress_sizes_device(&[archive_ptr], &[archive_size]))?[0];
            self.allocate_raw_file(&tmp, tar_size)?;
            let tmp_ptr = self.contiguous_file_ptr(&tmp, tar_size)?;
            cuda(codec.decompress_device(
                &[archive_ptr],
                &[archive_size],
                &[tmp_ptr],
                &[tar_size],
            ))?;
            (tar_size, Some(packed_archive))
        };

        let out_base = crate::lookup::normalize(output_dir);
        self.ensure_dir_path(&out_base)?;
        let mut pos = 0u64;
        let mut files = 0usize;
        let mut output_bytes = 0u64;
        while pos + 512 <= tar_size {
            let hdr = self.read(&tmp, pos, 512)?;
            if hdr.iter().all(|&b| b == 0) {
                break;
            }
            let name_end = hdr[..100].iter().position(|&b| b == 0).unwrap_or(100);
            let name = std::str::from_utf8(&hdr[..name_end])
                .map_err(|e| EngineError::Cuda(format!("invalid tar path UTF-8: {e}")))?;
            let size = parse_tar_octal(&hdr[124..136])?;
            let out_path = join_archive_output(&out_base, name)?;
            self.ensure_parent_dirs(&out_path)?;
            self.create_or_truncate_file(&out_path)?;
            self.copy_file_payload_raw(&tmp, pos + 512, &out_path, 0, size)?;
            self.set_size(&out_path, size)?;
            files += 1;
            output_bytes += size;
            pos += 512 + size + pad512(size);
        }
        let _ = self.remove(&tmp);
        if let Some(packed_archive) = packed_archive {
            let _ = self.remove(&packed_archive);
        }
        Ok(ArchiveExtractStats {
            format: match format {
                NvcompFrameCodec::Zstd => "tar.zst".to_string(),
                NvcompFrameCodec::Lz4 => "tar.lz4".to_string(),
                NvcompFrameCodec::Gzip => "tar.gz".to_string(),
                NvcompFrameCodec::Deflate => "zip".to_string(),
            },
            archive,
            output_dir: out_base,
            file_count: files,
            archive_bytes: archive_size,
            output_bytes,
            elapsed_ms: start.elapsed().as_millis(),
        })
    }

    fn create_or_truncate_file(&mut self, path: &str) -> EResult<()> {
        let path = crate::lookup::normalize(path);
        match self.table.get(&path) {
            Some(node) if node.is_dir => return Err(EngineError::NotAFile),
            Some(_) => self.set_size(&path, 0)?,
            None => {
                self.table.create_file(&path, 0)?;
            }
        }
        Ok(())
    }

    fn copy_file_payload_raw(
        &mut self,
        src_path: &str,
        src_offset: u64,
        dst_path: &str,
        dst_offset: u64,
        len: u64,
    ) -> EResult<()> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_logical_len(dst_path, dst_offset + len)?;
        let mut done = 0u64;
        while done < len {
            let src_pos = src_offset + done;
            let src_lc = (src_pos / CHUNK_SIZE) as usize;
            let src_in = src_pos % CHUNK_SIZE;
            let take = (len - done).min(CHUNK_SIZE - src_in);
            let src_ptr = match self.coord(src_path, src_lc) {
                Some(Placement::Raw { chunk }) => {
                    self.vram_base + chunk as u64 * CHUNK_SIZE + src_in
                }
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Cuda(
                        "archive jobs currently require raw source file placements".into(),
                    ));
                }
                None => {
                    self.write(
                        &vec_path(dst_path),
                        dst_offset + done,
                        &vec![0u8; take as usize],
                    )?;
                    done += take;
                    continue;
                }
            };
            self.write_device_bytes(dst_path, dst_offset + done, src_ptr, take)?;
            done += take;
        }
        Ok(())
    }

    fn write_device_bytes(
        &mut self,
        path: &str,
        offset: u64,
        src_ptr: u64,
        len: u64,
    ) -> EResult<()> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_logical_len(path, offset + len)?;
        let mut done = 0u64;
        while done < len {
            let pos = offset + done;
            let lc = (pos / CHUNK_SIZE) as usize;
            let in_off = pos % CHUNK_SIZE;
            let take = (len - done).min(CHUNK_SIZE - in_off);
            let chunk = self.ensure_raw_output_chunk(path, lc)?;
            cuda(self.vram.copy_dev_into(
                chunk as u64 * CHUNK_SIZE + in_off,
                src_ptr + done,
                take,
            ))?;
            done += take;
        }
        cuda(self.vram.sync())?;
        let node = self.table.get_mut(path).ok_or(LookupError::NotFound)?;
        node.size = node.size.max(offset + len);
        let now = crate::lookup::now_filetime();
        node.modified = now;
        node.changed = now;
        Ok(())
    }

    fn allocate_raw_file(&mut self, path: &str, len: u64) -> EResult<()> {
        self.ensure_logical_len(path, len)?;
        let chunks = logical_chunks(len);
        if chunks == 0 {
            return self.set_size(path, 0);
        }
        let (start, got) = self.alloc_contiguous_run(chunks as u32)?;
        if got != chunks as u32 {
            return Err(EngineError::NoSpace);
        }
        for lc in 0..chunks {
            self.set_coord(
                path,
                lc,
                Some(Placement::Raw {
                    chunk: start + lc as u32,
                }),
            );
        }
        self.set_size(path, len)
    }

    fn archive_zip_compress_gpu(
        &mut self,
        files: &[String],
        output: &str,
        start: Instant,
    ) -> EResult<ArchiveJobStats> {
        let mut planned = Vec::with_capacity(files.len());
        for file in files {
            let path = crate::lookup::normalize(file);
            let (size, is_dir) = {
                let node = self.table.get(&path).ok_or(LookupError::NotFound)?;
                (node.size, node.is_dir)
            };
            if is_dir {
                return Err(EngineError::NotAFile);
            }
            let name = path.trim_start_matches('\\').replace('\\', "/");
            if !name.is_ascii() {
                return Err(EngineError::Cuda(format!("zip path must be ASCII: {name}")));
            }
            let crc = self.crc32_file_gpu(&path, size)?;
            planned.push((path, name, size, 0u64, crc));
        }
        self.create_or_truncate_file(output)?;
        let mut deflate = cuda(NvcompBatchedCodec::load(
            &self.vram,
            NvcompFrameCodec::Deflate,
        ))?;
        let mut out_pos = 0u64;
        let mut central = Vec::new();
        let mut input_bytes = 0u64;
        for (path, name, size, _comp_size, crc) in &planned {
            let local_offset = out_pos;
            let name_bytes = name.as_bytes();
            let zip_chunks = zip_deflate_chunk_count(*size);
            let extra = zip_local_extra(*size, &vec![0; zip_chunks]);
            let mut hdr = Vec::with_capacity(30 + name_bytes.len());
            push_u32(&mut hdr, 0x0403_4b50);
            push_u16(&mut hdr, 45);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 8);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 0);
            push_u32(&mut hdr, *crc);
            push_u32(&mut hdr, u32::MAX);
            push_u32(&mut hdr, u32::MAX);
            push_u16(
                &mut hdr,
                u16_checked(name_bytes.len(), "zip file name length")?,
            );
            push_u16(&mut hdr, u16_checked(extra.len(), "zip64 local extra length")?);
            hdr.extend_from_slice(name_bytes);
            hdr.extend_from_slice(&extra);
            self.write(output, out_pos, &hdr)?;
            out_pos += hdr.len() as u64;
            let (comp_size, chunk_sizes) =
                self.write_zip_deflate_payload(path, *size, output, out_pos, &mut deflate)?;
            patch_zip64_local_sizes(
                self,
                output,
                local_offset + 30 + name_bytes.len() as u64,
                *size,
                comp_size,
                &chunk_sizes,
            )?;
            out_pos += comp_size;
            central.push(ZipCentralEntry {
                crc: *crc,
                size: *size,
                comp_size,
                local_offset,
                name: name.clone(),
            });
            input_bytes += *size;
        }
        let cd_start = out_pos;
        for entry in &central {
            let name_bytes = entry.name.as_bytes();
            let extra = zip64_central_extra(entry.size, entry.comp_size, entry.local_offset);
            let mut hdr = Vec::with_capacity(46 + name_bytes.len() + extra.len());
            push_u32(&mut hdr, 0x0201_4b50);
            push_u16(&mut hdr, 45);
            push_u16(&mut hdr, 45);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 8);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 0);
            push_u32(&mut hdr, entry.crc);
            push_u32(&mut hdr, u32::MAX);
            push_u32(&mut hdr, u32::MAX);
            push_u16(
                &mut hdr,
                u16_checked(name_bytes.len(), "zip central file name length")?,
            );
            push_u16(&mut hdr, u16_checked(extra.len(), "zip64 central extra length")?);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 0);
            push_u32(&mut hdr, 0);
            push_u32(&mut hdr, u32::MAX);
            hdr.extend_from_slice(name_bytes);
            hdr.extend_from_slice(&extra);
            self.write(output, out_pos, &hdr)?;
            out_pos += hdr.len() as u64;
        }
        let cd_len = out_pos - cd_start;
        let zip64_eocd_offset = out_pos;
        let mut zip64 = Vec::with_capacity(56);
        push_u32(&mut zip64, 0x0606_4b50);
        push_u64(&mut zip64, 44);
        push_u16(&mut zip64, 45);
        push_u16(&mut zip64, 45);
        push_u32(&mut zip64, 0);
        push_u32(&mut zip64, 0);
        push_u64(&mut zip64, central.len() as u64);
        push_u64(&mut zip64, central.len() as u64);
        push_u64(&mut zip64, cd_len);
        push_u64(&mut zip64, cd_start);
        self.write(output, out_pos, &zip64)?;
        out_pos += zip64.len() as u64;

        let mut zip64_locator = Vec::with_capacity(20);
        push_u32(&mut zip64_locator, 0x0706_4b50);
        push_u32(&mut zip64_locator, 0);
        push_u64(&mut zip64_locator, zip64_eocd_offset);
        push_u32(&mut zip64_locator, 1);
        self.write(output, out_pos, &zip64_locator)?;
        out_pos += zip64_locator.len() as u64;

        let mut eocd = Vec::with_capacity(22);
        push_u32(&mut eocd, 0x0605_4b50);
        push_u16(&mut eocd, 0);
        push_u16(&mut eocd, 0);
        push_u16(&mut eocd, u16::MAX);
        push_u16(&mut eocd, u16::MAX);
        push_u32(&mut eocd, u32::MAX);
        push_u32(&mut eocd, u32::MAX);
        push_u16(&mut eocd, 0);
        self.write(output, out_pos, &eocd)?;
        out_pos += eocd.len() as u64;
        self.set_size(output, out_pos)?;
        Ok(ArchiveJobStats {
            format: "zip".to_string(),
            output: output.to_string(),
            file_count: planned.len(),
            input_bytes,
            archive_bytes: out_pos,
            elapsed_ms: start.elapsed().as_millis(),
        })
    }

    fn archive_zip_extract_gpu(
        &mut self,
        archive: &str,
        output_dir: &str,
        start: Instant,
    ) -> EResult<ArchiveExtractStats> {
        let archive_size = {
            let node = self.table.get(archive).ok_or(LookupError::NotFound)?;
            if node.is_dir {
                return Err(EngineError::NotAFile);
            }
            node.size
        };
        let out_base = crate::lookup::normalize(output_dir);
        self.ensure_dir_path(&out_base)?;
        let mut deflate_codec = cuda(NvcompBatchedCodec::load(
            &self.vram,
            NvcompFrameCodec::Deflate,
        ))?;
        let mut pos = 0u64;
        let mut files = 0usize;
        let mut output_bytes = 0u64;
        while pos + 4 <= archive_size {
            let sig = self.read(archive, pos, 4)?;
            let sig = read_u32_le(&sig);
            if sig == 0x0201_4b50 || sig == 0x0605_4b50 {
                break;
            }
            if sig != 0x0403_4b50 {
                return Err(EngineError::Cuda(format!(
                    "unsupported zip signature {sig:08x} at {pos}"
                )));
            }
            let hdr = self.read(archive, pos, 30)?;
            let method = read_u16_le(&hdr[8..10]);
            let comp32 = read_u32_le(&hdr[18..22]);
            let uncomp32 = read_u32_le(&hdr[22..26]);
            let name_len = read_u16_le(&hdr[26..28]) as u64;
            let extra_len = read_u16_le(&hdr[28..30]) as u64;
            let name_bytes = self.read(archive, pos + 30, name_len as usize)?;
            let extra = self.read(
                archive,
                pos + 30 + name_len,
                usize::try_from(extra_len)
                    .map_err(|_| EngineError::Cuda("zip extra length exceeds usize".into()))?,
            )?;
            let (uncomp_size, comp_size) = zip_sizes_from_local_extra(uncomp32, comp32, &extra)?;
            let name = std::str::from_utf8(&name_bytes)
                .map_err(|e| EngineError::Cuda(format!("invalid zip path UTF-8: {e}")))?;
            let data_pos = pos + 30 + name_len + extra_len;
            let out_path = join_archive_output(&out_base, name)?;
            self.ensure_parent_dirs(&out_path)?;
            self.create_or_truncate_file(&out_path)?;
            match method {
                0 => self.copy_file_payload_raw(archive, data_pos, &out_path, 0, uncomp_size)?,
                8 => {
                    if let Some(chunk_sizes) = zip_deflate_chunks_from_extra(&extra)? {
                        self.extract_zip_deflate_chunks(
                            &mut deflate_codec,
                            archive,
                            data_pos,
                            &chunk_sizes,
                            &out_path,
                            uncomp_size,
                        )?;
                    } else if comp_size == stored_deflate_len(uncomp_size) {
                        let written = self.extract_stored_deflate_stream(
                            archive, data_pos, comp_size, &out_path,
                        )?;
                        if written != uncomp_size {
                            return Err(EngineError::Cuda(format!(
                                "zip stored-deflate size mismatch: expected {uncomp_size}, got {written}"
                            )));
                        }
                    } else {
                        self.extract_deflate_payload(
                            &mut deflate_codec,
                            archive,
                            data_pos,
                            comp_size,
                            &out_path,
                            0,
                            uncomp_size,
                        )?;
                    }
                }
                _ => {
                    return Err(EngineError::Cuda(format!(
                        "unsupported zip compression method: {method}"
                    )));
                }
            }
            self.set_size(&out_path, uncomp_size)?;
            files += 1;
            output_bytes += uncomp_size;
            pos = data_pos + comp_size;
        }
        Ok(ArchiveExtractStats {
            format: "zip".to_string(),
            archive: archive.to_string(),
            output_dir: out_base,
            file_count: files,
            archive_bytes: archive_size,
            output_bytes,
            elapsed_ms: start.elapsed().as_millis(),
        })
    }

    fn write_zip_deflate_payload(
        &mut self,
        src_path: &str,
        len: u64,
        dst_path: &str,
        mut out_pos: u64,
        codec: &mut NvcompBatchedCodec,
    ) -> EResult<(u64, Vec<u64>)> {
        if len == 0 {
            self.write(dst_path, out_pos, &[1, 0, 0, 255, 255])?;
            return Ok((5, vec![5]));
        }
        let tmp = format!("\\.__vramdisk_zip_src_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&tmp)?;
        self.allocate_raw_file(&tmp, len)?;
        self.copy_file_payload_raw(src_path, 0, &tmp, 0, len)?;
        let base = self.contiguous_file_ptr(&tmp, len)?;
        let total_chunks = len.div_ceil(ZIP_DEFLATE_CHUNK);
        let mut chunk_idx = 0u64;
        let mut written = 0u64;
        let mut chunk_comp_sizes = Vec::with_capacity(total_chunks as usize);
        while chunk_idx < total_chunks {
            let n = ((total_chunks - chunk_idx).min(crate::nvcomp::BATCH as u64)) as usize;
            let mut ptrs = Vec::with_capacity(n);
            let mut sizes = Vec::with_capacity(n);
            for i in 0..n {
                let off = (chunk_idx + i as u64) * ZIP_DEFLATE_CHUNK;
                let take = (len - off).min(ZIP_DEFLATE_CHUNK);
                ptrs.push(base + off);
                sizes.push(take);
            }
            let comp_sizes = cuda(codec.compress_device(&ptrs, &sizes))?;
            for (i, comp_size) in comp_sizes.into_iter().enumerate() {
                let is_last = chunk_idx + i as u64 + 1 == total_chunks;
                self.write_device_bytes(dst_path, out_pos, codec.compressed_slot_ptr(i), comp_size)?;
                if !is_last {
                    self.clear_zip_deflate_bfinal(dst_path, out_pos)?;
                }
                out_pos += comp_size;
                written += comp_size;
                chunk_comp_sizes.push(comp_size);
            }
            chunk_idx += n as u64;
        }
        let _ = self.remove(&tmp);
        Ok((written, chunk_comp_sizes))
    }

    fn clear_zip_deflate_bfinal(&mut self, path: &str, offset: u64) -> EResult<()> {
        let mut b = self.read(path, offset, 1)?;
        if b.len() != 1 {
            return Err(EngineError::Cuda("truncated deflate payload".into()));
        }
        b[0] &= !1;
        self.write(path, offset, &b)?;
        Ok(())
    }

    fn set_zip_deflate_bfinal(&mut self, path: &str, offset: u64) -> EResult<()> {
        let mut b = self.read(path, offset, 1)?;
        if b.len() != 1 {
            return Err(EngineError::Cuda("truncated deflate payload".into()));
        }
        b[0] |= 1;
        self.write(path, offset, &b)?;
        Ok(())
    }

    fn extract_zip_deflate_chunks(
        &mut self,
        codec: &mut NvcompBatchedCodec,
        src_path: &str,
        src_offset: u64,
        chunk_comp_sizes: &[u64],
        dst_path: &str,
        out_len: u64,
    ) -> EResult<()> {
        if out_len == 0 {
            self.set_size(dst_path, 0)?;
            return Ok(());
        }
        self.allocate_raw_file(dst_path, out_len)?;
        let dst_base = self.contiguous_file_ptr(dst_path, out_len)?;
        let mut comp_pos = src_offset;
        let mut out_pos = 0u64;
        for &comp_len in chunk_comp_sizes {
            let take = (out_len - out_pos).min(ZIP_DEFLATE_CHUNK);
            let tmp = format!("\\.__vramdisk_zip_deflate_src_{}", crate::lookup::now_filetime());
            self.create_or_truncate_file(&tmp)?;
            self.allocate_raw_file(&tmp, comp_len)?;
            self.copy_file_payload_raw(src_path, comp_pos, &tmp, 0, comp_len)?;
            self.set_zip_deflate_bfinal(&tmp, 0)?;
            let src_ptr = self.contiguous_file_ptr(&tmp, comp_len)?;
            cuda(codec.decompress_device(
                &[src_ptr],
                &[comp_len],
                &[dst_base + out_pos],
                &[take],
            ))?;
            let _ = self.remove(&tmp);
            comp_pos += comp_len;
            out_pos += take;
            if out_pos == out_len {
                break;
            }
        }
        if out_pos != out_len {
            return Err(EngineError::Cuda("ZIP Deflate chunk table ended early".into()));
        }
        Ok(())
    }

    fn crc32_file_gpu(&mut self, path: &str, len: u64) -> EResult<u32> {
        let segs = self.file_segments_raw(path, 0, len)?;
        let out = cuda(self.api_kernel()?.crc32_many(&[segs]))?;
        Ok(out[0])
    }

    fn crc32_range_gpu(&mut self, path: &str, offset: u64, len: u64) -> EResult<u32> {
        let segs = self.file_segments_raw(path, offset, len)?;
        let out = cuda(self.api_kernel()?.crc32_many(&[segs]))?;
        Ok(out[0])
    }

    fn raw_file_ptr(&self, path: &str, offset: u64) -> EResult<u64> {
        let lc = (offset / CHUNK_SIZE) as usize;
        let in_off = offset % CHUNK_SIZE;
        match self.coord(path, lc) {
            Some(Placement::Raw { chunk }) => {
                Ok(self.vram_base + chunk as u64 * CHUNK_SIZE + in_off)
            }
            Some(Placement::Compressed { .. }) => Err(EngineError::Cuda(
                "archive codec input currently requires raw placement".into(),
            )),
            None => Err(EngineError::Cuda("archive codec input is sparse".into())),
        }
    }

    fn raw_output_ptr(&mut self, path: &str, offset: u64, len: u64) -> EResult<u64> {
        if len > CHUNK_SIZE - (offset % CHUNK_SIZE) {
            return Err(EngineError::Cuda(
                "archive codec output slice crosses a chunk boundary".into(),
            ));
        }
        self.ensure_logical_len(path, offset + len)?;
        let lc = (offset / CHUNK_SIZE) as usize;
        let in_off = offset % CHUNK_SIZE;
        let chunk = self.ensure_raw_output_chunk(path, lc)?;
        let node = self.table.get_mut(path).ok_or(LookupError::NotFound)?;
        node.size = node.size.max(offset + len);
        Ok(self.vram_base + chunk as u64 * CHUNK_SIZE + in_off)
    }

    fn file_segments_raw(&self, path: &str, offset: u64, len: u64) -> EResult<Vec<HashSegment>> {
        let mut segs = Vec::new();
        let mut done = 0u64;
        while done < len {
            let pos = offset + done;
            let lc = (pos / CHUNK_SIZE) as usize;
            let in_off = pos % CHUNK_SIZE;
            let take = (len - done).min(CHUNK_SIZE - in_off) as u32;
            match self.coord(path, lc) {
                None => segs.push(HashSegment {
                    ptr: 0,
                    len: take,
                    kind: 1,
                }),
                Some(Placement::Raw { chunk }) => segs.push(HashSegment {
                    ptr: self.vram_base + chunk as u64 * CHUNK_SIZE + in_off,
                    len: take,
                    kind: 0,
                }),
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Cuda(
                        "archive CRC32 currently requires raw/sparse placements".into(),
                    ));
                }
            }
            done += take as u64;
        }
        Ok(segs)
    }

    fn extract_deflate_payload(
        &mut self,
        codec: &mut NvcompBatchedCodec,
        src_path: &str,
        src_offset: u64,
        comp_len: u64,
        dst_path: &str,
        dst_offset: u64,
        out_len: u64,
    ) -> EResult<()> {
        let tmp = format!("\\.__vramdisk_deflate_src_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&tmp)?;
        self.allocate_raw_file(&tmp, comp_len)?;
        self.copy_file_payload_raw(src_path, src_offset, &tmp, 0, comp_len)?;
        let src_ptr = self.contiguous_file_ptr(&tmp, comp_len)?;
        let dst_ptr = if dst_offset == 0 {
            self.allocate_raw_file(dst_path, out_len)?;
            self.contiguous_file_ptr(dst_path, out_len)?
        } else {
            self.raw_output_ptr(dst_path, dst_offset, out_len)?
        };
        cuda(codec.decompress_device(&[src_ptr], &[comp_len], &[dst_ptr], &[out_len]))?;
        let _ = self.remove(&tmp);
        Ok(())
    }

    fn extract_lz4_payload(
        &mut self,
        codec: &mut NvcompBatchedCodec,
        src_path: &str,
        src_offset: u64,
        comp_len: u64,
        dst_path: &str,
        dst_offset: u64,
        out_len: u64,
    ) -> EResult<()> {
        let tmp = format!("\\.__vramdisk_lz4_src_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&tmp)?;
        self.allocate_raw_file(&tmp, comp_len)?;
        self.copy_file_payload_raw(src_path, src_offset, &tmp, 0, comp_len)?;
        let src_ptr = self.contiguous_file_ptr(&tmp, comp_len)?;
        let dst_ptr = self.raw_output_ptr(dst_path, dst_offset, out_len)?;
        cuda(codec.decompress_device(&[src_ptr], &[comp_len], &[dst_ptr], &[out_len]))?;
        let _ = self.remove(&tmp);
        Ok(())
    }

    fn write_gzip_deflate_members(
        &mut self,
        src_path: &str,
        len: u64,
        dst_path: &str,
    ) -> EResult<u64> {
        let mut codec = cuda(NvcompBatchedCodec::load(
            &self.vram,
            NvcompFrameCodec::Deflate,
        ))?;
        let mut out_pos = 0u64;
        let chunks = logical_chunks(len);
        let mut base = 0usize;
        while base < chunks {
            let n = (chunks - base).min(crate::nvcomp::BATCH);
            let mut ptrs = Vec::with_capacity(n);
            let mut sizes = Vec::with_capacity(n);
            let mut crc_files = Vec::with_capacity(n);
            for i in 0..n {
                let off = (base + i) as u64 * CHUNK_SIZE;
                let take = (len - off).min(CHUNK_SIZE);
                let ptr = self.raw_file_ptr(src_path, off)?;
                ptrs.push(ptr);
                sizes.push(take);
                crc_files.push(self.file_segments_raw(src_path, off, take)?);
            }
            let crcs = cuda(self.api_kernel()?.crc32_many(&crc_files))?;
            let comp_sizes = cuda(codec.compress_device(&ptrs, &sizes))?;
            for i in 0..n {
                let mut header = Vec::with_capacity(24);
                header.extend_from_slice(&[0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 255]);
                push_u16(&mut header, 12);
                header.extend_from_slice(b"GS");
                push_u16(&mut header, 8);
                header.extend_from_slice(&comp_sizes[i].to_le_bytes());
                self.write(dst_path, out_pos, &header)?;
                out_pos += header.len() as u64;
                self.write_device_bytes(
                    dst_path,
                    out_pos,
                    codec.compressed_slot_ptr(i),
                    comp_sizes[i],
                )?;
                out_pos += comp_sizes[i];
                let mut trailer = Vec::with_capacity(8);
                push_u32(&mut trailer, crcs[i]);
                push_u32(&mut trailer, (sizes[i] & 0xffff_ffff) as u32);
                self.write(dst_path, out_pos, &trailer)?;
                out_pos += 8;
            }
            base += n;
        }
        if len == 0 {
            let mut header = Vec::with_capacity(24);
            header.extend_from_slice(&[0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 255]);
            push_u16(&mut header, 12);
            header.extend_from_slice(b"GS");
            push_u16(&mut header, 8);
            header.extend_from_slice(&0u64.to_le_bytes());
            self.write(dst_path, out_pos, &header)?;
            out_pos += header.len() as u64;
            self.write(dst_path, out_pos, &[0, 0, 0, 0, 0, 0, 0, 0])?;
            out_pos += 8;
        }
        Ok(out_pos)
    }

    fn extract_gzip_deflate_members(
        &mut self,
        archive: &str,
        archive_size: u64,
        dst_path: &str,
    ) -> EResult<u64> {
        let mut codec = cuda(NvcompBatchedCodec::load(
            &self.vram,
            NvcompFrameCodec::Deflate,
        ))?;
        let mut src_pos = 0u64;
        let mut out_pos = 0u64;
        while src_pos < archive_size {
            let hdr = self.read(archive, src_pos, 10)?;
            if hdr.len() != 10 || hdr[0] != 0x1f || hdr[1] != 0x8b || hdr[2] != 8 {
                return Err(EngineError::Cuda("unsupported gzip header".into()));
            }
            if hdr[3] & 4 == 0 {
                return Err(EngineError::Cuda(
                    "gzip member is missing VRAMDISK compressed-size extra field".into(),
                ));
            }
            src_pos += 10;
            let xlen_buf = self.read(archive, src_pos, 2)?;
            let xlen = read_u16_le(&xlen_buf) as u64;
            src_pos += 2;
            let extra = self.read(archive, src_pos, xlen as usize)?;
            src_pos += xlen;
            let comp_len = gzip_extra_comp_len(&extra)?;
            let trailer_pos = src_pos + comp_len;
            if trailer_pos + 8 > archive_size {
                return Err(EngineError::Cuda("truncated gzip member".into()));
            }
            let trailer = self.read(archive, trailer_pos, 8)?;
            let expected_crc = read_u32_le(&trailer[0..4]);
            let expected_size = read_u32_le(&trailer[4..8]) as u64;
            if expected_size > 0 {
                self.extract_deflate_payload(
                    &mut codec,
                    archive,
                    src_pos,
                    comp_len,
                    dst_path,
                    out_pos,
                    expected_size,
                )?;
            }
            let actual_crc = self.crc32_range_gpu(dst_path, out_pos, expected_size)?;
            if actual_crc != expected_crc {
                return Err(EngineError::Cuda("gzip CRC32 mismatch".into()));
            }
            out_pos += expected_size;
            src_pos = trailer_pos + 8;
        }
        self.set_size(dst_path, out_pos)?;
        Ok(out_pos)
    }

    fn write_lz4_frame(&mut self, src_path: &str, len: u64, dst_path: &str) -> EResult<u64> {
        let mut codec = cuda(NvcompBatchedCodec::load(&self.vram, NvcompFrameCodec::Lz4))?;
        let mut header = Vec::with_capacity(15);
        header.extend_from_slice(&0x184d_2204u32.to_le_bytes());
        let flg = 0x68u8;
        let bd = 0x40u8;
        header.push(flg);
        header.push(bd);
        header.extend_from_slice(&len.to_le_bytes());
        let hc = lz4_header_checksum(&header[4..]);
        header.push(hc);
        self.write(dst_path, 0, &header)?;
        let mut out_pos = header.len() as u64;
        let chunks = logical_chunks(len);
        let mut base = 0usize;
        while base < chunks {
            let n = (chunks - base).min(crate::nvcomp::BATCH);
            let mut ptrs = Vec::with_capacity(n);
            let mut sizes = Vec::with_capacity(n);
            for i in 0..n {
                let off = (base + i) as u64 * CHUNK_SIZE;
                let take = (len - off).min(CHUNK_SIZE);
                ptrs.push(self.raw_file_ptr(src_path, off)?);
                sizes.push(take);
            }
            let comp_sizes = cuda(codec.compress_device(&ptrs, &sizes))?;
            for i in 0..n {
                let sz = comp_sizes[i];
                if sz > 0 && sz < sizes[i] {
                    self.write(dst_path, out_pos, &(sz as u32).to_le_bytes())?;
                    out_pos += 4;
                    self.write_device_bytes(dst_path, out_pos, codec.compressed_slot_ptr(i), sz)?;
                    out_pos += sz;
                } else {
                    let marker = (sizes[i] as u32) | 0x8000_0000;
                    self.write(dst_path, out_pos, &marker.to_le_bytes())?;
                    out_pos += 4;
                    self.copy_file_payload_raw(
                        src_path,
                        (base + i) as u64 * CHUNK_SIZE,
                        dst_path,
                        out_pos,
                        sizes[i],
                    )?;
                    out_pos += sizes[i];
                }
            }
            base += n;
        }
        self.write(dst_path, out_pos, &0u32.to_le_bytes())?;
        out_pos += 4;
        Ok(out_pos)
    }

    fn extract_lz4_frame(
        &mut self,
        archive: &str,
        archive_size: u64,
        dst_path: &str,
    ) -> EResult<u64> {
        let header = self.read(archive, 0, 15)?;
        if header.len() != 15 || read_u32_le(&header[0..4]) != 0x184d_2204 {
            return Err(EngineError::Cuda("unsupported LZ4 frame header".into()));
        }
        if header[4] != 0x68 || header[5] != 0x40 {
            return Err(EngineError::Cuda("unsupported LZ4 frame flags".into()));
        }
        if lz4_header_checksum(&header[4..14]) != header[14] {
            return Err(EngineError::Cuda(
                "LZ4 frame header checksum mismatch".into(),
            ));
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&header[6..14]);
        let total = u64::from_le_bytes(len_bytes);
        self.allocate_raw_file(dst_path, total)?;
        let mut codec = cuda(NvcompBatchedCodec::load(&self.vram, NvcompFrameCodec::Lz4))?;
        let mut src_pos = 15u64;
        let mut out_pos = 0u64;
        while src_pos + 4 <= archive_size {
            let sz_buf = self.read(archive, src_pos, 4)?;
            let marker = read_u32_le(&sz_buf);
            src_pos += 4;
            if marker == 0 {
                break;
            }
            let uncompressed = marker & 0x8000_0000 != 0;
            let block_len = (marker & 0x7fff_ffff) as u64;
            let out_len = (total - out_pos).min(CHUNK_SIZE);
            if src_pos + block_len > archive_size {
                return Err(EngineError::Cuda("truncated LZ4 frame block".into()));
            }
            if uncompressed {
                if block_len != out_len {
                    return Err(EngineError::Cuda("LZ4 raw block size mismatch".into()));
                }
                self.copy_file_payload_raw(archive, src_pos, dst_path, out_pos, out_len)?;
            } else {
                self.extract_lz4_payload(
                    &mut codec, archive, src_pos, block_len, dst_path, out_pos, out_len,
                )?;
            }
            src_pos += block_len;
            out_pos += out_len;
        }
        if out_pos != total {
            return Err(EngineError::Cuda("LZ4 frame content size mismatch".into()));
        }
        self.set_size(dst_path, total)?;
        Ok(total)
    }

    #[allow(dead_code)]
    fn write_stored_deflate_stream(
        &mut self,
        src_path: &str,
        len: u64,
        dst_path: &str,
        mut out_pos: u64,
        final_stream: bool,
    ) -> EResult<u64> {
        let mut done = 0u64;
        if len == 0 {
            self.write(dst_path, out_pos, &[1, 0, 0, 255, 255])?;
            return Ok(out_pos + 5);
        }
        while done < len {
            let take = (len - done).min(65_535);
            let is_last = done + take == len;
            let bfinal = if final_stream && is_last { 1 } else { 0 };
            let len16 = take as u16;
            let nlen = !len16;
            let hdr = [
                bfinal,
                (len16 & 0xff) as u8,
                (len16 >> 8) as u8,
                (nlen & 0xff) as u8,
                (nlen >> 8) as u8,
            ];
            self.write(dst_path, out_pos, &hdr)?;
            out_pos += 5;
            self.copy_file_payload_raw(src_path, done, dst_path, out_pos, take)?;
            out_pos += take;
            done += take;
        }
        Ok(out_pos)
    }

    fn extract_stored_deflate_stream(
        &mut self,
        src_path: &str,
        mut src_pos: u64,
        comp_len: u64,
        dst_path: &str,
    ) -> EResult<u64> {
        let end = src_pos + comp_len;
        let mut out_pos = 0u64;
        while src_pos < end {
            let hdr = self.read(src_path, src_pos, 5)?;
            if hdr.len() != 5 {
                return Err(EngineError::Cuda("truncated stored deflate block".into()));
            }
            if hdr[0] & 0b0000_0110 != 0 {
                return Err(EngineError::Cuda(
                    "expected a stored deflate block in ZIP fallback".into(),
                ));
            }
            let len = u16::from_le_bytes([hdr[1], hdr[2]]) as u64;
            let nlen = u16::from_le_bytes([hdr[3], hdr[4]]);
            if nlen != !(len as u16) {
                return Err(EngineError::Cuda("invalid stored deflate LEN/NLEN".into()));
            }
            src_pos += 5;
            if src_pos + len > end {
                return Err(EngineError::Cuda(
                    "stored deflate block exceeds stream".into(),
                ));
            }
            self.copy_file_payload_raw(src_path, src_pos, dst_path, out_pos, len)?;
            out_pos += len;
            src_pos += len;
            if hdr[0] & 1 != 0 {
                break;
            }
        }
        if src_pos != end {
            return Err(EngineError::Cuda(
                "stored deflate stream length mismatch".into(),
            ));
        }
        Ok(out_pos)
    }

    fn ensure_raw_output_chunk(&mut self, path: &str, lc: usize) -> EResult<ChunkId> {
        match self.coord(path, lc) {
            Some(Placement::Raw { chunk }) => Ok(chunk),
            Some(Placement::Compressed { .. }) => Err(EngineError::Cuda(
                "archive output unexpectedly contains compressed placement".into(),
            )),
            None => {
                let chunk = self.alloc_chunk()?;
                cuda(self.vram.zero_at(chunk as u64 * CHUNK_SIZE, CHUNK_SIZE))?;
                self.set_coord(path, lc, Some(Placement::Raw { chunk }));
                Ok(chunk)
            }
        }
    }

    fn contiguous_file_ptr(&self, path: &str, len: u64) -> EResult<u64> {
        if len == 0 {
            return Ok(self.vram_base);
        }
        let chunks = logical_chunks(len);
        let first = match self.coord(path, 0) {
            Some(Placement::Raw { chunk }) => chunk,
            Some(Placement::Compressed { .. }) => {
                return Err(EngineError::Cuda(
                    "archive temp stream must be raw and contiguous".into(),
                ));
            }
            None => return Err(EngineError::Cuda("archive temp stream is sparse".into())),
        };
        for lc in 0..chunks {
            match self.coord(path, lc) {
                Some(Placement::Raw { chunk }) if chunk == first + lc as u32 => {}
                Some(Placement::Raw { .. }) => {
                    return Err(EngineError::Cuda(
                        "archive temp stream is not physically contiguous".into(),
                    ));
                }
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Cuda(
                        "archive temp stream must be raw and contiguous".into(),
                    ));
                }
                None => return Err(EngineError::Cuda("archive temp stream is sparse".into())),
            }
        }
        Ok(self.vram_base + first as u64 * CHUNK_SIZE)
    }

    fn ensure_dir_path(&mut self, path: &str) -> EResult<()> {
        let path = crate::lookup::normalize(path);
        if path == "\\" {
            return Ok(());
        }
        let mut cur = String::new();
        for comp in path.split('\\').filter(|s| !s.is_empty()) {
            cur.push('\\');
            cur.push_str(comp);
            match self.table.get(&cur) {
                Some(node) if node.is_dir => {}
                Some(_) => return Err(EngineError::NotAFile),
                None => {
                    self.table.create_dir(&cur, 0)?;
                }
            }
        }
        Ok(())
    }

    fn ensure_parent_dirs(&mut self, path: &str) -> EResult<()> {
        let path = crate::lookup::normalize(path);
        let Some(pos) = path.rfind('\\') else {
            return Ok(());
        };
        if pos == 0 {
            return Ok(());
        }
        self.ensure_dir_path(&path[..pos])
    }
}

struct ZipCentralEntry {
    crc: u32,
    size: u64,
    comp_size: u64,
    local_offset: u64,
    name: String,
}

fn vec_path(path: &str) -> String {
    path.to_string()
}

fn pad512(n: u64) -> u64 {
    (512 - (n % 512)) % 512
}

fn tar_header(path: &str, size: u64) -> EResult<[u8; 512]> {
    if path.is_empty() || path.len() > 100 {
        return Err(EngineError::Cuda(format!(
            "tar path must be 1..100 ASCII bytes for current GPU archive writer: {path}"
        )));
    }
    if !path.is_ascii() {
        return Err(EngineError::Cuda(format!(
            "tar path must be ASCII for current GPU archive writer: {path}"
        )));
    }
    let mut h = [0u8; 512];
    h[..path.len()].copy_from_slice(path.as_bytes());
    write_octal(&mut h[100..108], 0o644);
    write_octal(&mut h[108..116], 0);
    write_octal(&mut h[116..124], 0);
    write_octal(&mut h[124..136], size);
    write_octal(&mut h[136..148], 0);
    for b in &mut h[148..156] {
        *b = b' ';
    }
    h[156] = b'0';
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    let chk = format!("{sum:06o}\0 ");
    h[148..156].copy_from_slice(chk.as_bytes());
    Ok(h)
}

fn write_octal(dst: &mut [u8], value: u64) {
    for b in dst.iter_mut() {
        *b = 0;
    }
    let width = dst.len().saturating_sub(1);
    let s = format!("{value:0width$o}");
    let bytes = s.as_bytes();
    let start = width.saturating_sub(bytes.len());
    dst[start..start + bytes.len()].copy_from_slice(bytes);
}

fn parse_tar_octal(src: &[u8]) -> EResult<u64> {
    let s = src
        .iter()
        .copied()
        .take_while(|&b| b != 0 && b != b' ')
        .filter(|&b| b != 0)
        .collect::<Vec<_>>();
    let text = std::str::from_utf8(&s)
        .map_err(|e| EngineError::Cuda(format!("invalid tar octal field: {e}")))?;
    u64::from_str_radix(text.trim(), 8)
        .map_err(|e| EngineError::Cuda(format!("invalid tar octal value: {e}")))
}

/// Join an archive entry name onto the extraction base, rejecting `.`/`..`
/// path components so a crafted archive can't place a literal `..` node in
/// the namespace (tar/zip "slip") instead of extracting under `base`.
fn join_archive_output(base: &str, name: &str) -> EResult<String> {
    let clean = name.trim_start_matches('/').replace('/', "\\");
    if clean.split('\\').any(|comp| comp == "." || comp == "..") {
        return Err(EngineError::Cuda(format!(
            "refusing archive entry with unsafe path component: {name}"
        )));
    }
    Ok(if base == "\\" {
        format!("\\{clean}")
    } else {
        format!("{base}\\{clean}")
    })
}

fn gzip_extra_comp_len(extra: &[u8]) -> EResult<u64> {
    let mut pos = 0usize;
    while pos + 4 <= extra.len() {
        let si1 = extra[pos];
        let si2 = extra[pos + 1];
        let len = read_u16_le(&extra[pos + 2..pos + 4]) as usize;
        pos += 4;
        if pos + len > extra.len() {
            return Err(EngineError::Cuda("invalid gzip extra length".into()));
        }
        if si1 == b'G' && si2 == b'S' && len == 8 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&extra[pos..pos + 8]);
            return Ok(u64::from_le_bytes(bytes));
        }
        pos += len;
    }
    Err(EngineError::Cuda(
        "gzip member is missing VRAMDISK compressed-size subfield".into(),
    ))
}

fn zip_local_extra(uncomp_size: u64, chunk_comp_sizes: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + 8 + chunk_comp_sizes.len() * 8);
    push_u16(&mut out, 0x0001);
    push_u16(&mut out, 16);
    push_u64(&mut out, uncomp_size);
    push_u64(&mut out, chunk_comp_sizes.iter().sum());
    let mut start = 0usize;
    while start < chunk_comp_sizes.len() {
        let take = (chunk_comp_sizes.len() - start).min(8190);
        push_u16(&mut out, 0x4753);
        push_u16(&mut out, (8 + take * 8) as u16);
        push_u32(&mut out, start as u32);
        push_u32(&mut out, take as u32);
        for &size in &chunk_comp_sizes[start..start + take] {
            push_u64(&mut out, size);
        }
        start += take;
    }
    out
}

fn zip_deflate_chunk_count(len: u64) -> usize {
    let count = len.div_ceil(ZIP_DEFLATE_CHUNK).max(1);
    usize::try_from(count).unwrap_or(usize::MAX)
}

fn zip64_central_extra(uncomp_size: u64, comp_size: u64, local_offset: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(28);
    push_u16(&mut out, 0x0001);
    push_u16(&mut out, 24);
    push_u64(&mut out, uncomp_size);
    push_u64(&mut out, comp_size);
    push_u64(&mut out, local_offset);
    out
}

fn zip_sizes_from_local_extra(uncomp32: u32, comp32: u32, extra: &[u8]) -> EResult<(u64, u64)> {
    if uncomp32 != u32::MAX && comp32 != u32::MAX {
        return Ok((uncomp32 as u64, comp32 as u64));
    }
    let mut pos = 0usize;
    while pos + 4 <= extra.len() {
        let tag = read_u16_le(&extra[pos..pos + 2]);
        let len = read_u16_le(&extra[pos + 2..pos + 4]) as usize;
        pos += 4;
        if pos + len > extra.len() {
            return Err(EngineError::Cuda("invalid ZIP extra field length".into()));
        }
        if tag == 0x0001 {
            let field = &extra[pos..pos + len];
            if field.len() < 16 {
                return Err(EngineError::Cuda("truncated ZIP64 size extra field".into()));
            }
            return Ok((read_u64_le(&field[0..8]), read_u64_le(&field[8..16])));
        }
        pos += len;
    }
    Err(EngineError::Cuda(
        "ZIP entry uses 0xffffffff sizes without ZIP64 extra field".into(),
    ))
}

fn zip_deflate_chunks_from_extra(extra: &[u8]) -> EResult<Option<Vec<u64>>> {
    let mut pos = 0usize;
    let mut sizes = Vec::new();
    while pos + 4 <= extra.len() {
        let tag = read_u16_le(&extra[pos..pos + 2]);
        let len = read_u16_le(&extra[pos + 2..pos + 4]) as usize;
        pos += 4;
        if pos + len > extra.len() {
            return Err(EngineError::Cuda("invalid ZIP extra field length".into()));
        }
        if tag == 0x4753 {
            if len < 8 || (len - 8) % 8 != 0 {
                return Err(EngineError::Cuda("invalid VRAMDISK ZIP chunk table".into()));
            }
            let start = read_u32_le(&extra[pos..pos + 4]) as usize;
            let count = read_u32_le(&extra[pos + 4..pos + 8]) as usize;
            if count != (len - 8) / 8 {
                return Err(EngineError::Cuda("VRAMDISK ZIP chunk table count mismatch".into()));
            }
            if sizes.len() < start {
                return Err(EngineError::Cuda("VRAMDISK ZIP chunk table has a gap".into()));
            }
            if sizes.len() == start {
                sizes.reserve(count);
            }
            let mut p = pos + 8;
            for _ in 0..count {
                sizes.push(read_u64_le(&extra[p..p + 8]));
                p += 8;
            }
        }
        pos += len;
    }
    if sizes.is_empty() {
        Ok(None)
    } else {
        Ok(Some(sizes))
    }
}

fn patch_zip64_local_sizes(
    engine: &mut StorageEngine,
    path: &str,
    extra_offset: u64,
    uncomp_size: u64,
    comp_size: u64,
    chunk_comp_sizes: &[u64],
) -> EResult<()> {
    let extra = zip_local_extra(uncomp_size, chunk_comp_sizes);
    debug_assert_eq!(chunk_comp_sizes.iter().sum::<u64>(), comp_size);
    engine.write(path, extra_offset, &extra)?;
    Ok(())
}

fn lz4_header_checksum(desc: &[u8]) -> u8 {
    ((xxhash32(desc, 0) >> 8) & 0xff) as u8
}

fn xxhash32(data: &[u8], seed: u32) -> u32 {
    const P1: u32 = 0x9E37_79B1;
    const P2: u32 = 0x85EB_CA77;
    const P3: u32 = 0xC2B2_AE3D;
    const P4: u32 = 0x27D4_EB2F;
    const P5: u32 = 0x1656_67B1;

    fn round(acc: u32, input: u32) -> u32 {
        acc.wrapping_add(input.wrapping_mul(P2))
            .rotate_left(13)
            .wrapping_mul(P1)
    }

    let mut i = 0usize;
    let mut h = if data.len() >= 16 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        while i + 16 <= data.len() {
            v1 = round(v1, read_u32_le(&data[i..i + 4]));
            v2 = round(v2, read_u32_le(&data[i + 4..i + 8]));
            v3 = round(v3, read_u32_le(&data[i + 8..i + 12]));
            v4 = round(v4, read_u32_le(&data[i + 12..i + 16]));
            i += 16;
        }
        v1.rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18))
    } else {
        seed.wrapping_add(P5)
    };
    h = h.wrapping_add(data.len() as u32);
    while i + 4 <= data.len() {
        h = h
            .wrapping_add(read_u32_le(&data[i..i + 4]).wrapping_mul(P3))
            .rotate_left(17)
            .wrapping_mul(P4);
        i += 4;
    }
    while i < data.len() {
        h = h
            .wrapping_add((data[i] as u32).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
        i += 1;
    }
    h ^= h >> 15;
    h = h.wrapping_mul(P2);
    h ^= h >> 13;
    h = h.wrapping_mul(P3);
    h ^ (h >> 16)
}

fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u16_le(src: &[u8]) -> u16 {
    u16::from_le_bytes([src[0], src[1]])
}

fn read_u32_le(src: &[u8]) -> u32 {
    u32::from_le_bytes([src[0], src[1], src[2], src[3]])
}

fn read_u64_le(src: &[u8]) -> u64 {
    u64::from_le_bytes([
        src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
    ])
}

fn u16_checked(v: usize, what: &str) -> EResult<u16> {
    u16::try_from(v).map_err(|_| EngineError::Cuda(format!("{what} exceeds u16")))
}

#[allow(dead_code)]
fn u32_checked(v: u64, what: &str) -> EResult<u32> {
    u32::try_from(v).map_err(|_| EngineError::Cuda(format!("{what} exceeds u32")))
}

fn stored_deflate_len(len: u64) -> u64 {
    if len == 0 {
        return 5;
    }
    len + len.div_ceil(65_535) * 5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_archive_output_joins_under_base() {
        assert_eq!(join_archive_output("\\out", "a.txt").unwrap(), "\\out\\a.txt");
        assert_eq!(join_archive_output("\\", "a.txt").unwrap(), "\\a.txt");
        assert_eq!(
            join_archive_output("\\out", "sub/a.txt").unwrap(),
            "\\out\\sub\\a.txt"
        );
    }

    #[test]
    fn join_archive_output_rejects_dot_dot_and_dot() {
        assert!(join_archive_output("\\out", "../evil.txt").is_err());
        assert!(join_archive_output("\\out", "a/../../evil.txt").is_err());
        assert!(join_archive_output("\\out", "./a.txt").is_err());
        assert!(join_archive_output("\\out", "sub/../evil.txt").is_err());
    }

    fn engine(mib: u64, dedup: bool) -> StorageEngine {
        let vram = Vram::new(0, mib * 1024 * 1024).expect("alloc vram for test");
        StorageEngine::new(vram, false, dedup).expect("engine")
    }

    fn engine_compress(mib: u64) -> StorageEngine {
        let vram = Vram::new(0, mib * 1024 * 1024).expect("alloc vram for test");
        StorageEngine::new(vram, true, false).expect("compress engine (nvCOMP)")
    }

    fn archive_engine(mib: u64) -> StorageEngine {
        let vram = Vram::new(0, mib * 1024 * 1024).expect("alloc vram for archive test");
        StorageEngine::new(vram, false, false).expect("archive engine")
    }

    #[test]
    #[ignore]
    fn archive_gpu_tar_zstd_gzip_roundtrip_and_bench() {
        let mut e = archive_engine(512);
        e.table_mut().create_dir("\\data", 0).unwrap();
        let specs = [
            ("\\data\\a.bin", 8 * 1024 * 1024usize),
            ("\\data\\b.bin", 5 * 1024 * 1024usize),
            ("\\data\\c.bin", 3 * 1024 * 1024usize),
        ];
        let mut originals = Vec::new();
        for (idx, (path, len)) in specs.iter().enumerate() {
            e.table_mut().create_file(path, 0).unwrap();
            let mut data = vec![0u8; *len];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i / 97 + idx * 31) & 0xff) as u8;
            }
            e.write(path, 0, &data).unwrap();
            originals.push((path.to_string(), data));
        }
        let paths: Vec<String> = specs.iter().map(|(p, _)| p.to_string()).collect();
        for (codec, out, extracted) in [
            (NvcompFrameCodec::Zstd, "\\bench.tar.zst", "\\unzstd"),
            (NvcompFrameCodec::Lz4, "\\bench.tar.lz4", "\\unlz4"),
            (NvcompFrameCodec::Gzip, "\\bench.tar.gz", "\\ungzip"),
            (NvcompFrameCodec::Deflate, "\\bench.zip", "\\unzip"),
        ] {
            let c = e.archive_compress_gpu(codec, &paths, out).unwrap();
            println!(
                "{} compress: input={} archive={} elapsed={}ms throughput={:.2} MiB/s",
                c.format,
                c.input_bytes,
                c.archive_bytes,
                c.elapsed_ms,
                (c.input_bytes as f64 / 1048576.0) / (c.elapsed_ms.max(1) as f64 / 1000.0)
            );
            let x = e.archive_extract_gpu(codec, out, extracted).unwrap();
            println!(
                "{} extract: archive={} output={} elapsed={}ms throughput={:.2} MiB/s",
                x.format,
                x.archive_bytes,
                x.output_bytes,
                x.elapsed_ms,
                (x.output_bytes as f64 / 1048576.0) / (x.elapsed_ms.max(1) as f64 / 1000.0)
            );
            for (path, expected) in &originals {
                let rel = path.trim_start_matches('\\');
                let got = e
                    .read(&format!("{extracted}\\{rel}"), 0, expected.len())
                    .unwrap();
                assert_eq!(got, *expected);
            }
        }
    }

    #[allow(dead_code)]
    fn engine_compress_dedup(mib: u64) -> StorageEngine {
        let vram = Vram::new(0, mib * 1024 * 1024).expect("alloc vram for test");
        StorageEngine::new(vram, true, true).expect("compress+dedup engine (nvCOMP)")
    }

    #[test]
    fn write_read_roundtrip_small() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\a", 0).unwrap();
        let data = b"hello VRAMDISK";
        assert_eq!(e.write("\\a", 0, data).unwrap(), data.len() as u64);
        assert_eq!(e.get("\\a").unwrap().size, data.len() as u64);
        assert_eq!(e.read("\\a", 0, 1024).unwrap(), data);
    }

    #[test]
    fn gpu_api_hash_known_vectors() {
        let mut e = engine(2, false);
        e.table_mut().create_file("\\abc", 0).unwrap();
        e.write("\\abc", 0, b"abc").unwrap();
        assert_eq!(
            crate::api_kernel::digest_hex(
                &e.hash_file_gpu("\\abc", crate::api_kernel::HashAlgorithm::Md5)
                    .unwrap()
            ),
            "900150983cd24fb0d6963f7d28e17f72"
        );
        assert_eq!(
            crate::api_kernel::digest_hex(
                &e.hash_file_gpu("\\abc", crate::api_kernel::HashAlgorithm::Sha1)
                    .unwrap()
            ),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            crate::api_kernel::digest_hex(
                &e.hash_file_gpu("\\abc", crate::api_kernel::HashAlgorithm::Sha256)
                    .unwrap()
            ),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        e.table_mut().create_file("\\sparse", 0).unwrap();
        e.set_size("\\sparse", CHUNK_SIZE + 3).unwrap();
        e.write("\\sparse", CHUNK_SIZE, b"abc").unwrap();
        assert_eq!(
            e.read("\\sparse", CHUNK_SIZE, 3).unwrap(),
            b"abc",
            "sparse fixture sanity"
        );
        let _ = e
            .hash_file_gpu("\\sparse", crate::api_kernel::HashAlgorithm::Sha256)
            .unwrap();
    }

    #[test]
    fn write_spans_multiple_chunks() {
        let mut e = engine(2, false);
        e.table_mut().create_file("\\big", 0).unwrap();
        let n = 150 * 1024usize;
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let off = 70 * 1024u64;
        e.write("\\big", off, &data).unwrap();
        assert_eq!(e.get("\\big").unwrap().size, off + n as u64);
        let head = e.read("\\big", 0, off as usize).unwrap();
        assert!(head.iter().all(|&b| b == 0));
        assert_eq!(e.read("\\big", off, n).unwrap(), data);
    }

    #[test]
    fn raw_full_chunk_write_uses_contiguous_storage() {
        let mut e = engine(8, false);
        e.table_mut().create_file("\\big", 0).unwrap();
        let n = CHUNK_SIZE as usize * 8;
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        e.write("\\big", 0, &data).unwrap();
        assert_eq!(e.used_chunks(), 8);
        for lc in 0..8 {
            assert_eq!(
                e.coord("\\big", lc),
                Some(Placement::Raw { chunk: lc as u32 })
            );
        }
        assert_eq!(e.read("\\big", 0, n).unwrap(), data);
    }

    #[test]
    fn partial_chunk_zero_fill() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\p", 0).unwrap();
        e.write("\\p", 10, b"XYZWV").unwrap();
        let got = e.read("\\p", 0, 20).unwrap();
        let mut want = vec![0u8; 15];
        want[10..15].copy_from_slice(b"XYZWV");
        assert_eq!(got, want);
    }

    #[test]
    fn read_clamps_to_eof() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\c", 0).unwrap();
        e.write("\\c", 0, b"abcdef").unwrap();
        assert_eq!(e.read("\\c", 4, 100).unwrap(), b"ef");
        assert!(e.read("\\c", 6, 100).unwrap().is_empty());
    }

    #[test]
    fn truncate_frees_and_extends() {
        let mut e = engine(2, false);
        e.table_mut().create_file("\\t", 0).unwrap();
        let data = vec![7u8; 200 * 1024];
        e.write("\\t", 0, &data).unwrap();
        let used_full = e.used_chunks();
        assert!(used_full >= 4);
        e.set_size("\\t", 1).unwrap();
        assert!(e.used_chunks() < used_full);
        assert_eq!(e.read("\\t", 0, 10).unwrap(), vec![7u8]);
        e.set_size("\\t", 100 * 1024).unwrap();
        let tail = e.read("\\t", 1, 100 * 1024).unwrap();
        assert!(tail.iter().all(|&b| b == 0));
    }

    #[test]
    fn overwrite_updates_in_place() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\o", 0).unwrap();
        e.write("\\o", 0, b"AAAAAAAA").unwrap();
        let before = e.used_chunks();
        e.write("\\o", 2, b"bb").unwrap();
        assert_eq!(e.used_chunks(), before);
        assert_eq!(e.read("\\o", 0, 8).unwrap(), b"AAbbAAAA");
    }

    #[test]
    fn write_past_capacity_is_rejected_not_panic() {
        // 1 MiB volume = 16 chunks. A write whose end exceeds the volume's
        // logical capacity must return NoSpace, never try to allocate a giant
        // coordinate vector (which would OOM-panic and tear the mount down).
        let mut e = engine(1, false);
        e.table_mut().create_file("\\big", 0).unwrap();
        let off = 16 * CHUNK_SIZE; // first byte beyond the last addressable chunk
        assert!(matches!(
            e.write("\\big", off, b"x"),
            Err(EngineError::NoSpace)
        ));
        // The file is unchanged and the engine still works.
        assert_eq!(e.get("\\big").unwrap().size, 0);
        e.write("\\big", 0, b"ok").unwrap();
        assert_eq!(e.read("\\big", 0, 2).unwrap(), b"ok");
    }

    #[test]
    fn write_offset_overflow_is_rejected() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\o", 0).unwrap();
        // offset + len overflows u64: must error, not panic.
        assert!(matches!(
            e.write("\\o", u64::MAX - 4, b"abcdefgh"),
            Err(EngineError::NoSpace)
        ));
    }

    #[test]
    fn set_size_huge_is_rejected_not_panic() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\t", 0).unwrap();
        assert!(matches!(
            e.set_size("\\t", u64::MAX),
            Err(EngineError::NoSpace)
        ));
        assert!(matches!(
            e.set_size("\\t", 17 * CHUNK_SIZE),
            Err(EngineError::NoSpace)
        ));
        // A within-capacity truncate still works.
        e.set_size("\\t", 4 * CHUNK_SIZE).unwrap();
        assert_eq!(e.get("\\t").unwrap().size, 4 * CHUNK_SIZE);
    }

    #[test]
    fn remove_frees_chunks() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\r", 0).unwrap();
        e.write("\\r", 0, &[1u8; 4096]).unwrap();
        assert!(e.used_chunks() >= 1);
        e.remove("\\r").unwrap();
        assert_eq!(e.used_chunks(), 0);
    }

    // ---- dedup tests --------------------------------------------------------

    fn chunk_pattern(seed: u8) -> Vec<u8> {
        (0..CHUNK_SIZE as usize).map(|i| (i as u8) ^ seed).collect()
    }

    #[test]
    fn dedup_shares_identical_chunks() {
        let mut e = engine(4, true);
        let block = chunk_pattern(0xAB);
        e.table_mut().create_file("\\f1", 0).unwrap();
        e.table_mut().create_file("\\f2", 0).unwrap();
        e.write("\\f1", 0, &block).unwrap();
        let after_first = e.used_chunks();
        e.write("\\f2", 0, &block).unwrap();
        // Second identical chunk must not consume another physical chunk.
        assert_eq!(
            e.used_chunks(),
            after_first,
            "identical chunk was not deduped"
        );
        // Both files read back the same content.
        assert_eq!(e.read("\\f1", 0, CHUNK_SIZE as usize).unwrap(), block);
        assert_eq!(e.read("\\f2", 0, CHUNK_SIZE as usize).unwrap(), block);
    }

    #[test]
    fn stats_report_dedup_physical_savings() {
        let mut e = engine(4, true);
        let block = chunk_pattern(0x42);
        e.table_mut().create_file("\\a", 0).unwrap();
        e.table_mut().create_file("\\b", 0).unwrap();
        e.write("\\a", 0, &block).unwrap();
        e.write("\\b", 0, &block).unwrap();

        let s = e.stats();
        assert_eq!(s.file_count, 2);
        assert_eq!(s.logical_file_bytes, CHUNK_SIZE * 2);
        assert_eq!(s.raw_logical_chunks, 2);
        assert_eq!(s.raw_unique_chunks, 1);
        assert_eq!(s.dedup_shared_logical_chunks, 1);
        assert_eq!(s.dedup_saved_bytes, CHUNK_SIZE);
        assert_eq!(s.used_physical_bytes, CHUNK_SIZE);
    }

    #[test]
    fn dedup_distinct_chunks_not_shared() {
        let mut e = engine(4, true);
        e.table_mut().create_file("\\a", 0).unwrap();
        e.table_mut().create_file("\\b", 0).unwrap();
        e.write("\\a", 0, &chunk_pattern(1)).unwrap();
        let n1 = e.used_chunks();
        e.write("\\b", 0, &chunk_pattern(2)).unwrap();
        assert_eq!(e.used_chunks(), n1 + 1);
    }

    #[test]
    fn dedup_cow_on_partial_write() {
        let mut e = engine(4, true);
        let block = chunk_pattern(0x5A);
        e.table_mut().create_file("\\x", 0).unwrap();
        e.table_mut().create_file("\\y", 0).unwrap();
        e.write("\\x", 0, &block).unwrap();
        e.write("\\y", 0, &block).unwrap();
        let shared = e.used_chunks();
        // Partially modify y -> must CoW, allocating a new chunk.
        e.write("\\y", 0, b"DIFFERENT").unwrap();
        assert_eq!(e.used_chunks(), shared + 1, "CoW did not allocate");
        // x is untouched; y has the edit.
        assert_eq!(e.read("\\x", 0, CHUNK_SIZE as usize).unwrap(), block);
        let y = e.read("\\y", 0, 9).unwrap();
        assert_eq!(&y, b"DIFFERENT");
    }

    #[test]
    fn clone_range_shares_full_chunks_and_cows() {
        let mut e = engine(4, true);
        e.table_mut().create_file("\\src", 0).unwrap();
        e.table_mut().create_file("\\dst", 0).unwrap();
        let data = vec![7u8; (2 * CHUNK_SIZE) as usize];
        e.write("\\src", 0, &data).unwrap();

        e.clone_range("\\src", "\\dst", 0, 0, 2 * CHUNK_SIZE)
            .unwrap();
        let s = e.stats();
        assert_eq!(s.raw_logical_chunks, 4);
        assert_eq!(s.raw_unique_chunks, 2);
        assert_eq!(e.read("\\dst", 0, data.len()).unwrap(), data);

        e.write("\\dst", 0, &[9]).unwrap();
        assert_eq!(e.read("\\src", 0, 1).unwrap(), vec![7]);
        assert_eq!(e.read("\\dst", 0, 1).unwrap(), vec![9]);
    }

    // ---- compression tests (require nvCOMP: cargo test -- --ignored) --------

    #[test]
    #[ignore]
    fn compress_roundtrip_and_space() {
        let mut e = engine_compress(8);
        e.table_mut().create_file("\\c", 0).unwrap();
        // 256KiB of highly compressible data (4 logical chunks).
        let n = 256 * 1024usize;
        let data: Vec<u8> = (0..n).map(|i| ((i / 1024) % 5) as u8).collect();
        e.write("\\c", 0, &data).unwrap();
        // Compresses well: far fewer physical chunks than the 4 logical ones.
        assert!(
            e.used_chunks() < 4,
            "expected compression, used {}",
            e.used_chunks()
        );
        assert_eq!(e.read("\\c", 0, n).unwrap(), data);

        // Partial overwrite (read-modify-recompress).
        e.write("\\c", 5, b"HELLO").unwrap();
        let mut exp = data.clone();
        exp[5..10].copy_from_slice(b"HELLO");
        assert_eq!(e.read("\\c", 0, n).unwrap(), exp);

        // Truncate then regrow -> tail reads as zeros.
        e.set_size("\\c", 100 * 1024).unwrap();
        e.set_size("\\c", n as u64).unwrap();
        let tail = e.read("\\c", 100 * 1024, n - 100 * 1024).unwrap();
        assert!(tail.iter().all(|&b| b == 0));

        e.remove("\\c").unwrap();
        assert_eq!(e.used_chunks(), 0, "all storage freed");
    }

    #[test]
    #[ignore]
    fn gpu_api_hash_lz4_compressed_chunk() {
        let mut e = engine_compress(8);
        e.table_mut().create_file("\\c", 0).unwrap();
        let data = vec![b'A'; CHUNK_SIZE as usize * 2];
        e.write("\\c", 0, &data).unwrap();
        assert!(e.used_chunks() < 2, "fixture should be stored compressed");
        assert_eq!(
            crate::api_kernel::digest_hex(
                &e.hash_file_gpu("\\c", crate::api_kernel::HashAlgorithm::Md5)
                    .unwrap()
            ),
            "d6011631b1fa3890bcce53ef6fc422fa"
        );
    }

    #[test]
    #[ignore]
    fn compress_incompressible_roundtrip() {
        let mut e = engine_compress(8);
        e.table_mut().create_file("\\r", 0).unwrap();
        // Pseudo-random data won't shrink; stored raw but must round-trip.
        let n = 128 * 1024usize;
        let mut s = 0x9e37_79b9u32;
        let data: Vec<u8> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        e.write("\\r", 0, &data).unwrap();
        assert_eq!(e.read("\\r", 0, n).unwrap(), data);
    }

    // ---- compress + dedup combined tests (require nvCOMP: cargo test -- --ignored) ---

    #[test]
    #[ignore]
    fn compress_dedup_identical_raw_chunks_are_shared() {
        let mut e = engine_compress_dedup(8);
        let block = chunk_pattern(0xCC);
        e.table_mut().create_file("\\f1", 0).unwrap();
        e.table_mut().create_file("\\f2", 0).unwrap();
        e.write("\\f1", 0, &block).unwrap();
        let after_first = e.used_chunks();
        e.write("\\f2", 0, &block).unwrap();
        assert_eq!(
            e.used_chunks(),
            after_first,
            "identical raw fallback chunk was not deduped"
        );
        assert_eq!(e.read("\\f1", 0, CHUNK_SIZE as usize).unwrap(), block);
        assert_eq!(e.read("\\f2", 0, CHUNK_SIZE as usize).unwrap(), block);
    }

    #[test]
    #[ignore]
    fn compress_dedup_identical_compressed_chunks_are_shared() {
        let mut e = engine_compress_dedup(8);
        let block = vec![b'A'; CHUNK_SIZE as usize];
        e.table_mut().create_file("\\f1", 0).unwrap();
        e.table_mut().create_file("\\f2", 0).unwrap();
        e.write("\\f1", 0, &block).unwrap();
        let after_first = e.used_chunks();
        e.write("\\f2", 0, &block).unwrap();
        assert_eq!(
            e.used_chunks(),
            after_first,
            "identical compressed chunk was not deduped"
        );
        let s = e.stats();
        assert_eq!(s.compressed_logical_chunks, 2);
        assert_eq!(s.dedup_shared_logical_chunks, 1);
        assert_eq!(s.dedup_saved_bytes, CHUNK_SIZE);
        assert_eq!(e.read("\\f1", 0, CHUNK_SIZE as usize).unwrap(), block);
        assert_eq!(e.read("\\f2", 0, CHUNK_SIZE as usize).unwrap(), block);
    }

    #[test]
    #[ignore]
    fn compress_dedup_unique_data_compressed() {
        let mut e = engine_compress_dedup(8);
        e.table_mut().create_file("\\c", 0).unwrap();
        // Highly compressible unique data (no dedup match).
        let n = 256 * 1024usize;
        let data: Vec<u8> = (0..n).map(|i| ((i / 1024) % 3) as u8).collect();
        e.write("\\c", 0, &data).unwrap();
        assert!(
            e.used_chunks() < 4,
            "expected compression for unique data, used {}",
            e.used_chunks()
        );
        assert_eq!(e.read("\\c", 0, n).unwrap(), data);
        e.remove("\\c").unwrap();
        assert_eq!(e.used_chunks(), 0);
    }

    #[test]
    #[ignore]
    fn compress_dedup_partial_write_keeps_files_independent() {
        // x and y start by sharing one placement. A partial write to y must
        // materialize a new placement and leave x untouched.
        let mut e = engine_compress_dedup(8);
        let block = chunk_pattern(0xBB);
        e.table_mut().create_file("\\x", 0).unwrap();
        e.table_mut().create_file("\\y", 0).unwrap();
        e.write("\\x", 0, &block).unwrap();
        e.write("\\y", 0, &block).unwrap();
        e.write("\\y", 0, b"PATCHED").unwrap();
        // x is unaffected; y has the edit.
        assert_eq!(e.read("\\x", 0, CHUNK_SIZE as usize).unwrap(), block);
        let y_head = e.read("\\y", 0, 7).unwrap();
        assert_eq!(&y_head, b"PATCHED");
        // ...and the rest of y still matches the original block.
        let y_tail = e.read("\\y", 7, CHUNK_SIZE as usize - 7).unwrap();
        assert_eq!(y_tail, block[7..], "remainder of y corrupted");
    }

    #[test]
    fn dedup_refcount_frees_only_at_zero() {
        let mut e = engine(4, true);
        let block = chunk_pattern(0x33);
        e.table_mut().create_file("\\p", 0).unwrap();
        e.table_mut().create_file("\\q", 0).unwrap();
        e.write("\\p", 0, &block).unwrap();
        e.write("\\q", 0, &block).unwrap();
        let shared = e.used_chunks();
        // Deleting one sharer keeps the physical chunk for the other.
        e.remove("\\p").unwrap();
        assert_eq!(e.used_chunks(), shared, "shared chunk freed too early");
        assert_eq!(e.read("\\q", 0, CHUNK_SIZE as usize).unwrap(), block);
        // Deleting the last sharer frees it.
        e.remove("\\q").unwrap();
        assert_eq!(e.used_chunks(), shared - 1);
    }
}
