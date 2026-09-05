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
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

use crate::api_kernel::{ApiKernel, HashAlgorithm, HashSegment, SEARCH_MAX_PATTERN};
use crate::arena::CompressedAllocator;
use crate::chunk::{ChunkAllocator, ChunkId};
use crate::cuda::Vram;
use crate::gpu_hash::GpuHasher;
use crate::lookup::{Codec, LookupError, LookupTable, Node, Placement};
use crate::nvcomp::{Lz4Codec, NvcompBatchedCodec, NvcompFrameCodec};
use crate::CHUNK_SIZE;
use digest::Digest;
use md5::Md5;
use sha1::Sha1;
use sha2::Sha256;

const ZIP_DEFLATE_CHUNK: u64 = 1024 * 1024;

/// Compressed chunks pulled back to the host at a time for the end-bit walk.
///
/// The GPU compresses [`crate::nvcomp::BATCH`] chunks per launch, but the walk
/// runs on the CPU and so needs those chunks in host memory. Working through a
/// launch in groups bounds that transient buffer — a group of incompressible
/// chunks is a group's worth of megabytes — while still handing every core
/// several chunks per [`walk_deflate_blobs`] call.
const ZIP_DEFLATE_WALK_GROUP: usize = 64;

/// Extra-field id of VRAMDISK's private per-chunk compressed-size table.
///
/// Purely an extraction accelerator: it lets `archive.extract` hand every
/// chunk of a member to nvCOMP as its own decompression job instead of walking
/// one serial stream. Other tools ignore it, as APPNOTE requires of extra
/// fields they do not recognise.
const ZIP_CHUNK_TABLE_TAG: u16 = 0x4754;

/// Extra-field id of the *old* chunk table, from before members were spliced
/// into a single valid DEFLATE stream.
///
/// Still read, never written. The chunks it describes are byte-concatenated
/// with only bit 0 of each one's first byte cleared, so they have to be closed
/// differently — see [`StorageEngine::extract_zip_deflate_chunks`]. Archives
/// carrying this tag are exactly the ones Windows could not open; VRAMDISK
/// keeps reading them so nothing already written becomes unreadable.
const ZIP_CHUNK_TABLE_TAG_LEGACY: u16 = 0x4753;

/// Payload bytes handed to one CRC-32 lane, i.e. to one CUDA thread.
///
/// `vramdisk_crc32_many` runs a scalar, byte-at-a-time table CRC per thread,
/// so aggregate throughput is purely a function of how many lanes are in
/// flight; the lane size only decides how much serial work each thread does.
/// One logical chunk keeps a lane to exactly one [`HashSegment`] for a raw or
/// sparse file — no descriptor amplification — while still being large enough
/// that the per-lane launch bookkeeping disappears next to the scan itself.
const CRC32_LANE_BYTES: u64 = CHUNK_SIZE;

/// Payload bytes covered by one `crc32_many` launch.
///
/// This is the CRC analogue of [`StorageEngine::gpu_hash_launch_budget`], but
/// it is a constant rather than a calibrated value because the two are bounded
/// by different things. The hash budget bounds how long a *single* thread runs,
/// since a single-stream digest cannot be split; here the range is already cut
/// into [`CRC32_LANE_BYTES`] lanes, so wall time per launch is one lane's worth
/// of work no matter how large this is, and what it actually bounds is the
/// descriptor and per-lane host bookkeeping of one launch. 512 MiB is 8192
/// lanes: enough to saturate the device, small enough that the segment and
/// state scratch stay well under a megabyte.
const CRC32_LAUNCH_BYTES: u64 = 512 * 1024 * 1024;

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
pub(crate) const CPU_HASH_WINDOW_BYTES: usize = 32 * 1024 * 1024;

/// GPU hash routing is calibrated by hashing one small VRAM sample *both ways*
/// and comparing what comes back:
///   - launch budget targets ~100 ms of GPU work per kernel, to stay well
///     below the Windows TDR timeout (a driver reset loses the mounted disk)
///   - the CPU routing threshold is where the GPU stops being the faster place
///     to put a single file
///
/// # Why both sides have to be measured
///
/// `vramdisk_hash_update` runs on exactly one CUDA thread, because MD5, SHA-1,
/// SHA-256 and FNV-1a are each a strictly sequential chain over the message:
/// block *n*'s state feeds block *n+1*, so one stream admits no parallelism to
/// spread over the device. A single GPU thread walking that chain reaches
/// roughly 30–60 MB/s, while the CPU path — device-to-host in windows, then
/// RustCrypto with SHA-NI/AVX2 — runs at 0.6–1.6 GB/s. Where the GPU wins is
/// the *batched* path, which gives one thread to each of many files and so
/// scales with file count, not with file size.
///
/// The routing threshold therefore has to answer "is this file small enough to
/// be worth batching", and the earlier one-sided rule — GPU throughput times a
/// fixed 0.25 s, then clamped to at most 128 MiB — could not, because it never
/// looked at what the CPU would have done. On a machine that measures 33 MB/s
/// on the GPU against 1.6 GB/s on the CPU it derived an 8.2 MB threshold and
/// sent every file below that to the ~30x slower path, while the 128 MiB clamp
/// blamed for keeping big files off the GPU never came anywhere near binding.
/// Comparing the two measurements instead removes the clamp entirely: where
/// the GPU digest really is faster (a slow CPU, or an algorithm x86 has no
/// instructions for) files of any size route to the GPU with no ceiling, and
/// where it is not, only files small enough to leave room in a batch for other
/// files go there at all.
const GPU_HASH_CALIBRATION_BYTES: u64 = 1_048_576; // 1 MiB
const GPU_HASH_LAUNCH_TARGET_SECS: f64 = 0.10;
const GPU_HASH_ROUTE_TARGET_SECS: f64 = 0.25;
const GPU_HASH_LAUNCH_BUDGET_MIN_BYTES: u64 = 4 * 1024 * 1024;
const GPU_HASH_LAUNCH_BUDGET_MAX_BYTES: u64 = 512 * 1024 * 1024;
const GPU_HASH_ROUTE_THRESHOLD_MIN_BYTES: u64 = 1_048_576; // 1 MiB
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuHashCalibration {
    pub sample_bytes: u64,
    pub sample_elapsed_secs: f64,
    pub throughput_bytes_per_sec: f64,
    /// What the same sample cost on the CPU route (materialize to host, then
    /// RustCrypto), measured so routing can compare rather than guess.
    pub cpu_sample_elapsed_secs: f64,
    pub cpu_throughput_bytes_per_sec: f64,
    pub launch_budget_bytes: u64,
    pub cpu_route_threshold_bytes: u64,
}

fn clamp_gpu_hash_launch_budget(budget: u64) -> u64 {
    budget.clamp(
        GPU_HASH_LAUNCH_BUDGET_MIN_BYTES,
        GPU_HASH_LAUNCH_BUDGET_MAX_BYTES,
    )
}

fn clamp_hash_cpu_route_threshold(threshold: u64) -> u64 {
    threshold.max(GPU_HASH_ROUTE_THRESHOLD_MIN_BYTES)
}

/// Turn the two measured throughputs into the launch budget and the routing
/// threshold. Split out from the measurement so the policy can be unit tested
/// without a GPU.
fn calibration_from_throughput(
    sample_bytes: u64,
    sample_elapsed_secs: f64,
    throughput_bytes_per_sec: f64,
    cpu_sample_elapsed_secs: f64,
    cpu_throughput_bytes_per_sec: f64,
) -> GpuHashCalibration {
    let launch_budget_bytes = clamp_gpu_hash_launch_budget(
        (throughput_bytes_per_sec * GPU_HASH_LAUNCH_TARGET_SECS) as u64,
    );
    // Per byte, the GPU is at least as fast as the CPU: there is nothing to
    // trade off, so no size is too large for the GPU and files of every size
    // route there. `hash_file_gpu_cancellable` already streams a file as a
    // sequence of launch-budget passes, so an unbounded threshold does not put
    // an unbounded amount of work in any one kernel.
    let cpu_route_threshold_bytes = if throughput_bytes_per_sec >= cpu_throughput_bytes_per_sec {
        u64::MAX
    } else {
        // The GPU only pays off through batching, so admit a file only while
        // it is small enough to share a launch with others: never more than
        // one launch budget, and never more than the 0.25 s of single-thread
        // GPU work that bounds how long one batch can stall its neighbours.
        clamp_hash_cpu_route_threshold(
            ((throughput_bytes_per_sec * GPU_HASH_ROUTE_TARGET_SECS) as u64)
                .min(launch_budget_bytes),
        )
    };
    GpuHashCalibration {
        sample_bytes,
        sample_elapsed_secs,
        throughput_bytes_per_sec,
        cpu_sample_elapsed_secs,
        cpu_throughput_bytes_per_sec,
        launch_budget_bytes,
        cpu_route_threshold_bytes,
    }
}

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

/// Everything `StorageEngine` can fail with.
///
/// The taxonomy exists so the WinFsp layer can pick an `NTSTATUS` and the Jobs
/// API can pick a message without inspecting the error string: the variant
/// alone says whether the caller handed us bad data (`InvalidInput`), asked for
/// something this build deliberately does not do (`Unsupported`), hit a
/// resource wall (`NoSpace` / `OutOfVram`), or tripped over an engine bug
/// (`Internal`). Only genuine driver and kernel failures are `Cuda`.
#[derive(Debug)]
pub enum EngineError {
    /// Namespace lookup failure (missing path, wrong node type, ...).
    Lookup(LookupError),
    /// No free physical chunk: the volume is full.
    NoSpace,
    /// Ran out of VRAM part-way through a job, with actionable advice.
    ///
    /// Distinct from `NoSpace` because a job can exhaust VRAM while staging
    /// temporary buffers even on a volume that is not itself full, and the
    /// remedy the user needs to hear differs.
    OutOfVram(String),
    /// Operation expected a file but found a directory.
    NotAFile,
    /// Cooperative cancellation requested by the caller.
    Cancelled,
    /// Underlying CUDA driver or kernel failure.
    ///
    /// Produced only by [`cuda`], which wraps a real `cuda`/nvCOMP result. A
    /// parse or validation failure must never land here: users read "CUDA
    /// error" as "my GPU is broken", which sends them debugging the wrong
    /// thing.
    Cuda(String),
    /// Caller-supplied data or parameters were malformed.
    ///
    /// Covers archive framing that does not parse, non-ASCII/non-UTF-8 paths,
    /// bad base64/hex, and values that overflow an on-disk field. Retrying
    /// will not help; the input has to change.
    InvalidInput(String),
    /// A well-formed request the engine deliberately cannot serve
    /// (missing nvCOMP, unimplemented placement, ...).
    ///
    /// The input is fine — this build or this mount configuration simply has
    /// no path for it, so the message names the missing capability.
    Unsupported(String),
    /// An engine invariant was violated — internally stored data did not
    /// round-trip. Indicates a bug or memory corruption, not bad input.
    Internal(String),
}

impl From<LookupError> for EngineError {
    fn from(e: LookupError) -> Self {
        EngineError::Lookup(e)
    }
}

impl std::fmt::Display for EngineError {
    /// One clean English line per error, with no variant name and no Rust
    /// quoting, because these strings are surfaced verbatim in `result.json`
    /// and in the GUI. `InvalidInput` and `Unsupported` messages are already
    /// written as complete sentences, so they render bare; the others get the
    /// minimum prefix needed to make the sentence stand on its own.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // `LookupError` lives in a module that has no `Display` of its
            // own, so the human phrasing is spelled out here rather than
            // leaking `NotADirectory`-style Debug output to the user.
            EngineError::Lookup(e) => match e {
                LookupError::NotFound => f.write_str("path not found"),
                LookupError::AlreadyExists => f.write_str("path already exists"),
                LookupError::NotADirectory => f.write_str("path is not a directory"),
                LookupError::IsADirectory => f.write_str("path is a directory"),
                LookupError::NotEmpty => f.write_str("directory is not empty"),
                LookupError::InvalidName => f.write_str("invalid path name"),
            },
            EngineError::NoSpace => f.write_str("no free space left on the VRAM disk"),
            EngineError::OutOfVram(msg) => f.write_str(msg),
            EngineError::NotAFile => f.write_str("path is not a file"),
            EngineError::Cancelled => f.write_str("cancelled"),
            EngineError::Cuda(msg) => write!(f, "CUDA error: {msg}"),
            EngineError::InvalidInput(msg) => f.write_str(msg),
            EngineError::Unsupported(msg) => f.write_str(msg),
            EngineError::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub type EResult<T> = Result<T, EngineError>;

fn cuda<T>(r: anyhow::Result<T>) -> EResult<T> {
    r.map_err(|e| EngineError::Cuda(format!("{e:#}")))
}

fn cancelled<T>() -> EResult<T> {
    Err(EngineError::Cancelled)
}

fn archive_vram_exhausted(detail: &str) -> EngineError {
    EngineError::OutOfVram(format!(
        "archive job needs more free VRAM to {detail}; free space, use a larger mount, or retry on an uncompressed mount"
    ))
}

fn map_archive_job_result<T>(result: EResult<T>) -> EResult<T> {
    result.map_err(|err| match err {
        EngineError::NoSpace => {
            archive_vram_exhausted("materialize compressed data or write temporary archive buffers")
        }
        other => other,
    })
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

/// Public, plain-`u64` snapshot of the engine's activity counters.
///
/// This is what `$VRAMDISK\trace.json` and `trace.txt` render (see
/// [`crate::internal_api`]); it is a value copied out of the live counters by
/// [`StorageEngine::trace_snapshot`], never the counters themselves. Keeping it
/// plain means consumers can compare, clone and arithmetic on it freely — the
/// live side's atomicity is an implementation detail of the private
/// `TraceCounters` this is copied out of.
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
    /// Candidates that the hash index offered but confirmation turned down, so
    /// the chunk was stored on its own instead of being shared. Under the
    /// default byte verification a non-zero value means an actual FNV-1a
    /// collision was caught before it could corrupt a file; under
    /// `--dedup-trust-hash` it only counts stale index entries.
    pub dedup_rejected_chunks: u64,
    pub dedup_unique_chunks: u64,
    pub gpu_hash_chunks: u64,
}

/// Declares the live counter struct that mirrors [`EngineTrace`] field for
/// field, plus its snapshot/reset.
///
/// The two structs are kept in step by the compiler rather than by hand: the
/// generated `snapshot` builds an `EngineTrace` struct literal, so a field
/// present in one and missing from the other fails to compile.
macro_rules! trace_counters {
    ($($field:ident),+ $(,)?) => {
        /// Live activity counters, held inside [`StorageEngine`].
        ///
        /// These are `AtomicU64` rather than plain `u64` because the shared
        /// read fast path ([`StorageEngine::read_into_shared`]) runs under an
        /// `RwLock` *read* guard: several threads are inside the engine at once
        /// with only `&self`, and they still have to be counted. Counting only
        /// the exclusive path would make `$VRAMDISK\trace.json` under-report
        /// every concurrent read, i.e. lie.
        ///
        /// `Ordering::Relaxed` is the right ordering throughout: these are pure
        /// statistics that publish no other memory and order nothing else. The
        /// only guarantee needed is that no increment is lost, which relaxed
        /// read-modify-write already gives.
        #[derive(Debug, Default)]
        struct TraceCounters {
            $($field: AtomicU64,)+
        }

        impl TraceCounters {
            /// Copy the counters into the plain-`u64` public struct.
            ///
            /// The loads are not atomic *as a group*, so a snapshot taken
            /// while reads are in flight can straddle an in-progress update
            /// (e.g. `read_calls` already incremented but `raw_read_bytes` not
            /// yet). That was equally true of the old non-atomic field reads
            /// under a shared guard, and these are monotonic statistics, so a
            /// momentarily skewed sample is harmless.
            fn snapshot(&self) -> EngineTrace {
                EngineTrace {
                    $($field: self.$field.load(Ordering::Relaxed),)+
                }
            }

            /// Zero every counter. Takes `&self` only because atomics allow it;
            /// the sole caller ([`StorageEngine::reset_trace`]) holds `&mut`.
            fn reset(&self) {
                $(self.$field.store(0, Ordering::Relaxed);)+
            }
        }
    };
}

trace_counters!(
    read_calls,
    write_calls,
    logical_read_bytes,
    logical_write_bytes,
    raw_read_ops,
    raw_read_bytes,
    raw_write_ops,
    raw_write_bytes,
    compressed_read_chunks,
    compressed_read_requested_bytes,
    compressed_read_full_bytes,
    compress_batches,
    compress_chunks,
    compress_raw_fallback_chunks,
    dedup_hash_chunks,
    dedup_candidate_chunks,
    dedup_shared_chunks,
    dedup_rejected_chunks,
    dedup_unique_chunks,
    gpu_hash_chunks,
);

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

#[derive(Debug, Default)]
struct ArchiveMaterializedSegments {
    segs: Vec<HashSegment>,
    temp_chunks: Vec<ChunkId>,
}

/// One `crc32_many` launch, described lane by lane.
///
/// `segments[i]` is what lane `i` reads and `lengths[i]` how many payload
/// bytes that adds up to. The lengths are kept alongside because folding the
/// per-lane checksums back into one needs each lane's length, and it is not
/// recoverable from the segment list once sparse holes and materialized
/// compressed chunks are in it.
#[derive(Default)]
struct Crc32Lanes {
    segments: Vec<Vec<HashSegment>>,
    lengths: Vec<u64>,
}

/// Upper bound for one encode staging pass (input side). Big enough to
/// amortise launches, small enough to fit comfortably next to the user's
/// data. Must be a multiple of 12 (lcm of all transcoding group sizes).
const ENCODE_STAGE_BYTES: u64 = 48 * 1024 * 1024;

/// Window the search kernel scans per launch.
///
/// Files are stored as 64 KiB chunks that need not be adjacent and may be
/// compressed, so a scan materializes a contiguous raw window first. Bigger
/// windows amortize the launch and the staging setup; this is capped again at
/// run time against free space, since the window is real VRAM.
const SEARCH_STAGE_BYTES: u64 = 64 * 1024 * 1024;

/// Transcoding codec for GPU encode/decode jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeCodec {
    Base64,
    Hex,
}

impl EncodeCodec {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "base64" | "b64" => Some(Self::Base64),
            "hex" | "base16" => Some(Self::Hex),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Base64 => "base64",
            Self::Hex => "hex",
        }
    }
}

/// Direction of a GPU encode/decode job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeDirection {
    Encode,
    Decode,
}

impl EncodeDirection {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "encode" | "enc" => Some(Self::Encode),
            "decode" | "dec" => Some(Self::Decode),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Encode => "encode",
            Self::Decode => "decode",
        }
    }
}

/// Everything one search pass needs beyond the file list.
///
/// Grouped rather than passed positionally because the scan loop already takes
/// the engine, the paths, their sizes and a progress callback; six more loose
/// arguments is where a transposed pair stops being a compile error.
struct SearchPlan<'a> {
    needle: &'a [u8],
    ignore_case: bool,
    max_offsets_per_file: usize,
    /// Contiguous raw scratch file each window is materialized into.
    staging: &'a str,
    /// Size of that scratch file, and so of one scan window.
    window: u64,
    /// Total bytes across every file, for progress reporting.
    total_bytes: u64,
}

/// One file that matched a search, and where.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub path: String,
    /// Every match in the file, even if `offsets` holds fewer.
    pub matches: u64,
    /// Ascending byte offsets, truncated to the caller's limit.
    pub offsets: Vec<u64>,
    /// Set when `offsets` is a subset of the matches found.
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct SearchJobStats {
    pub pattern_len: usize,
    pub ignore_case: bool,
    pub files_scanned: u64,
    pub bytes_scanned: u64,
    pub files_matched: u64,
    pub total_matches: u64,
    pub hits: Vec<SearchHit>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncodeJobStats {
    pub codec: String,
    pub direction: String,
    pub input: String,
    pub output: String,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub elapsed_ms: u128,
}

/// Decode the final 4-character Base64 group, which is the only place `=`
/// padding is legal. Returns the 1–3 decoded bytes.
fn decode_base64_quad(quad: &[u8]) -> EResult<Vec<u8>> {
    if quad.len() != 4 {
        return Err(EngineError::InvalidInput("truncated base64 group".into()));
    }
    fn val(c: u8) -> EResult<u32> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a') as u32 + 26),
            b'0'..=b'9' => Ok((c - b'0') as u32 + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(EngineError::InvalidInput(format!(
                "invalid base64 character 0x{c:02x}"
            ))),
        }
    }
    let pads = match (quad[2], quad[3]) {
        (b'=', b'=') => 2,
        (b'=', _) => {
            return Err(EngineError::InvalidInput(
                "invalid base64 padding: '=' may only end the data".into(),
            ))
        }
        (_, b'=') => 1,
        _ => 0,
    };
    let v0 = val(quad[0])?;
    let v1 = val(quad[1])?;
    let v2 = if pads >= 2 { 0 } else { val(quad[2])? };
    let v3 = if pads >= 1 { 0 } else { val(quad[3])? };
    let word = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
    let mut out = vec![(word >> 16) as u8];
    if pads < 2 {
        out.push((word >> 8) as u8);
    }
    if pads < 1 {
        out.push(word as u8);
    }
    Ok(out)
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
    // Phase 1: hash each 256-byte segment independently.
    let mut seg_hashes = [0u64; 256];
    for t in 0..256 {
        let mut h = FNV1A64_OFFSET_BASIS;
        for &b in &data[t * 256..(t + 1) * 256] {
            h ^= b as u64;
            h = h.wrapping_mul(FNV1A64_PRIME);
        }
        seg_hashes[t] = h;
    }

    // Phase 2: FNV-1a over the segment hashes (8 LE bytes each).
    let mut h = FNV1A64_OFFSET_BASIS;
    for sh in seg_hashes {
        for byte_idx in 0..8u64 {
            h ^= (sh >> (byte_idx * 8)) & 0xff;
            h = h.wrapping_mul(FNV1A64_PRIME);
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

pub(crate) enum CpuHashState {
    Md5(Md5),
    Sha1(Sha1),
    Sha256(Sha256),
    Fnv1a64(u64),
}

impl CpuHashState {
    pub(crate) fn new(alg: HashAlgorithm) -> Self {
        match alg {
            HashAlgorithm::Md5 => Self::Md5(Md5::new()),
            HashAlgorithm::Sha1 => Self::Sha1(Sha1::new()),
            HashAlgorithm::Sha256 => Self::Sha256(Sha256::new()),
            HashAlgorithm::Fnv1a64 => Self::Fnv1a64(FNV1A64_OFFSET_BASIS),
        }
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        match self {
            Self::Md5(hasher) => hasher.update(bytes),
            Self::Sha1(hasher) => hasher.update(bytes),
            Self::Sha256(hasher) => hasher.update(bytes),
            Self::Fnv1a64(state) => {
                for &b in bytes {
                    *state ^= b as u64;
                    *state = state.wrapping_mul(FNV1A64_PRIME);
                }
            }
        }
    }

    pub(crate) fn finalize(self) -> Vec<u8> {
        match self {
            Self::Md5(hasher) => hasher.finalize().to_vec(),
            Self::Sha1(hasher) => hasher.finalize().to_vec(),
            Self::Sha256(hasher) => hasher.finalize().to_vec(),
            Self::Fnv1a64(state) => state.to_be_bytes().to_vec(),
        }
    }
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
    /// Confirm every dedup hit by comparing the candidate's stored bytes with
    /// the bytes being written, instead of trusting the FNV-1a hash.
    ///
    /// The index is *keyed* by a two-level FNV-1a 64-bit hash, which is the
    /// right structure — it turns "find an identical chunk" into one hash-map
    /// probe. But FNV-1a is not collision resistant and collisions are cheap
    /// to construct on purpose, so the hash can only ever be a *candidate*
    /// filter. Sharing on the hash alone lets anyone who can write chosen
    /// bytes to the volume make an unrelated file's chunk alias theirs, which
    /// is silent data corruption of somebody else's data. Verification is
    /// therefore the default: a wrong answer here is undetectable and
    /// unrecoverable, while the cost is bounded and measurable (one 64 KiB
    /// device-to-host read per *confirmed duplicate*, nothing at all on
    /// unique data).
    ///
    /// Set to `false` (`--dedup-trust-hash`) to restore the pre-verification
    /// behaviour, where a candidate is confirmed by re-hashing it on the GPU.
    /// That is only safe when every writer to the volume is trusted.
    dedup_verify_bytes: bool,
    /// Reusable 64 KiB landing buffer for the byte-verification read-back.
    /// A batched write confirms many candidates in a row, so this is allocated
    /// once and reused rather than per candidate.
    verify_scratch: Vec<u8>,

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
    gpu_hash_launch_budget: u64,
    hash_cpu_route_threshold: u64,
    gpu_hash_launch_budget_override: bool,
    hash_cpu_route_threshold_override: bool,
    gpu_hash_calibration: Option<GpuHashCalibration>,
    /// Payload covered by one parallel CRC-32 launch; see
    /// [`CRC32_LAUNCH_BYTES`]. Held as a field only so tests can shrink it and
    /// exercise the multi-launch fold without staging a half-gigabyte file.
    crc32_launch_bytes: u64,
    /// Bytes materialized and scanned per search launch. A field only so tests
    /// can shrink it and exercise the window seam without writing a file
    /// larger than [`SEARCH_STAGE_BYTES`].
    search_window_bytes: u64,
    trace: TraceCounters,
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
        let engine = StorageEngine {
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
            dedup_verify_bytes: true,
            verify_scratch: Vec::new(),
            carena: CompressedAllocator::new(),
            codec,
            vram_base,
            gpu_hasher,
            api_kernel: None,
            gpu_hash_launch_budget: GPU_HASH_LAUNCH_BUDGET_MIN_BYTES,
            hash_cpu_route_threshold: GPU_HASH_ROUTE_THRESHOLD_MIN_BYTES,
            gpu_hash_launch_budget_override: false,
            hash_cpu_route_threshold_override: false,
            gpu_hash_calibration: None,
            crc32_launch_bytes: CRC32_LAUNCH_BYTES,
            search_window_bytes: SEARCH_STAGE_BYTES,
            trace: TraceCounters::default(),
        };
        engine.warm_gpu_api();
        Ok(engine)
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

    /// O(1) volume totals for `GetDiskFreeSpaceEx`-style callers. Explorer
    /// polls volume info continuously during copies, so this must not walk the
    /// namespace the way [`stats`](Self::stats) does.
    pub fn volume_usage(&self) -> (u64, u64) {
        let total = self.total_chunks() as u64 * CHUNK_SIZE;
        let free = (self.total_chunks() - self.used_chunks()) as u64 * CHUNK_SIZE;
        (total, free)
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
        self.trace.snapshot()
    }

    pub fn gpu_hash_launch_budget(&self) -> u64 {
        self.gpu_hash_launch_budget
    }

    pub fn hash_cpu_route_threshold(&self) -> u64 {
        self.hash_cpu_route_threshold
    }

    pub fn gpu_hash_calibration(&self) -> Option<GpuHashCalibration> {
        self.gpu_hash_calibration
    }

    pub fn set_gpu_hash_launch_budget(&mut self, budget: u64) {
        self.gpu_hash_launch_budget = clamp_gpu_hash_launch_budget(budget);
        self.gpu_hash_launch_budget_override = true;
    }

    pub fn set_hash_cpu_route_threshold(&mut self, threshold: u64) {
        self.hash_cpu_route_threshold = clamp_hash_cpu_route_threshold(threshold);
        self.hash_cpu_route_threshold_override = true;
    }

    /// Override how much payload one parallel CRC-32 launch covers.
    ///
    /// The default ([`CRC32_LAUNCH_BYTES`]) is deliberately larger than any
    /// realistic test fixture, so this exists to let a test drive the
    /// multi-launch fold — where per-launch lane checksums have to be
    /// concatenated across launches, not just within one — on a small file.
    pub fn set_crc32_launch_bytes(&mut self, bytes: u64) {
        self.crc32_launch_bytes = bytes.max(CRC32_LANE_BYTES);
    }

    /// Shrink the search window. Rounded up to a whole chunk, because the
    /// window is a contiguous raw scratch file.
    pub fn set_search_window_bytes(&mut self, bytes: u64) {
        self.search_window_bytes = bytes.max(CHUNK_SIZE).div_ceil(CHUNK_SIZE) * CHUNK_SIZE;
    }

    /// Whether dedup confirms a hash hit by comparing bytes. See
    /// [`set_dedup_verify_bytes`](Self::set_dedup_verify_bytes).
    pub fn dedup_verify_bytes(&self) -> bool {
        self.dedup_verify_bytes
    }

    /// Choose how a dedup candidate is confirmed before two logical chunks are
    /// made to share one physical chunk.
    ///
    /// `true` (the default) compares the candidate's stored bytes against the
    /// bytes being written. `false` trusts the two-level FNV-1a 64-bit hash the
    /// index is keyed by, confirming only that the candidate still hashes to
    /// the same value.
    ///
    /// The default is verification because FNV-1a is not collision resistant:
    /// collisions can be constructed cheaply and deliberately, so hash-only
    /// matching lets an attacker who can write chosen bytes to the volume
    /// alias an unrelated file's chunk onto their own — silent, unrecoverable
    /// corruption of data the attacker never had to be able to write. Trusting
    /// the hash buys back one 64 KiB device-to-host read per confirmed
    /// duplicate and is only appropriate when every writer is trusted.
    pub fn set_dedup_verify_bytes(&mut self, verify: bool) {
        self.dedup_verify_bytes = verify;
    }

    pub fn file_size(&self, path: &str) -> EResult<u64> {
        let node = self.table.get(path).ok_or(LookupError::NotFound)?;
        if node.is_dir {
            return Err(EngineError::NotAFile);
        }
        Ok(node.size)
    }

    pub fn should_hash_on_gpu_routed(&mut self, path: &str) -> EResult<bool> {
        self.ensure_hash_calibration()?;
        self.should_hash_on_gpu(path)
    }

    #[allow(dead_code)]
    pub fn reset_trace(&mut self) {
        self.trace.reset();
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

    /// Hash-only confirmation of a raw dedup candidate: re-hash the stored
    /// chunk on the GPU and compare with the CPU hash of the incoming data.
    /// Both use the same two-level FNV-1a algorithm, so identical content
    /// produces identical values.
    ///
    /// This is the `--dedup-trust-hash` path only. It cannot separate
    /// identical content from an FNV-1a collision — it re-derives the very
    /// value that selected the candidate — so it answers "is this still the
    /// chunk the index says it is", not "is this the same data".
    fn verify_chunk(&mut self, c: ChunkId, expected_hash: u64) -> EResult<bool> {
        let hasher = self
            .gpu_hasher
            .as_mut()
            .expect("gpu_hasher present when dedup");
        let gpu_hash = cuda(hasher.hash_chunk(self.vram_base, c as u64 * CHUNK_SIZE))?;
        Ok(gpu_hash == expected_hash)
    }

    /// Hash-only confirmation of any placement. See [`verify_chunk`](Self::verify_chunk)
    /// for why this is not a safety check.
    fn verify_placement(&mut self, p: Placement, expected_hash: u64) -> EResult<bool> {
        match p {
            Placement::Raw { chunk } => self.verify_chunk(chunk, expected_hash),
            Placement::Compressed { offset, .. } => {
                Ok(self.compressed_hash.get(&offset) == Some(&expected_hash))
            }
        }
    }

    /// Exact confirmation of a dedup candidate: compare the bytes that are
    /// really stored behind `p` against the 64 KiB the caller is about to
    /// write. This is the only check that can tell identical content apart
    /// from an FNV-1a collision, which is why it is the default.
    ///
    /// A raw candidate costs one 64 KiB device-to-host read into the reusable
    /// [`verify_scratch`](Self::verify_scratch) buffer; a compressed candidate
    /// costs a decompression of its blob. Both are paid only for chunks the
    /// index already matched, i.e. once per *confirmed duplicate*, never on
    /// unique data.
    ///
    /// Errors are genuine device or codec failures and propagate, exactly as
    /// the hash-only path's kernel launch does; a plain content mismatch is
    /// `Ok(false)` and makes the caller store the chunk normally.
    fn candidate_matches_bytes(&mut self, p: Placement, incoming: &[u8]) -> EResult<bool> {
        debug_assert_eq!(incoming.len(), CHUNK_SIZE as usize);
        match p {
            Placement::Raw { chunk } => {
                if self.verify_scratch.len() != CHUNK_SIZE as usize {
                    self.verify_scratch.resize(CHUNK_SIZE as usize, 0);
                }
                cuda(
                    self.vram
                        .read_at(chunk as u64 * CHUNK_SIZE, &mut self.verify_scratch),
                )?;
                Ok(self.verify_scratch.as_slice() == incoming)
            }
            Placement::Compressed { offset, len, codec } => {
                let full = self.decompress_blob(offset, len, codec)?;
                Ok(full.as_slice() == incoming)
            }
        }
    }

    /// Confirm a single dedup candidate the way the current mode demands:
    /// byte comparison by default, hash re-check under `--dedup-trust-hash`.
    fn confirm_candidate(&mut self, p: Placement, h: u64, incoming: &[u8]) -> EResult<bool> {
        if self.dedup_verify_bytes {
            self.candidate_matches_bytes(p, incoming)
        } else {
            self.verify_placement(p, h)
        }
    }

    /// Make logical chunk `lc` of `path` point at the already-stored placement
    /// `p`, taking a reference on it and releasing whatever the chunk held
    /// before. Callers must have confirmed that `p` really holds the intended
    /// content.
    fn share_placement(&mut self, path: &str, lc: usize, p: Placement) {
        self.ref_inc_placement(p);
        if let Some(old) = self.coord(path, lc) {
            self.free_placement(old);
        }
        self.set_coord(path, lc, Some(p));
    }

    /// Single-chunk dedup attempt: look the content hash up, confirm the
    /// candidate really holds `incoming`, and share it if so.
    ///
    /// `incoming` is the full 64 KiB the caller is about to store; every caller
    /// already has it on the host, which is what makes exact confirmation
    /// affordable.
    fn try_share_hashed(
        &mut self,
        path: &str,
        lc: usize,
        h: u64,
        incoming: &[u8],
    ) -> EResult<bool> {
        let Some(cand) = self.hash_index.get(&h).copied() else {
            return Ok(false);
        };
        if !self.confirm_candidate(cand, h, incoming)? {
            self.trace
                .dedup_rejected_chunks
                .fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        self.share_placement(path, lc, cand);
        Ok(true)
    }

    /// `--dedup-trust-hash` pre-pass for a batched write: confirm every
    /// candidate of the batch up front, re-hashing all raw candidates in a
    /// single GPU launch. `Some(ok)` is indexed like `hashes`.
    ///
    /// Returns `None` in the default byte-verifying mode, where confirmation
    /// happens one chunk at a time inside the placement loop instead — that is
    /// both where the incoming bytes are addressable and the only point at
    /// which the index is guaranteed still to describe the candidate.
    fn batch_trusted_candidates(&mut self, hashes: &[u64]) -> EResult<Option<Vec<bool>>> {
        if self.dedup_verify_bytes {
            return Ok(None);
        }
        let mut ok = vec![false; hashes.len()];
        let mut offsets = Vec::new();
        let mut items = Vec::new();
        for (i, h) in hashes.iter().enumerate() {
            match self.hash_index.get(h) {
                Some(Placement::Raw { chunk }) => {
                    offsets.push(*chunk as u64 * CHUNK_SIZE);
                    items.push(i);
                }
                Some(Placement::Compressed { offset, .. }) => {
                    ok[i] = self.compressed_hash.get(offset) == Some(h);
                }
                None => {}
            }
        }
        if !offsets.is_empty() {
            let mut out = vec![0u64; offsets.len()];
            let base = self.vram_base;
            let hasher = self
                .gpu_hasher
                .as_mut()
                .expect("gpu_hasher present when dedup");
            cuda(hasher.hash_chunks(base, &offsets, &mut out))?;
            self.trace
                .gpu_hash_chunks
                .fetch_add(out.len() as u64, Ordering::Relaxed);
            for (slot, &i) in items.iter().enumerate() {
                ok[i] = out[slot] == hashes[i];
            }
        }
        Ok(Some(ok))
    }

    /// Batched dedup attempt for chunk `i` of a write whose per-chunk content
    /// hashes are `hashes` and whose chunk `i` holds `incoming`.
    ///
    /// The index is re-read here rather than from a snapshot taken before the
    /// batch started placing chunks. While a batch runs, entries are only ever
    /// *removed* from the index (a chunk rewritten in place, or released when
    /// its last reference went away) and never replaced, so a fresh lookup can
    /// only be the snapshot's placement or nothing at all. Taking it fresh is
    /// what makes the byte comparison meaningful: a candidate the index still
    /// vouches for cannot be a chunk this same batch has already overwritten
    /// or freed, so the bytes read back are the bytes the index promised.
    fn try_share_batched(
        &mut self,
        path: &str,
        lc: usize,
        i: usize,
        hashes: &[u64],
        incoming: &[u8],
        trusted: Option<&Vec<bool>>,
    ) -> EResult<bool> {
        let Some(cand) = self.hash_index.get(&hashes[i]).copied() else {
            return Ok(false);
        };
        self.trace
            .dedup_candidate_chunks
            .fetch_add(1, Ordering::Relaxed);
        let ok = match trusted {
            Some(t) => t[i],
            None => self.candidate_matches_bytes(cand, incoming)?,
        };
        if !ok {
            self.trace
                .dedup_rejected_chunks
                .fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        self.share_placement(path, lc, cand);
        self.trace
            .dedup_shared_chunks
            .fetch_add(1, Ordering::Relaxed);
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
                    } else {
                        // Every compressed placement created under dedup seeds
                        // its refcount via compressed_index_insert; a missing
                        // entry means the invariant broke somewhere and
                        // freeing the blob could double-free a shared region.
                        debug_assert!(
                            false,
                            "compressed placement at {offset} missing refcount under dedup"
                        );
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
        let full = match codec {
            Codec::Lz4 => {
                let lz4 = self.codec.as_mut().ok_or_else(|| {
                    EngineError::Unsupported("LZ4 chunk without nvCOMP codec".into())
                })?;
                cuda(lz4.decompress(&buf, CHUNK_SIZE as usize))?
            }
            Codec::Zstd => zstd::decode_all(buf.as_slice())
                .map_err(|e| EngineError::Internal(e.to_string()))?,
        };
        // Consumers slice/overwrite the result as a full chunk; a short or
        // oversized decode would panic or spill into a neighbouring chunk.
        if full.len() != CHUNK_SIZE as usize {
            return Err(EngineError::Internal(format!(
                "compressed blob decoded to {} bytes, expected {CHUNK_SIZE}",
                full.len()
            )));
        }
        Ok(full)
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

    /// Start compiling the API kernels so the first job does not have to wait
    /// for NVRTC.
    ///
    /// Building the first [`ApiKernel`] in a process compiles the CUDA source
    /// with NVRTC, which takes about 5.5 s on a desktop; everything after that
    /// — loading the module, taking the calibration samples — is a couple of
    /// hundred milliseconds. Before this, that 5.5 s landed on whichever hash,
    /// archive or encode job happened to be first, which simply looked like a
    /// hung job.
    ///
    /// It runs on its own thread rather than inline in [`StorageEngine::new`]
    /// because doing it inline moved the stall rather than removing it: the
    /// drive letter did not appear for 5.4 s (measured 0.2 s before, 5.6 s
    /// after), and a mount that takes six seconds to show up is a worse
    /// symptom than a first job that takes six seconds to finish. NVRTC needs
    /// no CUDA context, so a plain detached thread is enough; the compiled PTX
    /// is process-global, and a job that arrives before the thread finishes
    /// simply waits for it rather than compiling a second copy.
    fn warm_gpu_api(&self) {
        thread::spawn(crate::api_kernel::precompile);
    }

    fn ensure_hash_calibration(&mut self) -> EResult<()> {
        if self.gpu_hash_calibration.is_some() {
            return Ok(());
        }
        let vram_base = self.vram_base;
        let vram_size = self.vram.size();
        let sample_bytes = GPU_HASH_CALIBRATION_BYTES.min(vram_size).max(1);
        let gpu_elapsed = {
            let kernel = self.api_kernel()?;
            Self::measure_gpu_hash_calibration(kernel, vram_base, sample_bytes)?
        };
        // Time the CPU route over the same bytes, through the same
        // materialize-then-digest path a CPU-routed file would take, so the
        // comparison in `calibration_from_throughput` is like for like.
        let mut host = vec![0u8; sample_bytes as usize];
        let started = Instant::now();
        cuda(self.vram.read_at(0, &mut host))?;
        let mut hasher = Sha256::new();
        hasher.update(&host);
        let _ = hasher.finalize();
        let cpu_elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
        let calibration = calibration_from_throughput(
            sample_bytes,
            gpu_elapsed,
            sample_bytes as f64 / gpu_elapsed,
            cpu_elapsed,
            sample_bytes as f64 / cpu_elapsed,
        );
        if !self.gpu_hash_launch_budget_override {
            self.gpu_hash_launch_budget = calibration.launch_budget_bytes;
        }
        if !self.hash_cpu_route_threshold_override {
            self.hash_cpu_route_threshold = calibration.cpu_route_threshold_bytes;
        }
        self.gpu_hash_calibration = Some(calibration);
        Ok(())
    }

    /// Time a single-thread GPU SHA-256 over `sample_bytes` of device memory at
    /// `vram_base`, returning the elapsed seconds. Launch overhead is inside
    /// the measurement on purpose: routing pays it too.
    fn measure_gpu_hash_calibration(
        kernel: &mut ApiKernel,
        vram_base: u64,
        sample_bytes: u64,
    ) -> EResult<f64> {
        let started = Instant::now();
        cuda(kernel.begin(HashAlgorithm::Sha256))?;
        cuda(kernel.update(&[HashSegment {
            ptr: vram_base,
            len: u32::try_from(sample_bytes).unwrap_or(u32::MAX),
            kind: 0,
        }]))?;
        let _ = cuda(kernel.finish(HashAlgorithm::Sha256))?;
        Ok(started.elapsed().as_secs_f64().max(f64::EPSILON))
    }

    fn file_has_only_raw_sparse(&self, path: &str) -> EResult<bool> {
        let node = self.table.get(path).ok_or(LookupError::NotFound)?;
        if node.is_dir {
            return Err(EngineError::NotAFile);
        }
        Ok(node
            .coords
            .iter()
            .all(|placement| matches!(placement, None | Some(Placement::Raw { .. }))))
    }

    /// Whether `[offset, offset + len)` is backed only by raw chunks and
    /// sparse holes, so a GPU pass over it needs no materialization scratch.
    ///
    /// Scoped to the range rather than the whole file because the gzip reader
    /// verifies a 512 MiB output one 64 KiB member at a time: a whole-file scan
    /// there would be quadratic in the file size, while this is proportional to
    /// the range actually being read.
    fn range_has_only_raw_sparse(&self, path: &str, offset: u64, len: u64) -> EResult<bool> {
        let node = self.table.get(path).ok_or(LookupError::NotFound)?;
        if node.is_dir {
            return Err(EngineError::NotAFile);
        }
        if len == 0 {
            return Ok(true);
        }
        let first = (offset / CHUNK_SIZE) as usize;
        let last = ((offset + len - 1) / CHUNK_SIZE) as usize;
        Ok(node
            .coords
            .iter()
            .take(last + 1)
            .skip(first)
            .all(|placement| matches!(placement, None | Some(Placement::Raw { .. }))))
    }

    fn file_supports_gpu_hash(&self, path: &str) -> EResult<bool> {
        let node = self.table.get(path).ok_or(LookupError::NotFound)?;
        if node.is_dir {
            return Err(EngineError::NotAFile);
        }
        Ok(node.coords.iter().all(|placement| {
            matches!(
                placement,
                None | Some(Placement::Raw { .. })
                    | Some(Placement::Compressed {
                        codec: Codec::Lz4,
                        ..
                    })
            )
        }))
    }

    fn should_hash_on_gpu(&self, path: &str) -> EResult<bool> {
        let node = self.table.get(path).ok_or(LookupError::NotFound)?;
        if node.is_dir {
            return Err(EngineError::NotAFile);
        }
        Ok(node.size <= self.hash_cpu_route_threshold && self.file_supports_gpu_hash(path)?)
    }

    pub fn hash_file_cpu(&mut self, path: &str, alg: HashAlgorithm) -> EResult<Vec<u8>> {
        let (size, is_dir) = {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            (node.size, node.is_dir)
        };
        if is_dir {
            return Err(EngineError::NotAFile);
        }
        let mut hasher = CpuHashState::new(alg);
        let mut offset = 0u64;
        while offset < size {
            let take = ((size - offset).min(CPU_HASH_WINDOW_BYTES as u64)) as usize;
            let chunk = self.read(path, offset, take)?;
            hasher.update(&chunk);
            offset += chunk.len() as u64;
        }
        Ok(hasher.finalize())
    }

    pub fn hash_file(&mut self, path: &str, alg: HashAlgorithm) -> EResult<Vec<u8>> {
        self.ensure_hash_calibration()?;
        if self.should_hash_on_gpu(path)? {
            self.hash_file_gpu(path, alg)
        } else {
            self.hash_file_cpu(path, alg)
        }
    }

    /// Hash a file using CUDA-resident data only. Raw chunks are read in place
    /// from the VRAM buffer; LZ4-compressed chunks are decompressed by nvCOMP
    /// into device scratch and immediately consumed by the API hash kernel.
    /// Sparse holes are represented as zero segments and generated on GPU.
    ///
    /// CPU zstd fallback chunks cannot satisfy the GPU-only contract and are
    /// rejected instead of silently pulling file bytes through host memory.
    pub fn hash_file_gpu(&mut self, path: &str, alg: HashAlgorithm) -> EResult<Vec<u8>> {
        self.hash_file_gpu_cancellable(path, alg, |_, _| false)
    }

    /// Hash one file on the GPU, reporting progress and honouring cancellation.
    ///
    /// # The `progress` callback
    ///
    /// Every long-running engine job takes the same callback, and this is the
    /// canonical description of its contract; the other `*_cancellable`
    /// entry points refer back here.
    ///
    /// The engine calls `progress(done_bytes, total_bytes)` at each of its
    /// safe points — the places where it can abandon the job without leaving
    /// VRAM or the namespace inconsistent — meaning *"I have completed
    /// `done_bytes` of an estimated `total_bytes`; return `true` if I should
    /// stop"*. Returning `true` makes the job unwind at that point and fail
    /// with [`EngineError::Cancelled`] after cleaning up its staging temps;
    /// returning `false` lets it continue.
    ///
    /// The counters exist because these jobs routinely move tens of gigabytes:
    /// a caller that only learns "still running" cannot tell the user whether
    /// to keep waiting, so every poll carries the most meaningful byte counts
    /// available at that point in the code.
    ///
    /// Guarantees the engine makes, which callers may rely on:
    ///
    /// - `done_bytes` never goes backwards within a single call.
    /// - `total_bytes` is a *best estimate*, not a promise. It is whatever the
    ///   job can cheaply know when it ticks, it may be revised (usually
    ///   upward) as the job learns more, and the unit it counts is the one
    ///   that makes the ratio meaningful for that job — staged tar bytes,
    ///   archive bytes consumed, payload bytes transcoded — not necessarily
    ///   the size of any one file. Consequently `done_bytes` can momentarily
    ///   exceed a total that has just been revised, so treat the ratio as
    ///   advisory rather than clamped to 1.
    /// - `(0, 0)` is the documented "no useful counter here" signal, used at
    ///   pre-flight checks that run before any measurable work. Callers that
    ///   store the counters should ignore such a tick rather than let it wipe
    ///   out a meaningful total reported earlier.
    ///
    /// The callback is invoked from the engine thread while the engine lock is
    /// held, so it must be cheap and must not re-enter the engine.
    pub fn hash_file_gpu_cancellable<F>(
        &mut self,
        path: &str,
        alg: HashAlgorithm,
        mut progress: F,
    ) -> EResult<Vec<u8>>
    where
        F: FnMut(u64, u64) -> bool,
    {
        self.ensure_hash_calibration()?;
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

        let budget = self.gpu_hash_launch_budget;
        let raw_seg_limit = budget.div_ceil(CHUNK_SIZE) as usize;
        let comp_seg_limit = raw_seg_limit.min(crate::nvcomp::BATCH);
        let mut pos = 0u64;
        while pos < size {
            // `pos` is exactly how many of the file's bytes have been fed to
            // the hash kernel, so the file size is an exact total here.
            if progress(pos, size) {
                return cancelled();
            }
            let lc = (pos / CHUNK_SIZE) as usize;
            match self.coord(path, lc) {
                None | Some(Placement::Raw { .. }) => {
                    let mut segs = Vec::with_capacity(raw_seg_limit);
                    let mut used = 0u64;
                    while pos < size && segs.len() < raw_seg_limit && used < budget {
                        let lc = (pos / CHUNK_SIZE) as usize;
                        let in_off = pos % CHUNK_SIZE;
                        let take = (size - pos).min(CHUNK_SIZE - in_off).min(budget - used) as u32;
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
                            Some(Placement::Compressed { .. }) => break,
                        }
                        pos += take as u64;
                        used += take as u64;
                    }
                    if !segs.is_empty() {
                        self.update_api_hash(&segs)?;
                    }
                }
                Some(Placement::Compressed {
                    codec: Codec::Lz4, ..
                }) => {
                    let mut blobs = Vec::with_capacity(comp_seg_limit);
                    let mut parts = Vec::with_capacity(comp_seg_limit);
                    let mut used = 0u64;
                    while pos < size && blobs.len() < comp_seg_limit && used < budget {
                        let lc = (pos / CHUNK_SIZE) as usize;
                        let in_off = pos % CHUNK_SIZE;
                        let take = (size - pos).min(CHUNK_SIZE - in_off).min(budget - used) as u32;
                        match self.coord(path, lc) {
                            Some(Placement::Compressed {
                                offset,
                                len,
                                codec: Codec::Lz4,
                            }) => {
                                blobs.push((offset, len));
                                parts.push((in_off, take));
                                pos += take as u64;
                                used += take as u64;
                            }
                            Some(Placement::Compressed {
                                codec: Codec::Zstd, ..
                            }) => {
                                return Err(EngineError::Unsupported(
                                    "GPU-only hash is unavailable for CPU zstd fallback chunks"
                                        .into(),
                                ));
                            }
                            _ => break,
                        }
                    }

                    let vram_base = self.vram_base;
                    let codec = self.codec.as_mut().ok_or_else(|| {
                        EngineError::Unsupported("LZ4 chunk without nvCOMP codec".into())
                    })?;
                    cuda(codec.decompress_from_arena_dev(vram_base, &blobs))?;

                    let mut segs = Vec::with_capacity(parts.len());
                    for (i, (in_off, take)) in parts.into_iter().enumerate() {
                        segs.push(HashSegment {
                            ptr: codec.uncomp_slot_ptr(i) + in_off,
                            len: take,
                            kind: 0,
                        });
                    }
                    self.update_api_hash(&segs)?;
                }
                Some(Placement::Compressed {
                    codec: Codec::Zstd, ..
                }) => {
                    return Err(EngineError::Unsupported(
                        "GPU-only hash is unavailable for CPU zstd fallback chunks".into(),
                    ));
                }
            }
        }

        cuda(self.api_kernel()?.finish(alg))
    }

    pub fn hash_files_gpu_many(
        &mut self,
        paths: &[String],
        alg: HashAlgorithm,
    ) -> EResult<Vec<Vec<u8>>> {
        self.hash_files_gpu_many_cancellable(paths, alg, |_, _| false)
    }

    /// Hash many files in one batched GPU pass.
    ///
    /// `progress` follows the contract documented on
    /// [`StorageEngine::hash_file_gpu_cancellable`]. The counters here span
    /// the *whole* set: the total is the summed size of every requested file
    /// and `done_bytes` accumulates across both phases — first the files that
    /// must be hashed one at a time because they are not purely raw/sparse,
    /// then the batched round-robin below — so the caller sees one bar for the
    /// set instead of one that restarts per file.
    pub fn hash_files_gpu_many_cancellable<F>(
        &mut self,
        paths: &[String],
        alg: HashAlgorithm,
        mut progress: F,
    ) -> EResult<Vec<Vec<u8>>>
    where
        F: FnMut(u64, u64) -> bool,
    {
        self.ensure_hash_calibration()?;
        // Total the set up front so the very first tick already carries a real
        // denominator. Sizes come from the in-memory lookup table, so this pass
        // is cheap; a path that is missing (or is a directory) contributes 0
        // here and is rejected with the proper error by the loop below.
        let total_bytes: u64 = paths
            .iter()
            .map(|path| self.table.get(path).map(|node| node.size).unwrap_or(0))
            .sum();
        let mut done_bytes = 0u64;
        let mut out = vec![Vec::new(); paths.len()];
        let mut batch_paths = Vec::new();
        let mut batch_sizes = Vec::new();
        let mut batch_indices = Vec::new();
        for (idx, path) in paths.iter().enumerate() {
            let (size, is_dir) = {
                let node = self.table.get(path).ok_or(LookupError::NotFound)?;
                (node.size, node.is_dir)
            };
            if is_dir {
                return Err(EngineError::NotAFile);
            }
            if self.file_has_only_raw_sparse(path)? {
                batch_indices.push(idx);
                batch_paths.push(path.clone());
                batch_sizes.push(size);
            } else {
                // Shift the single-file hash's own counters into the set's
                // frame: it reports 0..size for this file, which continues the
                // set's running total instead of restarting the bar.
                let done_before = done_bytes;
                out[idx] = self.hash_file_gpu_cancellable(path, alg, |file_done, _| {
                    progress(done_before.saturating_add(file_done), total_bytes)
                })?;
                done_bytes = done_bytes.saturating_add(size);
            }
        }

        if batch_paths.is_empty() {
            return Ok(out);
        }

        let budget = self.gpu_hash_launch_budget;
        let nfiles = batch_paths.len();
        let mut offsets = vec![0u64; nfiles];
        cuda(self.api_kernel()?.begin_many(alg, nfiles))?;
        while offsets
            .iter()
            .zip(batch_sizes.iter())
            .any(|(&offset, &size)| offset < size)
        {
            if progress(done_bytes, total_bytes) {
                return cancelled();
            }
            let mut round = Vec::with_capacity(nfiles);
            let mut round_bytes = 0u64;
            let mut progressed = false;
            for ((path, &size), offset) in batch_paths
                .iter()
                .zip(batch_sizes.iter())
                .zip(offsets.iter_mut())
            {
                let take = size.saturating_sub(*offset).min(budget);
                if take == 0 {
                    round.push(Vec::new());
                    continue;
                }
                round.push(self.file_segments_raw(path, *offset, take)?);
                *offset += take;
                round_bytes += take;
                progressed = true;
            }
            if !progressed {
                break;
            }
            cuda(self.api_kernel()?.update_many(&round))?;
            // Only credit the round once its kernel has actually run, so the
            // reported count never runs ahead of the work.
            done_bytes = done_bytes.saturating_add(round_bytes);
        }

        let digests = cuda(self.api_kernel()?.finish_many(alg, nfiles))?;
        for (idx, digest) in batch_indices.into_iter().zip(digests.into_iter()) {
            out[idx] = digest;
        }
        Ok(out)
    }

    pub fn hash_files_many(
        &mut self,
        paths: &[String],
        alg: HashAlgorithm,
    ) -> EResult<Vec<Vec<u8>>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        self.ensure_hash_calibration()?;

        let mut out = vec![Vec::new(); paths.len()];
        let mut gpu_paths = Vec::new();
        let mut gpu_indices = Vec::new();
        let mut cpu_paths = Vec::new();
        let mut cpu_indices = Vec::new();

        for (idx, path) in paths.iter().enumerate() {
            if self.should_hash_on_gpu(path)? {
                gpu_indices.push(idx);
                gpu_paths.push(path.clone());
            } else {
                cpu_indices.push(idx);
                cpu_paths.push(path.clone());
            }
        }

        if !gpu_paths.is_empty() {
            let digests = self.hash_files_gpu_many(&gpu_paths, alg)?;
            for (idx, digest) in gpu_indices.into_iter().zip(digests.into_iter()) {
                out[idx] = digest;
            }
        }

        for (idx, path) in cpu_indices.into_iter().zip(cpu_paths.iter()) {
            out[idx] = self.hash_file_cpu(path, alg)?;
        }

        Ok(out)
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
                self.trace.compress_batches.fetch_add(1, Ordering::Relaxed);
                self.trace.compress_chunks.fetch_add(1, Ordering::Relaxed);
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
                self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
                self.trace
                    .raw_write_bytes
                    .fetch_add(CHUNK_SIZE, Ordering::Relaxed);
                if self.compress {
                    self.trace
                        .compress_raw_fallback_chunks
                        .fetch_add(1, Ordering::Relaxed);
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

    /// Rename/move `from` to `to`, freeing the VRAM placements of any file
    /// that was replaced at the destination. Going through the lookup table
    /// directly would leak the replaced file's chunks (the classic editor
    /// save pattern — write temp file, rename over the original — would then
    /// leak the original's entire content on every save).
    pub fn rename(&mut self, from: &str, to: &str, replace: bool) -> EResult<()> {
        let replaced = self.table.rename(from, to, replace)?;
        if let Some(node) = replaced {
            self.free_coords(&node.coords);
        }
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

    /// Length in bytes of the maximal run of *physically contiguous* `Raw`
    /// chunks that starts at logical position `pos`.
    ///
    /// This is what turns a large sequential read into one big device-to-host
    /// transfer instead of one 64 KiB transfer per logical chunk, and it is the
    /// single biggest factor in read throughput — so both read paths
    /// ([`read_into`](Self::read_into) and
    /// [`read_into_shared`](Self::read_into_shared)) call this rather than
    /// carrying two copies of the walk that could drift apart.
    ///
    /// `first_take` is the byte count already claimed for the chunk containing
    /// `pos` (which may start mid-chunk), and `remaining` is how many bytes of
    /// the request are still unserved. The run only ever extends across chunk
    /// boundaries, so a mid-chunk start simply stops the walk unless the first
    /// span happens to end exactly on a boundary.
    ///
    /// Takes `&self`, so it is usable from a shared (`RwLock` read) guard.
    fn raw_run_bytes(
        &self,
        path: &str,
        pos: u64,
        first_take: usize,
        remaining: usize,
        first_chunk: ChunkId,
    ) -> usize {
        let mut run = first_take;
        let mut prev_chunk = first_chunk;
        while run < remaining {
            let next_pos = pos + run as u64;
            if next_pos % CHUNK_SIZE != 0 {
                break;
            }
            let next_lc = (next_pos / CHUNK_SIZE) as usize;
            match self.coord(path, next_lc) {
                Some(Placement::Raw { chunk }) if chunk == prev_chunk + 1 => {
                    run += (CHUNK_SIZE as usize).min(remaining - run);
                    prev_chunk = chunk;
                }
                _ => break,
            }
        }
        run
    }

    /// Shared-guard fast path for [`read_into`](Self::read_into): serves a read
    /// through `&self` when — and only when — nothing about it needs exclusive
    /// access, so concurrent readers of the volume genuinely run in parallel.
    ///
    /// Returns `Ok(Some(n))` when the request was served in full (`n` is the
    /// same count `read_into` would return), and `Ok(None)` when the caller
    /// must retry under the exclusive guard. On `Ok(None)` **`buf` has not been
    /// touched**: the decision is made by a pre-scan of the placement array
    /// before a single byte is transferred or zeroed, so a fallback can never
    /// see a half-filled buffer and no caller can mistake a bail-out for a
    /// short read.
    ///
    /// # Why this is sound
    ///
    /// Everything the raw/sparse path touches is either immutable behind the
    /// shared guard or already internally synchronised:
    ///
    /// * the namespace ([`LookupTable`]) and each node's `coords` are read-only
    ///   while any shared guard is held, because every mutation of them goes
    ///   through a `&mut self` method and therefore the exclusive guard;
    /// * [`Vram::read_at`] takes `&self` and is thread-safe on its own — its
    ///   pinned staging buffers live behind an internal `Mutex` and the
    ///   host-registered path fans out over the per-`Vram` transfer streams,
    ///   each copy reading a disjoint device range into the caller's own
    ///   buffer;
    /// * the trace counters are [`AtomicU64`] (see the private `TraceCounters`),
    ///   so the read is still counted exactly as the exclusive path counts it.
    ///
    /// # Bail-out conditions (`Ok(None)`)
    ///
    /// The pre-scan walks every logical chunk the request touches and gives up
    /// on anything that is not `Placement::Raw` or a sparse hole (`None`):
    ///
    /// * **`Placement::Compressed { codec: Codec::Lz4, .. }`** — decompression
    ///   goes through [`Lz4Codec`], whose nvCOMP scratch and device buffers are
    ///   mutated per call (`self.codec.as_mut()`), i.e. exclusive by nature.
    /// * **`Placement::Compressed { codec: Codec::Zstd, .. }`** — the CPU
    ///   fallback runs through `decompress_blob`, which is `&mut self` for the
    ///   same reason.
    ///
    /// Error cases (`NotFound`, `NotAFile`) and the trivially empty cases are
    /// resolved here rather than deferred, because they cost nothing and are
    /// identical under either guard.
    ///
    /// **If a new [`Placement`] variant or a new per-read side effect is added,
    /// it must be added to the bail-out list above unless it is provably safe
    /// under `&self`.** The pre-scan matches variants exhaustively precisely so
    /// that a new variant is a compile error here, not a silent data race: keep
    /// it that way — do not add a catch-all arm.
    pub fn read_into_shared(
        &self,
        path: &str,
        offset: u64,
        buf: &mut [u8],
    ) -> EResult<Option<usize>> {
        let node = self.table.get(path).ok_or(LookupError::NotFound)?;
        if node.is_dir {
            return Err(EngineError::NotAFile);
        }
        let size = node.size;
        if offset >= size || buf.is_empty() {
            return Ok(Some(0));
        }
        let n = ((size - offset).min(buf.len() as u64)) as usize;

        // Pre-scan: decide before writing anything. `coords` may be shorter
        // than the file's logical chunk count (a sparse tail), and a missing
        // entry means the same as `None` — a hole — exactly as `coord` treats
        // it.
        let first_lc = (offset / CHUNK_SIZE) as usize;
        let last_lc = ((offset + n as u64 - 1) / CHUNK_SIZE) as usize;
        for lc in first_lc..=last_lc {
            match node.coords.get(lc).copied().flatten() {
                None | Some(Placement::Raw { .. }) => {}
                Some(Placement::Compressed { .. }) => return Ok(None),
            }
        }

        let out = &mut buf[..n];
        self.trace.read_calls.fetch_add(1, Ordering::Relaxed);
        self.trace
            .logical_read_bytes
            .fetch_add(n as u64, Ordering::Relaxed);

        let mut done = 0usize;
        let mut pos = offset;
        while done < n {
            let lc = (pos / CHUNK_SIZE) as usize;
            let in_off = pos % CHUNK_SIZE;
            let take = ((CHUNK_SIZE - in_off) as usize).min(n - done);
            match node.coords.get(lc).copied().flatten() {
                None => {
                    // Sparse hole: the caller's buffer is not pre-zeroed.
                    out[done..done + take].fill(0);
                    done += take;
                    pos += take as u64;
                }
                Some(Placement::Raw { chunk }) => {
                    let phys = chunk as u64 * CHUNK_SIZE + in_off;
                    let run_take = self.raw_run_bytes(path, pos, take, n - done, chunk);
                    self.trace.raw_read_ops.fetch_add(1, Ordering::Relaxed);
                    self.trace
                        .raw_read_bytes
                        .fetch_add(run_take as u64, Ordering::Relaxed);
                    cuda(self.vram.read_at(phys, &mut out[done..done + run_take]))?;
                    done += run_take;
                    pos += run_take as u64;
                }
                // Unreachable: the pre-scan already returned `Ok(None)` for
                // every compressed placement, and nothing can change the
                // placement array while this shared borrow is alive. It is
                // still handled explicitly (rather than with a catch-all) so
                // that adding a `Placement` variant breaks the build in both
                // matches. An error — not `Ok(None)` — because bytes have
                // already been written into `buf` by this point, and `Ok(None)`
                // promises the opposite.
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Internal(format!(
                        "shared read reached a compressed placement at logical chunk {lc} of \
                         {path} after the pre-scan accepted it"
                    )))
                }
            }
        }
        Ok(Some(n))
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
        self.trace.read_calls.fetch_add(1, Ordering::Relaxed);
        self.trace
            .logical_read_bytes
            .fetch_add(n as u64, Ordering::Relaxed);

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
                    let run_take = self.raw_run_bytes(path, pos, take, n - done, chunk);
                    self.trace.raw_read_ops.fetch_add(1, Ordering::Relaxed);
                    self.trace
                        .raw_read_bytes
                        .fetch_add(run_take as u64, Ordering::Relaxed);
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
                    self.trace
                        .compressed_read_chunks
                        .fetch_add(1, Ordering::Relaxed);
                    self.trace
                        .compressed_read_requested_bytes
                        .fetch_add(take as u64, Ordering::Relaxed);
                    self.trace
                        .compressed_read_full_bytes
                        .fetch_add(CHUNK_SIZE, Ordering::Relaxed);
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
            if pieces.len() != pending.len() {
                return Err(EngineError::Internal(format!(
                    "decompress returned {} pieces for {} requests",
                    pieces.len(),
                    pending.len()
                )));
            }
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
        self.trace.write_calls.fetch_add(1, Ordering::Relaxed);
        self.trace
            .logical_write_bytes
            .fetch_add(data.len() as u64, Ordering::Relaxed);

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
        // Guard the destination end against u64 overflow: offsets arrive
        // straight from a user-issued FSCTL_DUPLICATE_EXTENTS_TO_FILE.
        let dst_end = dst_offset.checked_add(n).ok_or(EngineError::NoSpace)?;
        if !self.dedup {
            let data = self.read(src_path, src_offset, n as usize)?;
            return self.write(dst_path, dst_offset, &data);
        }
        if src_path.eq_ignore_ascii_case(dst_path)
            && ranges_overlap(src_offset, src_offset + n, dst_offset, dst_end)
        {
            let data = self.read(src_path, src_offset, n as usize)?;
            return self.write(dst_path, dst_offset, &data);
        }

        self.ensure_logical_len(dst_path, dst_end)?;
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
            self.trace
                .dedup_shared_chunks
                .fetch_add(full_chunks as u64, Ordering::Relaxed);
        }

        if done < n {
            let take = n - done;
            let data = self.read(src_path, src_offset + done, take as usize)?;
            self.write(dst_path, dst_offset + done, &data)?;
        }

        let node = self.table.get_mut(dst_path).unwrap();
        node.size = node.size.max(dst_end);
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
        self.trace
            .dedup_hash_chunks
            .fetch_add(n as u64, Ordering::Relaxed);

        // `--dedup-trust-hash` only: confirm the whole batch up front with one
        // GPU re-hash launch. The default byte-verifying mode confirms inside
        // the loop below instead, where the incoming bytes are in hand.
        let trusted = self.batch_trusted_candidates(&hashes)?;

        let mut indexed_writes = Vec::new();
        let mut wrote = false;
        for i in 0..n {
            let lc = lc0 + i;
            let s = i * CHUNK_SIZE as usize;
            let incoming = &data[s..s + CHUNK_SIZE as usize];
            if self.try_share_batched(path, lc, i, &hashes, incoming, trusted.as_ref())? {
                continue;
            }
            let old = self.coord(path, lc);

            let chunk = match old {
                Some(Placement::Raw { chunk }) if self.refcount[chunk as usize] == 1 => {
                    self.index_remove(chunk);
                    chunk
                }
                _ => self.alloc_chunk()?,
            };
            cuda(
                self.vram
                    .write_at_async(chunk as u64 * CHUNK_SIZE, incoming),
            )?;
            self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
            self.trace
                .raw_write_bytes
                .fetch_add(CHUNK_SIZE, Ordering::Relaxed);
            wrote = true;
            if !matches!(old, Some(Placement::Raw { chunk: c }) if c == chunk) {
                if let Some(old) = old {
                    self.free_placement(old);
                }
            }
            self.set_coord(path, lc, Some(Placement::Raw { chunk }));
            indexed_writes.push((chunk, hashes[i]));
            self.trace
                .dedup_unique_chunks
                .fetch_add(1, Ordering::Relaxed);
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
            self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
            self.trace
                .raw_write_bytes
                .fetch_add(bytes as u64, Ordering::Relaxed);
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
        self.trace
            .gpu_hash_chunks
            .fetch_add(hashes.len() as u64, Ordering::Relaxed);
        self.trace
            .dedup_unique_chunks
            .fetch_add(hashes.len() as u64, Ordering::Relaxed);
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
                    self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
                    self.trace
                        .raw_write_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
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
                    self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
                    self.trace
                        .raw_write_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    j += count;
                }
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Internal(
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

    /// Allocate up to `count` contiguous chunks, halving the request on
    /// failure (O(log count) bitmap scans instead of the old decrement-by-one
    /// retry, which was O(count) full scans on a fragmented volume). When
    /// dedup is enabled the refcount slot of every returned chunk is
    /// initialised to 1, matching `alloc_chunk` — `release_chunk` relies on it.
    fn alloc_contiguous_run(&mut self, count: u32) -> EResult<(ChunkId, u32)> {
        let mut n = count;
        loop {
            if let Some(start) = self.alloc.alloc_contiguous(n) {
                if self.dedup {
                    for c in start..start + n {
                        self.refcount[c as usize] = 1;
                    }
                }
                return Ok((start, n));
            }
            if n == 1 {
                return Err(EngineError::NoSpace);
            }
            n = n.div_ceil(2);
        }
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
                    self.trace
                        .compress_raw_fallback_chunks
                        .fetch_add(1, Ordering::Relaxed);
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
            self.trace.compress_batches.fetch_add(1, Ordering::Relaxed);
            self.trace
                .compress_chunks
                .fetch_add(m as u64, Ordering::Relaxed);

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
                        self.trace
                            .compress_raw_fallback_chunks
                            .fetch_add(1, Ordering::Relaxed);
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
            self.trace
                .dedup_hash_chunks
                .fetch_add(m as u64, Ordering::Relaxed);

            // `--dedup-trust-hash` only: one batched GPU re-hash confirms the
            // whole group. Byte verification instead compares each candidate's
            // stored bytes with `group` inside the loop below.
            let trusted = self.batch_trusted_candidates(&hashes)?;

            let mut misses = Vec::new();
            let mut miss_buf = Vec::new();
            for i in 0..m {
                let lc = lc0 + j + i;
                let s = i * cs as usize;
                let incoming = &group[s..s + cs as usize];
                if !self.try_share_batched(path, lc, i, &hashes, incoming, trusted.as_ref())? {
                    misses.push((i, lc, self.coord(path, lc), hashes[i]));
                    miss_buf.extend_from_slice(incoming);
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
                    self.trace
                        .compress_raw_fallback_chunks
                        .fetch_add(1, Ordering::Relaxed);
                    self.trace
                        .dedup_unique_chunks
                        .fetch_add(1, Ordering::Relaxed);
                }
                // `miss_buf` is dropped at the end of this iteration, but the
                // fallback writes above were enqueued async *from* it — flush
                // them before the buffer goes away.
                cuda(self.vram.sync())?;
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
            self.trace.compress_batches.fetch_add(1, Ordering::Relaxed);
            self.trace
                .compress_chunks
                .fetch_add(miss_count as u64, Ordering::Relaxed);

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
                        self.trace
                            .compress_raw_fallback_chunks
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                self.trace
                    .dedup_unique_chunks
                    .fetch_add(1, Ordering::Relaxed);
            }
            // Flush the arena copies and any raw-fallback writes enqueued from
            // this iteration's `miss_buf` before it is dropped/reused.
            cuda(self.vram.sync())?;
            j += m;
        }

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
        self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
        self.trace
            .raw_write_bytes
            .fetch_add(CHUNK_SIZE, Ordering::Relaxed);
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
            if self.try_share_hashed(path, lc, h, sub)? {
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
                if self.try_share_hashed(path, lc, h, &whole)? {
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
                    return Err(EngineError::Unsupported(
                        "compressed write not implemented".into(),
                    ));
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
            self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
            self.trace
                .raw_write_bytes
                .fetch_add(sub.len() as u64, Ordering::Relaxed);
            if fresh {
                self.set_coord(path, lc, Some(Placement::Raw { chunk }));
            }
            return Ok(());
        }

        // Dedup + partial write: copy-on-write, then modify in place. The
        // chunk is left out of the dedup index until it is next fully written.
        let chunk = self.make_exclusive(path, lc)?;
        cuda(self.vram.write_at(chunk as u64 * CHUNK_SIZE + in_off, sub))?;
        self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
        self.trace
            .raw_write_bytes
            .fetch_add(sub.len() as u64, Ordering::Relaxed);
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
                            self.try_share_hashed(path, last, h, &whole)?
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
        self.archive_compress_gpu_cancellable(format, files, output, |_, _| false)
    }

    /// Build an archive from `files` on the GPU.
    ///
    /// `progress` follows the contract documented on
    /// [`StorageEngine::hash_file_gpu_cancellable`]. For the tar-based formats
    /// the counters measure the *staged tar image*: `total_bytes` is the tar
    /// length computed while planning (headers, payloads, padding and the
    /// 1024-byte trailer) and `done_bytes` is how much of it has been written
    /// into the staging file. The codec pass that follows is a single nvCOMP
    /// launch with no interior safe point, so the counters park at the staged
    /// total while it runs. The ZIP path counts differently — see
    /// `archive_zip_compress_gpu`.
    pub fn archive_compress_gpu_cancellable<F>(
        &mut self,
        format: NvcompFrameCodec,
        files: &[String],
        output: &str,
        mut progress: F,
    ) -> EResult<ArchiveJobStats>
    where
        F: FnMut(u64, u64) -> bool,
    {
        let mut output_created = false;
        let result = map_archive_job_result((|| {
            let start = Instant::now();
            let output = crate::lookup::normalize(output);
            if format == NvcompFrameCodec::Deflate {
                let stats = self.archive_zip_compress_gpu(
                    files,
                    &output,
                    start,
                    &mut output_created,
                    &mut progress,
                )?;
                return Ok(stats);
            }
            let tmp = format!(
                "\\.__vramdisk_archive_tmp_{}",
                crate::lookup::now_filetime()
            );
            self.create_or_truncate_file(&tmp)?;
            let mut planned = Vec::with_capacity(files.len());
            let mut tar_total = 1024u64;
            for file in files {
                // Planning only reads metadata: no payload has been staged and
                // `tar_total` is still being accumulated, so a denominator
                // taken from it would grow under the caller's feet. `(0, 0)`
                // is the documented "nothing meaningful to report" tick.
                if progress(0, 0) {
                    let _ = self.remove(&tmp);
                    return cancelled();
                }
                let path = crate::lookup::normalize(file);
                let (size, is_dir) = {
                    let node = self.table.get(&path).ok_or(LookupError::NotFound)?;
                    (node.size, node.is_dir)
                };
                if is_dir {
                    return Err(EngineError::NotAFile);
                }
                // Build (and thereby validate) the tar header now, before the
                // whole tar buffer is reserved: a single over-long or
                // non-ASCII name must fail the job while it is still cheap.
                let name = path.trim_start_matches('\\').replace('\\', "/");
                let header = tar_header(&name, size)?;
                tar_total += 512 + size + pad512(size);
                planned.push((path, size, header));
            }
            self.allocate_raw_file(&tmp, tar_total)?;
            let mut tar_pos = 0u64;
            let mut input_bytes = 0u64;
            for (path, size, header) in planned {
                // `tar_pos` and `tar_total` are the same unit — bytes of the
                // tar image, headers and padding included — so this ratio is
                // exact rather than an estimate.
                if progress(tar_pos, tar_total) {
                    let _ = self.remove(&tmp);
                    return cancelled();
                }
                self.write_raw_internal(&tmp, tar_pos, &header)?;
                tar_pos += 512;
                self.copy_file_payload_raw(&path, 0, &tmp, tar_pos, size)?;
                tar_pos += size;
                input_bytes += size;
                let pad = pad512(size);
                if pad != 0 {
                    self.write_raw_internal(&tmp, tar_pos, &vec![0u8; pad as usize])?;
                    tar_pos += pad;
                }
            }
            self.write_raw_internal(&tmp, tar_pos, &[0u8; 1024])?;
            tar_pos += 1024;

            // The tar image is complete here (`tar_pos == tar_total`); the
            // codec pass below has no interior safe point, so this is the last
            // chance to stop before it.
            if progress(tar_pos, tar_total) {
                let _ = self.remove(&tmp);
                return cancelled();
            }
            self.create_or_truncate_file(&output)?;
            output_created = true;
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
        })());
        if result.is_err() {
            // A failed (or cancelled) job must not leave a half-written
            // archive at the requested output path, nor leak staging temps.
            if output_created {
                let _ = self.remove(&crate::lookup::normalize(output));
            }
            self.cleanup_archive_temp_files();
        }
        result
    }

    pub fn archive_extract_gpu(
        &mut self,
        format: NvcompFrameCodec,
        archive: &str,
        output_dir: &str,
    ) -> EResult<ArchiveExtractStats> {
        self.archive_extract_gpu_cancellable(format, archive, output_dir, |_, _| false)
    }

    /// Extract an archive on the GPU.
    ///
    /// `progress` follows the contract documented on
    /// [`StorageEngine::hash_file_gpu_cancellable`]. For the tar-based formats
    /// the counters measure the *decompressed tar image*: the archive is
    /// inflated into a staging file first (one nvCOMP launch, no interior safe
    /// point, so the counters stay at zero for it) and the per-member loop
    /// then walks that image, reporting its offset against the tar length. The
    /// ZIP path instead walks the archive itself — see
    /// `archive_zip_extract_gpu`.
    pub fn archive_extract_gpu_cancellable<F>(
        &mut self,
        format: NvcompFrameCodec,
        archive: &str,
        output_dir: &str,
        mut progress: F,
    ) -> EResult<ArchiveExtractStats>
    where
        F: FnMut(u64, u64) -> bool,
    {
        let result = map_archive_job_result((|| {
            let start = Instant::now();
            let archive = crate::lookup::normalize(archive);
            if format == NvcompFrameCodec::Deflate {
                return self.archive_zip_extract_gpu(&archive, output_dir, start, &mut progress);
            }
            let archive_size = {
                let node = self.table.get(&archive).ok_or(LookupError::NotFound)?;
                if node.is_dir {
                    return Err(EngineError::NotAFile);
                }
                node.size
            };
            let tmp = format!(
                "\\.__vramdisk_extract_tmp_{}",
                crate::lookup::now_filetime()
            );
            self.create_or_truncate_file(&tmp)?;
            let (tar_size, packed_archive) = if format == NvcompFrameCodec::Gzip {
                let tar_size = self.extract_gzip_deflate_members(&archive, archive_size, &tmp)?;
                (tar_size, None)
            } else if format == NvcompFrameCodec::Lz4 {
                let tar_size = self.extract_lz4_frame(&archive, archive_size, &tmp)?;
                (tar_size, None)
            } else {
                let packed_archive = format!(
                    "\\.__vramdisk_extract_src_{}",
                    crate::lookup::now_filetime()
                );
                self.create_or_truncate_file(&packed_archive)?;
                self.allocate_raw_file(&packed_archive, archive_size)?;
                self.copy_file_payload_raw(&archive, 0, &packed_archive, 0, archive_size)?;
                let archive_ptr = self.contiguous_file_ptr(&packed_archive, archive_size)?;
                let mut codec = cuda(NvcompBatchedCodec::load(&self.vram, format))?;
                let tar_size =
                    cuda(codec.decompress_sizes_device(&[archive_ptr], &[archive_size]))?[0];
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
                // `pos` is the offset of the member about to be extracted
                // within the decompressed tar, so it and `tar_size` share a
                // unit and the ratio is exact.
                if progress(pos, tar_size) {
                    let _ = self.remove(&tmp);
                    if let Some(packed_archive) = packed_archive.as_ref() {
                        let _ = self.remove(packed_archive);
                    }
                    return cancelled();
                }
                let hdr = self.read(&tmp, pos, 512)?;
                if hdr.len() < 512 {
                    return Err(EngineError::InvalidInput("truncated tar header".into()));
                }
                if hdr.iter().all(|&b| b == 0) {
                    break;
                }
                let name_end = hdr[..100].iter().position(|&b| b == 0).unwrap_or(100);
                let name = std::str::from_utf8(&hdr[..name_end]).map_err(|e| {
                    EngineError::InvalidInput(format!("invalid tar path UTF-8: {e}"))
                })?;
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
        })());
        if result.is_err() {
            // Partially extracted output files are left in place (useful for
            // diagnosing a bad archive), but staging temps must not leak.
            self.cleanup_archive_temp_files();
        }
        result
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

    fn write_raw_internal(&mut self, path: &str, offset: u64, data: &[u8]) -> EResult<u64> {
        if data.is_empty() {
            return Ok(0);
        }
        {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            if node.is_dir {
                return Err(EngineError::NotAFile);
            }
        }
        let end = offset
            .checked_add(data.len() as u64)
            .ok_or(EngineError::NoSpace)?;
        self.ensure_logical_len(path, end)?;
        self.trace.write_calls.fetch_add(1, Ordering::Relaxed);
        self.trace
            .logical_write_bytes
            .fetch_add(data.len() as u64, Ordering::Relaxed);

        let mut done = 0usize;
        let mut pos = offset;
        while done < data.len() {
            let lc = (pos / CHUNK_SIZE) as usize;
            let in_off = pos % CHUNK_SIZE;
            let take = ((CHUNK_SIZE - in_off) as usize).min(data.len() - done);
            self.write_chunk_raw_internal(path, lc, in_off, &data[done..done + take])?;
            done += take;
            pos += take as u64;
        }

        let node = self.table.get_mut(path).unwrap();
        node.size = node.size.max(end);
        let now = crate::lookup::now_filetime();
        node.modified = now;
        node.changed = now;
        Ok(data.len() as u64)
    }

    fn write_chunk_raw_internal(
        &mut self,
        path: &str,
        lc: usize,
        in_off: u64,
        sub: &[u8],
    ) -> EResult<()> {
        let full = in_off == 0 && sub.len() as u64 == CHUNK_SIZE;
        let existing = self.coord(path, lc);
        let (chunk, fresh) = match existing {
            Some(Placement::Raw { chunk }) => (chunk, false),
            Some(Placement::Compressed { .. }) => {
                return Err(EngineError::Internal(
                    "archive internal raw stream unexpectedly contains compressed placement".into(),
                ));
            }
            None if self.dedup => (self.alloc_chunk()?, true),
            None => (self.alloc.alloc_one().ok_or(EngineError::NoSpace)?, true),
        };
        let base = chunk as u64 * CHUNK_SIZE;
        if fresh && !full {
            cuda(self.vram.zero_at_async(base, CHUNK_SIZE))?;
            cuda(self.vram.write_at_async(base + in_off, sub))?;
            cuda(self.vram.sync())?;
        } else {
            cuda(self.vram.write_at(base + in_off, sub))?;
        }
        self.trace.raw_write_ops.fetch_add(1, Ordering::Relaxed);
        self.trace
            .raw_write_bytes
            .fetch_add(sub.len() as u64, Ordering::Relaxed);
        if fresh {
            self.set_coord(path, lc, Some(Placement::Raw { chunk }));
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
            let mut take = (len - done).min(CHUNK_SIZE - src_in);
            let src_ptr = match self.coord(src_path, src_lc) {
                Some(Placement::Raw { chunk }) => {
                    // Physically adjacent chunks become one transfer. Copying a
                    // chunk at a time meant 64 KiB per device memcpy *and* a
                    // stream sync per chunk inside `write_device_bytes`, which
                    // is what made staging a large file cost hundreds of
                    // milliseconds instead of a few.
                    take = self.raw_run_bytes(
                        src_path,
                        src_pos,
                        take as usize,
                        (len - done) as usize,
                        chunk,
                    ) as u64;
                    self.vram_base + chunk as u64 * CHUNK_SIZE + src_in
                }
                Some(Placement::Compressed {
                    codec: Codec::Lz4, ..
                }) => {
                    let mut blobs = Vec::new();
                    let mut parts = Vec::new();
                    while done < len && blobs.len() < crate::nvcomp::BATCH {
                        let src_pos = src_offset + done;
                        let src_lc = (src_pos / CHUNK_SIZE) as usize;
                        let src_in = src_pos % CHUNK_SIZE;
                        let take = (len - done).min(CHUNK_SIZE - src_in);
                        match self.coord(src_path, src_lc) {
                            Some(Placement::Compressed {
                                offset,
                                len,
                                codec: Codec::Lz4,
                            }) => {
                                blobs.push((offset, len));
                                parts.push((done, src_in, take));
                                done += take;
                            }
                            _ => break,
                        }
                    }
                    let slot_ptrs = {
                        let vram_base = self.vram_base;
                        let codec = self.codec.as_mut().ok_or_else(|| {
                            EngineError::Unsupported(
                                "archive job found an LZ4-compressed source chunk but nvCOMP LZ4 is unavailable".into(),
                            )
                        })?;
                        cuda(codec.decompress_from_arena_dev(vram_base, &blobs))?;
                        (0..blobs.len())
                            .map(|i| codec.uncomp_slot_ptr(i))
                            .collect::<Vec<_>>()
                    };
                    for (i, (dst_done, src_in, take)) in parts.into_iter().enumerate() {
                        self.write_device_bytes(
                            dst_path,
                            dst_offset + dst_done,
                            slot_ptrs[i] + src_in,
                            take,
                        )?;
                    }
                    continue;
                }
                Some(Placement::Compressed {
                    offset,
                    len,
                    codec: Codec::Zstd,
                }) => {
                    let full = self.decompress_blob(offset, len, Codec::Zstd)?;
                    let start = src_in as usize;
                    self.write_raw_internal(
                        dst_path,
                        dst_offset + done,
                        &full[start..start + take as usize],
                    )?;
                    done += take;
                    continue;
                }
                None => {
                    self.write_raw_internal(
                        dst_path,
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

    /// Remove any leftover `\.__vramdisk_*` temp files from the root.
    ///
    /// Archive jobs stage data in uniquely named root-level temp files; the
    /// success and cancellation paths remove them inline, and this sweep backs
    /// up every `?` error path so a failed job can never leak the temp file's
    /// VRAM (or leave a stray node visible in the namespace). Only one archive
    /// job runs at a time (single job worker + engine mutex), so sweeping by
    /// prefix cannot hit another job's temps.
    fn cleanup_archive_temp_files(&mut self) {
        let stale: Vec<String> = match self.table.readdir("\\") {
            Ok(entries) => entries
                .into_iter()
                .filter(|(name, node)| !node.is_dir && name.starts_with(".__vramdisk_"))
                .map(|(name, _)| format!("\\{}", name.to_ascii_lowercase()))
                .collect(),
            Err(_) => return,
        };
        for path in stale {
            let _ = self.remove(&path);
        }
    }

    fn archive_temp_chunk(&mut self) -> EResult<ChunkId> {
        self.alloc_chunk().map_err(|err| match err {
            EngineError::NoSpace => archive_vram_exhausted("materialize compressed source data"),
            other => other,
        })
    }

    fn release_temp_chunks(&mut self, temp_chunks: Vec<ChunkId>) {
        for chunk in temp_chunks {
            self.release_chunk(chunk);
        }
    }

    /// Build the GPU hash segments for `[offset, offset+len)` of `path` into
    /// `out`, materializing compressed chunks into freshly allocated temp
    /// chunks recorded in `out.temp_chunks`.
    ///
    /// `out` is caller-provided (rather than returned) so that on an error
    /// mid-build the caller still sees — and can release — the temp chunks
    /// allocated so far; returning the struct would drop them on the `?` path
    /// and leak the VRAM.
    fn archive_crc32_segments(
        &mut self,
        path: &str,
        offset: u64,
        len: u64,
        out: &mut ArchiveMaterializedSegments,
    ) -> EResult<()> {
        let mut done = 0u64;
        while done < len {
            let pos = offset + done;
            let lc = (pos / CHUNK_SIZE) as usize;
            let in_off = pos % CHUNK_SIZE;
            let take = (len - done).min(CHUNK_SIZE - in_off) as u32;
            match self.coord(path, lc) {
                None => {
                    out.segs.push(HashSegment {
                        ptr: 0,
                        len: take,
                        kind: 1,
                    });
                    done += take as u64;
                }
                Some(Placement::Raw { chunk }) => {
                    out.segs.push(HashSegment {
                        ptr: self.vram_base + chunk as u64 * CHUNK_SIZE + in_off,
                        len: take,
                        kind: 0,
                    });
                    done += take as u64;
                }
                Some(Placement::Compressed {
                    offset,
                    len,
                    codec: Codec::Zstd,
                }) => {
                    let full = self.decompress_blob(offset, len, Codec::Zstd)?;
                    let chunk = self.archive_temp_chunk()?;
                    cuda(self.vram.write_at(chunk as u64 * CHUNK_SIZE, &full))?;
                    out.temp_chunks.push(chunk);
                    out.segs.push(HashSegment {
                        ptr: self.vram_base + chunk as u64 * CHUNK_SIZE + in_off,
                        len: take,
                        kind: 0,
                    });
                    done += take as u64;
                }
                Some(Placement::Compressed {
                    codec: Codec::Lz4, ..
                }) => {
                    let mut blobs = Vec::new();
                    let mut parts = Vec::new();
                    while done < len && blobs.len() < crate::nvcomp::BATCH {
                        let pos = offset + done;
                        let lc = (pos / CHUNK_SIZE) as usize;
                        let in_off = pos % CHUNK_SIZE;
                        let take = (len - done).min(CHUNK_SIZE - in_off) as u32;
                        match self.coord(path, lc) {
                            Some(Placement::Compressed {
                                offset,
                                len,
                                codec: Codec::Lz4,
                            }) => {
                                blobs.push((offset, len));
                                parts.push((in_off, take));
                                done += take as u64;
                            }
                            _ => break,
                        }
                    }
                    let base_chunk = out.temp_chunks.len();
                    for _ in 0..blobs.len() {
                        out.temp_chunks.push(self.archive_temp_chunk()?);
                    }
                    let slot_ptrs = {
                        let vram_base = self.vram_base;
                        let codec = self.codec.as_mut().ok_or_else(|| {
                            EngineError::Unsupported(
                                "archive job found an LZ4-compressed source chunk but nvCOMP LZ4 is unavailable".into(),
                            )
                        })?;
                        cuda(codec.decompress_from_arena_dev(vram_base, &blobs))?;
                        (0..blobs.len())
                            .map(|i| codec.uncomp_slot_ptr(i))
                            .collect::<Vec<_>>()
                    };
                    for (i, (in_off, take)) in parts.into_iter().enumerate() {
                        let chunk = out.temp_chunks[base_chunk + i];
                        cuda(self.vram.copy_dev_into(
                            chunk as u64 * CHUNK_SIZE,
                            slot_ptrs[i],
                            CHUNK_SIZE,
                        ))?;
                        out.segs.push(HashSegment {
                            ptr: self.vram_base + chunk as u64 * CHUNK_SIZE + in_off,
                            len: take,
                            kind: 0,
                        });
                    }
                    cuda(self.vram.sync())?;
                }
            }
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
            let mut take = (len - done).min(CHUNK_SIZE - in_off);
            let chunk = self.ensure_raw_output_chunk(path, lc)?;
            // Extend across destination chunks that are already allocated and
            // physically adjacent -- the usual case for a freshly allocated
            // contiguous output file, which turns 8192 device memcpys into one.
            take =
                self.raw_run_bytes(path, pos, take as usize, (len - done) as usize, chunk) as u64;
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
            // A shorter run than requested is useless here: give it back
            // instead of leaking `got` freshly marked chunks.
            for c in start..start + got {
                self.release_chunk(c);
            }
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
        // The recycled chunks still hold whatever a previously freed file left
        // in them. Callers overwrite [0, len) but not the tail of the last
        // chunk, which would otherwise leak stale data if the file is later
        // grown with SetFileSize.
        let tail = len % CHUNK_SIZE;
        if tail != 0 {
            let last = start as u64 + chunks as u64 - 1;
            cuda(
                self.vram
                    .zero_at(last * CHUNK_SIZE + tail, CHUNK_SIZE - tail),
            )?;
        }
        self.set_size(path, len)
    }

    /// ZIP writer: GPU CRC32 per member, then GPU Deflate per member.
    ///
    /// `progress` follows the contract documented on
    /// [`StorageEngine::hash_file_gpu_cancellable`]. Unlike the tar formats,
    /// ZIP reads every payload *twice* — once for the CRC32 that has to go in
    /// the local header, once for the Deflate pass — so the denominator here
    /// is twice the summed input size: the CRC pass fills the first half and
    /// the compression pass the second. Counting only one pass would make the
    /// bar reach 100% at the halfway point and then sit there.
    fn archive_zip_compress_gpu<F>(
        &mut self,
        files: &[String],
        output: &str,
        start: Instant,
        output_created: &mut bool,
        progress: &mut F,
    ) -> EResult<ArchiveJobStats>
    where
        F: FnMut(u64, u64) -> bool,
    {
        // Sizes come from the in-memory lookup table; a path that is missing
        // or is a directory contributes 0 and is rejected with the proper
        // error by the CRC loop below.
        let payload_total: u64 = files
            .iter()
            .map(|file| {
                self.table
                    .get(&crate::lookup::normalize(file))
                    .map(|node| node.size)
                    .unwrap_or(0)
            })
            .sum();
        let progress_total = payload_total.saturating_mul(2);
        let mut crc_done = 0u64;
        let mut planned = Vec::with_capacity(files.len());
        for file in files {
            if progress(crc_done, progress_total) {
                return cancelled();
            }
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
                return Err(EngineError::InvalidInput(format!(
                    "zip path must be ASCII: {name}"
                )));
            }
            // Shift the CRC pass's own 0..size counters into the job's frame so
            // a single large member still moves the bar while it is scanned.
            let crc_done_before = crc_done;
            let crc = self.crc32_file_gpu(&path, size, &mut |file_done, _| {
                progress(crc_done_before.saturating_add(file_done), progress_total)
            })?;
            crc_done = crc_done.saturating_add(size);
            planned.push((path, name, size, 0u64, crc));
        }
        self.create_or_truncate_file(output)?;
        *output_created = true;
        let mut deflate = cuda(NvcompBatchedCodec::load(
            &self.vram,
            NvcompFrameCodec::Deflate,
        ))?;
        let mut out_pos = 0u64;
        let mut central = Vec::new();
        let mut input_bytes = 0u64;
        for (path, name, size, _comp_size, crc) in &planned {
            // The CRC pass covered `payload_total`; this pass adds the payload
            // bytes it has deflated so far, so the count continues rather than
            // restarting at the phase boundary.
            if progress(payload_total.saturating_add(input_bytes), progress_total) {
                return cancelled();
            }
            let local_offset = out_pos;
            let name_bytes = name.as_bytes();
            let zip_chunks = zip_deflate_chunk_count(*size);
            let hdr = zip_local_header(
                8,
                *crc,
                name_bytes,
                &zip_local_extra(*size, 0, &vec![0; zip_chunks]),
            )?;
            self.write_raw_internal(output, out_pos, &hdr)?;
            out_pos += hdr.len() as u64;
            let mut method = 8u16;
            let comp_size =
                match self.write_zip_deflate_payload(path, *size, output, out_pos, &mut deflate)? {
                    Some((comp_size, chunk_sizes)) => {
                        patch_zip64_local_sizes(
                            self,
                            output,
                            local_offset + 30 + name_bytes.len() as u64,
                            *size,
                            comp_size,
                            &chunk_sizes,
                        )?;
                        comp_size
                    }
                    None => {
                        // The member's DEFLATE chunks could not be spliced into one
                        // conformant stream, so rewrite it from the local header
                        // down as a stored (method 0) member. Always readable,
                        // never compressed — see `write_zip_deflate_payload`.
                        method = 0;
                        let hdr = zip_local_header(
                            0,
                            *crc,
                            name_bytes,
                            &zip_local_extra(*size, *size, &[]),
                        )?;
                        out_pos = local_offset;
                        self.write_raw_internal(output, out_pos, &hdr)?;
                        out_pos += hdr.len() as u64;
                        self.copy_file_payload_raw(path, 0, output, out_pos, *size)?;
                        *size
                    }
                };
            out_pos += comp_size;
            central.push(ZipCentralEntry {
                method,
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
            push_u16(&mut hdr, entry.method);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 0);
            push_u32(&mut hdr, entry.crc);
            push_u32(&mut hdr, u32::MAX);
            push_u32(&mut hdr, u32::MAX);
            push_u16(
                &mut hdr,
                u16_checked(name_bytes.len(), "zip central file name length")?,
            );
            push_u16(
                &mut hdr,
                u16_checked(extra.len(), "zip64 central extra length")?,
            );
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 0);
            push_u16(&mut hdr, 0);
            push_u32(&mut hdr, 0);
            push_u32(&mut hdr, u32::MAX);
            hdr.extend_from_slice(name_bytes);
            hdr.extend_from_slice(&extra);
            self.write_raw_internal(output, out_pos, &hdr)?;
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
        self.write_raw_internal(output, out_pos, &zip64)?;
        out_pos += zip64.len() as u64;

        let mut zip64_locator = Vec::with_capacity(20);
        push_u32(&mut zip64_locator, 0x0706_4b50);
        push_u32(&mut zip64_locator, 0);
        push_u64(&mut zip64_locator, zip64_eocd_offset);
        push_u32(&mut zip64_locator, 1);
        self.write_raw_internal(output, out_pos, &zip64_locator)?;
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
        self.write_raw_internal(output, out_pos, &eocd)?;
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

    /// ZIP reader: walk the local headers, inflating each member on the GPU.
    ///
    /// `progress` follows the contract documented on
    /// [`StorageEngine::hash_file_gpu_cancellable`]. The ZIP is read in place
    /// rather than staged, so the natural counters are the walk's own: how far
    /// into the archive the header cursor has advanced, against the archive
    /// size. The central directory at the tail is never reached (the walk
    /// stops at its signature), so the count ends slightly short of the total
    /// — harmless, since a job that returns `Ok` is snapped to its total.
    fn archive_zip_extract_gpu<F>(
        &mut self,
        archive: &str,
        output_dir: &str,
        start: Instant,
        progress: &mut F,
    ) -> EResult<ArchiveExtractStats>
    where
        F: FnMut(u64, u64) -> bool,
    {
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
            if progress(pos, archive_size) {
                return cancelled();
            }
            let sig = self.read(archive, pos, 4)?;
            let sig = read_u32_le(&sig);
            if sig == 0x0201_4b50 || sig == 0x0605_4b50 {
                break;
            }
            if sig != 0x0403_4b50 {
                return Err(EngineError::InvalidInput(format!(
                    "unsupported zip signature {sig:08x} at {pos}"
                )));
            }
            let hdr = self.read(archive, pos, 30)?;
            if hdr.len() < 30 {
                return Err(EngineError::InvalidInput(
                    "truncated zip local header".into(),
                ));
            }
            let method = read_u16_le(&hdr[8..10]);
            let comp32 = read_u32_le(&hdr[18..22]);
            let uncomp32 = read_u32_le(&hdr[22..26]);
            let name_len = read_u16_le(&hdr[26..28]) as u64;
            let extra_len = read_u16_le(&hdr[28..30]) as u64;
            let name_bytes = self.read(archive, pos + 30, name_len as usize)?;
            let extra = self.read(
                archive,
                pos + 30 + name_len,
                usize::try_from(extra_len).map_err(|_| {
                    EngineError::InvalidInput("zip extra length exceeds usize".into())
                })?,
            )?;
            if name_bytes.len() as u64 != name_len || extra.len() as u64 != extra_len {
                return Err(EngineError::InvalidInput(
                    "truncated zip local header fields".into(),
                ));
            }
            let (uncomp_size, comp_size) = zip_sizes_from_local_extra(uncomp32, comp32, &extra)?;
            let name = std::str::from_utf8(&name_bytes)
                .map_err(|e| EngineError::InvalidInput(format!("invalid zip path UTF-8: {e}")))?;
            let data_pos = pos + 30 + name_len + extra_len;
            let out_path = join_archive_output(&out_base, name)?;
            self.ensure_parent_dirs(&out_path)?;
            self.create_or_truncate_file(&out_path)?;
            match method {
                0 => self.copy_file_payload_raw(archive, data_pos, &out_path, 0, uncomp_size)?,
                8 => {
                    if let Some(table) = zip_deflate_chunks_from_extra(&extra)? {
                        self.extract_zip_deflate_chunks(
                            &mut deflate_codec,
                            archive,
                            data_pos,
                            &table,
                            &out_path,
                            uncomp_size,
                        )?;
                    } else if comp_size == stored_deflate_len(uncomp_size) {
                        let written = self.extract_stored_deflate_stream(
                            archive, data_pos, comp_size, &out_path,
                        )?;
                        if written != uncomp_size {
                            return Err(EngineError::InvalidInput(format!(
                                "zip stored-deflate size mismatch: expected {uncomp_size}, got {written}"
                            )));
                        }
                    } else {
                        self.extract_deflate_payload(
                            &mut deflate_codec,
                            PayloadSpan {
                                src_path: archive,
                                src_offset: data_pos,
                                comp_len: comp_size,
                                dst_path: &out_path,
                                dst_offset: 0,
                                out_len: uncomp_size,
                            },
                        )?;
                    }
                }
                _ => {
                    return Err(EngineError::Unsupported(format!(
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

    /// Deflate one ZIP member, splicing nvCOMP's per-chunk streams into a
    /// single standards-conformant DEFLATE stream.
    ///
    /// The member is cut into [`ZIP_DEFLATE_CHUNK`] pieces because that is the
    /// only axis nvCOMP's Deflate parallelises over: measured on this machine,
    /// 256 MiB compresses in 0.163 s as 256 × 1 MiB chunks and in 46.7 s as one
    /// chunk, so "just compress the member as one stream" is a 280× regression
    /// and not an option.
    ///
    /// Joining those pieces is the delicate part. Each is a complete DEFLATE
    /// stream ending at an arbitrary *bit*, and the old code cleared the
    /// `BFINAL` flag and concatenated at byte boundaries — which left the
    /// encoder's zero padding sitting between two blocks, where a decoder reads
    /// it as the header of a stored block and then swallows the next chunk's
    /// first bytes as that block's LEN/NLEN. VRAMDISK's own extractor never
    /// noticed (it decompresses chunk by chunk from the private table below),
    /// but Explorer and .NET's `ZipArchive` both refused any member over
    /// 1 MiB.
    ///
    /// So instead: [`DeflateWalker`] finds each chunk's exact end bit, the last
    /// block's `BFINAL` is cleared wherever it actually lives, and an empty
    /// stored block ([`zip_deflate_joiner`]) is spliced on to carry the stream
    /// back to a byte boundary before the next chunk starts. The member's final
    /// chunk keeps nvCOMP's own `BFINAL` and is written untouched — which is
    /// also why a member of one chunk or less costs nothing extra.
    ///
    /// Returns `None` when a chunk's stream could not be walked or did not
    /// account for exactly the bytes it was compressed from; the caller then
    /// stores the member instead of risking a corrupt one.
    fn write_zip_deflate_payload(
        &mut self,
        src_path: &str,
        len: u64,
        dst_path: &str,
        out_pos: u64,
        codec: &mut NvcompBatchedCodec,
    ) -> EResult<Option<(u64, Vec<u64>)>> {
        if len == 0 {
            self.write_raw_internal(dst_path, out_pos, &ZIP_DEFLATE_TERMINATOR)?;
            return Ok(Some((5, vec![5])));
        }
        let tmp = format!("\\.__vramdisk_zip_src_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&tmp)?;
        let spliced = self.write_zip_deflate_chunks(&tmp, src_path, len, dst_path, out_pos, codec);
        let _ = self.remove(&tmp);
        spliced
    }

    /// The chunk loop behind [`StorageEngine::write_zip_deflate_payload`], split
    /// out so the staging file is removed on every exit path.
    fn write_zip_deflate_chunks(
        &mut self,
        tmp: &str,
        src_path: &str,
        len: u64,
        dst_path: &str,
        mut out_pos: u64,
        codec: &mut NvcompBatchedCodec,
    ) -> EResult<Option<(u64, Vec<u64>)>> {
        self.allocate_raw_file(tmp, len)?;
        self.copy_file_payload_raw(src_path, 0, tmp, 0, len)?;
        let base = self.contiguous_file_ptr(tmp, len)?;
        let total_chunks = len.div_ceil(ZIP_DEFLATE_CHUNK);
        let mut chunk_idx = 0u64;
        let mut written = 0u64;
        let mut chunk_comp_sizes = Vec::with_capacity(total_chunks as usize);
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        while chunk_idx < total_chunks {
            let n = ((total_chunks - chunk_idx).min(crate::nvcomp::BATCH as u64)) as usize;
            let mut ptrs = Vec::with_capacity(n);
            let mut sizes = Vec::with_capacity(n);
            for i in 0..n {
                let off = (chunk_idx + i as u64) * ZIP_DEFLATE_CHUNK;
                ptrs.push(base + off);
                sizes.push((len - off).min(ZIP_DEFLATE_CHUNK));
            }
            let comp_sizes = cuda(codec.compress_device(&ptrs, &sizes))?;
            // The compressed blobs live in codec scratch that the next launch
            // overwrites, so this batch is walked and written out before the
            // loop comes back around.
            let mut slot = 0usize;
            while slot < n {
                let group = (n - slot).min(ZIP_DEFLATE_WALK_GROUP);
                blobs.clear();
                for j in 0..group {
                    let i = slot + j;
                    if chunk_idx + i as u64 + 1 == total_chunks {
                        // Nothing is spliced onto the member's last chunk, so
                        // it needs no end bit and no host round trip.
                        blobs.push(Vec::new());
                        continue;
                    }
                    let mut blob = vec![0u8; comp_sizes[i] as usize];
                    cuda(codec.copy_compressed_slot_to_host(i, &mut blob))?;
                    blobs.push(blob);
                }
                let walks = walk_deflate_blobs(&blobs);
                for j in 0..group {
                    let i = slot + j;
                    let comp_size = comp_sizes[i];
                    if blobs[j].is_empty() {
                        self.write_device_bytes(
                            dst_path,
                            out_pos,
                            codec.compressed_slot_ptr(i),
                            comp_size,
                        )?;
                        out_pos += comp_size;
                        written += comp_size;
                        chunk_comp_sizes.push(comp_size);
                        continue;
                    }
                    let off = (chunk_idx + i as u64) * ZIP_DEFLATE_CHUNK;
                    let Some(walk) = walks[j] else {
                        return Ok(None);
                    };
                    if walk.out_len != (len - off).min(ZIP_DEFLATE_CHUNK)
                        || walk.end_bit == 0
                        || walk.end_bit > comp_size * 8
                    {
                        return Ok(None);
                    }
                    let used = walk.end_bit.div_ceil(8);
                    self.write_device_bytes(dst_path, out_pos, codec.compressed_slot_ptr(i), used)?;
                    // Open the stream up (clear the last block's BFINAL) and
                    // clear anything the encoder left past the end bit, so the
                    // joiner's all-zero block header lands on clean padding.
                    let blob = &mut blobs[j];
                    let bfinal_byte = (walk.final_bfinal_bit / 8) as usize;
                    blob[bfinal_byte] &= !(1u8 << (walk.final_bfinal_bit % 8));
                    let last = used as usize - 1;
                    let rem = (walk.end_bit % 8) as u32;
                    if rem != 0 {
                        blob[last] &= ((1u16 << rem) - 1) as u8;
                    }
                    let joiner = zip_deflate_joiner(walk.end_bit);
                    // The rewritten tail byte and the joiner are adjacent, so
                    // they go out as one write; only the BFINAL byte (byte 0 of
                    // a single-block chunk) needs a second.
                    let mut tail = Vec::with_capacity(1 + joiner.len());
                    tail.push(blob[last]);
                    tail.extend_from_slice(joiner);
                    if bfinal_byte != last {
                        self.write_raw_internal(
                            dst_path,
                            out_pos + bfinal_byte as u64,
                            &blob[bfinal_byte..bfinal_byte + 1],
                        )?;
                    } else {
                        tail[0] = blob[last];
                    }
                    self.write_raw_internal(dst_path, out_pos + last as u64, &tail)?;
                    let total = used + joiner.len() as u64;
                    out_pos += total;
                    written += total;
                    chunk_comp_sizes.push(total);
                }
                slot += group;
            }
            chunk_idx += n as u64;
        }
        Ok(Some((written, chunk_comp_sizes)))
    }

    /// Inflate a member from the private per-chunk table, one nvCOMP launch per
    /// chunk straight into the output file's VRAM.
    ///
    /// [`ZipChunkTable::spliced`] distinguishes the two on-disk shapes the
    /// table can describe. A spliced chunk is one written by the current
    /// [`StorageEngine::write_zip_deflate_payload`]: its last block's `BFINAL`
    /// was cleared and an empty stored block appended, which leaves it
    /// byte-aligned, so appending [`ZIP_DEFLATE_TERMINATOR`] closes it no
    /// matter how many blocks it holds. A legacy chunk was written by the code
    /// that produced the broken archives — it ends on a bit boundary and had
    /// only bit 0 of byte 0 cleared, so setting that bit back is both the only
    /// thing that can close it and the only thing that ever did.
    fn extract_zip_deflate_chunks(
        &mut self,
        codec: &mut NvcompBatchedCodec,
        src_path: &str,
        src_offset: u64,
        table: &ZipChunkTable,
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
        for (i, &comp_len) in table.sizes.iter().enumerate() {
            let take = (out_len - out_pos).min(ZIP_DEFLATE_CHUNK);
            let is_last = i + 1 == table.sizes.len();
            // A spliced chunk that is not the member's last one is still an
            // open stream and needs the terminator; the last one already ends
            // in its own final block.
            let terminate = table.spliced && !is_last;
            let staged = comp_len
                + if terminate {
                    ZIP_DEFLATE_TERMINATOR.len() as u64
                } else {
                    0
                };
            let tmp = format!(
                "\\.__vramdisk_zip_deflate_src_{}",
                crate::lookup::now_filetime()
            );
            self.create_or_truncate_file(&tmp)?;
            self.allocate_raw_file(&tmp, staged)?;
            self.copy_file_payload_raw(src_path, comp_pos, &tmp, 0, comp_len)?;
            if terminate {
                self.write_raw_internal(&tmp, comp_len, &ZIP_DEFLATE_TERMINATOR)?;
            } else if !table.spliced {
                self.set_zip_deflate_bfinal(&tmp, 0)?;
            }
            let src_ptr = self.contiguous_file_ptr(&tmp, staged)?;
            let produced = cuda(codec.decompress_device(
                &[src_ptr],
                &[staged],
                &[dst_base + out_pos],
                &[take],
            ))?;
            let _ = self.remove(&tmp);
            if produced.first().copied() != Some(take) {
                return Err(EngineError::InvalidInput(format!(
                    "ZIP Deflate chunk {i} produced {:?} bytes, expected {take}",
                    produced.first()
                )));
            }
            comp_pos += comp_len;
            out_pos += take;
            if out_pos == out_len {
                break;
            }
        }
        if out_pos != out_len {
            return Err(EngineError::InvalidInput(
                "ZIP Deflate chunk table ended early".into(),
            ));
        }
        Ok(())
    }

    fn set_zip_deflate_bfinal(&mut self, path: &str, offset: u64) -> EResult<()> {
        let mut b = self.read(path, offset, 1)?;
        if b.len() != 1 {
            return Err(EngineError::InvalidInput(
                "truncated deflate payload".into(),
            ));
        }
        b[0] |= 1;
        self.write_raw_internal(path, offset, &b)?;
        Ok(())
    }

    /// CRC32 over a whole file. `progress` reports `0..len` for this file
    /// alone; callers that are part of a larger job wrap it to shift those
    /// counters into their own frame.
    fn crc32_file_gpu<F>(&mut self, path: &str, len: u64, progress: &mut F) -> EResult<u32>
    where
        F: FnMut(u64, u64) -> bool,
    {
        self.crc32_range_gpu_cancellable(path, 0, len, progress)
    }

    fn crc32_range_gpu(&mut self, path: &str, offset: u64, len: u64) -> EResult<u32> {
        self.crc32_range_gpu_cancellable(path, offset, len, &mut |_, _| false)
    }

    /// CRC32 over `[offset, offset + len)`, computed as many parallel lanes.
    ///
    /// # Why lanes rather than one running state
    ///
    /// `vramdisk_crc32_many` gives one *thread* to each entry of the batch it
    /// is handed, because that batch is normally a set of independent files.
    /// Feeding it a single entry therefore checksums the whole range on a
    /// single CUDA thread, and a lone GPU thread walking a byte-at-a-time
    /// table CRC runs at roughly 5 MB/s — three orders of magnitude below what
    /// the device can do and around 300x slower than the same loop on a CPU
    /// core. That is what made the ZIP writer (one CRC pass per member) and
    /// the gzip reader (one verification pass per member) dominate their jobs:
    /// a 256 MiB ZIP member spent 7.98 s in this function against 0.16 s in
    /// the Deflate pass that followed it.
    ///
    /// CRC-32 is a linear function over GF(2), so the range can be cut into
    /// independent lanes that are checksummed concurrently and then folded
    /// back together with [`crc32_combine_with`] — the exact same value as a
    /// serial scan, bit for bit, which matters because these checksums go into
    /// ZIP local headers and gzip trailers that other tools verify.
    ///
    /// `progress` follows the contract documented on
    /// [`StorageEngine::hash_file_gpu_cancellable`], counting bytes of the
    /// requested range: `done_bytes` is what previous passes have fed to the
    /// kernel and `total_bytes` is `len`, so the ratio is exact.
    fn crc32_range_gpu_cancellable<F>(
        &mut self,
        path: &str,
        offset: u64,
        len: u64,
        progress: &mut F,
    ) -> EResult<u32>
    where
        F: FnMut(u64, u64) -> bool,
    {
        self.ensure_hash_calibration()?;
        // A launch over raw and sparse placements reads VRAM in place and so
        // costs nothing but descriptors, but a compressed placement is first
        // decompressed into a temp chunk that has to stay allocated until the
        // launch has read it. Sizing a compressed launch by lane count would
        // ask for one temp chunk per 64 KiB of the whole launch — half a
        // gigabyte of scratch for the default — so those fall back to the hash
        // launch budget, which is what bounded this scratch before lanes
        // existed. 64 lanes is still 64x the parallelism of a single thread.
        let launch_bytes = if self.range_has_only_raw_sparse(path, offset, len)? {
            self.crc32_launch_bytes
        } else {
            self.crc32_launch_bytes.min(self.gpu_hash_launch_budget)
        };
        // CRC-32 of the empty string, which is also the identity for the fold
        // below: `crc32_combine_with(shift, 0, c) == c`.
        let mut crc = 0u32;
        // Lanes all share one length except the last of the range, so the
        // GF(2) operator that advances a CRC across a lane is built once and
        // reused for every fold step instead of once per lane.
        let mut shift_cache: Option<(u64, [u32; 32])> = None;
        let mut done = 0u64;
        while done < len {
            if progress(done, len) {
                return cancelled();
            }
            let take = (len - done).min(launch_bytes);
            // Build segments into a caller-owned struct, then release its temp
            // chunks whether or not the build or the kernel update succeeded.
            let mut materialized = ArchiveMaterializedSegments::default();
            let mut lanes = Crc32Lanes::default();
            let built =
                self.build_crc32_lanes(path, offset + done, take, &mut materialized, &mut lanes);
            let launched =
                built.and_then(|()| cuda(self.api_kernel()?.crc32_many(&lanes.segments)));
            self.release_temp_chunks(std::mem::take(&mut materialized.temp_chunks));
            let lane_crcs = launched?;
            for (&lane_len, &lane_crc) in lanes.lengths.iter().zip(lane_crcs.iter()) {
                let shift = match shift_cache {
                    Some((cached_len, shift)) if cached_len == lane_len => shift,
                    _ => {
                        let shift = crc32_zero_shift(lane_len);
                        shift_cache = Some((lane_len, shift));
                        shift
                    }
                };
                crc = crc32_combine_with(&shift, crc, lane_crc);
            }
            done += take;
        }
        Ok(crc)
    }

    /// Cut `[offset, offset + len)` into [`CRC32_LANE_BYTES`] lanes and build
    /// one GPU segment list per lane.
    ///
    /// Temp chunks materialized for compressed placements accumulate in `out`
    /// (shared by every lane) so the caller can release them once the launch
    /// that reads them has completed, exactly as the single-lane version did.
    fn build_crc32_lanes(
        &mut self,
        path: &str,
        offset: u64,
        len: u64,
        out: &mut ArchiveMaterializedSegments,
        lanes: &mut Crc32Lanes,
    ) -> EResult<()> {
        let mut done = 0u64;
        while done < len {
            let take = (len - done).min(CRC32_LANE_BYTES);
            self.archive_crc32_segments(path, offset + done, take, out)?;
            lanes.segments.push(std::mem::take(&mut out.segs));
            lanes.lengths.push(take);
            done += take;
        }
        Ok(())
    }

    fn raw_file_ptr(&self, path: &str, offset: u64) -> EResult<u64> {
        let lc = (offset / CHUNK_SIZE) as usize;
        let in_off = offset % CHUNK_SIZE;
        match self.coord(path, lc) {
            Some(Placement::Raw { chunk }) => {
                Ok(self.vram_base + chunk as u64 * CHUNK_SIZE + in_off)
            }
            Some(Placement::Compressed { .. }) => Err(EngineError::Unsupported(
                "archive codec input currently requires raw placement".into(),
            )),
            None => Err(EngineError::Unsupported(
                "archive codec input is sparse".into(),
            )),
        }
    }

    fn raw_output_ptr(&mut self, path: &str, offset: u64, len: u64) -> EResult<u64> {
        if len > CHUNK_SIZE - (offset % CHUNK_SIZE) {
            return Err(EngineError::Unsupported(
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
                    return Err(EngineError::Unsupported(
                        "GPU batch hash currently requires raw/sparse placements".into(),
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
        span: PayloadSpan<'_>,
    ) -> EResult<()> {
        let tmp = format!(
            "\\.__vramdisk_deflate_src_{}",
            crate::lookup::now_filetime()
        );
        self.create_or_truncate_file(&tmp)?;
        self.allocate_raw_file(&tmp, span.comp_len)?;
        self.copy_file_payload_raw(span.src_path, span.src_offset, &tmp, 0, span.comp_len)?;
        let src_ptr = self.contiguous_file_ptr(&tmp, span.comp_len)?;
        let dst_ptr = if span.dst_offset == 0 {
            self.allocate_raw_file(span.dst_path, span.out_len)?;
            self.contiguous_file_ptr(span.dst_path, span.out_len)?
        } else {
            self.raw_output_ptr(span.dst_path, span.dst_offset, span.out_len)?
        };
        cuda(codec.decompress_device(&[src_ptr], &[span.comp_len], &[dst_ptr], &[span.out_len]))?;
        let _ = self.remove(&tmp);
        Ok(())
    }

    fn extract_lz4_payload(
        &mut self,
        codec: &mut NvcompBatchedCodec,
        span: PayloadSpan<'_>,
    ) -> EResult<()> {
        let tmp = format!("\\.__vramdisk_lz4_src_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&tmp)?;
        self.allocate_raw_file(&tmp, span.comp_len)?;
        self.copy_file_payload_raw(span.src_path, span.src_offset, &tmp, 0, span.comp_len)?;
        let src_ptr = self.contiguous_file_ptr(&tmp, span.comp_len)?;
        let dst_ptr = self.raw_output_ptr(span.dst_path, span.dst_offset, span.out_len)?;
        cuda(codec.decompress_device(&[src_ptr], &[span.comp_len], &[dst_ptr], &[span.out_len]))?;
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
                self.write_raw_internal(dst_path, out_pos, &header)?;
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
                self.write_raw_internal(dst_path, out_pos, &trailer)?;
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
            self.write_raw_internal(dst_path, out_pos, &header)?;
            out_pos += header.len() as u64;
            self.write_raw_internal(dst_path, out_pos, &[0, 0, 0, 0, 0, 0, 0, 0])?;
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
                return Err(EngineError::InvalidInput("unsupported gzip header".into()));
            }
            if hdr[3] & 4 == 0 {
                return Err(EngineError::Unsupported(
                    "gzip member is missing VRAMDISK compressed-size extra field".into(),
                ));
            }
            src_pos += 10;
            let xlen_buf = self.read(archive, src_pos, 2)?;
            if xlen_buf.len() < 2 {
                return Err(EngineError::InvalidInput(
                    "truncated gzip extra length".into(),
                ));
            }
            let xlen = read_u16_le(&xlen_buf) as u64;
            src_pos += 2;
            let extra = self.read(archive, src_pos, xlen as usize)?;
            if extra.len() as u64 != xlen {
                return Err(EngineError::InvalidInput(
                    "truncated gzip extra field".into(),
                ));
            }
            src_pos += xlen;
            let comp_len = gzip_extra_comp_len(&extra)?;
            let trailer_pos = src_pos + comp_len;
            if trailer_pos + 8 > archive_size {
                return Err(EngineError::InvalidInput("truncated gzip member".into()));
            }
            let trailer = self.read(archive, trailer_pos, 8)?;
            let expected_crc = read_u32_le(&trailer[0..4]);
            let expected_size = read_u32_le(&trailer[4..8]) as u64;
            if expected_size > 0 {
                self.extract_deflate_payload(
                    &mut codec,
                    PayloadSpan {
                        src_path: archive,
                        src_offset: src_pos,
                        comp_len,
                        dst_path,
                        dst_offset: out_pos,
                        out_len: expected_size,
                    },
                )?;
            }
            let actual_crc = self.crc32_range_gpu(dst_path, out_pos, expected_size)?;
            if actual_crc != expected_crc {
                return Err(EngineError::InvalidInput("gzip CRC32 mismatch".into()));
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
        self.write_raw_internal(dst_path, 0, &header)?;
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
                    self.write_raw_internal(dst_path, out_pos, &(sz as u32).to_le_bytes())?;
                    out_pos += 4;
                    self.write_device_bytes(dst_path, out_pos, codec.compressed_slot_ptr(i), sz)?;
                    out_pos += sz;
                } else {
                    let marker = (sizes[i] as u32) | 0x8000_0000;
                    self.write_raw_internal(dst_path, out_pos, &marker.to_le_bytes())?;
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
        self.write_raw_internal(dst_path, out_pos, &0u32.to_le_bytes())?;
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
            return Err(EngineError::InvalidInput(
                "unsupported LZ4 frame header".into(),
            ));
        }
        // A frame that does not carry a content size, or that uses a block size
        // other than 64 KiB, is a perfectly valid LZ4 frame that this extractor
        // simply cannot drive -- not a malformed one.
        if header[4] != 0x68 || header[5] != 0x40 {
            return Err(EngineError::Unsupported(
                "LZ4 frame must declare a content size and use 64 KiB blocks".into(),
            ));
        }
        if lz4_header_checksum(&header[4..14]) != header[14] {
            return Err(EngineError::InvalidInput(
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
                return Err(EngineError::InvalidInput(
                    "truncated LZ4 frame block".into(),
                ));
            }
            if uncompressed {
                if block_len != out_len {
                    return Err(EngineError::InvalidInput(
                        "LZ4 raw block size mismatch".into(),
                    ));
                }
                self.copy_file_payload_raw(archive, src_pos, dst_path, out_pos, out_len)?;
            } else {
                self.extract_lz4_payload(
                    &mut codec,
                    PayloadSpan {
                        src_path: archive,
                        src_offset: src_pos,
                        comp_len: block_len,
                        dst_path,
                        dst_offset: out_pos,
                        out_len,
                    },
                )?;
            }
            src_pos += block_len;
            out_pos += out_len;
        }
        if out_pos != total {
            return Err(EngineError::InvalidInput(
                "LZ4 frame content size mismatch".into(),
            ));
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
            self.write_raw_internal(dst_path, out_pos, &[1, 0, 0, 255, 255])?;
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
            self.write_raw_internal(dst_path, out_pos, &hdr)?;
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
                return Err(EngineError::InvalidInput(
                    "truncated stored deflate block".into(),
                ));
            }
            if hdr[0] & 0b0000_0110 != 0 {
                return Err(EngineError::InvalidInput(
                    "expected a stored deflate block in ZIP fallback".into(),
                ));
            }
            let len = u16::from_le_bytes([hdr[1], hdr[2]]) as u64;
            let nlen = u16::from_le_bytes([hdr[3], hdr[4]]);
            if nlen != !(len as u16) {
                return Err(EngineError::InvalidInput(
                    "invalid stored deflate LEN/NLEN".into(),
                ));
            }
            src_pos += 5;
            if src_pos + len > end {
                return Err(EngineError::InvalidInput(
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
            return Err(EngineError::InvalidInput(
                "stored deflate stream length mismatch".into(),
            ));
        }
        Ok(out_pos)
    }

    fn ensure_raw_output_chunk(&mut self, path: &str, lc: usize) -> EResult<ChunkId> {
        match self.coord(path, lc) {
            Some(Placement::Raw { chunk }) => Ok(chunk),
            Some(Placement::Compressed { .. }) => Err(EngineError::Internal(
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
                return Err(EngineError::Unsupported(
                    "archive temp stream must be raw and contiguous".into(),
                ));
            }
            None => {
                return Err(EngineError::Unsupported(
                    "archive temp stream is sparse".into(),
                ))
            }
        };
        for lc in 0..chunks {
            match self.coord(path, lc) {
                Some(Placement::Raw { chunk }) if chunk == first + lc as u32 => {}
                Some(Placement::Raw { .. }) => {
                    return Err(EngineError::Unsupported(
                        "archive temp stream is not physically contiguous".into(),
                    ));
                }
                Some(Placement::Compressed { .. }) => {
                    return Err(EngineError::Unsupported(
                        "archive temp stream must be raw and contiguous".into(),
                    ));
                }
                None => {
                    return Err(EngineError::Unsupported(
                        "archive temp stream is sparse".into(),
                    ))
                }
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

    // ---- GPU encode/decode jobs (Base64 / hex) ------------------------------

    /// GPU file transcoding: Base64/hex encode or decode `input` into
    /// `output`, both on the mounted volume.
    ///
    /// Works in fixed-size staged passes so it never needs one contiguous
    /// VRAM run the size of the file: each pass materializes a slice of the
    /// input (raw/sparse/compressed placements all accepted) into a small
    /// contiguous staging area, transcodes it on the GPU, and scatters the
    /// result into the output file's chunks device-to-device. File payload
    /// bytes never round-trip through host memory; only the trailing partial
    /// Base64 group is finished on the CPU.
    ///
    /// `progress` follows the contract documented on
    /// [`StorageEngine::hash_file_gpu_cancellable`] and counts *input* bytes
    /// transcoded on the GPU: one staged pass per tick.
    pub fn encode_file_gpu_cancellable<F>(
        &mut self,
        codec: EncodeCodec,
        direction: EncodeDirection,
        input: &str,
        output: &str,
        mut progress: F,
    ) -> EResult<EncodeJobStats>
    where
        F: FnMut(u64, u64) -> bool,
    {
        let start = Instant::now();
        let input = crate::lookup::normalize(input);
        let output = crate::lookup::normalize(output);
        if input == output {
            return Err(EngineError::InvalidInput(
                "encode input and output must be different files".into(),
            ));
        }
        let mut output_created = false;
        let result = map_archive_job_result(self.encode_file_gpu_inner(
            codec,
            direction,
            &input,
            &output,
            &mut output_created,
            &mut progress,
        ));
        match result {
            Ok((input_bytes, output_bytes)) => Ok(EncodeJobStats {
                codec: codec.name().to_string(),
                direction: direction.name().to_string(),
                input,
                output,
                input_bytes,
                output_bytes,
                elapsed_ms: start.elapsed().as_millis(),
            }),
            Err(e) => {
                if output_created {
                    let _ = self.remove(&output);
                }
                self.cleanup_archive_temp_files();
                Err(e)
            }
        }
    }

    fn encode_file_gpu_inner<F>(
        &mut self,
        codec: EncodeCodec,
        direction: EncodeDirection,
        input: &str,
        output: &str,
        output_created: &mut bool,
        progress: &mut F,
    ) -> EResult<(u64, u64)>
    where
        F: FnMut(u64, u64) -> bool,
    {
        let size = {
            let node = self.table.get(input).ok_or(LookupError::NotFound)?;
            if node.is_dir {
                return Err(EngineError::NotAFile);
            }
            node.size
        };

        // Effective input length: decodes ignore trailing ASCII whitespace
        // (a final newline is near-universal in encoded text files).
        let m = match direction {
            EncodeDirection::Encode => size,
            EncodeDirection::Decode => self.trim_trailing_whitespace_len(input, size)?,
        };

        // Input-unit / output-unit byte sizes for one transcoding group.
        let (in_unit, out_unit): (u64, u64) = match (codec, direction) {
            (EncodeCodec::Base64, EncodeDirection::Encode) => (3, 4),
            (EncodeCodec::Base64, EncodeDirection::Decode) => (4, 3),
            (EncodeCodec::Hex, EncodeDirection::Encode) => (1, 2),
            (EncodeCodec::Hex, EncodeDirection::Decode) => (2, 1),
        };

        // Validate decode alignment and work out the exact output length.
        let (gpu_in_len, out_len, tail_host): (u64, u64, Option<Vec<u8>>) = match (codec, direction)
        {
            (EncodeCodec::Base64, EncodeDirection::Encode) => (m, m.div_ceil(3) * 4, None),
            (EncodeCodec::Hex, EncodeDirection::Encode) => (m, m * 2, None),
            (EncodeCodec::Hex, EncodeDirection::Decode) => {
                if m % 2 != 0 {
                    return Err(EngineError::InvalidInput(
                        "hex decode requires an even number of hex digits".into(),
                    ));
                }
                (m, m / 2, None)
            }
            (EncodeCodec::Base64, EncodeDirection::Decode) => {
                if m % 4 != 0 {
                    return Err(EngineError::InvalidInput(
                        "base64 decode requires input length to be a multiple of 4 \
                             (single-line base64 without embedded line breaks)"
                            .into(),
                    ));
                }
                if m == 0 {
                    (0, 0, None)
                } else {
                    // Decode the final (possibly '='-padded) group on the
                    // host; the GPU handles only full non-padded groups.
                    let last = self.read(input, m - 4, 4)?;
                    let tail = decode_base64_quad(&last)?;
                    let out_len = (m / 4 - 1) * 3 + tail.len() as u64;
                    (m - 4, out_len, Some(tail))
                }
            }
        };

        self.create_or_truncate_file(output)?;
        *output_created = true;
        if out_len == 0 {
            self.set_size(output, 0)?;
            return Ok((size, 0));
        }

        // Contiguous staging area: input slice + transcoded output slice in
        // one temp file. Sized to the input (small files stage in one pass)
        // and clamped so staging never eats more than half the free space.
        let unit_lcm = 12u64; // lcm of every in_unit above
        let free_bytes = (self.alloc.free() as u64) * CHUNK_SIZE;
        let max_in_stage = ENCODE_STAGE_BYTES
            .min((free_bytes / 2) / (1 + out_unit.div_ceil(in_unit)))
            .max(unit_lcm);
        let in_stage = gpu_in_len
            .div_ceil(unit_lcm)
            .saturating_mul(unit_lcm)
            .min(max_in_stage / unit_lcm * unit_lcm)
            .max(unit_lcm);
        let out_stage = in_stage / in_unit * out_unit;
        let staging = format!("\\.__vramdisk_encode_tmp_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&staging)?;
        self.allocate_raw_file(&staging, in_stage + out_stage)
            .map_err(|err| match err {
                EngineError::NoSpace => {
                    archive_vram_exhausted("stage encode input and output slices")
                }
                other => other,
            })?;
        let stage_base = self.contiguous_file_ptr(&staging, in_stage + out_stage)?;
        let out_stage_ptr = stage_base + in_stage;

        let mut in_pos = 0u64;
        let mut out_pos = 0u64;
        while in_pos < gpu_in_len {
            // `gpu_in_len` rather than the file size: it is the length this
            // loop actually transcodes, i.e. the input minus the trailing
            // whitespace a decode ignores and minus the final padded Base64
            // group, which is finished on the host after the loop.
            if progress(in_pos, gpu_in_len) {
                let _ = self.remove(&staging);
                return cancelled();
            }
            let take = (gpu_in_len - in_pos).min(in_stage);
            self.copy_file_payload_raw(input, in_pos, &staging, 0, take)?;
            let kernel = self.api_kernel()?;
            match (codec, direction) {
                (EncodeCodec::Base64, EncodeDirection::Encode) => {
                    cuda(kernel.base64_encode(stage_base, take, out_stage_ptr))?
                }
                (EncodeCodec::Base64, EncodeDirection::Decode) => {
                    cuda(kernel.base64_decode(stage_base, take / 4, out_stage_ptr))?
                }
                (EncodeCodec::Hex, EncodeDirection::Encode) => {
                    cuda(kernel.hex_encode(stage_base, take, out_stage_ptr))?
                }
                (EncodeCodec::Hex, EncodeDirection::Decode) => {
                    cuda(kernel.hex_decode(stage_base, take / 2, out_stage_ptr))?
                }
            }
            let out_take = take.div_ceil(in_unit) * out_unit;
            self.write_device_bytes(output, out_pos, out_stage_ptr, out_take)?;
            in_pos += take;
            out_pos += out_take;
        }
        if let Some(tail) = tail_host {
            self.write_raw_internal(output, out_pos, &tail)?;
        }
        self.set_size(output, out_len)?;
        let _ = self.remove(&staging);
        Ok((size, out_len))
    }

    /// Scan every file in `paths` for the literal byte string `needle`.
    ///
    /// This is the operation the hardware is actually good at. The bytes are
    /// already in VRAM, so the scan runs at device memory bandwidth and never
    /// crosses PCIe -- unlike a CPU tool, which has to pull the whole volume
    /// through the bus before it can look at any of it.
    ///
    /// A file's chunks need not be adjacent and may be compressed, so each pass
    /// materializes a contiguous raw window first (a device-to-device copy) and
    /// scans that. Consecutive windows step by `window - (needle.len() - 1)` so
    /// a match straddling the seam is still found, and because the kernel's
    /// last candidate in a window is exactly one before the next window's
    /// first, no match is counted twice.
    ///
    /// `max_offsets_per_file` bounds what is *reported*; the match counts are
    /// always exact.
    pub fn search_files_gpu_cancellable<F>(
        &mut self,
        paths: &[String],
        needle: &[u8],
        ignore_case: bool,
        max_offsets_per_file: usize,
        mut progress: F,
    ) -> EResult<SearchJobStats>
    where
        F: FnMut(u64, u64) -> bool,
    {
        let start = Instant::now();
        if needle.is_empty() {
            return Err(EngineError::InvalidInput(
                "search pattern must not be empty".into(),
            ));
        }
        if needle.len() > SEARCH_MAX_PATTERN {
            return Err(EngineError::InvalidInput(format!(
                "search pattern must be at most {SEARCH_MAX_PATTERN} bytes"
            )));
        }

        let mut sizes = Vec::with_capacity(paths.len());
        let mut total_bytes = 0u64;
        for path in paths {
            let node = self.table.get(path).ok_or(LookupError::NotFound)?;
            if node.is_dir {
                return Err(EngineError::NotAFile);
            }
            sizes.push(node.size);
            total_bytes = total_bytes.saturating_add(node.size);
        }

        // The window is real VRAM, so never take more than half of what is
        // free: a search must not push the volume into no-space.
        let free_bytes = (self.alloc.free() as u64) * CHUNK_SIZE;
        let window = self
            .search_window_bytes
            .min((free_bytes / 2) / CHUNK_SIZE * CHUNK_SIZE)
            .max(CHUNK_SIZE);
        let staging = format!("\\.__vramdisk_search_tmp_{}", crate::lookup::now_filetime());
        self.create_or_truncate_file(&staging)?;
        self.allocate_raw_file(&staging, window)
            .map_err(|err| match err {
                EngineError::NoSpace => archive_vram_exhausted("stage a search window"),
                other => other,
            })?;
        let plan = SearchPlan {
            needle,
            ignore_case,
            max_offsets_per_file,
            staging: &staging,
            window,
            total_bytes,
        };
        let result = self.search_inner(paths, &sizes, &plan, &mut progress);
        let _ = self.remove(&staging);
        let (hits, files_matched, total_matches, bytes_scanned) = result?;
        Ok(SearchJobStats {
            pattern_len: needle.len(),
            ignore_case,
            files_scanned: paths.len() as u64,
            bytes_scanned,
            files_matched,
            total_matches,
            hits,
            elapsed_ms: start.elapsed().as_millis(),
        })
    }

    /// Every physically contiguous raw run backing `[0, size)` of `path`, or
    /// `None` when any of it is compressed or a sparse hole.
    ///
    /// `None` sends the caller to the staging path, which handles those
    /// uniformly. Holes are excluded rather than skipped because a needle of
    /// all-zero bytes really can match inside one, and getting that subtly
    /// wrong is worse than materializing the file.
    fn raw_runs(&self, path: &str, size: u64) -> Option<Vec<(u64, u64, u64)>> {
        if size == 0 {
            return Some(Vec::new());
        }
        let mut runs: Vec<(u64, u64, u64)> = Vec::new();
        let mut pos = 0u64;
        while pos < size {
            let lc = (pos / CHUNK_SIZE) as usize;
            let take = (size - pos).min(CHUNK_SIZE);
            let chunk = match self.coord(path, lc)? {
                Placement::Raw { chunk } => chunk,
                Placement::Compressed { .. } => return None,
            };
            let ptr = self.vram_base + chunk as u64 * CHUNK_SIZE;
            match runs.last_mut() {
                Some((_, last_ptr, last_len)) if *last_ptr + *last_len == ptr => {
                    *last_len += take;
                }
                _ => runs.push((pos, ptr, take)),
            }
            pos += take;
        }
        Some(runs)
    }

    /// Scan `runs` in place, stitching each run boundary on the host.
    ///
    /// A run's kernel launch finds every match starting at or before
    /// `run.end - needle.len()`. The few candidate starts left over -- the last
    /// `needle.len() - 1` positions of a run -- span into the next run, which is
    /// somewhere else in VRAM entirely. Those are at most a few hundred bytes,
    /// so they are read back and matched on the host rather than staged.
    fn search_raw_runs<F>(
        &mut self,
        path: &str,
        runs: &[(u64, u64, u64)],
        plan: &SearchPlan<'_>,
        done: &mut u64,
        progress: &mut F,
    ) -> EResult<(u64, Vec<u64>, bool)>
    where
        F: FnMut(u64, u64) -> bool,
    {
        let SearchPlan {
            needle,
            ignore_case,
            max_offsets_per_file: max_offsets,
            total_bytes,
            ..
        } = *plan;
        let overlap = needle.len() as u64 - 1;
        let mut matches = 0u64;
        let mut offsets: Vec<u64> = Vec::new();
        let mut truncated = false;
        let push = |off: u64, offsets: &mut Vec<u64>, truncated: &mut bool| {
            if offsets.len() < max_offsets {
                offsets.push(off);
            } else {
                *truncated = true;
            }
        };

        for (i, &(file_off, ptr, len)) in runs.iter().enumerate() {
            if progress(*done, total_bytes) {
                return cancelled();
            }
            if len >= needle.len() as u64 {
                let launch = {
                    let kernel = self.api_kernel()?;
                    cuda(kernel.search(ptr, len, needle, ignore_case, file_off))?
                };
                matches = matches.saturating_add(launch.total);
                for off in launch.offsets {
                    push(off, &mut offsets, &mut truncated);
                }
            }
            *done = done.saturating_add(len);

            // Candidates straddling the seam into the next run.
            if overlap > 0 && i + 1 < runs.len() {
                let seam_start = (file_off + len).saturating_sub(overlap);
                let seam_len = (overlap * 2).min(runs[i + 1].0 + runs[i + 1].2 - seam_start);
                let bytes = self.read(path, seam_start, seam_len as usize)?;
                for (at, w) in bytes.windows(needle.len()).enumerate() {
                    let hit = if ignore_case {
                        w.iter().zip(needle).all(|(a, b)| a.eq_ignore_ascii_case(b))
                    } else {
                        w == needle
                    };
                    // Only starts the run's own launch could not reach.
                    if hit && (at as u64) < overlap {
                        matches += 1;
                        push(seam_start + at as u64, &mut offsets, &mut truncated);
                    }
                }
            }
        }
        truncated = truncated || (offsets.len() as u64) < matches;
        Ok((matches, offsets, truncated))
    }

    fn search_inner<F>(
        &mut self,
        paths: &[String],
        sizes: &[u64],
        plan: &SearchPlan<'_>,
        progress: &mut F,
    ) -> EResult<(Vec<SearchHit>, u64, u64, u64)>
    where
        F: FnMut(u64, u64) -> bool,
    {
        let SearchPlan {
            needle,
            ignore_case,
            max_offsets_per_file,
            staging,
            window,
            total_bytes,
        } = *plan;
        let base = self.contiguous_file_ptr(staging, window)?;
        let overlap = needle.len() as u64 - 1;
        let mut hits = Vec::new();
        let mut files_matched = 0u64;
        let mut total_matches = 0u64;
        let mut done = 0u64;

        for (path, &size) in paths.iter().zip(sizes.iter()) {
            // Fast path: a file whose data is entirely raw can be scanned where
            // it already lies in VRAM. Staging it into a contiguous window
            // first costs a full device-to-device copy of the file, which
            // dominates the scan itself by an order of magnitude.
            if let Some(runs) = self.raw_runs(path, size) {
                let (file_matches, offsets, truncated) =
                    self.search_raw_runs(path, &runs, plan, &mut done, progress)?;
                if file_matches > 0 {
                    files_matched += 1;
                    total_matches = total_matches.saturating_add(file_matches);
                    let mut offsets = offsets;
                    offsets.sort_unstable();
                    hits.push(SearchHit {
                        path: path.clone(),
                        matches: file_matches,
                        offsets,
                        truncated,
                    });
                }
                continue;
            }
            let mut pos = 0u64;
            let mut file_matches = 0u64;
            let mut offsets: Vec<u64> = Vec::new();
            let mut truncated = false;
            while pos < size {
                if progress(done, total_bytes) {
                    return cancelled();
                }
                let take = (size - pos).min(window);
                if take < needle.len() as u64 {
                    done = done.saturating_add(take);
                    break;
                }
                self.copy_file_payload_raw(path, pos, staging, 0, take)?;
                let launch = {
                    let kernel = self.api_kernel()?;
                    cuda(kernel.search(base, take, needle, ignore_case, pos))?
                };
                file_matches = file_matches.saturating_add(launch.total);
                for off in launch.offsets {
                    if offsets.len() < max_offsets_per_file {
                        offsets.push(off);
                    } else {
                        truncated = true;
                        break;
                    }
                }
                let last_window = pos + take >= size;
                let step = if last_window { take } else { take - overlap };
                done = done.saturating_add(step);
                if last_window {
                    break;
                }
                pos += step;
            }
            if file_matches > 0 {
                files_matched += 1;
                total_matches = total_matches.saturating_add(file_matches);
                truncated = truncated || (offsets.len() as u64) < file_matches;
                offsets.sort_unstable();
                hits.push(SearchHit {
                    path: path.clone(),
                    matches: file_matches,
                    offsets,
                    truncated,
                });
            }
        }
        progress(done, total_bytes);
        Ok((hits, files_matched, total_matches, done))
    }

    /// Length of `path`'s content once trailing ASCII whitespace is dropped.
    fn trim_trailing_whitespace_len(&mut self, path: &str, size: u64) -> EResult<u64> {
        let mut end = size;
        while end > 0 {
            let take = end.min(4096);
            let block = self.read(path, end - take, take as usize)?;
            let kept = block
                .iter()
                .rposition(|b| !matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
                .map(|i| i as u64 + 1)
                .unwrap_or(0);
            end = end - take + kept;
            if kept != 0 {
                break;
            }
        }
        Ok(end)
    }
}

/// One compressed archive payload: where its bytes live now and where the
/// decompressed bytes must land.
///
/// `extract_deflate_payload` and `extract_lz4_payload` both need two
/// `(path, offset, length)` triples, and passing six positional values meant a
/// transposed source/destination pair would still compile and would quietly
/// corrupt the extraction. Naming the fields makes that class of mistake
/// visible at the call site.
struct PayloadSpan<'a> {
    /// File holding the compressed bytes.
    src_path: &'a str,
    /// Byte offset of the payload within `src_path`.
    src_offset: u64,
    /// Compressed length, i.e. how many bytes to feed the codec.
    comp_len: u64,
    /// File the plaintext is written to.
    dst_path: &'a str,
    /// Byte offset within `dst_path` to write at. Zero additionally means the
    /// whole destination may be (re)allocated as one contiguous raw run.
    dst_offset: u64,
    /// Expected decompressed length; the codec is told to produce exactly this.
    out_len: u64,
}

struct ZipCentralEntry {
    /// Compression method actually used: 8 (Deflate), or 0 when the member had
    /// to be stored because its chunks could not be spliced.
    method: u16,
    crc: u32,
    size: u64,
    comp_size: u64,
    local_offset: u64,
    name: String,
}

/// Build a ZIP64 local file header. Sizes are always the `0xffffffff` escape,
/// with the real values living in the ZIP64 extra field.
fn zip_local_header(method: u16, crc: u32, name: &[u8], extra: &[u8]) -> EResult<Vec<u8>> {
    let mut hdr = Vec::with_capacity(30 + name.len() + extra.len());
    push_u32(&mut hdr, 0x0403_4b50);
    push_u16(&mut hdr, 45);
    push_u16(&mut hdr, 0);
    push_u16(&mut hdr, method);
    push_u16(&mut hdr, 0);
    push_u16(&mut hdr, 0);
    push_u32(&mut hdr, crc);
    push_u32(&mut hdr, u32::MAX);
    push_u32(&mut hdr, u32::MAX);
    push_u16(&mut hdr, u16_checked(name.len(), "zip file name length")?);
    push_u16(
        &mut hdr,
        u16_checked(extra.len(), "zip64 local extra length")?,
    );
    hdr.extend_from_slice(name);
    hdr.extend_from_slice(extra);
    Ok(hdr)
}

fn pad512(n: u64) -> u64 {
    (512 - (n % 512)) % 512
}

fn tar_header(path: &str, size: u64) -> EResult<[u8; 512]> {
    if path.is_empty() || path.len() > 100 {
        return Err(EngineError::InvalidInput(format!(
            "tar path must be 1..100 ASCII bytes for current GPU archive writer: {path}"
        )));
    }
    if !path.is_ascii() {
        return Err(EngineError::InvalidInput(format!(
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
        .map_err(|e| EngineError::InvalidInput(format!("invalid tar octal field: {e}")))?;
    u64::from_str_radix(text.trim(), 8)
        .map_err(|e| EngineError::InvalidInput(format!("invalid tar octal value: {e}")))
}

/// Join an archive entry name onto the extraction base, rejecting `.`/`..`
/// path components so a crafted archive can't place a literal `..` node in
/// the namespace (tar/zip "slip") instead of extracting under `base`.
fn join_archive_output(base: &str, name: &str) -> EResult<String> {
    let clean = name.trim_start_matches('/').replace('/', "\\");
    if clean.split('\\').any(|comp| comp == "." || comp == "..") {
        return Err(EngineError::InvalidInput(format!(
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
            return Err(EngineError::InvalidInput(
                "invalid gzip extra length".into(),
            ));
        }
        if si1 == b'G' && si2 == b'S' && len == 8 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&extra[pos..pos + 8]);
            return Ok(u64::from_le_bytes(bytes));
        }
        pos += len;
    }
    Err(EngineError::Unsupported(
        "gzip member is missing VRAMDISK compressed-size subfield".into(),
    ))
}

fn zip_local_extra(uncomp_size: u64, comp_size: u64, chunk_comp_sizes: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + 8 + chunk_comp_sizes.len() * 8);
    push_u16(&mut out, 0x0001);
    push_u16(&mut out, 16);
    push_u64(&mut out, uncomp_size);
    push_u64(&mut out, comp_size);
    let mut start = 0usize;
    while start < chunk_comp_sizes.len() {
        let take = (chunk_comp_sizes.len() - start).min(8190);
        push_u16(&mut out, ZIP_CHUNK_TABLE_TAG);
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
            return Err(EngineError::InvalidInput(
                "invalid ZIP extra field length".into(),
            ));
        }
        if tag == 0x0001 {
            let field = &extra[pos..pos + len];
            if field.len() < 16 {
                return Err(EngineError::InvalidInput(
                    "truncated ZIP64 size extra field".into(),
                ));
            }
            return Ok((read_u64_le(&field[0..8]), read_u64_le(&field[8..16])));
        }
        pos += len;
    }
    Err(EngineError::InvalidInput(
        "ZIP entry uses 0xffffffff sizes without ZIP64 extra field".into(),
    ))
}

/// A member's private per-chunk compressed-size table, and which of the two
/// on-disk chunk shapes it describes.
///
/// `spliced` is false only for [`ZIP_CHUNK_TABLE_TAG_LEGACY`] archives, whose
/// chunks are closed by setting bit 0 of their first byte rather than by
/// appending a terminator.
struct ZipChunkTable {
    sizes: Vec<u64>,
    spliced: bool,
}

fn zip_deflate_chunks_from_extra(extra: &[u8]) -> EResult<Option<ZipChunkTable>> {
    let mut pos = 0usize;
    let mut sizes = Vec::new();
    let mut spliced = true;
    while pos + 4 <= extra.len() {
        let tag = read_u16_le(&extra[pos..pos + 2]);
        let len = read_u16_le(&extra[pos + 2..pos + 4]) as usize;
        pos += 4;
        if pos + len > extra.len() {
            return Err(EngineError::InvalidInput(
                "invalid ZIP extra field length".into(),
            ));
        }
        if tag == ZIP_CHUNK_TABLE_TAG || tag == ZIP_CHUNK_TABLE_TAG_LEGACY {
            if tag == ZIP_CHUNK_TABLE_TAG_LEGACY {
                spliced = false;
            }
            if len < 8 || (len - 8) % 8 != 0 {
                return Err(EngineError::InvalidInput(
                    "invalid VRAMDISK ZIP chunk table".into(),
                ));
            }
            let start = read_u32_le(&extra[pos..pos + 4]) as usize;
            let count = read_u32_le(&extra[pos + 4..pos + 8]) as usize;
            if count != (len - 8) / 8 {
                return Err(EngineError::InvalidInput(
                    "VRAMDISK ZIP chunk table count mismatch".into(),
                ));
            }
            if sizes.len() < start {
                return Err(EngineError::InvalidInput(
                    "VRAMDISK ZIP chunk table has a gap".into(),
                ));
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
        Ok(Some(ZipChunkTable { sizes, spliced }))
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
    let extra = zip_local_extra(uncomp_size, comp_size, chunk_comp_sizes);
    debug_assert!(chunk_comp_sizes.is_empty() || chunk_comp_sizes.iter().sum::<u64>() == comp_size);
    engine.write_raw_internal(path, extra_offset, &extra)?;
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
    u16::try_from(v).map_err(|_| EngineError::InvalidInput(format!("{what} exceeds u16")))
}

#[allow(dead_code)]
fn u32_checked(v: u64, what: &str) -> EResult<u32> {
    u32::try_from(v).map_err(|_| EngineError::InvalidInput(format!("{what} exceeds u32")))
}

/// Apply a GF(2) linear operator, held as its 32 column vectors, to a CRC
/// register.
///
/// A reflected CRC-32 register is a vector over GF(2), and every operation the
/// algorithm performs on it — shifting in a bit, feeding a byte, appending a
/// run of zeros — is linear. Such an operator is fully described by where it
/// sends each of the 32 basis vectors, which is what `mat` holds, so applying
/// it is XOR-ing together the columns selected by the set bits of `crc`.
fn crc32_apply(mat: &[u32; 32], mut crc: u32) -> u32 {
    let mut out = 0u32;
    let mut col = 0usize;
    while crc != 0 {
        if crc & 1 != 0 {
            out ^= mat[col];
        }
        crc >>= 1;
        col += 1;
    }
    out
}

/// Compose two GF(2) operators: the result applies `b` and then `a`.
fn crc32_compose(a: &[u32; 32], b: &[u32; 32]) -> [u32; 32] {
    let mut out = [0u32; 32];
    for (slot, column) in out.iter_mut().zip(b.iter()) {
        *slot = crc32_apply(a, *column);
    }
    out
}

/// The operator for advancing a CRC-32 register across `len` zero bytes.
///
/// Built by repeated squaring over the bit-level operator, so the cost is
/// logarithmic in `len` rather than linear — the whole point of precomputing
/// it once per lane length instead of running zlib's `crc32_combine` per lane,
/// which rebuilds this ladder every call and turned out to cost more than the
/// GPU scan it was folding.
fn crc32_zero_shift(len: u64) -> [u32; 32] {
    // The identity operator: basis vector `n` maps to itself.
    let mut result = [0u32; 32];
    for (n, slot) in result.iter_mut().enumerate() {
        *slot = 1u32 << n;
    }
    if len == 0 {
        return result;
    }
    // `odd` starts as the operator for one zero *bit*: the low bit falls out
    // and, when set, the polynomial is XOR-ed back in.
    let mut odd = [0u32; 32];
    odd[0] = 0xedb8_8320;
    for (n, slot) in odd.iter_mut().enumerate().skip(1) {
        *slot = 1u32 << (n - 1);
    }
    let mut even = crc32_compose(&odd, &odd); // two bits
    odd = crc32_compose(&even, &even); // four bits
    let mut len = len;
    loop {
        even = crc32_compose(&odd, &odd); // eight bits: one byte, then 2, 4, ...
        if len & 1 != 0 {
            result = crc32_compose(&even, &result);
        }
        len >>= 1;
        if len == 0 {
            break;
        }
        odd = crc32_compose(&even, &even);
        if len & 1 != 0 {
            result = crc32_compose(&odd, &result);
        }
        len >>= 1;
        if len == 0 {
            break;
        }
    }
    result
}

/// Concatenate two CRC-32 values: the checksum of `a` followed by `b`, where
/// `shift` is [`crc32_zero_shift`] for the length of `b`.
///
/// This is zlib's `crc32_combine` with the operator lifted out of the call, so
/// a fold over thousands of equal-length lanes builds the ladder once. Both
/// inputs and the result are finalized CRC values (post `^ 0xffffffff`), which
/// is what `vramdisk_crc32_many_final` returns and what ZIP and gzip store.
fn crc32_combine_with(shift: &[u32; 32], a: u32, b: u32) -> u32 {
    crc32_apply(shift, a) ^ b
}

fn stored_deflate_len(len: u64) -> u64 {
    if len == 0 {
        return 5;
    }
    len + len.div_ceil(65_535) * 5
}

// ---------------------------------------------------------------------------
// DEFLATE bitstream structure walker
// ---------------------------------------------------------------------------

/// Longest DEFLATE Huffman code, in bits (RFC 1951 §3.2.7).
const DEFLATE_MAX_CODE_BITS: u32 = 15;

/// Entries in a flat Huffman decode table: one per [`DEFLATE_MAX_CODE_BITS`]
/// bit lookahead value.
///
/// A single flat table rather than zlib's two-level one because the walker only
/// ever builds a handful of tables per megabyte of payload (nvCOMP emits one
/// dynamic block per chunk), so the 64 KiB fill is amortised over hundreds of
/// thousands of symbol decodes, and the decode itself becomes one load.
const DEFLATE_TABLE_LEN: usize = 1 << DEFLATE_MAX_CODE_BITS;

/// Entries in the code-length alphabet's decode table (codes are ≤ 7 bits).
const DEFLATE_CLEN_TABLE_LEN: usize = 1 << 7;

/// Order in which the 19 code-length code lengths appear in a dynamic header.
const DEFLATE_CLEN_ORDER: [u8; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Match length for literal/length symbols 257..=285, before the extra bits.
const DEFLATE_LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra bits carried by literal/length symbols 257..=285.
const DEFLATE_LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Extra bits carried by distance symbols 0..=29.
const DEFLATE_DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// What a walk of one DEFLATE stream found out about its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeflateWalk {
    /// Bit offset, counted from the stream's first bit, one past the last
    /// block's end-of-block symbol. Everything at or after this offset inside
    /// the final byte is the encoder's zero padding.
    end_bit: u64,
    /// Bit offset of the `BFINAL` flag of the stream's *last* block — the bit
    /// that has to be cleared to make the stream continue into the next one.
    /// Not necessarily bit 0 of byte 0: a stream may hold several blocks.
    final_bfinal_bit: u64,
    /// Bytes the stream expands to. Checking this against the size the chunk
    /// was compressed from proves the whole symbol stream was decoded
    /// correctly, which is what makes [`DeflateWalk::end_bit`] trustworthy.
    out_len: u64,
}

/// A cursor that reads DEFLATE's LSB-first bit packing out of a byte slice.
///
/// Bits are pulled into a 64-bit accumulator eight bytes at a time so the hot
/// path (peek 15, consume `n`) is a mask and two shifts. Past the end of the
/// slice the accumulator reads as zeros, which is harmless because every
/// consumer checks `have` before committing to a code length — a code that
/// would need bits beyond the buffer is rejected instead of being decoded out
/// of phantom zeros.
struct DeflateBits<'a> {
    data: &'a [u8],
    /// Index of the next byte to pull into `acc`.
    next: usize,
    /// Bit buffer, LSB = next bit of the stream.
    acc: u64,
    /// Valid bits currently in `acc`.
    have: u32,
    /// Bits consumed so far, from the start of `data`.
    pos: u64,
}

impl<'a> DeflateBits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            next: 0,
            acc: 0,
            have: 0,
            pos: 0,
        }
    }

    #[inline]
    fn refill(&mut self) {
        while self.have <= 56 {
            let Some(&b) = self.data.get(self.next) else {
                break;
            };
            self.acc |= (b as u64) << self.have;
            self.have += 8;
            self.next += 1;
        }
    }

    /// Consume the next `n` (≤ 32) bits, LSB first.
    #[inline]
    fn take(&mut self, n: u32) -> EResult<u32> {
        self.refill();
        if self.have < n {
            return Err(deflate_error("stream ends inside a code"));
        }
        let v = (self.acc & ((1u64 << n) - 1)) as u32;
        self.acc >>= n;
        self.have -= n;
        self.pos += u64::from(n);
        Ok(v)
    }

    /// Consume one Huffman code and return its symbol. `table` is a flat
    /// lookup table with a power-of-two length, indexed by that many bits of
    /// lookahead.
    #[inline]
    fn decode(&mut self, table: &[u16]) -> EResult<u16> {
        self.refill();
        let entry = table[(self.acc & (table.len() as u64 - 1)) as usize];
        let len = u32::from(entry & 0xf);
        if len == 0 {
            return Err(deflate_error("undefined Huffman code"));
        }
        if len > self.have {
            return Err(deflate_error("stream ends inside a Huffman code"));
        }
        self.acc >>= len;
        self.have -= len;
        self.pos += u64::from(len);
        Ok(entry >> 4)
    }

    /// Discard bits up to the next byte boundary (what a stored block header
    /// does before its LEN/NLEN pair).
    fn align(&mut self) -> EResult<()> {
        let rem = (self.pos % 8) as u32;
        if rem != 0 {
            self.take(8 - rem)?;
        }
        Ok(())
    }

    /// Jump `n` bytes forward from the current (byte-aligned) position.
    fn skip_bytes(&mut self, n: u64) -> EResult<()> {
        debug_assert_eq!(self.pos % 8, 0);
        let byte = self.pos / 8 + n;
        if byte > self.data.len() as u64 {
            return Err(deflate_error(
                "stored block runs past the end of the stream",
            ));
        }
        self.next = byte as usize;
        self.acc = 0;
        self.have = 0;
        self.pos = byte * 8;
        Ok(())
    }
}

fn deflate_error(what: &str) -> EngineError {
    EngineError::InvalidInput(format!("malformed DEFLATE stream: {what}"))
}

/// Reverse the low `len` bits of `code`.
///
/// Canonical Huffman codes are defined MSB-first but written to the bitstream
/// LSB-first, so the flat lookup table has to be indexed by the reversed code.
fn reverse_code_bits(mut code: u32, len: u32) -> u32 {
    let mut out = 0u32;
    for _ in 0..len {
        out = (out << 1) | (code & 1);
        code >>= 1;
    }
    out
}

/// Fill `table` (`1 << bits` entries) with a canonical Huffman code built from
/// `lengths`, packing each entry as `symbol << 4 | code_length`.
///
/// Entries left at zero decode as "undefined": DEFLATE explicitly permits
/// *incomplete* codes (a distance tree with a single one-bit code, or none at
/// all, is legal), and those unreachable slots must be rejected rather than
/// silently aliased onto a real symbol. Over-subscribed codes are rejected up
/// front, since they have no canonical assignment at all.
fn build_deflate_table(lengths: &[u8], bits: u32, table: &mut [u16]) -> EResult<()> {
    let n = 1usize << bits;
    table[..n].fill(0);
    let mut count = [0u16; 16];
    for &l in lengths {
        if u32::from(l) > bits {
            return Err(deflate_error(
                "Huffman code longer than the alphabet allows",
            ));
        }
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut left = 1i32;
    for &at_len in count.iter().take(bits as usize + 1).skip(1) {
        left <<= 1;
        left -= i32::from(at_len);
        if left < 0 {
            return Err(deflate_error("over-subscribed Huffman code"));
        }
    }
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for l in 1..=bits as usize {
        code = (code + u32::from(count[l - 1])) << 1;
        next_code[l] = code;
    }
    for (sym, &l) in lengths.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let l = u32::from(l);
        let assigned = next_code[l as usize];
        next_code[l as usize] += 1;
        let entry = ((sym as u16) << 4) | l as u16;
        let step = 1usize << l;
        let mut i = reverse_code_bits(assigned, l) as usize;
        while i < n {
            table[i] = entry;
            i += step;
        }
    }
    Ok(())
}

/// Walks DEFLATE streams to find where they end, reusing its decode tables.
///
/// # Why this exists
///
/// nvCOMP compresses a ZIP member as many independent 1 MiB chunks so the GPU
/// has something to parallelise over, but a ZIP member is one DEFLATE stream.
/// DEFLATE is a *bitstream*: a block ends at an arbitrary bit inside its last
/// byte and the encoder zero-pads the rest of that byte. Concatenating chunks
/// at byte boundaries therefore hands the decoder the padding as if it were the
/// next block header, which is exactly the bug this walker was written to fix —
/// archives that VRAMDISK could re-read (it used its own chunk table) but that
/// Windows Explorer and .NET's `ZipArchive` could not open at all.
///
/// Splicing correctly needs one number per chunk that nvCOMP does not report:
/// the bit at which the compressed stream ends. Nothing short of following the
/// block headers *and* every Huffman code recovers it, so that is what this
/// does — it decodes the whole symbol stream but produces no output, keeping
/// neither a window nor the decompressed bytes.
///
/// # Trusting the answer
///
/// A wrong end bit corrupts an archive silently, so the walk is self-checking:
/// it tracks the number of bytes the stream *would* have produced, and the
/// caller compares that against the size the chunk was compressed from. The
/// two can only agree if every code length along the way was right, which is
/// what makes [`DeflateWalk::end_bit`] safe to splice on.
struct DeflateWalker {
    litlen: Vec<u16>,
    dist: Vec<u16>,
    clen: Vec<u16>,
    /// Code lengths for the literal/length and distance alphabets, back to
    /// back, as a dynamic header spells them out (288 + 32 at most).
    lengths: [u8; 320],
}

impl DeflateWalker {
    fn new() -> Self {
        Self {
            litlen: vec![0; DEFLATE_TABLE_LEN],
            dist: vec![0; DEFLATE_TABLE_LEN],
            clen: vec![0; DEFLATE_CLEN_TABLE_LEN],
            lengths: [0; 320],
        }
    }

    /// Walk `data` from its first bit to the end of its final block.
    fn walk(&mut self, data: &[u8]) -> EResult<DeflateWalk> {
        let mut bits = DeflateBits::new(data);
        let mut out_len = 0u64;
        // Assigned on every iteration; the stream is only left through the
        // `BFINAL` break, so the last block's flag is what survives.
        let mut final_bfinal_bit: u64;
        loop {
            final_bfinal_bit = bits.pos;
            let bfinal = bits.take(1)?;
            let btype = bits.take(2)?;
            match btype {
                0 => {
                    bits.align()?;
                    let len = bits.take(16)?;
                    let nlen = bits.take(16)?;
                    if nlen != (!len & 0xffff) {
                        return Err(deflate_error("stored block LEN/NLEN mismatch"));
                    }
                    bits.skip_bytes(u64::from(len))?;
                    out_len += u64::from(len);
                }
                1 => {
                    self.set_fixed_tables()?;
                    out_len += walk_deflate_symbols(&mut bits, &self.litlen, &self.dist)?;
                }
                2 => {
                    self.read_dynamic_tables(&mut bits)?;
                    out_len += walk_deflate_symbols(&mut bits, &self.litlen, &self.dist)?;
                }
                _ => return Err(deflate_error("reserved block type 3")),
            }
            if bfinal == 1 {
                break;
            }
        }
        Ok(DeflateWalk {
            end_bit: bits.pos,
            final_bfinal_bit,
            out_len,
        })
    }

    /// Install the fixed Huffman code of RFC 1951 §3.2.6.
    fn set_fixed_tables(&mut self) -> EResult<()> {
        let mut litlen = [0u8; 288];
        for (sym, l) in litlen.iter_mut().enumerate() {
            *l = match sym {
                0..=143 => 8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            };
        }
        build_deflate_table(&litlen, DEFLATE_MAX_CODE_BITS, &mut self.litlen)?;
        build_deflate_table(&[5u8; 32], DEFLATE_MAX_CODE_BITS, &mut self.dist)
    }

    /// Read a dynamic block's header and install the codes it describes.
    fn read_dynamic_tables(&mut self, bits: &mut DeflateBits<'_>) -> EResult<()> {
        let hlit = bits.take(5)? as usize + 257;
        let hdist = bits.take(5)? as usize + 1;
        let hclen = bits.take(4)? as usize + 4;
        if hlit > 288 || hdist > 32 {
            return Err(deflate_error("dynamic header alphabet is too large"));
        }
        let mut clen_lengths = [0u8; 19];
        for &slot in DEFLATE_CLEN_ORDER.iter().take(hclen) {
            clen_lengths[slot as usize] = bits.take(3)? as u8;
        }
        build_deflate_table(&clen_lengths, 7, &mut self.clen)?;

        let total = hlit + hdist;
        // Disjoint field borrows: the code-length code is read while the
        // alphabet it describes is written.
        let clen = &self.clen;
        let lengths = &mut self.lengths[..total];
        lengths.fill(0);
        let mut i = 0usize;
        while i < total {
            let sym = bits.decode(clen)?;
            let (repeat, value) = match sym {
                0..=15 => {
                    lengths[i] = sym as u8;
                    i += 1;
                    continue;
                }
                16 => {
                    if i == 0 {
                        return Err(deflate_error("code-length repeat with nothing to repeat"));
                    }
                    (3 + bits.take(2)? as usize, lengths[i - 1])
                }
                17 => (3 + bits.take(3)? as usize, 0u8),
                18 => (11 + bits.take(7)? as usize, 0u8),
                _ => return Err(deflate_error("invalid code-length symbol")),
            };
            if i + repeat > total {
                return Err(deflate_error("code-length repeat overruns the alphabet"));
            }
            lengths[i..i + repeat].fill(value);
            i += repeat;
        }
        build_deflate_table(
            &self.lengths[..hlit],
            DEFLATE_MAX_CODE_BITS,
            &mut self.litlen,
        )?;
        build_deflate_table(
            &self.lengths[hlit..total],
            DEFLATE_MAX_CODE_BITS,
            &mut self.dist,
        )
    }
}

/// Consume one Huffman-coded block's symbols, returning the bytes it emits.
fn walk_deflate_symbols(bits: &mut DeflateBits<'_>, litlen: &[u16], dist: &[u16]) -> EResult<u64> {
    let mut out_len = 0u64;
    loop {
        let sym = bits.decode(litlen)?;
        if sym < 256 {
            out_len += 1;
            continue;
        }
        if sym == 256 {
            return Ok(out_len);
        }
        let idx = sym as usize - 257;
        if idx >= DEFLATE_LENGTH_BASE.len() {
            return Err(deflate_error("literal/length symbol 286 or 287"));
        }
        let extra = u32::from(DEFLATE_LENGTH_EXTRA[idx]);
        let len = u64::from(DEFLATE_LENGTH_BASE[idx]) + u64::from(bits.take(extra)?);
        let dsym = bits.decode(dist)? as usize;
        if dsym >= DEFLATE_DIST_EXTRA.len() {
            return Err(deflate_error("distance symbol 30 or 31"));
        }
        bits.take(u32::from(DEFLATE_DIST_EXTRA[dsym]))?;
        out_len += len;
    }
}

/// Walk a group of compressed chunks, spreading them over the CPU cores.
///
/// The walk is the one part of the ZIP writer that runs on the host, and it is
/// proportional to the number of Huffman symbols in the batch, so leaving it on
/// one core would make it — not the GPU — the writer's bottleneck. Blobs left
/// empty by the caller (the member's final chunk, which is spliced to nothing
/// and so needs no end bit) are skipped, and a chunk whose walk fails comes
/// back as `None` so the caller can fall back to storing the member.
fn walk_deflate_blobs(blobs: &[Vec<u8>]) -> Vec<Option<DeflateWalk>> {
    let mut out: Vec<Option<DeflateWalk>> = vec![None; blobs.len()];
    let workers = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(blobs.len())
        .max(1);
    if workers == 1 {
        let mut walker = DeflateWalker::new();
        for (blob, slot) in blobs.iter().zip(out.iter_mut()) {
            if !blob.is_empty() {
                *slot = walker.walk(blob).ok();
            }
        }
        return out;
    }
    let per = blobs.len().div_ceil(workers);
    thread::scope(|scope| {
        for (src, dst) in blobs.chunks(per).zip(out.chunks_mut(per)) {
            scope.spawn(move || {
                let mut walker = DeflateWalker::new();
                for (blob, slot) in src.iter().zip(dst.iter_mut()) {
                    if !blob.is_empty() {
                        *slot = walker.walk(blob).ok();
                    }
                }
            });
        }
    });
    out
}

/// The bytes that turn a chunk boundary into a byte boundary.
///
/// `end_bit` is where the preceding chunk's stream stopped. The decoder resumes
/// there, so what it must find is an *empty stored block*: a `BFINAL=0`,
/// `BTYPE=00` header, then padding to the next byte boundary, then `LEN=0` and
/// `NLEN=0xffff`. Every bit of that header is a zero and the encoder already
/// zero-padded the tail of its last byte, so the whole joiner is four bytes of
/// `00 00 ff ff` — *provided* the three header bits still fit in that byte.
/// When fewer than three bits are spare (including the case where the stream
/// happened to end exactly on a byte boundary) the header spills into a byte of
/// its own and the joiner is five bytes instead. Getting this choice wrong by
/// one is precisely the failure the walker exists to prevent: the decoder would
/// read the following chunk's first bytes as a stored block's LEN/NLEN.
fn zip_deflate_joiner(end_bit: u64) -> &'static [u8] {
    let used = (end_bit % 8) as u32;
    let spare = if used == 0 { 0 } else { 8 - used };
    if spare >= 3 {
        &[0x00, 0x00, 0xff, 0xff]
    } else {
        &[0x00, 0x00, 0x00, 0xff, 0xff]
    }
}

/// The five bytes that terminate a spliced chunk when it is read back on its
/// own: a `BFINAL=1`, `BTYPE=00`, `LEN=0` stored block.
///
/// A non-final chunk written by [`StorageEngine::write_zip_deflate_payload`]
/// ends with the joiner above, which leaves the stream byte-aligned and still
/// open, so the extractor can close it by appending a whole byte-aligned block
/// rather than having to hunt for the `BFINAL` bit it cleared.
const ZIP_DEFLATE_TERMINATOR: [u8; 5] = [0x01, 0x00, 0x00, 0xff, 0xff];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_archive_output_joins_under_base() {
        assert_eq!(
            join_archive_output("\\out", "a.txt").unwrap(),
            "\\out\\a.txt"
        );
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

    fn engine_with_hash_budget(mib: u64, dedup: bool, budget: u64) -> StorageEngine {
        let mut e = engine(mib, dedup);
        e.set_gpu_hash_launch_budget(budget);
        e
    }

    fn engine_compress(mib: u64) -> StorageEngine {
        let vram = Vram::new(0, mib * 1024 * 1024).expect("alloc vram for test");
        StorageEngine::new(vram, true, false).expect("compress engine (nvCOMP)")
    }

    fn archive_engine(mib: u64) -> StorageEngine {
        let vram = Vram::new(0, mib * 1024 * 1024).expect("alloc vram for archive test");
        StorageEngine::new(vram, false, false).expect("archive engine")
    }

    fn write_archive_fixture(e: &mut StorageEngine) -> Vec<(String, Vec<u8>)> {
        e.table_mut().create_dir("\\data", 0).unwrap();
        let specs = [
            ("\\data\\a.bin", vec![b'A'; CHUNK_SIZE as usize * 2]),
            (
                "\\data\\b.bin",
                (0..(CHUNK_SIZE as usize + 12_345))
                    .map(|i| ((i / 251) & 0xff) as u8)
                    .collect(),
            ),
            ("\\data\\c.bin", vec![b'Z'; 8192]),
        ];
        let mut originals = Vec::with_capacity(specs.len());
        for (path, data) in specs {
            e.table_mut().create_file(path, 0).unwrap();
            e.write(path, 0, &data).unwrap();
            originals.push((path.to_string(), data));
        }
        originals
    }

    fn assert_archive_tree(e: &mut StorageEngine, base: &str, originals: &[(String, Vec<u8>)]) {
        for (path, expected) in originals {
            let rel = path.trim_start_matches('\\');
            let got = e
                .read(&format!("{base}\\{rel}"), 0, expected.len())
                .unwrap();
            assert_eq!(got, *expected, "archive roundtrip mismatch for {path}");
        }
    }

    fn assert_first_chunk_codec(e: &StorageEngine, path: &str, codec: Codec) {
        assert!(matches!(
            e.coord(path, 0),
            Some(Placement::Compressed {
                codec: got, ..
            }) if got == codec
        ));
    }

    fn force_file_zstd_compressed(e: &mut StorageEngine, path: &str) {
        let size = e.get(path).unwrap().size;
        assert!(
            size <= CHUNK_SIZE,
            "test fixture expects a single archive chunk"
        );
        let original = e.read(path, 0, size as usize).unwrap();
        for lc in 0..logical_chunks(size) {
            let chunk_start = lc * CHUNK_SIZE as usize;
            let chunk_end = ((lc + 1) * CHUNK_SIZE as usize).min(original.len());
            let mut full = vec![0u8; CHUNK_SIZE as usize];
            if chunk_start < chunk_end {
                full[..chunk_end - chunk_start].copy_from_slice(&original[chunk_start..chunk_end]);
            }
            let compressed = zstd::encode_all(full.as_slice(), 3).unwrap();
            assert!(
                compressed.len() < CHUNK_SIZE as usize,
                "test fixture archive chunk should fit in compressed storage"
            );
            let old = e.coord(path, lc);
            e.place_chunk(path, lc, &full, old, Some((compressed, Codec::Zstd)), None)
                .unwrap();
        }
        e.set_size(path, size).unwrap();
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

    #[test]
    #[ignore]
    fn archive_jobs_roundtrip_lz4_source_placements() {
        let mut e = engine_compress(512);
        let originals = write_archive_fixture(&mut e);
        assert_first_chunk_codec(&e, "\\data\\a.bin", Codec::Lz4);
        assert_first_chunk_codec(&e, "\\data\\b.bin", Codec::Lz4);
        let paths: Vec<String> = originals.iter().map(|(path, _)| path.clone()).collect();
        for (codec, archive, out_dir) in [
            (
                NvcompFrameCodec::Zstd,
                "\\lz4-src.tar.zst",
                "\\lz4-zstd-out",
            ),
            (NvcompFrameCodec::Deflate, "\\lz4-src.zip", "\\lz4-zip-out"),
        ] {
            e.archive_compress_gpu(codec, &paths, archive).unwrap();
            e.archive_extract_gpu(codec, archive, out_dir).unwrap();
            assert_archive_tree(&mut e, out_dir, &originals);
        }
    }

    #[test]
    #[ignore]
    fn archive_jobs_roundtrip_zstd_source_placements() {
        let mut e = engine_compress(512);
        e.codec = None;
        let originals = write_archive_fixture(&mut e);
        assert_first_chunk_codec(&e, "\\data\\a.bin", Codec::Zstd);
        assert_first_chunk_codec(&e, "\\data\\b.bin", Codec::Zstd);
        let paths: Vec<String> = originals.iter().map(|(path, _)| path.clone()).collect();
        for (codec, archive, out_dir) in [
            (
                NvcompFrameCodec::Zstd,
                "\\zstd-src.tar.zst",
                "\\zstd-zstd-out",
            ),
            (
                NvcompFrameCodec::Deflate,
                "\\zstd-src.zip",
                "\\zstd-zip-out",
            ),
        ] {
            e.archive_compress_gpu(codec, &paths, archive).unwrap();
            e.archive_extract_gpu(codec, archive, out_dir).unwrap();
            assert_archive_tree(&mut e, out_dir, &originals);
        }
    }

    #[test]
    #[ignore]
    fn archive_extract_accepts_compressed_archive_file() {
        let mut e = engine_compress(512);
        let originals = write_archive_fixture(&mut e);
        let paths: Vec<String> = originals.iter().map(|(path, _)| path.clone()).collect();
        let stats = e
            .archive_compress_gpu(NvcompFrameCodec::Zstd, &paths, "\\packed.tar.zst")
            .unwrap();
        assert!(stats.archive_bytes > 0);
        force_file_zstd_compressed(&mut e, "\\packed.tar.zst");
        assert_first_chunk_codec(&e, "\\packed.tar.zst", Codec::Zstd);
        e.archive_extract_gpu(NvcompFrameCodec::Zstd, "\\packed.tar.zst", "\\packed-out")
            .unwrap();
        assert_archive_tree(&mut e, "\\packed-out", &originals);
    }

    fn engine_compress_dedup(mib: u64) -> StorageEngine {
        let vram = Vram::new(0, mib * 1024 * 1024).expect("alloc vram for test");
        StorageEngine::new(vram, true, true).expect("compress+dedup engine (nvCOMP)")
    }

    fn patterned_bytes(len: usize, seed: u32) -> Vec<u8> {
        (0..len)
            .map(|i| {
                let x = i as u32;
                x.wrapping_mul(37)
                    .wrapping_add((x >> 3) * 11)
                    .wrapping_add(seed) as u8
            })
            .collect()
    }

    fn hash_reference(alg: HashAlgorithm, data: &[u8]) -> Vec<u8> {
        match alg {
            HashAlgorithm::Md5 => {
                let mut hasher = Md5::new();
                hasher.update(data);
                hasher.finalize().to_vec()
            }
            HashAlgorithm::Sha1 => {
                let mut hasher = Sha1::new();
                hasher.update(data);
                hasher.finalize().to_vec()
            }
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(data);
                hasher.finalize().to_vec()
            }
            HashAlgorithm::Fnv1a64 => {
                let mut state = FNV1A64_OFFSET_BASIS;
                for &b in data {
                    state ^= b as u64;
                    state = state.wrapping_mul(FNV1A64_PRIME);
                }
                state.to_be_bytes().to_vec()
            }
        }
    }

    // -----------------------------------------------------------------------
    // DEFLATE walker
    // -----------------------------------------------------------------------

    /// splitmix64, so the randomized walker tests are reproducible.
    struct TestRng(u64);

    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Writes DEFLATE's LSB-first bit packing and keeps an exact bit count, so
    /// a test knows the very answer [`DeflateWalker`] has to come back with.
    struct TestBitWriter {
        out: Vec<u8>,
        bit: u64,
    }

    impl TestBitWriter {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                bit: 0,
            }
        }

        /// Append the low `bits` bits of `value`, least significant first.
        fn put(&mut self, value: u32, bits: u32) {
            for i in 0..bits {
                if self.bit % 8 == 0 {
                    self.out.push(0);
                }
                let idx = (self.bit / 8) as usize;
                self.out[idx] |= (((value >> i) & 1) as u8) << (self.bit % 8);
                self.bit += 1;
            }
        }

        /// Append a Huffman code, which unlike everything else goes out most
        /// significant bit first.
        fn put_code(&mut self, code: u32, len: u32) {
            for i in (0..len).rev() {
                self.put((code >> i) & 1, 1);
            }
        }

        fn align(&mut self) {
            while self.bit % 8 != 0 {
                self.put(0, 1);
            }
        }
    }

    /// Canonical code for every symbol, matching what [`build_deflate_table`]
    /// expects to decode.
    fn canonical_codes(lengths: &[u8]) -> Vec<u32> {
        let mut count = [0u32; 16];
        for &l in lengths {
            if l > 0 {
                count[l as usize] += 1;
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for l in 1..16 {
            code = (code + count[l - 1]) << 1;
            next[l] = code;
        }
        let mut out = vec![0u32; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                out[sym] = next[l as usize];
                next[l as usize] += 1;
            }
        }
        out
    }

    /// `n` code lengths whose Kraft sum is exactly 1, i.e. a complete code.
    ///
    /// Built by repeatedly splitting a leaf in two, which preserves the sum by
    /// construction and reaches every shape a real encoder could produce.
    fn random_complete_lengths(rng: &mut TestRng, n: usize, max: u8) -> Vec<u8> {
        assert!(n >= 1);
        if n == 1 {
            // The one incomplete code DEFLATE explicitly allows.
            return vec![1];
        }
        let mut lengths = vec![0u8; n];
        lengths[0] = 1;
        lengths[1] = 1;
        for i in 2..n {
            let start = rng.below(i);
            let mut j = start;
            loop {
                if lengths[j] < max {
                    lengths[j] += 1;
                    lengths[i] = lengths[j];
                    break;
                }
                j = (j + 1) % i;
                assert_ne!(j, start, "no room left under a depth limit of {max}");
            }
        }
        // Shuffle, so the canonical assignment is not always in tree order.
        for i in (1..n).rev() {
            let j = rng.below(i + 1);
            lengths.swap(i, j);
        }
        lengths
    }

    /// Scatter a complete code over `slots` symbols, leaving the rest unused
    /// (length 0) so dynamic headers have zero runs to encode with 17/18.
    /// `required` is a symbol that must end up with a code.
    fn scatter_lengths(rng: &mut TestRng, slots: usize, used: usize, required: usize) -> Vec<u8> {
        let used = used.clamp(1, slots);
        let lengths = random_complete_lengths(rng, used, 15);
        let mut positions: Vec<usize> = (0..slots).collect();
        for i in (1..slots).rev() {
            let j = rng.below(i + 1);
            positions.swap(i, j);
        }
        if !positions[..used].contains(&required) {
            positions[0] = required;
        }
        let mut out = vec![0u8; slots];
        for (slot, len) in positions[..used].iter().zip(lengths) {
            out[*slot] = len;
        }
        out
    }

    /// The fixed literal/length code lengths of RFC 1951 §3.2.6.
    fn fixed_litlen_lengths() -> Vec<u8> {
        (0..288usize)
            .map(|sym| match sym {
                0..=143 => 8u8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            })
            .collect()
    }

    /// Emit the symbol stream of a Huffman-coded block, ending with symbol 256.
    /// Returns how many bytes it decodes to.
    fn emit_symbols(
        rng: &mut TestRng,
        w: &mut TestBitWriter,
        litlen: &[u8],
        dist: &[u8],
        count: usize,
    ) -> u64 {
        let lit_codes = canonical_codes(litlen);
        let dist_codes = canonical_codes(dist);
        let literals: Vec<usize> = (0..256).filter(|&s| litlen[s] > 0).collect();
        let lengths: Vec<usize> = (257..litlen.len().min(286))
            .filter(|&s| litlen[s] > 0)
            .collect();
        let dists: Vec<usize> = (0..dist.len().min(30)).filter(|&s| dist[s] > 0).collect();
        let mut out_len = 0u64;
        for _ in 0..count {
            let want_match = !lengths.is_empty() && !dists.is_empty() && rng.below(3) == 0;
            if !want_match {
                if literals.is_empty() {
                    continue;
                }
                let sym = literals[rng.below(literals.len())];
                w.put_code(lit_codes[sym], u32::from(litlen[sym]));
                out_len += 1;
                continue;
            }
            let sym = lengths[rng.below(lengths.len())];
            w.put_code(lit_codes[sym], u32::from(litlen[sym]));
            let idx = sym - 257;
            let extra = u32::from(DEFLATE_LENGTH_EXTRA[idx]);
            let extra_bits = if extra == 0 {
                0
            } else {
                rng.below(1 << extra) as u32
            };
            w.put(extra_bits, extra);
            out_len += u64::from(DEFLATE_LENGTH_BASE[idx]) + u64::from(extra_bits);
            let dsym = dists[rng.below(dists.len())];
            w.put_code(dist_codes[dsym], u32::from(dist[dsym]));
            let dextra = u32::from(DEFLATE_DIST_EXTRA[dsym]);
            let dextra_bits = if dextra == 0 {
                0
            } else {
                rng.below(1 << dextra) as u32
            };
            w.put(dextra_bits, dextra);
        }
        w.put_code(lit_codes[256], u32::from(litlen[256]));
        out_len
    }

    /// Write a dynamic block's header, spelling the two alphabets out through
    /// the code-length alphabet (with 16/17/18 runs where they apply).
    fn emit_dynamic_header(rng: &mut TestRng, w: &mut TestBitWriter, litlen: &[u8], dist: &[u8]) {
        w.put((litlen.len() - 257) as u32, 5);
        w.put((dist.len() - 1) as u32, 5);
        w.put(19 - 4, 4);
        let clen_lengths = random_complete_lengths(rng, 19, 7);
        let clen_codes = canonical_codes(&clen_lengths);
        for &slot in DEFLATE_CLEN_ORDER.iter() {
            w.put(u32::from(clen_lengths[slot as usize]), 3);
        }
        let all: Vec<u8> = litlen.iter().chain(dist.iter()).copied().collect();
        let mut i = 0usize;
        while i < all.len() {
            let value = all[i];
            let mut run = 1usize;
            while i + run < all.len() && all[i + run] == value {
                run += 1;
            }
            let emit = |w: &mut TestBitWriter, sym: usize| {
                w.put_code(clen_codes[sym], u32::from(clen_lengths[sym]));
            };
            if value == 0 && run >= 11 {
                let take = run.min(138);
                emit(w, 18);
                w.put((take - 11) as u32, 7);
                i += take;
            } else if value == 0 && run >= 3 {
                let take = run.min(10);
                emit(w, 17);
                w.put((take - 3) as u32, 3);
                i += take;
            } else if run >= 4 {
                emit(w, value as usize);
                let take = (run - 1).min(6);
                emit(w, 16);
                w.put((take - 3) as u32, 2);
                i += 1 + take;
            } else {
                emit(w, value as usize);
                i += 1;
            }
        }
    }

    /// Emit one random block (header included) and return the bytes it decodes
    /// to. `bfinal` decides whether the stream ends here.
    fn emit_random_block(rng: &mut TestRng, w: &mut TestBitWriter, bfinal: bool) -> u64 {
        let btype = rng.below(3) as u32;
        w.put(u32::from(bfinal), 1);
        w.put(btype, 2);
        match btype {
            0 => {
                w.align();
                let len = rng.below(300);
                w.put(len as u32, 16);
                w.put(!(len as u32) & 0xffff, 16);
                for _ in 0..len {
                    w.put(rng.below(256) as u32, 8);
                }
                len as u64
            }
            1 => {
                let litlen = fixed_litlen_lengths();
                let dist = vec![5u8; 32];
                let count = 1 + rng.below(400);
                emit_symbols(rng, w, &litlen, &dist, count)
            }
            _ => {
                let hlit = 257 + rng.below(32);
                let hdist = 1 + rng.below(30);
                let lit_used = 2 + rng.below(hlit - 1);
                let litlen = scatter_lengths(rng, hlit, lit_used, 256);
                let dist = if hdist == 1 {
                    vec![1u8]
                } else {
                    let dist_used = 1 + rng.below(hdist);
                    scatter_lengths(rng, hdist, dist_used, 0)
                };
                emit_dynamic_header(rng, w, &litlen, &dist);
                let count = 1 + rng.below(400);
                emit_symbols(rng, w, &litlen, &dist, count)
            }
        }
    }

    /// One synthetic stream plus the answers the walker must reproduce.
    struct TestStream {
        bytes: Vec<u8>,
        end_bit: u64,
        final_bfinal_bit: u64,
        out_len: u64,
    }

    fn random_stream(rng: &mut TestRng, blocks: usize) -> TestStream {
        let mut w = TestBitWriter::new();
        let mut out_len = 0u64;
        let mut final_bfinal_bit = 0u64;
        for b in 0..blocks {
            final_bfinal_bit = w.bit;
            out_len += emit_random_block(rng, &mut w, b + 1 == blocks);
        }
        let end_bit = w.bit;
        // Encoders zero-pad the tail of the last byte, and so does this.
        w.align();
        TestStream {
            bytes: w.out,
            end_bit,
            final_bfinal_bit,
            out_len,
        }
    }

    #[test]
    fn deflate_walker_matches_synthetic_streams() {
        let mut rng = TestRng(0x5EED_1234_ABCD_0001);
        let mut walker = DeflateWalker::new();
        for case in 0..600 {
            let blocks = 1 + case % 4;
            let stream = random_stream(&mut rng, blocks);
            let walk = walker
                .walk(&stream.bytes)
                .unwrap_or_else(|e| panic!("case {case} ({blocks} blocks) failed to walk: {e:?}"));
            assert_eq!(walk.end_bit, stream.end_bit, "case {case}: end bit");
            assert_eq!(
                walk.final_bfinal_bit, stream.final_bfinal_bit,
                "case {case}: final BFINAL bit"
            );
            assert_eq!(walk.out_len, stream.out_len, "case {case}: output length");
            assert!(
                stream.bytes.len() as u64 * 8 - walk.end_bit < 8,
                "case {case}: the walk must land inside the last byte"
            );
        }
    }

    #[test]
    fn deflate_walker_splices_streams_byte_aligned() {
        let mut rng = TestRng(0x5EED_1234_ABCD_0002);
        let mut walker = DeflateWalker::new();
        let mut four_byte = 0usize;
        let mut five_byte = 0usize;
        for case in 0..400 {
            let head = random_stream(&mut rng, 1 + case % 3);
            let tail = random_stream(&mut rng, 1 + (case + 1) % 3);
            let head_walk = walker.walk(&head.bytes).expect("head walks");

            // Exactly what `write_zip_deflate_chunks` does to a non-final chunk.
            let used = head_walk.end_bit.div_ceil(8) as usize;
            let mut joined = head.bytes[..used].to_vec();
            let bfinal_byte = (head_walk.final_bfinal_bit / 8) as usize;
            joined[bfinal_byte] &= !(1u8 << (head_walk.final_bfinal_bit % 8));
            let rem = (head_walk.end_bit % 8) as u32;
            if rem != 0 {
                joined[used - 1] &= ((1u16 << rem) - 1) as u8;
            }
            let joiner = zip_deflate_joiner(head_walk.end_bit);
            if joiner.len() == 4 {
                four_byte += 1;
            } else {
                five_byte += 1;
            }
            let prefix = used + joiner.len();
            joined.extend_from_slice(joiner);
            joined.extend_from_slice(&tail.bytes);

            let walk = walker
                .walk(&joined)
                .unwrap_or_else(|e| panic!("case {case}: spliced stream does not walk: {e:?}"));
            assert_eq!(
                walk.out_len,
                head.out_len + tail.out_len,
                "case {case}: spliced output length"
            );
            assert_eq!(
                walk.end_bit,
                prefix as u64 * 8 + tail.end_bit,
                "case {case}: spliced end bit"
            );
            assert_eq!(
                walk.final_bfinal_bit,
                prefix as u64 * 8 + tail.final_bfinal_bit,
                "case {case}: spliced final BFINAL bit"
            );
        }
        // Both joiner lengths have to be exercised, or the test proves nothing
        // about the alignment decision.
        assert!(four_byte > 20, "only {four_byte} four-byte joiners");
        assert!(five_byte > 20, "only {five_byte} five-byte joiners");
    }

    #[test]
    fn deflate_walker_handles_terminator_and_empty_stored_blocks() {
        let mut walker = DeflateWalker::new();
        let walk = walker.walk(&ZIP_DEFLATE_TERMINATOR).expect("terminator");
        assert_eq!(walk.out_len, 0);
        assert_eq!(walk.end_bit, 5 * 8);
        assert_eq!(walk.final_bfinal_bit, 0);

        // A non-final empty stored block followed by the terminator: the exact
        // shape a spliced chunk is closed with on the extract path.
        let mut bytes = vec![0x00, 0x00, 0x00, 0xff, 0xff];
        bytes.extend_from_slice(&ZIP_DEFLATE_TERMINATOR);
        let walk = walker.walk(&bytes).expect("joined empty blocks");
        assert_eq!(walk.out_len, 0);
        assert_eq!(walk.end_bit, 10 * 8);
        assert_eq!(walk.final_bfinal_bit, 5 * 8);
    }

    #[test]
    fn deflate_joiner_lengths_follow_the_spare_bits() {
        for used in 0..8u64 {
            let joiner = zip_deflate_joiner(64 + used);
            let spare = if used == 0 { 0 } else { 8 - used };
            let want = if spare >= 3 { 4 } else { 5 };
            assert_eq!(joiner.len(), want, "{used} bits used in the last byte");
            assert_eq!(joiner[joiner.len() - 2..], [0xff, 0xff]);
            assert!(joiner[..joiner.len() - 2].iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn deflate_walker_rejects_malformed_streams() {
        let mut walker = DeflateWalker::new();
        assert!(walker.walk(&[]).is_err(), "empty input");
        assert!(walker.walk(&[0x07]).is_err(), "reserved block type 3");
        // Final stored block whose NLEN does not complement LEN.
        assert!(
            walker.walk(&[0x01, 0x05, 0x00, 0x00, 0x00]).is_err(),
            "bad NLEN"
        );
        // Final stored block claiming more bytes than are present.
        assert!(
            walker.walk(&[0x01, 0x05, 0x00, 0xfa, 0xff, 0x01]).is_err(),
            "stored block overruns"
        );
        // A fixed-Huffman block that never reaches its end-of-block symbol.
        assert!(walker.walk(&[0x03]).is_err(), "truncated fixed block");
        // Truncated dynamic header.
        assert!(
            walker.walk(&[0x05, 0x00]).is_err(),
            "truncated dynamic header"
        );
    }

    #[test]
    fn deflate_table_rejects_over_subscribed_codes() {
        let mut table = vec![0u16; DEFLATE_TABLE_LEN];
        assert!(build_deflate_table(&[1, 1, 1], 15, &mut table).is_err());
        assert!(build_deflate_table(&[1, 1], 15, &mut table).is_ok());
        // Incomplete codes are legal; the unreachable half must stay undefined.
        assert!(build_deflate_table(&[1], 15, &mut table).is_ok());
        assert_eq!(table[0] & 0xf, 1, "the single one-bit code is decodable");
        assert_eq!(table[1] & 0xf, 0, "its complement is not");
    }

    /// The end-to-end proof that the splice is right: take real nvCOMP Deflate
    /// chunks, join them the way the ZIP writer does, and check that the result
    /// is one stream that inflates back to the original bytes.
    ///
    /// The synthetic tests above pin the walker down against streams whose
    /// answer is known by construction; this one pins it against the encoder
    /// whose output actually ships, including the padding it leaves behind.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn deflate_splice_of_nvcomp_chunks_roundtrips() {
        let mib = 1024 * 1024usize;
        let chunks = 8usize;
        let mut vram = Vram::new(0, 64 * 1024 * 1024).expect("vram");
        let mut codec =
            NvcompBatchedCodec::load(&vram, NvcompFrameCodec::Deflate).expect("nvcomp deflate");
        let base = vram.buf_device_ptr();

        // Runs, pseudo-random, mixed and text-like payloads, plus a short final
        // chunk — every shape a real member's chunks come in.
        let mut rng = TestRng(0xC0FF_EE00_1234_5678);
        let sizes: Vec<u64> = (0..chunks)
            .map(|k| if k + 1 == chunks { mib / 3 } else { mib } as u64)
            .collect();
        let mut source: Vec<u8> = Vec::new();
        for (k, &take) in sizes.iter().enumerate() {
            for i in 0..take as usize {
                source.push(match k % 4 {
                    0 => (i / 64) as u8,
                    1 => rng.next_u64() as u8,
                    2 => {
                        if i % 11 == 0 {
                            rng.next_u64() as u8
                        } else {
                            (i % 250) as u8
                        }
                    }
                    _ => b'a' + ((i / (2 + k)) % 26) as u8,
                });
            }
        }
        vram.write_at(0, &source).expect("stage source");
        vram.sync().expect("sync");
        let mut ptrs = Vec::with_capacity(chunks);
        let mut off = 0u64;
        for &s in &sizes {
            ptrs.push(base + off);
            off += s;
        }
        let comp = codec.compress_device(&ptrs, &sizes).expect("compress");

        let mut walker = DeflateWalker::new();
        let mut spliced: Vec<u8> = Vec::new();
        for (i, &clen) in comp.iter().enumerate() {
            let mut blob = vec![0u8; clen as usize];
            codec
                .copy_compressed_slot_to_host(i, &mut blob)
                .expect("compressed slot to host");
            if i + 1 == chunks {
                spliced.extend_from_slice(&blob);
                continue;
            }
            let walk = walker
                .walk(&blob)
                .unwrap_or_else(|e| panic!("nvCOMP chunk {i} does not walk: {e:?}"));
            assert_eq!(
                walk.out_len, sizes[i],
                "chunk {i} decodes to its input size"
            );
            assert!(
                blob.len() as u64 * 8 - walk.end_bit < 8,
                "chunk {i}: end bit is not inside the last byte"
            );
            let used = walk.end_bit.div_ceil(8) as usize;
            let rem = (walk.end_bit % 8) as u32;
            if rem != 0 {
                assert_eq!(
                    blob[used - 1] >> rem,
                    0,
                    "chunk {i}: nvCOMP left non-zero padding after the stream"
                );
            }
            blob.truncate(used);
            let bfinal_byte = (walk.final_bfinal_bit / 8) as usize;
            blob[bfinal_byte] &= !(1u8 << (walk.final_bfinal_bit % 8));
            spliced.extend_from_slice(&blob);
            spliced.extend_from_slice(zip_deflate_joiner(walk.end_bit));
        }

        let whole = walker.walk(&spliced).expect("spliced stream walks");
        assert_eq!(
            whole.out_len,
            source.len() as u64,
            "spliced stream accounts for the whole member"
        );

        let stage = 16 * 1024 * 1024u64;
        let out = 32 * 1024 * 1024u64;
        vram.write_at(stage, &spliced).expect("stage spliced");
        vram.sync().expect("sync");
        let produced = codec
            .decompress_device(
                &[base + stage],
                &[spliced.len() as u64],
                &[base + out],
                &[source.len() as u64],
            )
            .expect("decompress spliced");
        assert_eq!(produced, vec![source.len() as u64]);
        let mut back = vec![0u8; source.len()];
        vram.read_at(out, &mut back).expect("read back");
        assert!(back == source, "spliced stream does not round-trip");
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn write_read_roundtrip_small() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\a", 0).unwrap();
        let data = b"hello VRAMDISK";
        assert_eq!(e.write("\\a", 0, data).unwrap(), data.len() as u64);
        assert_eq!(e.get("\\a").unwrap().size, data.len() as u64);
        assert_eq!(e.read("\\a", 0, 1024).unwrap(), data);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn gpu_api_hash_small_budget_matches_known_digests() {
        let mut e = engine_with_hash_budget(8, false, 64 * 1024);
        e.table_mut().create_file("\\multi_a", 0).unwrap();
        let file1: Vec<u8> = (0..(CHUNK_SIZE as usize * 3 + 123))
            .map(|i| ((i * 17) + (i / 512) + 3) as u8)
            .collect();
        e.write("\\multi_a", 0, &file1).unwrap();

        e.table_mut().create_file("\\multi_b", 0).unwrap();
        e.set_size("\\multi_b", CHUNK_SIZE * 2 + 4096 + 17).unwrap();
        let part1: Vec<u8> = (0..7000).map(|i| ((i * 29) + 11) as u8).collect();
        e.write("\\multi_b", 8192, &part1).unwrap();
        let part2: Vec<u8> = (0..3000).map(|i| ((i * 7) + 5) as u8).collect();
        let tail_off = CHUNK_SIZE * 2 + 4096 + 17 - part2.len() as u64;
        e.write("\\multi_b", tail_off, &part2).unwrap();

        let expected = [
            (
                "\\multi_a",
                "52d82b3d1e16800efbe069a25e9d8869",
                "01f386977fa106de35ab560a750df36a7a453e2c",
                "f59cb9d9ab6d6f82efb85f5e7ce6e424a8db21a342b8623f77ba546e6296ec6c",
            ),
            (
                "\\multi_b",
                "9d662de4fe8a49a2f1eb727e7d891645",
                "cb9ad859560b033f7d3810cf0cd5fbdf3b7a30bd",
                "087746a4b3aa9a38a52802dfecfd99cd85a94a88f8fe0d7381772b4ae9569bd8",
            ),
        ];

        for &(path, md5, sha1, sha256) in &expected {
            assert_eq!(
                crate::api_kernel::digest_hex(&e.hash_file_gpu(path, HashAlgorithm::Md5).unwrap()),
                md5
            );
            assert_eq!(
                crate::api_kernel::digest_hex(&e.hash_file_gpu(path, HashAlgorithm::Sha1).unwrap()),
                sha1
            );
            assert_eq!(
                crate::api_kernel::digest_hex(
                    &e.hash_file_gpu(path, HashAlgorithm::Sha256).unwrap()
                ),
                sha256
            );
        }

        let paths = vec!["\\multi_a".to_string(), "\\multi_b".to_string()];
        let md5 = e.hash_files_gpu_many(&paths, HashAlgorithm::Md5).unwrap();
        let sha1 = e.hash_files_gpu_many(&paths, HashAlgorithm::Sha1).unwrap();
        let sha256 = e
            .hash_files_gpu_many(&paths, HashAlgorithm::Sha256)
            .unwrap();
        assert_eq!(crate::api_kernel::digest_hex(&md5[0]), expected[0].1);
        assert_eq!(crate::api_kernel::digest_hex(&md5[1]), expected[1].1);
        assert_eq!(crate::api_kernel::digest_hex(&sha1[0]), expected[0].2);
        assert_eq!(crate::api_kernel::digest_hex(&sha1[1]), expected[1].2);
        assert_eq!(crate::api_kernel::digest_hex(&sha256[0]), expected[0].3);
        assert_eq!(crate::api_kernel::digest_hex(&sha256[1]), expected[1].3);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn cpu_and_gpu_hash_digests_match_all_algorithms() {
        let mut e = engine(16, false);
        let path = "\\hash-fixture";
        let data = patterned_bytes(CHUNK_SIZE as usize * 2 + 12_345, 17);
        e.table_mut().create_file(path, 0).unwrap();
        e.write(path, 0, &data).unwrap();

        for alg in [
            HashAlgorithm::Md5,
            HashAlgorithm::Sha1,
            HashAlgorithm::Sha256,
            HashAlgorithm::Fnv1a64,
        ] {
            let cpu = e.hash_file_cpu(path, alg).unwrap();
            let gpu = e.hash_file_gpu(path, alg).unwrap();
            let reference = hash_reference(alg, &data);
            assert_eq!(cpu, gpu, "CPU/GPU digest mismatch for {}", alg.name());
            assert_eq!(
                cpu,
                reference,
                "reference digest mismatch for {}",
                alg.name()
            );
        }
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn routed_large_file_hash_uses_cpu_and_matches_rustcrypto() {
        let mut e = engine(16, false);
        e.hash_cpu_route_threshold = 1_048_576; // 1 MiB
        e.hash_cpu_route_threshold_override = true;

        let path = "\\large-hash";
        let data = patterned_bytes(5 * 1024 * 1024 + 321, 99);
        e.table_mut().create_file(path, 0).unwrap();
        e.write(path, 0, &data).unwrap();

        assert!(!e.should_hash_on_gpu(path).unwrap());
        let digest = e.hash_file(path, HashAlgorithm::Sha256).unwrap();
        assert_eq!(digest, hash_reference(HashAlgorithm::Sha256, &data));
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn routed_hash_succeeds_for_cpu_zstd_fallback_chunks() {
        let mut e = engine_compress(16);
        e.codec = None;

        let path = "\\zstd-hash";
        let data = vec![b'A'; CHUNK_SIZE as usize * 2];
        e.table_mut().create_file(path, 0).unwrap();
        e.write(path, 0, &data).unwrap();

        assert!(matches!(
            e.coord(path, 0),
            Some(Placement::Compressed {
                codec: Codec::Zstd,
                ..
            })
        ));
        assert!(!e.file_supports_gpu_hash(path).unwrap());

        let digest = e.hash_file(path, HashAlgorithm::Sha256).unwrap();
        assert_eq!(digest, hash_reference(HashAlgorithm::Sha256, &data));
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn read_clamps_to_eof() {
        let mut e = engine(1, false);
        e.table_mut().create_file("\\c", 0).unwrap();
        e.write("\\c", 0, b"abcdef").unwrap();
        assert_eq!(e.read("\\c", 4, 100).unwrap(), b"ef");
        assert!(e.read("\\c", 6, 100).unwrap().is_empty());
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
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

    // ---- dedup candidate verification --------------------------------------
    //
    // The dedup index is keyed by a two-level FNV-1a 64-bit hash, which is not
    // collision resistant: collisions can be constructed on purpose. These
    // tests forge the situation an attacker would engineer — the index says
    // "this hash lives at that placement" while the placement holds different
    // bytes — and assert that the engine refuses to share, because sharing
    // there means one file silently reading another's data.

    /// Point `hash` at `p` in the dedup index, standing in for an FNV-1a
    /// collision that an attacker constructed. For a compressed candidate the
    /// blob's reverse-map entry is rewritten too, so the hash-only check has
    /// nothing left to notice — only the bytes still disagree.
    fn forge_collision(e: &mut StorageEngine, hash: u64, p: Placement) {
        e.hash_index.insert(hash, p);
        if let Placement::Compressed { offset, .. } = p {
            e.compressed_hash.insert(offset, hash);
        }
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn dedup_verification_rejects_forged_raw_collision() {
        let mut e = engine(4, true);
        let a = chunk_pattern(0x11);
        let b = chunk_pattern(0x22);
        e.table_mut().create_file("\\a", 0).unwrap();
        e.table_mut().create_file("\\b", 0).unwrap();
        e.write("\\a", 0, &a).unwrap();
        e.write("\\b", 0, &b).unwrap();

        let b_place = e.coord("\\b", 0).unwrap();
        assert!(matches!(b_place, Placement::Raw { .. }));
        forge_collision(&mut e, fnv1a(&a), b_place);

        let before = e.used_chunks();
        e.table_mut().create_file("\\c", 0).unwrap();
        e.write("\\c", 0, &a).unwrap();

        assert_eq!(
            e.read("\\c", 0, CHUNK_SIZE as usize).unwrap(),
            a,
            "forged collision aliased \\c onto \\b's chunk"
        );
        assert_eq!(
            e.used_chunks(),
            before + 1,
            "a rejected candidate must be stored in its own chunk"
        );
        assert_eq!(
            e.read("\\b", 0, CHUNK_SIZE as usize).unwrap(),
            b,
            "\\b must be untouched by the rejected share"
        );
        assert_eq!(e.trace_snapshot().dedup_rejected_chunks, 1);
        assert_eq!(
            e.trace_snapshot().dedup_shared_chunks,
            0,
            "a rejected candidate must not be counted as shared"
        );

        // Refcount bookkeeping survives the mismatch: each file still owns
        // exactly one chunk, so deleting them all releases everything.
        e.remove("\\a").unwrap();
        e.remove("\\b").unwrap();
        e.remove("\\c").unwrap();
        assert_eq!(e.used_chunks(), 0, "rejected path leaked a chunk");
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn dedup_verification_rejects_forged_compressed_collision() {
        let mut e = engine_compress_dedup(8);
        let a = vec![b'A'; CHUNK_SIZE as usize];
        let b = vec![b'B'; CHUNK_SIZE as usize];
        e.table_mut().create_file("\\a", 0).unwrap();
        e.table_mut().create_file("\\b", 0).unwrap();
        e.write("\\a", 0, &a).unwrap();
        e.write("\\b", 0, &b).unwrap();

        let b_place = e.coord("\\b", 0).unwrap();
        assert!(
            matches!(b_place, Placement::Compressed { .. }),
            "fixture must produce a compressed candidate, got {b_place:?}"
        );
        forge_collision(&mut e, fnv1a(&a), b_place);

        e.table_mut().create_file("\\c", 0).unwrap();
        e.write("\\c", 0, &a).unwrap();

        assert_eq!(
            e.read("\\c", 0, CHUNK_SIZE as usize).unwrap(),
            a,
            "forged collision aliased \\c onto \\b's blob"
        );
        assert_eq!(e.read("\\b", 0, CHUNK_SIZE as usize).unwrap(), b);
        assert_eq!(e.trace_snapshot().dedup_rejected_chunks, 1);
        assert_eq!(e.trace_snapshot().dedup_shared_chunks, 0);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn dedup_verification_rejects_forged_collision_on_partial_write() {
        // Partial writes take the single-chunk `try_share_hashed` path: the
        // chunk is materialized, patched, then offered to the dedup index.
        let mut e = engine_compress_dedup(8);
        let victim = vec![b'V'; CHUNK_SIZE as usize];
        let mut patched = vec![b'W'; CHUNK_SIZE as usize];
        patched[10..12].copy_from_slice(b"XY");

        e.table_mut().create_file("\\victim", 0).unwrap();
        e.table_mut().create_file("\\w", 0).unwrap();
        e.write("\\victim", 0, &victim).unwrap();
        e.write("\\w", 0, &vec![b'W'; CHUNK_SIZE as usize]).unwrap();

        let victim_place = e.coord("\\victim", 0).unwrap();
        assert!(matches!(victim_place, Placement::Compressed { .. }));
        forge_collision(&mut e, fnv1a(&patched), victim_place);

        e.write("\\w", 10, b"XY").unwrap();

        assert_eq!(
            e.read("\\w", 0, CHUNK_SIZE as usize).unwrap(),
            patched,
            "forged collision aliased \\w onto \\victim's blob"
        );
        assert_eq!(e.read("\\victim", 0, CHUNK_SIZE as usize).unwrap(), victim);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn dedup_trust_hash_restores_hash_only_sharing() {
        // `--dedup-trust-hash` puts the pre-verification decision back: the
        // candidate is confirmed from the index's own hash bookkeeping, so a
        // forged collision is taken and \c silently reads \b's data. This test
        // documents the behaviour the flag opts back into — the corruption is
        // the point, and is why verification is the default.
        let mut e = engine_compress_dedup(8);
        e.set_dedup_verify_bytes(false);
        assert!(!e.dedup_verify_bytes());

        let a = vec![b'A'; CHUNK_SIZE as usize];
        let b = vec![b'B'; CHUNK_SIZE as usize];
        e.table_mut().create_file("\\a", 0).unwrap();
        e.table_mut().create_file("\\b", 0).unwrap();
        e.write("\\a", 0, &a).unwrap();
        e.write("\\b", 0, &b).unwrap();

        let b_place = e.coord("\\b", 0).unwrap();
        forge_collision(&mut e, fnv1a(&a), b_place);

        e.table_mut().create_file("\\c", 0).unwrap();
        e.write("\\c", 0, &a).unwrap();

        assert_eq!(
            e.coord("\\c", 0),
            Some(b_place),
            "trust-hash mode must share on the hash alone"
        );
        assert_eq!(e.trace_snapshot().dedup_shared_chunks, 1);
        assert_eq!(e.trace_snapshot().dedup_rejected_chunks, 0);
        // ...and the flag really did cost correctness: \c reads \b's bytes.
        assert_eq!(e.read("\\c", 0, CHUNK_SIZE as usize).unwrap(), b);

        // The same forgery is refused as soon as verification is turned on.
        e.set_dedup_verify_bytes(true);
        e.table_mut().create_file("\\d", 0).unwrap();
        e.write("\\d", 0, &a).unwrap();
        assert_eq!(e.read("\\d", 0, CHUNK_SIZE as usize).unwrap(), a);
        assert_eq!(e.trace_snapshot().dedup_rejected_chunks, 1);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn dedup_batched_write_still_shares_genuine_duplicates() {
        // Regression cover for the batched full-chunk path in both modes:
        // verification must not cost dedup its whole reason for existing.
        for verify in [true, false] {
            let mut e = engine(8, true);
            e.set_dedup_verify_bytes(verify);
            let data: Vec<u8> = (0..(CHUNK_SIZE as usize * 4))
                .map(|i| ((i * 31) / 7) as u8)
                .collect();
            e.table_mut().create_file("\\f1", 0).unwrap();
            e.table_mut().create_file("\\f2", 0).unwrap();
            e.write("\\f1", 0, &data).unwrap();
            let after_first = e.used_chunks();
            assert_eq!(after_first, 4, "fixture should occupy four chunks");

            let rehashes_before = e.trace_snapshot().gpu_hash_chunks;
            e.write("\\f2", 0, &data).unwrap();

            assert_eq!(
                e.used_chunks(),
                after_first,
                "batched duplicate write consumed extra chunks (verify={verify})"
            );
            assert_eq!(e.trace_snapshot().dedup_shared_chunks, 4);
            assert_eq!(e.trace_snapshot().dedup_rejected_chunks, 0);
            assert_eq!(e.read("\\f1", 0, data.len()).unwrap(), data);
            assert_eq!(e.read("\\f2", 0, data.len()).unwrap(), data);

            let s = e.stats();
            assert_eq!(s.dedup_shared_logical_chunks, 4);
            assert_eq!(s.dedup_saved_bytes, 4 * CHUNK_SIZE);

            // The two modes confirm by different means: only trust-hash runs
            // the batched GPU re-hash over the candidates.
            let rehashed = e.trace_snapshot().gpu_hash_chunks - rehashes_before;
            if verify {
                assert_eq!(rehashed, 0, "byte verification must not re-hash");
            } else {
                assert_eq!(rehashed, 4, "trust-hash must confirm by re-hashing");
            }
        }
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn dedup_batched_write_does_not_alias_a_chunk_it_just_rewrote() {
        // Swapping two chunks in one batched write offers each chunk of the
        // file as a candidate for the other's slot. Confirming against the
        // index at the moment of the decision — rather than a snapshot taken
        // before the batch started placing chunks — is what keeps a released
        // or already-rewritten chunk from being adopted.
        let mut e = engine(8, true);
        let a = chunk_pattern(0x0A);
        let b = chunk_pattern(0x0B);
        e.table_mut().create_file("\\f", 0).unwrap();

        let mut ab = a.clone();
        ab.extend_from_slice(&b);
        e.write("\\f", 0, &ab).unwrap();
        assert_eq!(e.used_chunks(), 2);

        let mut ba = b.clone();
        ba.extend_from_slice(&a);
        e.write("\\f", 0, &ba).unwrap();

        assert_eq!(e.read("\\f", 0, ba.len()).unwrap(), ba);
        assert_eq!(
            e.used_chunks(),
            2,
            "a chunk the file still points at was returned to the allocator"
        );

        // A later allocation must not be handed a chunk \f is still using.
        e.table_mut().create_file("\\g", 0).unwrap();
        let c = chunk_pattern(0x0C);
        e.write("\\g", 0, &c).unwrap();
        assert_eq!(e.read("\\f", 0, ba.len()).unwrap(), ba);
        assert_eq!(e.read("\\g", 0, CHUNK_SIZE as usize).unwrap(), c);
    }

    // ---- rename fixes --------------------------------------------------------

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn rename_replace_frees_target_chunks() {
        let mut e = engine(4, false);
        e.table_mut().create_file("\\a", 0).unwrap();
        e.table_mut().create_file("\\b", 0).unwrap();
        e.write("\\a", 0, &vec![1u8; CHUNK_SIZE as usize]).unwrap();
        e.write("\\b", 0, &vec![2u8; CHUNK_SIZE as usize * 2])
            .unwrap();
        assert_eq!(e.used_chunks(), 3);
        // The editor save pattern: write temp, rename over the original. The
        // replaced file's two chunks must be freed, not leaked.
        e.rename("\\a", "\\b", true).unwrap();
        assert_eq!(e.used_chunks(), 1, "replaced file's chunks must be freed");
        let got = e.read("\\b", 0, CHUNK_SIZE as usize).unwrap();
        assert!(got.iter().all(|&b| b == 1));
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn case_only_rename_updates_display_name() {
        let mut e = engine(2, false);
        e.table_mut().create_file("\\lower.txt", 0).unwrap();
        e.rename("\\lower.txt", "\\LOWER.TXT", false).unwrap();
        assert_eq!(e.get("\\lower.txt").unwrap().name, "LOWER.TXT");
        let kids = e.table().readdir("\\").unwrap();
        assert_eq!(kids[0].0, "LOWER.TXT");
    }

    // ---- GPU encode/decode ---------------------------------------------------

    fn b64_reference(data: &[u8]) -> String {
        const CHARS: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for group in data.chunks(3) {
            let b0 = group[0] as u32;
            let b1 = group.get(1).copied().unwrap_or(0) as u32;
            let b2 = group.get(2).copied().unwrap_or(0) as u32;
            let w = (b0 << 16) | (b1 << 8) | b2;
            out.push(CHARS[(w >> 18) as usize & 63] as char);
            out.push(CHARS[(w >> 12) as usize & 63] as char);
            out.push(if group.len() > 1 {
                CHARS[(w >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if group.len() > 2 {
                CHARS[w as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    fn encode_roundtrip_case(e: &mut StorageEngine, tag: &str, data: &[u8]) {
        let src = format!("\\{tag}.bin");
        let b64 = format!("\\{tag}.b64");
        let back = format!("\\{tag}.back");
        let hex = format!("\\{tag}.hex");
        let hexback = format!("\\{tag}.hexback");
        e.table_mut().create_file(&src, 0).unwrap();
        if !data.is_empty() {
            e.write(&src, 0, data).unwrap();
        }

        let stats = e
            .encode_file_gpu_cancellable(
                EncodeCodec::Base64,
                EncodeDirection::Encode,
                &src,
                &b64,
                |_, _| false,
            )
            .unwrap();
        assert_eq!(stats.output_bytes, (data.len() as u64).div_ceil(3) * 4);
        let encoded = e.read(&b64, 0, stats.output_bytes as usize).unwrap();
        assert_eq!(
            String::from_utf8(encoded).unwrap(),
            b64_reference(data),
            "base64 output mismatch for {tag}"
        );

        e.encode_file_gpu_cancellable(
            EncodeCodec::Base64,
            EncodeDirection::Decode,
            &b64,
            &back,
            |_, _| false,
        )
        .unwrap();
        assert_eq!(e.file_size(&back).unwrap(), data.len() as u64);
        assert_eq!(e.read(&back, 0, data.len().max(1)).unwrap(), data);

        let stats = e
            .encode_file_gpu_cancellable(
                EncodeCodec::Hex,
                EncodeDirection::Encode,
                &src,
                &hex,
                |_, _| false,
            )
            .unwrap();
        assert_eq!(stats.output_bytes, data.len() as u64 * 2);
        let hexed = e.read(&hex, 0, data.len() * 2).unwrap();
        let expect: String = data.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(String::from_utf8(hexed).unwrap(), expect);

        e.encode_file_gpu_cancellable(
            EncodeCodec::Hex,
            EncodeDirection::Decode,
            &hex,
            &hexback,
            |_, _| false,
        )
        .unwrap();
        assert_eq!(e.read(&hexback, 0, data.len().max(1)).unwrap(), data);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn encode_base64_hex_roundtrip_all_paddings() {
        let mut e = engine(16, false);
        // Lengths mod 3 = 0, 1, 2, plus empty and >1 chunk with all byte values.
        encode_roundtrip_case(&mut e, "empty", b"");
        encode_roundtrip_case(&mut e, "pad0", b"abcdef");
        encode_roundtrip_case(&mut e, "pad1", b"abcdefg");
        encode_roundtrip_case(&mut e, "pad2", b"abcdefgh");
        let big: Vec<u8> = (0..CHUNK_SIZE as usize * 2 + 7)
            .map(|i| (i % 256) as u8)
            .collect();
        encode_roundtrip_case(&mut e, "big", &big);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn encode_decode_accepts_trailing_newline_and_rejects_garbage() {
        let mut e = engine(8, false);
        e.table_mut().create_file("\\ok.b64", 0).unwrap();
        e.write("\\ok.b64", 0, b"aGVsbG8=\r\n").unwrap();
        e.encode_file_gpu_cancellable(
            EncodeCodec::Base64,
            EncodeDirection::Decode,
            "\\ok.b64",
            "\\ok.out",
            |_, _| false,
        )
        .unwrap();
        assert_eq!(e.read("\\ok.out", 0, 16).unwrap(), b"hello");

        e.table_mut().create_file("\\bad.b64", 0).unwrap();
        e.write("\\bad.b64", 0, b"aGVs!G8=").unwrap();
        let err = e.encode_file_gpu_cancellable(
            EncodeCodec::Base64,
            EncodeDirection::Decode,
            "\\bad.b64",
            "\\bad.out",
            |_, _| false,
        );
        assert!(err.is_err(), "invalid base64 must be rejected");
        assert!(
            e.get("\\bad.out").is_none(),
            "failed decode must not leave a partial output file"
        );

        e.table_mut().create_file("\\bad.hex", 0).unwrap();
        e.write("\\bad.hex", 0, b"00ff0z").unwrap();
        assert!(e
            .encode_file_gpu_cancellable(
                EncodeCodec::Hex,
                EncodeDirection::Decode,
                "\\bad.hex",
                "\\badhex.out",
                |_, _| false,
            )
            .is_err());
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn encode_handles_sparse_input_and_leaves_no_temp_files() {
        let mut e = engine(16, false);
        e.table_mut().create_file("\\sparse.bin", 0).unwrap();
        // First chunk is a hole, the write after it lands in a second chunk.
        e.set_size("\\sparse.bin", CHUNK_SIZE).unwrap();
        e.write("\\sparse.bin", CHUNK_SIZE, b"tail").unwrap();
        let size = e.file_size("\\sparse.bin").unwrap();
        let raw = e.read("\\sparse.bin", 0, size as usize).unwrap();

        e.encode_file_gpu_cancellable(
            EncodeCodec::Base64,
            EncodeDirection::Encode,
            "\\sparse.bin",
            "\\sparse.b64",
            |_, _| false,
        )
        .unwrap();
        e.encode_file_gpu_cancellable(
            EncodeCodec::Base64,
            EncodeDirection::Decode,
            "\\sparse.b64",
            "\\sparse.back",
            |_, _| false,
        )
        .unwrap();
        assert_eq!(e.read("\\sparse.back", 0, size as usize).unwrap(), raw);

        // No .__vramdisk_* staging temp may survive.
        let leftovers: Vec<String> = e
            .table()
            .readdir("\\")
            .unwrap()
            .into_iter()
            .filter(|(name, _)| name.starts_with(".__vramdisk_"))
            .map(|(name, _)| name.to_string())
            .collect();
        assert!(leftovers.is_empty(), "staging temp leaked: {leftovers:?}");
    }

    // ---- shared-guard read fast path ---------------------------------------

    /// Position-derived byte pattern.
    ///
    /// Every offset gets a different value with a long period, so a read that
    /// returns the right *number* of bytes from the wrong *place* (a shifted
    /// run, a swapped chunk, a stale staging buffer from another thread) fails
    /// the comparison. A constant fill would not catch any of those.
    fn pattern_byte(i: usize) -> u8 {
        ((i.wrapping_mul(131).wrapping_add(i / 977)) % 251) as u8
    }

    /// Create `path` of exactly `size` bytes, write `spans` into it, and return
    /// the bytes the whole file must read back as. Everything outside `spans`
    /// stays a sparse hole (or an unwritten part of a partially written chunk),
    /// so the fixture deliberately mixes raw runs with holes.
    fn mixed_raw_sparse_file(
        e: &mut StorageEngine,
        path: &str,
        size: u64,
        spans: &[(u64, usize)],
    ) -> Vec<u8> {
        e.table_mut().create_file(path, 0).unwrap();
        e.set_size(path, size).unwrap();
        let mut expected = vec![0u8; size as usize];
        for &(off, len) in spans {
            let data: Vec<u8> = (0..len).map(|j| pattern_byte(off as usize + j)).collect();
            e.write(path, off, &data).unwrap();
            expected[off as usize..off as usize + len].copy_from_slice(&data);
        }
        expected
    }

    /// The shared path must be indistinguishable from the exclusive one for
    /// every raw/sparse read shape: inside one chunk, across several chunks,
    /// across sparse holes, clamped at EOF, past EOF, and empty.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn shared_read_matches_exclusive_read() {
        let mut e = engine(16, false);
        let path = "\\mixed";
        let size = CHUNK_SIZE * 6 + 1234;
        let expected = mixed_raw_sparse_file(
            &mut e,
            path,
            size,
            &[
                (100, 5000),                               // partial chunk 0
                (CHUNK_SIZE * 2, CHUNK_SIZE as usize * 2), // chunks 2..3, contiguous run
                (CHUNK_SIZE * 5 + 77, 4096),               // partial chunk 5
                (CHUNK_SIZE * 6, 1234),                    // tail chunk 6
            ],
        );
        // chunks 1 and 4 were never written and stay sparse.
        assert!(e.coord(path, 1).is_none(), "fixture must contain a hole");
        assert!(e.coord(path, 4).is_none(), "fixture must contain a hole");

        let cases: &[(u64, usize)] = &[
            (0, 0),                                        // zero length
            (0, 64),                                       // start of a raw chunk
            (200, 10),                                     // inside one raw chunk
            (5200, 300),                                   // inside chunk 0, past the written span
            (CHUNK_SIZE + 10, 50),                         // inside a sparse hole
            (0, CHUNK_SIZE as usize * 4),                  // raw, hole, raw, raw
            (CHUNK_SIZE * 3 + 5, CHUNK_SIZE as usize * 3), // crosses the hole at chunk 4
            (CHUNK_SIZE - 7, 14),                          // straddles a chunk boundary
            (0, size as usize),                            // the whole file
            (0, size as usize + 4096),                     // clamped at EOF
            (size - 10, 100),                              // clamped at EOF from inside the tail
            (size, 16),                                    // exactly at EOF
            (size + CHUNK_SIZE * 10, 16),                  // far past EOF
        ];

        for &(off, len) in cases {
            let mut shared = vec![0xA5u8; len];
            let got = e
                .read_into_shared(path, off, &mut shared)
                .unwrap_or_else(|e| panic!("shared read off={off} len={len}: {e:?}"))
                .unwrap_or_else(|| panic!("raw/sparse read off={off} len={len} must not bail out"));

            let mut exclusive = vec![0x5Au8; len];
            let want = e.read_into(path, off, &mut exclusive).unwrap();

            assert_eq!(got, want, "byte count differs (off={off} len={len})");
            assert_eq!(
                &shared[..got],
                &exclusive[..want],
                "shared and exclusive reads differ (off={off} len={len})"
            );
            let lo = off.min(size) as usize;
            assert_eq!(
                &shared[..got],
                &expected[lo..lo + got],
                "shared read returned the wrong bytes (off={off} len={len})"
            );
            assert!(
                shared[got..].iter().all(|&b| b == 0xA5),
                "shared read wrote past the {got} bytes it reported (off={off} len={len})"
            );
        }
    }

    /// `$VRAMDISK\trace.json` must not start lying once reads stop taking the
    /// exclusive guard, so the shared path has to move exactly the counters the
    /// exclusive path would have moved, by exactly the same amounts.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn shared_read_traces_identically_to_exclusive_read() {
        let mut e = engine(16, false);
        let path = "\\traced";
        let size = CHUNK_SIZE * 5 + 99;
        mixed_raw_sparse_file(
            &mut e,
            path,
            size,
            &[(0, CHUNK_SIZE as usize * 2), (CHUNK_SIZE * 4, 8192)],
        );

        let mut buf = vec![0u8; size as usize];

        e.reset_trace();
        e.read_into_shared(path, 0, &mut buf).unwrap().unwrap();
        let shared = e.trace_snapshot();

        e.reset_trace();
        e.read_into(path, 0, &mut buf).unwrap();
        let exclusive = e.trace_snapshot();

        assert_eq!(shared, exclusive);
        assert_eq!(shared.read_calls, 1);
        assert_eq!(shared.logical_read_bytes, size);
        assert!(shared.raw_read_bytes > 0, "fixture must exercise raw reads");
    }

    /// A compressed placement needs the nvCOMP codec's device scratch, which is
    /// mutated per call, so the shared path must decline — and it must decline
    /// having written nothing, since the caller's buffer is about to be reused
    /// verbatim by the exclusive retry.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn shared_read_bails_out_on_compressed_placement() {
        let mut e = engine_compress(8);
        let path = "\\compressible";
        e.table_mut().create_file(path, 0).unwrap();
        let data = vec![b'Q'; CHUNK_SIZE as usize * 2];
        e.write(path, 0, &data).unwrap();
        assert!(
            matches!(e.coord(path, 0), Some(Placement::Compressed { .. })),
            "fixture must actually be stored compressed"
        );

        e.reset_trace();
        let mut buf = vec![0x5Au8; data.len()];
        assert_eq!(e.read_into_shared(path, 0, &mut buf).unwrap(), None);
        assert!(
            buf.iter().all(|&b| b == 0x5A),
            "a bail-out must leave the caller's buffer untouched"
        );
        assert_eq!(
            e.trace_snapshot(),
            EngineTrace::default(),
            "a read that was not served must not be counted as one"
        );

        // A partially compressed read must bail too, not serve the raw prefix.
        e.table_mut().create_file("\\partial", 0).unwrap();
        e.set_size("\\partial", CHUNK_SIZE * 3).unwrap();
        e.write(
            "\\partial",
            CHUNK_SIZE * 2,
            &vec![b'Q'; CHUNK_SIZE as usize],
        )
        .unwrap();
        let mut buf = vec![0x5Au8; (CHUNK_SIZE * 3) as usize];
        assert_eq!(e.read_into_shared("\\partial", 0, &mut buf).unwrap(), None);
        assert!(buf.iter().all(|&b| b == 0x5A));

        // The exclusive fallback still serves what the shared path declined.
        let n = e.read_into(path, 0, &mut buf[..data.len()]).unwrap();
        assert_eq!(&buf[..n], &data[..]);
    }

    /// The bail-out keys on the *placement*, not on whether the volume has
    /// compression enabled: a sparse hole on a compressing engine is still a
    /// hole and still serviceable through a shared guard.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn shared_read_serves_sparse_data_on_a_compressing_engine() {
        let mut e = engine_compress(8);
        let path = "\\holes";
        e.table_mut().create_file(path, 0).unwrap();
        e.set_size(path, CHUNK_SIZE * 3).unwrap();
        let mut buf = vec![0xFFu8; (CHUNK_SIZE * 3) as usize];
        let n = e
            .read_into_shared(path, 0, &mut buf)
            .unwrap()
            .expect("an all-sparse file needs no exclusive access");
        assert_eq!(n, (CHUNK_SIZE * 3) as usize);
        assert!(
            buf.iter().all(|&b| b == 0),
            "sparse holes must read as zeros"
        );
    }

    /// The point of the whole exercise: many threads inside the engine at once,
    /// each holding only a shared guard, each getting its own bytes back.
    ///
    /// The reads deliberately overlap in the device ranges they touch and in
    /// the sizes they use, so they contend for `Vram`'s pinned staging buffers
    /// and its transfer streams. A shared-state bug in the read path — a
    /// staging buffer handed to two threads, a stream synchronised by the wrong
    /// one — shows up here as bytes from another thread's request.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn shared_read_is_correct_under_concurrent_readers() {
        use std::sync::{Arc, Barrier, RwLock};

        const THREADS: usize = 8;
        const ITERS: usize = 24;

        let mut e = engine(64, false);
        let path = "\\concurrent";
        let size = CHUNK_SIZE * 24 + 4096;
        // Two long raw runs with sparse holes between them, so the coalescing
        // walk has something to coalesce and something to stop at.
        let expected = mixed_raw_sparse_file(
            &mut e,
            path,
            size,
            &[
                (0, CHUNK_SIZE as usize * 8),
                (CHUNK_SIZE * 10, CHUNK_SIZE as usize * 6),
                (CHUNK_SIZE * 20 + 512, CHUNK_SIZE as usize * 4),
            ],
        );

        e.reset_trace();
        let engine = Arc::new(RwLock::new(e));
        let expected = Arc::new(expected);
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::with_capacity(THREADS);

        for t in 0..THREADS {
            let engine = Arc::clone(&engine);
            let expected = Arc::clone(&expected);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                for i in 0..ITERS {
                    let off = ((t * 7919 + i * 4093) as u64 * 512) % size;
                    let len = ((t + i) % 5 + 1) * (CHUNK_SIZE as usize / 2) + 333;
                    let mut buf = vec![0xA5u8; len];

                    let guard = engine.read().expect("engine read guard");
                    let got = guard
                        .read_into_shared(path, off, &mut buf)
                        .expect("shared read failed")
                        .expect("a raw/sparse file must never bail out");
                    drop(guard);

                    assert_eq!(
                        got,
                        ((size - off) as usize).min(len),
                        "thread {t} iter {i}: wrong length for off={off} len={len}"
                    );
                    let want = &expected[off as usize..off as usize + got];
                    if buf[..got] != *want {
                        let bad = buf[..got]
                            .iter()
                            .zip(want)
                            .position(|(a, b)| a != b)
                            .expect("slices differ");
                        panic!(
                            "thread {t} iter {i}: off={off} len={len} byte {bad} is {:#04x}, \
                             expected {:#04x}",
                            buf[bad], want[bad]
                        );
                    }
                    assert!(
                        buf[got..].iter().all(|&b| b == 0xA5),
                        "thread {t} iter {i}: wrote past the {got} reported bytes"
                    );
                }
            }));
        }

        for (t, h) in handles.into_iter().enumerate() {
            h.join()
                .unwrap_or_else(|_| panic!("reader thread {t} panicked"));
        }

        // Every read is still accounted for, from every thread.
        let trace = engine.read().unwrap().trace_snapshot();
        assert_eq!(trace.read_calls, (THREADS * ITERS) as u64);
        assert!(
            trace.raw_read_ops > 0 && trace.logical_read_bytes > 0,
            "concurrent shared reads must still land in the trace counters"
        );
    }

    /// The GF(2) fold has to reproduce a serial scan exactly, because these
    /// checksums end up in ZIP local headers and gzip trailers that other
    /// tools verify. Split points are chosen to cover the interesting shapes:
    /// an empty tail, a sub-byte-ladder length, an exact power of two, and an
    /// odd length that exercises both branches of the squaring loop.
    #[test]
    fn crc32_combine_reproduces_a_serial_scan() {
        let data = patterned_bytes(9_973, 7);
        let whole = crate::api_kernel::crc32_reference(&data);
        for split in [0usize, 1, 2, 63, 64, 255, 4096, 9_972, 9_973] {
            let (head, tail) = data.split_at(split);
            let combined = crc32_combine_with(
                &crc32_zero_shift(tail.len() as u64),
                crate::api_kernel::crc32_reference(head),
                crate::api_kernel::crc32_reference(tail),
            );
            assert_eq!(
                combined, whole,
                "combining at {split} must equal the serial CRC32"
            );
        }
    }

    /// Folding lane by lane, the way `crc32_range_gpu_cancellable` does, must
    /// also land on the serial value — and starting the fold from 0 (the CRC of
    /// the empty string) must be the identity for the first lane.
    #[test]
    fn crc32_combine_folds_equal_lanes() {
        let data = patterned_bytes(4_100, 3);
        let lane = 512usize;
        let shift = crc32_zero_shift(lane as u64);
        let mut crc = 0u32;
        let mut off = 0usize;
        while off < data.len() {
            let take = (data.len() - off).min(lane);
            let lane_shift = if take == lane {
                shift
            } else {
                crc32_zero_shift(take as u64)
            };
            crc = crc32_combine_with(
                &lane_shift,
                crc,
                crate::api_kernel::crc32_reference(&data[off..off + take]),
            );
            off += take;
        }
        assert_eq!(crc, crate::api_kernel::crc32_reference(&data));
    }

    /// A GPU device is not needed to pin the routing *policy*, only the two
    /// measured throughputs it is derived from.
    #[test]
    fn calibration_routes_large_files_to_the_gpu_when_the_gpu_is_faster() {
        // 2 GB/s on the GPU against 1 GB/s on the CPU: nothing to trade off,
        // so no file is too large for the GPU and the old 128 MiB ceiling is
        // simply absent.
        let c = calibration_from_throughput(1 << 20, 0.0005, 2e9, 0.001, 1e9);
        assert_eq!(c.cpu_route_threshold_bytes, u64::MAX);
        // The launch budget still bounds one kernel, TDR being unrelated to
        // which side is faster: 2 GB/s for 0.1 s.
        assert_eq!(c.launch_budget_bytes, 200_000_000);
    }

    #[test]
    fn calibration_keeps_large_files_off_a_slower_gpu() {
        // What this machine actually measures: ~33 MB/s single-thread on the
        // GPU against ~1.6 GB/s on the CPU.
        let c = calibration_from_throughput(1 << 20, 0.0318, 33e6, 0.00065, 1.6e9);
        assert!(
            c.cpu_route_threshold_bytes <= c.launch_budget_bytes,
            "a file admitted to the GPU must still fit in one launch so the \
             batch it joins has room for other files: threshold {} budget {}",
            c.cpu_route_threshold_bytes,
            c.launch_budget_bytes
        );
        assert!(c.cpu_route_threshold_bytes >= GPU_HASH_ROUTE_THRESHOLD_MIN_BYTES);
        assert!(
            c.cpu_route_threshold_bytes < 128 * 1024 * 1024,
            "the 0.25 s bound still applies when the GPU is the slower side"
        );
    }

    /// The GPU CRC32 is folded from independent lanes, so the interesting
    /// failures are at the seams: between lanes, between launches, and on a
    /// final short lane. The launch size is shrunk so a small fixture crosses
    /// all three.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn gpu_crc32_matches_reference_across_lanes_and_launches() {
        let mut e = engine(64, false);
        // Three full launches of four lanes each, plus a short final lane.
        e.set_crc32_launch_bytes(4 * CRC32_LANE_BYTES);
        let len = (12 * CRC32_LANE_BYTES + 1234) as usize;
        let data = patterned_bytes(len, 11);
        e.table_mut().create_file("\\crc", 0).unwrap();
        e.write("\\crc", 0, &data).unwrap();
        assert_eq!(
            e.crc32_range_gpu("\\crc", 0, len as u64).unwrap(),
            crate::api_kernel::crc32_reference(&data)
        );
        // Sub-ranges must fold the same way, which is what the gzip reader
        // verifies members with.
        let off = CRC32_LANE_BYTES + 7;
        let take = 5 * CRC32_LANE_BYTES + 99;
        assert_eq!(
            e.crc32_range_gpu("\\crc", off, take).unwrap(),
            crate::api_kernel::crc32_reference(&data[off as usize..(off + take) as usize])
        );
        // A sparse hole is synthesized inside the kernel rather than read, so
        // it has its own path through the lane builder.
        e.table_mut().create_file("\\hole", 0).unwrap();
        let hole_len = 3 * CRC32_LANE_BYTES;
        e.set_size("\\hole", hole_len).unwrap();
        assert_eq!(
            e.crc32_range_gpu("\\hole", 0, hole_len).unwrap(),
            crate::api_kernel::crc32_reference(&vec![0u8; hole_len as usize])
        );
    }

    /// Coverage for the sizes the new routing rule can send to the GPU that
    /// the old 128 MiB ceiling could not. The digests must still be the
    /// RustCrypto ones, and the GPU and CPU routes must agree with each other.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn gpu_hash_matches_reference_above_the_old_route_cap() {
        let mut e = engine(512, false);
        // Above `GPU_HASH_ROUTE_THRESHOLD_MAX_BYTES` as it used to be, so this
        // file is only reachable on the GPU under the measured rule.
        let len = 129 * 1024 * 1024usize;
        let data = patterned_bytes(len, 23);
        e.table_mut().create_file("\\big", 0).unwrap();
        e.write("\\big", 0, &data).unwrap();
        e.set_hash_cpu_route_threshold(u64::MAX);
        assert!(
            e.should_hash_on_gpu_routed("\\big").unwrap(),
            "a file this size must be routable to the GPU once the ceiling is \
             derived from measurement rather than fixed"
        );
        for alg in [HashAlgorithm::Fnv1a64, HashAlgorithm::Sha256] {
            let expected = hash_reference(alg, &data);
            assert_eq!(
                e.hash_file(r"\big", alg).unwrap(),
                expected,
                "{} routed to the GPU at {len} bytes",
                alg.name()
            );
            assert_eq!(
                e.hash_file_cpu(r"\big", alg).unwrap(),
                expected,
                "{} on the CPU route at {len} bytes",
                alg.name()
            );
        }
    }

    /// Naive reference scan, so the GPU result is checked against something
    /// obviously correct rather than against itself.
    fn search_reference(hay: &[u8], needle: &[u8], fold: bool) -> Vec<u64> {
        let norm = |b: u8| if fold { b.to_ascii_lowercase() } else { b };
        if needle.is_empty() || hay.len() < needle.len() {
            return Vec::new();
        }
        (0..=hay.len() - needle.len())
            .filter(|&i| (0..needle.len()).all(|k| norm(hay[i + k]) == norm(needle[k])))
            .map(|i| i as u64)
            .collect()
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn search_matches_a_reference_scan() {
        let mut e = engine(16, false);
        let mut data = Vec::new();
        for i in 0..40_000u32 {
            data.extend_from_slice(format!("line {i} needle-{}Q", i % 7).as_bytes());
        }
        e.table_mut().create_file("\\a.bin", 0).unwrap();
        e.write("\\a.bin", 0, &data).unwrap();
        for (pat, fold) in [
            (&b"needle-3"[..], false),
            (&b"NEEDLE-3"[..], true),
            (&b"Q"[..], false),
            (&b"line 39999 needle-3Q"[..], false),
            (&b"zzz-absent"[..], false),
        ] {
            let stats = e
                .search_files_gpu_cancellable(
                    &["\\a.bin".to_string()],
                    pat,
                    fold,
                    usize::MAX,
                    |_, _| false,
                )
                .unwrap();
            let want = search_reference(&data, pat, fold);
            assert_eq!(
                stats.total_matches,
                want.len() as u64,
                "count for {:?} fold={fold}",
                String::from_utf8_lossy(pat)
            );
            let got: Vec<u64> = stats
                .hits
                .first()
                .map(|h| h.offsets.clone())
                .unwrap_or_default();
            assert_eq!(got, want, "offsets for {:?}", String::from_utf8_lossy(pat));
        }
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn search_finds_matches_across_window_seams() {
        // The window is shrunk so the file spans many launches; a match placed
        // deliberately astride each seam must still be found exactly once.
        let mut e = engine(16, false);
        e.set_search_window_bytes(CHUNK_SIZE);
        let window = CHUNK_SIZE;
        let needle = b"SEAMMARK";
        let mut data = vec![b'.'; (window * 5) as usize];
        // Straddle every seam: half the needle before it, half after.
        for seam in 1..5u64 {
            let at = (seam * (window - (needle.len() as u64 - 1))) as usize - needle.len() / 2;
            data[at..at + needle.len()].copy_from_slice(needle);
        }
        e.table_mut().create_file("\\seam.bin", 0).unwrap();
        e.write("\\seam.bin", 0, &data).unwrap();
        let stats = e
            .search_files_gpu_cancellable(
                &["\\seam.bin".to_string()],
                needle,
                false,
                usize::MAX,
                |_, _| false,
            )
            .unwrap();
        assert_eq!(
            stats.total_matches,
            search_reference(&data, needle, false).len() as u64
        );
        assert_eq!(
            stats.hits[0].offsets,
            search_reference(&data, needle, false)
        );
        assert_eq!(stats.bytes_scanned, data.len() as u64);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn search_reads_compressed_and_sparse_files() {
        let mut e = engine_compress(16);
        // Highly compressible, so the chunks really do land compressed.
        let mut data = b"the quick brown fox ".repeat(20_000);
        let at = data.len() / 2;
        data[at..at + 6].copy_from_slice(b"MARKER");
        e.table_mut().create_file("\\c.bin", 0).unwrap();
        e.write("\\c.bin", 0, &data).unwrap();
        // Sparse: a hole, then content past it.
        e.table_mut().create_file("\\s.bin", 0).unwrap();
        e.write("\\s.bin", 4 * CHUNK_SIZE, b"MARKER").unwrap();

        let stats = e
            .search_files_gpu_cancellable(
                &["\\c.bin".to_string(), "\\s.bin".to_string()],
                b"MARKER",
                false,
                16,
                |_, _| false,
            )
            .unwrap();
        assert_eq!(stats.files_matched, 2);
        assert_eq!(stats.total_matches, 2);
        assert_eq!(stats.hits[0].offsets, vec![at as u64]);
        assert_eq!(stats.hits[1].offsets, vec![4 * CHUNK_SIZE]);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn search_reports_exact_counts_when_offsets_are_capped() {
        let mut e = engine(8, false);
        let data = b"ab".repeat(5_000);
        e.table_mut().create_file("\\many.bin", 0).unwrap();
        e.write("\\many.bin", 0, &data).unwrap();
        let stats = e
            .search_files_gpu_cancellable(&["\\many.bin".to_string()], b"ab", false, 10, |_, _| {
                false
            })
            .unwrap();
        assert_eq!(stats.total_matches, 5_000, "count must not be capped");
        assert_eq!(stats.hits[0].offsets.len(), 10, "offsets are capped");
        assert!(stats.hits[0].truncated);
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn search_rejects_unusable_patterns_and_honours_cancellation() {
        let mut e = engine(8, false);
        e.table_mut().create_file("\\x.bin", 0).unwrap();
        e.write("\\x.bin", 0, &b"hello".repeat(1000)).unwrap();
        assert!(matches!(
            e.search_files_gpu_cancellable(&["\\x.bin".to_string()], b"", false, 8, |_, _| false),
            Err(EngineError::InvalidInput(_))
        ));
        let long = vec![b'z'; SEARCH_MAX_PATTERN + 1];
        assert!(matches!(
            e.search_files_gpu_cancellable(&["\\x.bin".to_string()], &long, false, 8, |_, _| false),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            e.search_files_gpu_cancellable(&["\\x.bin".to_string()], b"hello", false, 8, |_, _| {
                true
            }),
            Err(EngineError::Cancelled)
        ));
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn search_stitches_matches_across_physical_run_boundaries() {
        // The in-place fast path scans each physically contiguous run with its
        // own launch, so a match straddling two runs is found only by the host
        // stitch. Fragment a file on purpose and put a needle on every seam.
        let mut e = engine(32, false);
        e.table_mut().create_file("\\a.bin", 0).unwrap();
        e.table_mut().create_file("\\b.bin", 0).unwrap();
        // Interleaved appends push the two files' chunks apart from each other.
        let block = vec![b'.'; CHUNK_SIZE as usize];
        for i in 0..24u64 {
            e.write("\\a.bin", i * CHUNK_SIZE, &block).unwrap();
            e.write("\\b.bin", i * CHUNK_SIZE, &block).unwrap();
        }
        let size = e.file_size("\\a.bin").unwrap();
        let runs = e.raw_runs("\\a.bin", size).expect("all raw");
        assert!(
            runs.len() > 1,
            "test needs a fragmented file, got {} run(s)",
            runs.len()
        );

        let needle = b"SEAM";
        for w in runs.windows(2) {
            let end = w[0].0 + w[0].2;
            e.write("\\a.bin", end - 2, needle).unwrap();
        }
        // One match wholly inside a run, as a control.
        e.write("\\a.bin", 100, needle).unwrap();

        let data = e.read("\\a.bin", 0, size as usize).unwrap();
        let want = search_reference(&data, needle, false);
        assert_eq!(
            want.len(),
            runs.len(),
            "fixture should place one per seam + 1"
        );

        let stats = e
            .search_files_gpu_cancellable(
                &["\\a.bin".to_string()],
                needle,
                false,
                usize::MAX,
                |_, _| false,
            )
            .unwrap();
        assert_eq!(
            stats.total_matches,
            want.len() as u64,
            "seam matches counted once"
        );
        assert_eq!(stats.hits[0].offsets, want, "seam match offsets");
    }
}
