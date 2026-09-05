//! CUDA layer: owns the single large VRAM buffer and moves bytes in/out of it.
//!
//! The buffer is one contiguous allocation in device memory. Higher layers
//! address it by byte offset; chunk math lives in [`crate::chunk`].
//!
//! # How host↔device transfers are staged, and why
//!
//! Transfers come in three sizes. Below [`OPTIMIZED_TRANSFER_THRESHOLD`] they
//! go through the default stream as an ordinary pageable copy. Above it they
//! are copied through *our own* page-locked staging buffers. Only host-to-device
//! transfers of [`HOST_REGISTER_THRESHOLD`] or more page-lock the caller's own
//! memory, and then only for the duration of that one call.
//!
//! Preferring staging over "just pin the caller's buffer and DMA straight into
//! it" is a deliberate reversal of the obvious optimisation, so it is worth
//! recording why.
//!
//! **1. Registering per call is usually slower than staging.**
//! `cuMemHostRegister` walks and locks every page in the range. Measured on
//! this project's target machine (i9-12900K / RTX 4070, PCIe 4.0 x16),
//! registering *and* unregistering costs ~154 µs for 1 MiB, ~658 µs for 16 MiB
//! and ~2.53 ms for 64 MiB of warm page-aligned private memory — but ~1.95 ms
//! for the 16 MiB buffer WinFsp actually hands us, whose pages are cold on
//! every request. Against a 16 MiB device-to-host copy that itself takes
//! ~2.15 ms that is a 47 % tax, and it made a filesystem-level 16 MiB read
//! spend 1.70 ms registering, 2.15 ms copying and 0.25 ms unregistering. It
//! only pays on large transfers out of memory that is already resident, which
//! is what [`HOST_REGISTER_THRESHOLD`] is calibrated to select.
//!
//! **2. Caching the registrations to amortise that cost is unsound.** The
//! obvious fix is to keep registered ranges alive in an LRU map keyed by the
//! buffer address — and the addresses do repeat: instrumenting a full
//! filesystem benchmark showed *two* distinct page-aligned base addresses
//! across ~2000 large transfers, 1999 of them the same one, because WinFsp's
//! `AlwaysUseDoubleBuffering` hands out pooled buffers. But [`Vram::read_at`]
//! and [`Vram::write_at`] are also called with ordinary `Vec` buffers from the
//! storage engine and the benchmarks, and those *are* freed and reallocated.
//! When a range CUDA still has registered is released and its address reused,
//! the driver does **not** notice — `cached_registration_would_outlive_its_buffer`
//! in this file's tests shows it still refusing to re-register the recycled
//! address, and a probe run with an 8 MiB range read `0xAA` through CUDA while
//! the CPU read `0xBB` from the same pointer. A device-to-host copy through
//! such a mapping writes into physical pages the OS has already handed to
//! somebody else. There is no cheap way to validate a cached registration
//! (`VirtualQuery` cannot distinguish a reused heap block, and CUDA's own
//! pointer attributes are answered from the same stale table), so the cache is
//! not implementable safely here and is not implemented. Registrations are
//! therefore strictly scoped to one call, guarded by [`RegisteredHost`].
//!
//! **3. Staging is fast enough that we give up very little.** The host-side
//! `memcpy` between the caller's buffer and a pinned stage is the only extra
//! cost, and it is parallelised across [`COPY_SHARDS`] worker threads
//! ([`Vram::read_staged`] / [`Vram::write_staged`]), which is what closes the
//! gap: single-threaded that copy ran at ~3.5 GB/s into WinFsp's buffer, and it
//! is the dominant term (per 512 KiB stage: 150 µs of `memcpy` against 17 µs of
//! waiting for the DMA).

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use cudarc::driver::{result, sys, CudaContext, CudaSlice, CudaStream, DevicePtr};

/// Upper bound on how much pinned memory a single staging slot may hold.
/// Only a ceiling: [`PIPELINE_STAGE_BYTES`] is what actually sizes a slot, and
/// it is far below this. Kept as a guard so a future stage-size change cannot
/// silently start allocating hundreds of megabytes of page-locked memory.
const PINNED_STAGE_MAX: usize = 64 * 1024 * 1024;

/// Transfers below this go straight through the default stream as an ordinary
/// pageable copy; above it they are staged through pinned memory.
///
/// Measured per-operation latency (host round trip, this machine): at 4 KiB
/// the pageable path costs 8.0 µs against 9.0 µs for a synchronous
/// `cuMemcpyDtoH` and 10.9 µs through a persistent pinned bounce buffer, so
/// nothing beats it for small reads and the threshold must stay above them.
/// At 256 KiB the staged path is ahead (~6.8 GB/s pageable against ~10 GB/s
/// staged), which is where this is set.
const OPTIMIZED_TRANSFER_THRESHOLD: usize = 256 * 1024;

/// Bytes moved per staging slot per round.
///
/// Swept at 1 / 2 / 4 / 8 MiB against `--bench`, best of three runs each so the
/// GPU's idle downclocking does not dominate. 4 MiB won nearly everywhere and
/// was never worse: `[1]` 4 MB write went 5.55 → 6.59 → **10.39** → 10.77 GB/s
/// and `[2]` 64 MB read 5.50 → 6.47 → **6.27** → 5.38 GB/s across those four
/// sizes. Below 4 MiB the per-submission cost of many small DMAs shows up; at
/// 8 MiB the engine's 16 MB reads lose their pipelining (4.39 GB/s) because a
/// request no longer splits into enough rounds.
///
/// The old 16 MiB value was the worst of both: a 16 MiB request became a
/// single un-pipelined stage — one full DMA followed by one full `memcpy` —
/// which is what made the read path serial.
const PIPELINE_STAGE_BYTES: usize = 4 * 1024 * 1024;

/// Number of staging slots per direction: exactly what one full-width transfer
/// wants, [`COPY_SHARDS`] × [`SLOTS_PER_SHARD`].
///
/// Each slot owns one pinned buffer *and* one copy stream — 16 streams in all —
/// which is what lets a slot's `synchronize` wait for its own copy and nothing
/// else. The four streams this replaced were shared between every in-flight
/// transfer, so waiting on one also waited on whatever another caller had
/// queued there.
///
/// Doubling the count to sixteen slots per direction, so that two full-width
/// transfers could never contend, measured no better on the 8-thread
/// concurrent-read benchmark (16 MB shared read 7.28 against 7.17 GB/s) — a
/// lease that cannot get every slot it asked for simply runs with fewer, so
/// contention costs parallelism rather than blocking. Eight slots of
/// [`PIPELINE_STAGE_BYTES`] per direction is 64 MiB of pinned memory in total,
/// the same as the 2×16 MiB×2 layout this replaced.
const PIPELINE_STAGES: usize = COPY_SHARDS * SLOTS_PER_SHARD;

/// How many shards a large transfer is split into, i.e. how many threads run
/// the host-side `memcpy` at once.
///
/// The copy out of (or into) the caller's buffer is the bottleneck, not the
/// DMA: per 512 KiB stage the filesystem read path spent 150.4 µs in `memcpy`
/// against 16.7 µs waiting on the DMA. Sharding it 4 ways took filesystem
/// 16 MiB reads from 2.21 to 2.97 GB/s; 6 and 8 shards measured no better
/// (2.54 and 2.96 GB/s), so 4 is where the curve flattens — which is where
/// memory bandwidth, not core count, starts to bind.
const COPY_SHARDS: usize = 4;

/// Transfers at least this large are sharded across [`COPY_SHARDS`] threads;
/// smaller ones are pipelined on the calling thread instead, because spawning
/// workers costs more than it saves on a transfer only a round or two long.
/// With 1 MiB stages, sharding a 4 MB transfer four ways measured 5.55 GB/s
/// against 10.39 GB/s for the same 4 MB moved as one inline stage — four
/// thread spawns are the same order as the copy itself at that size.
const PARALLEL_COPY_THRESHOLD: usize = 2 * PIPELINE_STAGE_BYTES;

/// Host-to-device transfers at least this large skip staging and DMA straight
/// out of the caller's buffer, page-locking it for the duration of the one call
/// and releasing it again before returning. Nothing is cached, so none of the
/// hazards in the module docs apply.
///
/// The threshold is deliberately far above anything the filesystem can produce.
/// WinFsp's largest single request measured here is 16 MiB, and its pooled
/// buffer is *cold* — page-locking it costs ~1.95 ms and the DMA out of it only
/// reaches 7.8 GB/s, so registering it is a large net loss. A caller handing us
/// 32 MiB or more in one go is the storage engine or a benchmark, passing a
/// long-lived `Vec` whose pages are already resident: there `cuMemHostRegister`
/// costs ~1.3 ms per 32 MiB (interpolated from 658 µs at 16 MiB and 2.53 ms at
/// 64 MiB) and buys a ~18 GB/s DMA with no host copy at all.
///
/// Measured on `[1] Raw VRAM Bandwidth` at 64 MB: 10.98 GB/s registered against
/// 8.63 GB/s staged, and on `[2]` at 256 MB, 11.19 against 8.91 GB/s.
///
/// Reads deliberately do *not* get this treatment: their destination is
/// typically a freshly allocated buffer whose pages are not yet resident, and
/// staging beat registration there by 36 % at 64 MB.
const HOST_REGISTER_THRESHOLD: usize = 32 * 1024 * 1024;

/// Staging slots handed to each shard, i.e. the depth of its pipeline.
///
/// With one slot a shard runs fill → DMA → wait → fill → …, so each round's DMA
/// is serialised behind the host copy that feeds it; two slots let the DMA of
/// one round overlap the copy of the next. Measured on `[1] Raw VRAM Bandwidth`
/// at 64 MB: 8.02 GB/s with one slot per shard against 8.45 with two. A third
/// gave no further gain (8.41) — one round of DMA is already fully hidden, and
/// the staged path is then bounded by the host copy itself. That ceiling is why
/// transfers this large take the [`HOST_REGISTER_THRESHOLD`] path instead,
/// where there is no host copy at all.
const SLOTS_PER_SHARD: usize = 2;

/// Owns the VRAM allocation and the CUDA context/stream used to touch it.
pub struct Vram {
    /// Kept alive so the primary context stays retained for the buffer's life.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    buf: CudaSlice<u8>,
    size: u64,
    h2d: StagePool,
    d2h: StagePool,
}

#[derive(Clone, Copy)]
enum PinnedKind {
    Normal,
    WriteCombined,
}

/// The single page-locked allocation a direction's staging slots carve up.
///
/// One `cuMemHostAlloc` rather than one per slot, because every registered host
/// range goes into a table the driver consults on *every* pageable copy: with
/// sixteen separate slot allocations the small-transfer path measurably slowed
/// down (4 KiB random reads through the filesystem fell from ~33.4k to ~29.4k
/// IOPS) even though that path does not touch the staging buffers at all.
/// Allocating up front also keeps `cuMemHostAlloc` — slow enough to dominate a
/// three-run benchmark — off the first measured transfer.
struct PinnedBlock {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

unsafe impl Send for PinnedBlock {}
unsafe impl Sync for PinnedBlock {}

impl PinnedBlock {
    fn new(ctx: &Arc<CudaContext>, len: usize, kind: PinnedKind) -> Result<Self> {
        ctx.bind_to_thread()
            .context("bind CUDA context for pinned host alloc")?;
        let flags = match kind {
            PinnedKind::Normal => 0,
            PinnedKind::WriteCombined => sys::CU_MEMHOSTALLOC_WRITECOMBINED,
        };
        let ptr = unsafe { result::malloc_host(len, flags) }.context("cuMemHostAlloc")?;
        let ptr = std::ptr::NonNull::new(ptr as *mut u8).context("cuMemHostAlloc returned null")?;
        Ok(Self { ptr, len })
    }
}

impl Drop for PinnedBlock {
    fn drop(&mut self) {
        let _ = unsafe { result::free_host(self.ptr.as_ptr() as *mut c_void) };
    }
}

/// One staging slot: a disjoint window into the direction's [`PinnedBlock`],
/// together with the copy stream that fills it.
///
/// Pairing a buffer with its own stream is what lets a transfer wait for *its
/// own* bytes and nothing else. The previous layout shared four streams between
/// every in-flight transfer, so `synchronize` on one also waited for whatever
/// unrelated work another caller had queued there.
struct Stage {
    /// Borrowed from the pool's [`PinnedBlock`], which outlives every slot.
    ptr: std::ptr::NonNull<u8>,
    cap: usize,
    stream: Arc<CudaStream>,
}

unsafe impl Send for Stage {}

impl Stage {
    /// The slot's staging buffer, truncated to `len`.
    ///
    /// `len` never exceeds `cap`: callers clamp every round to
    /// [`stage_bytes`], which is what the slot was carved to hold.
    fn buf(&mut self, len: usize) -> Result<&mut [u8]> {
        if len > self.cap {
            anyhow::bail!(
                "staging round of {len} bytes exceeds slot capacity {}",
                self.cap
            );
        }
        // SAFETY: `ptr` points at `cap` bytes inside the pool's block, the
        // block outlives the pool and therefore this slot, and a slot is only
        // ever held by one leaseholder at a time.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) })
    }
}

/// A fixed set of [`Stage`]s that callers lease for the duration of one
/// transfer.
///
/// WinFsp dispatches its callbacks from a pool of threads, so several
/// `read_at`s can be in flight at once. A single shared staging buffer behind
/// one mutex would serialise them; leasing from a pool lets independent
/// transfers proceed in parallel and only blocks when every slot is busy.
struct StagePool {
    free: Mutex<Vec<Stage>>,
    released: Condvar,
    /// The allocation every slot's buffer points into. Declared last so that
    /// fields drop in order and the slots go away before the memory they
    /// borrow; never accessed through this field.
    #[allow(dead_code)]
    block: PinnedBlock,
}

impl StagePool {
    /// Allocates one pinned block of `slots * slot_bytes` and carves it into
    /// `slots` disjoint windows, one per stream.
    fn new(
        ctx: &Arc<CudaContext>,
        slots: usize,
        kind: PinnedKind,
        slot_bytes: usize,
    ) -> Result<Self> {
        let slot_bytes = slot_bytes.max(1);
        let block = PinnedBlock::new(ctx, slots * slot_bytes, kind)?;
        let mut free = Vec::with_capacity(slots);
        for i in 0..slots {
            // SAFETY: `i * slot_bytes < block.len`, so the offset stays inside
            // the allocation and each window is disjoint from the others.
            let ptr = unsafe { block.ptr.as_ptr().add(i * slot_bytes) };
            free.push(Stage {
                ptr: std::ptr::NonNull::new(ptr).context("pinned block offset was null")?,
                cap: slot_bytes,
                stream: ctx.new_stream().context("create transfer stream")?,
            });
        }
        debug_assert_eq!(block.len, slots * slot_bytes);
        Ok(Self {
            free: Mutex::new(free),
            released: Condvar::new(),
            block,
        })
    }

    /// Lease up to `want` slots, blocking until at least one is free.
    ///
    /// Taking fewer than asked for is fine — a transfer simply runs with less
    /// parallelism — so a busy pool degrades throughput instead of deadlocking.
    fn lease(&self, want: usize) -> StageLease<'_> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        while free.is_empty() {
            free = self.released.wait(free).unwrap_or_else(|e| e.into_inner());
        }
        let take = want.clamp(1, free.len());
        let at = free.len() - take;
        let slots = free.split_off(at);
        StageLease { pool: self, slots }
    }
}

/// Returns its slots to the pool on drop, including on the error paths.
struct StageLease<'a> {
    pool: &'a StagePool,
    slots: Vec<Stage>,
}

impl Drop for StageLease<'_> {
    fn drop(&mut self) {
        let mut free = self.pool.free.lock().unwrap_or_else(|e| e.into_inner());
        free.append(&mut self.slots);
        drop(free);
        self.pool.released.notify_all();
    }
}

/// Page-locks a host range for the lifetime of the guard.
///
/// Deliberately scoped to a single transfer: the registration is released in
/// `Drop`, before the call that created it returns, so no registration can
/// outlive the buffer it describes. See the module docs for why keeping these
/// alive across calls is not safe.
struct RegisteredHost {
    ptr: *mut c_void,
}

impl RegisteredHost {
    fn try_register(ptr: *const u8, len: usize) -> Option<Self> {
        if len == 0 {
            return None;
        }
        let ptr = ptr as *mut c_void;
        unsafe { sys::cuMemHostRegister_v2(ptr, len, 0) }
            .result()
            .ok()?;
        Some(Self { ptr })
    }
}

impl Drop for RegisteredHost {
    fn drop(&mut self) {
        let _ = unsafe { sys::cuMemHostUnregister(self.ptr) }.result();
    }
}

/// Bytes a single staging round moves, clamped by the pinned-memory ceiling.
fn stage_bytes() -> usize {
    PIPELINE_STAGE_BYTES.min(PINNED_STAGE_MAX)
}

/// Pull `out.len()` bytes from device address `src` through `slots`.
///
/// Rounds are issued round-robin across the slots and drained in the same
/// order, so with more than one slot the device-to-host DMA of round *n+1* is
/// already in flight while round *n* is being copied out into the caller's
/// buffer. That host copy is the expensive half — measured at 150 µs per
/// 512 KiB against 17 µs of waiting on the DMA — which is why the copy, not
/// the transfer, is what gets spread across threads by the caller.
///
/// The slots are leased exclusively by the caller, so nothing locks in here.
fn read_pipelined(
    ctx: &Arc<CudaContext>,
    slots: &mut [Stage],
    src: u64,
    out: &mut [u8],
) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    ctx.bind_to_thread()
        .context("bind CUDA context for staged read")?;
    let chunk = stage_bytes();
    let n = slots.len().min(out.len().div_ceil(chunk)).max(1);
    let mut pending: Vec<Option<(usize, usize)>> = vec![None; n];
    let mut done = 0usize;
    let mut round = 0usize;
    while done < out.len() {
        let i = round % n;
        if let Some((at, len)) = pending[i].take() {
            let slot = &mut slots[i];
            slot.stream.synchronize()?;
            let buf = slot.buf(len)?;
            out[at..at + len].copy_from_slice(&buf[..len]);
        }
        let take = (out.len() - done).min(chunk);
        let slot = &mut slots[i];
        let buf = slot.buf(take)?;
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                buf.as_mut_ptr() as *mut c_void,
                src + done as u64,
                take,
                slot.stream.cu_stream(),
            )
            .result()
            .context("cuMemcpyDtoHAsync staged")?;
        }
        pending[i] = Some((done, take));
        done += take;
        round += 1;
    }
    for (i, p) in pending.into_iter().enumerate() {
        if let Some((at, len)) = p {
            let slot = &mut slots[i];
            slot.stream.synchronize()?;
            let buf = slot.buf(len)?;
            out[at..at + len].copy_from_slice(&buf[..len]);
        }
    }
    Ok(())
}

/// Push `data` to device address `dst` through `slots`; the mirror of
/// [`read_pipelined`].
///
/// A slot may only be refilled once the DMA reading out of it has finished, so
/// with a single slot every host copy waits on the previous transfer. The
/// second slot is what removes that stall.
fn write_pipelined(
    ctx: &Arc<CudaContext>,
    slots: &mut [Stage],
    dst: u64,
    data: &[u8],
) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    ctx.bind_to_thread()
        .context("bind CUDA context for staged write")?;
    let chunk = stage_bytes();
    let n = slots.len().min(data.len().div_ceil(chunk)).max(1);
    let mut inflight = vec![false; n];
    let mut done = 0usize;
    let mut round = 0usize;
    while done < data.len() {
        let i = round % n;
        let take = (data.len() - done).min(chunk);
        let slot = &mut slots[i];
        if inflight[i] {
            slot.stream.synchronize()?;
        }
        let buf = slot.buf(take)?;
        buf[..take].copy_from_slice(&data[done..done + take]);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst + done as u64,
                buf.as_ptr() as *const c_void,
                take,
                slot.stream.cu_stream(),
            )
            .result()
            .context("cuMemcpyHtoDAsync staged")?;
        }
        inflight[i] = true;
        done += take;
        round += 1;
    }
    for (i, busy) in inflight.into_iter().enumerate() {
        if busy {
            slots[i].stream.synchronize()?;
        }
    }
    Ok(())
}

/// Push `data` straight from page-locked caller memory, striped across the
/// slots' streams. No host copy and no staging buffer — just parallel DMA.
fn write_registered(slots: &[Stage], dst: u64, data: &[u8]) -> Result<()> {
    let n = slots.len().max(1);
    let stripe = data.len().div_ceil(n);
    let mut used = 0usize;
    for (i, slot) in slots.iter().enumerate() {
        let start = i * stripe;
        if start >= data.len() {
            break;
        }
        let take = (data.len() - start).min(stripe);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst + start as u64,
                data[start..start + take].as_ptr() as *const c_void,
                take,
                slot.stream.cu_stream(),
            )
            .result()
            .context("cuMemcpyHtoDAsync registered stripe")?;
        }
        used = i + 1;
    }
    for slot in &slots[..used] {
        slot.stream.synchronize()?;
    }
    Ok(())
}

impl Vram {
    /// Total physical VRAM (bytes) of the given device, without allocating.
    pub fn device_total_mem(ordinal: usize) -> Result<u64> {
        result::init().context("cuInit failed (no CUDA driver / GPU?)")?;
        let dev = result::device::get(ordinal as i32)
            .with_context(|| format!("no CUDA device with ordinal {ordinal}"))?;
        let total = unsafe { result::device::total_mem(dev) }.context("cuDeviceTotalMem failed")?;
        Ok(total as u64)
    }

    /// Name of the given CUDA device (for logging).
    pub fn device_name(ordinal: usize) -> Result<String> {
        result::init().context("cuInit failed (no CUDA driver / GPU?)")?;
        let dev = result::device::get(ordinal as i32)
            .with_context(|| format!("no CUDA device with ordinal {ordinal}"))?;
        let name = result::device::get_name(dev).context("cuDeviceGetName failed")?;
        Ok(name)
    }

    /// Number of CUDA devices visible to the driver, without allocating.
    pub fn device_count() -> Result<usize> {
        result::init().context("cuInit failed (no CUDA driver / GPU?)")?;
        let n = result::device::get_count().context("cuDeviceGetCount failed")?;
        Ok(n.max(0) as usize)
    }

    /// Allocate a zero-initialized contiguous buffer of `size` bytes on `ordinal`.
    pub fn new(ordinal: usize, size: u64) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("failed to create CUDA context on device {ordinal}"))?;
        let stream = ctx.default_stream();
        // No point reserving more staging memory than the buffer can ever hand
        // back; a small test allocation should not pin 32 MiB of host RAM.
        let slot = stage_bytes().min(size.max(1) as usize);
        let h2d = StagePool::new(&ctx, PIPELINE_STAGES, PinnedKind::WriteCombined, slot)?;
        let d2h = StagePool::new(&ctx, PIPELINE_STAGES, PinnedKind::Normal, slot)?;
        let buf = stream
            .alloc_zeros::<u8>(size as usize)
            .with_context(|| format!("failed to allocate {size} bytes of VRAM"))?;
        stream.synchronize().context("stream sync after alloc")?;
        Ok(Self {
            ctx,
            stream,
            buf,
            size,
            h2d,
            d2h,
        })
    }

    /// Total size of the buffer in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Raw device address of the VRAM buffer start. Used by the GPU hash
    /// kernel to address the buffer directly without a separate cudarc view.
    pub fn buf_device_ptr(&self) -> u64 {
        let (ptr, _guard) = self.buf.device_ptr(&self.stream);
        ptr
    }

    /// Bind the primary context to the calling thread. Required before any
    /// memcpy/memset from a thread that hasn't touched CUDA yet (WinFsp
    /// dispatches callbacks from a pool of threads).
    fn bind(&self) -> Result<()> {
        self.ctx
            .bind_to_thread()
            .context("bind CUDA context to thread")?;
        Ok(())
    }

    /// The CUDA context backing this buffer (shared primary context).
    pub fn context(&self) -> Arc<CudaContext> {
        self.ctx.clone()
    }

    /// The CUDA stream used for transfers into/out of this buffer.
    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    /// Order the staging streams behind everything already queued on the
    /// default stream.
    ///
    /// This fence is load-bearing and cannot be dropped: the staging slots'
    /// streams are created non-blocking, so they are *not* ordered against the
    /// default stream, and callers legitimately mix the two — the storage
    /// engine issues [`write_at_async`](Self::write_at_async) (default stream)
    /// and then reads the same chunk back through [`read_at`](Self::read_at)
    /// (staging streams). Without the fence that read can overtake the write.
    /// It costs ~1.5 µs against transfers of 256 KiB and up, which is the only
    /// reason a cheaper device-side event fence is not worth the extra state.
    fn fence_default_stream(&self) -> Result<()> {
        self.stream
            .synchronize()
            .context("fence default stream before staged transfer")
    }

    /// Copy `data` from host into the buffer starting at byte `offset`.
    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        if data.len() < OPTIMIZED_TRANSFER_THRESHOLD {
            self.write_at_async(offset, data)?;
            self.stream.synchronize()?;
            return Ok(());
        }
        let end = offset
            .checked_add(data.len() as u64)
            .filter(|&e| e <= self.size)
            .with_context(|| {
                format!("write_at out of bounds: offset={offset} len={}", data.len())
            })?;
        self.bind()?;
        self.fence_default_stream()?;
        let dst = self.buf_device_ptr() + offset;
        self.write_staged(dst, data)?;
        debug_assert_eq!(end, offset + data.len() as u64);
        Ok(())
    }

    /// Copy `out.len()` bytes from the buffer at byte `offset` into `out`.
    pub fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        if out.len() < OPTIMIZED_TRANSFER_THRESHOLD {
            self.read_at_async(offset, out)?;
            self.stream.synchronize()?;
            return Ok(());
        }
        let end = offset
            .checked_add(out.len() as u64)
            .filter(|&e| e <= self.size)
            .with_context(|| format!("read_at out of bounds: offset={offset} len={}", out.len()))?;
        self.bind()?;
        self.fence_default_stream()?;
        let src = self.buf_device_ptr() + offset;
        self.read_staged(src, out)?;
        debug_assert_eq!(end, offset + out.len() as u64);
        Ok(())
    }

    /// Device-to-host through the staging pool.
    ///
    /// Large transfers are split into [`COPY_SHARDS`] contiguous shards run on
    /// their own threads, because the host-side copy out of the stage — not the
    /// DMA — is the bottleneck, and one thread only reaches ~3.5 GB/s writing
    /// into the buffer WinFsp hands us. Smaller transfers skip the threads and
    /// pipeline on the calling thread instead, over whatever slots the lease
    /// produced.
    fn read_staged(&self, src: u64, out: &mut [u8]) -> Result<()> {
        let rounds = out.len().div_ceil(stage_bytes());
        let shards = if out.len() >= PARALLEL_COPY_THRESHOLD {
            rounds.min(COPY_SHARDS)
        } else {
            1
        };
        let mut lease = self.d2h.lease(rounds.min(shards * SLOTS_PER_SHARD));
        // The pool may have been busy; never ask for more shards than the
        // lease can actually feed.
        let shards = shards.min(lease.slots.len());
        if shards <= 1 {
            return read_pipelined(&self.ctx, &mut lease.slots, src, out);
        }

        let per = lease.slots.len() / shards;
        let bytes = out.len().div_ceil(shards);
        let ctx = &self.ctx;
        std::thread::scope(|sc| -> Result<()> {
            let mut workers = Vec::with_capacity(shards);
            for (i, (group, part)) in lease
                .slots
                .chunks_mut(per)
                .zip(out.chunks_mut(bytes))
                .enumerate()
            {
                let at = src + (i * bytes) as u64;
                workers.push(sc.spawn(move || read_pipelined(ctx, group, at, part)));
            }
            for w in workers {
                w.join()
                    .map_err(|_| anyhow::anyhow!("device-to-host copy worker panicked"))??;
            }
            Ok(())
        })
    }

    /// Host-to-device through the staging pool; the mirror of
    /// [`read_staged`](Self::read_staged).
    fn write_staged(&self, dst: u64, data: &[u8]) -> Result<()> {
        if data.len() >= HOST_REGISTER_THRESHOLD {
            // Borrow the slots purely for their streams; the pinned buffers go
            // unused because the DMA reads the caller's pages directly.
            let lease = self.h2d.lease(COPY_SHARDS);
            if let Some(_pinned) = RegisteredHost::try_register(data.as_ptr(), data.len()) {
                return write_registered(&lease.slots, dst, data);
            }
            // Registration can legitimately fail (an overlapping range is
            // already registered, or the pages cannot be locked); fall through
            // to staging, which needs no cooperation from the caller's memory.
            // `lease` ends with this block, so the slots are back in the pool
            // before the staged path leases again — holding both at once could
            // starve the pool under concurrent writers.
        }
        let rounds = data.len().div_ceil(stage_bytes());
        let shards = if data.len() >= PARALLEL_COPY_THRESHOLD {
            rounds.min(COPY_SHARDS)
        } else {
            1
        };
        let mut lease = self.h2d.lease(rounds.min(shards * SLOTS_PER_SHARD));
        let shards = shards.min(lease.slots.len());
        if shards <= 1 {
            return write_pipelined(&self.ctx, &mut lease.slots, dst, data);
        }

        let per = lease.slots.len() / shards;
        let bytes = data.len().div_ceil(shards);
        let ctx = &self.ctx;
        std::thread::scope(|sc| -> Result<()> {
            let mut workers = Vec::with_capacity(shards);
            for (i, (group, part)) in lease
                .slots
                .chunks_mut(per)
                .zip(data.chunks(bytes))
                .enumerate()
            {
                let at = dst + (i * bytes) as u64;
                workers.push(sc.spawn(move || write_pipelined(ctx, group, at, part)));
            }
            for w in workers {
                w.join()
                    .map_err(|_| anyhow::anyhow!("host-to-device copy worker panicked"))??;
            }
            Ok(())
        })
    }

    /// Enqueue a host-to-device copy without synchronising the stream.
    /// The caller must keep `data` alive and unchanged until [`sync`] completes.
    ///
    /// [`sync`]: Vram::sync
    pub fn write_at_async(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(data.len() as u64)
            .filter(|&e| e <= self.size)
            .with_context(|| {
                format!("write_at out of bounds: offset={offset} len={}", data.len())
            })?;
        self.bind()?;
        let mut view = self.buf.slice_mut(offset as usize..end as usize);
        self.stream.memcpy_htod(data, &mut view)?;
        Ok(())
    }

    /// Enqueue a device-to-host copy without synchronising the stream.
    /// The caller must keep `out` alive and untouched until [`sync`] completes.
    ///
    /// [`sync`]: Vram::sync
    pub fn read_at_async(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(out.len() as u64)
            .filter(|&e| e <= self.size)
            .with_context(|| format!("read_at out of bounds: offset={offset} len={}", out.len()))?;
        self.bind()?;
        let view = self.buf.slice(offset as usize..end as usize);
        self.stream.memcpy_dtoh(&view, out)?;
        Ok(())
    }

    /// Copy `len` bytes within the buffer from `src` to `dst` (used for CoW).
    /// Uses a device-to-device copy so shared dedup chunks can be split without
    /// bouncing 64 KiB through host memory.
    pub fn copy_within(&mut self, src: u64, dst: u64, len: u64) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        src.checked_add(len)
            .filter(|&e| e <= self.size)
            .with_context(|| format!("copy_within source out of bounds: offset={src} len={len}"))?;
        dst.checked_add(len)
            .filter(|&e| e <= self.size)
            .with_context(|| format!("copy_within dest out of bounds: offset={dst} len={len}"))?;
        self.bind()?;
        let base = self.buf_device_ptr();
        unsafe {
            result::memcpy_dtod_async(
                base + dst,
                base + src,
                len as usize,
                self.stream.cu_stream(),
            )
            .context("memcpy_dtod_async (copy_within)")?;
        }
        self.stream.synchronize().context("copy_within sync")?;
        Ok(())
    }

    /// Copy `len` bytes from an external device pointer `src_ptr` into the
    /// buffer at byte `dst_offset` (device-to-device, no host bounce). The copy
    /// is *enqueued* on the stream and not synchronised here; call [`sync`] (or
    /// any synchronising op on the same stream) before reading the bytes back.
    ///
    /// Used to move freshly compressed blobs straight from the codec's scratch
    /// into the packed arena without routing through host memory.
    ///
    /// [`sync`]: Vram::sync
    pub fn copy_dev_into(&self, dst_offset: u64, src_ptr: u64, len: u64) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        dst_offset
            .checked_add(len)
            .filter(|&e| e <= self.size)
            .with_context(|| {
                format!("copy_dev_into out of bounds: offset={dst_offset} len={len}")
            })?;
        self.bind()?;
        let dst = self.buf_device_ptr() + dst_offset;
        unsafe {
            result::memcpy_dtod_async(dst, src_ptr, len as usize, self.stream.cu_stream())
                .context("memcpy_dtod_async (copy_dev_into)")?;
        }
        Ok(())
    }

    /// Synchronise the stream, completing any enqueued (non-synchronising) work
    /// such as [`copy_dev_into`](Vram::copy_dev_into).
    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize().context("vram stream sync")
    }

    /// Zero `len` bytes of the buffer starting at byte `offset`.
    pub fn zero_at(&mut self, offset: u64, len: u64) -> Result<()> {
        self.zero_at_async(offset, len)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Enqueue a memset-to-zero without synchronising the stream.
    pub fn zero_at_async(&mut self, offset: u64, len: u64) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let end = offset
            .checked_add(len)
            .filter(|&e| e <= self.size)
            .with_context(|| format!("zero_at out of bounds: offset={offset} len={len}"))?;
        self.bind()?;
        let mut view = self.buf.slice_mut(offset as usize..end as usize);
        self.stream.memset_zeros(&mut view)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte pattern with a long period, so a stage boundary that duplicates or
    /// drops a run shows up as a mismatch rather than matching by luck.
    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u32).wrapping_mul(2654435761).rotate_left(seed as u32) as u8)
            .collect()
    }

    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn copy_within_uses_device_to_device_path() {
        let mut vram = Vram::new(0, 128 * 1024).expect("test vram");
        let src: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
        vram.write_at(0, &src).unwrap();
        vram.copy_within(0, 64 * 1024, 64 * 1024).unwrap();

        let mut dst = vec![0u8; 64 * 1024];
        vram.read_at(64 * 1024, &mut dst).unwrap();
        assert_eq!(dst, src);
    }

    /// Round-trips sizes that straddle every branch in the staged path: below
    /// [`OPTIMIZED_TRANSFER_THRESHOLD`], one partial round, an exact multiple of
    /// [`PIPELINE_STAGE_BYTES`], the sharded path, and a size that leaves a
    /// ragged final shard.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn staged_transfers_round_trip_every_size_class() {
        // Sized from the constants so retuning them cannot silently shrink what
        // the cases below actually exercise.
        let cap = 10 * PIPELINE_STAGE_BYTES + 65536;
        let mut vram = Vram::new(0, cap as u64).expect("test vram");
        for &len in &[
            1usize,
            4096,
            OPTIMIZED_TRANSFER_THRESHOLD - 1,
            OPTIMIZED_TRANSFER_THRESHOLD,
            PIPELINE_STAGE_BYTES,
            PIPELINE_STAGE_BYTES + 1,
            PARALLEL_COPY_THRESHOLD,
            PARALLEL_COPY_THRESHOLD + 12345,
            COPY_SHARDS * PIPELINE_STAGE_BYTES,
            9 * PIPELINE_STAGE_BYTES + 777,
        ] {
            let src = pattern(len, 3);
            vram.zero_at(0, cap as u64).unwrap();
            vram.write_at(0, &src).unwrap();
            let mut back = vec![0xffu8; len];
            vram.read_at(0, &mut back).unwrap();
            assert_eq!(back, src, "round trip mismatch at len={len}");
            // Nothing past the written range may have been touched.
            let mut tail = vec![0xffu8; 4096];
            vram.read_at(len as u64, &mut tail).unwrap();
            assert!(
                tail.iter().all(|&b| b == 0),
                "staged write overran at len={len}"
            );
        }
    }

    /// The sharded path splits on shard boundaries that have nothing to do with
    /// the caller's offset, so a non-zero, non-stage-aligned device offset is
    /// the case most likely to be mis-addressed.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn staged_transfers_honour_unaligned_offsets() {
        let len = 5 * PIPELINE_STAGE_BYTES + 4097;
        let off = 3 * PIPELINE_STAGE_BYTES as u64 + 1234;
        let cap = off as usize + len + 65536;
        let mut vram = Vram::new(0, cap as u64).expect("test vram");
        vram.zero_at(0, cap as u64).unwrap();
        let src = pattern(len, 7);
        vram.write_at(off, &src).unwrap();

        let mut back = vec![0u8; len];
        vram.read_at(off, &mut back).unwrap();
        assert_eq!(back, src);

        let mut before = vec![0xffu8; 4096];
        vram.read_at(off - 4096, &mut before).unwrap();
        assert!(before.iter().all(|&b| b == 0));
    }

    /// `read_at` takes `&self` and the storage engine's shared read path calls
    /// it from several threads at once. Each transfer must lease its own
    /// staging slots; a slot shared between two readers would interleave bytes.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn concurrent_reads_do_not_interleave_staging_slots() {
        const REGIONS: usize = 8;
        let len = 3 * PIPELINE_STAGE_BYTES;
        let mut vram = Vram::new(0, (REGIONS * len) as u64).expect("test vram");
        for r in 0..REGIONS {
            vram.write_at((r * len) as u64, &pattern(len, r as u8))
                .unwrap();
        }
        let vram = &vram;
        std::thread::scope(|sc| {
            let mut hs = Vec::new();
            for r in 0..REGIONS {
                hs.push(sc.spawn(move || {
                    for _ in 0..4 {
                        let mut got = vec![0u8; len];
                        vram.read_at((r * len) as u64, &mut got).unwrap();
                        assert_eq!(got, pattern(len, r as u8), "region {r} was corrupted");
                    }
                }));
            }
            for h in hs {
                h.join().unwrap();
            }
        });
    }

    /// Guards the invariant the module doc rests on: a transfer must never
    /// leave the *caller's* memory page-locked once it has returned. If it did,
    /// the buffer could be freed and its address reused while CUDA still held a
    /// mapping for it — see `cached_registration_would_outlive_its_buffer` for
    /// what that costs.
    ///
    /// Both sizes matter: below [`HOST_REGISTER_THRESHOLD`] the write path
    /// stages and never registers anything, at or above it the write path
    /// registers the caller's buffer for the duration of the call and must give
    /// it back. `cuMemHostRegister` fails with `HOST_MEMORY_ALREADY_REGISTERED`
    /// on a range overlapping a live registration, so its success here is the
    /// proof that nothing is still pinned.
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn transfers_leave_no_caller_memory_registered() {
        for len in [4 * PIPELINE_STAGE_BYTES, HOST_REGISTER_THRESHOLD + 4096] {
            let mut vram = Vram::new(0, len as u64).expect("test vram");
            let src = pattern(len, 11);
            let mut dst = vec![0u8; len];
            vram.write_at(0, &src).unwrap();
            vram.read_at(0, &mut dst).unwrap();
            assert_eq!(dst, src, "round trip mismatch at len={len}");

            vram.bind().unwrap();
            for buf in [src.as_ptr(), dst.as_ptr()] {
                let r = unsafe { sys::cuMemHostRegister_v2(buf as *mut c_void, len, 0) }.result();
                assert!(
                    r.is_ok(),
                    "a transfer of {len} bytes left caller memory page-locked: {r:?}"
                );
                unsafe { sys::cuMemHostUnregister(buf as *mut c_void) }
                    .result()
                    .unwrap();
            }
        }
    }

    /// Documents, by reproducing it, why the registration cache described in
    /// the module docs is not implementable safely.
    ///
    /// A range is registered with CUDA and then released back to the OS
    /// *without* being unregistered — exactly the state a cache that outlives
    /// its buffer would be in. The address is then re-committed, and CUDA still
    /// refuses to register it: the driver is holding a mapping for memory the
    /// process no longer owns and has no idea the pages went away. Whether that
    /// stale mapping then resolves to the old physical pages is down to what
    /// the OS does with them — sometimes it hands the same frames back, and a
    /// probe run on this machine with an 8 MiB range read `0xAA` through CUDA
    /// while the CPU read `0xBB` from the same address. The driver's
    /// obliviousness asserted here is the deterministic part, and it is enough:
    /// a cached registration cannot be validated, so it cannot be trusted.
    ///
    /// If this ever starts failing because a driver learned to drop its
    /// mappings on free, the caching optimisation becomes available and the
    /// module docs should be revisited.
    #[cfg(windows)]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires an NVIDIA GPU")]
    #[test]
    fn cached_registration_would_outlive_its_buffer() {
        #[link(name = "kernel32")]
        extern "system" {
            fn VirtualAlloc(addr: *mut c_void, size: usize, typ: u32, prot: u32) -> *mut c_void;
            fn VirtualFree(addr: *mut c_void, size: usize, typ: u32) -> i32;
        }
        const MEM_COMMIT: u32 = 0x1000;
        const MEM_RESERVE: u32 = 0x2000;
        const MEM_RELEASE: u32 = 0x8000;
        const PAGE_READWRITE: u32 = 0x04;
        const LEN: usize = 2 * 1024 * 1024;

        let vram = Vram::new(0, LEN as u64).expect("test vram");
        vram.bind().unwrap();

        let first = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                LEN,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        assert!(!first.is_null(), "VirtualAlloc failed");
        unsafe { sys::cuMemHostRegister_v2(first, LEN, 0) }
            .result()
            .expect("cuMemHostRegister");

        // Release the pages while CUDA still holds the registration, then take
        // the same address back.
        assert_ne!(unsafe { VirtualFree(first, 0, MEM_RELEASE) }, 0);
        let again = unsafe { VirtualAlloc(first, LEN, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
        if again != first {
            // The allocator did not hand the address back; nothing to show.
            if !again.is_null() {
                unsafe { VirtualFree(again, 0, MEM_RELEASE) };
            }
            let _ = unsafe { sys::cuMemHostUnregister(first) }.result();
            return;
        }

        let reregister = unsafe { sys::cuMemHostRegister_v2(again, LEN, 0) }.result();
        let _ = unsafe { sys::cuMemHostUnregister(again) }.result();
        unsafe { VirtualFree(again, 0, MEM_RELEASE) };

        assert!(
            reregister.is_err(),
            "the CUDA driver dropped its registration when the memory was freed              — a registration cache may now be safe; re-read the module docs              before adding one"
        );
    }
}
