//! CUDA layer: owns the single large VRAM buffer and moves bytes in/out of it.
//!
//! The buffer is one contiguous allocation in device memory. Higher layers
//! address it by byte offset; chunk math lives in [`crate::chunk`].

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cudarc::driver::{result, sys, CudaContext, CudaSlice, CudaStream, DevicePtr};

const PINNED_STAGE_MAX: usize = 64 * 1024 * 1024;
const OPTIMIZED_TRANSFER_THRESHOLD: usize = 1024 * 1024;
const HOST_REGISTER_THRESHOLD: usize = 4 * 1024 * 1024;
const TRANSFER_STREAMS: usize = 4;
const PIPELINE_STAGE_BYTES: usize = 16 * 1024 * 1024;
const PIPELINE_STAGES: usize = 2;

/// Owns the VRAM allocation and the CUDA context/stream used to touch it.
pub struct Vram {
    /// Kept alive so the primary context stays retained for the buffer's life.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    transfer_streams: Vec<Arc<CudaStream>>,
    buf: CudaSlice<u8>,
    size: u64,
    h2d_stage: Mutex<Vec<PinnedStage>>,
    d2h_stage: Mutex<Vec<PinnedStage>>,
}

#[derive(Clone, Copy)]
enum PinnedKind {
    Normal,
    WriteCombined,
}

struct PinnedStage {
    ptr: Option<std::ptr::NonNull<u8>>,
    cap: usize,
    kind: PinnedKind,
}

unsafe impl Send for PinnedStage {}
unsafe impl Sync for PinnedStage {}

impl PinnedStage {
    fn new(kind: PinnedKind) -> Self {
        Self {
            ptr: None,
            cap: 0,
            kind,
        }
    }

    fn slice_mut<'a>(&'a mut self, ctx: &Arc<CudaContext>, len: usize) -> Result<&'a mut [u8]> {
        self.reserve(ctx, len)?;
        let ptr = self.ptr.context("pinned stage not allocated")?;
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) })
    }

    fn reserve(&mut self, ctx: &Arc<CudaContext>, len: usize) -> Result<()> {
        if len <= self.cap {
            return Ok(());
        }
        self.release();
        ctx.bind_to_thread()
            .context("bind CUDA context for pinned host alloc")?;
        let flags = match self.kind {
            PinnedKind::Normal => 0,
            PinnedKind::WriteCombined => sys::CU_MEMHOSTALLOC_WRITECOMBINED,
        };
        let ptr = unsafe { result::malloc_host(len, flags) }.context("cuMemHostAlloc")?;
        let ptr = std::ptr::NonNull::new(ptr as *mut u8).context("cuMemHostAlloc returned null")?;
        self.ptr = Some(ptr);
        self.cap = len;
        Ok(())
    }

    fn release(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            let _ = unsafe { result::free_host(ptr.as_ptr() as *mut c_void) };
        }
        self.cap = 0;
    }
}

impl Drop for PinnedStage {
    fn drop(&mut self) {
        self.release();
    }
}

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

fn sync_used_streams(streams: &[Arc<CudaStream>], used: &[usize]) -> Result<()> {
    let mut seen = [false; TRANSFER_STREAMS];
    for &idx in used {
        if idx < streams.len() && !seen[idx] {
            streams[idx].synchronize()?;
            seen[idx] = true;
        }
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
        result::init().ok();
        let dev = result::device::get(ordinal as i32)?;
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
        let mut transfer_streams = Vec::with_capacity(TRANSFER_STREAMS);
        for _ in 0..TRANSFER_STREAMS {
            transfer_streams.push(ctx.new_stream().context("create transfer stream")?);
        }
        let buf = stream
            .alloc_zeros::<u8>(size as usize)
            .with_context(|| format!("failed to allocate {size} bytes of VRAM"))?;
        stream.synchronize().context("stream sync after alloc")?;
        Ok(Self {
            ctx,
            stream,
            transfer_streams,
            buf,
            size,
            h2d_stage: Mutex::new(
                (0..PIPELINE_STAGES)
                    .map(|_| PinnedStage::new(PinnedKind::WriteCombined))
                    .collect(),
            ),
            d2h_stage: Mutex::new(
                (0..PIPELINE_STAGES)
                    .map(|_| PinnedStage::new(PinnedKind::Normal))
                    .collect(),
            ),
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
        ptr as u64
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
        let base = self.buf_device_ptr();
        if data.len() >= HOST_REGISTER_THRESHOLD {
            if let Some(_registered) = RegisteredHost::try_register(data.as_ptr(), data.len()) {
                self.stream.synchronize()?;
                self.copy_registered_h2d(base + offset, data)?;
                debug_assert_eq!(end, offset + data.len() as u64);
                return Ok(());
            }
        }
        let mut done = 0usize;
        let mut stages = self.h2d_stage.lock().unwrap_or_else(|e| e.into_inner());
        let mut used_streams = Vec::new();
        while done < data.len() {
            let stage_idx = (done / PIPELINE_STAGE_BYTES) % stages.len();
            let stream_idx = stage_idx % self.transfer_streams.len();
            let stream = &self.transfer_streams[stream_idx];
            stream.synchronize()?;
            let take = (data.len() - done).min(PIPELINE_STAGE_BYTES.min(PINNED_STAGE_MAX));
            let slice = stages[stage_idx].slice_mut(&self.ctx, take)?;
            slice[..take].copy_from_slice(&data[done..done + take]);
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    base + offset + done as u64,
                    slice.as_ptr() as *const c_void,
                    take,
                    stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyHtoDAsync pinned")?;
            }
            used_streams.push(stream_idx);
            done += take;
        }
        sync_used_streams(&self.transfer_streams, &used_streams)?;
        debug_assert_eq!(end, offset + data.len() as u64);
        Ok(())
    }

    fn copy_registered_h2d(&self, dst: u64, data: &[u8]) -> Result<()> {
        let stripe = data.len().div_ceil(self.transfer_streams.len());
        let mut used = Vec::new();
        for (i, stream) in self.transfer_streams.iter().enumerate() {
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
                    stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyHtoDAsync registered stripe")?;
            }
            used.push(i);
        }
        sync_used_streams(&self.transfer_streams, &used)
    }

    /// Enqueue a host-to-device copy without synchronising the stream.
    /// The caller must keep `data` alive and unchanged until [`sync`] completes.
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
        let base = self.buf_device_ptr();
        if out.len() >= HOST_REGISTER_THRESHOLD {
            if let Some(_registered) = RegisteredHost::try_register(out.as_mut_ptr(), out.len()) {
                self.stream.synchronize()?;
                self.copy_registered_d2h(base + offset, out)?;
                debug_assert_eq!(end, offset + out.len() as u64);
                return Ok(());
            }
        }
        let mut done = 0usize;
        let mut stages = self.d2h_stage.lock().unwrap_or_else(|e| e.into_inner());
        let mut pending: Vec<Option<(usize, usize)>> = vec![None; stages.len()];
        while done < out.len() {
            let stage_idx = (done / PIPELINE_STAGE_BYTES) % stages.len();
            let stream_idx = stage_idx % self.transfer_streams.len();
            let stream = &self.transfer_streams[stream_idx];
            if let Some((prev_done, prev_take)) = pending[stage_idx].take() {
                stream.synchronize()?;
                let slice = stages[stage_idx].slice_mut(&self.ctx, prev_take)?;
                out[prev_done..prev_done + prev_take].copy_from_slice(&slice[..prev_take]);
            }
            let take = (out.len() - done).min(PIPELINE_STAGE_BYTES.min(PINNED_STAGE_MAX));
            let slice = stages[stage_idx].slice_mut(&self.ctx, take)?;
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    slice.as_mut_ptr() as *mut c_void,
                    base + offset + done as u64,
                    take,
                    stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyDtoHAsync pinned")?;
            }
            pending[stage_idx] = Some((done, take));
            done += take;
        }
        for (stage_idx, pending) in pending.into_iter().enumerate() {
            if let Some((prev_done, prev_take)) = pending {
                let stream_idx = stage_idx % self.transfer_streams.len();
                self.transfer_streams[stream_idx].synchronize()?;
                let slice = stages[stage_idx].slice_mut(&self.ctx, prev_take)?;
                out[prev_done..prev_done + prev_take].copy_from_slice(&slice[..prev_take]);
            }
        }
        debug_assert_eq!(end, offset + out.len() as u64);
        Ok(())
    }

    fn copy_registered_d2h(&self, src: u64, out: &mut [u8]) -> Result<()> {
        let stripe = out.len().div_ceil(self.transfer_streams.len());
        let mut used = Vec::new();
        for (i, stream) in self.transfer_streams.iter().enumerate() {
            let start = i * stripe;
            if start >= out.len() {
                break;
            }
            let take = (out.len() - start).min(stripe);
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    out[start..start + take].as_mut_ptr() as *mut c_void,
                    src + start as u64,
                    take,
                    stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyDtoHAsync registered stripe")?;
            }
            used.push(i);
        }
        sync_used_streams(&self.transfer_streams, &used)
    }

    /// Enqueue a device-to-host copy without synchronising the stream.
    /// The caller must keep `out` alive and untouched until [`sync`] completes.
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
}
