//! GPU-parallel FNV-1a chunk hasher (batched).
//!
//! Loads a PTX kernel that hashes one or more 64 KiB chunks already resident in
//! VRAM without moving the chunk data to the host. The two-level algorithm
//! (256 threads × 256 bytes each, then thread 0 reduces the 256 per-thread
//! hashes) is mirrored exactly on the CPU side in `engine.rs`, so both paths
//! produce identical hashes for identical data.
//!
//! One launch hashes a whole batch: `gridDim=(num_chunks,1,1)` puts one block
//! per chunk, so the GPU runs many blocks across its SMs in parallel instead of
//! the single block the chunk-at-a-time path used. Only the per-chunk offset
//! and result arrays cross the bus (8 bytes each way per chunk), never the
//! chunk bodies.
//!
//! The kernel is launched on the same stream the VRAM writes use, so it is
//! ordered after any prior writes without a whole-device synchronize.

use std::ffi::{c_void, CString};
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::result as dr;
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};

use crate::cuda::Vram;

/// PTX source compiled into the binary.
const PTX: &[u8] = include_bytes!("hash_kernel.ptx");

/// Loaded GPU hasher. Holds the CUDA module, function handle, and device
/// scratch for the per-chunk offset inputs and hash outputs. The scratch grows
/// on demand to fit the largest batch seen.
pub struct GpuHasher {
    module: sys::CUmodule,
    func: sys::CUfunction,
    offs_d: CudaSlice<u64>,
    out_d: CudaSlice<u64>,
    cap: usize,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

// SAFETY: CUmodule and CUfunction are opaque device handles. The context
// is always bound before any operation, so cross-thread use is safe under
// the same guarantees as the rest of the cudarc usage in this codebase.
unsafe impl Send for GpuHasher {}
unsafe impl Sync for GpuHasher {}

impl Drop for GpuHasher {
    fn drop(&mut self) {
        unsafe {
            dr::module::unload(self.module).ok();
        }
    }
}

impl GpuHasher {
    /// Load the PTX kernel, sharing `vram`'s CUDA context and stream.
    pub fn new(vram: &Vram) -> Result<Self> {
        let ctx = vram.context();
        let stream = vram.stream();
        ctx.bind_to_thread().context("bind ctx for GpuHasher")?;

        // PTX must be NUL-terminated for cuModuleLoadData.
        let mut ptx_nul = PTX.to_vec();
        ptx_nul.push(0);

        let module = unsafe {
            dr::module::load_data(ptx_nul.as_ptr() as *const c_void)
                .context("cuModuleLoadData for hash_kernel.ptx")?
        };
        let fname = CString::new("fnv1a_hash").unwrap();
        let func = unsafe {
            dr::module::get_function(module, fname).context("cuModuleGetFunction fnv1a_hash")?
        };

        // Start with room for a single chunk; grow on demand.
        let cap = 1usize;
        let offs_d = stream
            .alloc_zeros::<u64>(cap)
            .context("alloc hash offs buf")?;
        let out_d = stream
            .alloc_zeros::<u64>(cap)
            .context("alloc hash out buf")?;
        stream.synchronize().context("sync after hash buf alloc")?;

        Ok(GpuHasher {
            module,
            func,
            offs_d,
            out_d,
            cap,
            ctx,
            stream,
        })
    }

    /// Ensure the offset/output scratch can hold `n` chunks.
    fn reserve(&mut self, n: usize) -> Result<()> {
        if n <= self.cap {
            return Ok(());
        }
        self.offs_d = self
            .stream
            .alloc_zeros::<u64>(n)
            .context("grow hash offs buf")?;
        self.out_d = self
            .stream
            .alloc_zeros::<u64>(n)
            .context("grow hash out buf")?;
        self.stream
            .synchronize()
            .context("sync after hash buf grow")?;
        self.cap = n;
        Ok(())
    }

    /// Hash one 64 KiB chunk at byte `chunk_offset` within the VRAM buffer whose
    /// device base address is `vram_base`. Returns the 64-bit hash.
    ///
    /// Precondition: any writes to the chunk have been issued on the same stream
    /// (they are, via `Vram`), so the launch is correctly ordered after them.
    pub fn hash_chunk(&mut self, vram_base: u64, chunk_offset: u64) -> Result<u64> {
        let mut out = [0u64; 1];
        self.hash_chunks(vram_base, &[chunk_offset], &mut out)?;
        Ok(out[0])
    }

    /// Hash a batch of chunks in a single kernel launch. `offsets[i]` is the
    /// byte offset of chunk `i` within the VRAM buffer; `out[i]` receives its
    /// hash. `out.len()` must equal `offsets.len()`.
    pub fn hash_chunks(&mut self, vram_base: u64, offsets: &[u64], out: &mut [u64]) -> Result<()> {
        debug_assert_eq!(offsets.len(), out.len());
        let n = offsets.len();
        if n == 0 {
            return Ok(());
        }
        self.ctx.bind_to_thread()?;
        self.reserve(n)?;

        // Upload the per-chunk offsets.
        {
            let mut view = self.offs_d.slice_mut(0..n);
            self.stream.memcpy_htod(offsets, &mut view)?;
        }

        // Device addresses of the offset and output arrays.
        let offs_ptr = {
            let (p, _g) = self.offs_d.device_ptr(&self.stream);
            p
        };
        let out_ptr = {
            let (p, _g) = self.out_d.device_ptr(&self.stream);
            p
        };

        // Kernel arguments: (base: u64, offs_ptr: CUdeviceptr, out_ptr: CUdeviceptr).
        let mut p0 = vram_base;
        let mut p1 = offs_ptr;
        let mut p2 = out_ptr;
        let mut params: [*mut c_void; 3] = [
            &mut p0 as *mut u64 as *mut c_void,
            &mut p1 as *mut sys::CUdeviceptr as *mut c_void,
            &mut p2 as *mut sys::CUdeviceptr as *mut c_void,
        ];

        // One block per chunk, 256 threads per block. Launch on the VRAM stream
        // so it serialises after prior writes; a single stream sync fences it.
        let raw_stream = self.stream.cu_stream();
        unsafe {
            dr::launch_kernel(
                self.func,
                (n as u32, 1, 1), // grid: one block per chunk
                (256, 1, 1),      // block: 256 threads
                0,                // smem declared statically in PTX
                raw_stream,
                &mut params,
            )
            .context("cuLaunchKernel fnv1a_hash")?;
        }

        // Read back the n results (one stream sync covers launch + copy).
        let view = self.out_d.slice(0..n);
        self.stream
            .memcpy_dtoh(&view, out)
            .context("hash result D2H")?;
        self.stream.synchronize().context("sync after hash D2H")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CHUNK_SIZE;

    /// CPU reference (same two-level FNV-1a as engine.rs / the PTX kernel).
    fn fnv1a_ref(data: &[u8]) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        let seg: Vec<u64> = (0..256)
            .map(|t| {
                let mut h = BASIS;
                for &b in &data[t * 256..(t + 1) * 256] {
                    h ^= b as u64;
                    h = h.wrapping_mul(PRIME);
                }
                h
            })
            .collect();
        let mut h = BASIS;
        for sh in seg {
            for i in 0..8u64 {
                h ^= (sh >> (i * 8)) & 0xff;
                h = h.wrapping_mul(PRIME);
            }
        }
        h
    }

    // Requires a GPU. Validates that a single batched launch hashes many chunks
    // and that each matches the CPU reference.
    #[test]
    fn batch_matches_cpu() {
        let chunks = 8usize;
        let cs = CHUNK_SIZE as usize;
        let mut vram = Vram::new(0, (chunks as u64) * CHUNK_SIZE).expect("vram");
        let base = vram.buf_device_ptr();

        // Distinct content per chunk.
        let mut expected = vec![0u64; chunks];
        for c in 0..chunks {
            let data: Vec<u8> = (0..cs).map(|i| ((i + c * 7) as u8) ^ (c as u8)).collect();
            vram.write_at(c as u64 * CHUNK_SIZE, &data).unwrap();
            expected[c] = fnv1a_ref(&data);
        }

        let mut hasher = GpuHasher::new(&vram).unwrap();
        let offsets: Vec<u64> = (0..chunks as u64).map(|c| c * CHUNK_SIZE).collect();
        let mut out = vec![0u64; chunks];
        hasher.hash_chunks(base, &offsets, &mut out).unwrap();
        assert_eq!(out, expected, "batched GPU hash mismatch vs CPU");

        // Single-chunk path agrees too.
        assert_eq!(hasher.hash_chunk(base, 0).unwrap(), expected[0]);
    }
}
