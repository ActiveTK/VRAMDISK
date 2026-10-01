//! CUDA kernels for virtual internal APIs.
//!
//! File hashing is deliberately implemented as a generic streaming API kernel:
//! Rust passes an algorithm id plus a list of device-memory segments, and the
//! CUDA side updates one digest state. File bytes never round-trip through host
//! memory; only small descriptors and the final digest cross the bus.

use std::ffi::{c_void, CString};
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::result as dr;
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::cuda::Vram;

const MAX_DIGEST: usize = 32;
const DEFAULT_SEG_CAP: usize = 256;
const STATE_BYTES: usize = 256;
const MANY_THREADS_PER_BLOCK: u32 = 128;

/// Longest search pattern the kernel will take. Long literals are rare and the
/// per-thread inner loop is linear in this, so a generous but bounded cap keeps
/// the device-side buffer a fixed allocation.
pub const SEARCH_MAX_PATTERN: usize = 256;

/// How many match offsets one launch can record. Hits past this are still
/// counted -- only the recorded offsets are capped -- so a pathological pattern
/// reports an honest total instead of silently truncating it.
const SEARCH_HIT_CAP: usize = 1 << 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HashAlgorithm {
    Md5 = 1,
    Sha1 = 2,
    Sha256 = 3,
    Fnv1a64 = 4,
}

impl HashAlgorithm {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "md5" => Some(Self::Md5),
            "sha1" | "sha-1" => Some(Self::Sha1),
            "sha256" | "sha-256" => Some(Self::Sha256),
            "fnv1a64" | "fnv-1a-64" => Some(Self::Fnv1a64),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Md5 => "md5",
            Self::Sha1 => "sha1",
            Self::Sha256 => "sha256",
            Self::Fnv1a64 => "fnv1a64",
        }
    }

    pub fn digest_len(self) -> usize {
        match self {
            Self::Md5 => 16,
            Self::Sha1 => 20,
            Self::Sha256 => 32,
            Self::Fnv1a64 => 8,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct HashSegment {
    pub ptr: u64,
    pub len: u32,
    /// 0 = read bytes from `ptr`; 1 = synthesize `len` zero bytes.
    pub kind: u32,
}

unsafe impl DeviceRepr for HashSegment {}
unsafe impl ValidAsZeroBits for HashSegment {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HashFileDesc {
    pub seg_start: u32,
    pub seg_count: u32,
}

unsafe impl DeviceRepr for HashFileDesc {}
unsafe impl ValidAsZeroBits for HashFileDesc {}

/// Loaded API kernel module plus persistent digest state and descriptor scratch.
/// One search launch's result: every match counted, and the offsets that fit.
#[derive(Debug, Default, Clone)]
pub struct SearchLaunch {
    /// Total matches in the scanned window, including any beyond the offset cap.
    pub total: u64,
    /// Ascending match offsets, at most [`SEARCH_HIT_CAP`] of them.
    pub offsets: Vec<u64>,
}

pub struct ApiKernel {
    module: sys::CUmodule,
    init_func: sys::CUfunction,
    update_func: sys::CUfunction,
    final_func: sys::CUfunction,
    many_init_func: sys::CUfunction,
    many_update_func: sys::CUfunction,
    many_final_func: sys::CUfunction,
    crc32_init_func: sys::CUfunction,
    crc32_many_func: sys::CUfunction,
    crc32_final_func: sys::CUfunction,
    b64_encode_func: sys::CUfunction,
    b64_decode_func: sys::CUfunction,
    hex_encode_func: sys::CUfunction,
    hex_decode_func: sys::CUfunction,
    search_func: sys::CUfunction,
    segs_d: CudaSlice<HashSegment>,
    files_d: CudaSlice<HashFileDesc>,
    state_d: CudaSlice<u8>,
    states_d: CudaSlice<u8>,
    crc_states_d: CudaSlice<u32>,
    out_d: CudaSlice<u8>,
    outs_d: CudaSlice<u8>,
    crc_out_d: CudaSlice<u32>,
    status_d: CudaSlice<u32>,
    needle_d: CudaSlice<u8>,
    hits_d: CudaSlice<u64>,
    hit_count_d: CudaSlice<u64>,
    seg_cap: usize,
    file_cap: usize,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

unsafe impl Send for ApiKernel {}
unsafe impl Sync for ApiKernel {}

impl Drop for ApiKernel {
    fn drop(&mut self) {
        unsafe {
            dr::module::unload(self.module).ok();
        }
    }
}

/// Unloads a `CUmodule` unless it is handed off with [`into_raw`].
///
/// `ApiKernel::new` resolves nine kernel functions and allocates a dozen device
/// buffers *after* `cuModuleLoadData` succeeds; any `?` in between would
/// otherwise leak the module, and `new` is retried on every call that needs the
/// kernels.
///
/// [`into_raw`]: ModuleGuard::into_raw
struct ModuleGuard(sys::CUmodule);

impl ModuleGuard {
    /// Give the module to a fully constructed [`ApiKernel`], which unloads it
    /// in its own `Drop`.
    fn into_raw(self) -> sys::CUmodule {
        std::mem::ManuallyDrop::new(self).0
    }
}

impl Drop for ModuleGuard {
    fn drop(&mut self) {
        unsafe {
            dr::module::unload(self.0).ok();
        }
    }
}

impl ApiKernel {
    pub fn new(vram: &Vram) -> Result<Self> {
        let ctx = vram.context();
        let stream = vram.stream();
        ctx.bind_to_thread().context("bind ctx for ApiKernel")?;

        let ptx_nul = compiled_api_ptx()?;
        // Guarded from here on: every `?` below must unload the module again.
        let module = ModuleGuard(unsafe {
            dr::module::load_data(ptx_nul.as_ptr() as *const c_void)
                .context("cuModuleLoadData for API kernels")?
        });
        let init_func = get_func(module.0, "vramdisk_hash_init")?;
        let update_func = get_func(module.0, "vramdisk_hash_update")?;
        let final_func = get_func(module.0, "vramdisk_hash_final")?;
        let many_init_func = get_func(module.0, "vramdisk_hash_many_init")?;
        let many_update_func = get_func(module.0, "vramdisk_hash_many_update")?;
        let many_final_func = get_func(module.0, "vramdisk_hash_many_final")?;
        let crc32_init_func = get_func(module.0, "vramdisk_crc32_many_init")?;
        let crc32_many_func = get_func(module.0, "vramdisk_crc32_many")?;
        let crc32_final_func = get_func(module.0, "vramdisk_crc32_many_final")?;
        let b64_encode_func = get_func(module.0, "vramdisk_b64_encode")?;
        let b64_decode_func = get_func(module.0, "vramdisk_b64_decode")?;
        let hex_encode_func = get_func(module.0, "vramdisk_hex_encode")?;
        let hex_decode_func = get_func(module.0, "vramdisk_hex_decode")?;
        let search_func = get_func(module.0, "vramdisk_search")?;

        let segs_d = stream
            .alloc_zeros::<HashSegment>(DEFAULT_SEG_CAP)
            .context("alloc API segment scratch")?;
        let files_d = stream
            .alloc_zeros::<HashFileDesc>(1)
            .context("alloc API file scratch")?;
        let state_d = stream
            .alloc_zeros::<u8>(STATE_BYTES)
            .context("alloc API state")?;
        let states_d = stream
            .alloc_zeros::<u8>(STATE_BYTES)
            .context("alloc API batch states")?;
        let crc_states_d = stream
            .alloc_zeros::<u32>(1)
            .context("alloc API CRC32 states")?;
        let out_d = stream
            .alloc_zeros::<u8>(MAX_DIGEST)
            .context("alloc API digest")?;
        let outs_d = stream
            .alloc_zeros::<u8>(MAX_DIGEST)
            .context("alloc API batch digests")?;
        let crc_out_d = stream
            .alloc_zeros::<u32>(1)
            .context("alloc API CRC32 output")?;
        let status_d = stream.alloc_zeros::<u32>(1).context("alloc API status")?;
        let needle_d = stream
            .alloc_zeros::<u8>(SEARCH_MAX_PATTERN)
            .context("alloc API search pattern")?;
        let hits_d = stream
            .alloc_zeros::<u64>(SEARCH_HIT_CAP)
            .context("alloc API search hits")?;
        let hit_count_d = stream
            .alloc_zeros::<u64>(1)
            .context("alloc API search hit count")?;
        stream.synchronize()?;

        Ok(Self {
            // Construction succeeded: `ApiKernel::drop` owns the unload now.
            module: module.into_raw(),
            init_func,
            update_func,
            final_func,
            many_init_func,
            many_update_func,
            many_final_func,
            crc32_init_func,
            crc32_many_func,
            crc32_final_func,
            b64_encode_func,
            b64_decode_func,
            hex_encode_func,
            hex_decode_func,
            search_func,
            segs_d,
            files_d,
            state_d,
            states_d,
            crc_states_d,
            out_d,
            outs_d,
            crc_out_d,
            status_d,
            needle_d,
            hits_d,
            hit_count_d,
            seg_cap: DEFAULT_SEG_CAP,
            file_cap: 1,
            ctx,
            stream,
        })
    }

    pub fn max_segments(&self) -> usize {
        self.seg_cap
    }

    fn reserve_segments(&mut self, n: usize) -> Result<()> {
        if n <= self.seg_cap {
            return Ok(());
        }
        self.segs_d = self
            .stream
            .alloc_zeros::<HashSegment>(n)
            .context("grow API segment scratch")?;
        self.stream.synchronize()?;
        self.seg_cap = n;
        Ok(())
    }

    fn reserve_files(&mut self, n: usize) -> Result<()> {
        if n <= self.file_cap {
            return Ok(());
        }
        self.files_d = self
            .stream
            .alloc_zeros::<HashFileDesc>(n)
            .context("grow API file scratch")?;
        self.states_d = self
            .stream
            .alloc_zeros::<u8>(n * STATE_BYTES)
            .context("grow API batch states")?;
        self.crc_states_d = self
            .stream
            .alloc_zeros::<u32>(n)
            .context("grow API CRC32 states")?;
        self.outs_d = self
            .stream
            .alloc_zeros::<u8>(n * MAX_DIGEST)
            .context("grow API batch digests")?;
        self.crc_out_d = self
            .stream
            .alloc_zeros::<u32>(n)
            .context("grow API CRC32 output")?;
        self.stream.synchronize()?;
        self.file_cap = n;
        Ok(())
    }

    pub fn begin(&mut self, alg: HashAlgorithm) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let mut alg_id = alg as u32;
        let state_ptr = ptr_u8(&self.state_d, &self.stream);
        let status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut p_state = state_ptr;
        let mut p_status = status_ptr;
        let mut params: [*mut c_void; 3] = [
            &mut alg_id as *mut u32 as *mut c_void,
            &mut p_state as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.init_func,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_hash_init")?;
        }
        self.stream.synchronize()?;
        self.check_status()
    }

    pub fn update(&mut self, segments: &[HashSegment]) -> Result<()> {
        if segments.is_empty() {
            return Ok(());
        }
        self.ctx.bind_to_thread()?;
        self.reserve_segments(segments.len())?;
        {
            let mut view = self.segs_d.slice_mut(0..segments.len());
            self.stream.memcpy_htod(segments, &mut view)?;
        }
        let segs_ptr = ptr_seg(&self.segs_d, &self.stream);
        let state_ptr = ptr_u8(&self.state_d, &self.stream);
        let status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut p_segs = segs_ptr;
        let mut n = segments.len() as u32;
        let mut p_state = state_ptr;
        let mut p_status = status_ptr;
        let mut params: [*mut c_void; 4] = [
            &mut p_segs as *mut sys::CUdeviceptr as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
            &mut p_state as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.update_func,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_hash_update")?;
        }
        self.stream.synchronize()?;
        self.check_status()
    }

    pub fn finish(&mut self, alg: HashAlgorithm) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        let mut alg_id = alg as u32;
        let state_ptr = ptr_u8(&self.state_d, &self.stream);
        let out_ptr = ptr_u8(&self.out_d, &self.stream);
        let status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut p_state = state_ptr;
        let mut p_out = out_ptr;
        let mut p_status = status_ptr;
        let mut params: [*mut c_void; 4] = [
            &mut alg_id as *mut u32 as *mut c_void,
            &mut p_state as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_out as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.final_func,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_hash_final")?;
        }
        let mut out = vec![0u8; alg.digest_len()];
        let view = self.out_d.slice(0..out.len());
        self.stream.memcpy_dtoh(&view, &mut out)?;
        self.stream.synchronize()?;
        self.check_status()?;
        Ok(out)
    }

    pub fn begin_many(&mut self, alg: HashAlgorithm, nfiles: usize) -> Result<()> {
        if nfiles == 0 {
            return Ok(());
        }
        self.ctx.bind_to_thread()?;
        self.reserve_files(nfiles)?;
        let zero = [0u32; 1];
        self.stream.memcpy_htod(&zero, &mut self.status_d)?;

        let states_ptr = ptr_u8(&self.states_d, &self.stream);
        let status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut alg_id = alg as u32;
        let mut n = nfiles as u32;
        let mut p_states = states_ptr;
        let mut p_status = status_ptr;
        let mut params: [*mut c_void; 4] = [
            &mut alg_id as *mut u32 as *mut c_void,
            &mut p_states as *mut sys::CUdeviceptr as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.many_init_func,
                many_grid(nfiles),
                (MANY_THREADS_PER_BLOCK, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_hash_many_init")?;
        }
        self.stream.synchronize()?;
        self.check_status()
    }

    fn upload_many_files(&mut self, files: &[Vec<HashSegment>]) -> Result<usize> {
        let nfiles = files.len();
        // Deliberately not `reserve_files` here: growing replaces `states_d` /
        // `crc_states_d` with fresh zeroed allocations, which would discard the
        // in-flight digest state mid-accumulation and return a confidently
        // wrong hash. Only `begin_many` / `begin_crc32_many` may grow it.
        anyhow::ensure!(
            nfiles <= self.file_cap,
            "update covers {nfiles} files but per-file state holds only {}; \
             begin_many/begin_crc32_many must be called with the full count",
            self.file_cap
        );
        let total_segments: usize = files.iter().map(Vec::len).sum();
        self.reserve_segments(total_segments.max(1))?;

        let mut flat = Vec::with_capacity(total_segments);
        let mut desc = Vec::with_capacity(nfiles);
        for file in files {
            desc.push(HashFileDesc {
                seg_start: flat.len() as u32,
                seg_count: file.len() as u32,
            });
            flat.extend(file.iter().copied());
        }
        if !flat.is_empty() {
            let mut view = self.segs_d.slice_mut(0..flat.len());
            self.stream.memcpy_htod(&flat, &mut view)?;
        }
        if !desc.is_empty() {
            let mut view = self.files_d.slice_mut(0..desc.len());
            self.stream.memcpy_htod(&desc, &mut view)?;
        }
        self.stream.synchronize()?;
        Ok(nfiles)
    }

    pub fn update_many(&mut self, files: &[Vec<HashSegment>]) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }
        self.ctx.bind_to_thread()?;
        let nfiles = self.upload_many_files(files)?;
        let mut segs_ptr = ptr_seg(&self.segs_d, &self.stream);
        let mut files_ptr = ptr_file(&self.files_d, &self.stream);
        let mut states_ptr = ptr_u8(&self.states_d, &self.stream);
        let mut nfiles_u32 = nfiles as u32;
        let mut status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut params: [*mut c_void; 5] = [
            &mut segs_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut files_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut states_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut nfiles_u32 as *mut u32 as *mut c_void,
            &mut status_ptr as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.many_update_func,
                many_grid(nfiles),
                (MANY_THREADS_PER_BLOCK, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_hash_many_update")?;
        }
        self.stream.synchronize()?;
        self.check_status()
    }

    pub fn finish_many(&mut self, alg: HashAlgorithm, nfiles: usize) -> Result<Vec<Vec<u8>>> {
        if nfiles == 0 {
            return Ok(Vec::new());
        }
        // `outs_d` only holds `file_cap` digests; slicing past it would panic.
        anyhow::ensure!(
            nfiles <= self.file_cap,
            "finish_many covers {nfiles} files but digest scratch holds only {}",
            self.file_cap
        );
        self.ctx.bind_to_thread()?;
        let mut alg_id = alg as u32;
        let mut states_ptr = ptr_u8(&self.states_d, &self.stream);
        let mut outs_ptr = ptr_u8(&self.outs_d, &self.stream);
        let mut nfiles_u32 = nfiles as u32;
        let mut status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut params: [*mut c_void; 5] = [
            &mut alg_id as *mut u32 as *mut c_void,
            &mut states_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut outs_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut nfiles_u32 as *mut u32 as *mut c_void,
            &mut status_ptr as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.many_final_func,
                many_grid(nfiles),
                (MANY_THREADS_PER_BLOCK, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_hash_many_final")?;
        }

        let digest_len = alg.digest_len();
        let mut flat_out = vec![0u8; nfiles * MAX_DIGEST];
        let view = self.outs_d.slice(0..flat_out.len());
        self.stream.memcpy_dtoh(&view, &mut flat_out)?;
        self.stream.synchronize()?;
        self.check_status()?;
        Ok((0..nfiles)
            .map(|i| flat_out[i * MAX_DIGEST..i * MAX_DIGEST + digest_len].to_vec())
            .collect())
    }

    pub fn hash_many(
        &mut self,
        alg: HashAlgorithm,
        files: &[Vec<HashSegment>],
    ) -> Result<Vec<Vec<u8>>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        self.begin_many(alg, files.len())?;
        self.update_many(files)?;
        self.finish_many(alg, files.len())
    }

    pub fn begin_crc32_many(&mut self, nfiles: usize) -> Result<()> {
        if nfiles == 0 {
            return Ok(());
        }
        self.ctx
            .bind_to_thread()
            .context("bind ctx for crc32_many init")?;
        self.reserve_files(nfiles)?;
        let zero = [0u32; 1];
        self.stream.memcpy_htod(&zero, &mut self.status_d)?;

        let mut states_ptr = ptr_u32(&self.crc_states_d, &self.stream);
        let mut nfiles_u32 = nfiles as u32;
        let mut status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut params: [*mut c_void; 3] = [
            &mut states_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut nfiles_u32 as *mut u32 as *mut c_void,
            &mut status_ptr as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.crc32_init_func,
                many_grid(nfiles),
                (MANY_THREADS_PER_BLOCK, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_crc32_many_init")?;
        }
        self.stream.synchronize()?;
        self.check_status()
    }

    pub fn update_crc32_many(&mut self, files: &[Vec<HashSegment>]) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }
        self.ctx
            .bind_to_thread()
            .context("bind ctx for crc32_many update")?;
        let nfiles = self.upload_many_files(files)?;
        let mut segs_ptr = ptr_seg(&self.segs_d, &self.stream);
        let mut files_ptr = ptr_file(&self.files_d, &self.stream);
        let mut states_ptr = ptr_u32(&self.crc_states_d, &self.stream);
        let mut nfiles_u32 = nfiles as u32;
        let mut status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut params: [*mut c_void; 5] = [
            &mut segs_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut files_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut states_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut nfiles_u32 as *mut u32 as *mut c_void,
            &mut status_ptr as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.crc32_many_func,
                many_grid(nfiles),
                (MANY_THREADS_PER_BLOCK, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_crc32_many")?;
        }
        self.stream.synchronize()?;
        self.check_status()
    }

    pub fn finish_crc32_many(&mut self, nfiles: usize) -> Result<Vec<u32>> {
        if nfiles == 0 {
            return Ok(Vec::new());
        }
        // `crc_out_d` only holds `file_cap` entries; slicing past it would panic.
        anyhow::ensure!(
            nfiles <= self.file_cap,
            "finish_crc32_many covers {nfiles} files but CRC scratch holds only {}",
            self.file_cap
        );
        self.ctx
            .bind_to_thread()
            .context("bind ctx for crc32_many final")?;
        let mut states_ptr = ptr_u32(&self.crc_states_d, &self.stream);
        let mut out_ptr = ptr_u32(&self.crc_out_d, &self.stream);
        let mut nfiles_u32 = nfiles as u32;
        let mut status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut params: [*mut c_void; 4] = [
            &mut states_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut out_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut nfiles_u32 as *mut u32 as *mut c_void,
            &mut status_ptr as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.crc32_final_func,
                many_grid(nfiles),
                (MANY_THREADS_PER_BLOCK, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_crc32_many_final")?;
        }
        self.stream.synchronize()?;
        self.check_status()?;
        let mut out = vec![0u32; nfiles];
        let view = self.crc_out_d.slice(0..nfiles);
        self.stream.memcpy_dtoh(&view, &mut out)?;
        self.stream.synchronize()?;
        Ok(out)
    }

    pub fn crc32_many(&mut self, files: &[Vec<HashSegment>]) -> Result<Vec<u32>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        self.begin_crc32_many(files.len())?;
        self.update_crc32_many(files)?;
        self.finish_crc32_many(files.len())
    }

    /// Launch one GPU transcoding pass (`vramdisk_b64_*` / `vramdisk_hex_*`)
    /// over `units` fixed-size groups. `count` is the kernel's own length
    /// parameter (bytes for the encoders, groups for the decoders). The
    /// decoders report invalid input characters through the status flag.
    fn launch_transcode(
        &mut self,
        func: sys::CUfunction,
        name: &str,
        in_ptr: u64,
        count: u64,
        out_ptr: u64,
        units: u64,
    ) -> Result<()> {
        if units == 0 {
            return Ok(());
        }
        const THREADS: u32 = 256;
        anyhow::ensure!(
            units <= (u32::MAX as u64) * THREADS as u64,
            "transcode launch too large: {units} units"
        );
        self.ctx
            .bind_to_thread()
            .context("bind ctx for transcode")?;
        let zero = [0u32; 1];
        self.stream.memcpy_htod(&zero, &mut self.status_d)?;
        let status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut p_in = in_ptr as sys::CUdeviceptr;
        let mut n = count;
        let mut p_out = out_ptr as sys::CUdeviceptr;
        let mut p_status = status_ptr;
        let mut params: [*mut c_void; 4] = [
            &mut p_in as *mut sys::CUdeviceptr as *mut c_void,
            &mut n as *mut u64 as *mut c_void,
            &mut p_out as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        let blocks = units.div_ceil(THREADS as u64) as u32;
        unsafe {
            dr::launch_kernel(
                func,
                (blocks, 1, 1),
                (THREADS, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .with_context(|| format!("launch {name}"))?;
        }
        self.stream.synchronize()?;
        self.check_status()
            .with_context(|| format!("{name}: invalid input data"))
    }

    /// Base64-encode `in_len` bytes at device address `in_ptr` into
    /// `ceil(in_len/3)*4` output bytes at `out_ptr` (with `=` padding).
    pub fn base64_encode(&mut self, in_ptr: u64, in_len: u64, out_ptr: u64) -> Result<()> {
        let func = self.b64_encode_func;
        self.launch_transcode(
            func,
            "vramdisk_b64_encode",
            in_ptr,
            in_len,
            out_ptr,
            in_len.div_ceil(3),
        )
    }

    /// Decode `quads` full (non-padded) 4-character Base64 groups at `in_ptr`
    /// into `quads*3` bytes at `out_ptr`. Fails on any character outside the
    /// standard Base64 alphabet — including `=`, which callers must strip and
    /// handle as the final group.
    pub fn base64_decode(&mut self, in_ptr: u64, quads: u64, out_ptr: u64) -> Result<()> {
        let func = self.b64_decode_func;
        self.launch_transcode(func, "vramdisk_b64_decode", in_ptr, quads, out_ptr, quads)
    }

    /// Hex-encode (lowercase) `in_len` bytes at `in_ptr` into `in_len*2`
    /// characters at `out_ptr`.
    pub fn hex_encode(&mut self, in_ptr: u64, in_len: u64, out_ptr: u64) -> Result<()> {
        let func = self.hex_encode_func;
        self.launch_transcode(func, "vramdisk_hex_encode", in_ptr, in_len, out_ptr, in_len)
    }

    /// Decode `pairs` hex digit pairs (either case) at `in_ptr` into `pairs`
    /// bytes at `out_ptr`. Fails on any non-hex character.
    pub fn hex_decode(&mut self, in_ptr: u64, pairs: u64, out_ptr: u64) -> Result<()> {
        let func = self.hex_decode_func;
        self.launch_transcode(func, "vramdisk_hex_decode", in_ptr, pairs, out_ptr, pairs)
    }

    /// Scan `hay_len` bytes at device address `hay_ptr` for `needle`.
    ///
    /// Returns the total number of matches and up to [`SEARCH_HIT_CAP`] of
    /// their offsets, each biased by `base_offset` so the caller can report
    /// positions in the original file rather than in the staged window. The
    /// recorded offsets are the ones that happened to win the atomic, not the
    /// numerically first ones, so a caller that truncates must say so; they are
    /// returned sorted for convenience.
    ///
    /// `fold_case` folds ASCII `A-Z` on both sides. It deliberately does not
    /// attempt Unicode case folding: the data is arbitrary bytes, not
    /// necessarily text in any particular encoding.
    pub fn search(
        &mut self,
        hay_ptr: u64,
        hay_len: u64,
        needle: &[u8],
        fold_case: bool,
        base_offset: u64,
        want_offsets: usize,
    ) -> Result<SearchLaunch> {
        anyhow::ensure!(!needle.is_empty(), "search pattern must not be empty");
        anyhow::ensure!(
            needle.len() <= SEARCH_MAX_PATTERN,
            "search pattern longer than {SEARCH_MAX_PATTERN} bytes"
        );
        if hay_len < needle.len() as u64 {
            return Ok(SearchLaunch::default());
        }
        self.ctx.bind_to_thread().context("bind ctx for search")?;

        let mut staged = [0u8; SEARCH_MAX_PATTERN];
        staged[..needle.len()].copy_from_slice(needle);
        self.stream.memcpy_htod(&staged, &mut self.needle_d)?;
        let zero = [0u64; 1];
        self.stream.memcpy_htod(&zero, &mut self.hit_count_d)?;

        let mut p_hay = hay_ptr as sys::CUdeviceptr;
        let mut n_hay = hay_len;
        let mut p_needle = ptr_u8(&self.needle_d, &self.stream);
        let mut n_needle = needle.len() as u32;
        let mut fold = u32::from(fold_case);
        let mut base = base_offset;
        let mut p_hits = ptr_u64(&self.hits_d, &self.stream);
        let mut cap = SEARCH_HIT_CAP as u32;
        let mut p_count = ptr_u64(&self.hit_count_d, &self.stream);
        let mut params: [*mut c_void; 9] = [
            &mut p_hay as *mut sys::CUdeviceptr as *mut c_void,
            &mut n_hay as *mut u64 as *mut c_void,
            &mut p_needle as *mut sys::CUdeviceptr as *mut c_void,
            &mut n_needle as *mut u32 as *mut c_void,
            &mut fold as *mut u32 as *mut c_void,
            &mut base as *mut u64 as *mut c_void,
            &mut p_hits as *mut sys::CUdeviceptr as *mut c_void,
            &mut cap as *mut u32 as *mut c_void,
            &mut p_count as *mut sys::CUdeviceptr as *mut c_void,
        ];

        // Grid-strided, so the launch is sized to keep the device busy rather
        // than to cover the buffer: one thread per byte would be millions of
        // blocks for a large window and no faster.
        const THREADS: u32 = 256;
        const MAX_BLOCKS: u64 = 4096;
        let candidates = hay_len - needle.len() as u64 + 1;
        let blocks = candidates.div_ceil(THREADS as u64).clamp(1, MAX_BLOCKS) as u32;
        unsafe {
            dr::launch_kernel(
                self.search_func,
                (blocks, 1, 1),
                (THREADS, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_search")?;
        }
        self.stream.synchronize()?;

        let mut count = [0u64; 1];
        self.stream.memcpy_dtoh(&self.hit_count_d, &mut count)?;
        let total = count[0];
        // Only copy back what the caller will actually keep. A pattern with
        // millions of hits fills the whole buffer every launch, and copying and
        // sorting 64 Ki offsets that are about to be discarded costs more than
        // the scan itself.
        let kept = total.min(SEARCH_HIT_CAP as u64).min(want_offsets as u64) as usize;
        let mut offsets = vec![0u64; kept];
        if kept > 0 {
            let view = self.hits_d.slice(0..kept);
            self.stream.memcpy_dtoh(&view, &mut offsets)?;
        }
        self.stream.synchronize()?;
        offsets.sort_unstable();
        Ok(SearchLaunch { total, offsets })
    }

    fn check_status(&self) -> Result<()> {
        let mut status = [0u32; 1];
        self.stream.memcpy_dtoh(&self.status_d, &mut status)?;
        self.stream.synchronize()?;
        anyhow::ensure!(
            status[0] == 0,
            "API CUDA kernel failed with status {}",
            status[0]
        );
        Ok(())
    }
}

fn get_func(module: sys::CUmodule, name: &str) -> Result<sys::CUfunction> {
    let cname = CString::new(name).unwrap();
    unsafe { dr::module::get_function(module, cname).with_context(|| format!("get kernel {name}")) }
}

fn ptr_u8(slice: &CudaSlice<u8>, stream: &CudaStream) -> sys::CUdeviceptr {
    let (p, _g) = slice.device_ptr(stream);
    p
}

fn ptr_u64(slice: &CudaSlice<u64>, stream: &CudaStream) -> sys::CUdeviceptr {
    let (p, _g) = slice.device_ptr(stream);
    p
}

fn ptr_u32(slice: &CudaSlice<u32>, stream: &CudaStream) -> sys::CUdeviceptr {
    let (p, _g) = slice.device_ptr(stream);
    p
}

fn ptr_seg(slice: &CudaSlice<HashSegment>, stream: &CudaStream) -> sys::CUdeviceptr {
    let (p, _g) = slice.device_ptr(stream);
    p
}

fn ptr_file(slice: &CudaSlice<HashFileDesc>, stream: &CudaStream) -> sys::CUdeviceptr {
    let (p, _g) = slice.device_ptr(stream);
    p
}

fn many_grid(nfiles: usize) -> (u32, u32, u32) {
    ((nfiles as u32).div_ceil(MANY_THREADS_PER_BLOCK), 1, 1)
}

pub fn digest_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{b:02x}").unwrap();
    }
    out
}

/// Placeholder inside [`API_CUDA`] replaced with the CRC-32 table entries.
const CRC32_TABLE_MARKER: &str = "/*CRC32_TABLE*/";

/// Standard reflected CRC-32 table (polynomial 0xedb88320 — the zlib/PKZIP
/// one). Derived here rather than typed out so none of the 256 entries can be
/// mistyped; the unit tests pin the well-known anchor values.
const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// NUL-terminated PTX for [`API_CUDA`], compiled at most once per process.
///
/// NVRTC takes about six seconds to compile this translation unit on a typical
/// desktop, and the result depends on nothing but the source text and the
/// options below — both compile-time constants. Recompiling it per
/// [`ApiKernel`] made every fresh engine (each mount, and each test that
/// builds one) pay that again for an identical answer.
///
/// The failure is cached as a message rather than retried: a compile that
/// failed once for this fixed input will fail the same way every time, and
/// re-running a six-second compile per call to say so is worse than repeating
/// the diagnostic.
static API_PTX: std::sync::OnceLock<Result<Vec<u8>, String>> = std::sync::OnceLock::new();

/// Compile [`API_CUDA`] now, so the first [`ApiKernel::new`] does not have to.
///
/// NVRTC needs no CUDA context, so this is safe to call from any thread —
/// including a background one that does not own the device — and a later
/// `ApiKernel::new` on any thread reuses the result. Errors are left for that
/// call to report: a warm-up has nobody to report to.
pub fn precompile() {
    let _ = compiled_api_ptx();
}

fn compiled_api_ptx() -> Result<&'static Vec<u8>> {
    API_PTX
        .get_or_init(|| {
            let ptx = compile_ptx_with_opts(
                api_cuda_source(),
                CompileOptions {
                    // API_CUDA is plain scalar CUDA C++ (MD5/SHA1/SHA256/CRC32
                    // block math) with no warp/tensor intrinsics that need a
                    // newer architecture, so target the oldest arch NVRTC still
                    // supports rather than whatever a dev machine's default is —
                    // PTX JIT can run a lower-.target module on a newer GPU but
                    // never the reverse, so a needlessly high target here would
                    // silently break pre-Turing GPUs.
                    arch: Some("compute_50"),
                    options: vec!["--std=c++11".to_string()],
                    name: Some("vramdisk_api_kernel.cu".to_string()),
                    ..Default::default()
                },
            )
            .map_err(|e| format!("{e:#}"))?;
            let mut bytes = ptx.to_src().into_bytes();
            bytes.push(0);
            Ok(bytes)
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("compile API CUDA kernels with NVRTC: {e}"))
}

/// Host-side reference CRC-32, used by tests to pin what the device kernels
/// and the GF(2) lane combiner in `engine` must reproduce exactly.
#[cfg(test)]
pub(crate) fn crc32_reference(data: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc = (crc >> 8) ^ table[((crc ^ b as u32) & 0xff) as usize];
    }
    crc ^ 0xffff_ffff
}

/// [`API_CUDA`] with the CRC-32 table literals substituted in. NVRTC compiles
/// the source at runtime anyway, so assembling it here costs nothing that
/// matters next to the compile itself.
fn api_cuda_source() -> String {
    use std::fmt::Write as _;
    let table = crc32_table();
    let mut entries = String::with_capacity(256 * 13);
    for (i, v) in table.iter().enumerate() {
        entries.push_str(if i % 8 == 0 { "\n    " } else { " " });
        write!(&mut entries, "0x{v:08x}u,").unwrap();
    }
    entries.push('\n');
    API_CUDA.replace(CRC32_TABLE_MARKER, &entries)
}

const API_CUDA: &str = r#"
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

struct HashSegment { u64 ptr; u32 len; u32 kind; };
struct HashFileDesc { u32 seg_start; u32 seg_count; };
struct HashState {
    u32 alg;
    u64 len;
    u32 md5[4];
    u32 sha1[5];
    u32 sha256[8];
    u64 fnv;
    u8 buf[64];
    u32 buf_len;
};

static const u64 HASH_STATE_STRIDE = 256ULL;
__device__ __constant__ u8 ZERO_BLOCK[64] = {0};

__device__ __forceinline__ u32 rol32(u32 x, u32 n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ u32 ror32(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ u32 ld_le32(const u8* p) { return ((u32)p[0]) | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
__device__ __forceinline__ u32 ld_be32(const u8* p) { return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | ((u32)p[3]); }
__device__ __forceinline__ void st_le32(u8* p, u32 x) { p[0]=x; p[1]=x>>8; p[2]=x>>16; p[3]=x>>24; }
__device__ __forceinline__ void st_be32(u8* p, u32 x) { p[0]=x>>24; p[1]=x>>16; p[2]=x>>8; p[3]=x; }
__device__ __forceinline__ void st_be64(u8* p, u64 x) { for (int i=0;i<8;i++) p[i] = (u8)(x >> (56 - 8*i)); }

__device__ __forceinline__ void hash_init_state(HashState* s, u32 alg) {
    s->alg = alg;
    s->len = 0;
    s->buf_len = 0;
    for (int i=0;i<64;i++) s->buf[i]=0;
    s->md5[0]=0x67452301; s->md5[1]=0xefcdab89; s->md5[2]=0x98badcfe; s->md5[3]=0x10325476;
    s->sha1[0]=0x67452301; s->sha1[1]=0xefcdab89; s->sha1[2]=0x98badcfe; s->sha1[3]=0x10325476; s->sha1[4]=0xc3d2e1f0;
    s->sha256[0]=0x6a09e667; s->sha256[1]=0xbb67ae85; s->sha256[2]=0x3c6ef372; s->sha256[3]=0xa54ff53a;
    s->sha256[4]=0x510e527f; s->sha256[5]=0x9b05688c; s->sha256[6]=0x1f83d9ab; s->sha256[7]=0x5be0cd19;
    s->fnv=0xcbf29ce484222325ULL;
}

__device__ void md5_block(HashState* s, const u8* data) {
    const u32 K[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391 };
    const u32 R[64] = {
        7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
        4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21 };
    u32 m[16]; for (int i=0;i<16;i++) m[i]=ld_le32(data+i*4);
    u32 a=s->md5[0], b=s->md5[1], c=s->md5[2], d=s->md5[3];
    for (int i=0;i<64;i++) {
        u32 f,g;
        if (i<16) { f=(b&c)|((~b)&d); g=i; }
        else if (i<32) { f=(d&b)|((~d)&c); g=(5*i+1)&15; }
        else if (i<48) { f=b^c^d; g=(3*i+5)&15; }
        else { f=c^(b|(~d)); g=(7*i)&15; }
        u32 tmp=d; d=c; c=b; b=b+rol32(a+f+K[i]+m[g],R[i]); a=tmp;
    }
    s->md5[0]+=a; s->md5[1]+=b; s->md5[2]+=c; s->md5[3]+=d;
}

__device__ void sha1_block(HashState* s, const u8* data) {
    u32 w[80]; for (int i=0;i<16;i++) w[i]=ld_be32(data+i*4);
    for (int i=16;i<80;i++) w[i]=rol32(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
    u32 a=s->sha1[0],b=s->sha1[1],c=s->sha1[2],d=s->sha1[3],e=s->sha1[4];
    for (int i=0;i<80;i++) {
        u32 f,k;
        if (i<20) { f=(b&c)|((~b)&d); k=0x5a827999; }
        else if (i<40) { f=b^c^d; k=0x6ed9eba1; }
        else if (i<60) { f=(b&c)|(b&d)|(c&d); k=0x8f1bbcdc; }
        else { f=b^c^d; k=0xca62c1d6; }
        u32 t=rol32(a,5)+f+e+k+w[i]; e=d; d=c; c=rol32(b,30); b=a; a=t;
    }
    s->sha1[0]+=a; s->sha1[1]+=b; s->sha1[2]+=c; s->sha1[3]+=d; s->sha1[4]+=e;
}

__device__ void sha256_block(HashState* s, const u8* data) {
    const u32 K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
    u32 w[64]; for (int i=0;i<16;i++) w[i]=ld_be32(data+i*4);
    for (int i=16;i<64;i++) {
        u32 s0=ror32(w[i-15],7)^ror32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=ror32(w[i-2],17)^ror32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=s->sha256[0],b=s->sha256[1],c=s->sha256[2],d=s->sha256[3],e=s->sha256[4],f=s->sha256[5],g=s->sha256[6],h=s->sha256[7];
    for (int i=0;i<64;i++) {
        u32 S1=ror32(e,6)^ror32(e,11)^ror32(e,25);
        u32 ch=(e&f)^((~e)&g);
        u32 t1=h+S1+ch+K[i]+w[i];
        u32 S0=ror32(a,2)^ror32(a,13)^ror32(a,22);
        u32 maj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->sha256[0]+=a; s->sha256[1]+=b; s->sha256[2]+=c; s->sha256[3]+=d; s->sha256[4]+=e; s->sha256[5]+=f; s->sha256[6]+=g; s->sha256[7]+=h;
}

__device__ __forceinline__ void process_block(HashState* s, const u8* block) {
    if (s->alg==1) md5_block(s, block);
    else if (s->alg==2) sha1_block(s, block);
    else if (s->alg==3) sha256_block(s, block);
}

__device__ __forceinline__ void hash_update_block64(HashState* s, const u8* block) {
    if (s->alg == 4) {
        for (int i=0; i<64; i++) {
            s->fnv ^= (u64)block[i];
            s->fnv *= 0x100000001b3ULL;
        }
        return;
    }
    process_block(s, block);
}

__device__ __forceinline__ void hash_update_data(HashState* s, const u8* data, u32 len) {
    if (len == 0) return;
    s->len += (u64)len;
    if (s->alg == 4) {
        while (len >= 64) {
            hash_update_block64(s, data);
            data += 64;
            len -= 64;
        }
        for (u32 i=0; i<len; i++) {
            s->fnv ^= (u64)data[i];
            s->fnv *= 0x100000001b3ULL;
        }
        return;
    }

    if (s->buf_len != 0) {
        u32 take = 64 - s->buf_len;
        if (take > len) take = len;
        for (u32 i=0; i<take; i++) s->buf[s->buf_len + i] = data[i];
        s->buf_len += take;
        data += take;
        len -= take;
        if (s->buf_len != 64) return;
        if (s->buf_len == 64) {
            process_block(s, s->buf);
            s->buf_len = 0;
        }
    }
    while (len >= 64) {
        process_block(s, data);
        data += 64;
        len -= 64;
    }
    for (u32 i=0; i<len; i++) s->buf[i] = data[i];
    s->buf_len = len;
}

__device__ __forceinline__ void hash_update_zeros(HashState* s, u32 len) {
    if (len == 0) return;
    s->len += (u64)len;
    if (s->alg == 4) {
        while (len >= 64) {
            hash_update_block64(s, ZERO_BLOCK);
            len -= 64;
        }
        for (u32 i=0; i<len; i++) {
            s->fnv ^= 0ULL;
            s->fnv *= 0x100000001b3ULL;
        }
        return;
    }

    if (s->buf_len != 0) {
        u32 take = 64 - s->buf_len;
        if (take > len) take = len;
        for (u32 i=0; i<take; i++) s->buf[s->buf_len + i] = 0;
        s->buf_len += take;
        len -= take;
        if (s->buf_len != 64) return;
        if (s->buf_len == 64) {
            process_block(s, s->buf);
            s->buf_len = 0;
        }
    }
    while (len >= 64) {
        process_block(s, ZERO_BLOCK);
        len -= 64;
    }
    for (u32 i=0; i<len; i++) s->buf[i] = 0;
    s->buf_len = len;
}

__device__ __forceinline__ void hash_update_segment(HashState* s, HashSegment sg) {
    if (sg.kind == 1) hash_update_zeros(s, sg.len);
    else hash_update_data(s, (const u8*)sg.ptr, sg.len);
}

__device__ __forceinline__ void hash_finalize_state(HashState* s, u8* out, u32* status) {
    if (s->alg == 4) {
        st_be64(out, s->fnv);
        return;
    }
    u64 bits = s->len * 8ULL;
    u8 one = 0x80;
    hash_update_data(s, &one, 1);
    while (s->buf_len != 56) hash_update_zeros(s, 1);
    u8 len_bytes[8];
    if (s->alg == 1) {
        for (int i=0;i<8;i++) len_bytes[i] = (u8)(bits >> (8*i));
    } else {
        for (int i=0;i<8;i++) len_bytes[i] = (u8)(bits >> (56 - 8*i));
    }
    hash_update_data(s, len_bytes, 8);
    if (s->alg==1) { for (int i=0;i<4;i++) st_le32(out+i*4, s->md5[i]); }
    else if (s->alg==2) { for (int i=0;i<5;i++) st_be32(out+i*4, s->sha1[i]); }
    else if (s->alg==3) { for (int i=0;i<8;i++) st_be32(out+i*4, s->sha256[i]); }
    else atomicExch(status, 1u);
}

// The error flag is written atomically (every writer stores the same value, 1)
// so concurrent threads never tear it. The `status[0]!=0` reads below are
// deliberately plain and racy: they are only an early-out, and the host
// re-reads the flag after synchronizing.

extern "C" __global__ void vramdisk_hash_init(u32 alg, u8* state_raw, u32* status) {
    HashState* s = (HashState*)state_raw;
    status[0]=0;
    hash_init_state(s, alg);
    if (alg < 1 || alg > 4) atomicExch(status, 1u);
}

extern "C" __global__ void vramdisk_hash_update(const HashSegment* segs, u32 nsegs, u8* state_raw, u32* status) {
    if (status[0]!=0) return;
    HashState* g = (HashState*)state_raw;
    HashState s = *g;
    for (u32 si=0; si<nsegs; si++) hash_update_segment(&s, segs[si]);
    *g = s;
}

extern "C" __global__ void vramdisk_hash_final(u32 alg, u8* state_raw, u8* out, u32* status) {
    if (status[0]!=0) return;
    HashState* g = (HashState*)state_raw;
    HashState s = *g;
    if (s.alg != alg) { atomicExch(status, 1u); return; }
    hash_finalize_state(&s, out, status);
    *g = s;
}

extern "C" __global__ void vramdisk_hash_many_init(u32 alg, u8* states_raw, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nfiles) return;
    HashState* s = (HashState*)(states_raw + ((u64)fi * HASH_STATE_STRIDE));
    hash_init_state(s, alg);
    if (alg < 1 || alg > 4) atomicExch(status, 1u);
}

extern "C" __global__ void vramdisk_hash_many_update(const HashSegment* segs, const HashFileDesc* files, u8* states_raw, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nfiles || status[0]!=0) return;
    HashState* g = (HashState*)(states_raw + ((u64)fi * HASH_STATE_STRIDE));
    HashState s = *g;
    HashFileDesc fd = files[fi];
    for (u32 rel=0; rel<fd.seg_count; rel++) hash_update_segment(&s, segs[fd.seg_start + rel]);
    *g = s;
}

extern "C" __global__ void vramdisk_hash_many_final(u32 alg, u8* states_raw, u8* outs, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nfiles || status[0]!=0) return;
    HashState* g = (HashState*)(states_raw + ((u64)fi * HASH_STATE_STRIDE));
    HashState s = *g;
    u8* out = outs + ((u64)fi * 32ULL);
    if (s.alg != alg) { atomicExch(status, 1u); return; }
    hash_finalize_state(&s, out, status);
    *g = s;
}

// Standard reflected CRC-32 table (polynomial 0xedb88320, the zlib/PKZIP one).
// The entries are substituted in by api_cuda_source() before NVRTC sees this.
// One lookup per byte replaces the eight shift/mask iterations the bitwise loop
// needed: that version ran ~3x slower per byte than SHA-256 while sharing
// SHA-256's launch budget, putting large files within range of the Windows TDR
// timeout (a driver reset loses the whole mounted disk).
__device__ __constant__ u32 CRC32_TABLE[256] = {/*CRC32_TABLE*/};

__device__ __forceinline__ u32 crc32_update_byte(u32 crc, u8 b) {
    return (crc >> 8) ^ CRC32_TABLE[(crc ^ (u32)b) & 0xffu];
}

__device__ __forceinline__ u32 crc32_update_block64(u32 crc, const u8* block) {
    for (int i=0; i<64; i++) crc = (crc >> 8) ^ CRC32_TABLE[(crc ^ (u32)block[i]) & 0xffu];
    return crc;
}

__device__ __forceinline__ u32 crc32_update_segment(u32 crc, HashSegment sg) {
    if (sg.kind == 1) {
        while (sg.len >= 64) {
            crc = crc32_update_block64(crc, ZERO_BLOCK);
            sg.len -= 64;
        }
        for (u32 i=0; i<sg.len; i++) crc = crc32_update_byte(crc, 0);
        return crc;
    }

    const u8* p = (const u8*)sg.ptr;
    while (sg.len >= 64) {
        crc = crc32_update_block64(crc, p);
        p += 64;
        sg.len -= 64;
    }
    for (u32 i=0; i<sg.len; i++) crc = crc32_update_byte(crc, p[i]);
    return crc;
}

extern "C" __global__ void vramdisk_crc32_many_init(u32* states, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nfiles || status[0] != 0) return;
    states[fi] = 0xffffffffu;
}

extern "C" __global__ void vramdisk_crc32_many(const HashSegment* segs, const HashFileDesc* files, u32* states, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nfiles || status[0]!=0) return;
    HashFileDesc fd = files[fi];
    u32 crc = states[fi];
    for (u32 rel=0; rel<fd.seg_count; rel++) crc = crc32_update_segment(crc, segs[fd.seg_start + rel]);
    states[fi] = crc;
}

extern "C" __global__ void vramdisk_crc32_many_final(const u32* states, u32* outs, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= nfiles || status[0]!=0) return;
    outs[fi] = states[fi] ^ 0xffffffffu;
}

// ---- Base64 / hex transcoding ----------------------------------------------
//
// Each thread handles one fixed-size group (3 bytes -> 4 chars for Base64,
// 1 byte -> 2 chars for hex), so a pass is embarrassingly parallel and
// memory-bound. The encoders take the byte length and pad the final group
// themselves; the decoders only ever see full groups (the host strips and
// finishes the '='-padded Base64 tail) and flag invalid characters through
// `status` (racy reads of the flag are benign: every writer stores 1).

__device__ __constant__ char B64_CHARS[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

extern "C" __global__ void vramdisk_b64_encode(const u8* in, u64 in_len, u8* out, u32* status) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 groups = (in_len + 2) / 3;
    if (i >= groups) return;
    u64 s = i * 3;
    u32 b0 = in[s];
    u32 b1 = (s + 1 < in_len) ? in[s + 1] : 0;
    u32 b2 = (s + 2 < in_len) ? in[s + 2] : 0;
    u32 w = (b0 << 16) | (b1 << 8) | b2;
    u8* o = out + i * 4;
    o[0] = B64_CHARS[(w >> 18) & 63u];
    o[1] = B64_CHARS[(w >> 12) & 63u];
    o[2] = (s + 1 < in_len) ? B64_CHARS[(w >> 6) & 63u] : (u8)'=';
    o[3] = (s + 2 < in_len) ? B64_CHARS[w & 63u] : (u8)'=';
}

__device__ __forceinline__ int b64_val(u8 c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

extern "C" __global__ void vramdisk_b64_decode(const u8* in, u64 quads, u8* out, u32* status) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= quads) return;
    const u8* p = in + i * 4;
    int v0 = b64_val(p[0]);
    int v1 = b64_val(p[1]);
    int v2 = b64_val(p[2]);
    int v3 = b64_val(p[3]);
    if ((v0 | v1 | v2 | v3) < 0) { atomicExch(status, 1u); return; }
    u32 w = ((u32)v0 << 18) | ((u32)v1 << 12) | ((u32)v2 << 6) | (u32)v3;
    u8* o = out + i * 3;
    o[0] = (u8)(w >> 16);
    o[1] = (u8)(w >> 8);
    o[2] = (u8)w;
}

__device__ __constant__ char HEX_CHARS[17] = "0123456789abcdef";

extern "C" __global__ void vramdisk_hex_encode(const u8* in, u64 in_len, u8* out, u32* status) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_len) return;
    u8 b = in[i];
    out[i * 2] = HEX_CHARS[b >> 4];
    out[i * 2 + 1] = HEX_CHARS[b & 15u];
}

__device__ __forceinline__ int hex_val(u8 c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

extern "C" __global__ void vramdisk_hex_decode(const u8* in, u64 pairs, u8* out, u32* status) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    int hi = hex_val(in[i * 2]);
    int lo = hex_val(in[i * 2 + 1]);
    if ((hi | lo) < 0) { atomicExch(status, 1u); return; }
    out[i] = (u8)((hi << 4) | lo);
}

// --- literal substring search ------------------------------------------------
//
// One thread per candidate start offset, grid-strided so the launch shape does
// not depend on the buffer size. The filter is the point: a candidate is
// rejected on the first *and* last needle byte before the body is compared, so
// the overwhelmingly common non-match costs two loads and no branching beyond
// the compare. That keeps the kernel bandwidth-bound, which is what makes this
// worth doing on a GPU at all -- the data is already in VRAM, so the scan runs
// at device memory bandwidth instead of streaming over PCIe to the host.
//
// `count` is incremented for every hit, including hits past `cap`, so callers
// get a true total even when they only keep a sample of the offsets.

__device__ __forceinline__ u8 ascii_lower(u8 c) {
    return (c >= 'A' && c <= 'Z') ? (u8)(c + 32) : c;
}

extern "C" __global__ void vramdisk_search(
    const u8* hay, u64 hay_len,
    const u8* needle, u32 needle_len,
    u32 fold_case, u64 base_offset,
    u64* out_offsets, u32 cap, u64* count)
{
    if (needle_len == 0 || hay_len < (u64)needle_len) return;
    u64 last = hay_len - (u64)needle_len;
    u64 stride = (u64)blockDim.x * (u64)gridDim.x;
    u8 n_first = needle[0];
    u8 n_last = needle[needle_len - 1];
    if (fold_case) { n_first = ascii_lower(n_first); n_last = ascii_lower(n_last); }

    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i <= last; i += stride) {
        u8 c_first = hay[i];
        u8 c_last = hay[i + needle_len - 1];
        if (fold_case) { c_first = ascii_lower(c_first); c_last = ascii_lower(c_last); }
        if (c_first != n_first || c_last != n_last) continue;
        bool hit = true;
        for (u32 k = 1; k + 1 < needle_len; ++k) {
            u8 a = hay[i + k];
            u8 b = needle[k];
            if (fold_case) { a = ascii_lower(a); b = ascii_lower(b); }
            if (a != b) { hit = false; break; }
        }
        if (hit) {
            u64 slot = atomicAdd(count, 1ULL);
            if (slot < (u64)cap) out_offsets[slot] = base_offset + i;
        }
    }
}

"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchors for the standard reflected CRC-32 table. A wrong table would
    /// only surface as bad checksums at runtime, long after NVRTC is happy.
    #[test]
    fn crc32_table_matches_known_values() {
        let table = crc32_table();
        assert_eq!(table[0], 0x0000_0000);
        assert_eq!(table[1], 0x7707_3096);
        assert_eq!(table[2], 0xee0e_612c);
        assert_eq!(table[128], 0xedb8_8320);
        assert_eq!(table[255], 0x2d02_ef8d);
    }

    /// The same reflected update the kernel performs, checked against the
    /// well-known CRC-32 of "123456789".
    #[test]
    fn crc32_table_reproduces_reference_digest() {
        let table = crc32_table();
        let mut crc = 0xffff_ffffu32;
        for &b in b"123456789" {
            crc = (crc >> 8) ^ table[((crc ^ b as u32) & 0xff) as usize];
        }
        assert_eq!(crc ^ 0xffff_ffff, 0xcbf4_3926);
    }

    #[test]
    fn cuda_source_has_table_substituted() {
        let src = api_cuda_source();
        assert!(!src.contains(CRC32_TABLE_MARKER), "marker left in source");
        assert!(src.contains("0x77073096u,"), "table entry missing");
        assert_eq!(
            src.matches("0x2d02ef8du,").count(),
            1,
            "table entry missing"
        );
    }
}
