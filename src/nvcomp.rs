//! FFI wrapper around nvCOMP's batched LZ4 C API.
//!
//! The DLL is loaded dynamically with `libloading`, so the project builds and
//! runs without nvCOMP present; it is only required when `--compress` is given.
//! [`candidate_dll_paths`] scans every `v*\bin\<N>` combination actually
//! present under the default install root (plus `NVCOMP_DLL`), so nvCOMP
//! releases other than exactly v5.2/CUDA 12 are still found; the batched C
//! API's symbol names are stable across the versions this searches. Every
//! candidate is a fully qualified path — no bare filename fallback — so this
//! never depends on the ambient Windows DLL search order (app dir/PATH),
//! which would otherwise be plantable by anything that can write next to the
//! exe.
//!
//! nvCOMP's LZ4 entry points are *batched*: one launch compresses or
//! decompresses up to [`BATCH`] independent 64 KiB chunks in parallel, sharing a
//! single temp buffer and a single stream synchronize. Driving the batch with
//! many chunks (rather than one) is what lets the GPU saturate — the previous
//! batch-of-one path paid a kernel launch plus several synchronizes per 64 KiB.
//!
//! Persistent device scratch holds one whole batch (`d_in` / `d_out`), plus the
//! pointer and size arrays nvCOMP wants. Single-chunk [`compress`] /
//! [`decompress`] are thin wrappers over the batched calls with `n = 1`, used by
//! the engine's read-modify-write paths.

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use libloading::Library;

use crate::cuda::Vram;
use crate::CHUNK_SIZE;

/// Maximum chunks processed in one nvCOMP launch. Bounds the device scratch
/// (≈ `BATCH × (CHUNK_SIZE + max_comp)` plus temp) while being large enough to
/// amortise the launch + synchronize across the batch. A multi-megabyte write
/// is split into groups of this size.
pub const BATCH: usize = 256;

/// nvCOMP archive/job codecs that expose standard compressed frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvcompFrameCodec {
    Zstd,
    Lz4,
    Gzip,
    Deflate,
}

impl NvcompFrameCodec {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "zstd" | "tar.zst" | "tar.zstd" => Some(Self::Zstd),
            "lz4" | "tar.lz4" => Some(Self::Lz4),
            "gzip" | "gz" | "tar.gz" | "tgz" => Some(Self::Gzip),
            "deflate" | "zip" => Some(Self::Deflate),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Zstd => "zstd",
            Self::Lz4 => "lz4",
            Self::Gzip => "gzip",
            Self::Deflate => "deflate",
        }
    }

    fn symbol_prefix(self) -> &'static [u8] {
        match self {
            Self::Zstd => b"Zstd",
            Self::Lz4 => b"LZ4",
            Self::Gzip => b"Gzip",
            Self::Deflate => b"Deflate",
        }
    }

    fn compress_opts(self) -> GenericCompressOpts {
        let mut opts = GenericCompressOpts::default();
        match self {
            Self::Deflate => opts.words[0] = 1,
            Self::Lz4 => {
                opts.words[0] = 0;
                opts.words[1] = 0;
            }
            _ => {}
        }
        opts
    }

    fn decompress_opts(self) -> GenericDecompressOpts {
        GenericDecompressOpts::default()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GenericCompressOpts {
    words: [i32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GenericDecompressOpts {
    words: [i32; 16],
}

type FnGenericMaxOut = unsafe extern "C" fn(usize, GenericCompressOpts, *mut usize) -> i32;
type FnGenericCompTemp =
    unsafe extern "C" fn(usize, usize, GenericCompressOpts, *mut usize, usize) -> i32;
type FnGenericDecompTemp =
    unsafe extern "C" fn(usize, usize, GenericDecompressOpts, *mut usize, usize) -> i32;
type FnGenericGetDecompSize =
    unsafe extern "C" fn(*const *const c_void, *const usize, *mut usize, usize, *mut c_void) -> i32;
#[allow(clippy::type_complexity)]
type FnGenericCompress = unsafe extern "C" fn(
    *const *const c_void,
    *const usize,
    usize,
    usize,
    *mut c_void,
    usize,
    *const *mut c_void,
    *mut usize,
    GenericCompressOpts,
    *mut i32,
    *mut c_void,
) -> i32;
#[allow(clippy::type_complexity)]
type FnGenericDecompress = unsafe extern "C" fn(
    *const *const c_void,
    *const usize,
    *const usize,
    *mut usize,
    usize,
    *mut c_void,
    usize,
    *const *mut c_void,
    GenericDecompressOpts,
    *mut i32,
    *mut c_void,
) -> i32;

/// Generic nvCOMP codec used by jobs/archive paths.
///
/// Inputs and outputs are device pointers. The caller decides where the final
/// bytes live; this wrapper only owns one batch of scratch and never pulls file
/// payload bytes through host memory.
pub struct NvcompBatchedCodec {
    #[allow(dead_code)]
    lib: Library,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    codec: NvcompFrameCodec,
    f_compress: FnGenericCompress,
    f_decompress: FnGenericDecompress,
    f_get_decomp_size: FnGenericGetDecompSize,
    d_out: CudaSlice<u8>,
    d_temp: CudaSlice<u8>,
    arr_in: CudaSlice<u64>,
    arr_out: CudaSlice<u64>,
    s_in: CudaSlice<u64>,
    s_out_cap: CudaSlice<u64>,
    s_result: CudaSlice<u64>,
    statuses: CudaSlice<i32>,
    max_comp: usize,
    max_input: usize,
    out_slots: usize,
    temp_bytes: usize,
    a_d_out: usize,
    a_d_temp: usize,
    a_arr_in: usize,
    a_arr_out: usize,
    a_s_in: usize,
    a_s_out_cap: usize,
    a_s_result: usize,
    a_status: usize,
}

impl NvcompBatchedCodec {
    pub fn load(vram: &Vram, codec: NvcompFrameCodec) -> Result<Self> {
        let (lib, _path) = load_nvcomp_library()?;
        let prefix = codec.symbol_prefix();
        let symbol = |suffix: &[u8]| {
            let mut s = b"nvcompBatched".to_vec();
            s.extend_from_slice(prefix);
            s.extend_from_slice(suffix);
            s.push(0);
            s
        };
        let (f_max, f_comp_temp, f_decomp_temp, f_get_decomp_size, f_compress, f_decompress) = unsafe {
            let f_max: libloading::Symbol<FnGenericMaxOut> =
                lib.get(&symbol(b"CompressGetMaxOutputChunkSize"))?;
            let f_comp_temp: libloading::Symbol<FnGenericCompTemp> =
                lib.get(&symbol(b"CompressGetTempSizeAsync"))?;
            let f_decomp_temp: libloading::Symbol<FnGenericDecompTemp> =
                lib.get(&symbol(b"DecompressGetTempSizeAsync"))?;
            let f_get_decomp_size: libloading::Symbol<FnGenericGetDecompSize> =
                lib.get(&symbol(b"GetDecompressSizeAsync"))?;
            let f_compress: libloading::Symbol<FnGenericCompress> =
                lib.get(&symbol(b"CompressAsync"))?;
            let f_decompress: libloading::Symbol<FnGenericDecompress> =
                lib.get(&symbol(b"DecompressAsync"))?;
            (
                *f_max,
                *f_comp_temp,
                *f_decomp_temp,
                *f_get_decomp_size,
                *f_compress,
                *f_decompress,
            )
        };

        let ctx = vram.context();
        let stream = vram.stream();
        ctx.bind_to_thread()
            .context("bind ctx for nvcomp codec init")?;
        let cs = CHUNK_SIZE as usize;
        let total = BATCH * cs;
        let copts = codec.compress_opts();
        let dopts = codec.decompress_opts();

        let mut max_comp = 0usize;
        check(
            unsafe { f_max(cs, copts, &mut max_comp) },
            "GenericCompressGetMaxOutputChunkSize",
        )?;
        let mut ctemp = 0usize;
        check(
            unsafe { f_comp_temp(BATCH, cs, copts, &mut ctemp, total) },
            "GenericCompressGetTempSize",
        )?;
        let mut dtemp = 0usize;
        check(
            unsafe { f_decomp_temp(BATCH, cs, dopts, &mut dtemp, total) },
            "GenericDecompressGetTempSize",
        )?;
        let temp_bytes = ctemp.max(dtemp).max(1);

        let d_out = stream.alloc_zeros::<u8>(BATCH * max_comp)?;
        let d_temp = stream.alloc_zeros::<u8>(temp_bytes)?;
        let arr_in = stream.alloc_zeros::<u64>(BATCH)?;
        let mut arr_out = stream.alloc_zeros::<u64>(BATCH)?;
        let s_in = stream.alloc_zeros::<u64>(BATCH)?;
        let mut s_out_cap = stream.alloc_zeros::<u64>(BATCH)?;
        let s_result = stream.alloc_zeros::<u64>(BATCH)?;
        let statuses = stream.alloc_zeros::<i32>(BATCH)?;
        stream.synchronize()?;

        let a_d_out = addr_of(&d_out, &stream);
        let a_d_temp = addr_of(&d_temp, &stream);
        let a_arr_in = addr_of(&arr_in, &stream);
        let a_arr_out = addr_of(&arr_out, &stream);
        let a_s_in = addr_of(&s_in, &stream);
        let a_s_out_cap = addr_of(&s_out_cap, &stream);
        let a_s_result = addr_of(&s_result, &stream);
        let a_status = addr_of(&statuses, &stream);

        let out_ptrs: Vec<u64> = (0..BATCH)
            .map(|i| (a_d_out + i * max_comp) as u64)
            .collect();
        let out_caps: Vec<u64> = vec![cs as u64; BATCH];
        stream.memcpy_htod(&out_ptrs, &mut arr_out)?;
        stream.memcpy_htod(&out_caps, &mut s_out_cap)?;
        stream.synchronize()?;

        Ok(Self {
            lib,
            ctx,
            stream,
            codec,
            f_compress,
            f_decompress,
            f_get_decomp_size,
            d_out,
            d_temp,
            arr_in,
            arr_out,
            s_in,
            s_out_cap,
            s_result,
            statuses,
            max_comp,
            max_input: cs,
            out_slots: BATCH,
            temp_bytes,
            a_d_out,
            a_d_temp,
            a_arr_in,
            a_arr_out,
            a_s_in,
            a_s_out_cap,
            a_s_result,
            a_status,
        })
    }

    pub fn compressed_slot_ptr(&self, i: usize) -> u64 {
        (self.a_d_out + i * self.max_comp) as u64
    }

    pub fn compress_device(&mut self, input_ptrs: &[u64], input_sizes: &[u64]) -> Result<Vec<u64>> {
        if input_ptrs.len() != input_sizes.len() {
            bail!("input pointer/size length mismatch");
        }
        if input_ptrs.len() > BATCH {
            bail!(
                "nvCOMP {} accepts at most {BATCH} chunks",
                self.codec.name()
            );
        }
        if input_ptrs.is_empty() {
            return Ok(Vec::new());
        }
        self.ctx.bind_to_thread()?;
        let n = input_ptrs.len();
        let max_in = input_sizes.iter().copied().max().unwrap_or(0) as usize;
        let total_in = input_sizes.iter().sum::<u64>() as usize;
        self.reserve_for(max_in, n, total_in)?;
        {
            let mut view = self.arr_in.slice_mut(0..n);
            self.stream.memcpy_htod(input_ptrs, &mut view)?;
        }
        {
            let mut view = self.s_in.slice_mut(0..n);
            self.stream.memcpy_htod(input_sizes, &mut view)?;
        }
        self.stream.synchronize()?;
        let status = unsafe {
            (self.f_compress)(
                self.a_arr_in as *const *const c_void,
                self.a_s_in as *const usize,
                max_in,
                n,
                self.a_d_temp as *mut c_void,
                self.temp_bytes,
                self.a_arr_out as *const *mut c_void,
                self.a_s_result as *mut usize,
                self.codec.compress_opts(),
                self.a_status as *mut i32,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        check(status, "GenericCompressAsync(launch)")?;
        self.stream.synchronize()?;
        self.check_statuses(n, "compress")?;
        self.read_u64(&self.s_result, n)
    }

    fn reserve_for(&mut self, max_input: usize, n: usize, total_input: usize) -> Result<()> {
        if max_input <= self.max_input && n <= self.out_slots {
            return Ok(());
        }
        let copts = self.codec.compress_opts();
        let dopts = self.codec.decompress_opts();
        let mut max_comp = 0usize;
        let f_max = self.load_max_output_symbol()?;
        check(
            unsafe { f_max(max_input, copts, &mut max_comp) },
            "GenericCompressGetMaxOutputChunkSize(reserve)",
        )?;
        let mut ctemp = 0usize;
        let f_comp_temp = self.load_comp_temp_symbol()?;
        check(
            unsafe { f_comp_temp(n.max(1), max_input, copts, &mut ctemp, total_input.max(1)) },
            "GenericCompressGetTempSize(reserve)",
        )?;
        let mut dtemp = 0usize;
        let f_decomp_temp = self.load_decomp_temp_symbol()?;
        check(
            unsafe { f_decomp_temp(n.max(1), max_input, dopts, &mut dtemp, total_input.max(1)) },
            "GenericDecompressGetTempSize(reserve)",
        )?;
        let out_slots = n.max(1);
        self.d_out = self.stream.alloc_zeros::<u8>(out_slots * max_comp)?;
        self.d_temp = self.stream.alloc_zeros::<u8>(ctemp.max(dtemp).max(1))?;
        self.stream.synchronize()?;
        self.max_comp = max_comp;
        self.max_input = max_input;
        self.out_slots = out_slots;
        self.temp_bytes = ctemp.max(dtemp).max(1);
        self.a_d_out = addr_of(&self.d_out, &self.stream);
        self.a_d_temp = addr_of(&self.d_temp, &self.stream);
        let out_ptrs: Vec<u64> = (0..BATCH)
            .map(|i| (self.a_d_out + i.min(self.out_slots - 1) * self.max_comp) as u64)
            .collect();
        {
            let mut view = self.arr_out.slice_mut(0..BATCH);
            self.stream.memcpy_htod(&out_ptrs, &mut view)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    fn symbol_name(&self, suffix: &[u8]) -> Vec<u8> {
        let mut s = b"nvcompBatched".to_vec();
        s.extend_from_slice(self.codec.symbol_prefix());
        s.extend_from_slice(suffix);
        s.push(0);
        s
    }

    fn load_max_output_symbol(&self) -> Result<FnGenericMaxOut> {
        unsafe {
            let f: libloading::Symbol<FnGenericMaxOut> = self
                .lib
                .get(&self.symbol_name(b"CompressGetMaxOutputChunkSize"))?;
            Ok(*f)
        }
    }

    fn load_comp_temp_symbol(&self) -> Result<FnGenericCompTemp> {
        unsafe {
            let f: libloading::Symbol<FnGenericCompTemp> = self
                .lib
                .get(&self.symbol_name(b"CompressGetTempSizeAsync"))?;
            Ok(*f)
        }
    }

    fn load_decomp_temp_symbol(&self) -> Result<FnGenericDecompTemp> {
        unsafe {
            let f: libloading::Symbol<FnGenericDecompTemp> = self
                .lib
                .get(&self.symbol_name(b"DecompressGetTempSizeAsync"))?;
            Ok(*f)
        }
    }

    pub fn decompress_device(
        &mut self,
        input_ptrs: &[u64],
        input_sizes: &[u64],
        output_ptrs: &[u64],
        output_caps: &[u64],
    ) -> Result<Vec<u64>> {
        if input_ptrs.len() != input_sizes.len()
            || input_ptrs.len() != output_ptrs.len()
            || input_ptrs.len() != output_caps.len()
        {
            bail!("decompress pointer/size length mismatch");
        }
        if input_ptrs.len() > BATCH {
            bail!(
                "nvCOMP {} accepts at most {BATCH} chunks",
                self.codec.name()
            );
        }
        if input_ptrs.is_empty() {
            return Ok(Vec::new());
        }
        self.ctx.bind_to_thread()?;
        let n = input_ptrs.len();
        {
            let mut view = self.arr_in.slice_mut(0..n);
            self.stream.memcpy_htod(input_ptrs, &mut view)?;
        }
        {
            let mut view = self.s_in.slice_mut(0..n);
            self.stream.memcpy_htod(input_sizes, &mut view)?;
        }
        {
            let mut view = self.arr_out.slice_mut(0..n);
            self.stream.memcpy_htod(output_ptrs, &mut view)?;
        }
        {
            let mut view = self.s_out_cap.slice_mut(0..n);
            self.stream.memcpy_htod(output_caps, &mut view)?;
        }
        self.stream.synchronize()?;
        let max_out = output_caps.iter().copied().max().unwrap_or(0) as usize;
        let total_out = output_caps.iter().sum::<u64>() as usize;
        let status = unsafe {
            (self.f_decompress)(
                self.a_arr_in as *const *const c_void,
                self.a_s_in as *const usize,
                self.a_s_out_cap as *const usize,
                self.a_s_result as *mut usize,
                n,
                self.a_d_temp as *mut c_void,
                self.temp_bytes,
                self.a_arr_out as *const *mut c_void,
                self.codec.decompress_opts(),
                self.a_status as *mut i32,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        let _ = (max_out, total_out);
        check(status, "GenericDecompressAsync(launch)")?;
        self.stream.synchronize()?;
        self.check_statuses(n, "decompress")?;
        self.read_u64(&self.s_result, n)
    }

    pub fn decompress_sizes_device(
        &mut self,
        input_ptrs: &[u64],
        input_sizes: &[u64],
    ) -> Result<Vec<u64>> {
        if input_ptrs.len() != input_sizes.len() {
            bail!("decompress size pointer/size length mismatch");
        }
        if input_ptrs.len() > BATCH {
            bail!(
                "nvCOMP {} accepts at most {BATCH} chunks",
                self.codec.name()
            );
        }
        if input_ptrs.is_empty() {
            return Ok(Vec::new());
        }
        self.ctx.bind_to_thread()?;
        let n = input_ptrs.len();
        {
            let mut view = self.arr_in.slice_mut(0..n);
            self.stream.memcpy_htod(input_ptrs, &mut view)?;
        }
        {
            let mut view = self.s_in.slice_mut(0..n);
            self.stream.memcpy_htod(input_sizes, &mut view)?;
        }
        self.stream.synchronize()?;
        let status = unsafe {
            (self.f_get_decomp_size)(
                self.a_arr_in as *const *const c_void,
                self.a_s_in as *const usize,
                self.a_s_result as *mut usize,
                n,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        check(status, "GenericGetDecompressSizeAsync(launch)")?;
        self.stream.synchronize()?;
        self.read_u64(&self.s_result, n)
    }

    fn check_statuses(&self, n: usize, op: &str) -> Result<()> {
        let statuses = self.read_i32(&self.statuses, n)?;
        if let Some(bad) = statuses.iter().position(|&s| s != 0) {
            bail!(
                "nvCOMP {} {op} chunk {bad} reported status {}",
                self.codec.name(),
                statuses[bad]
            );
        }
        Ok(())
    }

    fn read_u64(&self, slice: &CudaSlice<u64>, n: usize) -> Result<Vec<u64>> {
        let mut v = vec![0u64; n];
        let view = slice.slice(0..n);
        self.stream.memcpy_dtoh(&view, &mut v)?;
        self.stream.synchronize()?;
        Ok(v)
    }

    fn read_i32(&self, slice: &CudaSlice<i32>, n: usize) -> Result<Vec<i32>> {
        let mut v = vec![0i32; n];
        let view = slice.slice(0..n);
        self.stream.memcpy_dtoh(&view, &mut v)?;
        self.stream.synchronize()?;
        Ok(v)
    }
}

/// nvCOMP LZ4 compression options (matches `nvcompBatchedLZ4CompressOpts_t`).
/// All-zero = `{ NVCOMP_TYPE_CHAR, NVCOMP_BITSHUFFLE_NONE, reserved }`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Lz4CompressOpts {
    data_type: i32,
    bitshuffle_mode: i32,
    reserved: [u8; 56],
}
impl Default for Lz4CompressOpts {
    fn default() -> Self {
        Lz4CompressOpts {
            data_type: 0,
            bitshuffle_mode: 0,
            reserved: [0; 56],
        }
    }
}

/// nvCOMP LZ4 decompression options (matches `nvcompBatchedLZ4DecompressOpts_t`).
#[repr(C)]
#[derive(Clone, Copy)]
struct Lz4DecompressOpts {
    backend: i32,
    sort_before_hw_decompress: i32,
    data_type: i32,
    bitshuffle_mode: i32,
    reserved: [u8; 48],
}
impl Default for Lz4DecompressOpts {
    fn default() -> Self {
        Lz4DecompressOpts {
            backend: 0,
            sort_before_hw_decompress: 0,
            data_type: 0,
            bitshuffle_mode: 0,
            reserved: [0; 48],
        }
    }
}

type FnMaxOut = unsafe extern "C" fn(usize, Lz4CompressOpts, *mut usize) -> i32;
type FnCompTemp = unsafe extern "C" fn(usize, usize, Lz4CompressOpts, *mut usize, usize) -> i32;
type FnDecompTemp = unsafe extern "C" fn(usize, usize, Lz4DecompressOpts, *mut usize, usize) -> i32;
#[allow(clippy::type_complexity)]
type FnCompress = unsafe extern "C" fn(
    *const *const c_void, // device_uncompressed_chunk_ptrs
    *const usize,         // device_uncompressed_chunk_bytes
    usize,                // max_uncompressed_chunk_bytes
    usize,                // num_chunks
    *mut c_void,          // device_temp_ptr
    usize,                // temp_bytes
    *const *mut c_void,   // device_compressed_chunk_ptrs
    *mut usize,           // device_compressed_chunk_bytes (out)
    Lz4CompressOpts,
    *mut i32,    // device_statuses
    *mut c_void, // stream
) -> i32;
#[allow(clippy::type_complexity)]
type FnDecompress = unsafe extern "C" fn(
    *const *const c_void, // device_compressed_chunk_ptrs
    *const usize,         // device_compressed_chunk_bytes
    *const usize,         // device_uncompressed_buffer_bytes
    *mut usize,           // device_uncompressed_chunk_bytes (out, optional)
    usize,                // num_chunks
    *mut c_void,          // device_temp_ptr
    usize,                // temp_bytes
    *const *mut c_void,   // device_uncompressed_chunk_ptrs
    Lz4DecompressOpts,
    *mut i32,    // device_statuses
    *mut c_void, // stream
) -> i32;

/// Loaded nvCOMP LZ4 codec with persistent device scratch for one batch.
pub struct Lz4Codec {
    #[allow(dead_code)]
    lib: Library,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    f_compress: FnCompress,
    f_decompress: FnDecompress,

    // Device scratch (kept alive; cached addresses below stay valid for life).
    d_in: CudaSlice<u8>,  // BATCH × CHUNK_SIZE  (uncompressed in / out)
    d_out: CudaSlice<u8>, // BATCH × max_comp    (compressed out / in)
    #[allow(dead_code)]
    d_temp: CudaSlice<u8>,
    #[allow(dead_code)]
    arr_in: CudaSlice<u64>, // [addr(d_in)+i*CHUNK_SIZE]
    #[allow(dead_code)]
    arr_out: CudaSlice<u64>, // [addr(d_out)+i*max_comp]
    #[allow(dead_code)]
    s_uncomp: CudaSlice<u64>, // [CHUNK_SIZE; BATCH] (constant)
    s_comp: CudaSlice<u64>,   // compressed sizes (decompress in)
    s_result: CudaSlice<u64>, // output sizes (compress out / decompress out)
    statuses: CudaSlice<i32>,

    max_comp: usize,
    temp_bytes: usize,

    // Cached device addresses (usize) of the buffers above.
    a_d_out: usize,
    a_d_temp: usize,
    a_arr_in: usize,
    a_arr_out: usize,
    a_s_uncomp: usize,
    a_s_comp: usize,
    a_s_result: usize,
    a_status: usize,
}

/// Root nvCOMP installs under on Windows, across releases.
const NVCOMP_INSTALL_ROOT: &str = r"C:\Program Files\NVIDIA nvCOMP";

/// Batched-API DLL base names nvCOMP has shipped under across major versions
/// (newest first; the trailing major-version suffix has moved 3 -> 4 -> 5).
const NVCOMP_DLL_NAMES: &[&str] = &[
    "nvcomp64_5.dll",
    "nvcomp64_4.dll",
    "nvcomp64_3.dll",
    "nvcomp64.dll",
];

/// `v*` version directories directly under [`NVCOMP_INSTALL_ROOT`], newest
/// first (plain lexicographic descending is good enough for `v5.2`-style
/// names).
fn installed_nvcomp_versions() -> Vec<String> {
    let mut versions: Vec<String> = std::fs::read_dir(NVCOMP_INSTALL_ROOT)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| name.starts_with('v'))
        .collect();
    versions.sort_unstable_by(|a, b| b.cmp(a));
    versions
}

/// `bin\<N>` CUDA-toolkit-version subdirectories under a given nvCOMP version
/// directory, newest (highest numbered) first.
fn cuda_bin_dirs(version_dir: &str) -> Vec<String> {
    let bin_root = format!(r"{NVCOMP_INSTALL_ROOT}\{version_dir}\bin");
    let mut dirs: Vec<String> = std::fs::read_dir(&bin_root)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    dirs.sort_unstable_by_key(|name| std::cmp::Reverse(name.parse::<i64>().unwrap_or(-1)));
    dirs
}

/// nvCOMP DLL paths worth trying, in priority order:
/// 1. `NVCOMP_DLL`, if set — an explicit override with no further fallback.
/// 2. Every `<install root>\v*\bin\<N>\<name>` combination actually present
///    on disk, so installs other than exactly v5.2/CUDA 12 are still found
///    (older/newer nvCOMP releases, or a machine with a different CUDA
///    toolkit's bin layout).
///
/// Deliberately *not* included: bare filenames like `nvcomp64_5.dll` passed
/// to `LoadLibraryExW` with no path fall back to the default Windows DLL
/// search order, which includes the executable's own directory and `PATH` —
/// letting an attacker who can drop a same-named file next to `vramdisk.exe`
/// (or earlier on `PATH`) get it loaded and executed in-process. Every
/// candidate here is a fully qualified path for that reason.
fn candidate_dll_paths() -> Vec<String> {
    if let Ok(p) = std::env::var("NVCOMP_DLL") {
        return vec![p];
    }
    let mut candidates = Vec::new();
    for version in installed_nvcomp_versions() {
        for bin in cuda_bin_dirs(&version) {
            for name in NVCOMP_DLL_NAMES {
                candidates.push(format!(r"{NVCOMP_INSTALL_ROOT}\{version}\bin\{bin}\{name}"));
            }
        }
    }
    candidates
}

/// Try each candidate nvCOMP DLL path in turn, returning the first that
/// loads successfully (and the path that worked, for diagnostics).
fn load_nvcomp_library() -> Result<(Library, String)> {
    let candidates = candidate_dll_paths();
    let mut last_err: Option<libloading::Error> = None;
    for path in &candidates {
        match unsafe { Library::new(path) } {
            Ok(lib) => return Ok((lib, path.clone())),
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow::anyhow!(
        "failed to load nvCOMP DLL; tried {} candidate path(s) (set NVCOMP_DLL to override): {}",
        candidates.len(),
        last_err.map(|e| e.to_string()).unwrap_or_default(),
    ))
}

/// Whether the nvCOMP DLL can be found and loaded on this machine (no CUDA
/// context needed for this check). The CLI doesn't need this — its codecs
/// just fall back to CPU zstd when the load fails — but the GUI uses it to
/// gray out GPU-compression options up front instead of offering a silent
/// CPU fallback.
pub fn nvcomp_available() -> bool {
    load_nvcomp_library().is_ok()
}

impl Lz4Codec {
    /// Load nvCOMP and allocate batch scratch, sharing `vram`'s context/stream.
    pub fn load(vram: &Vram) -> Result<Self> {
        let (lib, _path) = load_nvcomp_library()?;

        // Resolve the symbols we use.
        let (f_max, f_comp_temp, f_decomp_temp, f_compress, f_decompress) = unsafe {
            let f_max: libloading::Symbol<FnMaxOut> =
                lib.get(b"nvcompBatchedLZ4CompressGetMaxOutputChunkSize\0")?;
            let f_comp_temp: libloading::Symbol<FnCompTemp> =
                lib.get(b"nvcompBatchedLZ4CompressGetTempSizeAsync\0")?;
            let f_decomp_temp: libloading::Symbol<FnDecompTemp> =
                lib.get(b"nvcompBatchedLZ4DecompressGetTempSizeAsync\0")?;
            let f_compress: libloading::Symbol<FnCompress> =
                lib.get(b"nvcompBatchedLZ4CompressAsync\0")?;
            let f_decompress: libloading::Symbol<FnDecompress> =
                lib.get(b"nvcompBatchedLZ4DecompressAsync\0")?;
            (
                *f_max,
                *f_comp_temp,
                *f_decomp_temp,
                *f_compress,
                *f_decompress,
            )
        };

        let ctx = vram.context();
        let stream = vram.stream();
        ctx.bind_to_thread().context("bind ctx for nvcomp init")?;

        let copts = Lz4CompressOpts::default();
        let dopts = Lz4DecompressOpts::default();
        let cs = CHUNK_SIZE as usize;
        let total = BATCH * cs;

        // Query sizes on the host (these calls don't touch the device).
        let mut max_comp = 0usize;
        check(
            unsafe { f_max(cs, copts, &mut max_comp) },
            "GetMaxOutputChunkSize",
        )?;
        let mut ctemp = 0usize;
        check(
            unsafe { f_comp_temp(BATCH, cs, copts, &mut ctemp, total) },
            "CompressGetTempSize",
        )?;
        let mut dtemp = 0usize;
        check(
            unsafe { f_decomp_temp(BATCH, cs, dopts, &mut dtemp, total) },
            "DecompressGetTempSize",
        )?;
        let temp_bytes = ctemp.max(dtemp).max(1);

        // Allocate persistent batch scratch.
        let d_in = stream.alloc_zeros::<u8>(BATCH * cs)?;
        let d_out = stream.alloc_zeros::<u8>(BATCH * max_comp)?;
        let d_temp = stream.alloc_zeros::<u8>(temp_bytes)?;
        let mut arr_in = stream.alloc_zeros::<u64>(BATCH)?;
        let mut arr_out = stream.alloc_zeros::<u64>(BATCH)?;
        let mut s_uncomp = stream.alloc_zeros::<u64>(BATCH)?;
        let s_comp = stream.alloc_zeros::<u64>(BATCH)?;
        let s_result = stream.alloc_zeros::<u64>(BATCH)?;
        let statuses = stream.alloc_zeros::<i32>(BATCH)?;
        stream.synchronize()?;

        // Cache device base addresses.
        let a_d_in = addr_of(&d_in, &stream);
        let a_d_out = addr_of(&d_out, &stream);
        let a_d_temp = addr_of(&d_temp, &stream);
        let a_arr_in = addr_of(&arr_in, &stream);
        let a_arr_out = addr_of(&arr_out, &stream);
        let a_s_uncomp = addr_of(&s_uncomp, &stream);
        let a_s_comp = addr_of(&s_comp, &stream);
        let a_s_result = addr_of(&s_result, &stream);
        let a_status = addr_of(&statuses, &stream);

        // Per-chunk pointer arrays into the contiguous in/out regions, and the
        // constant uncompressed size for every slot.
        let in_ptrs: Vec<u64> = (0..BATCH).map(|i| (a_d_in + i * cs) as u64).collect();
        let out_ptrs: Vec<u64> = (0..BATCH)
            .map(|i| (a_d_out + i * max_comp) as u64)
            .collect();
        let sizes: Vec<u64> = vec![cs as u64; BATCH];
        stream.memcpy_htod(&in_ptrs, &mut arr_in)?;
        stream.memcpy_htod(&out_ptrs, &mut arr_out)?;
        stream.memcpy_htod(&sizes, &mut s_uncomp)?;
        stream.synchronize()?;

        Ok(Lz4Codec {
            lib,
            ctx,
            stream,
            f_compress,
            f_decompress,
            d_in,
            d_out,
            d_temp,
            arr_in,
            arr_out,
            s_uncomp,
            s_comp,
            s_result,
            statuses,
            max_comp,
            temp_bytes,
            a_d_out,
            a_d_temp,
            a_arr_in,
            a_arr_out,
            a_s_uncomp,
            a_s_comp,
            a_s_result,
            a_status,
        })
    }

    /// Compress one full 64 KiB chunk. Returns `Some(bytes)` if the result is
    /// strictly smaller than the input, else `None` (store uncompressed).
    pub fn compress(&mut self, src: &[u8]) -> Result<Option<Vec<u8>>> {
        assert_eq!(
            src.len(),
            CHUNK_SIZE as usize,
            "compress expects a full chunk"
        );
        Ok(self.compress_batch(src)?.pop().unwrap())
    }

    /// Compress a contiguous run of full 64 KiB chunks (`data.len()` must be a
    /// multiple of `CHUNK_SIZE`). Returns one entry per chunk: `Some(bytes)`
    /// when LZ4 shrank it, `None` when it did not (store raw).
    pub fn compress_batch(&mut self, data: &[u8]) -> Result<Vec<Option<Vec<u8>>>> {
        let cs = CHUNK_SIZE as usize;
        assert_eq!(data.len() % cs, 0, "compress_batch expects whole chunks");
        let n = data.len() / cs;
        self.ctx.bind_to_thread()?;

        let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
        let mut base = 0usize;
        while base < n {
            let m = (n - base).min(BATCH);
            self.compress_group(&data[base * cs..(base + m) * cs], m, &mut out)?;
            base += m;
        }
        Ok(out)
    }

    /// Largest number of chunks one batched call processes.
    pub const fn max_batch() -> usize {
        BATCH
    }

    /// Device address of compressed-blob slot `i` within `d_out` (where both the
    /// compressed output lands and the decompress input is read). Valid until
    /// the next compress/decompress call overwrites the scratch.
    pub fn comp_slot_ptr(&self, i: usize) -> u64 {
        (self.a_d_out + i * self.max_comp) as u64
    }

    /// Device address of uncompressed scratch slot `i` within `d_in`.
    pub fn uncomp_slot_ptr(&self, i: usize) -> u64 {
        let cs = CHUNK_SIZE as usize;
        let (ptr, _guard) = self.d_in.device_ptr(&self.stream);
        ptr as u64 + (i * cs) as u64
    }

    /// Compress `m` (≤ BATCH) contiguous chunks and *leave the results on the
    /// device* in `d_out` slots `0..m`. Returns `Some(len)` for chunks that
    /// shrank (blob at [`comp_slot_ptr`]) or `None` for chunks to store raw.
    ///
    /// Lets the caller move blobs device-to-device (e.g. into a packed VRAM
    /// arena) without a host round-trip. Blobs are only valid until the next
    /// call that touches the scratch.
    ///
    /// [`comp_slot_ptr`]: Lz4Codec::comp_slot_ptr
    pub fn compress_group_dev(&mut self, data: &[u8], m: usize) -> Result<Vec<Option<usize>>> {
        self.ctx.bind_to_thread()?;
        self.compress_core(data, m)
    }

    /// Compress exactly `m` (≤ BATCH) chunks given contiguously in `data`,
    /// appending one host-side result per chunk to `out`.
    fn compress_group(
        &mut self,
        data: &[u8],
        m: usize,
        out: &mut Vec<Option<Vec<u8>>>,
    ) -> Result<()> {
        let cs = CHUNK_SIZE as usize;
        let sizes = self.compress_core(data, m)?;
        // Pull back each blob that shrank; incompressible chunks need no copy.
        let mut blobs: Vec<Option<Vec<u8>>> = Vec::with_capacity(m);
        for (i, sz) in sizes.iter().enumerate() {
            match sz {
                Some(sz) => {
                    let mut buf = vec![0u8; *sz];
                    let start = i * self.max_comp;
                    let view = self.d_out.slice(start..start + *sz);
                    self.stream.memcpy_dtoh(&view, &mut buf)?;
                    blobs.push(Some(buf));
                }
                None => blobs.push(None),
            }
        }
        let _ = cs;
        self.stream.synchronize()?;
        out.extend(blobs);
        Ok(())
    }

    /// Core compression for one ≤ BATCH group: upload `m` chunks, launch nvCOMP,
    /// and return per-chunk `Some(compressed_len)` (blob resident in `d_out`
    /// slot i) or `None` when the chunk did not shrink.
    fn compress_core(&mut self, data: &[u8], m: usize) -> Result<Vec<Option<usize>>> {
        let cs = CHUNK_SIZE as usize;
        {
            let mut view = self.d_in.slice_mut(0..m * cs);
            self.stream.memcpy_htod(data, &mut view)?;
        }
        self.stream.synchronize()?;

        let status = unsafe {
            (self.f_compress)(
                self.a_arr_in as *const *const c_void,
                self.a_s_uncomp as *const usize,
                cs,
                m,
                self.a_d_temp as *mut c_void,
                self.temp_bytes,
                self.a_arr_out as *const *mut c_void,
                self.a_s_result as *mut usize,
                Lz4CompressOpts::default(),
                self.a_status as *mut i32,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        check(status, "CompressAsync(launch)")?;
        self.stream.synchronize()?;

        let statuses = self.read_i32(&self.statuses, m)?;
        if let Some(bad) = statuses.iter().position(|&s| s != 0) {
            bail!(
                "nvCOMP compress chunk {bad} reported status {}",
                statuses[bad]
            );
        }
        let sizes = self.read_u64(&self.s_result, m)?;
        Ok(sizes
            .iter()
            .map(|&sz| {
                let sz = sz as usize;
                if sz == 0 || sz >= cs {
                    None
                } else {
                    Some(sz)
                }
            })
            .collect())
    }

    /// Decompress `comp` back into a full 64 KiB chunk.
    pub fn decompress(&mut self, comp: &[u8], out_len: usize) -> Result<Vec<u8>> {
        let mut full = self.decompress_batch(&[comp])?.pop().unwrap();
        full.truncate(out_len);
        Ok(full)
    }

    /// Decompress a batch of blobs, each back to a full 64 KiB chunk. Returns
    /// one `Vec<u8>` of `CHUNK_SIZE` bytes per input blob.
    pub fn decompress_batch(&mut self, blobs: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        let cs = CHUNK_SIZE as usize;
        self.ctx.bind_to_thread()?;
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(blobs.len());
        for group in blobs.chunks(BATCH) {
            self.decompress_group(group, &mut out)?;
        }
        let _ = cs;
        Ok(out)
    }

    /// Decompress one group of ≤ BATCH blobs.
    fn decompress_group(&mut self, blobs: &[&[u8]], out: &mut Vec<Vec<u8>>) -> Result<()> {
        let cs = CHUNK_SIZE as usize;
        let m = blobs.len();
        let mut comp_sizes = vec![0u64; m];
        for (i, b) in blobs.iter().enumerate() {
            if b.len() > self.max_comp {
                bail!(
                    "compressed blob {} exceeds scratch {}",
                    b.len(),
                    self.max_comp
                );
            }
            let start = i * self.max_comp;
            let mut view = self.d_out.slice_mut(start..start + b.len());
            self.stream.memcpy_htod(*b, &mut view)?;
            comp_sizes[i] = b.len() as u64;
        }
        {
            let mut view = self.s_comp.slice_mut(0..m);
            self.stream.memcpy_htod(&comp_sizes, &mut view)?;
        }
        self.stream.synchronize()?;

        let status = unsafe {
            (self.f_decompress)(
                self.a_arr_out as *const *const c_void, // compressed in = d_out
                self.a_s_comp as *const usize,
                self.a_s_uncomp as *const usize, // output capacity = CHUNK_SIZE
                self.a_s_result as *mut usize,   // actual decompressed bytes (out)
                m,
                self.a_d_temp as *mut c_void,
                self.temp_bytes,
                self.a_arr_in as *const *mut c_void, // uncompressed out = d_in
                Lz4DecompressOpts::default(),
                self.a_status as *mut i32,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        check(status, "DecompressAsync(launch)")?;
        self.stream.synchronize()?;

        let statuses = self.read_i32(&self.statuses, m)?;
        if let Some(bad) = statuses.iter().position(|&s| s != 0) {
            bail!(
                "nvCOMP decompress chunk {bad} reported status {}",
                statuses[bad]
            );
        }

        // One D2H of the whole decompressed region, then split per chunk.
        let mut whole = vec![0u8; m * cs];
        {
            let view = self.d_in.slice(0..m * cs);
            self.stream.memcpy_dtoh(&view, &mut whole)?;
        }
        self.stream.synchronize()?;
        for i in 0..m {
            out.push(whole[i * cs..(i + 1) * cs].to_vec());
        }
        Ok(())
    }

    /// Decompress blobs that live directly in the VRAM buffer (the packed
    /// arena), reading them device-to-device into the codec scratch — no host
    /// upload. `vram_base` is the device address of the VRAM buffer start; each
    /// `(offset, len)` is a compressed blob at `vram_base + offset`. Returns one
    /// full `CHUNK_SIZE` chunk per blob.
    #[allow(dead_code)]
    pub fn decompress_from_arena(
        &mut self,
        vram_base: u64,
        blobs: &[(u64, u32)],
    ) -> Result<Vec<Vec<u8>>> {
        self.ctx.bind_to_thread()?;
        let cs = CHUNK_SIZE as usize;
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(blobs.len());
        for group in blobs.chunks(BATCH) {
            let m = group.len();
            // D2D each compressed blob from the arena into d_out slot i (aligned
            // scratch), so nvCOMP reads from the same layout as the host path.
            for (i, &(off, len)) in group.iter().enumerate() {
                if len as usize > self.max_comp {
                    bail!("compressed blob {} exceeds scratch {}", len, self.max_comp);
                }
                let dst = (self.a_d_out + i * self.max_comp) as u64;
                let src = vram_base + off;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst,
                        src,
                        len as usize,
                        self.stream.cu_stream(),
                    )
                    .context("memcpy_dtod_async (decompress_from_arena)")?;
                }
            }
            let sizes: Vec<u64> = group.iter().map(|&(_, l)| l as u64).collect();
            {
                let mut view = self.s_comp.slice_mut(0..m);
                self.stream.memcpy_htod(&sizes, &mut view)?;
            }
            self.stream.synchronize()?;

            let status = unsafe {
                (self.f_decompress)(
                    self.a_arr_out as *const *const c_void, // compressed in = d_out
                    self.a_s_comp as *const usize,
                    self.a_s_uncomp as *const usize, // output capacity = CHUNK_SIZE
                    self.a_s_result as *mut usize,
                    m,
                    self.a_d_temp as *mut c_void,
                    self.temp_bytes,
                    self.a_arr_in as *const *mut c_void, // uncompressed out = d_in
                    Lz4DecompressOpts::default(),
                    self.a_status as *mut i32,
                    self.stream.cu_stream() as *mut c_void,
                )
            };
            check(status, "DecompressAsync(launch)")?;
            self.stream.synchronize()?;
            let statuses = self.read_i32(&self.statuses, m)?;
            if let Some(bad) = statuses.iter().position(|&s| s != 0) {
                bail!(
                    "nvCOMP decompress chunk {bad} reported status {}",
                    statuses[bad]
                );
            }

            let mut whole = vec![0u8; m * cs];
            {
                let view = self.d_in.slice(0..m * cs);
                self.stream.memcpy_dtoh(&view, &mut whole)?;
            }
            self.stream.synchronize()?;
            for i in 0..m {
                out.push(whole[i * cs..(i + 1) * cs].to_vec());
            }
        }
        Ok(out)
    }

    /// Decompress arena-resident blobs like [`decompress_from_arena`], but copy
    /// only the requested uncompressed slice of each chunk back to host.
    pub fn decompress_from_arena_slices(
        &mut self,
        vram_base: u64,
        requests: &[(u64, u32, usize, usize)],
    ) -> Result<Vec<Vec<u8>>> {
        self.ctx.bind_to_thread()?;
        let cs = CHUNK_SIZE as usize;
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(requests.len());
        for group in requests.chunks(BATCH) {
            let m = group.len();
            for (i, &(off, len, in_off, take)) in group.iter().enumerate() {
                if len as usize > self.max_comp {
                    bail!("compressed blob {} exceeds scratch {}", len, self.max_comp);
                }
                if in_off.checked_add(take).is_none_or(|end| end > cs) {
                    bail!("decompress slice out of bounds: offset={in_off} len={take}");
                }
                let dst = (self.a_d_out + i * self.max_comp) as u64;
                let src = vram_base + off;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst,
                        src,
                        len as usize,
                        self.stream.cu_stream(),
                    )
                    .context("memcpy_dtod_async (decompress_from_arena_slices)")?;
                }
            }
            let sizes: Vec<u64> = group.iter().map(|&(_, len, _, _)| len as u64).collect();
            {
                let mut view = self.s_comp.slice_mut(0..m);
                self.stream.memcpy_htod(&sizes, &mut view)?;
            }
            self.stream.synchronize()?;

            let status = unsafe {
                (self.f_decompress)(
                    self.a_arr_out as *const *const c_void,
                    self.a_s_comp as *const usize,
                    self.a_s_uncomp as *const usize,
                    self.a_s_result as *mut usize,
                    m,
                    self.a_d_temp as *mut c_void,
                    self.temp_bytes,
                    self.a_arr_in as *const *mut c_void,
                    Lz4DecompressOpts::default(),
                    self.a_status as *mut i32,
                    self.stream.cu_stream() as *mut c_void,
                )
            };
            check(status, "DecompressAsync(launch slices)")?;
            self.stream.synchronize()?;
            let statuses = self.read_i32(&self.statuses, m)?;
            if let Some(bad) = statuses.iter().position(|&s| s != 0) {
                bail!(
                    "nvCOMP decompress chunk {bad} reported status {}",
                    statuses[bad]
                );
            }

            for (i, &(_, _, in_off, take)) in group.iter().enumerate() {
                let mut piece = vec![0u8; take];
                {
                    let start = i * cs + in_off;
                    let view = self.d_in.slice(start..start + take);
                    self.stream.memcpy_dtoh(&view, &mut piece)?;
                }
                out.push(piece);
            }
            self.stream.synchronize()?;
        }
        Ok(out)
    }

    /// Decompress up to one batch of arena-resident blobs into `d_in` slots
    /// without copying the uncompressed bytes back to host. The caller can read
    /// slot addresses with [`uncomp_slot_ptr`] until the next codec operation.
    pub fn decompress_from_arena_dev(
        &mut self,
        vram_base: u64,
        blobs: &[(u64, u32)],
    ) -> Result<()> {
        if blobs.is_empty() {
            return Ok(());
        }
        if blobs.len() > BATCH {
            bail!("decompress_from_arena_dev accepts at most {BATCH} blobs");
        }
        self.ctx.bind_to_thread()?;
        let m = blobs.len();
        for (i, &(off, len)) in blobs.iter().enumerate() {
            if len as usize > self.max_comp {
                bail!("compressed blob {} exceeds scratch {}", len, self.max_comp);
            }
            let dst = (self.a_d_out + i * self.max_comp) as u64;
            let src = vram_base + off;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst,
                    src,
                    len as usize,
                    self.stream.cu_stream(),
                )
                .context("memcpy_dtod_async (decompress_from_arena_dev)")?;
            }
        }
        let sizes: Vec<u64> = blobs.iter().map(|&(_, l)| l as u64).collect();
        {
            let mut view = self.s_comp.slice_mut(0..m);
            self.stream.memcpy_htod(&sizes, &mut view)?;
        }
        self.stream.synchronize()?;

        let status = unsafe {
            (self.f_decompress)(
                self.a_arr_out as *const *const c_void,
                self.a_s_comp as *const usize,
                self.a_s_uncomp as *const usize,
                self.a_s_result as *mut usize,
                m,
                self.a_d_temp as *mut c_void,
                self.temp_bytes,
                self.a_arr_in as *const *mut c_void,
                Lz4DecompressOpts::default(),
                self.a_status as *mut i32,
                self.stream.cu_stream() as *mut c_void,
            )
        };
        check(status, "DecompressAsync(launch dev)")?;
        self.stream.synchronize()?;
        let statuses = self.read_i32(&self.statuses, m)?;
        if let Some(bad) = statuses.iter().position(|&s| s != 0) {
            bail!(
                "nvCOMP decompress chunk {bad} reported status {}",
                statuses[bad]
            );
        }
        Ok(())
    }

    fn read_u64(&self, slice: &CudaSlice<u64>, n: usize) -> Result<Vec<u64>> {
        let mut v = vec![0u64; n];
        let view = slice.slice(0..n);
        self.stream.memcpy_dtoh(&view, &mut v)?;
        self.stream.synchronize()?;
        Ok(v)
    }

    fn read_i32(&self, slice: &CudaSlice<i32>, n: usize) -> Result<Vec<i32>> {
        let mut v = vec![0i32; n];
        let view = slice.slice(0..n);
        self.stream.memcpy_dtoh(&view, &mut v)?;
        self.stream.synchronize()?;
        Ok(v)
    }
}

/// Read the device address of a slice as a `usize`.
fn addr_of<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> usize {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr as usize
}

fn check(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("nvCOMP {what} failed with status {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Requires nvCOMP installed + a GPU; run with: cargo test -- --ignored
    #[test]
    #[ignore]
    fn lz4_roundtrip() {
        let vram = Vram::new(0, 4 * 1024 * 1024).expect("vram");
        let mut codec = Lz4Codec::load(&vram).expect("load nvcomp");
        let cs = CHUNK_SIZE as usize;

        // Highly compressible: long runs.
        let src: Vec<u8> = (0..cs).map(|i| (i / 512) as u8).collect();
        let comp = codec
            .compress(&src)
            .expect("compress")
            .expect("should shrink");
        assert!(comp.len() < cs, "compressed {} !< {}", comp.len(), cs);
        let back = codec.decompress(&comp, cs).expect("decompress");
        assert_eq!(back, src, "lz4 roundtrip mismatch");

        // Incompressible: pseudo-random. Should not shrink (returns None), but
        // if it does, it must still round-trip.
        let mut s = 0x1234_5678u32;
        let rnd: Vec<u8> = (0..cs)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        if let Some(cc) = codec.compress(&rnd).expect("compress rnd") {
            assert_eq!(codec.decompress(&cc, cs).expect("decompress rnd"), rnd);
        }
    }

    // Batched round-trip over many mixed chunks in one call.
    #[test]
    #[ignore]
    fn lz4_batch_roundtrip() {
        let vram = Vram::new(0, 8 * 1024 * 1024).expect("vram");
        let mut codec = Lz4Codec::load(&vram).expect("load nvcomp");
        let cs = CHUNK_SIZE as usize;
        let n = 300; // exceeds BATCH to exercise grouping

        // Build n compressible chunks with per-chunk distinct content.
        let mut data = vec![0u8; n * cs];
        for c in 0..n {
            for i in 0..cs {
                data[c * cs + i] = ((i / 64) as u8).wrapping_add(c as u8);
            }
        }
        let comps = codec.compress_batch(&data).expect("compress_batch");
        assert_eq!(comps.len(), n);
        assert!(comps.iter().all(|c| c.is_some()), "all should shrink");

        let blobs: Vec<&[u8]> = comps
            .iter()
            .map(|c| c.as_ref().unwrap().as_slice())
            .collect();
        let back = codec.decompress_batch(&blobs).expect("decompress_batch");
        assert_eq!(back.len(), n);
        for c in 0..n {
            assert_eq!(back[c], &data[c * cs..(c + 1) * cs], "chunk {c} mismatch");
        }
    }
}
