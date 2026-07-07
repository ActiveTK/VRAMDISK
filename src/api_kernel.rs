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
#[derive(Clone, Copy, Default)]
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
pub struct ApiKernel {
    module: sys::CUmodule,
    init_func: sys::CUfunction,
    update_func: sys::CUfunction,
    final_func: sys::CUfunction,
    many_init_func: sys::CUfunction,
    many_update_func: sys::CUfunction,
    many_final_func: sys::CUfunction,
    crc32_many_func: sys::CUfunction,
    segs_d: CudaSlice<HashSegment>,
    files_d: CudaSlice<HashFileDesc>,
    state_d: CudaSlice<u8>,
    states_d: CudaSlice<u8>,
    out_d: CudaSlice<u8>,
    outs_d: CudaSlice<u8>,
    crc_out_d: CudaSlice<u32>,
    status_d: CudaSlice<u32>,
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

impl ApiKernel {
    pub fn new(vram: &Vram) -> Result<Self> {
        let ctx = vram.context();
        let stream = vram.stream();
        ctx.bind_to_thread().context("bind ctx for ApiKernel")?;

        let ptx = compile_ptx_with_opts(
            API_CUDA,
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
        .context("compile API CUDA kernels with NVRTC")?;

        let mut ptx_nul = ptx.to_src().into_bytes();
        ptx_nul.push(0);
        let module = unsafe {
            dr::module::load_data(ptx_nul.as_ptr() as *const c_void)
                .context("cuModuleLoadData for API kernels")?
        };
        let init_func = get_func(module, "vramdisk_hash_init")?;
        let update_func = get_func(module, "vramdisk_hash_update")?;
        let final_func = get_func(module, "vramdisk_hash_final")?;
        let many_init_func = get_func(module, "vramdisk_hash_many_init")?;
        let many_update_func = get_func(module, "vramdisk_hash_many_update")?;
        let many_final_func = get_func(module, "vramdisk_hash_many_final")?;
        let crc32_many_func = get_func(module, "vramdisk_crc32_many")?;

        let segs_d = stream
            .alloc_zeros::<HashSegment>(DEFAULT_SEG_CAP)
            .context("alloc API segment scratch")?;
        let files_d = stream
            .alloc_zeros::<HashFileDesc>(1)
            .context("alloc API file scratch")?;
        let state_d = stream.alloc_zeros::<u8>(256).context("alloc API state")?;
        let states_d = stream
            .alloc_zeros::<u8>(256)
            .context("alloc API batch states")?;
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
        stream.synchronize()?;

        Ok(Self {
            module,
            init_func,
            update_func,
            final_func,
            many_init_func,
            many_update_func,
            many_final_func,
            crc32_many_func,
            segs_d,
            files_d,
            state_d,
            states_d,
            out_d,
            outs_d,
            crc_out_d,
            status_d,
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
            .alloc_zeros::<u8>(n * 256)
            .context("grow API batch states")?;
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

    pub fn hash_many(
        &mut self,
        alg: HashAlgorithm,
        files: &[Vec<HashSegment>],
    ) -> Result<Vec<Vec<u8>>> {
        self.ctx.bind_to_thread()?;
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let nfiles = files.len();
        let total_segments: usize = files.iter().map(Vec::len).sum();
        self.reserve_files(nfiles)?;
        self.reserve_segments(total_segments.max(1))?;

        let mut flat = Vec::with_capacity(total_segments);
        let mut descs = Vec::with_capacity(nfiles);
        for file in files {
            descs.push(HashFileDesc {
                seg_start: flat.len() as u32,
                seg_count: file.len() as u32,
            });
            flat.extend_from_slice(file);
        }
        if !flat.is_empty() {
            let mut seg_view = self.segs_d.slice_mut(0..flat.len());
            self.stream.memcpy_htod(&flat, &mut seg_view)?;
        }
        {
            let mut file_view = self.files_d.slice_mut(0..nfiles);
            self.stream.memcpy_htod(&descs, &mut file_view)?;
        }

        let zero = [0u32; 1];
        self.stream.memcpy_htod(&zero, &mut self.status_d)?;

        let mut alg_id = alg as u32;
        let states_ptr = ptr_u8(&self.states_d, &self.stream);
        let status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut n = nfiles as u32;
        let mut p_states = states_ptr;
        let mut p_status = status_ptr;
        let mut init_params: [*mut c_void; 4] = [
            &mut alg_id as *mut u32 as *mut c_void,
            &mut p_states as *mut sys::CUdeviceptr as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.many_init_func,
                (nfiles as u32, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut init_params,
            )
            .context("launch vramdisk_hash_many_init")?;
        }

        let segs_ptr = ptr_seg(&self.segs_d, &self.stream);
        let files_ptr = ptr_file(&self.files_d, &self.stream);
        let mut p_segs = segs_ptr;
        let mut p_files = files_ptr;
        let mut update_params: [*mut c_void; 5] = [
            &mut p_segs as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_files as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_states as *mut sys::CUdeviceptr as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.many_update_func,
                (nfiles as u32, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut update_params,
            )
            .context("launch vramdisk_hash_many_update")?;
        }

        let outs_ptr = ptr_u8(&self.outs_d, &self.stream);
        let mut p_outs = outs_ptr;
        let mut final_params: [*mut c_void; 5] = [
            &mut alg_id as *mut u32 as *mut c_void,
            &mut p_states as *mut sys::CUdeviceptr as *mut c_void,
            &mut p_outs as *mut sys::CUdeviceptr as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
            &mut p_status as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.many_final_func,
                (nfiles as u32, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut final_params,
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

    pub fn crc32_many(&mut self, files: &[Vec<HashSegment>]) -> Result<Vec<u32>> {
        let nfiles = files.len();
        self.ctx
            .bind_to_thread()
            .context("bind ctx for crc32_many")?;
        self.reserve_files(nfiles.max(1))?;
        let mut flat = Vec::new();
        let mut desc = Vec::with_capacity(nfiles);
        for file in files {
            desc.push(HashFileDesc {
                seg_start: flat.len() as u32,
                seg_count: file.len() as u32,
            });
            flat.extend(file.iter().copied());
        }
        self.reserve_segments(flat.len().max(1))?;
        if !flat.is_empty() {
            let mut view = self.segs_d.slice_mut(0..flat.len());
            self.stream.memcpy_htod(&flat, &mut view)?;
        }
        if !desc.is_empty() {
            let mut view = self.files_d.slice_mut(0..desc.len());
            self.stream.memcpy_htod(&desc, &mut view)?;
        }
        let zero = [0u32; 1];
        self.stream.memcpy_htod(&zero, &mut self.status_d)?;
        self.stream.synchronize()?;

        let mut segs_ptr = ptr_seg(&self.segs_d, &self.stream);
        let mut files_ptr = ptr_file(&self.files_d, &self.stream);
        let mut out_ptr = ptr_u32(&self.crc_out_d, &self.stream);
        let mut nfiles_u32 = nfiles as u32;
        let mut status_ptr = ptr_u32(&self.status_d, &self.stream);
        let mut params: [*mut c_void; 5] = [
            &mut segs_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut files_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut out_ptr as *mut sys::CUdeviceptr as *mut c_void,
            &mut nfiles_u32 as *mut u32 as *mut c_void,
            &mut status_ptr as *mut sys::CUdeviceptr as *mut c_void,
        ];
        unsafe {
            dr::launch_kernel(
                self.crc32_many_func,
                (nfiles as u32, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .context("launch vramdisk_crc32_many")?;
        }
        self.stream.synchronize()?;
        self.check_status()?;
        let mut out = vec![0u32; nfiles];
        if nfiles > 0 {
            let view = self.crc_out_d.slice(0..nfiles);
            self.stream.memcpy_dtoh(&view, &mut out)?;
            self.stream.synchronize()?;
        }
        Ok(out)
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

pub fn digest_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{b:02x}").unwrap();
    }
    out
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

__device__ __forceinline__ u32 rol32(u32 x, u32 n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ u32 ror32(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ u32 ld_le32(const u8* p) { return ((u32)p[0]) | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
__device__ __forceinline__ u32 ld_be32(const u8* p) { return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | ((u32)p[3]); }
__device__ __forceinline__ void st_le32(u8* p, u32 x) { p[0]=x; p[1]=x>>8; p[2]=x>>16; p[3]=x>>24; }
__device__ __forceinline__ void st_be32(u8* p, u32 x) { p[0]=x>>24; p[1]=x>>16; p[2]=x>>8; p[3]=x; }
__device__ __forceinline__ void st_be64(u8* p, u64 x) { for (int i=0;i<8;i++) p[i] = (u8)(x >> (56 - 8*i)); }

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

__device__ void process_block(HashState* s, const u8* block) {
    if (s->alg==1) md5_block(s, block);
    else if (s->alg==2) sha1_block(s, block);
    else if (s->alg==3) sha256_block(s, block);
}

__device__ void update_byte(HashState* s, u8 b) {
    if (s->alg==4) { s->fnv ^= (u64)b; s->fnv *= 0x100000001b3ULL; s->len++; return; }
    s->buf[s->buf_len++] = b; s->len++;
    if (s->buf_len == 64) { process_block(s, s->buf); s->buf_len = 0; }
}

extern "C" __global__ void vramdisk_hash_init(u32 alg, u8* state_raw, u32* status) {
    HashState* s = (HashState*)state_raw; status[0]=0; s->alg=alg; s->len=0; s->buf_len=0;
    for (int i=0;i<64;i++) s->buf[i]=0;
    s->md5[0]=0x67452301; s->md5[1]=0xefcdab89; s->md5[2]=0x98badcfe; s->md5[3]=0x10325476;
    s->sha1[0]=0x67452301; s->sha1[1]=0xefcdab89; s->sha1[2]=0x98badcfe; s->sha1[3]=0x10325476; s->sha1[4]=0xc3d2e1f0;
    s->sha256[0]=0x6a09e667; s->sha256[1]=0xbb67ae85; s->sha256[2]=0x3c6ef372; s->sha256[3]=0xa54ff53a;
    s->sha256[4]=0x510e527f; s->sha256[5]=0x9b05688c; s->sha256[6]=0x1f83d9ab; s->sha256[7]=0x5be0cd19;
    s->fnv=0xcbf29ce484222325ULL;
    if (alg < 1 || alg > 4) status[0]=1;
}

extern "C" __global__ void vramdisk_hash_update(const HashSegment* segs, u32 nsegs, u8* state_raw, u32* status) {
    HashState* s = (HashState*)state_raw; if (status[0]!=0) return;
    for (u32 si=0; si<nsegs; si++) {
        const HashSegment sg = segs[si];
        const u8* p = (const u8*)sg.ptr;
        for (u32 i=0; i<sg.len; i++) update_byte(s, sg.kind == 1 ? 0 : p[i]);
    }
}

extern "C" __global__ void vramdisk_hash_final(u32 alg, u8* state_raw, u8* out, u32* status) {
    HashState* s = (HashState*)state_raw; if (status[0]!=0) return;
    if (alg==4) { st_be64(out, s->fnv); return; }
    u64 bits = s->len * 8ULL;
    update_byte(s, 0x80);
    while (s->buf_len != 56) update_byte(s, 0);
    if (alg==1) { for (int i=0;i<8;i++) update_byte(s, (u8)(bits >> (8*i))); }
    else { for (int i=0;i<8;i++) update_byte(s, (u8)(bits >> (56 - 8*i))); }
    if (alg==1) { for (int i=0;i<4;i++) st_le32(out+i*4, s->md5[i]); }
    else if (alg==2) { for (int i=0;i<5;i++) st_be32(out+i*4, s->sha1[i]); }
    else if (alg==3) { for (int i=0;i<8;i++) st_be32(out+i*4, s->sha256[i]); }
    else status[0]=1;
}

extern "C" __global__ void vramdisk_hash_many_init(u32 alg, u8* states_raw, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x;
    if (fi >= nfiles) return;
    HashState* s = (HashState*)(states_raw + ((u64)fi * 256ULL));
    s->alg=alg; s->len=0; s->buf_len=0;
    for (int i=0;i<64;i++) s->buf[i]=0;
    s->md5[0]=0x67452301; s->md5[1]=0xefcdab89; s->md5[2]=0x98badcfe; s->md5[3]=0x10325476;
    s->sha1[0]=0x67452301; s->sha1[1]=0xefcdab89; s->sha1[2]=0x98badcfe; s->sha1[3]=0x10325476; s->sha1[4]=0xc3d2e1f0;
    s->sha256[0]=0x6a09e667; s->sha256[1]=0xbb67ae85; s->sha256[2]=0x3c6ef372; s->sha256[3]=0xa54ff53a;
    s->sha256[4]=0x510e527f; s->sha256[5]=0x9b05688c; s->sha256[6]=0x1f83d9ab; s->sha256[7]=0x5be0cd19;
    s->fnv=0xcbf29ce484222325ULL;
    if (alg < 1 || alg > 4) status[0]=1;
}

extern "C" __global__ void vramdisk_hash_many_update(const HashSegment* segs, const HashFileDesc* files, u8* states_raw, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x;
    if (fi >= nfiles || status[0]!=0) return;
    HashState* s = (HashState*)(states_raw + ((u64)fi * 256ULL));
    HashFileDesc fd = files[fi];
    for (u32 rel=0; rel<fd.seg_count; rel++) {
        const HashSegment sg = segs[fd.seg_start + rel];
        const u8* p = (const u8*)sg.ptr;
        for (u32 i=0; i<sg.len; i++) update_byte(s, sg.kind == 1 ? 0 : p[i]);
    }
}

extern "C" __global__ void vramdisk_hash_many_final(u32 alg, u8* states_raw, u8* outs, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x;
    if (fi >= nfiles || status[0]!=0) return;
    HashState* s = (HashState*)(states_raw + ((u64)fi * 256ULL));
    u8* out = outs + ((u64)fi * 32ULL);
    if (alg==4) { st_be64(out, s->fnv); return; }
    u64 bits = s->len * 8ULL;
    update_byte(s, 0x80);
    while (s->buf_len != 56) update_byte(s, 0);
    if (alg==1) { for (int i=0;i<8;i++) update_byte(s, (u8)(bits >> (8*i))); }
    else { for (int i=0;i<8;i++) update_byte(s, (u8)(bits >> (56 - 8*i))); }
    if (alg==1) { for (int i=0;i<4;i++) st_le32(out+i*4, s->md5[i]); }
    else if (alg==2) { for (int i=0;i<5;i++) st_be32(out+i*4, s->sha1[i]); }
    else if (alg==3) { for (int i=0;i<8;i++) st_be32(out+i*4, s->sha256[i]); }
    else status[0]=1;
}

__device__ __forceinline__ u32 crc32_update_byte(u32 crc, u8 b) {
    crc ^= (u32)b;
    for (int k=0; k<8; k++) {
        u32 mask = 0u - (crc & 1u);
        crc = (crc >> 1) ^ (0xedb88320u & mask);
    }
    return crc;
}

extern "C" __global__ void vramdisk_crc32_many(const HashSegment* segs, const HashFileDesc* files, u32* outs, u32 nfiles, u32* status) {
    u32 fi = blockIdx.x;
    if (fi >= nfiles || status[0]!=0) return;
    HashFileDesc fd = files[fi];
    u32 crc = 0xffffffffu;
    for (u32 rel=0; rel<fd.seg_count; rel++) {
        const HashSegment sg = segs[fd.seg_start + rel];
        const u8* p = (const u8*)sg.ptr;
        for (u32 i=0; i<sg.len; i++) crc = crc32_update_byte(crc, sg.kind == 1 ? 0 : p[i]);
    }
    outs[fi] = crc ^ 0xffffffffu;
}
"#;
