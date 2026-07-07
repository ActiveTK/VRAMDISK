//! WinFsp glue: exposes the [`StorageEngine`] as a Windows volume.
//!
//! WinFsp dispatches callbacks from a pool of kernel-managed threads. We use
//! `FineGuard` so independent callbacks may enter concurrently; shared engine
//! state remains protected by `Mutex`, which also serialises the single CUDA
//! stream until the engine grows finer region locks.

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use anyhow::Context as _;
use windows::Win32::Foundation::{
    LocalFree, HANDLE, HLOCAL, STATUS_ACCESS_DENIED, STATUS_DIRECTORY_NOT_EMPTY, STATUS_DISK_FULL,
    STATUS_END_OF_FILE, STATUS_FILE_IS_A_DIRECTORY, STATUS_INVALID_DEVICE_REQUEST,
    STATUS_INVALID_PARAMETER, STATUS_INVALID_SECURITY_DESCR, STATUS_NOT_A_DIRECTORY,
    STATUS_OBJECT_NAME_COLLISION, STATUS_OBJECT_NAME_NOT_FOUND,
};
use windows::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows::Win32::Security::{GetSecurityDescriptorLength, PSECURITY_DESCRIPTOR};
use windows::Win32::Storage::FileSystem::{
    GetFinalPathNameByHandleW, FILE_NAME_NORMALIZED, GETFINALPATHNAMEBYHANDLE_FLAGS,
    VOLUME_NAME_DOS,
};
use winfsp::constants::FspCleanupFlags;
use winfsp::filesystem::{
    DirInfo, DirMarker, FileInfo, FileSecurity, FileSystemContext, ModificationDescriptor,
    OpenFileInfo, VolumeInfo, WideNameInfo,
};
use winfsp::host::{FileSystemHost, FineGuard, VolumeParams};
use winfsp::FspError;
use winfsp::FspInit;
use winfsp::U16CStr;

use crate::api_kernel::{digest_hex, HashAlgorithm};
use crate::engine::{EngineError, StorageEngine};
use crate::internal_api::{self, Entry as InternalEntry};
use crate::jobs::{status_json, JobRegistry, JobState, JobSubmitError};
use crate::lookup::{LookupError, Node};
use crate::nvcomp::NvcompFrameCodec;
use crate::{round_up_to_chunk, CHUNK_SIZE};

/// `FILE_DIRECTORY_FILE` create option: the object being created is a directory.
const FILE_DIRECTORY_FILE: u32 = 0x0000_0001;
/// `INVALID_FILE_ATTRIBUTES`: sentinel meaning "do not change" in set_basic_info.
const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
const FILE_ATTRIBUTE_HIDDEN: u32 = 0x02;
const FILE_ATTRIBUTE_SYSTEM: u32 = 0x04;
const FILE_ATTRIBUTE_ARCHIVE: u32 = 0x20;
const INTERNAL_ATTRIBUTES: u32 =
    FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_ARCHIVE;
const DEFAULT_SECURITY_SDDL: &str = "O:BAG:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;WD)";
const FSCTL_DUPLICATE_EXTENTS_TO_FILE: u32 = 0x0009_8344;

/// Per-open-handle state stored by WinFsp (boxed behind a pointer).
pub struct OpenFile {
    /// Current (normalized) path of this handle.
    ///
    /// Wrapped in `Arc` so `VramDiskFs::open_paths` can hold a `Weak` reference
    /// to every live handle's path string. When a directory is renamed, the
    /// rename callback iterates those weak refs and rewrites any path that
    /// starts with the old prefix—including handles to files inside the
    /// renamed subtree that were NOT the direct target of the rename.
    path: Arc<Mutex<String>>,
    is_dir: bool,
    internal: Option<InternalOpen>,
    /// Set by `set_delete`; acted on in `cleanup`.
    delete_pending: AtomicBool,
    dir_buffer: winfsp::filesystem::DirBuffer,
}

pub struct InternalOpen {
    entry: InternalEntry,
    content: Mutex<Vec<u8>>,
    writable: bool,
    submitted: AtomicBool,
}

impl OpenFile {
    fn new(path_arc: Arc<Mutex<String>>, is_dir: bool) -> Self {
        OpenFile {
            path: path_arc,
            is_dir,
            internal: None,
            delete_pending: AtomicBool::new(false),
            dir_buffer: winfsp::filesystem::DirBuffer::new(),
        }
    }

    fn new_internal(path_arc: Arc<Mutex<String>>, entry: InternalEntry, content: Vec<u8>) -> Self {
        OpenFile {
            path: path_arc,
            is_dir: entry.is_dir(),
            internal: Some(InternalOpen {
                entry,
                content: Mutex::new(content),
                writable: false,
                submitted: AtomicBool::new(false),
            }),
            delete_pending: AtomicBool::new(false),
            dir_buffer: winfsp::filesystem::DirBuffer::new(),
        }
    }

    fn new_internal_writable(path_arc: Arc<Mutex<String>>, entry: InternalEntry) -> Self {
        OpenFile {
            path: path_arc,
            is_dir: entry.is_dir(),
            internal: Some(InternalOpen {
                entry,
                content: Mutex::new(Vec::new()),
                writable: true,
                submitted: AtomicBool::new(false),
            }),
            delete_pending: AtomicBool::new(false),
            dir_buffer: winfsp::filesystem::DirBuffer::new(),
        }
    }

    fn path(&self) -> String {
        // Recover from poisoning: a panic in some other callback must not make
        // every later path read panic and tear the whole volume down.
        self.path.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// The WinFsp filesystem context.
pub struct VramDiskFs {
    engine: Mutex<StorageEngine>,
    jobs: Arc<JobRegistry>,
    label: String,
    default_security_descriptor: Mutex<Vec<u8>>,
    /// Weak refs to the path `Arc` of every currently-open handle.
    /// Used by `rename` to propagate directory renames to all descendants.
    /// Dead refs are pruned lazily on each `open`/`create` call.
    open_paths: Mutex<Vec<Weak<Mutex<String>>>>,
}

impl VramDiskFs {
    pub fn new(engine: StorageEngine, label: impl Into<String>) -> Self {
        let default_security_descriptor =
            security_descriptor_from_sddl(DEFAULT_SECURITY_SDDL).unwrap_or_default();
        VramDiskFs {
            engine: Mutex::new(engine),
            jobs: Arc::new(JobRegistry::default()),
            label: label.into(),
            default_security_descriptor: Mutex::new(default_security_descriptor),
            open_paths: Mutex::new(Vec::new()),
        }
    }

    /// Lock the engine, recovering the guard if a previous callback poisoned
    /// the mutex by panicking. Keeping the volume alive (and at worst returning
    /// errors) is far better than aborting the whole mount on one bad call.
    fn engine(&self) -> std::sync::MutexGuard<'_, StorageEngine> {
        self.engine.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Lock `open_paths`, recovering from poisoning (see [`engine`]).
    ///
    /// [`engine`]: VramDiskFs::engine
    fn open_paths(&self) -> std::sync::MutexGuard<'_, Vec<Weak<Mutex<String>>>> {
        self.open_paths.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn default_security_descriptor(&self) -> winfsp::Result<Vec<u8>> {
        let mut cached = self
            .default_security_descriptor
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cached.is_empty() {
            *cached = security_descriptor_from_sddl(DEFAULT_SECURITY_SDDL)?;
        }
        Ok(cached.clone())
    }

    /// Allocate a normalized path `Arc`, register a `Weak` ref in
    /// `open_paths` (pruning dead entries first), and return the `Arc`.
    fn register_path(&self, normalized: String) -> Arc<Mutex<String>> {
        let arc = Arc::new(Mutex::new(normalized));
        let mut slots = self.open_paths();
        slots.retain(|w| w.upgrade().is_some()); // prune closed handles
        slots.push(Arc::downgrade(&arc));
        arc
    }
}

/// Map engine/lookup errors to NTSTATUS.
fn map_engine_err(e: EngineError) -> winfsp::FspError {
    match e {
        EngineError::Lookup(l) => map_lookup_err(l),
        EngineError::NoSpace => STATUS_DISK_FULL.into(),
        EngineError::NotAFile => STATUS_FILE_IS_A_DIRECTORY.into(),
        EngineError::Cuda(_) => STATUS_ACCESS_DENIED.into(),
    }
}

fn map_lookup_err(e: LookupError) -> winfsp::FspError {
    match e {
        LookupError::NotFound => STATUS_OBJECT_NAME_NOT_FOUND.into(),
        LookupError::AlreadyExists => STATUS_OBJECT_NAME_COLLISION.into(),
        LookupError::NotADirectory => STATUS_NOT_A_DIRECTORY.into(),
        LookupError::IsADirectory => STATUS_FILE_IS_A_DIRECTORY.into(),
        LookupError::NotEmpty => STATUS_DIRECTORY_NOT_EMPTY.into(),
        LookupError::InvalidName => STATUS_OBJECT_NAME_NOT_FOUND.into(),
    }
}

fn map_job_submit_err(e: JobSubmitError) -> winfsp::FspError {
    match e {
        JobSubmitError::InvalidId => STATUS_OBJECT_NAME_NOT_FOUND.into(),
        JobSubmitError::AlreadyExists | JobSubmitError::AlreadySubmitted => {
            STATUS_OBJECT_NAME_COLLISION.into()
        }
        JobSubmitError::TooManyJobs | JobSubmitError::DescriptorTooLarge => STATUS_DISK_FULL.into(),
        JobSubmitError::NotFound => STATUS_OBJECT_NAME_NOT_FOUND.into(),
    }
}

/// Populate a WinFsp `FileInfo` from a namespace node.
fn fill_file_info(node: &Node, info: &mut FileInfo) {
    info.file_attributes = if node.is_dir {
        node.attributes | FILE_ATTRIBUTE_DIRECTORY
    } else if node.attributes == 0 {
        FILE_ATTRIBUTE_NORMAL
    } else {
        node.attributes
    };
    info.reparse_tag = 0;
    info.file_size = node.size;
    info.allocation_size = round_up_to_chunk(node.size);
    info.creation_time = node.created;
    info.last_access_time = node.accessed;
    info.last_write_time = node.modified;
    info.change_time = node.changed;
    info.index_number = node.index_number;
    info.hard_links = 0;
    info.ea_size = 0;
}

/// Populate a WinFsp `FileInfo` for an internal virtual API entry.
fn fill_internal_file_info(entry: &InternalEntry, content_len: Option<u64>, info: &mut FileInfo) {
    let is_dir = entry.is_dir();
    let size = content_len.unwrap_or_else(|| internal_api::content_len(entry));
    info.file_attributes = if is_dir {
        INTERNAL_ATTRIBUTES | FILE_ATTRIBUTE_DIRECTORY
    } else {
        INTERNAL_ATTRIBUTES
    };
    info.reparse_tag = 0;
    info.file_size = size;
    info.allocation_size = if is_dir { 0 } else { round_up_to_chunk(size) };
    let now = crate::lookup::now_filetime();
    info.creation_time = now;
    info.last_access_time = now;
    info.last_write_time = now;
    info.change_time = now;
    info.index_number = match entry {
        InternalEntry::RootDir => 0xffff_0001,
        InternalEntry::HelpFile => 0xffff_0002,
        InternalEntry::StatsTextFile => 0xffff_0003,
        InternalEntry::StatsJsonFile => 0xffff_0004,
        InternalEntry::TraceTextFile => 0xffff_0005,
        InternalEntry::TraceJsonFile => 0xffff_0006,
        InternalEntry::ChunksRootDir => 0xffff_0007,
        InternalEntry::ChunksDir { .. } => 0xffff_0008,
        InternalEntry::ChunksJsonFile { .. } => 0xffff_0009,
        InternalEntry::JobsRootDir => 0xffff_000a,
        InternalEntry::JobsPendingDir => 0xffff_000b,
        InternalEntry::JobsCompletedDir => 0xffff_000c,
        InternalEntry::JobDir { .. } => 0xffff_000d,
        InternalEntry::JobSubmitFile { .. } => 0xffff_000e,
        InternalEntry::JobStatusFile { .. } => 0xffff_000f,
        InternalEntry::JobResultFile { .. } => 0xffff_0010,
        InternalEntry::JobWaitFile { .. } => 0xffff_0011,
        InternalEntry::JobCancelFile { .. } => 0xffff_0012,
        InternalEntry::HashRootDir => 0xffff_0013,
        InternalEntry::HashAlgDir { .. } => 0xffff_0014,
        InternalEntry::HashFile { .. } => 0xffff_0015,
    };
    info.hard_links = 0;
    info.ea_size = 0;
}

fn node_security_descriptor(fs: &VramDiskFs, node: &Node) -> winfsp::Result<Vec<u8>> {
    if node.security_descriptor.is_empty() {
        fs.default_security_descriptor()
    } else {
        Ok(node.security_descriptor.clone())
    }
}

fn write_security_descriptor(
    src: &[u8],
    out: Option<&mut [c_void]>,
) -> winfsp::Result<FileSecurityCopy> {
    if let Some(out) = out {
        let n = out.len().min(src.len());
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr().cast::<u8>(), n);
        }
    }
    Ok(FileSecurityCopy {
        size: src.len() as u64,
    })
}

fn job_content(entry: &InternalEntry, jobs: &JobRegistry, wait: bool) -> winfsp::Result<Vec<u8>> {
    match entry {
        InternalEntry::JobStatusFile { id } => {
            let snap = jobs.snapshot(id).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
            Ok(status_json(&snap).into_bytes())
        }
        InternalEntry::JobResultFile { id } => {
            let snap = jobs.snapshot(id).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
            Ok(snap.result.into_bytes())
        }
        InternalEntry::JobWaitFile { id } if wait => {
            let snap = jobs.wait(id).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
            Ok(snap.result.into_bytes())
        }
        InternalEntry::JobWaitFile { id } => {
            let snap = jobs.snapshot(id).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
            if snap.state.is_terminal() {
                Ok(snap.result.into_bytes())
            } else {
                Ok(status_json(&snap).into_bytes())
            }
        }
        InternalEntry::JobCancelFile { id } => {
            let ok = jobs.cancel(id);
            if !ok {
                return Err(STATUS_OBJECT_NAME_NOT_FOUND.into());
            }
            let snap = jobs.snapshot(id).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
            Ok(snap.result.into_bytes())
        }
        InternalEntry::JobSubmitFile { id } => {
            let snap = jobs.snapshot(id).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
            Ok(snap.descriptor.into_bytes())
        }
        _ => Err(STATUS_FILE_IS_A_DIRECTORY.into()),
    }
}

fn submit_internal_if_needed(context: &OpenFile, jobs: &JobRegistry) -> Option<String> {
    let Some(internal) = &context.internal else {
        return None;
    };
    if !internal.writable {
        return None;
    }
    let InternalEntry::JobSubmitFile { id } = &internal.entry else {
        return None;
    };
    if internal.submitted.swap(true, Ordering::SeqCst) {
        return None;
    }
    let data = internal
        .content
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    jobs.complete_submission(id, &data).ok()?;
    Some(id.clone())
}

fn execute_job(id: &str, jobs: &JobRegistry, engine: &mut StorageEngine) {
    let Some(descriptor) = jobs.start(id) else {
        return;
    };
    match execute_job_descriptor(id, &descriptor, engine) {
        Ok(result) => jobs.succeed(id, result),
        Err(e) => jobs.fail(id, e),
    }
}

fn execute_job_descriptor(
    id: &str,
    descriptor: &str,
    engine: &mut StorageEngine,
) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(descriptor)
        .map_err(|e| format!("invalid job descriptor JSON: {e}"))?;
    let op = v
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "job descriptor must contain string field \"op\"".to_string())?;
    match op {
        "noop" => Ok(serde_json::json!({
            "id": id,
            "state": JobState::Succeeded.as_str(),
            "ok": true,
            "op": "noop",
            "error": null,
        })
        .to_string()
            + "\r\n"),
        "hash" | "hash.calculate" => execute_hash_job(id, &v, engine),
        "archive.compress" | "compress.archive" => execute_archive_compress_job(id, &v, engine),
        "archive.extract" | "extract.archive" => execute_archive_extract_job(id, &v, engine),
        _ => Err("no GPU executor registered for requested operation".to_string()),
    }
}

fn execute_archive_compress_job(
    id: &str,
    descriptor: &serde_json::Value,
    engine: &mut StorageEngine,
) -> Result<String, String> {
    let format_name = descriptor
        .get("format")
        .or_else(|| descriptor.get("codec"))
        .and_then(|v| v.as_str())
        .unwrap_or("tar.zst");
    let format = NvcompFrameCodec::parse(format_name)
        .ok_or_else(|| format!("unsupported archive format: {format_name}"))?;
    let output = descriptor
        .get("output")
        .or_else(|| descriptor.get("destination"))
        .or_else(|| descriptor.get("dest"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "archive.compress job requires output".to_string())?;
    let recursive = descriptor
        .get("recursive")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let roots = descriptor_paths(descriptor, "path", "paths")?;
    let mut targets = BTreeSet::new();
    for root in roots {
        collect_hash_targets(engine, &root, recursive, &mut targets)?;
    }
    let paths: Vec<String> = targets.into_iter().collect();
    if paths.is_empty() {
        return Err("archive.compress resolved no files".to_string());
    }
    let stats = engine
        .archive_compress_gpu(format, &paths, output)
        .map_err(|e| format!("GPU archive compression failed: {e:?}"))?;
    let throughput = if stats.elapsed_ms == 0 {
        serde_json::Value::Null
    } else {
        serde_json::json!(
            (stats.input_bytes as f64 / 1048576.0) / (stats.elapsed_ms as f64 / 1000.0)
        )
    };
    Ok(serde_json::json!({
        "id": id,
        "state": JobState::Succeeded.as_str(),
        "ok": true,
        "op": "archive.compress",
        "format": stats.format,
        "output": stats.output,
        "file_count": stats.file_count,
        "input_bytes": stats.input_bytes,
        "archive_bytes": stats.archive_bytes,
        "elapsed_ms": stats.elapsed_ms,
        "throughput_mib_s": throughput,
        "error": null,
    })
    .to_string()
        + "\r\n")
}

fn execute_archive_extract_job(
    id: &str,
    descriptor: &serde_json::Value,
    engine: &mut StorageEngine,
) -> Result<String, String> {
    let format_name = descriptor
        .get("format")
        .or_else(|| descriptor.get("codec"))
        .and_then(|v| v.as_str())
        .unwrap_or("tar.zst");
    let format = NvcompFrameCodec::parse(format_name)
        .ok_or_else(|| format!("unsupported archive format: {format_name}"))?;
    let archive = descriptor
        .get("archive")
        .or_else(|| descriptor.get("input"))
        .or_else(|| descriptor.get("path"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "archive.extract job requires archive/input/path".to_string())?;
    let output_dir = descriptor
        .get("output_dir")
        .or_else(|| descriptor.get("destination"))
        .or_else(|| descriptor.get("dest"))
        .and_then(|v| v.as_str())
        .unwrap_or("\\");
    let stats = engine
        .archive_extract_gpu(format, archive, output_dir)
        .map_err(|e| format!("GPU archive extraction failed: {e:?}"))?;
    let throughput = if stats.elapsed_ms == 0 {
        serde_json::Value::Null
    } else {
        serde_json::json!(
            (stats.output_bytes as f64 / 1048576.0) / (stats.elapsed_ms as f64 / 1000.0)
        )
    };
    Ok(serde_json::json!({
        "id": id,
        "state": JobState::Succeeded.as_str(),
        "ok": true,
        "op": "archive.extract",
        "format": stats.format,
        "archive": stats.archive,
        "output_dir": stats.output_dir,
        "file_count": stats.file_count,
        "archive_bytes": stats.archive_bytes,
        "output_bytes": stats.output_bytes,
        "elapsed_ms": stats.elapsed_ms,
        "throughput_mib_s": throughput,
        "error": null,
    })
    .to_string()
        + "\r\n")
}

fn descriptor_paths(
    descriptor: &serde_json::Value,
    single_key: &str,
    list_key: &str,
) -> Result<Vec<String>, String> {
    let mut roots = Vec::new();
    if let Some(path) = descriptor.get(single_key).and_then(|v| v.as_str()) {
        roots.push(path.to_string());
    }
    if let Some(paths) = descriptor.get(list_key).and_then(|v| v.as_array()) {
        for p in paths {
            let Some(path) = p.as_str() else {
                return Err(format!("{list_key} must contain only strings"));
            };
            roots.push(path.to_string());
        }
    }
    if roots.is_empty() {
        return Err(format!("job requires {single_key} or {list_key}"));
    }
    Ok(roots)
}

fn execute_hash_job(
    id: &str,
    descriptor: &serde_json::Value,
    engine: &mut StorageEngine,
) -> Result<String, String> {
    let alg_name = descriptor
        .get("algorithm")
        .or_else(|| descriptor.get("alg"))
        .and_then(|v| v.as_str())
        .unwrap_or("sha256");
    let alg = HashAlgorithm::parse(alg_name)
        .ok_or_else(|| format!("unsupported hash algorithm: {alg_name}"))?;
    let recursive = descriptor
        .get("recursive")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let roots = descriptor_paths(descriptor, "path", "paths")?;

    let mut targets = BTreeSet::new();
    for root in roots {
        collect_hash_targets(engine, &root, recursive, &mut targets)?;
    }

    let paths: Vec<String> = targets.into_iter().collect();
    let digests = engine
        .hash_files_gpu_many(&paths, alg)
        .map_err(|e| format!("batched GPU hash failed: {e:?}"))?;
    let mut files = Vec::with_capacity(paths.len());
    for (path, digest) in paths.iter().zip(digests.iter()) {
        files.push(serde_json::json!({
            "path": path,
            "digest": digest_hex(digest),
        }));
    }

    Ok(serde_json::json!({
        "id": id,
        "state": JobState::Succeeded.as_str(),
        "ok": true,
        "op": "hash",
        "algorithm": alg.name(),
        "file_count": paths.len(),
        "files": files,
        "error": null,
    })
    .to_string()
        + "\r\n")
}

fn collect_hash_targets(
    engine: &StorageEngine,
    path: &str,
    recursive: bool,
    out: &mut BTreeSet<String>,
) -> Result<(), String> {
    let path = crate::lookup::normalize(path);
    let node = engine
        .get(&path)
        .ok_or_else(|| format!("path not found: {path}"))?;
    if !node.is_dir {
        out.insert(path);
        return Ok(());
    }
    if !recursive {
        return Err(format!("path is a directory and recursive=false: {path}"));
    }
    let entries = engine
        .table()
        .readdir(&path)
        .map_err(|_| format!("cannot read directory: {path}"))?;
    let base = if path == "\\" { String::new() } else { path };
    let children: Vec<(String, bool)> = entries
        .into_iter()
        .map(|(name, child)| {
            (
                format!("{base}\\{}", name.to_ascii_lowercase()),
                child.is_dir,
            )
        })
        .collect();
    for (child_path, is_dir) in children {
        if is_dir {
            collect_hash_targets(engine, &child_path, recursive, out)?;
        } else {
            out.insert(child_path);
        }
    }
    Ok(())
}

struct FileSecurityCopy {
    size: u64,
}

fn security_descriptor_from_sddl(sddl: &str) -> winfsp::Result<Vec<u8>> {
    use widestring::U16CString;
    use windows::core::PCWSTR;

    let wide = U16CString::from_str(sddl).map_err(|_| STATUS_INVALID_SECURITY_DESCR)?;
    let mut sd = PSECURITY_DESCRIPTOR::default();
    let ok = unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            PCWSTR(wide.as_ptr()),
            SDDL_REVISION_1,
            &mut sd,
            None,
        )
    };
    ok.map_err(|_| STATUS_INVALID_SECURITY_DESCR)?;
    if sd.is_invalid() {
        return Err(STATUS_INVALID_SECURITY_DESCR.into());
    }

    let len = unsafe { GetSecurityDescriptorLength(sd) };
    let bytes = unsafe { std::slice::from_raw_parts(sd.0.cast::<u8>(), len as usize) }.to_vec();
    unsafe {
        let _ = LocalFree(Some(HLOCAL(sd.0.cast())));
    }
    Ok(bytes)
}

fn security_descriptor_from_void_slice(security_descriptor: &[c_void]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            security_descriptor.as_ptr().cast::<u8>(),
            security_descriptor.len(),
        )
    }
    .to_vec()
}

fn apply_security_descriptor_modification(
    current: &[u8],
    security_information: u32,
    modification_descriptor: ModificationDescriptor,
) -> winfsp::Result<Vec<u8>> {
    let mut out = std::ptr::null_mut();
    let status = unsafe {
        winfsp_sys::FspSetSecurityDescriptor(
            current.as_ptr() as *mut c_void,
            security_information,
            modification_descriptor.as_mut_ptr(),
            &mut out,
        )
    };
    if status != 0 {
        return Err(FspError::NTSTATUS(status));
    }
    if out.is_null() {
        return Err(STATUS_INVALID_SECURITY_DESCR.into());
    }

    let sd = PSECURITY_DESCRIPTOR(out);
    let len = unsafe { GetSecurityDescriptorLength(sd) };
    let bytes = unsafe { std::slice::from_raw_parts(out.cast::<u8>(), len as usize) }.to_vec();
    unsafe {
        let create_func: unsafe extern "C" fn() -> i32 =
            std::mem::transmute(winfsp_sys::FspSetSecurityDescriptor as *const ());
        winfsp_sys::FspDeleteSecurityDescriptor(out, Some(create_func));
    }
    Ok(bytes)
}

impl FileSystemContext for VramDiskFs {
    type FileContext = OpenFile;

    fn get_security_by_name(
        &self,
        file_name: &U16CStr,
        security_descriptor: Option<&mut [c_void]>,
        _reparse_point_resolver: impl FnOnce(&U16CStr) -> Option<FileSecurity>,
    ) -> winfsp::Result<FileSecurity> {
        let path = crate::lookup::normalize(&file_name.to_string_lossy());
        let engine = self.engine();
        if let Some(entry) = internal_api::resolve(&path, &engine) {
            let sd = self.default_security_descriptor()?;
            let copied = write_security_descriptor(&sd, security_descriptor)?;
            return Ok(FileSecurity {
                reparse: false,
                sz_security_descriptor: copied.size,
                attributes: if entry.is_dir() {
                    INTERNAL_ATTRIBUTES | FILE_ATTRIBUTE_DIRECTORY
                } else {
                    INTERNAL_ATTRIBUTES
                },
            });
        }
        let node = engine.get(&path).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
        let sd = node_security_descriptor(self, node)?;
        let copied = write_security_descriptor(&sd, security_descriptor)?;
        Ok(FileSecurity {
            reparse: false,
            sz_security_descriptor: copied.size,
            attributes: if node.is_dir {
                node.attributes | FILE_ATTRIBUTE_DIRECTORY
            } else if node.attributes == 0 {
                FILE_ATTRIBUTE_NORMAL
            } else {
                node.attributes
            },
        })
    }

    fn open(
        &self,
        file_name: &U16CStr,
        _create_options: u32,
        _granted_access: u32,
        file_info: &mut OpenFileInfo,
    ) -> winfsp::Result<Self::FileContext> {
        let path = crate::lookup::normalize(&file_name.to_string_lossy());
        let mut engine = self.engine();
        if let Some(entry) = internal_api::resolve(&path, &engine) {
            if let InternalEntry::JobDir { id } = &entry {
                if !self.jobs.exists(id) {
                    return Err(STATUS_OBJECT_NAME_NOT_FOUND.into());
                }
            }
            let content = if entry.is_dir() {
                Vec::new()
            } else if matches!(
                entry,
                InternalEntry::JobStatusFile { .. }
                    | InternalEntry::JobResultFile { .. }
                    | InternalEntry::JobWaitFile { .. }
                    | InternalEntry::JobCancelFile { .. }
                    | InternalEntry::JobSubmitFile { .. }
            ) {
                drop(engine);
                let wait = matches!(entry, InternalEntry::JobWaitFile { .. });
                let content = job_content(&entry, &self.jobs, wait)?;
                fill_internal_file_info(&entry, Some(content.len() as u64), file_info.as_mut());
                return Ok(OpenFile::new_internal(
                    self.register_path(path),
                    entry,
                    content,
                ));
            } else {
                internal_api::content(&entry, &mut engine).map_err(map_engine_err)?
            };
            fill_internal_file_info(&entry, Some(content.len() as u64), file_info.as_mut());
            drop(engine);
            return Ok(OpenFile::new_internal(
                self.register_path(path),
                entry,
                content,
            ));
        }
        let node = engine.get(&path).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
        let is_dir = node.is_dir;
        fill_file_info(node, file_info.as_mut());
        drop(engine);
        Ok(OpenFile::new(self.register_path(path), is_dir))
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        &self,
        file_name: &U16CStr,
        create_options: u32,
        _granted_access: u32,
        file_attributes: u32,
        _security_descriptor: Option<&[c_void]>,
        _allocation_size: u64,
        _extra_buffer: Option<&[u8]>,
        _extra_buffer_is_reparse_point: bool,
        file_info: &mut OpenFileInfo,
    ) -> winfsp::Result<Self::FileContext> {
        let raw_path = file_name.to_string_lossy();
        let path = crate::lookup::normalize(&raw_path);
        if let Some(id) = internal_api::job_submit_id(&path) {
            self.jobs.reserve(&id).map_err(map_job_submit_err)?;
            let entry = InternalEntry::JobSubmitFile { id };
            fill_internal_file_info(&entry, Some(0), file_info.as_mut());
            return Ok(OpenFile::new_internal_writable(
                self.register_path(path),
                entry,
            ));
        }
        if internal_api::is_internal_path(&path) {
            return Err(STATUS_ACCESS_DENIED.into());
        }
        let is_dir = create_options & FILE_DIRECTORY_FILE != 0;
        let mut engine = self.engine();
        let node = if is_dir {
            engine.table_mut().create_dir(&raw_path, file_attributes)
        } else {
            engine.table_mut().create_file(&raw_path, file_attributes)
        }
        .map_err(map_lookup_err)?;
        node.security_descriptor = match _security_descriptor {
            Some(sd) => security_descriptor_from_void_slice(sd),
            None => self.default_security_descriptor()?,
        };
        fill_file_info(node, file_info.as_mut());
        drop(engine);
        Ok(OpenFile::new(self.register_path(path), is_dir))
    }

    fn close(&self, context: Self::FileContext) {
        if let Some(id) = submit_internal_if_needed(&context, &self.jobs) {
            let mut engine = self.engine();
            execute_job(&id, &self.jobs, &mut engine);
        }
    }

    fn get_file_info(
        &self,
        context: &Self::FileContext,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        if let Some(internal) = &context.internal {
            let len = internal
                .content
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len() as u64;
            fill_internal_file_info(&internal.entry, Some(len), file_info);
            return Ok(());
        }
        let engine = self.engine();
        let node = engine
            .get(&context.path())
            .ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
        fill_file_info(node, file_info);
        Ok(())
    }

    fn get_security(
        &self,
        context: &Self::FileContext,
        security_descriptor: Option<&mut [c_void]>,
    ) -> winfsp::Result<u64> {
        if context.internal.is_some() {
            let sd = self.default_security_descriptor()?;
            return Ok(write_security_descriptor(&sd, security_descriptor)?.size);
        }
        let engine = self.engine();
        let path = context.path();
        let node = engine.get(&path).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
        let sd = node_security_descriptor(self, node)?;
        Ok(write_security_descriptor(&sd, security_descriptor)?.size)
    }

    fn set_security(
        &self,
        context: &Self::FileContext,
        security_information: u32,
        modification_descriptor: ModificationDescriptor,
    ) -> winfsp::Result<()> {
        if context.internal.is_some() {
            return Err(STATUS_ACCESS_DENIED.into());
        }
        let mut engine = self.engine();
        let path = context.path();
        let current = {
            let node = engine.get(&path).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
            node_security_descriptor(self, node)?
        };
        let updated = apply_security_descriptor_modification(
            &current,
            security_information,
            modification_descriptor,
        )?;
        let node = engine
            .table_mut()
            .get_mut(&path)
            .ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
        node.security_descriptor = updated;
        node.changed = crate::lookup::now_filetime();
        Ok(())
    }

    fn read(
        &self,
        context: &Self::FileContext,
        buffer: &mut [u8],
        offset: u64,
    ) -> winfsp::Result<u32> {
        if let Some(internal) = &context.internal {
            if context.is_dir {
                return Err(STATUS_FILE_IS_A_DIRECTORY.into());
            }
            let data = internal.content.lock().unwrap_or_else(|e| e.into_inner());
            let off = offset as usize;
            if off >= data.len() {
                return Err(STATUS_END_OF_FILE.into());
            }
            let n = buffer.len().min(data.len() - off);
            buffer[..n].copy_from_slice(&data[off..off + n]);
            return Ok(n as u32);
        }
        let mut engine = self.engine();
        let path = context.path();
        let size = engine.get(&path).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?.size;
        if offset >= size {
            return Err(STATUS_END_OF_FILE.into());
        }
        // Read straight into WinFsp's output buffer: the device-to-host copy
        // lands in the final destination with no intermediate Vec allocation
        // and no second memcpy back into `buffer`.
        let n = engine
            .read_into(&path, offset, buffer)
            .map_err(map_engine_err)?;
        Ok(n as u32)
    }

    fn write(
        &self,
        context: &Self::FileContext,
        buffer: &[u8],
        offset: u64,
        write_to_eof: bool,
        constrained_io: bool,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<u32> {
        if let Some(internal) = &context.internal {
            if !internal.writable {
                return Err(STATUS_ACCESS_DENIED.into());
            }
            let mut data = internal.content.lock().unwrap_or_else(|e| e.into_inner());
            let eff_offset = if write_to_eof {
                data.len() as u64
            } else {
                offset
            };
            let end = eff_offset
                .checked_add(buffer.len() as u64)
                .ok_or(STATUS_DISK_FULL)?;
            if end as usize > 1024 * 1024 {
                return Err(STATUS_DISK_FULL.into());
            }
            let off = eff_offset as usize;
            if off > data.len() {
                data.resize(off, 0);
            }
            if end as usize > data.len() {
                data.resize(end as usize, 0);
            }
            data[off..off + buffer.len()].copy_from_slice(buffer);
            fill_internal_file_info(&internal.entry, Some(data.len() as u64), file_info);
            return Ok(buffer.len() as u32);
        }
        let mut engine = self.engine();
        let path = context.path();
        let size = engine.get(&path).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?.size;

        let eff_offset = if write_to_eof { size } else { offset };
        let mut data = buffer;
        if constrained_io {
            if eff_offset >= size {
                return Ok(0);
            }
            let max = (size - eff_offset) as usize;
            if data.len() > max {
                data = &data[..max];
            }
        }
        let n = engine
            .write(&path, eff_offset, data)
            .map_err(map_engine_err)?;
        let node = engine.get(&path).unwrap();
        fill_file_info(node, file_info);
        Ok(n as u32)
    }

    fn control(
        &self,
        context: &Self::FileContext,
        control_code: u32,
        input: &[u8],
        _output: &mut [u8],
    ) -> winfsp::Result<u32> {
        if control_code != FSCTL_DUPLICATE_EXTENTS_TO_FILE {
            return Err(STATUS_INVALID_DEVICE_REQUEST.into());
        }
        if context.internal.is_some() || context.is_dir {
            return Err(STATUS_ACCESS_DENIED.into());
        }
        let Some(req) = parse_duplicate_extents(input) else {
            return Err(STATUS_INVALID_PARAMETER.into());
        };
        let Some(src_path) = path_from_handle(req.source_handle) else {
            return Err(STATUS_INVALID_DEVICE_REQUEST.into());
        };
        let dst_path = context.path();
        let mut engine = self.engine();
        engine
            .clone_range(
                &src_path,
                &dst_path,
                req.source_offset,
                req.target_offset,
                req.byte_count,
            )
            .map_err(map_engine_err)?;
        Ok(0)
    }

    fn overwrite(
        &self,
        context: &Self::FileContext,
        file_attributes: u32,
        replace_file_attributes: bool,
        _allocation_size: u64,
        _extra_buffer: Option<&[u8]>,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        if context.internal.is_some() {
            return Err(STATUS_ACCESS_DENIED.into());
        }
        let mut engine = self.engine();
        let path = context.path();
        engine.set_size(&path, 0).map_err(map_engine_err)?;
        if let Some(node) = engine.table_mut().get_mut(&path) {
            if replace_file_attributes {
                node.attributes = file_attributes;
            } else {
                node.attributes |= file_attributes;
            }
            fill_file_info(node, file_info);
        }
        Ok(())
    }

    fn set_file_size(
        &self,
        context: &Self::FileContext,
        new_size: u64,
        set_allocation_size: bool,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        if context.internal.is_some() {
            return Err(STATUS_ACCESS_DENIED.into());
        }
        let mut engine = self.engine();
        let path = context.path();
        let size = engine.get(&path).ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?.size;
        // An allocation-size request only shrinks the file if it would no
        // longer fit; a file-size request always sets the logical size.
        if !set_allocation_size || new_size < size {
            engine.set_size(&path, new_size).map_err(map_engine_err)?;
        }
        let node = engine.get(&path).unwrap();
        fill_file_info(node, file_info);
        Ok(())
    }

    fn rename(
        &self,
        _context: &Self::FileContext,
        file_name: &U16CStr,
        new_file_name: &U16CStr,
        replace_if_exists: bool,
    ) -> winfsp::Result<()> {
        let from = file_name.to_string_lossy();
        let to = new_file_name.to_string_lossy();
        let from_norm = crate::lookup::normalize(&from);
        let to_norm = crate::lookup::normalize(&to);
        if internal_api::is_internal_path(&from_norm) || internal_api::is_internal_path(&to_norm) {
            return Err(STATUS_ACCESS_DENIED.into());
        }
        {
            let mut engine = self.engine();
            engine
                .table_mut()
                .rename(&from, &to, replace_if_exists)
                .map_err(map_lookup_err)?;
        } // release engine lock before touching open_paths

        // Rewrite the stored path of every open handle whose path is exactly
        // `old_prefix` (the renamed entry itself) or starts with
        // `old_prefix\` (a descendant inside a renamed directory).
        // This fixes the bug where open handles to files inside a renamed
        // directory would keep the stale pre-rename path.
        let old_prefix = crate::lookup::normalize(&from);
        let new_prefix = crate::lookup::normalize(&to);
        let old_with_sep = format!("{old_prefix}\\");

        let mut slots = self.open_paths();
        slots.retain(|weak| {
            let Some(arc) = weak.upgrade() else {
                return false; // prune closed handles
            };
            let mut p = arc.lock().unwrap_or_else(|e| e.into_inner());
            if *p == old_prefix {
                *p = new_prefix.clone();
            } else if let Some(suffix) = p.strip_prefix(old_with_sep.as_str()) {
                *p = format!("{new_prefix}\\{suffix}");
            }
            true
        });
        Ok(())
    }

    fn set_basic_info(
        &self,
        context: &Self::FileContext,
        file_attributes: u32,
        creation_time: u64,
        last_access_time: u64,
        last_write_time: u64,
        last_change_time: u64,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        if context.internal.is_some() {
            return Err(STATUS_ACCESS_DENIED.into());
        }
        let mut engine = self.engine();
        let path = context.path();
        let node = engine
            .table_mut()
            .get_mut(&path)
            .ok_or(STATUS_OBJECT_NAME_NOT_FOUND)?;
        if file_attributes != INVALID_FILE_ATTRIBUTES {
            node.attributes = file_attributes;
        }
        if creation_time != 0 {
            node.created = creation_time;
        }
        if last_access_time != 0 {
            node.accessed = last_access_time;
        }
        if last_write_time != 0 {
            node.modified = last_write_time;
        }
        if last_change_time != 0 {
            node.changed = last_change_time;
        }
        fill_file_info(node, file_info);
        Ok(())
    }

    fn set_delete(
        &self,
        context: &Self::FileContext,
        _file_name: &U16CStr,
        delete_file: bool,
    ) -> winfsp::Result<()> {
        if context.internal.is_some() {
            if delete_file {
                return Err(STATUS_ACCESS_DENIED.into());
            }
            return Ok(());
        }
        if delete_file && context.is_dir {
            let engine = self.engine();
            if let Some(node) = engine.get(&context.path()) {
                if !node.children.is_empty() {
                    return Err(STATUS_DIRECTORY_NOT_EMPTY.into());
                }
            }
        }
        context.delete_pending.store(delete_file, Ordering::SeqCst);
        Ok(())
    }

    fn cleanup(&self, context: &Self::FileContext, _file_name: Option<&U16CStr>, flags: u32) {
        if context.internal.is_some() {
            if let Some(id) = submit_internal_if_needed(context, &self.jobs) {
                let mut engine = self.engine();
                execute_job(&id, &self.jobs, &mut engine);
            }
            return;
        }
        let delete_requested = FspCleanupFlags::FspCleanupDelete.is_flagged(flags)
            || context.delete_pending.load(Ordering::SeqCst);
        if delete_requested {
            let mut engine = self.engine();
            let _ = engine.remove(&context.path());
        }
    }

    fn flush(
        &self,
        _context: Option<&Self::FileContext>,
        _file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        // All data lives in VRAM synchronously; nothing to flush.
        Ok(())
    }

    fn get_volume_info(&self, out: &mut VolumeInfo) -> winfsp::Result<()> {
        let engine = self.engine();
        let stats = engine.stats();
        out.total_size = stats.total_bytes;
        out.free_size = stats.free_physical_bytes;
        out.set_volume_label(&self.label);
        Ok(())
    }

    fn read_directory(
        &self,
        context: &Self::FileContext,
        pattern: Option<&U16CStr>,
        marker: DirMarker,
        buffer: &mut [u8],
    ) -> winfsp::Result<u32> {
        let path = context.path();
        let pattern = pattern.map(|p| p.to_string_lossy());
        if let Ok(lock) = context.dir_buffer.acquire(marker.is_none(), None) {
            let mut info = DirInfo::<255>::new();

            if let Some(internal) = &context.internal {
                if !context.is_dir {
                    return Err(STATUS_NOT_A_DIRECTORY.into());
                }

                let mut emit_internal = |name: &str, entry: InternalEntry| -> winfsp::Result<()> {
                    if !dir_pattern_matches(pattern.as_deref(), name) {
                        return Ok(());
                    }
                    info.reset();
                    let wide: Vec<u16> = name.encode_utf16().collect();
                    info.set_name_raw(wide.as_slice())?;
                    fill_internal_file_info(&entry, None, info.file_info_mut());
                    lock.write(&mut info)
                };

                if let InternalEntry::HashAlgDir { alg, target_dir } = &internal.entry {
                    emit_internal(".", internal.entry.clone())?;
                    emit_internal("..", internal_api::hash_target_parent(*alg, target_dir))?;
                }
                if let InternalEntry::ChunksDir { target_dir } = &internal.entry {
                    emit_internal(".", internal.entry.clone())?;
                    emit_internal("..", internal_api::chunks_target_parent(target_dir))?;
                }
                if matches!(
                    internal.entry,
                    InternalEntry::JobsPendingDir | InternalEntry::JobsCompletedDir
                ) {
                    emit_internal(".", internal.entry.clone())?;
                    emit_internal("..", InternalEntry::JobsRootDir)?;
                }
                if let InternalEntry::JobDir { .. } = &internal.entry {
                    emit_internal(".", internal.entry.clone())?;
                    emit_internal("..", InternalEntry::JobsRootDir)?;
                }

                match &internal.entry {
                    InternalEntry::RootDir => {
                        emit_internal("help.txt", InternalEntry::HelpFile)?;
                        emit_internal("stats.txt", InternalEntry::StatsTextFile)?;
                        emit_internal("stats.json", InternalEntry::StatsJsonFile)?;
                        emit_internal("trace.txt", InternalEntry::TraceTextFile)?;
                        emit_internal("trace.json", InternalEntry::TraceJsonFile)?;
                        emit_internal("chunks.json", InternalEntry::ChunksRootDir)?;
                        emit_internal("jobs", InternalEntry::JobsRootDir)?;
                    }
                    InternalEntry::JobsRootDir => {
                        emit_internal("pending", InternalEntry::JobsPendingDir)?;
                        emit_internal("completed", InternalEntry::JobsCompletedDir)?;
                    }
                    InternalEntry::JobsPendingDir => {
                        for id in self.jobs.receiving_ids() {
                            emit_internal(
                                &format!("{id}.json"),
                                InternalEntry::JobSubmitFile { id },
                            )?;
                        }
                    }
                    InternalEntry::JobsCompletedDir => {
                        for id in self.jobs.completed_ids() {
                            emit_internal(&id, InternalEntry::JobDir { id: id.clone() })?;
                        }
                    }
                    InternalEntry::JobDir { id } => {
                        emit_internal(
                            "status.json",
                            InternalEntry::JobStatusFile { id: id.clone() },
                        )?;
                        emit_internal(
                            "result.json",
                            InternalEntry::JobResultFile { id: id.clone() },
                        )?;
                        emit_internal("wait", InternalEntry::JobWaitFile { id: id.clone() })?;
                        emit_internal("cancel", InternalEntry::JobCancelFile { id: id.clone() })?;
                    }
                    InternalEntry::ChunksRootDir => {
                        let engine = self.engine();
                        let entries = engine.table().readdir("\\").map_err(map_lookup_err)?;
                        for (name, child) in entries {
                            if name.eq_ignore_ascii_case(internal_api::DISPLAY_ROOT) {
                                continue;
                            }
                            let target_path = format!("\\{}", name.to_ascii_lowercase());
                            emit_internal(
                                name,
                                internal_api::chunks_child_entry(child, target_path),
                            )?;
                        }
                    }
                    InternalEntry::ChunksDir { target_dir } => {
                        let engine = self.engine();
                        let base = if target_dir == "\\" {
                            String::new()
                        } else {
                            target_dir.clone()
                        };
                        let entries = engine.table().readdir(target_dir).map_err(map_lookup_err)?;
                        for (name, child) in entries {
                            let target_path = format!("{base}\\{}", name.to_ascii_lowercase());
                            emit_internal(
                                name,
                                internal_api::chunks_child_entry(child, target_path),
                            )?;
                        }
                    }
                    InternalEntry::HashRootDir => {
                        for alg in internal_api::supported_algorithms() {
                            emit_internal(
                                alg.name(),
                                InternalEntry::HashAlgDir {
                                    alg,
                                    target_dir: "\\".to_string(),
                                },
                            )?;
                        }
                    }
                    InternalEntry::HashAlgDir { alg, target_dir } => {
                        let engine = self.engine();
                        let base = if target_dir == "\\" {
                            String::new()
                        } else {
                            target_dir.clone()
                        };
                        let entries = engine.table().readdir(target_dir).map_err(map_lookup_err)?;
                        for (name, child) in entries {
                            if target_dir == "\\"
                                && name.eq_ignore_ascii_case(internal_api::DISPLAY_ROOT)
                            {
                                continue;
                            }
                            let target_path = format!("{base}\\{}", name.to_ascii_lowercase());
                            emit_internal(
                                name,
                                internal_api::hash_child_entry(child, *alg, target_path),
                            )?;
                        }
                    }
                    InternalEntry::HelpFile
                    | InternalEntry::StatsTextFile
                    | InternalEntry::StatsJsonFile
                    | InternalEntry::TraceTextFile
                    | InternalEntry::TraceJsonFile
                    | InternalEntry::ChunksJsonFile { .. }
                    | InternalEntry::JobSubmitFile { .. }
                    | InternalEntry::JobStatusFile { .. }
                    | InternalEntry::JobResultFile { .. }
                    | InternalEntry::JobWaitFile { .. }
                    | InternalEntry::JobCancelFile { .. }
                    | InternalEntry::HashFile { .. } => {}
                }
            } else {
                let engine = self.engine();
                // Set the entry name as raw UTF-16 *without* a NUL terminator;
                // WinFsp derives the name length from the entry size.
                let mut emit = |name: &str, node: &Node| -> winfsp::Result<()> {
                    if !dir_pattern_matches(pattern.as_deref(), name) {
                        return Ok(());
                    }
                    info.reset();
                    let wide: Vec<u16> = name.encode_utf16().collect();
                    info.set_name_raw(wide.as_slice())?;
                    fill_file_info(node, info.file_info_mut());
                    lock.write(&mut info)
                };

                // "." and ".." for non-root directories.
                if path != "\\" && !path.is_empty() {
                    if let Some(node) = engine.get(&path) {
                        emit(".", node)?;
                    }
                    let parent = parent_path(&path);
                    if let Some(node) = engine.get(&parent) {
                        emit("..", node)?;
                    }
                }

                let entries = engine.table().readdir(&path).map_err(map_lookup_err)?;
                for (name, child) in entries {
                    emit(name, child)?;
                }
                if path == "\\" {
                    let name = internal_api::DISPLAY_ROOT;
                    if dir_pattern_matches(pattern.as_deref(), name) {
                        info.reset();
                        let wide: Vec<u16> = name.encode_utf16().collect();
                        info.set_name_raw(wide.as_slice())?;
                        fill_internal_file_info(
                            &InternalEntry::RootDir,
                            None,
                            info.file_info_mut(),
                        );
                        lock.write(&mut info)?;
                    }
                }
            }
        }
        Ok(context.dir_buffer.read(marker, buffer))
    }
}

/// Parent path of a `\`-rooted path (for "..").
fn parent_path(path: &str) -> String {
    match path.rfind('\\') {
        Some(0) | None => "\\".to_string(),
        Some(i) => path[..i].to_string(),
    }
}

struct DuplicateExtentsRequest {
    source_handle: usize,
    source_offset: u64,
    target_offset: u64,
    byte_count: u64,
}

fn parse_duplicate_extents(input: &[u8]) -> Option<DuplicateExtentsRequest> {
    let handle_size = std::mem::size_of::<usize>();
    let need = handle_size + 24;
    if input.len() < need {
        return None;
    }
    let source_handle = read_usize_le(&input[..handle_size])?;
    Some(DuplicateExtentsRequest {
        source_handle,
        source_offset: read_u64_le(&input[handle_size..handle_size + 8])?,
        target_offset: read_u64_le(&input[handle_size + 8..handle_size + 16])?,
        byte_count: read_u64_le(&input[handle_size + 16..handle_size + 24])?,
    })
}

fn read_usize_le(bytes: &[u8]) -> Option<usize> {
    if bytes.len() == 8 {
        Some(u64::from_le_bytes(bytes.try_into().ok()?) as usize)
    } else if bytes.len() == 4 {
        Some(u32::from_le_bytes(bytes.try_into().ok()?) as usize)
    } else {
        None
    }
}

fn read_u64_le(bytes: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

fn path_from_handle(raw_handle: usize) -> Option<String> {
    if raw_handle == 0 {
        return None;
    }
    let mut buf = vec![0u16; 32_768];
    let n = unsafe {
        GetFinalPathNameByHandleW(
            HANDLE(raw_handle as *mut c_void),
            &mut buf,
            GETFINALPATHNAMEBYHANDLE_FLAGS(VOLUME_NAME_DOS.0 | FILE_NAME_NORMALIZED.0),
        )
    };
    if n == 0 || n as usize > buf.len() {
        return None;
    }
    let mut s = String::from_utf16_lossy(&buf[..n as usize]);
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        s = rest.to_string();
    }
    let bytes = s.as_bytes();
    if bytes.len() >= 3 && bytes[1] == b':' && (bytes[2] == b'\\' || bytes[2] == b'/') {
        return Some(crate::lookup::normalize(&s[2..]));
    }
    None
}

fn dir_pattern_matches(pattern: Option<&str>, name: &str) -> bool {
    let Some(pattern) = pattern else {
        return true;
    };
    if pattern.is_empty() || pattern == "*" {
        return true;
    }
    wildcard_match_ci(pattern.as_bytes(), name.as_bytes())
}

fn wildcard_match_ci(pattern: &[u8], name: &[u8]) -> bool {
    let mut p = 0;
    let mut n = 0;
    let mut star = None;
    let mut retry_name = 0;

    while n < name.len() {
        if p < pattern.len() && (pattern[p] == b'?' || pattern[p].eq_ignore_ascii_case(&name[n])) {
            p += 1;
            n += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            p += 1;
            retry_name = n;
        } else if let Some(star_pos) = star {
            p = star_pos + 1;
            retry_name += 1;
            n = retry_name;
        } else {
            return false;
        }
    }

    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

/// Locate the installed WinFsp DLL via the registry (with a default-path
/// fallback) and load it, so `winfsp_init`'s bare-name `LoadLibrary` resolves.
fn preload_winfsp_dll() -> anyhow::Result<()> {
    use widestring::U16CString;
    use windows::core::PCWSTR;
    use windows::Win32::System::LibraryLoader::LoadLibraryW;
    use windows::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};

    fn read_install_dir(subkey: &str) -> Option<String> {
        let sub = U16CString::from_str(subkey).ok()?;
        let val = U16CString::from_str("InstallDir").ok()?;
        let mut buf = [0u16; 260];
        let mut size = (buf.len() * 2) as u32;
        let status = unsafe {
            RegGetValueW(
                HKEY_LOCAL_MACHINE,
                PCWSTR(sub.as_ptr()),
                PCWSTR(val.as_ptr()),
                RRF_RT_REG_SZ,
                None,
                Some(buf.as_mut_ptr().cast()),
                Some(&mut size),
            )
        };
        if status.is_err() {
            return None;
        }
        let len = (size as usize / 2).saturating_sub(1); // drop trailing NUL
        Some(String::from_utf16_lossy(&buf[..len]))
    }

    let install_dir = read_install_dir("SOFTWARE\\WOW6432Node\\WinFsp")
        .or_else(|| read_install_dir("SOFTWARE\\WinFsp"))
        .unwrap_or_else(|| r"C:\Program Files (x86)\WinFsp".to_string());

    let dll = format!("{install_dir}\\bin\\winfsp-x64.dll");
    let wide = U16CString::from_str(&dll).context("bad WinFsp DLL path")?;
    unsafe {
        LoadLibraryW(PCWSTR(wide.as_ptr()))
            .with_context(|| format!("failed to load WinFsp DLL at {dll}"))?;
    }
    Ok(())
}

type VramDiskHost = FileSystemHost<VramDiskFs, FineGuard>;

pub struct MountedVramDisk {
    host: VramDiskHost,
    _init: FspInit,
    active: bool,
}

impl MountedVramDisk {
    pub fn unmount(mut self) {
        self.stop_unmount();
    }

    fn stop_unmount(&mut self) {
        if self.active {
            self.host.stop();
            self.host.unmount();
            self.active = false;
        }
    }
}

impl Drop for MountedVramDisk {
    fn drop(&mut self) {
        self.stop_unmount();
    }
}

fn volume_params() -> VolumeParams {
    let mut vp = VolumeParams::new();
    vp.sector_size(512)
        .sectors_per_allocation_unit((CHUNK_SIZE / 512) as u16)
        .max_component_length(255)
        .volume_creation_time(crate::lookup::now_filetime())
        .volume_serial_number(0x5652_414D) // "VRAM"
        .file_info_timeout(1000)
        .case_sensitive_search(false)
        .case_preserved_names(true)
        .unicode_on_disk(true)
        .persistent_acls(true)
        .device_control(true)
        .post_cleanup_when_modified_only(true)
        .filesystem_name("VRAMDISK");
    vp
}

pub fn mount(engine: StorageEngine, mount: &str, label: &str) -> anyhow::Result<MountedVramDisk> {
    preload_winfsp_dll().context("could not locate/load WinFsp (is it installed?)")?;
    let init = winfsp::winfsp_init().context("WinFsp init failed (is WinFsp installed?)")?;

    let fs = VramDiskFs::new(engine, label);
    let mut host = VramDiskHost::new(volume_params(), fs)
        .map_err(|e| anyhow::anyhow!("failed to create WinFsp host: {e:?}"))?;
    host.mount(mount)
        .map_err(|e| anyhow::anyhow!("failed to mount at {mount}: {e:?}"))?;
    host.start()
        .map_err(|e| anyhow::anyhow!("failed to start dispatcher: {e:?}"))?;
    Ok(MountedVramDisk {
        host,
        _init: init,
        active: true,
    })
}

/// Mount the engine as a WinFsp volume and block until the user unmounts.
pub fn run(engine: StorageEngine, mount_point: &str, label: &str) -> anyhow::Result<()> {
    let mounted = mount(engine, mount_point, label)?;
    println!("\nMounted at {mount_point}. Press Enter (or Ctrl-C / kill the process) to unmount.");

    // Wait for the user to press Enter. A bare EOF (e.g. launched with no
    // attached console) is ignored so the mount stays up until the process is
    // killed; WinFsp tears the volume down on process exit either way.
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF: park forever; rely on Ctrl-C / kill to stop.
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(3600));
                }
            }
            Ok(_) => break,
            Err(_) => break,
        }
    }

    mounted.unmount();
    println!("Unmounted.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_security_descriptor_is_self_relative() {
        let sd = security_descriptor_from_sddl(DEFAULT_SECURITY_SDDL).unwrap();
        assert!(sd.len() >= 20);
        let reported = unsafe {
            GetSecurityDescriptorLength(PSECURITY_DESCRIPTOR(sd.as_ptr() as *mut c_void))
        };
        assert_eq!(reported as usize, sd.len());
    }

    #[test]
    fn directory_pattern_matching_is_case_insensitive() {
        assert!(dir_pattern_matches(None, "$VRAMDISK"));
        assert!(dir_pattern_matches(Some("*"), "$VRAMDISK"));
        assert!(dir_pattern_matches(Some("$vramdisk"), "$VRAMDISK"));
        assert!(dir_pattern_matches(Some("$VRAMDISK*"), "$VRAMDISK"));
        assert!(dir_pattern_matches(Some("$VRAMDIS?"), "$VRAMDISK"));
        assert!(!dir_pattern_matches(Some("hash"), "$VRAMDISK"));
    }

    fn test_fs_with_file(path: &str) -> (VramDiskFs, OpenFile) {
        let _ = preload_winfsp_dll();
        let vram = crate::cuda::Vram::new(0, crate::CHUNK_SIZE).expect("test vram");
        let mut engine = StorageEngine::new(vram, false, false).expect("engine");
        engine.table_mut().create_file(path, 0).unwrap();
        let fs = VramDiskFs::new(engine, "test");
        let context = OpenFile::new(fs.register_path(crate::lookup::normalize(path)), false);
        (fs, context)
    }

    #[test]
    fn create_preserves_display_case() {
        let _ = preload_winfsp_dll();
        let vram = crate::cuda::Vram::new(0, crate::CHUNK_SIZE).expect("test vram");
        let engine = StorageEngine::new(vram, false, false).expect("engine");
        let fs = VramDiskFs::new(engine, "test");
        let name = widestring::U16CString::from_str("\\UPPER-Mixed.TXT").unwrap();
        let mut info: OpenFileInfo = unsafe { std::mem::zeroed() };

        let context = fs
            .create(
                name.as_ucstr(),
                0,
                0,
                FILE_ATTRIBUTE_NORMAL,
                None,
                0,
                None,
                false,
                &mut info,
            )
            .unwrap();

        let engine = fs.engine();
        let kids = engine.table().readdir("\\").unwrap();
        assert_eq!(kids[0].0, "UPPER-Mixed.TXT");
        assert_eq!(
            engine.get("\\upper-mixed.txt").unwrap().name,
            "UPPER-Mixed.TXT"
        );
        drop(engine);
        fs.close(context);
    }

    #[test]
    fn hash_job_hashes_multiple_paths_and_directories() {
        let vram = crate::cuda::Vram::new(0, crate::CHUNK_SIZE * 8).expect("test vram");
        let mut engine = StorageEngine::new(vram, false, false).expect("engine");
        engine.table_mut().create_file("\\a.txt", 0).unwrap();
        engine.table_mut().create_dir("\\dir", 0).unwrap();
        engine.table_mut().create_file("\\dir\\b.txt", 0).unwrap();
        engine.write("\\a.txt", 0, b"abc").unwrap();
        engine.write("\\dir\\b.txt", 0, b"hello").unwrap();

        let result = execute_job_descriptor(
            "job-hash",
            r#"{"op":"hash","algorithm":"sha256","paths":["\\a.txt","\\dir"],"recursive":true}"#,
            &mut engine,
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["algorithm"], "sha256");
        assert_eq!(json["file_count"], 2);
        let files = json["files"].as_array().unwrap();
        assert!(files.iter().any(|f| f["path"] == "\\a.txt"));
        assert!(files.iter().any(|f| f["path"] == "\\dir\\b.txt"));
        assert!(result.contains("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
        assert!(result.contains("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"));
    }

    #[test]
    fn cleanup_deletes_when_winfsp_sets_delete_flag() {
        let (fs, context) = test_fs_with_file("\\victim");
        fs.cleanup(&context, None, FspCleanupFlags::FspCleanupDelete as u32);
        assert!(fs.engine().get("\\victim").is_none());
    }

    #[test]
    fn cleanup_deletes_when_set_delete_marked_pending() {
        let (fs, context) = test_fs_with_file("\\victim");
        context.delete_pending.store(true, Ordering::SeqCst);
        fs.cleanup(&context, None, 0);
        assert!(fs.engine().get("\\victim").is_none());
    }
}
