//! Virtual files exposed under `\$VRAMDISK`.
//!
//! These entries are not part of the regular lookup table. The WinFsp layer
//! dispatches paths in this namespace here and serves them as read-only files
//! and directories.

use crate::api_kernel::{digest_hex, HashAlgorithm};
use crate::cli::format_size;
use crate::engine::{ChunkPlacementReport, EngineError, StorageEngine};
use crate::lookup::{Codec, Node};

pub const DISPLAY_ROOT: &str = "$VRAMDISK";
pub const ROOT: &str = "\\$vramdisk";
pub const HELP: &str = "\\$vramdisk\\help.txt";
pub const STATS_TXT: &str = "\\$vramdisk\\stats.txt";
pub const STATS_JSON: &str = "\\$vramdisk\\stats.json";
pub const TRACE_TXT: &str = "\\$vramdisk\\trace.txt";
pub const TRACE_JSON: &str = "\\$vramdisk\\trace.json";
pub const CHUNKS_JSON: &str = "\\$vramdisk\\chunks.json";
pub const JOBS: &str = "\\$vramdisk\\jobs";
pub const JOBS_PENDING: &str = "\\$vramdisk\\jobs\\pending";
pub const JOBS_COMPLETED: &str = "\\$vramdisk\\jobs\\completed";

const HELP_TEXT: &str = "\
VRAMDISK 内部API

ここは管理ディレクトリです。通常ファイルAPIは読み取り専用ですが、jobs\\pending だけはジョブ投入用に作成/書き込みできます。

  \\$VRAMDISK\\help.txt
      この説明を表示します。

  \\$VRAMDISK\\stats.txt
      容量、重複排除、圧縮の状態を人間向けに表示します。

  \\$VRAMDISK\\stats.json
      stats.txt と同じ情報を JSON で表示します。

  \\$VRAMDISK\\trace.txt
      起動後の read/write 経路カウンタを表示します。

  \\$VRAMDISK\\trace.json
      trace.txt と同じ情報を JSON で表示します。

  \\$VRAMDISK\\chunks.json\\<path>
      指定ファイルの論理チャンクごとの格納状態を JSON で表示します。

  \\$VRAMDISK\\jobs\\pending\\<id>.json
      クライアント指定IDでジョブを投入します。ID は英数字、'.'、'-'、'_'。

  \\$VRAMDISK\\jobs\\<id>\\status.json
      ジョブ状態を JSON で表示します。

  \\$VRAMDISK\\jobs\\<id>\\wait
      ジョブ完了まで read をブロックし、完了時に result.json と同じ内容を返します。

  hash job descriptor 例:
      {\"op\":\"hash\",\"algorithm\":\"sha256\",\"paths\":[\"\\\\data\"],\"recursive\":true}
      md5, sha1, sha256, fnv1a64 に対応します。
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    RootDir,
    HelpFile,
    StatsTextFile,
    StatsJsonFile,
    TraceTextFile,
    TraceJsonFile,
    ChunksRootDir,
    ChunksDir {
        target_dir: String,
    },
    ChunksJsonFile {
        target_file: String,
    },
    JobsRootDir,
    JobsPendingDir,
    JobsCompletedDir,
    JobDir {
        id: String,
    },
    JobSubmitFile {
        id: String,
    },
    JobStatusFile {
        id: String,
    },
    JobResultFile {
        id: String,
    },
    JobWaitFile {
        id: String,
    },
    JobCancelFile {
        id: String,
    },
    HashRootDir,
    HashAlgDir {
        alg: HashAlgorithm,
        target_dir: String,
    },
    HashFile {
        alg: HashAlgorithm,
        target_file: String,
    },
}

impl Entry {
    pub fn is_dir(&self) -> bool {
        matches!(
            self,
            Entry::RootDir
                | Entry::ChunksRootDir
                | Entry::ChunksDir { .. }
                | Entry::JobsRootDir
                | Entry::JobsPendingDir
                | Entry::JobsCompletedDir
                | Entry::JobDir { .. }
                | Entry::HashRootDir
                | Entry::HashAlgDir { .. }
        )
    }
}

pub fn is_internal_path(path: &str) -> bool {
    path == ROOT || path.starts_with("\\$vramdisk\\")
}

pub fn resolve(path: &str, engine: &StorageEngine) -> Option<Entry> {
    if path == ROOT {
        return Some(Entry::RootDir);
    }
    if path == HELP {
        return Some(Entry::HelpFile);
    }
    if path == STATS_TXT {
        return Some(Entry::StatsTextFile);
    }
    if path == STATS_JSON {
        return Some(Entry::StatsJsonFile);
    }
    if path == TRACE_TXT {
        return Some(Entry::TraceTextFile);
    }
    if path == TRACE_JSON {
        return Some(Entry::TraceJsonFile);
    }
    if path == CHUNKS_JSON {
        return Some(Entry::ChunksRootDir);
    }
    if let Some(suffix) = path.strip_prefix("\\$vramdisk\\chunks.json\\") {
        return resolve_chunks_target(suffix, engine);
    }
    if path == JOBS {
        return Some(Entry::JobsRootDir);
    }
    if path == JOBS_PENDING {
        return Some(Entry::JobsPendingDir);
    }
    if path == JOBS_COMPLETED {
        return Some(Entry::JobsCompletedDir);
    }
    if let Some(entry) = resolve_job_path(path) {
        return Some(entry);
    }
    None
}

pub fn job_submit_id(path: &str) -> Option<String> {
    let rest = path.strip_prefix("\\$vramdisk\\jobs\\pending\\")?;
    let id = rest.strip_suffix(".json")?;
    if crate::jobs::is_valid_job_id(id) {
        Some(id.to_string())
    } else {
        None
    }
}

fn resolve_job_path(path: &str) -> Option<Entry> {
    let rest = path.strip_prefix("\\$vramdisk\\jobs\\")?;
    if let Some(id) = rest
        .strip_prefix("pending\\")
        .and_then(|s| s.strip_suffix(".json"))
    {
        return crate::jobs::is_valid_job_id(id)
            .then(|| Entry::JobSubmitFile { id: id.to_string() });
    }
    let (id, leaf) = split_first(rest);
    if !crate::jobs::is_valid_job_id(id) {
        return None;
    }
    if leaf.is_empty() {
        return Some(Entry::JobDir { id: id.to_string() });
    }
    match leaf {
        "status.json" => Some(Entry::JobStatusFile { id: id.to_string() }),
        "result.json" => Some(Entry::JobResultFile { id: id.to_string() }),
        "wait" => Some(Entry::JobWaitFile { id: id.to_string() }),
        "cancel" => Some(Entry::JobCancelFile { id: id.to_string() }),
        _ => None,
    }
}

fn resolve_chunks_target(suffix: &str, engine: &StorageEngine) -> Option<Entry> {
    if suffix.is_empty() {
        return Some(Entry::ChunksRootDir);
    }
    let target = format!("\\{suffix}");
    let node = engine.get(&target)?;
    if node.is_dir {
        Some(Entry::ChunksDir { target_dir: target })
    } else {
        Some(Entry::ChunksJsonFile {
            target_file: target,
        })
    }
}

pub fn content(entry: &Entry, engine: &mut StorageEngine) -> Result<Vec<u8>, EngineError> {
    match entry {
        Entry::HelpFile => Ok(HELP_TEXT.as_bytes().to_vec()),
        Entry::StatsTextFile => Ok(stats_text(engine).into_bytes()),
        Entry::StatsJsonFile => Ok(stats_json(engine).into_bytes()),
        Entry::TraceTextFile => Ok(trace_text(engine).into_bytes()),
        Entry::TraceJsonFile => Ok(trace_json(engine).into_bytes()),
        Entry::ChunksJsonFile { target_file } => Ok(chunks_json(engine, target_file)?.into_bytes()),
        Entry::HashFile { alg, target_file } => {
            let digest = engine.hash_file_gpu(target_file, *alg)?;
            Ok(format!("{}\r\n", digest_hex(&digest)).into_bytes())
        }
        Entry::RootDir
        | Entry::ChunksRootDir
        | Entry::ChunksDir { .. }
        | Entry::JobsRootDir
        | Entry::JobsPendingDir
        | Entry::JobsCompletedDir
        | Entry::JobDir { .. }
        | Entry::JobSubmitFile { .. }
        | Entry::JobStatusFile { .. }
        | Entry::JobResultFile { .. }
        | Entry::JobWaitFile { .. }
        | Entry::JobCancelFile { .. }
        | Entry::HashRootDir
        | Entry::HashAlgDir { .. } => Err(EngineError::NotAFile),
    }
}

pub fn content_len(entry: &Entry) -> u64 {
    match entry {
        Entry::RootDir
        | Entry::ChunksRootDir
        | Entry::ChunksDir { .. }
        | Entry::JobsRootDir
        | Entry::JobsPendingDir
        | Entry::JobsCompletedDir
        | Entry::JobDir { .. }
        | Entry::HashRootDir
        | Entry::HashAlgDir { .. } => 0,
        Entry::HelpFile => HELP_TEXT.len() as u64,
        Entry::StatsTextFile
        | Entry::StatsJsonFile
        | Entry::TraceTextFile
        | Entry::TraceJsonFile
        | Entry::ChunksJsonFile { .. }
        | Entry::JobSubmitFile { .. }
        | Entry::JobStatusFile { .. }
        | Entry::JobResultFile { .. }
        | Entry::JobWaitFile { .. }
        | Entry::JobCancelFile { .. } => 0,
        Entry::HashFile { alg, .. } => (alg.digest_len() as u64 * 2) + 2,
    }
}

fn trace_text(engine: &StorageEngine) -> String {
    let t = engine.trace_snapshot();
    format!(
        "\
VRAMDISK trace\r\n\
\r\n\
calls.read: {}\r\n\
calls.write: {}\r\n\
logical.read_bytes: {}\r\n\
logical.write_bytes: {}\r\n\
\r\n\
raw.read_ops: {}\r\n\
raw.read_bytes: {}\r\n\
raw.write_ops: {}\r\n\
raw.write_bytes: {}\r\n\
\r\n\
compressed.read_chunks: {}\r\n\
compressed.read_requested_bytes: {}\r\n\
compressed.read_full_bytes: {}\r\n\
compressed.read_d2h_saved_bytes: {}\r\n\
compress.batches: {}\r\n\
compress.chunks: {}\r\n\
compress.raw_fallback_chunks: {}\r\n\
\r\n\
dedup.hash_chunks: {}\r\n\
dedup.candidate_chunks: {}\r\n\
dedup.shared_chunks: {}\r\n\
dedup.unique_chunks: {}\r\n\
gpu.hash_chunks: {}\r\n",
        t.read_calls,
        t.write_calls,
        t.logical_read_bytes,
        t.logical_write_bytes,
        t.raw_read_ops,
        t.raw_read_bytes,
        t.raw_write_ops,
        t.raw_write_bytes,
        t.compressed_read_chunks,
        t.compressed_read_requested_bytes,
        t.compressed_read_full_bytes,
        t.compressed_read_full_bytes
            .saturating_sub(t.compressed_read_requested_bytes),
        t.compress_batches,
        t.compress_chunks,
        t.compress_raw_fallback_chunks,
        t.dedup_hash_chunks,
        t.dedup_candidate_chunks,
        t.dedup_shared_chunks,
        t.dedup_unique_chunks,
        t.gpu_hash_chunks,
    )
}

fn trace_json(engine: &StorageEngine) -> String {
    let t = engine.trace_snapshot();
    format!(
        concat!(
            "{{\r\n",
            "  \"calls\": {{\"read\": {}, \"write\": {}}},\r\n",
            "  \"logical\": {{\"read_bytes\": {}, \"write_bytes\": {}}},\r\n",
            "  \"raw\": {{\"read_ops\": {}, \"read_bytes\": {}, \"write_ops\": {}, \"write_bytes\": {}}},\r\n",
            "  \"compressed\": {{\r\n",
            "    \"read_chunks\": {},\r\n",
            "    \"read_requested_bytes\": {},\r\n",
            "    \"read_full_bytes\": {},\r\n",
            "    \"read_d2h_saved_bytes\": {},\r\n",
            "    \"compress_batches\": {},\r\n",
            "    \"compress_chunks\": {},\r\n",
            "    \"raw_fallback_chunks\": {}\r\n",
            "  }},\r\n",
            "  \"dedup\": {{\r\n",
            "    \"hash_chunks\": {},\r\n",
            "    \"candidate_chunks\": {},\r\n",
            "    \"shared_chunks\": {},\r\n",
            "    \"unique_chunks\": {}\r\n",
            "  }},\r\n",
            "  \"gpu\": {{\"hash_chunks\": {}}}\r\n",
            "}}\r\n"
        ),
        t.read_calls,
        t.write_calls,
        t.logical_read_bytes,
        t.logical_write_bytes,
        t.raw_read_ops,
        t.raw_read_bytes,
        t.raw_write_ops,
        t.raw_write_bytes,
        t.compressed_read_chunks,
        t.compressed_read_requested_bytes,
        t.compressed_read_full_bytes,
        t.compressed_read_full_bytes
            .saturating_sub(t.compressed_read_requested_bytes),
        t.compress_batches,
        t.compress_chunks,
        t.compress_raw_fallback_chunks,
        t.dedup_hash_chunks,
        t.dedup_candidate_chunks,
        t.dedup_shared_chunks,
        t.dedup_unique_chunks,
        t.gpu_hash_chunks,
    )
}

fn stats_text(engine: &StorageEngine) -> String {
    let s = engine.stats();
    format!(
        "\
VRAMDISK stats\r\n\
\r\n\
mode.compress: {}\r\n\
mode.dedup: {}\r\n\
mode.nvcomp_lz4: {}\r\n\
\r\n\
volume.total: {} ({})\r\n\
volume.used_physical: {} ({})\r\n\
volume.free_physical: {} ({})\r\n\
\r\n\
namespace.files: {}\r\n\
namespace.directories: {}\r\n\
namespace.logical_file_bytes: {} ({})\r\n\
namespace.logical_allocated: {} ({})\r\n\
\r\n\
chunks.total: {}\r\n\
chunks.used_physical: {}\r\n\
chunks.free_physical: {}\r\n\
chunks.raw_unique: {}\r\n\
chunks.raw_logical: {}\r\n\
chunks.compressed_logical: {}\r\n\
chunks.sparse_logical: {}\r\n\
\r\n\
dedup.shared_logical_chunks: {}\r\n\
dedup.saved: {} ({})\r\n\
compression.payload: {} ({})\r\n\
compression.saved: {} ({})\r\n",
        on_off(s.compress_enabled),
        on_off(s.dedup_enabled),
        yes_no(s.nvcomp_lz4_available),
        s.total_bytes,
        format_size(s.total_bytes),
        s.used_physical_bytes,
        format_size(s.used_physical_bytes),
        s.free_physical_bytes,
        format_size(s.free_physical_bytes),
        s.file_count,
        s.dir_count,
        s.logical_file_bytes,
        format_size(s.logical_file_bytes),
        s.logical_allocated_bytes,
        format_size(s.logical_allocated_bytes),
        s.total_chunks,
        s.used_chunks,
        s.free_chunks,
        s.raw_unique_chunks,
        s.raw_logical_chunks,
        s.compressed_logical_chunks,
        s.sparse_logical_chunks,
        s.dedup_shared_logical_chunks,
        s.dedup_saved_bytes,
        format_size(s.dedup_saved_bytes),
        s.compressed_payload_bytes,
        format_size(s.compressed_payload_bytes),
        s.compression_saved_bytes,
        format_size(s.compression_saved_bytes),
    )
}

fn stats_json(engine: &StorageEngine) -> String {
    let s = engine.stats();
    format!(
        concat!(
            "{{\r\n",
            "  \"mode\": {{\r\n",
            "    \"compress\": {},\r\n",
            "    \"dedup\": {},\r\n",
            "    \"nvcomp_lz4_available\": {}\r\n",
            "  }},\r\n",
            "  \"volume\": {{\r\n",
            "    \"total_bytes\": {},\r\n",
            "    \"used_physical_bytes\": {},\r\n",
            "    \"free_physical_bytes\": {},\r\n",
            "    \"total_chunks\": {},\r\n",
            "    \"used_chunks\": {},\r\n",
            "    \"free_chunks\": {}\r\n",
            "  }},\r\n",
            "  \"namespace\": {{\r\n",
            "    \"file_count\": {},\r\n",
            "    \"dir_count\": {},\r\n",
            "    \"logical_file_bytes\": {},\r\n",
            "    \"logical_allocated_bytes\": {}\r\n",
            "  }},\r\n",
            "  \"chunks\": {{\r\n",
            "    \"raw_unique_chunks\": {},\r\n",
            "    \"raw_logical_chunks\": {},\r\n",
            "    \"compressed_logical_chunks\": {},\r\n",
            "    \"compressed_payload_bytes\": {},\r\n",
            "    \"sparse_logical_chunks\": {}\r\n",
            "  }},\r\n",
            "  \"dedup\": {{\r\n",
            "    \"shared_logical_chunks\": {},\r\n",
            "    \"saved_bytes\": {}\r\n",
            "  }},\r\n",
            "  \"compression\": {{\r\n",
            "    \"saved_bytes\": {}\r\n",
            "  }}\r\n",
            "}}\r\n"
        ),
        s.compress_enabled,
        s.dedup_enabled,
        s.nvcomp_lz4_available,
        s.total_bytes,
        s.used_physical_bytes,
        s.free_physical_bytes,
        s.total_chunks,
        s.used_chunks,
        s.free_chunks,
        s.file_count,
        s.dir_count,
        s.logical_file_bytes,
        s.logical_allocated_bytes,
        s.raw_unique_chunks,
        s.raw_logical_chunks,
        s.compressed_logical_chunks,
        s.compressed_payload_bytes,
        s.sparse_logical_chunks,
        s.dedup_shared_logical_chunks,
        s.dedup_saved_bytes,
        s.compression_saved_bytes,
    )
}

fn chunks_json(engine: &StorageEngine, target_file: &str) -> Result<String, EngineError> {
    let report = engine.file_chunks(target_file)?;
    let mut out = String::new();
    out.push_str("{\r\n");
    out.push_str(&format!(
        "  \"path\": \"{}\",\r\n",
        json_escape(&report.path)
    ));
    out.push_str(&format!("  \"size\": {},\r\n", report.size));
    out.push_str(&format!("  \"chunk_size\": {},\r\n", report.chunk_size));
    out.push_str(&format!(
        "  \"logical_chunks\": {},\r\n",
        report.logical_chunks
    ));
    out.push_str("  \"chunks\": [\r\n");
    for (i, chunk) in report.chunks.iter().enumerate() {
        out.push_str("    {\r\n");
        out.push_str(&format!(
            "      \"logical_chunk\": {},\r\n",
            chunk.logical_chunk
        ));
        out.push_str(&format!(
            "      \"logical_offset\": {},\r\n",
            chunk.logical_offset
        ));
        out.push_str(&format!(
            "      \"logical_len\": {},\r\n",
            chunk.logical_len
        ));
        match &chunk.placement {
            ChunkPlacementReport::Sparse => {
                out.push_str("      \"kind\": \"sparse\"\r\n");
            }
            ChunkPlacementReport::Raw {
                physical_chunk,
                physical_offset,
                refcount,
                content_hash,
            } => {
                out.push_str("      \"kind\": \"raw\",\r\n");
                out.push_str(&format!(
                    "      \"physical_chunk\": {},\r\n",
                    physical_chunk
                ));
                out.push_str(&format!(
                    "      \"physical_offset\": {},\r\n",
                    physical_offset
                ));
                out.push_str(&format!("      \"refcount\": {},\r\n", refcount));
                out.push_str(&format!("      \"shared\": {},\r\n", *refcount > 1));
                out.push_str(&format!(
                    "      \"content_hash\": {}\r\n",
                    json_hash(*content_hash)
                ));
            }
            ChunkPlacementReport::Compressed {
                offset,
                len,
                codec,
                refcount,
                content_hash,
            } => {
                out.push_str("      \"kind\": \"compressed\",\r\n");
                out.push_str(&format!("      \"offset\": {},\r\n", offset));
                out.push_str(&format!("      \"len\": {},\r\n", len));
                out.push_str(&format!("      \"codec\": \"{}\",\r\n", codec_name(*codec)));
                out.push_str(&format!("      \"refcount\": {},\r\n", refcount));
                out.push_str(&format!("      \"shared\": {},\r\n", *refcount > 1));
                out.push_str(&format!(
                    "      \"content_hash\": {}\r\n",
                    json_hash(*content_hash)
                ));
            }
        }
        out.push_str("    }");
        if i + 1 != report.chunks.len() {
            out.push(',');
        }
        out.push_str("\r\n");
    }
    out.push_str("  ]\r\n");
    out.push_str("}\r\n");
    Ok(out)
}

fn json_hash(h: Option<u64>) -> String {
    match h {
        Some(h) => format!("\"0x{h:016x}\""),
        None => "null".to_string(),
    }
}

fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::Lz4 => "lz4",
        Codec::Zstd => "zstd",
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn on_off(v: bool) -> &'static str {
    if v {
        "on"
    } else {
        "off"
    }
}

fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

pub fn hash_child_entry(child: &Node, alg: HashAlgorithm, target_path: String) -> Entry {
    if child.is_dir {
        Entry::HashAlgDir {
            alg,
            target_dir: target_path,
        }
    } else {
        Entry::HashFile {
            alg,
            target_file: target_path,
        }
    }
}

pub fn chunks_child_entry(child: &Node, target_path: String) -> Entry {
    if child.is_dir {
        Entry::ChunksDir {
            target_dir: target_path,
        }
    } else {
        Entry::ChunksJsonFile {
            target_file: target_path,
        }
    }
}

pub fn chunks_target_parent(target_dir: &str) -> Entry {
    if target_dir == "\\" {
        Entry::ChunksRootDir
    } else {
        Entry::ChunksDir {
            target_dir: parent_path(target_dir),
        }
    }
}

pub fn hash_target_parent(alg: HashAlgorithm, target_dir: &str) -> Entry {
    if target_dir == "\\" {
        Entry::HashRootDir
    } else {
        Entry::HashAlgDir {
            alg,
            target_dir: parent_path(target_dir),
        }
    }
}

pub fn supported_algorithms() -> [HashAlgorithm; 4] {
    [
        HashAlgorithm::Md5,
        HashAlgorithm::Sha1,
        HashAlgorithm::Sha256,
        HashAlgorithm::Fnv1a64,
    ]
}

pub fn parent_path(path: &str) -> String {
    match path.rfind('\\') {
        Some(0) | None => "\\".to_string(),
        Some(i) => path[..i].to_string(),
    }
}

fn split_first(s: &str) -> (&str, &str) {
    match s.split_once('\\') {
        Some((first, rest)) => (first, rest),
        None => (s, ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_internal_paths() {
        assert!(is_internal_path("\\$vramdisk"));
        assert!(is_internal_path("\\$vramdisk\\help.txt"));
        assert!(!is_internal_path("\\normal"));
    }

    #[test]
    fn resolves_public_entries_without_md5_alias() {
        let vram = crate::cuda::Vram::new(0, crate::CHUNK_SIZE).expect("test vram");
        let mut engine = StorageEngine::new(vram, false, false).expect("engine");
        engine.table_mut().create_file("\\file.bin", 0).unwrap();

        assert_eq!(resolve("\\$vramdisk", &engine), Some(Entry::RootDir));
        assert_eq!(
            resolve("\\$vramdisk\\help.txt", &engine),
            Some(Entry::HelpFile)
        );
        assert_eq!(
            resolve("\\$vramdisk\\stats.txt", &engine),
            Some(Entry::StatsTextFile)
        );
        assert_eq!(
            resolve("\\$vramdisk\\stats.json", &engine),
            Some(Entry::StatsJsonFile)
        );
        assert_eq!(
            resolve("\\$vramdisk\\trace.txt", &engine),
            Some(Entry::TraceTextFile)
        );
        assert_eq!(
            resolve("\\$vramdisk\\trace.json", &engine),
            Some(Entry::TraceJsonFile)
        );
        assert_eq!(
            resolve("\\$vramdisk\\chunks.json", &engine),
            Some(Entry::ChunksRootDir)
        );
        assert_eq!(
            resolve("\\$vramdisk\\chunks.json\\file.bin", &engine),
            Some(Entry::ChunksJsonFile {
                target_file: "\\file.bin".to_string()
            })
        );
        assert_eq!(resolve("\\$vramdisk\\md5", &engine), None);
        assert_eq!(resolve("\\$vramdisk\\md5\\file.bin", &engine), None);
    }

    #[test]
    fn chunks_json_reports_raw_and_sparse_chunks() {
        let vram = crate::cuda::Vram::new(0, crate::CHUNK_SIZE * 4).expect("test vram");
        let mut engine = StorageEngine::new(vram, false, false).expect("engine");
        engine.table_mut().create_file("\\file.bin", 0).unwrap();
        engine
            .write("\\file.bin", crate::CHUNK_SIZE, b"abc")
            .unwrap();

        let entry = resolve("\\$vramdisk\\chunks.json\\file.bin", &engine).unwrap();
        let json = String::from_utf8(content(&entry, &mut engine).unwrap()).unwrap();
        assert!(json.contains("\"path\": \"\\\\file.bin\""));
        assert!(json.contains("\"logical_chunks\": 2"));
        assert!(json.contains("\"kind\": \"sparse\""));
        assert!(json.contains("\"kind\": \"raw\""));
        assert!(json.contains("\"physical_chunk\": 0"));
    }

    #[test]
    fn chunks_json_reports_dedup_refcount_and_hash() {
        let vram = crate::cuda::Vram::new(0, crate::CHUNK_SIZE * 4).expect("test vram");
        let mut engine = StorageEngine::new(vram, false, true).expect("engine");
        let block = vec![7u8; crate::CHUNK_SIZE as usize];
        engine.table_mut().create_file("\\a.bin", 0).unwrap();
        engine.table_mut().create_file("\\b.bin", 0).unwrap();
        engine.write("\\a.bin", 0, &block).unwrap();
        engine.write("\\b.bin", 0, &block).unwrap();

        let entry = resolve("\\$vramdisk\\chunks.json\\a.bin", &engine).unwrap();
        let json = String::from_utf8(content(&entry, &mut engine).unwrap()).unwrap();
        assert!(json.contains("\"refcount\": 2"));
        assert!(json.contains("\"shared\": true"));
        assert!(json.contains("\"content_hash\": \"0x"));
    }

    #[test]
    fn parses_hash_algorithms() {
        assert_eq!(HashAlgorithm::parse("md5"), Some(HashAlgorithm::Md5));
        assert_eq!(HashAlgorithm::parse("sha-1"), Some(HashAlgorithm::Sha1));
        assert_eq!(HashAlgorithm::parse("sha256"), Some(HashAlgorithm::Sha256));
        assert_eq!(
            HashAlgorithm::parse("fnv-1a-64"),
            Some(HashAlgorithm::Fnv1a64)
        );
        assert_eq!(HashAlgorithm::parse("nope"), None);
    }
}
