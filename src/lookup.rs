//! The lookup-table: the in-memory namespace, file metadata and the
//! per-logical-chunk coordinate arrays.
//!
//! This module owns *no* VRAM. It is a pure data structure: a flat
//! `HashMap<normalized path, Node>` where directory nodes list their
//! children. The storage engine fills in each file's `coords` and the
//! chunk allocator backs the physical placements.

use std::collections::{BTreeSet, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::chunk::ChunkId;

/// Per-chunk compression codec. Chosen per logical chunk so already-compressed
/// data (jpeg/zip/mp4) can be stored verbatim while text/binaries shrink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Lz4,
    Zstd,
}

/// Where a single logical 64KiB chunk physically lives in the VRAM buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Uncompressed: occupies exactly one physical chunk, byte offset = id * CHUNK_SIZE.
    Raw { chunk: ChunkId },
    /// Compressed: `len` bytes of `codec` data at absolute byte `offset`.
    Compressed { offset: u64, len: u32, codec: Codec },
}

/// A filesystem node: a directory or a file.
#[derive(Debug, Clone)]
pub struct Node {
    /// Final path component in its original case (for display / readdir).
    pub name: String,
    pub is_dir: bool,
    /// Windows file attributes (FILE_ATTRIBUTE_*).
    pub attributes: u32,
    /// Logical file size in bytes (0 for directories).
    pub size: u64,
    pub created: u64,
    pub accessed: u64,
    pub modified: u64,
    pub changed: u64,
    /// Unique, stable inode-like id (WinFsp index number).
    pub index_number: u64,
    /// Child display names (directories only).
    pub children: BTreeSet<String>,
    /// Self-relative Windows security descriptor. Empty means "use the volume
    /// default descriptor" until one is supplied by create/set_security.
    pub security_descriptor: Vec<u8>,
    /// Placement of each logical chunk, indexed by logical chunk number (files only).
    /// `None` is a sparse hole that reads as zeros. Empty for zero-length files.
    pub coords: Vec<Option<Placement>>,
}

impl Node {
    fn new(name: String, is_dir: bool, attributes: u32, index_number: u64) -> Self {
        let now = now_filetime();
        Node {
            name,
            is_dir,
            attributes,
            size: 0,
            created: now,
            accessed: now,
            modified: now,
            changed: now,
            index_number,
            children: BTreeSet::new(),
            security_descriptor: Vec::new(),
            coords: Vec::new(),
        }
    }
}

/// Error type for namespace operations, mapped to NTSTATUS in the fs layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupError {
    NotFound,
    AlreadyExists,
    NotADirectory,
    IsADirectory,
    NotEmpty,
    InvalidName,
}

pub type LResult<T> = Result<T, LookupError>;

/// The whole namespace.
pub struct LookupTable {
    /// Keyed by normalized (lowercased, `\`-rooted) full path.
    nodes: HashMap<String, Node>,
    next_index: u64,
}

impl LookupTable {
    /// Create a table containing just the root directory `\`.
    pub fn new() -> Self {
        let mut nodes = HashMap::new();
        // FILE_ATTRIBUTE_DIRECTORY = 0x10
        nodes.insert(ROOT.to_string(), Node::new(String::new(), true, 0x10, 1));
        LookupTable {
            nodes,
            next_index: 2,
        }
    }

    fn alloc_index(&mut self) -> u64 {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    pub fn get(&self, path: &str) -> Option<&Node> {
        self.nodes.get(&normalize(path))
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    #[allow(dead_code)] // used by the storage engine (Phase 4)
    pub fn get_mut(&mut self, path: &str) -> Option<&mut Node> {
        self.nodes.get_mut(&normalize(path))
    }

    #[cfg(test)]
    pub fn exists(&self, path: &str) -> bool {
        self.nodes.contains_key(&normalize(path))
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Create a file at `path`. Parent must exist and be a directory.
    pub fn create_file(&mut self, path: &str, attributes: u32) -> LResult<&mut Node> {
        self.create(path, false, attributes)
    }

    /// Create a directory at `path`. Parent must exist and be a directory.
    pub fn create_dir(&mut self, path: &str, attributes: u32) -> LResult<&mut Node> {
        self.create(path, true, attributes | 0x10)
    }

    fn create(&mut self, path: &str, is_dir: bool, attributes: u32) -> LResult<&mut Node> {
        let key = normalize(path);
        if key == ROOT {
            return Err(LookupError::AlreadyExists);
        }
        if self.nodes.contains_key(&key) {
            return Err(LookupError::AlreadyExists);
        }
        let (parent, name) = split_parent(&key).ok_or(LookupError::InvalidName)?;
        if name.is_empty() {
            return Err(LookupError::InvalidName);
        }
        match self.nodes.get_mut(&parent) {
            None => return Err(LookupError::NotFound),
            Some(p) if !p.is_dir => return Err(LookupError::NotADirectory),
            Some(p) => {
                p.children.insert(display_name(path));
            }
        }
        let index = self.alloc_index();
        let node = Node::new(display_name(path), is_dir, attributes, index);
        self.nodes.insert(key.clone(), node);
        Ok(self.nodes.get_mut(&key).unwrap())
    }

    /// Remove a file or empty directory. Returns the removed node (so the
    /// engine can free its chunks). Root cannot be removed.
    pub fn remove(&mut self, path: &str) -> LResult<Node> {
        let key = normalize(path);
        if key == ROOT {
            return Err(LookupError::InvalidName);
        }
        let node = self.nodes.get(&key).ok_or(LookupError::NotFound)?;
        if node.is_dir && !node.children.is_empty() {
            return Err(LookupError::NotEmpty);
        }
        let (parent, _) = split_parent(&key).ok_or(LookupError::InvalidName)?;
        let removed = self.nodes.remove(&key).unwrap();
        if let Some(p) = self.nodes.get_mut(&parent) {
            p.children.remove(&removed.name);
        }
        Ok(removed)
    }

    /// List the children of a directory as (display name, &Node).
    pub fn readdir(&self, path: &str) -> LResult<Vec<(&str, &Node)>> {
        let key = normalize(path);
        let dir = self.nodes.get(&key).ok_or(LookupError::NotFound)?;
        if !dir.is_dir {
            return Err(LookupError::NotADirectory);
        }
        let base = if key == ROOT {
            String::new()
        } else {
            key.clone()
        };
        let mut out = Vec::with_capacity(dir.children.len());
        for child in &dir.children {
            let child_key = format!("{base}\\{}", child.to_ascii_lowercase());
            if let Some(n) = self.nodes.get(&child_key) {
                out.push((child.as_str(), n));
            }
        }
        Ok(out)
    }

    /// Rename/move `from` to `to`. If `to` exists: replaced when `replace`,
    /// else `AlreadyExists`. Moves the whole subtree for directories.
    pub fn rename(&mut self, from: &str, to: &str, replace: bool) -> LResult<()> {
        let from_key = normalize(from);
        let to_key = normalize(to);
        if from_key == ROOT {
            return Err(LookupError::InvalidName);
        }
        if !self.nodes.contains_key(&from_key) {
            return Err(LookupError::NotFound);
        }
        if from_key == to_key {
            return Ok(());
        }
        // Disallow moving a directory into itself or one of its own
        // descendants (e.g. `\a` -> `\a\b`): re-rooting the subtree under a key
        // that lives inside it would orphan and corrupt the namespace.
        let from_is_dir = self.nodes.get(&from_key).map(|n| n.is_dir).unwrap_or(false);
        if from_is_dir && to_key.starts_with(&format!("{from_key}\\")) {
            return Err(LookupError::InvalidName);
        }
        if self.nodes.contains_key(&to_key) {
            if !replace {
                return Err(LookupError::AlreadyExists);
            }
            let existing = self.nodes.get(&to_key).unwrap();
            if existing.is_dir {
                return Err(LookupError::IsADirectory);
            }
            self.remove(to)?;
        }
        let (to_parent, _) = split_parent(&to_key).ok_or(LookupError::InvalidName)?;
        match self.nodes.get(&to_parent) {
            None => return Err(LookupError::NotFound),
            Some(p) if !p.is_dir => return Err(LookupError::NotADirectory),
            _ => {}
        }

        // Collect the subtree (the node itself plus, for dirs, all descendants).
        let descendants: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| *k == &from_key || k.starts_with(&format!("{from_key}\\")))
            .cloned()
            .collect();

        // Detach from old parent; attach new display name to new parent.
        let (from_parent, _) = split_parent(&from_key).unwrap();
        let new_name = display_name(to);
        let old_name = self.nodes.get(&from_key).unwrap().name.clone();
        if let Some(p) = self.nodes.get_mut(&from_parent) {
            p.children.remove(&old_name);
        }
        if let Some(p) = self.nodes.get_mut(&to_parent) {
            p.children.insert(new_name.clone());
        }

        for old in descendants {
            let mut node = self.nodes.remove(&old).unwrap();
            let new_key = if old == from_key {
                to_key.clone()
            } else {
                // Re-root the descendant path under to_key.
                format!("{}{}", to_key, &old[from_key.len()..])
            };
            if old == from_key {
                node.name = new_name.clone();
            }
            self.nodes.insert(new_key, node);
        }
        Ok(())
    }
}

impl Default for LookupTable {
    fn default() -> Self {
        Self::new()
    }
}

/// The canonical root key.
const ROOT: &str = "\\";

/// Current time as a Windows FILETIME (100ns ticks since 1601-01-01).
pub fn now_filetime() -> u64 {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    // 11644473600 seconds between 1601 and 1970.
    const EPOCH_DIFF_100NS: u64 = 11_644_473_600 * 10_000_000;
    EPOCH_DIFF_100NS + dur.as_nanos() as u64 / 100
}

/// Normalize a path into its lookup key: `\`-separated, lowercased, no
/// trailing separator (except the root). Accepts `/` and `\`.
pub fn normalize(path: &str) -> String {
    let mut s = String::with_capacity(path.len() + 1);
    s.push('\\');
    for comp in path
        .split(['\\', '/'])
        .filter(|c| !c.is_empty() && *c != ".")
    {
        if s.len() > 1 {
            s.push('\\');
        }
        s.push_str(&comp.to_ascii_lowercase());
    }
    s
}

/// Final path component in original case.
pub fn display_name(path: &str) -> String {
    path.split(['\\', '/'])
        .filter(|c| !c.is_empty())
        .next_back()
        .unwrap_or("")
        .to_string()
}

/// Split a normalized key into (parent key, lowercased final component).
fn split_parent(key: &str) -> Option<(String, String)> {
    if key == ROOT {
        return None;
    }
    match key.rfind('\\') {
        Some(0) => Some((ROOT.to_string(), key[1..].to_string())),
        Some(i) => Some((key[..i].to_string(), key[i + 1..].to_string())),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_paths() {
        assert_eq!(normalize("\\"), "\\");
        assert_eq!(normalize("/"), "\\");
        assert_eq!(normalize("\\Foo\\Bar"), "\\foo\\bar");
        assert_eq!(normalize("Foo/Bar/"), "\\foo\\bar");
        assert_eq!(normalize("\\a\\\\b"), "\\a\\b");
    }

    #[test]
    fn root_exists() {
        let t = LookupTable::new();
        assert!(t.get("\\").unwrap().is_dir);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn create_and_lookup() {
        let mut t = LookupTable::new();
        t.create_dir("\\dir", 0).unwrap();
        t.create_file("\\dir\\file.txt", 0).unwrap();
        assert!(t.get("\\dir").unwrap().is_dir);
        assert_eq!(t.get("\\DIR\\FILE.TXT").unwrap().name, "file.txt");
        // case-insensitive lookup, case-preserving display.
        let kids = t.readdir("\\dir").unwrap();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].0, "file.txt");
    }

    #[test]
    fn create_errors() {
        let mut t = LookupTable::new();
        assert_eq!(
            t.create_dir("\\", 0).unwrap_err(),
            LookupError::AlreadyExists
        );
        assert_eq!(
            t.create_file("\\nope\\f", 0).unwrap_err(),
            LookupError::NotFound
        );
        t.create_file("\\f", 0).unwrap();
        assert_eq!(
            t.create_file("\\f", 0).unwrap_err(),
            LookupError::AlreadyExists
        );
        // parent is a file, not a dir.
        assert_eq!(
            t.create_file("\\f\\child", 0).unwrap_err(),
            LookupError::NotADirectory
        );
    }

    #[test]
    fn remove_rules() {
        let mut t = LookupTable::new();
        t.create_dir("\\d", 0).unwrap();
        t.create_file("\\d\\f", 0).unwrap();
        assert_eq!(t.remove("\\d").unwrap_err(), LookupError::NotEmpty);
        t.remove("\\d\\f").unwrap();
        assert!(!t.exists("\\d\\f"));
        t.remove("\\d").unwrap();
        assert!(!t.exists("\\d"));
        assert!(t.readdir("\\").unwrap().is_empty());
    }

    #[test]
    fn rename_file() {
        let mut t = LookupTable::new();
        t.create_file("\\a.txt", 0).unwrap();
        t.rename("\\a.txt", "\\b.txt", false).unwrap();
        assert!(!t.exists("\\a.txt"));
        assert_eq!(t.get("\\b.txt").unwrap().name, "b.txt");
    }

    #[test]
    fn rename_dir_subtree() {
        let mut t = LookupTable::new();
        t.create_dir("\\src", 0).unwrap();
        t.create_dir("\\src\\sub", 0).unwrap();
        t.create_file("\\src\\sub\\deep.txt", 0).unwrap();
        t.create_dir("\\dst", 0).unwrap();
        t.rename("\\src", "\\dst\\moved", false).unwrap();
        assert!(!t.exists("\\src"));
        assert!(t.exists("\\dst\\moved"));
        assert!(t.exists("\\dst\\moved\\sub\\deep.txt"));
        assert_eq!(t.get("\\dst\\moved").unwrap().name, "moved");
        let kids = t.readdir("\\dst").unwrap();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].0, "moved");
    }

    #[test]
    fn rename_replace() {
        let mut t = LookupTable::new();
        t.create_file("\\a", 0).unwrap();
        t.create_file("\\b", 0).unwrap();
        assert_eq!(
            t.rename("\\a", "\\b", false).unwrap_err(),
            LookupError::AlreadyExists
        );
        t.rename("\\a", "\\b", true).unwrap();
        assert!(!t.exists("\\a"));
        assert!(t.exists("\\b"));
    }

    #[test]
    fn rename_into_own_subtree_rejected() {
        let mut t = LookupTable::new();
        t.create_dir("\\a", 0).unwrap();
        t.create_dir("\\a\\b", 0).unwrap();
        t.create_file("\\a\\b\\f", 0).unwrap();
        // Moving \a into its own descendant must be refused, not corrupt the tree.
        assert_eq!(
            t.rename("\\a", "\\a\\b\\c", false).unwrap_err(),
            LookupError::InvalidName
        );
        assert_eq!(
            t.rename("\\a", "\\a\\moved", false).unwrap_err(),
            LookupError::InvalidName
        );
        // The namespace is intact and still navigable.
        assert!(t.exists("\\a"));
        assert!(t.exists("\\a\\b\\f"));
        assert_eq!(t.readdir("\\").unwrap().len(), 1);
        assert_eq!(t.readdir("\\a").unwrap().len(), 1);
        // A sibling move (not into the subtree) still works.
        t.create_dir("\\dst", 0).unwrap();
        t.rename("\\a", "\\dst\\a", false).unwrap();
        assert!(t.exists("\\dst\\a\\b\\f"));
    }

    #[test]
    fn filetime_is_reasonable() {
        // Well after the year 2000 (~1.26e17 100ns ticks) and before 2200.
        let t = now_filetime();
        assert!(t > 125_000_000_000_000_000);
        assert!(t < 190_000_000_000_000_000);
    }
}
