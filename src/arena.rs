//! Byte-granular sub-allocator for packed variable-length compressed blobs.
//!
//! The bitmap [`ChunkAllocator`](crate::chunk) only deals in whole 64KiB
//! chunks, which would waste all of compression's savings. This allocator
//! carves byte ranges out of *arenas*, where each arena is one 64KiB chunk
//! obtained from the bitmap allocator. A compressed blob is always smaller
//! than 64KiB (we only store it compressed when it shrank), so every blob
//! fits inside a single arena and never spans chunks.
//!
//! The engine drives chunk lifecycle: when no existing arena has room it
//! grabs a fresh chunk from the bitmap and hands it here via [`add_arena`];
//! when an arena empties, [`free`] returns its chunk id so the engine can
//! release it back to the bitmap.
//!
//! [`add_arena`]: CompressedAllocator::add_arena
//! [`free`]: CompressedAllocator::free

use std::collections::HashMap;

use crate::chunk::ChunkId;
use crate::CHUNK_SIZE;

struct Arena {
    /// Next never-used offset within the chunk.
    bump: u32,
    /// Live (allocated) bytes; the arena is dropped when this hits zero.
    used: u32,
    /// Freed `(offset, len)` slots available for reuse (no coalescing).
    free: Vec<(u32, u32)>,
}

impl Arena {
    fn new() -> Self {
        Arena {
            bump: 0,
            used: 0,
            free: Vec::new(),
        }
    }

    /// Try to carve `len` bytes from this arena, returning the local offset.
    fn alloc(&mut self, len: u32) -> Option<u32> {
        // First-fit over freed slots.
        for i in 0..self.free.len() {
            let (off, slot_len) = self.free[i];
            if slot_len >= len {
                if slot_len == len {
                    self.free.swap_remove(i);
                } else {
                    self.free[i] = (off + len, slot_len - len);
                }
                self.used += len;
                return Some(off);
            }
        }
        // Otherwise bump.
        if (CHUNK_SIZE as u32 - self.bump) >= len {
            let off = self.bump;
            self.bump += len;
            self.used += len;
            return Some(off);
        }
        None
    }

    fn free(&mut self, off: u32, len: u32) {
        debug_assert!(self.used >= len);
        self.used -= len;
        if off + len == self.bump {
            // Freed the tail: reclaim bump space directly.
            self.bump -= len;
        } else {
            self.free.push((off, len));
        }
    }
}

/// Sub-allocator over a set of 64KiB arena chunks.
#[derive(Default)]
pub struct CompressedAllocator {
    arenas: HashMap<ChunkId, Arena>,
}

impl CompressedAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to allocate `len` bytes in an existing arena, returning the
    /// absolute byte offset into the VRAM buffer. Returns `None` if no current
    /// arena has room (the engine should then [`add_arena`] and retry).
    ///
    /// [`add_arena`]: CompressedAllocator::add_arena
    pub fn try_alloc(&mut self, len: u32) -> Option<u64> {
        debug_assert!(len > 0 && len as u64 <= CHUNK_SIZE);
        for (&chunk, arena) in self.arenas.iter_mut() {
            if let Some(off) = arena.alloc(len) {
                return Some(chunk as u64 * CHUNK_SIZE + off as u64);
            }
        }
        None
    }

    /// Register a freshly obtained chunk as a new arena and allocate `len`
    /// bytes from it (which always succeeds for `len <= CHUNK_SIZE`).
    pub fn add_arena(&mut self, chunk: ChunkId, len: u32) -> u64 {
        let mut arena = Arena::new();
        let off = arena.alloc(len).expect("fresh arena must fit len <= chunk");
        self.arenas.insert(chunk, arena);
        chunk as u64 * CHUNK_SIZE + off as u64
    }

    /// Free a previously allocated region. Returns `Some(chunk)` when the arena
    /// became completely empty, so the engine can release that chunk back to
    /// the bitmap allocator.
    pub fn free(&mut self, offset: u64, len: u32) -> Option<ChunkId> {
        let chunk = (offset / CHUNK_SIZE) as ChunkId;
        let local = (offset % CHUNK_SIZE) as u32;
        let arena = self.arenas.get_mut(&chunk)?;
        arena.free(local, len);
        if arena.used == 0 {
            self.arenas.remove(&chunk);
            Some(chunk)
        } else {
            None
        }
    }

    /// Number of live arena chunks (for diagnostics/tests).
    #[cfg(test)]
    pub fn arena_count(&self) -> usize {
        self.arenas.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper mimicking the engine: hand out fresh chunk ids on demand.
    struct Sim {
        ca: CompressedAllocator,
        next_chunk: ChunkId,
        live_chunks: std::collections::BTreeSet<ChunkId>,
    }
    impl Sim {
        fn new() -> Self {
            Sim {
                ca: CompressedAllocator::new(),
                next_chunk: 0,
                live_chunks: Default::default(),
            }
        }
        fn alloc(&mut self, len: u32) -> u64 {
            if let Some(o) = self.ca.try_alloc(len) {
                return o;
            }
            let c = self.next_chunk;
            self.next_chunk += 1;
            self.live_chunks.insert(c);
            self.ca.add_arena(c, len)
        }
        fn free(&mut self, off: u64, len: u32) {
            if let Some(c) = self.ca.free(off, len) {
                self.live_chunks.remove(&c);
            }
        }
    }

    #[test]
    fn packs_multiple_blobs_into_one_chunk() {
        let mut s = Sim::new();
        let a = s.alloc(1000);
        let b = s.alloc(2000);
        let c = s.alloc(3000);
        // All three fit in a single 64KiB chunk.
        assert_eq!(s.live_chunks.len(), 1);
        // Distinct, non-overlapping offsets.
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert!(b >= a + 1000);
        assert!(c >= b + 2000);
    }

    #[test]
    fn spills_to_new_chunk_when_full() {
        let mut s = Sim::new();
        let big = (CHUNK_SIZE as u32) - 100;
        s.alloc(big);
        assert_eq!(s.live_chunks.len(), 1);
        // Won't fit in the remaining 100 bytes -> new arena.
        s.alloc(500);
        assert_eq!(s.live_chunks.len(), 2);
    }

    #[test]
    fn frees_chunk_when_arena_empties() {
        let mut s = Sim::new();
        let a = s.alloc(4000);
        let b = s.alloc(4000);
        assert_eq!(s.live_chunks.len(), 1);
        s.free(a, 4000);
        assert_eq!(s.live_chunks.len(), 1, "still one live blob");
        s.free(b, 4000);
        assert_eq!(s.live_chunks.len(), 0, "arena should be released");
        assert_eq!(s.ca.arena_count(), 0);
    }

    #[test]
    fn reuses_freed_slots() {
        let mut s = Sim::new();
        let a = s.alloc(1000);
        let _b = s.alloc(1000);
        s.free(a, 1000);
        // A 1000-byte request should reuse the freed slot, not grow the chunk.
        let c = s.alloc(1000);
        assert_eq!(c, a, "freed slot not reused");
        assert_eq!(s.live_chunks.len(), 1);
    }

    #[test]
    fn tail_free_reclaims_bump() {
        let mut s = Sim::new();
        let _a = s.alloc(1000);
        let b = s.alloc(2000);
        // Freeing the tail allocation should let the next big alloc reuse it.
        s.free(b, 2000);
        let c = s.alloc(2000);
        assert_eq!(c, b, "tail bump space not reclaimed");
        assert_eq!(s.live_chunks.len(), 1);
    }
}
