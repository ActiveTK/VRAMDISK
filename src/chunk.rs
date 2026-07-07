//! Free-chunk management via a bitmap.
//!
//! One bit per chunk in the VRAM buffer: `0` = free, `1` = used. The
//! allocator hands out single chunks (the dedup/CoW primitive) and
//! best-effort contiguous runs (so uncompressed files can be stored as
//! `(start, count)` extents).

/// Index of a chunk within the VRAM buffer.
pub type ChunkId = u32;

/// Bitmap allocator over a fixed number of fixed-size chunks.
#[derive(Debug)]
pub struct ChunkAllocator {
    /// Packed bits, 64 chunks per word. Bit set => used.
    words: Vec<u64>,
    total: u32,
    used: u32,
    /// Word index to start the next first-fit scan from.
    cursor: usize,
}

impl ChunkAllocator {
    /// Create an allocator for `total` chunks, all initially free.
    pub fn new(total: u32) -> Self {
        let words = (total as usize).div_ceil(64);
        let mut a = ChunkAllocator {
            words: vec![0u64; words],
            total,
            used: 0,
            cursor: 0,
        };
        // Mark padding bits (beyond `total`) in the last word as used so they
        // are never handed out.
        let rem = (total % 64) as u32;
        if rem != 0 {
            let last = words - 1;
            let valid_mask = (1u64 << rem) - 1;
            a.words[last] = !valid_mask;
        }
        a
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn free(&self) -> u32 {
        self.total - self.used
    }

    #[inline]
    fn loc(chunk: ChunkId) -> (usize, u32) {
        (chunk as usize / 64, chunk % 64)
    }

    /// Whether `chunk` is currently allocated.
    pub fn is_used(&self, chunk: ChunkId) -> bool {
        let (w, b) = Self::loc(chunk);
        self.words[w] & (1u64 << b) != 0
    }

    /// Allocate a single free chunk, or `None` if the disk is full.
    pub fn alloc_one(&mut self) -> Option<ChunkId> {
        let n = self.words.len();
        for i in 0..n {
            let w = (self.cursor + i) % n;
            let word = self.words[w];
            if word != u64::MAX {
                let bit = word.trailing_ones(); // first zero bit
                let chunk = (w as u32) * 64 + bit;
                debug_assert!(chunk < self.total);
                self.words[w] |= 1u64 << bit;
                self.used += 1;
                self.cursor = w;
                return Some(chunk);
            }
        }
        None
    }

    /// Allocate `count` contiguous free chunks, returning the start id.
    ///
    /// Best-effort first-fit; returns `None` if no run that long exists even
    /// when enough total chunks are free (fragmentation).
    pub fn alloc_contiguous(&mut self, count: u32) -> Option<ChunkId> {
        if count == 0 {
            return None;
        }
        if count == 1 {
            return self.alloc_one();
        }
        if count > self.free() {
            return None;
        }

        // Simple bit scan. Walks the run of free bits; resets on a used bit.
        let mut run_start: u32 = 0;
        let mut run_len: u32 = 0;
        let mut chunk: u32 = 0;
        while chunk < self.total {
            if self.is_used(chunk) {
                run_len = 0;
                run_start = chunk + 1;
            } else {
                if run_len == 0 {
                    run_start = chunk;
                }
                run_len += 1;
                if run_len == count {
                    self.mark_used(run_start, count);
                    return Some(run_start);
                }
            }
            chunk += 1;
        }
        None
    }

    /// Mark `[start, start+count)` used (internal; assumes currently free).
    fn mark_used(&mut self, start: ChunkId, count: u32) {
        for c in start..start + count {
            let (w, b) = Self::loc(c);
            debug_assert!(
                self.words[w] & (1u64 << b) == 0,
                "double-alloc of chunk {c}"
            );
            self.words[w] |= 1u64 << b;
        }
        self.used += count;
        self.cursor = Self::loc(start).0;
    }

    /// Free a single chunk. Panics in debug if it was already free.
    pub fn free_one(&mut self, chunk: ChunkId) {
        let (w, b) = Self::loc(chunk);
        let mask = 1u64 << b;
        debug_assert!(self.words[w] & mask != 0, "double-free of chunk {chunk}");
        if self.words[w] & mask != 0 {
            self.words[w] &= !mask;
            self.used -= 1;
        }
    }

    /// Free a contiguous run `[start, start+count)`.
    #[allow(dead_code)] // used by the storage engine (Phase 4)
    pub fn free_contiguous(&mut self, start: ChunkId, count: u32) {
        for c in start..start + count {
            self.free_one(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc_free() {
        let mut a = ChunkAllocator::new(10);
        assert_eq!(a.total(), 10);
        assert_eq!(a.free(), 10);
        let c0 = a.alloc_one().unwrap();
        let c1 = a.alloc_one().unwrap();
        assert_ne!(c0, c1);
        assert!(a.is_used(c0));
        assert_eq!(a.used(), 2);
        a.free_one(c0);
        assert!(!a.is_used(c0));
        assert_eq!(a.used(), 1);
    }

    #[test]
    fn exhaustion() {
        let mut a = ChunkAllocator::new(3);
        let ids: Vec<_> = (0..3).map(|_| a.alloc_one().unwrap()).collect();
        assert_eq!(a.free(), 0);
        assert!(a.alloc_one().is_none());
        // Distinct ids covering all chunks.
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn padding_bits_never_allocated() {
        // 65 chunks => 2 words, 63 padding bits in the second word.
        let mut a = ChunkAllocator::new(65);
        let mut seen = Vec::new();
        while let Some(c) = a.alloc_one() {
            assert!(c < 65, "allocated out-of-range chunk {c}");
            seen.push(c);
        }
        assert_eq!(seen.len(), 65);
        assert_eq!(a.free(), 0);
    }

    #[test]
    fn contiguous_runs() {
        let mut a = ChunkAllocator::new(100);
        let start = a.alloc_contiguous(10).unwrap();
        assert_eq!(start, 0);
        for c in 0..10 {
            assert!(a.is_used(c));
        }
        let next = a.alloc_contiguous(5).unwrap();
        assert_eq!(next, 10);
    }

    #[test]
    fn contiguous_handles_fragmentation() {
        let mut a = ChunkAllocator::new(10);
        // Use up chunks, then free alternating ones to fragment.
        let ids: Vec<_> = (0..10).map(|_| a.alloc_one().unwrap()).collect();
        for &c in ids.iter().filter(|c| *c % 2 == 0) {
            a.free_one(c);
        }
        assert_eq!(a.free(), 5);
        // No run of 2 exists despite 5 free chunks.
        assert!(a.alloc_contiguous(2).is_none());
        // But singles still work.
        assert!(a.alloc_one().is_some());
    }

    #[test]
    fn reuses_freed_chunks() {
        let mut a = ChunkAllocator::new(64);
        let ids: Vec<_> = (0..64).map(|_| a.alloc_one().unwrap()).collect();
        assert!(a.alloc_one().is_none());
        a.free_one(ids[40]);
        assert_eq!(a.alloc_one(), Some(40));
    }
}
