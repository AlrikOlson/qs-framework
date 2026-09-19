//! Bounded LRU cache for shaped text.
//!
//! Cache hits compare the stored text as well as its hash, so a hash collision
//! cannot return another string's glyphs. Eviction uses a linked list over a slab
//! and takes constant time.

use std::collections::HashMap;
use std::sync::Arc;

use crate::fontdb::FontId;
use crate::shape::{ShapedRun, Shaper};
use crate::{Features, PxSize};

const NIL: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RunKey {
    pub text_hash: u64,
    pub font: FontId,
    pub size: PxSize,
    pub features: Features,
}

impl RunKey {
    pub fn new(text: &str, font: FontId, size: PxSize, features: Features) -> Self {
        Self {
            text_hash: fnv1a(text.as_bytes()),
            font,
            size,
            features,
        }
    }
}

/// FNV-1a, 64-bit.
///
/// Deliberately **not** `DefaultHasher` or `RandomState`. `RandomState` reseeds per
/// instance, so the same filename hashes differently on every call and the cache never
/// hits -- a failure that costs no correctness and all of the performance, and that a
/// frame-time graph shows only as "shaping is slow". Determinism is the requirement here,
/// and FNV is deterministic by construction rather than by documentation.
///
/// A weak hash is acceptable precisely because [`ShapedRunCache`] verifies the stored text
/// on every hit: a collision costs one reshape, not a wrong filename on screen.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Misses since the last [`ShapedRunCache::begin_frame`]. Reported per frame as
    /// `shaped_runs_new` -- when a frame-time regression appears, this is the first number
    /// to read (see the bench-report contract).
    pub new_this_frame: u32,
    /// Collisions caught by the text verification described in the module docs. Expected
    /// to be zero forever; if it is not, that is worth knowing rather than not knowing.
    pub hash_collisions: u64,
}

struct Node {
    key: RunKey,
    text: Box<str>,
    run: Arc<ShapedRun>,
    prev: u32,
    next: u32,
}

/// Bounded LRU over shaped runs.
pub struct ShapedRunCache {
    index: HashMap<RunKey, u32>,
    slab: Vec<Node>,
    free: Vec<u32>,
    head: u32,
    tail: u32,
    capacity: usize,
    stats: CacheStats,
}

impl std::fmt::Debug for ShapedRunCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapedRunCache")
            .field("len", &self.index.len())
            .field("capacity", &self.capacity)
            .field("stats", &self.stats)
            .finish()
    }
}

impl ShapedRunCache {
    /// `capacity` is in entries. The default sizing rationale: a 1080p viewport at the
    /// Compact density shows ~45 rows, and a fling covers a few thousand distinct names
    /// before the user lets go. 8192 holds a full fling without evicting, which is what
    /// keeps `new_this_frame` near zero in steady state.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            index: HashMap::with_capacity(capacity),
            slab: Vec::with_capacity(capacity),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            capacity,
            stats: CacheStats::default(),
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Reset the per-frame counters. Called once at the top of a frame.
    pub fn begin_frame(&mut self) {
        self.stats.new_this_frame = 0;
    }

    /// Shape `text`, or return the cached result.
    pub fn get_or_shape(
        &mut self,
        shaper: &mut Shaper,
        text: &str,
        font: FontId,
        size: PxSize,
        features: Features,
    ) -> Arc<ShapedRun> {
        let key = RunKey::new(text, font, size, features);

        if let Some(&slot) = self.index.get(&key) {
            let matches = self
                .slab
                .get(slot as usize)
                .is_some_and(|n| &*n.text == text);
            if matches {
                self.touch(slot);
                self.stats.hits += 1;
                // `touch` does not move the node out, so this index is still valid.
                if let Some(node) = self.slab.get(slot as usize) {
                    return Arc::clone(&node.run);
                }
            } else {
                // Same 64-bit hash, different string. Drop the old entry and shape afresh.
                self.stats.hash_collisions += 1;
                self.remove(slot);
            }
        }

        self.stats.misses += 1;
        self.stats.new_this_frame += 1;
        let run = Arc::new(shaper.shape(text, font, size, features));
        self.insert(key, text, Arc::clone(&run));
        run
    }

    /// Drop everything. Called when the density or text scale changes, since every cached
    /// run was shaped at the old size and none of them are reusable.
    pub fn clear(&mut self) {
        self.index.clear();
        self.slab.clear();
        self.free.clear();
        self.head = NIL;
        self.tail = NIL;
    }

    fn insert(&mut self, key: RunKey, text: &str, run: Arc<ShapedRun>) {
        while self.index.len() >= self.capacity && self.tail != NIL {
            let victim = self.tail;
            self.remove(victim);
            self.stats.evictions += 1;
        }

        let node = Node {
            key,
            text: text.into(),
            run,
            prev: NIL,
            next: self.head,
        };

        let slot = match self.free.pop() {
            Some(slot) => {
                if let Some(existing) = self.slab.get_mut(slot as usize) {
                    *existing = node;
                }
                slot
            }
            None => {
                self.slab.push(node);
                (self.slab.len() - 1) as u32
            }
        };

        if self.head != NIL {
            if let Some(old_head) = self.slab.get_mut(self.head as usize) {
                old_head.prev = slot;
            }
        }
        self.head = slot;
        if self.tail == NIL {
            self.tail = slot;
        }
        self.index.insert(key, slot);
    }

    fn remove(&mut self, slot: u32) {
        let Some(node) = self.slab.get(slot as usize) else {
            return;
        };
        let (prev, next, key) = (node.prev, node.next, node.key);

        if prev != NIL {
            if let Some(n) = self.slab.get_mut(prev as usize) {
                n.next = next;
            }
        } else {
            self.head = next;
        }
        if next != NIL {
            if let Some(n) = self.slab.get_mut(next as usize) {
                n.prev = prev;
            }
        } else {
            self.tail = prev;
        }

        self.index.remove(&key);
        self.free.push(slot);

        // Release the run and the string immediately. Holding them until the slot is
        // reused would make the cache's memory high-water mark its steady state.
        if let Some(n) = self.slab.get_mut(slot as usize) {
            n.run = Arc::new(ShapedRun::default());
            n.text = "".into();
            n.prev = NIL;
            n.next = NIL;
        }
    }

    fn touch(&mut self, slot: u32) {
        if self.head == slot {
            return;
        }
        let Some(node) = self.slab.get(slot as usize) else {
            return;
        };
        let (prev, next) = (node.prev, node.next);

        if prev != NIL {
            if let Some(n) = self.slab.get_mut(prev as usize) {
                n.next = next;
            }
        }
        if next != NIL {
            if let Some(n) = self.slab.get_mut(next as usize) {
                n.prev = prev;
            }
        } else {
            self.tail = prev;
        }

        let old_head = self.head;
        if let Some(n) = self.slab.get_mut(slot as usize) {
            n.prev = NIL;
            n.next = old_head;
        }
        if old_head != NIL {
            if let Some(n) = self.slab.get_mut(old_head as usize) {
                n.prev = slot;
            }
        }
        self.head = slot;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;
    use crate::fontdb::{FontDb, SystemFontDb};

    fn fixture() -> Option<(Shaper, FontId)> {
        let db = SystemFontDb::scan();
        if db.is_empty() {
            return None;
        }
        let ui = db.ui_font();
        Some((Shaper::new(Arc::new(db)), ui))
    }

    #[test]
    fn a_repeated_name_is_shaped_once() {
        let Some((mut shaper, ui)) = fixture() else {
            return;
        };
        let mut cache = ShapedRunCache::new(64);
        let size = PxSize::new(14.0);

        for _ in 0..10 {
            cache.get_or_shape(&mut shaper, "readme.md", ui, size, Features::default());
        }
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 9);
    }

    #[test]
    fn eviction_is_least_recently_used() {
        let Some((mut shaper, ui)) = fixture() else {
            return;
        };
        let mut cache = ShapedRunCache::new(2);
        let size = PxSize::new(14.0);
        let f = Features::default();

        cache.get_or_shape(&mut shaper, "a.txt", ui, size, f);
        cache.get_or_shape(&mut shaper, "b.txt", ui, size, f);
        // Re-touch "a" so "b" becomes the least recently used.
        cache.get_or_shape(&mut shaper, "a.txt", ui, size, f);
        cache.get_or_shape(&mut shaper, "c.txt", ui, size, f);

        assert_eq!(cache.len(), 2);
        let before = cache.stats().misses;
        cache.get_or_shape(&mut shaper, "a.txt", ui, size, f);
        assert_eq!(
            cache.stats().misses,
            before,
            "\"a\" should still be resident"
        );

        cache.get_or_shape(&mut shaper, "b.txt", ui, size, f);
        assert_eq!(
            cache.stats().misses,
            before + 1,
            "\"b\" should have been evicted"
        );
    }

    #[test]
    fn thrashing_does_not_leak_slots() {
        let Some((mut shaper, ui)) = fixture() else {
            return;
        };
        let mut cache = ShapedRunCache::new(8);
        let size = PxSize::new(14.0);

        for i in 0..500 {
            let name = format!("file-{i}.rs");
            cache.get_or_shape(&mut shaper, &name, ui, size, Features::default());
        }
        assert!(cache.len() <= 8);
        // The slab is allowed to reach capacity, never to grow past it.
        assert!(cache.slab.len() <= 8, "slab grew to {}", cache.slab.len());
        assert!(cache.stats().evictions >= 490);
    }

    #[test]
    fn per_frame_counter_resets() {
        let Some((mut shaper, ui)) = fixture() else {
            return;
        };
        let mut cache = ShapedRunCache::new(64);
        let size = PxSize::new(14.0);

        cache.begin_frame();
        cache.get_or_shape(&mut shaper, "x.md", ui, size, Features::default());
        assert_eq!(cache.stats().new_this_frame, 1);

        cache.begin_frame();
        assert_eq!(cache.stats().new_this_frame, 0);
        cache.get_or_shape(&mut shaper, "x.md", ui, size, Features::default());
        assert_eq!(cache.stats().new_this_frame, 0, "a hit is not new work");
    }

    #[test]
    fn clearing_on_size_change_drops_everything() {
        let Some((mut shaper, ui)) = fixture() else {
            return;
        };
        let mut cache = ShapedRunCache::new(64);
        cache.get_or_shape(
            &mut shaper,
            "a.txt",
            ui,
            PxSize::new(14.0),
            Features::default(),
        );
        cache.clear();
        assert!(cache.is_empty());
        cache.get_or_shape(
            &mut shaper,
            "a.txt",
            ui,
            PxSize::new(21.0),
            Features::default(),
        );
        assert_eq!(cache.len(), 1);
    }
}
