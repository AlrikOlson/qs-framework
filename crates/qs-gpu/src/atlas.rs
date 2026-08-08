//! The R8 coverage glyph atlas.
//!
//! # The eviction problem, stated precisely
//!
//! Research R8 flags this as the second-most-likely place M0 fails, and names the
//! constraint: a CJK corpus with tens of thousands of distinct glyphs will exceed atlas
//! capacity, and "eviction must not flicker rows still on screen".
//!
//! That sentence rules out the obvious implementation. Resetting the atlas when it fills
//! is trivial and wrong -- it invalidates glyphs that the *current* frame has already
//! emitted draw calls for, and the row they belong to renders as garbage for a frame. So
//! eviction here has one hard rule: **an entry used in the current frame is never
//! evicted.** When no entry is evictable, the working set genuinely exceeds the atlas, and
//! that is the cliff R8 exists to locate. It is counted in
//! [`AtlasStats::overflow_events`] and reported, not smoothed over.
//!
//! # Why CLOCK and not true LRU
//!
//! Eviction runs inside a frame that is already over budget, so it must be O(1) amortized.
//! A true LRU needs an intrusive list; CLOCK (FIFO with a second chance) needs a queue and
//! a bit, approximates LRU closely for this access pattern, and cannot degrade into a scan.
//! Under thrash the difference between LRU and CLOCK is a few percent of hit rate; the
//! difference between either and an O(n) scan is the frame budget.
//!
//! # Why a per-frame upload bound
//!
//! Rasterizing is CPU work and uploading is bandwidth. A fling that reveals two thousand
//! new glyphs at once would spend the whole frame on them. The bound spreads that across
//! frames: some glyphs are missing for a frame or two during a violent scroll, which is
//! far less visible than a 40 ms stall, and it is recorded either way.

use std::collections::{HashMap, VecDeque};

use qs_text::{FontDb, GlyphKey, GlyphRaster, RasterizedGlyph};

use crate::icon::IconKey;

/// What one atlas entry is a picture of.
///
/// The atlas is a cache of R8 coverage bitmaps. Nothing in the shelf packing, the CLOCK
/// eviction, the never-evict-in-use rule or the per-frame upload bound is specific to
/// fonts -- only [`GlyphAtlas::get_or_insert`] is, and it is one thin wrapper over
/// [`GlyphAtlas::get_or_render`]. Widening the *key* is therefore all it took to give icons
/// every guarantee this module makes, and it is why an icon needs no second texture, no
/// second bind group, and no tier-specific code path.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum AtlasKey {
    Glyph(GlyphKey),
    Icon(IconKey),
}

impl From<GlyphKey> for AtlasKey {
    fn from(key: GlyphKey) -> Self {
        Self::Glyph(key)
    }
}

impl From<IconKey> for AtlasKey {
    fn from(key: IconKey) -> Self {
        Self::Icon(key)
    }
}

/// Glyph heights are rounded up to a multiple of this before choosing a shelf. Quantizing
/// means glyphs of similar height share shelves and their slots are interchangeable when
/// freed; without it every shelf holds one height and the atlas fragments immediately.
const SHELF_QUANTUM: u32 = 4;

/// One pixel of transparent padding around each glyph, so bilinear sampling at a
/// fractional position cannot pick up the neighbouring glyph's ink.
const GUTTER: u32 = 1;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AtlasEntry {
    /// `[u0, v0, u1, v1]`, normalized.
    pub uv: [f32; 4],
    /// Offset from the pen position to the bitmap's left edge, in pixels.
    pub left: i32,
    /// Offset from the baseline to the bitmap's top edge, positive upward.
    pub top: i32,
    pub width: u32,
    pub height: u32,
    /// The atlas generation this entry belongs to.
    ///
    /// A caller that holds an entry across frames must compare this against
    /// [`GlyphAtlas::generation`] before using the UVs. Nothing in M0 does -- the row
    /// builder re-queries every frame -- but the field is the difference between "we
    /// happen not to have that bug" and "that bug is detectable".
    pub generation: u64,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct AtlasStats {
    /// Glyphs rasterized and uploaded since the last [`GlyphAtlas::begin_frame`].
    pub uploaded_this_frame: u32,
    /// Glyphs wanted this frame that were deferred by the per-frame upload bound.
    pub deferred_this_frame: u32,
    pub evictions: u64,
    /// Times the atlas could not make room because every resident glyph was in use this
    /// frame. **This is the R8 cliff.** Non-zero means the visible working set does not
    /// fit, and the correct response is to widen the atlas or narrow the corpus, not to
    /// tune the eviction policy.
    pub overflow_events: u64,
    pub hits: u64,
    pub misses: u64,
    pub resident: u32,
    /// Fraction of atlas area currently allocated, 0.0..=1.0.
    pub occupancy: f32,
}

#[derive(Debug)]
struct Shelf {
    y: u32,
    height: u32,
    cursor: u32,
    /// `(x, width)` slots freed by eviction, reusable by any glyph of this shelf's class.
    free: Vec<(u32, u32)>,
}

#[derive(Debug)]
struct Resident {
    entry: AtlasEntry,
    shelf: usize,
    x: u32,
    slot_width: u32,
    last_used_frame: u64,
    /// CLOCK's second-chance bit.
    referenced: bool,
}

/// CPU-side atlas state. The GPU texture is owned by the caller and updated through
/// [`GlyphAtlas::take_uploads`], which keeps this type free of any `wgpu` dependency and
/// therefore testable without a device -- and usable unchanged by the CPU rasterizer,
/// which is what makes RP-2 (all tiers consume the same draw lists) true for text as well.
#[derive(Debug)]
pub struct GlyphAtlas {
    size: u32,
    shelves: Vec<Shelf>,
    next_shelf_y: u32,
    resident: HashMap<AtlasKey, Resident>,
    clock: VecDeque<AtlasKey>,
    generation: u64,
    frame: u64,
    max_uploads_per_frame: u32,
    stats: AtlasStats,
    pending: Vec<PendingUpload>,
    allocated_area: u64,
}

/// A bitmap that needs to reach the GPU texture.
#[derive(Clone, Debug)]
pub struct PendingUpload {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub coverage: Vec<u8>,
}

impl GlyphAtlas {
    /// `size` is the square texture edge in pixels. 2048 holds roughly 12,000 Latin glyphs
    /// at UI sizes -- far more than any viewport needs -- and roughly 2,000 CJK glyphs at
    /// 28 px, which is where the mixed-scripts corpus starts to bite.
    pub fn new(size: u32, max_uploads_per_frame: u32) -> Self {
        Self {
            size: size.max(256),
            shelves: Vec::new(),
            next_shelf_y: 0,
            resident: HashMap::new(),
            clock: VecDeque::new(),
            generation: 0,
            frame: 0,
            max_uploads_per_frame: max_uploads_per_frame.max(1),
            stats: AtlasStats::default(),
            pending: Vec::new(),
            allocated_area: 0,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn stats(&self) -> AtlasStats {
        let mut stats = self.stats;
        stats.resident = self.resident.len() as u32;
        stats.occupancy = self.allocated_area as f32 / (self.size as f32 * self.size as f32);
        stats
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.stats.uploaded_this_frame = 0;
        self.stats.deferred_this_frame = 0;
        self.pending.clear();
    }

    /// Hand the frame's uploads to the caller, which copies them into the GPU texture.
    pub fn take_uploads(&mut self) -> Vec<PendingUpload> {
        std::mem::take(&mut self.pending)
    }

    /// Whether `key` is known to have no ink -- a space, or a mark that rendered empty.
    ///
    /// Callers need this to tell the two harmless `None`s from [`GlyphAtlas::get_or_insert`]
    /// apart from the harmful one. A blank is text that is *supposed* to be invisible;
    /// counting it as a missing glyph would make the SC-006 signal fire on every line that
    /// contains a space, which is every line.
    pub fn is_blank(&self, key: impl Into<AtlasKey>) -> bool {
        self.resident
            .get(&key.into())
            .is_some_and(|res| res.entry.width == 0 || res.entry.height == 0)
    }

    /// Look up a glyph, rasterizing and inserting it if it is not resident.
    ///
    /// `None` means the glyph is not available *this frame*: either it has no ink, the
    /// per-frame upload bound was reached, or the atlas overflowed. All three are
    /// survivable and all three are counted.
    pub fn get_or_insert(
        &mut self,
        db: &dyn FontDb,
        raster: &mut GlyphRaster,
        key: GlyphKey,
    ) -> Option<AtlasEntry> {
        // Blanks are memoized here and *only* here. A glyph with no ink is a space, which is
        // text that is supposed to be invisible; memoizing costs one rasterization ever
        // instead of one per frame. Nothing else in the atlas has a legitimate blank, which
        // is why `blank_is_ok` is a parameter of this call rather than a property of the
        // atlas -- see `crate::icon`.
        self.get_or_render(key, true, |slot| raster.rasterize(db, slot))
    }

    /// Look up any atlas entry, producing its coverage bitmap on a miss.
    ///
    /// `render` is called at most once per key per process, and only after the per-frame
    /// upload bound has been checked -- so a caller whose rasterization is expensive does
    /// not pay for it on a frame that could not have uploaded the result anyway.
    ///
    /// `blank_is_ok` decides what an inkless bitmap means. For glyphs it is a space and gets
    /// memoized, so [`GlyphAtlas::is_blank`] can tell callers to stop counting it as a
    /// missing glyph. For icons it is a bug, and refusing to memoize keeps it visible.
    pub fn get_or_render<K>(
        &mut self,
        key: K,
        blank_is_ok: bool,
        render: impl FnOnce(K) -> Option<RasterizedGlyph>,
    ) -> Option<AtlasEntry>
    where
        K: Into<AtlasKey> + Copy,
    {
        let atlas_key = key.into();
        if let Some(res) = self.resident.get_mut(&atlas_key) {
            res.last_used_frame = self.frame;
            res.referenced = true;
            self.stats.hits += 1;
            return Some(res.entry);
        }

        self.stats.misses += 1;

        if self.stats.uploaded_this_frame >= self.max_uploads_per_frame {
            // Spread the work rather than blowing the budget. The entry will be picked up
            // next frame; see the module docs.
            self.stats.deferred_this_frame += 1;
            return None;
        }

        let bitmap = render(key)?;
        if bitmap.is_blank() {
            if blank_is_ok {
                self.insert_blank(atlas_key);
            }
            return None;
        }

        self.insert(atlas_key, &bitmap)
    }

    fn insert_blank(&mut self, key: AtlasKey) {
        let entry = AtlasEntry {
            uv: [0.0; 4],
            left: 0,
            top: 0,
            width: 0,
            height: 0,
            generation: self.generation,
        };
        self.resident.insert(
            key,
            Resident {
                entry,
                shelf: usize::MAX,
                x: 0,
                slot_width: 0,
                last_used_frame: self.frame,
                referenced: true,
            },
        );
        self.clock.push_back(key);
    }

    fn insert(&mut self, key: AtlasKey, bitmap: &RasterizedGlyph) -> Option<AtlasEntry> {
        let padded_w = bitmap.width + GUTTER * 2;
        let padded_h = bitmap.height + GUTTER * 2;
        if padded_w > self.size || padded_h > self.size {
            // A single glyph larger than the atlas. qs-text refuses sizes above
            // MAX_ATLAS_PX, so reaching this means the atlas was configured too small.
            self.stats.overflow_events += 1;
            return None;
        }

        let (shelf_index, x) = match self.allocate(padded_w, padded_h) {
            Some(slot) => slot,
            None => {
                if !self.evict_until_room(padded_w, padded_h) {
                    // Every resident glyph is in use this frame. This is the cliff.
                    self.stats.overflow_events += 1;
                    return None;
                }
                self.allocate(padded_w, padded_h)?
            }
        };

        let shelf_y = self.shelves.get(shelf_index)?.y;
        let gx = x + GUTTER;
        let gy = shelf_y + GUTTER;

        let inv = 1.0 / self.size as f32;
        let entry = AtlasEntry {
            uv: [
                gx as f32 * inv,
                gy as f32 * inv,
                (gx + bitmap.width) as f32 * inv,
                (gy + bitmap.height) as f32 * inv,
            ],
            left: bitmap.left,
            top: bitmap.top,
            width: bitmap.width,
            height: bitmap.height,
            generation: self.generation,
        };

        self.pending.push(PendingUpload {
            x: gx,
            y: gy,
            width: bitmap.width,
            height: bitmap.height,
            coverage: bitmap.coverage.clone(),
        });
        self.stats.uploaded_this_frame += 1;
        self.allocated_area += u64::from(padded_w) * u64::from(padded_h);

        self.resident.insert(
            key,
            Resident {
                entry,
                shelf: shelf_index,
                x,
                slot_width: padded_w,
                last_used_frame: self.frame,
                referenced: true,
            },
        );
        self.clock.push_back(key);
        Some(entry)
    }

    /// Find or open a shelf with room for `w x h`. Returns `(shelf index, x)`.
    fn allocate(&mut self, w: u32, h: u32) -> Option<(usize, u32)> {
        let class = h.div_ceil(SHELF_QUANTUM) * SHELF_QUANTUM;

        for (index, shelf) in self.shelves.iter_mut().enumerate() {
            if shelf.height != class {
                continue;
            }
            // Prefer a freed slot: reusing one keeps the shelf from growing past its
            // cursor while holes accumulate behind it.
            if let Some(pos) = shelf.free.iter().position(|&(_, fw)| fw >= w) {
                let (x, _) = shelf.free.swap_remove(pos);
                return Some((index, x));
            }
            if shelf.cursor + w <= self.size {
                let x = shelf.cursor;
                shelf.cursor += w;
                return Some((index, x));
            }
        }

        if self.next_shelf_y + class <= self.size {
            let y = self.next_shelf_y;
            self.next_shelf_y += class;
            self.shelves.push(Shelf {
                y,
                height: class,
                cursor: w,
                free: Vec::new(),
            });
            return Some((self.shelves.len() - 1, 0));
        }

        None
    }

    /// Free entries until `w x h` fits, never touching one used this frame.
    ///
    /// Returns `false` when nothing is evictable.
    fn evict_until_room(&mut self, w: u32, h: u32) -> bool {
        let class = h.div_ceil(SHELF_QUANTUM) * SHELF_QUANTUM;
        // Bounded by the queue length: one full sweep gives every entry its second chance,
        // and a second sweep evicts. Anything still unevictable after that is in use.
        let budget = self.clock.len().saturating_mul(2).max(1);

        for _ in 0..budget {
            let Some(key) = self.clock.pop_front() else {
                return false;
            };
            let Some(res) = self.resident.get_mut(&key) else {
                continue; // already gone
            };

            if res.last_used_frame == self.frame {
                // In use right now. Untouchable -- this is the rule that prevents the
                // flicker R8 names. Put it back and move on.
                self.clock.push_back(key);
                continue;
            }
            if res.referenced {
                // Second chance.
                res.referenced = false;
                self.clock.push_back(key);
                continue;
            }

            let (shelf_index, x, slot_width) = (res.shelf, res.x, res.slot_width);
            let shelf_height = self.shelves.get(shelf_index).map(|s| s.height);
            self.resident.remove(&key);
            self.stats.evictions += 1;

            if let (Some(shelf), true) =
                (self.shelves.get_mut(shelf_index), shelf_index != usize::MAX)
            {
                shelf.free.push((x, slot_width));
                self.allocated_area = self
                    .allocated_area
                    .saturating_sub(u64::from(slot_width) * u64::from(shelf.height));
            }

            // Did that open a slot the caller can use?
            if shelf_height == Some(class) && slot_width >= w {
                return true;
            }
            if self.allocate_probe(w, class) {
                return true;
            }
        }
        false
    }

    /// Non-mutating check for whether `allocate` would now succeed.
    fn allocate_probe(&self, w: u32, class: u32) -> bool {
        self.shelves.iter().any(|s| {
            s.height == class
                && (s.free.iter().any(|&(_, fw)| fw >= w) || s.cursor + w <= self.size)
        }) || self.next_shelf_y + class <= self.size
    }

    /// Drop everything and start over. Bumps the generation so any cached entry is
    /// detectably stale. Called when the text size changes -- every glyph in the atlas was
    /// rasterized at the old size and none of them are reusable.
    pub fn clear(&mut self) {
        self.shelves.clear();
        self.next_shelf_y = 0;
        self.resident.clear();
        self.clock.clear();
        self.pending.clear();
        self.allocated_area = 0;
        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;
    use qs_text::{Features, FontId, PxSize, Shaper, SystemFontDb};
    use std::sync::Arc;

    /// `(font database, rasterizer, glyphs to atlas)`.
    type Fixture = (Arc<dyn FontDb>, GlyphRaster, Vec<(u16, FontId)>);

    fn fixture() -> Option<Fixture> {
        let db = SystemFontDb::scan();
        if db.is_empty() {
            return None;
        }
        let ui = db.ui_font();
        let db: Arc<dyn FontDb> = Arc::new(db);
        let mut shaper = Shaper::new(Arc::clone(&db));
        let run = shaper.shape(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            ui,
            PxSize::new(16.0),
            Features::default(),
        );
        let glyphs: Vec<_> = run.glyphs.iter().map(|g| (g.glyph_id, g.font)).collect();
        Some((db, GlyphRaster::new(), glyphs))
    }

    fn key(font: FontId, id: u16) -> GlyphKey {
        GlyphKey {
            font,
            glyph_id: id,
            size: PxSize::new(16.0),
            subpixel: 0,
        }
    }

    #[test]
    fn a_repeated_glyph_is_uploaded_once() {
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        let (id, font) = glyphs[0];
        let mut atlas = GlyphAtlas::new(512, 256);

        atlas.begin_frame();
        let first = atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id));
        assert!(first.is_some());
        assert_eq!(atlas.stats().uploaded_this_frame, 1);

        let second = atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id));
        assert_eq!(first, second);
        assert_eq!(
            atlas.stats().uploaded_this_frame,
            1,
            "a hit must not re-upload"
        );
    }

    #[test]
    fn uvs_stay_inside_the_texture_and_respect_the_gutter() {
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        let mut atlas = GlyphAtlas::new(512, 4096);
        atlas.begin_frame();

        for &(id, font) in &glyphs {
            let Some(entry) = atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id)) else {
                continue;
            };
            for c in entry.uv {
                assert!((0.0..=1.0).contains(&c), "uv {c} outside the texture");
            }
            assert!(entry.uv[0] < entry.uv[2] && entry.uv[1] < entry.uv[3]);
            // The gutter means no glyph starts at the very edge.
            assert!(entry.uv[0] > 0.0 && entry.uv[1] > 0.0);
        }
    }

    #[test]
    fn uploaded_regions_never_overlap() {
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        let mut atlas = GlyphAtlas::new(512, 4096);
        atlas.begin_frame();
        for &(id, font) in &glyphs {
            atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id));
        }
        let uploads = atlas.take_uploads();

        for (i, a) in uploads.iter().enumerate() {
            for b in uploads.iter().skip(i + 1) {
                let disjoint = a.x + a.width <= b.x
                    || b.x + b.width <= a.x
                    || a.y + a.height <= b.y
                    || b.y + b.height <= a.y;
                assert!(
                    disjoint,
                    "glyph regions {a:?} and {b:?} overlap -- one would overwrite the other"
                );
            }
        }
    }

    #[test]
    fn the_per_frame_upload_bound_is_honoured() {
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        if glyphs.len() < 10 {
            return;
        }
        let mut atlas = GlyphAtlas::new(1024, 3);
        atlas.begin_frame();
        for &(id, font) in &glyphs {
            atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id));
        }
        assert_eq!(atlas.stats().uploaded_this_frame, 3);
        assert!(atlas.stats().deferred_this_frame > 0);

        // The deferred work must actually arrive on a later frame, not be dropped.
        atlas.begin_frame();
        for &(id, font) in &glyphs {
            atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id));
        }
        assert_eq!(atlas.stats().uploaded_this_frame, 3);
    }

    #[test]
    fn a_glyph_used_this_frame_is_never_evicted() {
        // The R8 guarantee. A tiny atlas guarantees eviction pressure; every glyph is
        // touched in the same frame, so the atlas must overflow rather than evict.
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        if glyphs.len() < 20 {
            return;
        }
        let mut atlas = GlyphAtlas::new(256, 4096);
        atlas.begin_frame();

        let mut live = Vec::new();
        for &(id, font) in &glyphs {
            if let Some(entry) = atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id)) {
                live.push((key(font, id), entry));
            }
        }
        // Everything that was handed out this frame must still resolve to the same slot.
        for (k, expected) in &live {
            let now = atlas.get_or_insert(db.as_ref(), &mut raster, *k);
            assert_eq!(
                now.as_ref(),
                Some(expected),
                "a glyph in use this frame moved or was evicted"
            );
        }
    }

    #[test]
    fn eviction_reclaims_space_across_frames() {
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        if glyphs.len() < 20 {
            return;
        }
        let mut atlas = GlyphAtlas::new(256, 4096);

        // Ten frames, each touching a different slice, so earlier glyphs age out.
        for frame in 0..10 {
            atlas.begin_frame();
            let start = (frame * 3) % glyphs.len();
            for &(id, font) in glyphs.iter().cycle().skip(start).take(8) {
                atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id));
            }
        }
        let stats = atlas.stats();
        assert!(
            stats.resident > 0,
            "the atlas emptied itself instead of recycling"
        );
        assert!(stats.occupancy <= 1.0);
    }

    #[test]
    fn clearing_bumps_the_generation_so_stale_entries_are_detectable() {
        let mut atlas = GlyphAtlas::new(256, 16);
        let before = atlas.generation();
        atlas.clear();
        assert_eq!(atlas.generation(), before + 1);
    }
}
