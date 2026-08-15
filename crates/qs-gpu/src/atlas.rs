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
//!
//! # How the bound is spent, and why it is not first-come
//!
//! A bound alone says how much may be uploaded, not *what*. Spending it on whoever asked
//! first means spending it in draw order, and draw order is top-to-bottom: a cold frame of
//! realistic filenames wants around 73 distinct glyphs against the CPU tier's 64, so the
//! first rows got their text and the last rows rendered shredded. That is not a budget
//! problem -- the budget is nearly enough -- it is an *allocation* problem, and the atlas
//! had no vocabulary for it. It has two now.
//!
//! **[`UploadClass`] separates the bounded from the unbounded.** Icons and emblems are
//! [`UploadClass::Structural`]: there are at most a dozen of them in any frame, each one
//! serves every row of its kind, and once resident they never cost anything again. Glyphs
//! are [`UploadClass::Content`]: there is no bound on how many distinct ones a corpus can
//! want, so they are the thing that has to be rationed. Rationing them against the *same*
//! counter meant a page of text could starve the folder icon, which is why the row builder
//! grew a hand-written prepass that resolved icons before asking for a single glyph. The
//! two classes have separate bounds now, so that ordering is a rule here rather than a
//! property of which loop a caller happened to run first.
//!
//! **Within content, admission is demand-ordered.** [`GlyphAtlas::want`] records that a
//! draw wants a key without rasterizing it; [`GlyphAtlas::admit_demanded`] then spends the
//! content bound on the keys the frame wanted *most*. A letter in forty filenames is
//! admitted before a letter in one, so the first cold frame reads as text with a few
//! characters missing rather than as the top half of a list. The caller pays for this by
//! deferring the instances it could not emit until after `admit_demanded` -- see
//! `qs_ui::ListRenderer::flush_text`, which is the only correct way to use `want`.
//!
//! # Two pages, one atlas
//!
//! A coverage mask has one channel and a picture has four, so they cannot share texels.
//! They share everything else. [`AtlasPage::Coverage`] is the R8 texture this module was
//! built around; [`AtlasPage::Colour`] is an RGBA8 texture beside it, and *beside* is the
//! only thing about it that is separate: one [`GlyphAtlas`] owns both, one `resident` map
//! addresses both, one CLOCK queue sweeps both, one [`AtlasStats`] reports both, and one
//! [`ATLAS_CAP_BYTES`] is checked over both in [`AtlasBudget`].
//!
//! **The split is static, and that is stronger than a shared pool.** The failure worth
//! preventing is a folder of photographs evicting the file list's glyphs -- the user sees
//! scrolling get slow, a long way from anything they did. A byte pool shared between
//! pictures and text permits exactly that. Two fixed pages make it unreachable: an image
//! cannot occupy a coverage slot, so it cannot take one. What the single cap buys instead
//! is that the two page sizes are chosen together, once, against one number, rather than
//! one of them being added later without anybody re-deriving the total.
//!
//! **Eviction is one policy with a page filter.** [`GlyphAtlas::evict_until_room`] sweeps
//! the same CLOCK queue with the same second chance and the same never-evict-in-use rule;
//! a resident on the wrong page is put back untouched, exactly as an in-use one is. There
//! is no second policy to keep in step with the first.

use std::collections::{HashMap, VecDeque};

use qs_text::{FontDb, GlyphKey, GlyphRaster, RasterizedGlyph};

use crate::icon::IconKey;

/// Which texture an entry lives in.
///
/// Not a property of the cache, a property of the *pixels*: one channel of coverage and
/// four channels of colour need different formats and therefore different textures. Every
/// other guarantee this module makes is indifferent to which page a key landed on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub enum AtlasPage {
    /// R8Unorm. Glyph coverage and icon coverage.
    #[default]
    Coverage,
    /// RGBA8Unorm, non-premultiplied on the way in and premultiplied on the way out --
    /// see [`RgbaImage`]. Thumbnails and any other picture.
    Colour,
}

impl AtlasPage {
    /// Bytes per texel on this page. The only number that differs between them.
    #[must_use]
    pub const fn bytes_per_texel(self) -> u64 {
        match self {
            Self::Coverage => 1,
            Self::Colour => 4,
        }
    }
}

/// Identity of one colour image in the atlas.
///
/// Opaque on purpose. The atlas does not decode anything and has no opinion about what an
/// image *is*; it needs a stable identity and nothing else, so the producer -- today
/// `thumbnail-cache-proxy`, when it exists -- supplies one and owns what it means. Making
/// this a path or a `(path, mtime)` here would put file identity in the renderer, and
/// `qs_vfs::Name` is already the one answer to that question.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ImageKey(pub u64);

/// A decoded picture on its way into the colour page.
///
/// **Straight alpha, sRGB, exactly as a decoder produces it.** The pipeline blends
/// `One / OneMinusSrcAlpha`, so the multiply has to happen somewhere, and admission is the
/// tempting place -- once per image ever instead of once per fragment. It is the wrong
/// place, and the reason is the colour space rather than the cost.
///
/// The colour page is `Rgba8UnormSrgb`, so sampling it decodes RGB through the sRGB curve
/// in hardware and leaves alpha alone. Premultiplying the stored bytes would make the
/// sample `srgb_to_linear(c * a)`, and what premultiplied blending needs is
/// `srgb_to_linear(c) * a`. Those are not equal and the error is largest exactly where
/// thumbnails have soft edges. So the bytes stay straight, the hardware linearizes, and the
/// multiply happens after it -- in the fragment shader on the GPU tier and in
/// `CpuRasterizer::draw_image` on the CPU tier, which is why both do it in the same order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, RGBA, **straight** alpha, sRGB-encoded.
    pub rgba: Vec<u8>,
}

impl RgbaImage {
    /// Whether this image is too malformed to admit: no pixels, or a buffer that does not
    /// match the dimensions it claims.
    ///
    /// Checked at admission rather than trusted, because the caller is a decoder and a
    /// decoder's output is the least trustworthy thing that reaches this module.
    #[must_use]
    pub fn is_malformed(&self) -> bool {
        self.width == 0
            || self.height == 0
            || self.rgba.len() != (self.width as usize) * (self.height as usize) * 4
    }
}

/// The whole atlas's memory ceiling, in bytes. SDD 7.3.
///
/// **One number over both pages**, which is the point: a colour cache sized on its own
/// merits is a colour cache nobody added to the glyph budget, and the total is what the
/// host actually pays. [`AtlasBudget::fits`] is the one place it is checked.
pub const ATLAS_CAP_BYTES: u64 = 96 * 1024 * 1024;

/// The default colour page edge, in texels.
///
/// 1024 x 1024 x 4 is 4 MiB, which holds sixty-four 128px thumbnails or four 512px
/// previews. Deliberately smaller than the coverage page's usual 2048: the coverage page
/// serves every row on screen at once, and a colour page serves what is in the inspector
/// or the visible part of a gallery. Both together are 8 MiB against a 96 MiB cap, so the
/// headroom for a wider page later is real rather than notional.
pub const DEFAULT_COLOUR_PAGE: u32 = 1024;

/// How the [`ATLAS_CAP_BYTES`] ceiling is spent across the two pages.
///
/// A type rather than two arguments, because the thing worth checking is the *sum* and a
/// sum has to be computed somewhere that sees both halves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AtlasBudget {
    pub coverage_size: u32,
    pub colour_size: u32,
}

impl AtlasBudget {
    /// Bytes this configuration would occupy on the GPU, both pages together.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        let coverage = (self.coverage_size as u64) * (self.coverage_size as u64);
        let colour = (self.colour_size as u64) * (self.colour_size as u64) * 4;
        coverage + colour
    }

    /// Whether both pages together fit under [`ATLAS_CAP_BYTES`].
    #[must_use]
    pub const fn fits(self) -> bool {
        self.bytes() <= ATLAS_CAP_BYTES
    }
}

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
    Image(ImageKey),
}

impl AtlasKey {
    /// Which page this key's pixels belong on.
    ///
    /// Derived from the key rather than stored beside it, so a caller cannot ask for a
    /// glyph on the colour page or an image on the coverage page. The two channel counts
    /// are a fact about the kind of picture, not a choice at the call site.
    #[must_use]
    pub const fn page(self) -> AtlasPage {
        match self {
            Self::Glyph(_) | Self::Icon(_) => AtlasPage::Coverage,
            Self::Image(_) => AtlasPage::Colour,
        }
    }
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

impl From<ImageKey> for AtlasKey {
    fn from(key: ImageKey) -> Self {
        Self::Image(key)
    }
}

/// Which bound a request is rationed against, and why the two are not one number.
///
/// The distinction is *how many distinct entries the class can ever want in one frame*, and
/// it is the whole argument for the split. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum UploadClass {
    /// Bounded by construction: the icon kinds and emblems, at most a dozen, each serving
    /// every row that uses it. Drawn from [`STRUCTURAL_UPLOADS_PER_FRAME`], a separate and
    /// deliberately small bound, so no amount of text can starve them.
    Structural,
    /// Unbounded by construction: glyphs. A corpus decides how many distinct ones a frame
    /// wants, so this is the class the per-frame bound exists to ration.
    #[default]
    Content,
    /// Pictures, rationed against [`IMAGE_UPLOADS_PER_FRAME`].
    ///
    /// A third class for the same reason there is a second one, and the argument is the
    /// module's own: rationing pictures against the text counter lets one starve the other,
    /// and the direction that matters is a folder of photographs starving the filenames
    /// beside them. What is different about images is the *unit cost* rather than the
    /// count -- one 256px thumbnail is 256 KiB, which is more bytes than every glyph in a
    /// screenful of text -- so this bound is small for a reason the other two are not.
    Image,
}

/// The structural bound, in entries per frame.
///
/// Nine icon kinds, two emblems, seven state icons and two chevrons is twenty, so twenty-one is
/// one frame's worst case with a slot to spare. It is a constant rather than tier configuration
/// because it is a fact about `crate::icon` and not about the machine: a tier that could
/// afford more would have nothing to spend it on. Twenty-one 20px coverage masks is under 9 KB,
/// which is why giving structural entries their own bound costs less than taking twenty-one
/// uploads away from text would.
///
/// It moved from twelve when `harness-adapters` added [`crate::icon::StateIcon`], again when
/// `session-persistence` added [`crate::icon::StateIcon::Remembered`], and again when the
/// text-as-layout sweep added [`crate::icon::Chevron`] — and the worst case is real rather than
/// theoretical: the all-sessions overview can put every state on screen at once, over a file
/// list already drawing every kind, under a breadcrumb bar drawing chevrons. The spare slot is
/// kept deliberately: a bound that exactly equals the shape count passes its own test on the day
/// it is written and starts dropping one icon a frame the day a shape is added.
pub const STRUCTURAL_UPLOADS_PER_FRAME: u32 = 21;

/// The image bound, in entries per frame.
///
/// Two, and the number comes from bytes rather than from entries. A 256 x 256 thumbnail is
/// 256 KiB of upload; two is half a megabyte a frame, which is the same order as the
/// content bound spends on a violent scroll. Four would double the worst frame's bandwidth
/// to serve a case -- more than two *newly visible* pictures in one frame -- that only
/// arises on the first frame of a gallery, where being two frames behind is invisible and
/// a dropped frame is not.
pub const IMAGE_UPLOADS_PER_FRAME: u32 = 2;

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
    /// Glyphs rasterized and uploaded since the last [`GlyphAtlas::begin_frame`]. Counts
    /// both classes, because it answers "what did this frame cost".
    pub uploaded_this_frame: u32,
    /// Glyphs wanted this frame that were deferred by the per-frame upload bound.
    pub deferred_this_frame: u32,
    /// Distinct content keys this frame wanted but did not have. The denominator of the
    /// convergence question: `deferred_this_frame` says how many draws went without,
    /// this says how many *entries* the frame is still short of, which is what decides how
    /// many more frames it takes to converge.
    pub demanded_this_frame: u32,
    /// Structural entries uploaded since the last [`GlyphAtlas::begin_frame`], against
    /// [`STRUCTURAL_UPLOADS_PER_FRAME`]. Reported separately because a structural
    /// deferral means [`crate::icon`] wanted more shapes in one frame than exist, which is
    /// a different bug from text not keeping up.
    pub structural_this_frame: u32,
    /// Content entries uploaded since the last [`GlyphAtlas::begin_frame`], against the
    /// configured per-frame bound. This is the number the bound actually gates;
    /// [`AtlasStats::uploaded_this_frame`] is the two classes together.
    pub content_this_frame: u32,
    /// Image entries uploaded since the last [`GlyphAtlas::begin_frame`], against
    /// [`IMAGE_UPLOADS_PER_FRAME`].
    pub image_this_frame: u32,
    pub evictions: u64,
    /// Times the atlas could not make room because every resident glyph was in use this
    /// frame. **This is the R8 cliff.** Non-zero means the visible working set does not
    /// fit, and the correct response is to widen the atlas or narrow the corpus, not to
    /// tune the eviction policy.
    ///
    /// It counts *contention only*. A bitmap that is simply larger than its page is
    /// [`AtlasStats::rejected_oversize`] instead, and keeping the two apart is what lets
    /// this number keep the meaning its name claims.
    pub overflow_events: u64,
    /// Bitmaps refused at admission because they are larger than their page could ever
    /// hold, whatever it was holding at the time.
    ///
    /// **Not an overflow.** Overflow means the visible working set no longer fits, which
    /// is a statement about the corpus and the atlas size together; a 4000 px photograph
    /// is a statement about one file. Counting them in one number made a single oversized
    /// image indistinguishable from the thrash cliff `atlas-thrash-sweep` exists to find,
    /// which is a bad trade at any ratio -- the responses are "resize the source" and
    /// "widen the atlas", and they are not interchangeable.
    ///
    /// Glyphs cannot reach this in practice: `qs_text::raster::MAX_ATLAS_PX` refuses the
    /// sizes that would. Images can and will, which is why it exists now.
    pub rejected_oversize: u64,
    /// Images refused at admission because their buffer did not match their dimensions.
    /// A decoder bug, and it is counted rather than silently skipped.
    pub rejected_malformed: u64,
    pub hits: u64,
    pub misses: u64,
    pub resident: u32,
    /// Fraction of the **coverage** page's area currently allocated, 0.0..=1.0.
    ///
    /// Still the coverage page alone, because every consumer of this number is asking
    /// about text. The colour page is [`AtlasStats::colour_occupancy`]; one blended figure
    /// would answer neither question.
    pub occupancy: f32,
    /// Fraction of the **colour** page's area currently allocated, 0.0..=1.0.
    pub colour_occupancy: f32,
    /// GPU bytes both pages occupy together, against [`ATLAS_CAP_BYTES`].
    ///
    /// The *reserved* size rather than the used one: a texture costs its full extent the
    /// moment it is created, so this is what the host actually pays and what the cap is
    /// about. Occupancy is the question about how much of it is doing work.
    pub bytes_reserved: u64,
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
    /// Which page `shelf` indexes into. Stored rather than re-derived from the key,
    /// because eviction walks the CLOCK queue and holds a `Resident` before it has any
    /// reason to look at what kind of key it belongs to.
    page: AtlasPage,
    shelf: usize,
    x: u32,
    slot_width: u32,
    last_used_frame: u64,
    /// CLOCK's second-chance bit.
    referenced: bool,
}

/// One texture's worth of shelf-packing state.
///
/// Extracted so the coverage and colour pages run the *same* allocator rather than two
/// that have to be kept in step. Nothing here knows which page it is; the format lives on
/// [`AtlasPage`] and the only thing the allocator needs is an edge length in texels.
#[derive(Debug)]
struct Page {
    size: u32,
    shelves: Vec<Shelf>,
    next_shelf_y: u32,
    allocated_area: u64,
}

impl Page {
    fn new(size: u32) -> Self {
        Self {
            size,
            shelves: Vec::new(),
            next_shelf_y: 0,
            allocated_area: 0,
        }
    }

    fn clear(&mut self) {
        self.shelves.clear();
        self.next_shelf_y = 0;
        self.allocated_area = 0;
    }

    fn occupancy(&self) -> f32 {
        self.allocated_area as f32 / (self.size as f32 * self.size as f32)
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

    /// Non-mutating check for whether [`Page::allocate`] would now succeed.
    fn allocate_probe(&self, w: u32, class: u32) -> bool {
        self.shelves.iter().any(|s| {
            s.height == class
                && (s.free.iter().any(|&(_, fw)| fw >= w) || s.cursor + w <= self.size)
        }) || self.next_shelf_y + class <= self.size
    }
}

/// CPU-side atlas state. The GPU texture is owned by the caller and updated through
/// [`GlyphAtlas::take_uploads`], which keeps this type free of any `wgpu` dependency and
/// therefore testable without a device -- and usable unchanged by the CPU rasterizer,
/// which is what makes RP-2 (all tiers consume the same draw lists) true for text as well.
#[derive(Debug)]
pub struct GlyphAtlas {
    coverage: Page,
    colour: Page,
    resident: HashMap<AtlasKey, Resident>,
    clock: VecDeque<AtlasKey>,
    generation: u64,
    frame: u64,
    max_uploads_per_frame: u32,
    stats: AtlasStats,
    pending: Vec<PendingUpload>,
    /// Colour-page uploads, kept in their own list so the R8 path is untouched by
    /// construction rather than by inspection. A frame with no images produces an empty
    /// vector and spends nothing: `Vec::new` does not allocate.
    pending_images: Vec<PendingImage>,
    /// This frame's content demand: how many draws wanted each key the atlas did not have.
    /// Cleared rather than reallocated each frame -- steady-state frame building must not
    /// allocate, and the same reasoning applies here as to `DrawList::reset`.
    demand: HashMap<AtlasKey, u32>,
    /// Scratch for sorting `demand`, kept for its capacity for the same reason.
    ranked: Vec<(AtlasKey, u32)>,
}

/// A coverage bitmap that needs to reach the R8 page.
#[derive(Clone, Debug)]
pub struct PendingUpload {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub coverage: Vec<u8>,
}

/// A picture that needs to reach the RGBA8 page.
///
/// A separate type from [`PendingUpload`] rather than a page tag on it, and the reason is
/// criterion-shaped: with two lists, "a frame with no images costs exactly what it costs
/// today" is true because the R8 list never learns that images exist. A tagged single list
/// would make every existing upload site carry a field about a feature it has nothing to do
/// with, and would put a branch in the one loop that runs per glyph.
#[derive(Clone, Debug)]
pub struct PendingImage {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, RGBA, **straight** alpha and sRGB-encoded -- see
    /// [`RgbaImage`] for why the premultiply is downstream of here rather than upstream.
    pub rgba: Vec<u8>,
}

impl GlyphAtlas {
    /// `size` is the coverage page's square edge in pixels. 2048 holds roughly 12,000 Latin
    /// glyphs at UI sizes -- far more than any viewport needs -- and roughly 2,000 CJK
    /// glyphs at 28 px, which is where the mixed-scripts corpus starts to bite.
    ///
    /// The colour page gets [`DEFAULT_COLOUR_PAGE`]. Use [`GlyphAtlas::with_budget`] to say
    /// otherwise.
    pub fn new(size: u32, max_uploads_per_frame: u32) -> Self {
        Self::with_budget(
            AtlasBudget {
                coverage_size: size,
                colour_size: DEFAULT_COLOUR_PAGE,
            },
            max_uploads_per_frame,
        )
    }

    /// Both pages at once, against one cap.
    ///
    /// A budget over [`ATLAS_CAP_BYTES`] is **refused rather than clamped**: the caller
    /// asked for a specific amount of memory and silently giving them a smaller atlas is
    /// how a thrash cliff arrives with no cause anybody can find. `None` says so, and the
    /// only caller that can hit it is one choosing sizes from configuration.
    pub fn try_with_budget(budget: AtlasBudget, max_uploads_per_frame: u32) -> Option<Self> {
        if !budget.fits() {
            return None;
        }
        Some(Self::build(budget, max_uploads_per_frame))
    }

    /// [`GlyphAtlas::try_with_budget`] for a budget the caller already knows fits, falling
    /// back to the default pages if it does not.
    pub fn with_budget(budget: AtlasBudget, max_uploads_per_frame: u32) -> Self {
        let budget = if budget.fits() {
            budget
        } else {
            AtlasBudget {
                coverage_size: 2048,
                colour_size: DEFAULT_COLOUR_PAGE,
            }
        };
        Self::build(budget, max_uploads_per_frame)
    }

    fn build(budget: AtlasBudget, max_uploads_per_frame: u32) -> Self {
        let coverage_size = budget.coverage_size.max(256);
        let colour_size = budget.colour_size.max(256);
        Self {
            coverage: Page::new(coverage_size),
            colour: Page::new(colour_size),
            resident: HashMap::new(),
            clock: VecDeque::new(),
            generation: 0,
            frame: 0,
            max_uploads_per_frame: max_uploads_per_frame.max(1),
            stats: AtlasStats::default(),
            pending: Vec::new(),
            pending_images: Vec::new(),
            demand: HashMap::new(),
            ranked: Vec::new(),
        }
    }

    /// The coverage page's edge, in texels. What `size` has always meant here.
    pub fn size(&self) -> u32 {
        self.coverage.size
    }

    /// The colour page's edge, in texels.
    pub fn colour_size(&self) -> u32 {
        self.colour.size
    }

    /// What this atlas reserves on the GPU, both pages together.
    pub fn budget(&self) -> AtlasBudget {
        AtlasBudget {
            coverage_size: self.coverage.size,
            colour_size: self.colour.size,
        }
    }

    fn page(&self, page: AtlasPage) -> &Page {
        match page {
            AtlasPage::Coverage => &self.coverage,
            AtlasPage::Colour => &self.colour,
        }
    }

    fn page_mut(&mut self, page: AtlasPage) -> &mut Page {
        match page {
            AtlasPage::Coverage => &mut self.coverage,
            AtlasPage::Colour => &mut self.colour,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn stats(&self) -> AtlasStats {
        let mut stats = self.stats;
        stats.resident = self.resident.len() as u32;
        stats.occupancy = self.coverage.occupancy();
        stats.colour_occupancy = self.colour.occupancy();
        stats.bytes_reserved = self.budget().bytes();
        stats
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.stats.uploaded_this_frame = 0;
        self.stats.deferred_this_frame = 0;
        self.stats.demanded_this_frame = 0;
        self.stats.structural_this_frame = 0;
        self.stats.content_this_frame = 0;
        self.stats.image_this_frame = 0;
        self.pending.clear();
        self.pending_images.clear();
        self.demand.clear();
    }

    /// The per-frame content bound this atlas was configured with.
    pub fn content_bound(&self) -> u32 {
        self.max_uploads_per_frame
    }

    /// Whether the frame that just finished was fully served: nothing deferred, and nothing
    /// demanded that [`GlyphAtlas::admit_demanded`] could not admit.
    ///
    /// This is the convergence predicate, and it is only meaningful *after* the frame's
    /// `admit_demanded` -- before it, every content key the frame wanted is still
    /// outstanding by construction.
    pub fn converged(&self) -> bool {
        self.stats.deferred_this_frame == 0 && self.stats.demanded_this_frame == 0
    }

    /// Hand the frame's uploads to the caller, which copies them into the GPU texture.
    pub fn take_uploads(&mut self) -> Vec<PendingUpload> {
        std::mem::take(&mut self.pending)
    }

    /// Hand the frame's colour uploads to the caller, which copies them into the RGBA
    /// texture. Empty on every frame that admitted no picture.
    pub fn take_image_uploads(&mut self) -> Vec<PendingImage> {
        std::mem::take(&mut self.pending_images)
    }

    /// Look up a colour image, admitting it if it is not resident.
    ///
    /// `decode` produces the picture on a miss and is called **at most once per key per
    /// process**, and only after the per-frame image bound has been checked -- so a caller
    /// whose decode is expensive does not pay for a frame that could not have uploaded the
    /// result. That is the whole reason this takes a closure instead of an [`RgbaImage`].
    ///
    /// `None` has four causes and the statistics tell them apart: the bound was spent this
    /// frame ([`AtlasStats::deferred_this_frame`]), the decoder gave up (nothing counted --
    /// it is the decoder's failure to report), the picture is larger than the page
    /// ([`AtlasStats::rejected_oversize`]), or its buffer did not match its dimensions
    /// ([`AtlasStats::rejected_malformed`]). Only the first is worth retrying next frame.
    pub fn get_or_decode_image(
        &mut self,
        key: ImageKey,
        decode: impl FnOnce(ImageKey) -> Option<RgbaImage>,
    ) -> Option<AtlasEntry> {
        let atlas_key = AtlasKey::Image(key);
        if let Some(res) = self.resident.get_mut(&atlas_key) {
            res.last_used_frame = self.frame;
            res.referenced = true;
            self.stats.hits += 1;
            return Some(res.entry);
        }

        self.stats.misses += 1;

        if self.stats.image_this_frame >= IMAGE_UPLOADS_PER_FRAME {
            self.stats.deferred_this_frame += 1;
            return None;
        }

        let image = decode(key)?;
        if image.is_malformed() {
            self.stats.rejected_malformed += 1;
            return None;
        }

        self.insert_image(atlas_key, &image)
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
    /// Look up `key` without rasterizing it, recording that one more draw wanted it.
    ///
    /// `Some` means resident: draw it now. `None` means the atlas does not have it yet and
    /// the caller must **defer the instance** until after [`GlyphAtlas::admit_demanded`],
    /// which is where the content bound is actually spent. A caller that treats `None` as
    /// "dropped" throws away exactly the glyphs this mechanism exists to rescue.
    ///
    /// Demand is counted per *draw*, not per key, and that is the ranking: a letter that
    /// appears in forty visible filenames is wanted forty times and outranks one that
    /// appears once.
    pub fn want(&mut self, key: impl Into<AtlasKey>) -> Option<AtlasEntry> {
        let key = key.into();
        if let Some(res) = self.resident.get_mut(&key) {
            res.last_used_frame = self.frame;
            res.referenced = true;
            self.stats.hits += 1;
            return Some(res.entry);
        }
        self.stats.misses += 1;
        *self.demand.entry(key).or_insert(0) += 1;
        None
    }

    /// Spend the content bound on the keys this frame wanted most.
    ///
    /// Ranks everything [`GlyphAtlas::want`] recorded by how many draws wanted it, admits
    /// down that order until the per-frame bound is spent, and leaves the rest for a later
    /// frame. Ties break on the key so a given frame always admits the same set: a
    /// screenshot that changed between runs because a `HashMap` iterated differently would
    /// make every visual test flaky for a reason nobody could reproduce.
    ///
    /// `render` produces the coverage bitmap for a key, and is called at most once per key
    /// admitted -- never for one the bound could not reach, so a caller whose rasterization
    /// is expensive does not pay for work this frame could not have used.
    ///
    /// An inkless content key is a space. It is memoized as a blank exactly as
    /// [`GlyphAtlas::get_or_insert`] does and costs no upload, so a page full of spaces
    /// cannot consume the bound.
    pub fn admit_demanded(&mut self, mut render: impl FnMut(AtlasKey) -> Option<RasterizedGlyph>) {
        self.ranked.clear();
        self.ranked
            .extend(self.demand.iter().map(|(&k, &n)| (k, n)));
        // Descending demand, then ascending key. `sort_unstable_by` is fine because the
        // second term makes the order total.
        self.ranked
            .sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for index in 0..self.ranked.len() {
            if self.stats.content_this_frame >= self.max_uploads_per_frame {
                break;
            }
            let Some(&(key, _)) = self.ranked.get(index) else {
                break;
            };
            match render(key) {
                Some(bitmap) if !bitmap.is_blank() => {
                    if self.insert(key, &bitmap, UploadClass::Content).is_some() {
                        self.demand.remove(&key);
                    }
                }
                Some(_) => {
                    // A space. Memoizing costs one rasterization ever instead of one per
                    // frame, and no upload -- see `get_or_render`.
                    self.insert_blank(key);
                    self.demand.remove(&key);
                }
                // The face could not raster it. Dropping it from demand stops the atlas
                // retrying a glyph that will fail identically every frame, which would
                // otherwise hold a slot at the head of the ranking forever.
                None => {
                    self.demand.remove(&key);
                }
            }
        }

        // Whatever is left is this frame's shortfall. Two different numbers, and the
        // difference matters: `demanded` counts distinct *entries* still missing, which is
        // what decides how many more frames convergence takes, while `deferred` counts the
        // *draws* that went without, which is what the reader actually sees missing.
        self.stats.demanded_this_frame = self.demand.len() as u32;
        self.stats.deferred_this_frame += self.demand.values().copied().sum::<u32>();
        self.demand.clear();
    }

    /// Resident lookup with no demand recorded and no insertion.
    ///
    /// The second half of [`GlyphAtlas::want`]: the deferring caller calls this after
    /// [`GlyphAtlas::admit_demanded`] to find out which of its parked draws can now be
    /// emitted. Counting demand again here would double every glyph's rank.
    pub fn get(&mut self, key: impl Into<AtlasKey>) -> Option<AtlasEntry> {
        let key = key.into();
        let res = self.resident.get_mut(&key)?;
        res.last_used_frame = self.frame;
        res.referenced = true;
        Some(res.entry)
    }

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
        self.get_or_render(key, true, UploadClass::Content, |slot| {
            raster.rasterize(db, slot)
        })
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
    ///
    /// `class` decides which bound the request is rationed against -- see [`UploadClass`].
    /// This is the *immediate* path: it admits on the spot and therefore in call order,
    /// which is correct for [`UploadClass::Structural`] (bounded by construction, so order
    /// cannot starve anything) and is the thing [`GlyphAtlas::want`] exists to replace for
    /// content.
    pub fn get_or_render<K>(
        &mut self,
        key: K,
        blank_is_ok: bool,
        class: UploadClass,
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

        let (spent, bound) = match class {
            UploadClass::Structural => (
                self.stats.structural_this_frame,
                STRUCTURAL_UPLOADS_PER_FRAME,
            ),
            UploadClass::Content => (self.stats.content_this_frame, self.max_uploads_per_frame),
            // A picture cannot arrive here: `render` yields a RasterizedGlyph, which is
            // coverage. The arm exists so the bound is stated for every class rather than
            // defaulted by a wildcard, which is what would let a fourth class ship
            // silently rationed against text.
            UploadClass::Image => (self.stats.image_this_frame, IMAGE_UPLOADS_PER_FRAME),
        };
        if spent >= bound {
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

        self.insert(atlas_key, &bitmap, class)
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
                page: key.page(),
                shelf: usize::MAX,
                x: 0,
                slot_width: 0,
                last_used_frame: self.frame,
                referenced: true,
            },
        );
        self.clock.push_back(key);
    }

    /// Find room for `padded_w x padded_h` on `page`, evicting if it has to.
    ///
    /// Returns `(shelf index, x, shelf y)`. The three refusals are counted separately here
    /// rather than merged into one, which is the whole of the fourth acceptance criterion:
    /// "too big to ever fit" and "everything that would fit is in use" call for opposite
    /// responses.
    fn place(
        &mut self,
        page: AtlasPage,
        padded_w: u32,
        padded_h: u32,
    ) -> Option<(usize, u32, u32)> {
        let size = self.page(page).size;
        if padded_w > size || padded_h > size {
            // Larger than the page could ever hold, whatever it currently holds. Not the
            // cliff -- see AtlasStats::rejected_oversize.
            self.stats.rejected_oversize += 1;
            return None;
        }

        let (shelf_index, x) = match self.page_mut(page).allocate(padded_w, padded_h) {
            Some(slot) => slot,
            None => {
                if !self.evict_until_room(page, padded_w, padded_h) {
                    // Every entry on this page that would fit is in use this frame. This is
                    // the cliff.
                    self.stats.overflow_events += 1;
                    return None;
                }
                self.page_mut(page).allocate(padded_w, padded_h)?
            }
        };

        let shelf_y = self.page(page).shelves.get(shelf_index)?.y;
        self.page_mut(page).allocated_area += u64::from(padded_w) * u64::from(padded_h);
        Some((shelf_index, x, shelf_y))
    }

    fn insert(
        &mut self,
        key: AtlasKey,
        bitmap: &RasterizedGlyph,
        class: UploadClass,
    ) -> Option<AtlasEntry> {
        let padded_w = bitmap.width + GUTTER * 2;
        let padded_h = bitmap.height + GUTTER * 2;
        let (shelf_index, x, shelf_y) = self.place(AtlasPage::Coverage, padded_w, padded_h)?;

        let gx = x + GUTTER;
        let gy = shelf_y + GUTTER;

        let inv = 1.0 / self.coverage.size as f32;
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
        match class {
            UploadClass::Structural => self.stats.structural_this_frame += 1,
            UploadClass::Content => self.stats.content_this_frame += 1,
            // Unreachable through the coverage page -- an image is never a RasterizedGlyph.
            // Counted rather than ignored so a future caller that gets it wrong is visible
            // in the statistics instead of silently uncounted.
            UploadClass::Image => self.stats.image_this_frame += 1,
        }

        self.resident.insert(
            key,
            Resident {
                entry,
                page: AtlasPage::Coverage,
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

    /// The colour page's half of [`GlyphAtlas::insert`].
    ///
    /// Structurally the same and deliberately not merged with it: the two differ in what
    /// they carry (`left`/`top` are a baseline offset a picture does not have), in which
    /// pending list they push to, and in the premultiply. A merged function would take a
    /// discriminated bitmap and branch three times, which is the same code with an extra
    /// type in front of it.
    fn insert_image(&mut self, key: AtlasKey, image: &RgbaImage) -> Option<AtlasEntry> {
        let padded_w = image.width + GUTTER * 2;
        let padded_h = image.height + GUTTER * 2;
        let (shelf_index, x, shelf_y) = self.place(AtlasPage::Colour, padded_w, padded_h)?;

        let gx = x + GUTTER;
        let gy = shelf_y + GUTTER;

        let inv = 1.0 / self.colour.size as f32;
        let entry = AtlasEntry {
            uv: [
                gx as f32 * inv,
                gy as f32 * inv,
                (gx + image.width) as f32 * inv,
                (gy + image.height) as f32 * inv,
            ],
            // A picture has no baseline. Zero rather than a guess, and the field is on the
            // shared AtlasEntry because a caller reading it for an image should get
            // "nothing to offset by" rather than a number that meant something else.
            left: 0,
            top: 0,
            width: image.width,
            height: image.height,
            generation: self.generation,
        };

        self.pending_images.push(PendingImage {
            x: gx,
            y: gy,
            width: image.width,
            height: image.height,
            rgba: image.rgba.clone(),
        });
        self.stats.uploaded_this_frame += 1;
        self.stats.image_this_frame += 1;

        self.resident.insert(
            key,
            Resident {
                entry,
                page: AtlasPage::Colour,
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

    /// Free entries on `page` until `w x h` fits there, never touching one used this frame.
    ///
    /// One policy, one queue, one second-chance bit. A resident on the *other* page is put
    /// back exactly as an in-use one is: it cannot help, and evicting it would be a
    /// picture paying for a glyph's slot.
    ///
    /// Returns `false` when nothing on this page is evictable.
    fn evict_until_room(&mut self, page: AtlasPage, w: u32, h: u32) -> bool {
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

            if res.page != page {
                // On the other page. Its slot is the wrong shape and the wrong format, so
                // freeing it would cost a picture to seat a glyph and open nothing. Put it
                // back untouched -- including its second-chance bit, which is not this
                // sweep's to spend.
                self.clock.push_back(key);
                continue;
            }
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
            let target = self.page_mut(page);
            let shelf_height = target.shelves.get(shelf_index).map(|s| s.height);
            self.resident.remove(&key);
            self.stats.evictions += 1;

            let target = self.page_mut(page);
            if let (Some(shelf), true) = (
                target.shelves.get_mut(shelf_index),
                shelf_index != usize::MAX,
            ) {
                let height = shelf.height;
                shelf.free.push((x, slot_width));
                target.allocated_area = target
                    .allocated_area
                    .saturating_sub(u64::from(slot_width) * u64::from(height));
            }

            // Did that open a slot the caller can use?
            if shelf_height == Some(class) && slot_width >= w {
                return true;
            }
            if self.page(page).allocate_probe(w, class) {
                return true;
            }
        }
        false
    }

    /// Drop everything and start over. Bumps the generation so any cached entry is
    /// detectably stale. Called when the text size changes -- every glyph in the atlas was
    /// rasterized at the old size and none of them are reusable.
    ///
    /// Both pages, because the generation is one number: leaving colour residents behind
    /// would leave entries stamped with a generation the atlas has moved past, which is
    /// exactly the staleness [`AtlasEntry::generation`] exists to make detectable.
    pub fn clear(&mut self) {
        self.coverage.clear();
        self.colour.clear();
        self.resident.clear();
        self.clock.clear();
        self.pending.clear();
        self.pending_images.clear();
        self.demand.clear();
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
    fn the_content_bound_goes_to_the_most_wanted_keys_not_the_first_asked() {
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        if glyphs.len() < 3 {
            return;
        }
        // One upload for three keys, so exactly one can win and the choice is forced to be
        // visible. The rare key is asked for *first*: under the old first-come rule it
        // would take the slot, which is the defect this whole mechanism replaces.
        let mut atlas = GlyphAtlas::new(1024, 1);
        atlas.begin_frame();

        let (rare_id, rare_font) = glyphs[0];
        let (common_id, common_font) = glyphs[1];
        let rare = key(rare_font, rare_id);
        let common = key(common_font, common_id);

        assert!(atlas.want(rare).is_none());
        for _ in 0..40 {
            assert!(atlas.want(common).is_none());
        }
        atlas.admit_demanded(|k| match k {
            AtlasKey::Glyph(g) => raster.rasterize(db.as_ref(), g),
            AtlasKey::Icon(_) | AtlasKey::Image(_) => None,
        });

        assert!(
            atlas.get(common).is_some(),
            "the key forty draws wanted did not get the frame's one upload"
        );
        assert!(
            atlas.get(rare).is_none(),
            "the key one draw wanted took the slot because it asked first"
        );
        assert_eq!(atlas.stats().content_this_frame, 1);
        assert_eq!(
            atlas.stats().demanded_this_frame,
            1,
            "one distinct entry is still owed"
        );
        assert_eq!(
            atlas.stats().deferred_this_frame,
            1,
            "and exactly one draw went without"
        );
    }

    #[test]
    fn structural_entries_draw_from_their_own_bound_and_text_cannot_starve_them() {
        // The rule that let the row builder's icon prepass go. The content bound is spent
        // to the last upload before the structural request arrives; under one shared
        // counter it would be refused, which is exactly how a cold frame used to lose its
        // icons to the text above them.
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        if glyphs.len() < 4 {
            return;
        }
        let mut atlas = GlyphAtlas::new(1024, 2);
        atlas.begin_frame();
        for &(id, font) in glyphs.iter().take(4) {
            atlas.want(key(font, id));
        }
        atlas.admit_demanded(|k| match k {
            AtlasKey::Glyph(g) => raster.rasterize(db.as_ref(), g),
            AtlasKey::Icon(_) | AtlasKey::Image(_) => None,
        });
        assert_eq!(
            atlas.stats().content_this_frame,
            2,
            "the content bound was not exhausted, so this proves nothing"
        );

        // A structural request now, with the content bound gone.
        let shape = IconKey {
            shape: crate::icon::IconShape::Kind(crate::icon::IconKind::Folder),
            px: 20,
        };
        let entry = atlas.get_or_render(
            shape,
            false,
            UploadClass::Structural,
            crate::icon::rasterize,
        );
        assert!(
            entry.is_some(),
            "text spent the frame and the icon was refused -- the two classes share a bound"
        );
        assert_eq!(atlas.stats().structural_this_frame, 1);
    }

    #[test]
    fn the_structural_bound_holds_every_shape_the_icon_module_can_produce() {
        // `STRUCTURAL_UPLOADS_PER_FRAME` is a claim about `crate::icon`, not about the
        // machine, so it is only correct as long as that claim is. A tenth icon kind added
        // without raising the constant would silently start dropping one icon per frame.
        let shapes = crate::icon::IconKind::ALL.len()
            + crate::icon::Emblem::ALL.len()
            + crate::icon::StateIcon::ALL.len()
            + crate::icon::Chevron::ALL.len();
        // Strictly under, not at: the constant's own docs claim a spare slot, and a bound
        // that exactly equals the shape count is one that passes on the day it is written
        // and drops an icon a frame on the day a shape is added. Asserting the claim is what
        // makes the next person raise the constant deliberately instead of discovering it.
        assert!(
            (shapes as u32) < STRUCTURAL_UPLOADS_PER_FRAME,
            "qs_gpu::icon can produce {shapes} distinct shapes in one frame and the \
             structural bound is {STRUCTURAL_UPLOADS_PER_FRAME}, which leaves no spare"
        );
    }

    #[test]
    fn a_blank_is_memoized_by_demanded_admission_and_costs_no_upload() {
        // Spaces are common, so they rank high, and a space that spent an upload every
        // frame would take a slot from a letter forever.
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        let Some(&(id, font)) = glyphs.first() else {
            return;
        };
        let mut atlas = GlyphAtlas::new(1024, 4);
        atlas.begin_frame();
        let k = key(font, id);
        atlas.want(k);
        // Force the blank path: render reports an inkless bitmap for this key.
        atlas.admit_demanded(|_| {
            Some(RasterizedGlyph {
                width: 0,
                height: 0,
                left: 0,
                top: 0,
                coverage: Vec::new(),
                was_color: false,
            })
        });
        assert!(atlas.is_blank(k), "the blank was not memoized");
        assert_eq!(
            atlas.stats().content_this_frame,
            0,
            "a blank spent an upload"
        );
        assert_eq!(atlas.stats().demanded_this_frame, 0);

        // And it stays memoized: a second frame must not re-rasterize it.
        atlas.begin_frame();
        assert!(atlas.want(k).is_some());
        let _ = (&mut raster, db);
    }

    #[test]
    fn clearing_bumps_the_generation_so_stale_entries_are_detectable() {
        let mut atlas = GlyphAtlas::new(256, 16);
        let before = atlas.generation();
        atlas.clear();
        assert_eq!(atlas.generation(), before + 1);
    }

    // -- the colour page ----------------------------------------------------------------

    /// A solid picture, `w x h`, straight alpha.
    fn picture(w: u32, h: u32) -> RgbaImage {
        RgbaImage {
            width: w,
            height: h,
            rgba: (0..w * h)
                .flat_map(|i| [(i % 251) as u8, (i % 241) as u8, (i % 239) as u8, 255])
                .collect(),
        }
    }

    /// An atlas with pages small enough that filling one is cheap.
    fn two_page_atlas(coverage: u32, colour: u32) -> GlyphAtlas {
        GlyphAtlas::with_budget(
            AtlasBudget {
                coverage_size: coverage,
                colour_size: colour,
            },
            64,
        )
    }

    #[test]
    fn a_picture_and_a_glyph_live_in_one_atlas_under_one_cap() {
        let atlas = two_page_atlas(2048, DEFAULT_COLOUR_PAGE);
        let budget = atlas.budget();
        assert!(
            budget.fits(),
            "the shipped pages are {} bytes against a cap of {ATLAS_CAP_BYTES}",
            budget.bytes()
        );
        // One number covering both, which is the acceptance criterion: a colour cache
        // measured on its own would make this figure a lie by however much it costs.
        assert_eq!(
            atlas.stats().bytes_reserved,
            u64::from(2048u32) * 2048
                + u64::from(DEFAULT_COLOUR_PAGE) * u64::from(DEFAULT_COLOUR_PAGE) * 4
        );
    }

    #[test]
    fn a_budget_over_the_cap_is_refused_rather_than_quietly_shrunk() {
        // 8192 x 8192 x 4 is 256 MiB on the colour page alone. Silently clamping it would
        // hand back an atlas that thrashes for a reason nothing in the statistics explains.
        let over = AtlasBudget {
            coverage_size: 2048,
            colour_size: 8192,
        };
        assert!(!over.fits());
        assert!(GlyphAtlas::try_with_budget(over, 64).is_none());
    }

    #[test]
    fn a_picture_too_large_for_its_page_is_refused_and_says_which_refusal_it_is() {
        let mut atlas = two_page_atlas(512, 256);
        atlas.begin_frame();

        // 300 px on a 256 px page: no arrangement of the page could ever hold it.
        let admitted = atlas.get_or_decode_image(ImageKey(1), |_| Some(picture(300, 32)));
        assert!(admitted.is_none());

        let stats = atlas.stats();
        assert_eq!(stats.rejected_oversize, 1);
        assert_eq!(
            stats.overflow_events, 0,
            "an oversized picture is not the working set failing to fit -- counting it as \
             one makes a single bad file indistinguishable from the atlas thrash cliff"
        );
    }

    #[test]
    fn a_malformed_picture_is_refused_at_admission_rather_than_uploaded() {
        let mut atlas = two_page_atlas(512, 256);
        atlas.begin_frame();

        let lying = RgbaImage {
            width: 16,
            height: 16,
            rgba: vec![0; 16 * 16 * 3], // three channels, not four
        };
        assert!(lying.is_malformed());
        assert!(
            atlas
                .get_or_decode_image(ImageKey(2), |_| Some(lying.clone()))
                .is_none()
        );
        assert_eq!(atlas.stats().rejected_malformed, 1);
        assert!(
            atlas.take_image_uploads().is_empty(),
            "a buffer that does not match its dimensions must never reach write_texture -- \
             the row stride there is computed from the width and would read past the end"
        );
    }

    #[test]
    fn a_full_colour_page_cannot_evict_a_glyph() {
        // The failure this whole two-page shape exists to prevent, stated as a test: a
        // folder of photographs must not cost the file list its text.
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        // A colour page (256, its floor) that holds exactly one of the pictures below,
        // beside a roomy coverage page.
        let mut atlas = two_page_atlas(512, 256);

        atlas.begin_frame();
        let (id, font) = glyphs[0];
        assert!(
            atlas
                .get_or_insert(db.as_ref(), &mut raster, key(font, id))
                .is_some()
        );
        let resident_after_glyph = atlas.stats().resident;

        // Now fill and overfill the colour page across several frames, so the pictures are
        // not protected by the never-evict-in-use rule and eviction genuinely runs.
        for frame in 0..8u64 {
            atlas.begin_frame();
            for n in 0..2 {
                atlas.get_or_decode_image(ImageKey(frame * 2 + n), |_| Some(picture(200, 200)));
            }
        }

        assert!(
            atlas.stats().evictions > 0,
            "the colour page never filled, so this proves nothing about what eviction takes"
        );
        assert!(
            atlas.get(key(font, id)).is_some(),
            "a picture evicted a glyph: the pages are sharing slots, which is exactly the \
             failure the static split exists to make unreachable"
        );
        assert!(atlas.stats().resident >= resident_after_glyph);
    }

    #[test]
    fn a_frame_with_no_pictures_costs_the_r8_path_nothing() {
        // Acceptance criterion 2, measured rather than asserted: a frame that admits only
        // glyphs must produce the same uploads, the same statistics and no colour work at
        // all. `take_image_uploads` returning an empty vector is the whole of the second
        // half -- Vec::new does not allocate, so an image-free frame does not touch the
        // allocator on the colour path.
        let Some((db, mut raster, glyphs)) = fixture() else {
            return;
        };
        let mut atlas = two_page_atlas(512, 256);

        atlas.begin_frame();
        for &(id, font) in glyphs.iter().take(8) {
            atlas.get_or_insert(db.as_ref(), &mut raster, key(font, id));
        }

        let stats = atlas.stats();
        assert_eq!(stats.image_this_frame, 0);
        assert_eq!(stats.rejected_oversize, 0);
        assert_eq!(stats.rejected_malformed, 0);
        assert_eq!(stats.colour_occupancy, 0.0);
        assert!(stats.content_this_frame > 0, "no glyphs were admitted");
        assert!(
            atlas.take_image_uploads().is_empty(),
            "a glyph-only frame produced colour uploads"
        );
        assert!(!atlas.take_uploads().is_empty());
    }

    #[test]
    fn the_image_bound_rations_pictures_separately_from_text() {
        // The third upload class earns its keep the same way the second did: pictures must
        // not be able to spend the bound that text is rationed against, in either direction.
        let mut atlas = two_page_atlas(512, 256);
        atlas.begin_frame();

        let mut admitted = 0;
        for n in 0..(IMAGE_UPLOADS_PER_FRAME + 3) {
            if atlas
                .get_or_decode_image(ImageKey(u64::from(n)), |_| Some(picture(24, 24)))
                .is_some()
            {
                admitted += 1;
            }
        }

        assert_eq!(admitted, IMAGE_UPLOADS_PER_FRAME);
        assert_eq!(
            atlas.stats().content_this_frame,
            0,
            "admitting pictures spent the content bound, so a gallery would starve the \
             filenames beside it"
        );
        assert_eq!(atlas.stats().image_this_frame, IMAGE_UPLOADS_PER_FRAME);
    }

    #[test]
    fn a_picture_is_uploaded_once_and_hit_thereafter() {
        let mut atlas = two_page_atlas(512, 256);
        atlas.begin_frame();

        let mut decodes = 0;
        let first = atlas.get_or_decode_image(ImageKey(7), |_| {
            decodes += 1;
            Some(picture(20, 20))
        });
        assert!(first.is_some());

        let second = atlas.get_or_decode_image(ImageKey(7), |_| {
            decodes += 1;
            Some(picture(20, 20))
        });
        assert_eq!(first, second);
        assert_eq!(
            decodes, 1,
            "the decoder ran twice for one key -- a hit must not pay for a decode, which is \
             the entire reason this takes a closure"
        );
        assert_eq!(atlas.take_image_uploads().len(), 1);
    }

    #[test]
    fn every_key_knows_which_page_it_belongs_on() {
        // The page is derived from the key and not chosen at the call site, so a caller
        // cannot put a picture on the coverage page by asking nicely.
        assert_eq!(AtlasKey::Image(ImageKey(0)).page(), AtlasPage::Colour);
        assert_eq!(AtlasPage::Colour.bytes_per_texel(), 4);
        assert_eq!(AtlasPage::Coverage.bytes_per_texel(), 1);
        let Some((_, _, glyphs)) = fixture() else {
            return;
        };
        let (id, font) = glyphs[0];
        assert_eq!(AtlasKey::from(key(font, id)).page(), AtlasPage::Coverage);
    }
}
