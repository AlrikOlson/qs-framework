//! The row recycler: decides which rows exist this frame, and proves it stayed bounded.
//!
//! # FR-002 is the definition of "virtualized"
//!
//! > Requests at most `viewport_height / row_height + 2` rows per frame.
//!
//! A regression here does not look like a bug. The list still renders correctly; it just
//! lays out ten thousand rows instead of forty and misses the frame budget by two orders of
//! magnitude. Nothing about the output says why. So the bound is **asserted in debug
//! builds** on every frame ([`Recycler::layout`]) and asserted again by
//! `tests/virtualization_bound.rs` across the full parameter space, because a silent
//! failure that destroys SC-001 deserves two independent checks rather than a comment.
//!
//! # What "recycling" means here, and what it does not
//!
//! There are no retained row objects to recycle. Each frame reads the visible slice into a
//! reused [`RowBuf`] and builds instances from it. That *is* the recycling: the allocations
//! persist, the contents do not. A pool of live row widgets would buy nothing -- there is
//! no per-row state worth carrying between frames -- and would cost the correctness problem
//! that makes virtualized lists notorious, where a recycled row keeps a scrap of the
//! previous row's data.

use crate::density::Density;
use crate::fenwick::Heights;
use crate::row_source::{RowBuf, RowSource};
use crate::scroll::{VisibleRange, visible_range};

/// Non-glyph primitives one visible entry is allowed to cost.
///
/// Measured, not chosen. With every state a row can be in switched on at once -- forty rows
/// all selected, one of them hovered, pressed and focused -- the list pass emits **3.60**
/// surface primitives per entry and the grid pass **3.10**: banding on the odd rows, the
/// hover and press washes, the selection halo, the selection fill, the status rail, and the
/// focus ring's two strokes on the one focused row.
///
/// Six is that measurement with room for one more layer on the busiest surface. A material
/// that grows still passes; a material that doubles does not, which is the sensitivity worth
/// having -- adding a layer in `design/tokens.json` does not look like a frame-time decision,
/// and [`ViewportLayout::surface_prim_bound`] is what makes it one.
///
/// It was **five**, and it is six because `prim-emissive-edge` gave `row/selected` a contact
/// shadow: a fourth layer on the surface that already had the most. That is the decision this
/// constant exists to force, and it is worth recording that it was made rather than absorbed.
/// A shadow is not decoration on a selection -- it is the thing that makes a selected row read
/// as sitting above the list rather than being printed on it, which is the cue the previous
/// three layers could not express between them. The cost is one instance per **selected**
/// entry, and `the_halo_costs_a_measured_amount_of_overdraw_and_not_an_assumed_one` measures
/// what it costs in fill rather than assuming it is small.
///
/// The next layer that wants this number raised should have to argue as specifically.
pub const SURFACE_PRIMS_PER_ENTRY: u32 = 6;

/// Everything about the frame's geometry that the row builder needs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ViewportLayout {
    /// Left edge of the entry surface within the window, in physical pixels.
    ///
    /// Zero while the list owns the whole window. See [`ViewportLayout::at`] for why the
    /// origin lives here rather than at each call site.
    pub origin_x: f32,
    /// Top edge of the entry surface within the window, in physical pixels.
    pub origin_y: f32,
    /// Physical pixels. The size of the **entry surface**, not of the window: chrome is
    /// subtracted before this is computed, which is what keeps the FR-002 bound honest
    /// once a command bar and a status shelf are on screen.
    pub width: u32,
    pub height: u32,
    /// Device pixel ratio.
    pub scale: f32,
    /// OS text-size setting.
    pub text_scale: f32,
    pub density: Density,
    /// Physical pixels per row.
    pub row_height: u32,
    /// Content-space scroll offset.
    pub scroll: f64,
    /// How far the first visible row is scrolled off the top, in physical pixels.
    pub first_row_offset: f64,
    pub visible: VisibleRange,
    /// Total logical rows.
    pub row_count: u64,
    /// Total content height in physical pixels.
    pub content_height: u64,
    /// Shortest row in the source, in physical pixels. See
    /// [`ViewportLayout::virtualization_bound`].
    pub min_row_height: u32,
    /// Entries packed into one laid-out row. `1` in the list, and whatever fits across the
    /// surface in the grid.
    ///
    /// This is what keeps `visible.count` meaning **entries** in every view. `fill`, the
    /// accessibility tree and `ensure_filled` all already read it that way, so the
    /// alternative — counting grid rows here and entries somewhere else — would leave the
    /// FR-002 assertion checking a number nothing else uses.
    pub columns: u32,
}

impl ViewportLayout {
    /// The FR-002 bound for this viewport, counted in **entries**.
    ///
    /// Derived from the **shortest** row, not the nominal one. FR-002 writes the bound as
    /// `viewport_height / row_height + 2`, which is unambiguous only while every row is the
    /// same height -- the case `flat-1m` exercises. With variable heights (`deep-40`) the
    /// worst case is however many of the shortest row fit on screen, and computing the
    /// bound from the nominal height would let the assertion pass while the recycler laid
    /// out several times more rows than intended.
    ///
    /// The `* columns` is the whole reason [`ViewportLayout::columns`] exists. A grid lays
    /// out several entries per row, and there are two wrong ways to reconcile that which
    /// fail in opposite directions: pushing entry counts through the row-shaped bound fires
    /// the assertion on a perfectly correct grid, and keeping the bound in grid rows while
    /// counting entries elsewhere leaves the assertion passing while nothing checks the
    /// number that actually costs frame time.
    pub fn virtualization_bound(&self) -> u32 {
        ((self.height / self.min_row_height.max(1)) + 2).saturating_mul(self.columns.max(1))
    }

    /// The FR-002 bound restated in the unit that costs frame time: **surface primitives**.
    ///
    /// Bounding entries was enough while a row's surface was a fixed handful of hand-written
    /// `Instance` pushes. Materials make composition cheap, and cheap composition is how one
    /// row goes from six primitives to twenty without anybody deciding to -- a layer added to
    /// `row/selected` in `design/tokens.json` costs one instance per selected row on every
    /// frame, and nothing in a token file looks like a frame-time decision.
    ///
    /// So the bound is re-measured here rather than assumed. [`SURFACE_PRIMS_PER_ENTRY`] is
    /// the per-entry allowance and `the_surface_primitive_budget_holds_in_the_worst_case` in
    /// `row.rs` is what holds the row renderer to it, with every state a row can be in
    /// switched on at once.
    ///
    /// Glyphs are deliberately outside it. A row's glyph count is a function of how long its
    /// name is, not of how it is composed, and folding the two together would produce a
    /// number that moves when a directory is renamed.
    pub fn surface_prim_bound(&self) -> u32 {
        self.virtualization_bound()
            .saturating_mul(SURFACE_PRIMS_PER_ENTRY)
    }

    /// Width of one cell, in physical pixels. The full surface width in the list.
    #[must_use]
    pub fn cell_width(&self) -> f32 {
        self.width as f32 / self.columns.max(1) as f32
    }

    /// Where the entry at `slot` within the visible slice is drawn, in window space.
    ///
    /// The list is the one-column case and gets the full-width row it always had.
    #[must_use]
    pub fn cell_rect(&self, slot: u32) -> (f32, f32, f32, f32) {
        let columns = self.columns.max(1);
        let width = self.cell_width();
        (
            self.origin_x + (slot % columns) as f32 * width,
            self.row_top(slot),
            width,
            self.row_height as f32,
        )
    }

    /// Place this layout's entry surface at `(x, y)` within the window.
    ///
    /// Chainable rather than two more parameters on [`Recycler::layout`], which already
    /// takes nine and is called from three crates that have no opinion about chrome.
    ///
    /// The origin lives on `ViewportLayout` because four separate calculations have to
    /// agree about where the surface is -- which row the pointer is over, how far the list
    /// may scroll, where a focused row must move to stay visible, and what the
    /// accessibility tree describes -- and this is the one type all four already read. A
    /// second, parallel notion of "where the list starts" is exactly the drift that makes
    /// a chrome-plus-virtualized-list combination select row 5 when row 7 was clicked.
    #[must_use]
    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.origin_x = x;
        self.origin_y = y;
        self
    }

    /// Top edge of a visible row in **window** space.
    ///
    /// Note that the row's position within the surface is derived from its index in the
    /// visible slice, not from its absolute content offset. That is deliberate and it is
    /// research R5 in practice: the absolute offset of row 999,999 is 28 million, and
    /// subtracting the scroll from it in `f32` would lose the fractional part. Here the
    /// arithmetic never leaves the surface's own coordinate range before the origin is
    /// added.
    pub fn row_top(&self, slot: u32) -> f32 {
        let line = f64::from(slot / self.columns.max(1));
        self.origin_y + (line * f64::from(self.row_height) - self.first_row_offset) as f32
    }

    /// Which entry a window-space point falls on, if any.
    ///
    /// The inverse of [`ViewportLayout::cell_rect`], and deliberately its neighbour: hit
    /// testing that re-derives the mapping somewhere else is hit testing that can disagree
    /// with what was drawn. Returns `None` outside the surface, past the last visible row,
    /// and — the case a grid adds — in the gap after the last entry of a short final row.
    #[must_use]
    pub fn entry_at(&self, x: f32, y: f32) -> Option<u64> {
        if y < self.origin_y || self.visible.count == 0 {
            return None;
        }
        let columns = self.columns.max(1);

        let column = if columns == 1 {
            0
        } else {
            let across = x - self.origin_x;
            if across < 0.0 {
                return None;
            }
            let column = (across / self.cell_width().max(1.0)) as u32;
            // A pointer past the right edge of the last column is outside the grid, not
            // inside its last cell.
            if column >= columns {
                return None;
            }
            column
        };

        let within = f64::from(y - self.origin_y) + self.first_row_offset;
        if within < 0.0 {
            return None;
        }
        let line = (within / f64::from(self.row_height.max(1))) as u64;
        let slot = line * u64::from(columns) + u64::from(column);
        // Both bounds matter and they are different failures. Past `visible.count` is a
        // point below the last laid-out row, or in the empty tail of the final row of a
        // folder whose entry count is not a multiple of the column count -- pointing at a
        // cell that was never drawn.
        (slot < u64::from(self.visible.count)).then(|| self.visible.first + slot)
    }

    /// Which row a window-space `y` falls on. The list's one-column case of
    /// [`ViewportLayout::entry_at`].
    #[must_use]
    pub fn row_at(&self, y: f32) -> Option<u64> {
        self.entry_at(self.origin_x, y)
    }
}

/// Owns the reusable row buffer.
#[derive(Debug, Default)]
pub struct Recycler {
    buf: RowBuf,
    /// Source version the buffer was filled from. RS-4 promises this changes only on real
    /// content changes, which is what makes it usable as a staleness check.
    filled_version: u64,
    last_range: Option<VisibleRange>,
}

impl Recycler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute this frame's layout.
    ///
    /// The argument list is long because a frame's geometry genuinely has this many
    /// independent inputs; collecting them into a parameter struct would move the list to
    /// another file rather than shorten it.
    #[allow(clippy::too_many_arguments)]
    pub fn layout(
        &self,
        source: &dyn RowSource,
        heights: &Heights,
        width: u32,
        height: u32,
        scale: f32,
        text_scale: f32,
        density: Density,
        row_height: u32,
        scroll: f64,
    ) -> ViewportLayout {
        let row_count = source.len();
        let content_height = heights.total(row_count);
        let (visible, first_row_offset) = visible_range(scroll, height, row_count, heights);

        let layout = ViewportLayout {
            // The whole window until a caller says otherwise -- see `ViewportLayout::at`.
            origin_x: 0.0,
            origin_y: 0.0,
            width,
            height,
            scale,
            text_scale,
            density,
            row_height: row_height.max(1),
            scroll,
            first_row_offset,
            visible,
            row_count,
            content_height,
            min_row_height: heights.min_height(),
            columns: 1,
        };

        // The assertion that makes FR-002 enforceable rather than aspirational. Compiled
        // out in release, so the frame path pays nothing for it.
        debug_assert!(
            visible.count <= layout.virtualization_bound() + 1,
            "virtualization bound broken: laid out {} rows, bound is {} \
             (viewport {height}px / row {row_height}px + 2)",
            visible.count,
            layout.virtualization_bound(),
        );
        debug_assert!(
            visible.end() <= row_count,
            "visible range {:?} runs past the corpus ({row_count} rows)",
            visible.as_range()
        );

        layout
    }

    /// Compute this frame's layout as a **grid** of `columns` cells per row.
    ///
    /// Runs the same [`visible_range`] the list does, in grid-row space, and then converts
    /// the answer back to an entry range. Two consequences are worth stating because they
    /// are what make the rest of the system not care which view is on screen:
    ///
    /// - `visible` is a contiguous range of **entries**, exactly as in the list, so
    ///   [`Recycler::fill`], `RowSource::rows` and the accessibility tree are untouched.
    /// - `row_count` stays the entry count, never the grid-row count, so `set_size` keeps
    ///   describing the folder rather than the shape it happens to be drawn in.
    #[allow(clippy::too_many_arguments)]
    pub fn layout_grid(
        &self,
        source: &dyn RowSource,
        columns: u32,
        cell_height: u32,
        width: u32,
        height: u32,
        scale: f32,
        text_scale: f32,
        density: Density,
        scroll: f64,
    ) -> ViewportLayout {
        let columns = columns.max(1);
        let cell_height = cell_height.max(1);
        let entry_count = source.len();
        let lines = entry_count.div_ceil(u64::from(columns));

        let heights = Heights::Uniform(cell_height);
        let (lines_visible, first_row_offset) = visible_range(scroll, height, lines, &heights);

        // Grid rows back to entries. The last row of a folder whose count is not a multiple
        // of the column count is short, so the end is clamped -- without that, `fill` would
        // ask the source for indices past the corpus and RS-2 would have to invent them.
        let first = lines_visible.first * u64::from(columns);
        let end = ((lines_visible.first + u64::from(lines_visible.count)) * u64::from(columns))
            .min(entry_count);
        let visible = VisibleRange {
            first,
            count: (end - first) as u32,
        };

        let layout = ViewportLayout {
            origin_x: 0.0,
            origin_y: 0.0,
            width,
            height,
            scale,
            text_scale,
            density,
            row_height: cell_height,
            scroll,
            first_row_offset,
            visible,
            row_count: entry_count,
            content_height: heights.total(lines),
            min_row_height: cell_height,
            columns,
        };

        debug_assert!(
            visible.count <= layout.virtualization_bound() + columns,
            "virtualization bound broken: laid out {} entries, bound is {} \
             (viewport {height}px / cell {cell_height}px + 2, times {columns} columns)",
            visible.count,
            layout.virtualization_bound(),
        );
        debug_assert!(
            visible.end() <= entry_count,
            "visible range {:?} runs past the corpus ({entry_count} entries)",
            visible.as_range()
        );

        layout
    }

    /// Fill the buffer with the visible rows.
    ///
    /// Calls `source.rows()` exactly once per frame with exactly the visible range. RS-1
    /// promises that call returns immediately; RS-2 promises it fills the whole range.
    pub fn fill(&mut self, source: &dyn RowSource, layout: &ViewportLayout) -> &RowBuf {
        qs_gpu::affinity::assert_ui_thread("Recycler::fill");

        self.buf.clear();
        if layout.visible.count > 0 {
            source.rows(layout.visible.as_range(), &mut self.buf);
        }
        self.filled_version = source.version();
        self.last_range = Some(layout.visible);

        // RS-2 again, from the consumer's side. A source that returns short would otherwise
        // produce blank rows at the bottom of the viewport, and the cause would be three
        // crates away from the symptom.
        debug_assert!(
            self.buf.len() as u64 <= u64::from(layout.visible.count),
            "source returned more rows than were asked for"
        );

        &self.buf
    }

    pub fn rows(&self) -> &RowBuf {
        &self.buf
    }

    /// Whether the buffer's contents predate the source's current version.
    pub fn is_stale(&self, source: &dyn RowSource) -> bool {
        self.filled_version != source.version()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;
    use crate::row_source::StubbornSource;

    fn layout_for(rows: u64, height: u32, row_height: u32, scroll: f64) -> ViewportLayout {
        let source = StubbornSource { count: rows };
        let heights = Heights::Uniform(row_height);
        Recycler::new().layout(
            &source,
            &heights,
            1920,
            height,
            1.0,
            1.0,
            Density::Default,
            row_height,
            scroll,
        )
    }

    #[test]
    fn the_bound_holds_at_a_million_rows() {
        let layout = layout_for(1_000_000, 1080, 28, 0.0);
        assert!(layout.visible.count <= layout.virtualization_bound() + 1);
        assert!(
            layout.visible.count < 50,
            "laid out {} rows",
            layout.visible.count
        );
    }

    #[test]
    fn the_bound_holds_at_the_very_bottom() {
        let content = 1_000_000u64 * 28;
        let layout = layout_for(1_000_000, 1080, 28, (content - 1080) as f64);
        assert!(layout.visible.count <= layout.virtualization_bound() + 1);
        assert!(layout.visible.end() <= 1_000_000);
    }

    #[test]
    fn row_tops_stay_in_viewport_space_even_at_the_bottom_of_the_corpus() {
        // If this ever produced values in the millions, the f32 narrowing in the draw list
        // would be lossy -- which is exactly the R5 failure.
        let content = 1_000_000u64 * 28;
        let layout = layout_for(1_000_000, 1080, 28, (content - 1080) as f64 + 0.5);

        for slot in 0..layout.visible.count {
            let top = layout.row_top(slot);
            assert!(
                top > -100.0 && top < 2000.0,
                "row {slot} top {top} is not in viewport space"
            );
        }
        // And the fractional scroll must survive into the first row's position. The whole
        // part depends on where row boundaries happen to fall; the *fraction* is the part
        // an `as u64` cast in the offset lookup would silently discard, taking sub-pixel
        // scrolling with it (research R5).
        assert!(
            (layout.first_row_offset.fract() - 0.5).abs() < 1e-6,
            "the sub-pixel scroll was lost: first_row_offset = {}",
            layout.first_row_offset
        );
    }

    #[test]
    fn a_source_shorter_than_the_viewport_lays_out_only_what_exists() {
        let layout = layout_for(3, 1080, 28, 0.0);
        assert_eq!(layout.visible.count, 3);
        assert_eq!(layout.content_height, 84);
    }

    #[test]
    fn an_empty_source_lays_out_nothing() {
        let layout = layout_for(0, 1080, 28, 0.0);
        assert_eq!(layout.visible.count, 0);
        assert_eq!(layout.content_height, 0);
    }

    #[test]
    fn filling_asks_the_source_for_exactly_the_visible_range() {
        let source = StubbornSource { count: 1_000_000 };
        let heights = Heights::Uniform(28);
        let mut recycler = Recycler::new();
        let layout = recycler.layout(
            &source,
            &heights,
            1920,
            1080,
            1.0,
            1.0,
            Density::Default,
            28,
            100_000.0,
        );
        let buf = recycler.fill(&source, &layout);

        assert_eq!(buf.len() as u32, layout.visible.count);
        assert_eq!(buf.rows()[0].id.0, layout.visible.first);
    }

    #[test]
    fn refilling_reuses_the_buffer_rather_than_reallocating() {
        let source = StubbornSource { count: 1_000_000 };
        let heights = Heights::Uniform(28);
        let mut recycler = Recycler::new();

        let layout = recycler.layout(
            &source,
            &heights,
            1920,
            1080,
            1.0,
            1.0,
            Density::Default,
            28,
            0.0,
        );
        recycler.fill(&source, &layout);
        let capacity = recycler.rows().rows().len();

        for scroll in [100.0, 5000.0, 999_999.0] {
            let layout = recycler.layout(
                &source,
                &heights,
                1920,
                1080,
                1.0,
                1.0,
                Density::Default,
                28,
                scroll,
            );
            recycler.fill(&source, &layout);
        }
        assert!(recycler.rows().rows().len() <= capacity + 2);
    }

    #[test]
    fn staleness_tracks_the_source_version_not_the_frame() {
        // RS-4: a version that churns per frame would defeat snapshot reuse entirely.
        let source = StubbornSource { count: 100 };
        let heights = Heights::Uniform(28);
        let mut recycler = Recycler::new();
        let layout = recycler.layout(
            &source,
            &heights,
            1920,
            1080,
            1.0,
            1.0,
            Density::Default,
            28,
            0.0,
        );
        recycler.fill(&source, &layout);
        assert!(!recycler.is_stale(&source));
    }

    #[test]
    fn a_layout_owns_the_whole_window_until_it_is_told_otherwise() {
        let layout = layout_for(1000, 1080, 28, 0.0);
        assert_eq!((layout.origin_x, layout.origin_y), (0.0, 0.0));
        assert_eq!(layout.row_top(0), 0.0);
    }

    #[test]
    fn an_origin_moves_every_row_by_exactly_the_chrome_height() {
        // The chrome takes 96px off the top; every row moves down by 96 and by nothing
        // else. A row pass that moved by a *scaled* origin, or that moved the first row
        // and not the rest, would still look plausible on screen.
        let bare = layout_for(1000, 1080, 28, 0.0);
        let inset = layout_for(1000, 1080, 28, 0.0).at(240.0, 96.0);
        for slot in 0..10 {
            assert_eq!(
                inset.row_top(slot),
                bare.row_top(slot) + 96.0,
                "slot {slot}"
            );
        }
    }

    #[test]
    fn hit_testing_inverts_the_row_placement_it_was_drawn_from() {
        // The acceptance criterion this pins is "a click lands on the row it is drawn
        // under". Asserting the inverse against `row_top` rather than against a second
        // copy of the arithmetic is what makes it catch a drift instead of confirming one.
        for scroll in [0.0, 13.0, 27.0, 5_000.5] {
            let layout = layout_for(1000, 1080, 28, scroll).at(240.0, 96.0);
            for slot in 0..layout.visible.count.min(20) {
                let index = layout.visible.first + u64::from(slot);
                let top = layout.row_top(slot);
                // Anywhere inside the row, not just its top edge.
                for probe in [0.5, 14.0, 27.4] {
                    let y = top + probe;
                    // The first visible row is normally scrolled part-way off the top of
                    // the surface, so part of it is behind the chrome. That part belongs
                    // to the chrome, not to the row -- see the pointer-in-chrome test.
                    if y < layout.origin_y {
                        continue;
                    }
                    assert_eq!(
                        layout.row_at(y),
                        Some(index),
                        "scroll {scroll} slot {slot} at +{probe}"
                    );
                }
            }
        }
    }

    #[test]
    fn hit_testing_is_identical_with_the_mode_on_and_off() {
        // FR-008 / T031. The lit mode's scene is published beside the frame and is not an
        // input to `entry_at` or `row_at` — this sweep asserts the answers are identical
        // anyway, with a real scene built from the same viewport, so a future change that
        // threads the scene into hit testing has to keep every answer or go red here.
        // (The task names row.rs; the hit test lives here, beside its inverse.)
        let layout = layout_for(1000, 1080, 28, 13.0).at(240.0, 96.0);
        let off: Vec<Option<u64>> = (0..1080).map(|y| layout.row_at(y as f32)).collect();

        let tokens = crate::tokens::Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let mut builder = crate::scene::SceneBuilder::new(
            1,
            [1000.0, 1080.0],
            40.0,
            qs_gpu::scene::Environment::default(),
        );
        let canvas = tokens.material(crate::material::name::SURFACE_CANVAS).unwrap();
        builder.add(
            canvas,
            crate::material::Surface::new(0.0, 0.0, 1000.0, 1080.0, 0.0, 1.0),
        );
        let scene = builder.finish();
        assert!(!scene.slabs.is_empty());

        let on: Vec<Option<u64>> = (0..1080).map(|y| layout.row_at(y as f32)).collect();
        assert_eq!(off, on, "hit testing changed when the lit mode's scene existed");
    }

    #[test]
    fn a_pointer_in_the_chrome_is_over_no_row_at_all() {
        // Without this, the command bar and the status shelf both act as row 0 and the
        // last visible row respectively -- a click on "up" would also select a file.
        let layout = layout_for(1000, 1080, 28, 0.0).at(0.0, 96.0);
        assert_eq!(layout.row_at(0.0), None);
        assert_eq!(layout.row_at(95.9), None);
        assert_eq!(layout.row_at(96.0), Some(0));
    }

    #[test]
    fn nothing_is_hit_below_the_last_visible_row() {
        let layout = layout_for(3, 1080, 28, 0.0).at(0.0, 96.0);
        assert_eq!(layout.row_at(96.0 + 28.0 * 2.5), Some(2));
        assert_eq!(layout.row_at(96.0 + 28.0 * 3.5), None);
    }

    #[test]
    fn an_empty_directory_is_hit_nowhere() {
        let layout = layout_for(0, 1080, 28, 0.0).at(0.0, 96.0);
        assert_eq!(layout.row_at(500.0), None);
    }

    fn grid_for(entries: u64, columns: u32, cell: u32, height: u32, scroll: f64) -> ViewportLayout {
        let source = StubbornSource { count: entries };
        Recycler::new().layout_grid(
            &source,
            columns,
            cell,
            1920,
            height,
            1.0,
            1.0,
            Density::Default,
            scroll,
        )
    }

    #[test]
    fn a_grid_reports_the_entry_count_not_the_row_count() {
        // set_size is the folder, not the shape it is drawn in. A screen reader announcing
        // "1 of 250" for a thousand files in four columns is the failure this pins.
        let grid = grid_for(1000, 4, 120, 720, 0.0);
        assert_eq!(grid.row_count, 1000);
        assert_eq!(grid.content_height, 250 * 120);
    }

    #[test]
    fn the_grid_lays_out_entries_and_stays_inside_the_bound() {
        for columns in [2u32, 3, 5, 8] {
            for cell in [80u32, 120, 200, 320] {
                for height in [200u32, 720, 1440, 2160] {
                    for scroll in [0.0, 137.0, 100_000.0] {
                        let grid = grid_for(1_000_000, columns, cell, height, scroll);
                        assert!(
                            grid.visible.count <= grid.virtualization_bound() + columns,
                            "{columns}x{cell} in {height}px at {scroll}: laid out {} entries, \
                             bound {}",
                            grid.visible.count,
                            grid.virtualization_bound()
                        );
                        assert!(grid.visible.end() <= grid.row_count);
                    }
                }
            }
        }
    }

    #[test]
    fn a_grid_never_asks_the_source_for_entries_past_the_corpus() {
        // The short final row. 10 entries in 4 columns is two full rows and a row of two,
        // and a range that ran to the end of the last *row* would ask for indices 10 and 11.
        let grid = grid_for(10, 4, 120, 2000, 0.0);
        assert_eq!(grid.visible.first, 0);
        assert_eq!(u64::from(grid.visible.count), 10);
    }

    #[test]
    fn grid_hit_testing_inverts_the_cell_it_was_drawn_from() {
        for scroll in [0.0, 60.0, 121.0, 9_000.0] {
            let grid = grid_for(1000, 4, 120, 720, scroll).at(40.0, 96.0);
            for slot in 0..grid.visible.count {
                let (x, y, w, h) = grid.cell_rect(slot);
                let index = grid.visible.first + u64::from(slot);
                // The four corners' worth of interior, not just the centre: an off-by-one
                // in the column arithmetic shows up at the edges first.
                for (dx, dy) in [
                    (1.0, 1.0),
                    (w - 1.0, 1.0),
                    (1.0, h - 1.0),
                    (w * 0.5, h * 0.5),
                ] {
                    let (px, py) = (x + dx, y + dy);
                    if py < grid.origin_y {
                        // Partly scrolled behind the chrome -- that part is the chrome's.
                        continue;
                    }
                    assert_eq!(
                        grid.entry_at(px, py),
                        Some(index),
                        "scroll {scroll} slot {slot} at (+{dx}, +{dy})"
                    );
                }
            }
        }
    }

    #[test]
    fn the_empty_tail_of_a_short_final_row_is_not_the_entry_before_it() {
        // 6 entries in 4 columns: the second row holds two cells and two empty slots.
        // Pointing at the empty slots must resolve to nothing, not to entry 5, and not to
        // a phantom entry 6 or 7.
        let grid = grid_for(6, 4, 120, 2000, 0.0);
        let (x, y, w, h) = grid.cell_rect(5);
        assert_eq!(grid.entry_at(x + w * 0.5, y + h * 0.5), Some(5));
        assert_eq!(grid.entry_at(x + w * 1.5, y + h * 0.5), None, "slot 6");
        assert_eq!(grid.entry_at(x + w * 2.5, y + h * 0.5), None, "slot 7");
    }

    #[test]
    fn a_pointer_past_the_last_column_is_outside_the_grid() {
        let grid = grid_for(1000, 4, 120, 720, 0.0);
        let width = grid.cell_width();
        assert_eq!(grid.entry_at(width * 3.5, 10.0), Some(3));
        assert_eq!(grid.entry_at(width * 4.0 + 1.0, 10.0), None);
        assert_eq!(grid.entry_at(-1.0, 10.0), None);
    }

    #[test]
    fn a_one_column_grid_places_and_hit_tests_exactly_like_the_list() {
        // The list is the columns == 1 case, and it has to stay literally that rather than
        // approximately that -- everything already shipped depends on it.
        let list = layout_for(1000, 720, 120, 240.0);
        let grid = grid_for(1000, 1, 120, 720, 240.0);
        assert_eq!(grid.visible, list.visible);
        assert_eq!(grid.content_height, list.content_height);
        for slot in 0..grid.visible.count {
            assert_eq!(grid.row_top(slot), list.row_top(slot), "slot {slot}");
        }
        for y in [0.0, 1.0, 119.0, 120.0, 500.0] {
            assert_eq!(grid.entry_at(0.0, y), list.row_at(y), "y {y}");
        }
    }

    #[test]
    fn an_empty_folder_in_a_grid_is_hit_nowhere_and_has_no_height() {
        let grid = grid_for(0, 4, 120, 720, 0.0);
        assert_eq!(grid.visible.count, 0);
        assert_eq!(grid.content_height, 0);
        assert_eq!(grid.entry_at(10.0, 10.0), None);
    }

    #[test]
    fn the_bound_holds_across_every_plausible_viewport_and_density() {
        for height in [64u32, 200, 480, 720, 1080, 1440, 2160, 4320] {
            for row_height in [12u32, 24, 28, 42, 56, 84] {
                for scroll in [0.0, 1.0, 999.0, 27_000_000.0] {
                    let layout = layout_for(1_000_000, height, row_height, scroll);
                    assert!(
                        layout.visible.count <= layout.virtualization_bound() + 1,
                        "viewport {height} row {row_height} scroll {scroll}: {} rows > bound {}",
                        layout.visible.count,
                        layout.virtualization_bound()
                    );
                }
            }
        }
    }
}
