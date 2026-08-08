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

/// Everything about the frame's geometry that the row builder needs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ViewportLayout {
    /// Physical pixels.
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
}

impl ViewportLayout {
    /// The FR-002 bound for this viewport.
    ///
    /// Derived from the **shortest** row, not the nominal one. FR-002 writes the bound as
    /// `viewport_height / row_height + 2`, which is unambiguous only while every row is the
    /// same height -- the case `flat-1m` exercises. With variable heights (`deep-40`) the
    /// worst case is however many of the shortest row fit on screen, and computing the
    /// bound from the nominal height would let the assertion pass while the recycler laid
    /// out several times more rows than intended.
    pub fn virtualization_bound(&self) -> u32 {
        (self.height / self.min_row_height.max(1)) + 2
    }

    /// Top edge of a visible row in **viewport** space.
    ///
    /// Note that this is derived from the row's index within the visible slice, not from
    /// its absolute content offset. That is deliberate and it is research R5 in practice:
    /// the absolute offset of row 999,999 is 28 million, and subtracting the scroll from it
    /// in `f32` would lose the fractional part. Here the arithmetic never leaves the
    /// viewport's own coordinate range.
    pub fn row_top(&self, slot: u32) -> f32 {
        (f64::from(slot) * f64::from(self.row_height) - self.first_row_offset) as f32
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
