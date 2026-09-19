//! Scroll offsets, fling integration and viewport-relative coordinates.
//!
//! Content offsets use `f64`: a million 28-pixel rows exceed the range where
//! `f32` can represent each pixel. Positions are rebased against the viewport
//! before conversion to `f32` for rendering.
//!
//! The fling integrator handles the scroll bounds before storing its next
//! offset, so motion settles when it reaches either end of the list.

use crate::fenwick::Heights;

/// Physics for a fling. The numbers are the ones that make a trackpad flick feel like the
/// platform's own list, and they are here rather than in tokens because they are motion,
/// not appearance.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FlingParams {
    /// Fraction of velocity retained per second. 0.001 means "1/1000 after one second",
    /// which reads as a firm, quick stop.
    pub friction: f64,
    /// Speed below which a fling stops, in physical pixels per second.
    pub stop_speed: f64,
    /// Maximum speed, to stop a pathological input event from crossing the whole corpus.
    pub max_speed: f64,
}

impl Default for FlingParams {
    fn default() -> Self {
        Self {
            friction: 0.001,
            stop_speed: 12.0,
            max_speed: 40_000.0,
        }
    }
}

/// Content-space scroll position and velocity.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct ScrollState {
    /// Content-space offset in physical pixels. `f64` -- see the module docs.
    offset: f64,
    /// Physical pixels per second.
    velocity: f64,
    params: FlingParams,
}

impl ScrollState {
    pub fn new() -> Self {
        Self {
            offset: 0.0,
            velocity: 0.0,
            params: FlingParams::default(),
        }
    }

    pub fn with_params(params: FlingParams) -> Self {
        Self {
            offset: 0.0,
            velocity: 0.0,
            params,
        }
    }

    pub fn offset(self) -> f64 {
        self.offset
    }

    pub fn velocity(self) -> f64 {
        self.velocity
    }

    /// Whether a fling is in progress. Drives whether the frame loop keeps an animation
    /// ticket open -- when this is false and nothing else is animating, the loop sleeps.
    pub fn is_animating(self) -> bool {
        self.velocity != 0.0
    }

    /// Largest valid offset for this content and viewport.
    ///
    /// Content shorter than the viewport pins to zero rather than producing a negative
    /// maximum, which would let the list scroll upward into empty space.
    pub fn max_offset(content_height: u64, viewport_height: u32) -> f64 {
        (content_height as f64 - f64::from(viewport_height)).max(0.0)
    }

    /// Jump to an absolute offset, clamped.
    pub fn set_offset(&mut self, offset: f64, content_height: u64, viewport_height: u32) {
        self.offset = clamp(offset, content_height, viewport_height);
    }

    /// Scroll by a delta, cancelling any fling.
    ///
    /// A wheel notch or a scrollbar drag is a direct manipulation and should override
    /// momentum immediately -- a flick that keeps coasting after the user grabs the
    /// scrollbar feels broken.
    pub fn scroll_by(&mut self, delta: f64, content_height: u64, viewport_height: u32) {
        self.velocity = 0.0;
        self.offset = clamp(self.offset + delta, content_height, viewport_height);
    }

    /// Start a fling at `velocity` physical pixels per second.
    pub fn fling(&mut self, velocity: f64) {
        self.velocity = if velocity.is_finite() {
            velocity.clamp(-self.params.max_speed, self.params.max_speed)
        } else {
            0.0
        };
    }

    pub fn stop(&mut self) {
        self.velocity = 0.0;
    }

    /// Advance the fling by `dt` seconds. Returns `true` while still moving.
    ///
    /// The clamp is applied to the candidate offset **before** it is stored, and hitting a
    /// bound kills the velocity in the same step. See the module docs for why the order is
    /// not negotiable.
    pub fn advance(&mut self, dt: f64, content_height: u64, viewport_height: u32) -> bool {
        // `is_sign_positive` alone would admit NaN; the explicit comparison rejects a
        // non-finite step, which a stalled frame clock can produce.
        if self.velocity == 0.0 || !dt.is_finite() || dt <= 0.0 {
            return self.velocity != 0.0;
        }

        let max = Self::max_offset(content_height, viewport_height);

        // Exponential decay integrated exactly over the step, rather than
        // `v *= friction * dt`. An Euler step makes the deceleration frame-rate dependent,
        // so the same flick travels a different distance at 60 Hz and 120 Hz -- which would
        // also make two benchmark runs at different present modes incomparable.
        let decay = self.params.friction.powf(dt);
        let travelled = if (self.params.friction.ln()).abs() > f64::EPSILON {
            self.velocity * (decay - 1.0) / self.params.friction.ln()
        } else {
            self.velocity * dt
        };

        let candidate = self.offset + travelled;

        if candidate <= 0.0 {
            self.offset = 0.0;
            self.velocity = 0.0;
            return false;
        }
        if candidate >= max {
            self.offset = max;
            self.velocity = 0.0;
            return false;
        }

        self.offset = candidate;
        self.velocity *= decay;

        if self.velocity.abs() < self.params.stop_speed {
            self.velocity = 0.0;
            return false;
        }
        true
    }

    /// Reposition after the content height changes, keeping `anchor` in view.
    pub fn rebase_for_content_change(
        &mut self,
        anchor_index: u64,
        anchor_viewport_y: f64,
        heights: &Heights,
        content_height: u64,
        viewport_height: u32,
    ) {
        let anchor_top = heights.offset_of(anchor_index) as f64;
        self.offset = clamp(
            anchor_top - anchor_viewport_y,
            content_height,
            viewport_height,
        );
    }
}

fn clamp(offset: f64, content_height: u64, viewport_height: u32) -> f64 {
    if !offset.is_finite() {
        return 0.0;
    }
    offset.clamp(
        0.0,
        ScrollState::max_offset(content_height, viewport_height),
    )
}

/// The visible slice of a corpus.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct VisibleRange {
    pub first: u64,
    pub count: u32,
}

impl VisibleRange {
    pub fn end(self) -> u64 {
        self.first + u64::from(self.count)
    }

    pub fn as_range(self) -> std::ops::Range<u64> {
        self.first..self.end()
    }
}

/// Visible row range and the clipped portion of its first row.
///
/// The count allows an extra row at each edge for partial visibility.
pub fn visible_range(
    offset: f64,
    viewport_height: u32,
    row_count: u64,
    heights: &Heights,
) -> (VisibleRange, f64) {
    if row_count == 0 || viewport_height == 0 {
        return (VisibleRange::default(), 0.0);
    }

    let offset = offset.max(0.0);

    // Split into whole and fractional parts before indexing. The index lookup works in
    // integer content pixels -- that is what makes the Fenwick tree exact -- but casting the
    // offset straight to `u64` would discard the sub-pixel scroll, and sub-pixel scrolling
    // is the entire reason this file is written in `f64`. The fraction is carried through to
    // `first_row_offset`, which is what positions the first visible row.
    let whole = offset.floor();
    let fraction = offset - whole;
    let (first, within) = heights.index_at(whole as u64, row_count);
    let within = within as f64 + fraction;

    let count = match heights {
        Heights::Uniform(h) => {
            let h = u64::from((*h).max(1));
            // ceil(viewport / row) + 2, computed in integers so a fractional row height
            // cannot round the bound down and leave a gap at the bottom.
            let visible = u64::from(viewport_height).div_ceil(h) + 2;
            visible.min(row_count - first) as u32
        }
        Heights::Variable(tree) => {
            // Walk until the accumulated height covers the viewport. Bounded by the same
            // `+ 2` slack: variable heights make the count data-dependent, but never
            // unbounded.
            let start_offset = tree.offset_of(first as usize);
            let target = start_offset + u64::from(viewport_height) + within.ceil() as u64;
            let mut index = first;
            while index < row_count && tree.offset_of(index as usize) < target {
                index += 1;
            }
            ((index - first) + 2).min(row_count - first) as u32
        }
    };

    (VisibleRange { first, count }, within)
}

/// Convert a content position to viewport-relative `f32` coordinates.
///
/// Subtraction happens in `f64` before narrowing. Reversing that order
/// loses subpixel precision near the end of a long list.
#[inline]
pub fn rebase_to_viewport(content_y: f64, scroll_offset: f64) -> f32 {
    (content_y - scroll_offset) as f32
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;
    use std::sync::Arc;

    use crate::fenwick::Fenwick;

    const ROWS: u64 = 1_000_000;
    const ROW_H: u32 = 28;
    const VIEWPORT: u32 = 1080;

    fn uniform() -> Heights {
        Heights::Uniform(ROW_H)
    }

    fn content() -> u64 {
        uniform().total(ROWS)
    }

    #[test]
    fn offsets_stay_exact_at_the_bottom_of_a_million_rows() {
        // The R5 regression test. In f32, 27_999_972.0 and 27_999_974.0 are the same value,
        // so a sub-pixel scroll here would snap or jitter.
        let heights = uniform();
        let last_top = heights.offset_of(ROWS - 1) as f64;
        assert_eq!(last_top, 27_999_972.0);

        for fraction in [0.0, 0.25, 0.5, 0.75] {
            let offset = last_top + fraction;
            let rebased = rebase_to_viewport(last_top, offset);
            assert!(
                (rebased - (-fraction as f32)).abs() < 0.001,
                "rebasing lost the fractional part at the bottom of the corpus"
            );
        }
    }

    #[test]
    fn rebasing_is_what_makes_f32_safe() {
        // Demonstrates the failure the rebasing prevents, so the test documents the bug and
        // not just the fix. Narrowing before subtracting loses the offset entirely.
        let content_y = 27_999_972.0f64;
        let scroll = 27_999_971.5f64;

        let correct = rebase_to_viewport(content_y, scroll);
        assert!((correct - 0.5).abs() < 0.001);

        let naive = content_y as f32 - scroll as f32;
        assert!(
            (naive - 0.5).abs() > 0.1,
            "if this ever passes, f32 got more precise and the comment above is stale"
        );
    }

    #[test]
    fn scrolling_clamps_at_both_ends() {
        let mut s = ScrollState::new();
        s.scroll_by(-1000.0, content(), VIEWPORT);
        assert_eq!(s.offset(), 0.0);

        s.scroll_by(f64::MAX, content(), VIEWPORT);
        assert_eq!(s.offset(), ScrollState::max_offset(content(), VIEWPORT));
    }

    #[test]
    fn a_non_finite_offset_does_not_poison_the_scroll_state() {
        let mut s = ScrollState::new();
        s.set_offset(f64::NAN, content(), VIEWPORT);
        assert_eq!(s.offset(), 0.0);
        s.set_offset(f64::INFINITY, content(), VIEWPORT);
        assert!(s.offset().is_finite());
    }

    #[test]
    fn content_shorter_than_the_viewport_pins_to_zero() {
        let short = 100u64;
        assert_eq!(ScrollState::max_offset(short, VIEWPORT), 0.0);
        let mut s = ScrollState::new();
        s.scroll_by(500.0, short, VIEWPORT);
        assert_eq!(s.offset(), 0.0);
    }

    #[test]
    fn a_fling_settles_and_stops_rather_than_asymptoting() {
        // SC-003 depends on this: a fling that never formally ends keeps an animation
        // ticket open and the loop never goes idle.
        let mut s = ScrollState::new();
        s.set_offset(1_000_000.0, content(), VIEWPORT);
        s.fling(8000.0);

        let mut frames = 0;
        while s.advance(1.0 / 120.0, content(), VIEWPORT) {
            frames += 1;
            assert!(frames < 100_000, "the fling never terminated");
        }
        assert!(!s.is_animating());
        assert!(frames > 10, "the fling ended suspiciously fast");
    }

    #[test]
    fn hitting_an_end_stop_kills_the_velocity_in_the_same_step() {
        // The oscillation bug: clamp after integrating and the velocity survives, so the
        // next frame overshoots again and the list judders at the end stop.
        let mut s = ScrollState::new();
        s.set_offset(10.0, content(), VIEWPORT);
        s.fling(-50_000.0);

        s.advance(1.0 / 60.0, content(), VIEWPORT);
        assert_eq!(s.offset(), 0.0);
        assert_eq!(s.velocity(), 0.0, "velocity must not survive the clamp");
    }

    #[test]
    fn fling_distance_does_not_depend_on_the_frame_rate() {
        // Two runs of the same flick at 60 Hz and 120 Hz must travel the same distance, or
        // a Fifo run and a Mailbox run of the same scenario are not comparable.
        let distance = |dt: f64| {
            let mut s = ScrollState::new();
            s.set_offset(1_000_000.0, content(), VIEWPORT);
            s.fling(10_000.0);
            let start = s.offset();
            while s.advance(dt, content(), VIEWPORT) {}
            s.offset() - start
        };

        let at_60 = distance(1.0 / 60.0);
        let at_120 = distance(1.0 / 120.0);
        let error = (at_60 - at_120).abs() / at_60.max(1.0);
        assert!(
            error < 0.02,
            "fling distance is frame-rate dependent: {at_60} vs {at_120}"
        );
    }

    #[test]
    fn direct_manipulation_cancels_momentum() {
        let mut s = ScrollState::new();
        s.set_offset(5000.0, content(), VIEWPORT);
        s.fling(9000.0);
        assert!(s.is_animating());
        s.scroll_by(100.0, content(), VIEWPORT);
        assert!(
            !s.is_animating(),
            "grabbing the scrollbar must stop the fling"
        );
    }

    #[test]
    fn a_pathological_fling_velocity_is_clamped() {
        let mut s = ScrollState::new();
        s.fling(f64::INFINITY);
        assert_eq!(s.velocity(), 0.0);
        s.fling(1e30);
        assert!(s.velocity().abs() <= FlingParams::default().max_speed);
    }

    #[test]
    fn the_visible_count_honours_the_virtualization_bound() {
        // FR-002. This bound is the definition of "virtualized".
        let heights = uniform();
        let bound = (VIEWPORT / ROW_H) + 2;

        for offset in [0.0, 1.0, 13.7, 999_999.5, 27_998_000.0] {
            let (range, _) = visible_range(offset, VIEWPORT, ROWS, &heights);
            assert!(
                range.count <= bound + 1,
                "offset {offset} produced {} rows, bound is {bound}",
                range.count
            );
            assert!(range.end() <= ROWS);
        }
    }

    #[test]
    fn the_visible_range_covers_the_viewport_with_no_gap() {
        let heights = uniform();
        for offset in [0.0, 5.0, 27.9, 28.0, 100_000.3] {
            let (range, within) = visible_range(offset, VIEWPORT, ROWS, &heights);
            let covered = u64::from(range.count) * u64::from(ROW_H);
            assert!(
                covered >= u64::from(VIEWPORT) + within as u64,
                "offset {offset} left a gap at the bottom"
            );
        }
    }

    #[test]
    fn the_variable_height_path_agrees_with_the_uniform_one() {
        let count = 10_000u64;
        let tree = Fenwick::from_heights(&vec![ROW_H; count as usize]);
        let variable = Heights::Variable(Arc::new(tree));
        let uniform = Heights::Uniform(ROW_H);

        for offset in [0.0, 27.0, 28.0, 5000.0, 100_000.0] {
            let (a, wa) = visible_range(offset, VIEWPORT, count, &uniform);
            let (b, wb) = visible_range(offset, VIEWPORT, count, &variable);
            assert_eq!(a.first, b.first, "offset {offset}");
            assert_eq!(wa, wb);
            // Counts may differ by the slack row; both must cover the viewport.
            assert!(b.count >= (VIEWPORT / ROW_H));
        }
    }

    #[test]
    fn an_empty_corpus_produces_an_empty_range() {
        let (range, within) = visible_range(0.0, VIEWPORT, 0, &uniform());
        assert_eq!(range.count, 0);
        assert_eq!(within, 0.0);

        let (range, _) = visible_range(0.0, 0, ROWS, &uniform());
        assert_eq!(range.count, 0);
    }

    #[test]
    fn growing_content_keeps_the_anchor_row_in_place() {
        // RC-4: M1's enumeration streams, so the count grows while the user scrolls.
        let heights = uniform();
        let mut s = ScrollState::new();
        s.set_offset(280_000.0, content(), VIEWPORT);

        let (range, within) = visible_range(s.offset(), VIEWPORT, ROWS, &heights);
        let anchor = range.first;

        s.rebase_for_content_change(anchor, -within, &heights, content() * 2, VIEWPORT);

        let (after, _) = visible_range(s.offset(), VIEWPORT, ROWS * 2, &heights);
        assert_eq!(after.first, anchor, "the anchor row drifted");
    }
}
