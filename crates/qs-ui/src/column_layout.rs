//! How the entry surface is divided into columns, and where the horizontal scroll sits.
//!
//! # One thing knows how the space is divided
//!
//! `chrome::Regions::carve` is the only thing that knows how the *window* is divided, and
//! [`ViewportLayout::at`] puts the resulting origin on the type every consumer already
//! reads. This is the same rule one level down: the entry surface is divided into columns
//! here and nowhere else, and each column's rect becomes one `ViewportLayout` origin.
//!
//! A second notion of "where column 3 starts" — a hit test that divides the width itself, a
//! renderer that adds up column widths as it goes — is the defect this shape exists to
//! prevent, and it shows up as clicking in one column and selecting in another.
//!
//! # Content space and viewport space
//!
//! A band's `x` is in **content** space: measured from the left edge of the first column,
//! ignoring the scroll. Subtracting the scroll offset gives viewport space. Keeping the two
//! named separately is what stops the horizontal scroll being added twice, which is the
//! horizontal version of the bug `ViewportLayout::origin` exists to prevent vertically.

use std::ops::Range;

/// One column's horizontal band, in content space.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ColumnBand {
    pub index: usize,
    /// Left edge, measured from the first column rather than from the viewport.
    pub x: f32,
    pub width: f32,
}

impl ColumnBand {
    /// Right edge, in content space.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }
}

/// The horizontal division of the entry surface into columns.
///
/// Uniform widths on purpose. Miller columns with per-column widths needs a resize
/// affordance per divider and a rule for what happens to the ones off screen, and neither
/// is designed — UXDD §5.1 asks for horizontal scrolling that follows focus and nothing
/// about resizing. A uniform width is the honest version of what is specified.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ColumnLayout {
    /// How wide the visible entry surface is.
    pub surface_width: f32,
    /// How wide one column is.
    pub column_width: f32,
    /// Space between columns. Not inside a column: a divider belongs to neither side.
    pub gap: f32,
    /// How many columns are open.
    pub count: usize,
}

impl ColumnLayout {
    /// A layout over `count` columns.
    ///
    /// `column_width` is clamped so a column can never be wider than the surface: a single
    /// column wider than the viewport can never be scrolled fully into view, so
    /// [`Self::scroll_to_show`] would have no answer to give.
    #[must_use]
    pub fn new(surface_width: f32, column_width: f32, gap: f32, count: usize) -> Self {
        Self {
            surface_width: surface_width.max(0.0),
            column_width: column_width.max(1.0).min(surface_width.max(1.0)),
            gap: gap.max(0.0),
            count,
        }
    }

    /// The band for `index`, in content space.
    #[must_use]
    pub fn band(&self, index: usize) -> Option<ColumnBand> {
        if index >= self.count {
            return None;
        }
        Some(ColumnBand {
            index,
            x: index as f32 * (self.column_width + self.gap),
            width: self.column_width,
        })
    }

    /// How wide all the columns are together, gaps between but not after.
    #[must_use]
    pub fn content_width(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.count as f32 * self.column_width + (self.count - 1) as f32 * self.gap
    }

    /// The furthest left the surface can be scrolled to.
    ///
    /// Zero when everything fits, which is what keeps a two-column layout in a wide window
    /// from being scrollable at all.
    #[must_use]
    pub fn max_scroll(&self) -> f32 {
        (self.content_width() - self.surface_width).max(0.0)
    }

    /// Clamp a proposed scroll offset into range.
    #[must_use]
    pub fn clamp_scroll(&self, scroll_x: f32) -> f32 {
        scroll_x.clamp(0.0, self.max_scroll())
    }

    /// Which columns intersect the viewport at `scroll_x`.
    ///
    /// A half-visible column is *in* the range: it has rows on screen, so it has to be laid
    /// out and it has to answer a hit test. Excluding it is how a column that is partly
    /// visible becomes unclickable.
    #[must_use]
    pub fn visible(&self, scroll_x: f32) -> Range<usize> {
        if self.count == 0 || self.surface_width <= 0.0 {
            return 0..0;
        }
        let scroll = self.clamp_scroll(scroll_x);
        let stride = self.column_width + self.gap;
        if stride <= 0.0 {
            return 0..self.count;
        }
        let first = (scroll / stride).floor().max(0.0) as usize;
        // `ceil` on the right edge, so a column showing one pixel is included.
        let last = (((scroll + self.surface_width) / stride).ceil() as usize).min(self.count);
        first.min(self.count)..last.max(first.min(self.count))
    }

    /// The smallest scroll offset that shows `index` completely.
    ///
    /// Minimal rather than centring: UXDD §5.1 says horizontal scroll *follows focus*, and a
    /// layout that recentres on every focus move throws the columns the user is reading
    /// sideways for no reason. A column already fully visible does not move at all.
    #[must_use]
    pub fn scroll_to_show(&self, index: usize, scroll_x: f32) -> f32 {
        let Some(band) = self.band(index) else {
            return self.clamp_scroll(scroll_x);
        };
        let scroll = self.clamp_scroll(scroll_x);
        let next = if band.x < scroll {
            band.x
        } else if band.right() > scroll + self.surface_width {
            band.right() - self.surface_width
        } else {
            scroll
        };
        self.clamp_scroll(next)
    }

    /// Which column contains viewport `x`, if any.
    ///
    /// `None` for a gap, which is deliberate: a click on a divider belongs to neither
    /// column, and picking the nearer one means a click one pixel either side of a boundary
    /// selects in two different columns.
    #[must_use]
    pub fn column_at(&self, x: f32, scroll_x: f32) -> Option<usize> {
        if x < 0.0 || x >= self.surface_width {
            return None;
        }
        let content_x = x + self.clamp_scroll(scroll_x);
        let stride = self.column_width + self.gap;
        if stride <= 0.0 {
            return None;
        }
        let index = (content_x / stride).floor() as usize;
        let band = self.band(index)?;
        // Past the column's own width is the gap after it.
        (content_x < band.right()).then_some(index)
    }

    /// Where `index` is drawn, in viewport space, or `None` when it is off screen.
    #[must_use]
    pub fn viewport_x(&self, index: usize, scroll_x: f32) -> Option<f32> {
        let band = self.band(index)?;
        let x = band.x - self.clamp_scroll(scroll_x);
        (x + band.width > 0.0 && x < self.surface_width).then_some(x)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    /// 300-wide columns, 4 gap, in a 1000-wide surface: 3.28 columns fit.
    fn layout(count: usize) -> ColumnLayout {
        ColumnLayout::new(1000.0, 300.0, 4.0, count)
    }

    #[test]
    fn bands_are_laid_out_left_to_right_with_the_gap_between_them() {
        let l = layout(3);
        assert_eq!(l.band(0).unwrap().x, 0.0);
        assert_eq!(l.band(1).unwrap().x, 304.0);
        assert_eq!(l.band(2).unwrap().x, 608.0);
        assert_eq!(l.band(3), None);
        // Gaps between, not after: three columns is two gaps.
        assert_eq!(l.content_width(), 908.0);
    }

    #[test]
    fn everything_fitting_means_nothing_scrolls() {
        let l = layout(3);
        assert_eq!(l.max_scroll(), 0.0);
        assert_eq!(l.clamp_scroll(500.0), 0.0);
        assert_eq!(l.scroll_to_show(2, 0.0), 0.0, "a visible column moved");
    }

    #[test]
    fn scrolling_follows_focus_minimally_rather_than_recentring() {
        // Six columns is 1824 wide in a 1000 surface. Showing the last one should move
        // exactly far enough and no further -- recentring would throw the columns the user
        // is reading sideways on every focus move.
        let l = layout(6);
        let last = l.band(5).unwrap();
        let scroll = l.scroll_to_show(5, 0.0);
        assert_eq!(scroll, last.right() - l.surface_width);
        assert_eq!(scroll, l.max_scroll(), "the last column pins to the end");

        // Now focus a column already fully on screen: nothing moves.
        assert_eq!(l.scroll_to_show(4, scroll), scroll);
    }

    #[test]
    fn scrolling_back_to_a_column_off_the_left_edge_shows_its_left_edge() {
        let l = layout(6);
        let scroll = l.max_scroll();
        let back = l.scroll_to_show(0, scroll);
        assert_eq!(back, 0.0);
        assert_eq!(l.band(0).unwrap().x, back, "column 0 is not flush left");
    }

    #[test]
    fn a_column_wider_than_the_surface_is_clamped_so_it_can_always_be_shown() {
        // Otherwise `scroll_to_show` has no answer: the column cannot be fully visible at
        // any offset, and the two branches disagree about which edge to align.
        let l = ColumnLayout::new(400.0, 900.0, 4.0, 3);
        assert_eq!(l.column_width, 400.0);
        let scroll = l.scroll_to_show(2, 0.0);
        assert!(scroll <= l.max_scroll());
        assert_eq!(l.scroll_to_show(2, scroll), scroll, "it did not settle");
    }

    #[test]
    fn a_half_visible_column_is_still_laid_out() {
        // It has rows on screen. Excluding it is how a partly visible column becomes
        // unclickable and draws as a blank band.
        let l = layout(6);
        let visible = l.visible(0.0);
        assert_eq!(visible.start, 0);
        assert!(
            visible.end >= 4,
            "the column straddling the right edge was dropped: {visible:?}"
        );
    }

    #[test]
    fn hit_testing_returns_the_column_under_the_point() {
        let l = layout(3);
        assert_eq!(l.column_at(0.0, 0.0), Some(0));
        assert_eq!(l.column_at(299.0, 0.0), Some(0));
        assert_eq!(l.column_at(304.0, 0.0), Some(1));
        assert_eq!(l.column_at(907.0, 0.0), Some(2));
    }

    #[test]
    fn a_click_in_the_gap_belongs_to_neither_column() {
        // Picking the nearer one means one pixel either side of a divider selects in two
        // different columns, which is the same class of bug as clicking row 7 and getting
        // row 5.
        let l = layout(3);
        assert_eq!(l.column_at(301.0, 0.0), None, "the gap answered");
        assert_eq!(l.column_at(303.0, 0.0), None);
    }

    #[test]
    fn hit_testing_accounts_for_the_scroll_exactly_once() {
        // The horizontal version of the bug `ViewportLayout::origin` exists to prevent:
        // adding the scroll twice puts the answer a whole column out.
        let l = layout(6);
        let scroll = 304.0; // exactly one column
        assert_eq!(l.column_at(0.0, scroll), Some(1));
        assert_eq!(l.column_at(304.0, scroll), Some(2));
    }

    #[test]
    fn hit_testing_outside_the_surface_answers_nothing() {
        let l = layout(3);
        assert_eq!(l.column_at(-1.0, 0.0), None);
        assert_eq!(l.column_at(1000.0, 0.0), None);
    }

    #[test]
    fn viewport_x_and_column_at_are_inverses_for_every_visible_column() {
        // The property that keeps drawing and hit testing from disagreeing.
        let l = layout(6);
        for scroll in [0.0, 100.0, 304.0, l.max_scroll()] {
            for index in l.visible(scroll) {
                let Some(x) = l.viewport_x(index, scroll) else {
                    continue;
                };
                // A pixel inside the column, clamped into the surface for the half-visible
                // ones at either edge.
                let probe = (x + 1.0).clamp(0.0, l.surface_width - 1.0);
                if probe >= x && probe < x + l.column_width {
                    assert_eq!(
                        l.column_at(probe, scroll),
                        Some(index),
                        "drawing put column {index} at {x} but the hit test disagreed at {probe} (scroll {scroll})"
                    );
                }
            }
        }
    }

    #[test]
    fn an_empty_stack_lays_out_nothing_rather_than_dividing_by_zero() {
        let l = layout(0);
        assert_eq!(l.content_width(), 0.0);
        assert_eq!(l.visible(0.0), 0..0);
        assert_eq!(l.column_at(10.0, 0.0), None);
        assert_eq!(l.band(0), None);
    }
}
