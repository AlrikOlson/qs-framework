//! Property tests for the visible-row count bound.
//!
//! The cases vary viewport size, row height and scroll position, including
//! partially visible rows at either edge.

// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the
// vocabulary of a test, not a hazard in one. The workspace lints deny them for
// production code, so each test binary opts out at its root.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
use proptest::prelude::*;
use qs_ui::density::Density;
use qs_ui::fenwick::{Fenwick, Heights};
use qs_ui::recycler::Recycler;
use qs_ui::row_source::{RowSource, StubbornSource};

/// Visible-row bound with the extra row needed for fractional scrolling.
fn bound(viewport_height: u32, row_height: u32) -> u32 {
    viewport_height / row_height.max(1) + 3
}

fn lay_out(rows: u64, viewport: u32, row_height: u32, scroll: f64) -> u32 {
    let source = StubbornSource { count: rows };
    let heights = Heights::Uniform(row_height);
    Recycler::new()
        .layout(
            &source,
            &heights,
            1920,
            viewport,
            1.0,
            1.0,
            Density::Default,
            row_height,
            scroll,
        )
        .visible
        .count
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn the_bound_holds_for_any_viewport_and_scroll(
        viewport in 1u32..4320,
        row_height in 1u32..200,
        scroll in 0f64..30_000_000.0,
    ) {
        let count = lay_out(1_000_000, viewport, row_height, scroll);
        prop_assert!(
            count <= bound(viewport, row_height),
            "viewport {} / row {} at scroll {} laid out {} rows, bound is {}",
            viewport, row_height, scroll, count, bound(viewport, row_height)
        );
    }

    /// A source smaller than the viewport must lay out what exists and no more -- never
    /// padding out to the bound with rows that are not there.
    #[test]
    fn a_short_source_lays_out_only_what_exists(rows in 0u64..40) {
        let count = lay_out(rows, 1080, 28, 0.0);
        prop_assert!(u64::from(count) <= rows);
    }

    /// The range never runs past the end of the corpus, at any scroll position.
    #[test]
    fn the_range_never_exceeds_the_corpus(
        rows in 1u64..100_000,
        scroll in 0f64..10_000_000.0,
    ) {
        let source = StubbornSource { count: rows };
        let heights = Heights::Uniform(28);
        let layout = Recycler::new().layout(
            &source, &heights, 1920, 1080, 1.0, 1.0, Density::Default, 28, scroll,
        );
        prop_assert!(layout.visible.end() <= rows);
    }

    /// Variable heights are bounded too. This is the path `deep-40` exercises, and it is
    /// the one where an unbounded walk is easiest to write by accident.
    #[test]
    fn variable_heights_are_bounded(
        heights_vec in prop::collection::vec(8u32..80, 100..3000),
        scroll in 0f64..200_000.0,
    ) {
        let count = heights_vec.len() as u64;
        let tallest = heights_vec.iter().copied().max().unwrap_or(28);
        let shortest = heights_vec.iter().copied().min().unwrap_or(28);
        let heights = Heights::Variable(std::sync::Arc::new(Fenwick::from_heights(&heights_vec)));

        let source = StubbornSource { count };
        let layout = Recycler::new().layout(
            &source, &heights, 1920, 1080, 1.0, 1.0, Density::Default, tallest, scroll,
        );

        // With variable heights the bound is set by the *shortest* row: that is how many
        // can fit in the viewport at once. The layout reports the same figure, so this
        // also checks the recycler derived it from the source rather than from the
        // nominal row height it was handed.
        let worst_case = 1080 / shortest.max(1) + 3;
        prop_assert_eq!(layout.min_row_height, shortest);
        prop_assert!(
            layout.visible.count <= worst_case,
            "laid out {} rows, worst case is {}",
            layout.visible.count, worst_case
        );
    }
}

#[test]
fn the_bound_holds_at_every_exact_row_boundary() {
    // Off-by-one errors in virtualization hide exactly here: at a scroll position that is
    // an exact multiple of the row height, and one pixel either side of it.
    let row_height = 28u32;
    for row in [0u64, 1, 2, 999, 500_000, 999_997, 999_998] {
        let exact = (row * u64::from(row_height)) as f64;
        for scroll in [exact - 1.0, exact, exact + 1.0, exact + 0.5] {
            if scroll < 0.0 {
                continue;
            }
            let count = lay_out(1_000_000, 1080, row_height, scroll);
            assert!(
                count <= bound(1080, row_height),
                "row {row} at scroll {scroll}: {count} rows"
            );
        }
    }
}

#[test]
fn the_bound_is_tight_rather_than_merely_satisfied() {
    // A recycler that laid out one row would also satisfy the bound. The bound is only
    // meaningful if the range also *covers* the viewport, so this asserts the count is
    // close to the theoretical minimum rather than far below it.
    let count = lay_out(1_000_000, 1080, 28, 100_000.0);
    let minimum = 1080 / 28; // 38
    assert!(
        count >= minimum && count <= minimum + 3,
        "expected about {minimum} rows, got {count}"
    );
}

#[test]
fn an_enormous_viewport_still_bounds_correctly() {
    // An 8K display in portrait, at the compact density.
    let count = lay_out(1_000_000, 7680, 24, 1_000_000.0);
    assert!(count <= bound(7680, 24), "{count} rows");
}

#[test]
fn a_source_that_grows_between_frames_does_not_break_the_bound() {
    // RC-4: M1's directory enumeration streams, so `len()` grows while the user scrolls.
    let heights = Heights::Uniform(28);
    let recycler = Recycler::new();
    let mut previous = 0;

    for rows in [100u64, 1_000, 50_000, 1_000_000] {
        let source = StubbornSource { count: rows };
        let layout = recycler.layout(
            &source,
            &heights,
            1920,
            1080,
            1.0,
            1.0,
            Density::Default,
            28,
            2000.0,
        );
        assert!(layout.visible.count <= bound(1080, 28));
        assert!(layout.visible.end() <= source.len());
        previous = layout.visible.count.max(previous);
    }
    assert!(previous > 0);
}
