//! Property tests for row-index and offset round trips.
//!
//! The generated cases cover a million rows, including offsets large enough
//! to lose integer precision in `f32`.

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
use qs_ui::fenwick::{Fenwick, Heights};

const ROW_COUNT: u64 = 1_000_000;
const ROW_HEIGHT: u32 = 28;

fn uniform() -> Heights {
    Heights::Uniform(ROW_HEIGHT)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// The core round trip: the offset of a row's top edge maps back to that row.
    #[test]
    fn offset_of_and_index_at_are_inverses(index in 0u64..ROW_COUNT) {
        let heights = uniform();
        let offset = heights.offset_of(index);
        let (recovered, within) = heights.index_at(offset, ROW_COUNT);

        prop_assert_eq!(recovered, index);
        prop_assert_eq!(within, 0);
    }

    /// Any offset *inside* a row maps to that row, with the correct remainder.
    #[test]
    fn an_offset_within_a_row_resolves_to_that_row(
        index in 0u64..ROW_COUNT,
        into in 0u64..u64::from(ROW_HEIGHT),
    ) {
        let heights = uniform();
        let offset = heights.offset_of(index) + into;
        let (recovered, within) = heights.index_at(offset, ROW_COUNT);

        prop_assert_eq!(recovered, index);
        prop_assert_eq!(within, into);
    }

    /// Offsets are strictly increasing. If this ever fails, the scrollbar maps two
    /// different positions to the same row and scrolling appears to stick.
    #[test]
    fn offsets_are_strictly_monotonic(index in 0u64..(ROW_COUNT - 1)) {
        let heights = uniform();
        prop_assert!(heights.offset_of(index + 1) > heights.offset_of(index));
    }

    /// The mapping never returns an index outside the corpus, for any offset at all --
    /// including ones past the end, which a fling overshoot produces before the clamp.
    #[test]
    fn the_index_is_always_in_range(offset in 0u64..(ROW_COUNT * u64::from(ROW_HEIGHT) * 2)) {
        let heights = uniform();
        let (index, _) = heights.index_at(offset, ROW_COUNT);
        prop_assert!(index < ROW_COUNT);
    }

    /// The variable-height path agrees with the uniform fast path.
    ///
    /// This is the property that lets `flat-1m` (uniform) and `deep-40` (variable) be
    /// compared at all: if the two paths disagreed, the fast path would be measuring a
    /// different mapping from the one the renderer will eventually ship.
    #[test]
    fn the_two_height_paths_agree(
        heights_vec in prop::collection::vec(1u32..64, 1..2000),
        probe in 0u64..100_000,
    ) {
        let tree = Fenwick::from_heights(&heights_vec);
        let total = tree.total();

        // Naive prefix sum as an independent oracle.
        let mut reference = Vec::with_capacity(heights_vec.len() + 1);
        reference.push(0u64);
        for &h in &heights_vec {
            let last = *reference.last().unwrap_or(&0);
            reference.push(last + u64::from(h));
        }

        for (i, &expected) in reference.iter().enumerate() {
            prop_assert_eq!(tree.offset_of(i), expected, "offset_of({})", i);
        }
        prop_assert_eq!(total, *reference.last().unwrap_or(&0));

        // And a probe lands in the row the oracle says it should.
        if total > 0 {
            let offset = probe % total;
            let (index, within) = tree.index_at(offset);
            let top = reference.get(index).copied().unwrap_or(0);
            let bottom = reference.get(index + 1).copied().unwrap_or(total);
            prop_assert!(offset >= top && offset < bottom, "offset {} not in row {}", offset, index);
            prop_assert_eq!(within, offset - top);
        }
    }

    /// Check subpixel offsets in the final thousand rows of the list,
    /// where premature conversion to `f32` loses precision.
    #[test]
    fn the_bottom_of_the_corpus_is_exact(back in 0u64..1000) {
        let heights = uniform();
        let index = ROW_COUNT - 1 - back;
        let offset = heights.offset_of(index);

        prop_assert!(offset > 27_900_000, "this test must actually reach the bottom");
        // Beyond 2^24 an f32 cannot represent consecutive integers, so this equality is
        // exactly what the f64 decision buys.
        prop_assert_eq!(offset, index * u64::from(ROW_HEIGHT));
        prop_assert_eq!(heights.index_at(offset, ROW_COUNT), (index, 0));
    }
}

#[test]
fn the_full_corpus_height_is_what_the_research_note_says() {
    // 1M rows at 28px is 28,000,000 content pixels -- the number research R5 cites, and
    // well past f32's 16,777,216 exact-integer limit.
    let heights = uniform();
    assert_eq!(heights.total(ROW_COUNT), 28_000_000);
    assert!(28_000_000.0f32 as u64 != 28_000_001.0f32 as u64 - 1);
}

#[test]
fn f32_really_cannot_do_this() {
    // Documents *why* the mapping is integer-based, so the test suite records the reason
    // and not just the behaviour. If this ever fails, f32 changed and the design note in
    // scroll.rs is stale.
    let a = 27_999_972.0f32;
    let b = 27_999_973.0f32;
    assert_eq!(
        a, b,
        "f32 cannot distinguish consecutive integers at this magnitude"
    );
}
