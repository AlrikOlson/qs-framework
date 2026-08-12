//! What is selected.
//!
//! # Why this is a set of runs and not a set of indices
//!
//! `Ctrl+A` in a folder of a million entries is one keystroke, and the two obvious
//! representations both answer it badly: a `HashSet<u64>` allocates a million times and a
//! bitset allocates 125 KB per pane per directory. A sorted, disjoint list of half-open
//! ranges answers it with **one** run, and answers the gestures that produce it -- a range
//! click, a marquee drag, an invert of a range selection -- with a handful more. The
//! representation is chosen by the operations that are actually cheap in it, which is the
//! same reason the directory cache stores an arena rather than a `Vec<String>`.
//!
//! # Why it lives here rather than in the shell
//!
//! Selection is view state, exactly like [`ScrollState`](crate::scroll::ScrollState): two
//! panes showing the same directory have different selections, so it cannot live on the
//! `RowSource` any more than hover can. And it is an **index** set, because this crate is
//! not allowed to know what a file is. Making a selection outlive a navigation means
//! mapping those indices back to entries, and that mapping needs names -- so it happens one
//! layer up, in the application, where names already are.
//!
//! # The anchor is not the lead, and the base is neither
//!
//! A range click extends from the **anchor** -- the last row a plain or toggle click landed
//! on -- to the row just clicked, which becomes the **lead**. Shift-clicking twice in a row
//! must therefore re-range from the same anchor rather than growing from the previous
//! shift-click, which is what every list on every platform does and what collapsing the two
//! into one "current row" silently breaks.
//!
//! Replacing the whole selection with the new range gets that second shift-click right and
//! a different case wrong: click 2, `Ctrl`-click 8, `Shift`-click 10 leaves 2 selected in
//! both Explorer and Finder. So a range click is applied over a **base** -- everything that
//! was committed before the range gesture began -- rather than over nothing or over the
//! previous range. Every other gesture commits its result as the new base, which is why
//! `range_to` is the one method that does not.

use std::ops::Range;

/// A set of selected entry indices.
///
/// Indices are logical corpus ordinals -- the same numbering
/// [`RowSource::len`](crate::row_source::RowSource::len) counts and the accessibility tree
/// announces, never a slot in the recycled row buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Selection {
    /// Sorted, disjoint, non-adjacent, non-empty half-open ranges.
    ///
    /// The three adjectives are the invariant every mutation restores, and they are what
    /// make [`Selection::contains`] a binary search rather than a scan.
    runs: Vec<Range<u64>>,
    /// How many indices are selected. Maintained incrementally: recomputing it by summing
    /// the runs would make a sequence of *n* toggles quadratic.
    count: u64,
    /// What a range click is applied *over* -- see the module docs. Every gesture except
    /// [`Selection::range_to`] replaces it with the result it just produced.
    base: Vec<Range<u64>>,
    anchor: Option<u64>,
    lead: Option<u64>,
    /// Bumped whenever the set changes. Callers that cache something derived from the
    /// selection -- the shelf's size total is the first -- compare this instead of comparing
    /// the runs, which is what keeps a held selection off the per-frame path.
    generation: u64,
}

/// The selection every default [`Interaction`](crate::row::Interaction) borrows.
///
/// A `static` rather than a field so `Interaction` stays `Copy` and the renderer keeps
/// taking it by value.
pub static NOTHING: Selection = Selection::new();

impl Default for Selection {
    fn default() -> Self {
        Self::new()
    }
}

impl Selection {
    /// An empty selection.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            runs: Vec::new(),
            count: 0,
            base: Vec::new(),
            anchor: None,
            lead: None,
            generation: 0,
        }
    }

    /// How many entries are selected.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.count
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Whether `index` is selected.
    #[must_use]
    pub fn contains(&self, index: u64) -> bool {
        let i = self.runs.partition_point(|run| run.end <= index);
        self.runs.get(i).is_some_and(|run| run.start <= index)
    }

    /// The runs, in ascending order.
    #[must_use]
    pub fn runs(&self) -> &[Range<u64>] {
        &self.runs
    }

    /// The lowest selected index.
    #[must_use]
    pub fn first(&self) -> Option<u64> {
        self.runs.first().map(|run| run.start)
    }

    /// Where a range click extends *from*.
    #[must_use]
    pub const fn anchor(&self) -> Option<u64> {
        self.anchor
    }

    /// The most recently affected index -- where a range click extends *to*.
    #[must_use]
    pub const fn lead(&self) -> Option<u64> {
        self.lead
    }

    /// A counter that changes whenever the set does.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// The row the travelling selection region should animate to, if any.
    ///
    /// `None` for a multiple selection on purpose: the morph in
    /// [`InteractionMotion`](crate::motion::InteractionMotion) is one region moving between
    /// two rows, which is a single-selection idea. Several selected rows are drawn as
    /// several regions and no morph, which is why this is a question the selection answers
    /// rather than something the renderer infers.
    #[must_use]
    pub fn morph_target(&self) -> Option<u64> {
        if self.count == 1 { self.first() } else { None }
    }

    /// Selected indices within `range`, ascending.
    ///
    /// Iterates the runs, not the range: asking a million-entry select-all which of rows
    /// 40..80 are in it must cost forty steps, not a million.
    pub fn iter_in(&self, range: Range<u64>) -> impl Iterator<Item = u64> + '_ {
        let start = self.runs.partition_point(move |run| run.end <= range.start);
        self.runs
            .get(start..)
            .unwrap_or(&[])
            .iter()
            .take_while(move |run| run.start < range.end)
            .flat_map(move |run| run.start.max(range.start)..run.end.min(range.end))
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        if self.runs.is_empty() && self.anchor.is_none() && self.lead.is_none() {
            self.base.clear();
            return;
        }
        self.runs.clear();
        self.count = 0;
        self.anchor = None;
        self.lead = None;
        self.base.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    /// A plain click: `index` alone, and it becomes the anchor.
    pub fn select_only(&mut self, index: u64) {
        let already = self.count == 1 && self.contains(index);
        self.runs.clear();
        self.runs.push(index..index + 1);
        self.count = 1;
        self.anchor = Some(index);
        self.lead = Some(index);
        if !already {
            self.generation = self.generation.wrapping_add(1);
        }
        self.commit_base();
    }

    /// A `Ctrl`-click: flip `index`, and re-anchor there.
    ///
    /// Re-anchoring is what makes toggle-then-shift-click select the run the user just
    /// started rather than a run reaching back to some forgotten earlier click.
    pub fn toggle(&mut self, index: u64) {
        if self.contains(index) {
            self.remove(index..index + 1);
        } else {
            self.insert(index..index + 1);
        }
        self.anchor = Some(index);
        self.lead = Some(index);
        self.commit_base();
    }

    /// A `Shift`-click: the anchor-to-`index` run, over the base.
    ///
    /// The anchor is left where it was and the base is left alone, so a second shift-click
    /// re-ranges from the same place instead of growing from the first one -- and whatever
    /// was selected before the range gesture started survives it.
    pub fn range_to(&mut self, index: u64) {
        let anchor = self.anchor.unwrap_or(index);
        let (lo, hi) = (anchor.min(index), anchor.max(index));
        self.restore_base();
        self.insert(lo..hi + 1);
        self.anchor = Some(anchor);
        self.lead = Some(index);
    }

    /// A `Ctrl+Shift`-click: add the anchor-to-`index` run to what is already selected, and
    /// keep it -- unlike [`Selection::range_to`], the next range click ranges over this.
    pub fn extend_to(&mut self, index: u64) {
        let anchor = self.anchor.unwrap_or(index);
        let (lo, hi) = (anchor.min(index), anchor.max(index));
        self.insert(lo..hi + 1);
        self.lead = Some(index);
        self.commit_base();
    }

    /// `Ctrl+A`: every entry.
    ///
    /// One run whatever `count` is -- the reason the representation was chosen.
    pub fn select_all(&mut self, count: u64) {
        if count == 0 {
            self.clear();
            return;
        }
        self.set_span(0..count);
        self.anchor = Some(0);
        self.lead = Some(count - 1);
        self.commit_base();
    }

    /// `Ctrl+I`: everything not currently selected, within `count`.
    pub fn invert(&mut self, count: u64) {
        let mut inverted = Vec::with_capacity(self.runs.len() + 1);
        let mut cursor = 0u64;
        for run in &self.runs {
            if run.start > cursor {
                inverted.push(cursor..run.start.min(count));
            }
            cursor = run.end;
            if cursor >= count {
                break;
            }
        }
        if cursor < count {
            inverted.push(cursor..count);
        }
        inverted.retain(|run| run.start < run.end);

        self.count = inverted.iter().map(|run| run.end - run.start).sum();
        self.runs = inverted;
        self.generation = self.generation.wrapping_add(1);
        // The anchor and the lead survive only if they are still in the set: a range click
        // from a row that is no longer selected reads as a jump from nowhere.
        self.anchor = self
            .anchor
            .filter(|&i| self.contains(i))
            .or_else(|| self.first());
        self.lead = self
            .lead
            .filter(|&i| self.contains(i))
            .or_else(|| self.first());
        self.commit_base();
    }

    /// Add `range` to the selection.
    pub fn insert(&mut self, range: Range<u64>) {
        if range.start >= range.end {
            return;
        }
        // `end < start` rather than `<=`: a run ending exactly where this one begins is
        // *adjacent*, and leaving the two unmerged would break the non-adjacency invariant
        // and make `contains` correct but the run count unbounded.
        let lo = self.runs.partition_point(|run| run.end < range.start);
        let hi = self.runs.partition_point(|run| run.start <= range.end);

        if lo < hi {
            let merged = {
                let overlapped = self.runs.get(lo..hi).unwrap_or(&[]);
                let start = overlapped
                    .first()
                    .map_or(range.start, |run| run.start.min(range.start));
                let end = overlapped
                    .last()
                    .map_or(range.end, |run| run.end.max(range.end));
                self.count -= overlapped
                    .iter()
                    .map(|run| run.end - run.start)
                    .sum::<u64>();
                start..end
            };
            self.count += merged.end - merged.start;
            self.runs.splice(lo..hi, [merged]);
        } else {
            self.count += range.end - range.start;
            self.runs.insert(lo, range);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    /// Take `range` out of the selection.
    pub fn remove(&mut self, range: Range<u64>) {
        if range.start >= range.end {
            return;
        }
        let lo = self.runs.partition_point(|run| run.end <= range.start);
        let hi = self.runs.partition_point(|run| run.start < range.end);
        if lo >= hi {
            return;
        }

        let mut replacement = Vec::new();
        let overlapped = self.runs.get(lo..hi).unwrap_or(&[]);
        self.count -= overlapped
            .iter()
            .map(|run| run.end - run.start)
            .sum::<u64>();
        if let Some(head) = overlapped.first() {
            if head.start < range.start {
                replacement.push(head.start..range.start);
            }
        }
        if let Some(tail) = overlapped.last() {
            if tail.end > range.end {
                replacement.push(range.end..tail.end);
            }
        }
        self.count += replacement
            .iter()
            .map(|run| run.end - run.start)
            .sum::<u64>();
        self.runs.splice(lo..hi, replacement);
        self.generation = self.generation.wrapping_add(1);
    }

    /// Put the anchor at `index` without changing what is selected.
    ///
    /// For the callers that build a selection out of the primitives -- a marquee band, a
    /// restore from a previous visit -- and then have to say where a subsequent range click
    /// should extend from. Also commits the base, because a set assembled this way *is* the
    /// committed set.
    pub fn set_anchor(&mut self, index: u64) {
        self.anchor = Some(index);
        self.lead = Some(index);
        self.commit_base();
    }

    /// Drop anything at or past `count`.
    ///
    /// A listing that came back shorter must not leave indices selected that no entry
    /// answers to, because every consumer downstream -- the shelf's total, the accessibility
    /// count, and eventually the operations engine -- would then be counting a file that is
    /// not there.
    pub fn clamp_to(&mut self, count: u64) {
        self.remove(count..u64::MAX);
        self.anchor = self.anchor.filter(|&i| i < count);
        self.lead = self.lead.filter(|&i| i < count);
        self.commit_base();
    }

    /// Replace the whole set with one run, keeping the generation honest.
    fn set_span(&mut self, span: Range<u64>) {
        let unchanged = self.runs.len() == 1 && self.runs.first() == Some(&span);
        if !unchanged {
            self.runs.clear();
            self.runs.push(span.clone());
            self.count = span.end - span.start;
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Make the current set the one the next range click is applied over.
    fn commit_base(&mut self) {
        self.base.clear();
        self.base.extend_from_slice(&self.runs);
    }

    /// Go back to the committed set, discarding the range gesture in progress.
    fn restore_base(&mut self) {
        if self.runs == self.base {
            return;
        }
        self.runs.clear();
        self.runs.extend_from_slice(&self.base);
        self.count = self.runs.iter().map(|run| run.end - run.start).sum();
        self.generation = self.generation.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    fn indices(selection: &Selection, upto: u64) -> Vec<u64> {
        (0..upto).filter(|&i| selection.contains(i)).collect()
    }

    /// The invariant every mutation has to restore: sorted, disjoint, non-adjacent,
    /// non-empty, and a `count` that agrees with the runs.
    fn check(selection: &Selection) {
        let mut previous_end = None;
        for run in selection.runs() {
            assert!(run.start < run.end, "empty run {run:?}");
            if let Some(end) = previous_end {
                assert!(
                    run.start > end,
                    "run {run:?} touches or overlaps the one ending at {end}"
                );
            }
            previous_end = Some(run.end);
        }
        let summed: u64 = selection.runs().iter().map(|r| r.end - r.start).sum();
        assert_eq!(summed, selection.len(), "count drifted from the runs");
    }

    #[test]
    fn select_all_over_a_million_entries_is_one_run() {
        // The reason the representation is runs. A HashSet here is a million allocations
        // for one keystroke.
        let mut selection = Selection::new();
        selection.select_all(1_000_000);
        assert_eq!(selection.len(), 1_000_000);
        assert_eq!(selection.runs().len(), 1);
        assert!(selection.contains(0) && selection.contains(999_999));
        assert!(!selection.contains(1_000_000));
        check(&selection);
    }

    #[test]
    fn a_click_replaces_and_a_toggle_does_not() {
        let mut selection = Selection::new();
        selection.select_only(4);
        selection.select_only(9);
        assert_eq!(indices(&selection, 20), vec![9]);

        selection.toggle(4);
        assert_eq!(indices(&selection, 20), vec![4, 9]);
        selection.toggle(9);
        assert_eq!(indices(&selection, 20), vec![4]);
        check(&selection);
    }

    #[test]
    fn a_second_range_click_re_ranges_from_the_anchor_rather_than_growing() {
        // The failure this pins: shift-clicking 10 and then 6 must leave 4..=6 selected,
        // not 4..=10. Collapsing the anchor and the lead into one "current row" gets this
        // wrong and it is the most-noticed selection bug there is.
        let mut selection = Selection::new();
        selection.select_only(4);
        selection.range_to(10);
        assert_eq!(indices(&selection, 20), (4..=10).collect::<Vec<_>>());

        selection.range_to(6);
        assert_eq!(indices(&selection, 20), vec![4, 5, 6]);
        assert_eq!(selection.anchor(), Some(4));
        assert_eq!(selection.lead(), Some(6));
        check(&selection);
    }

    #[test]
    fn a_range_click_backwards_selects_the_same_span_as_forwards() {
        let mut back = Selection::new();
        back.select_only(10);
        back.range_to(4);
        assert_eq!(indices(&back, 20), (4..=10).collect::<Vec<_>>());
        assert_eq!(back.lead(), Some(4));
        check(&back);
    }

    #[test]
    fn a_toggle_re_anchors_so_the_next_range_click_starts_there() {
        let mut selection = Selection::new();
        selection.select_only(2);
        selection.toggle(8);
        selection.range_to(10);
        // The run comes from 8, the toggle's row -- not from 2.
        assert!(
            selection.contains(2),
            "the toggled-in run replaced the rest"
        );
        assert_eq!(indices(&selection, 20), vec![2, 8, 9, 10]);
        check(&selection);
    }

    #[test]
    fn a_range_click_after_a_toggle_keeps_the_toggled_rows_and_still_re_ranges() {
        // Both halves of the base rule in one gesture sequence, because getting either one
        // alone is what the two obvious implementations do.
        let mut selection = Selection::new();
        selection.select_only(2);
        selection.toggle(8);
        selection.range_to(12);
        assert_eq!(indices(&selection, 20), vec![2, 8, 9, 10, 11, 12]);

        selection.range_to(9);
        assert_eq!(
            indices(&selection, 20),
            vec![2, 8, 9],
            "the second range click grew from the first instead of replacing it"
        );
        check(&selection);
    }

    #[test]
    fn ctrl_shift_extends_without_discarding_what_is_already_selected() {
        let mut selection = Selection::new();
        selection.select_only(1);
        selection.toggle(5);
        selection.extend_to(7);
        assert_eq!(indices(&selection, 20), vec![1, 5, 6, 7]);
        check(&selection);
    }

    #[test]
    fn adjacent_runs_merge_instead_of_accumulating() {
        // Without merging, marqueeing down a list one row at a time leaves one run per row
        // and `contains` degrades to a scan over a structure that was chosen to avoid one.
        let mut selection = Selection::new();
        for i in 0..64 {
            selection.insert(i..i + 1);
        }
        assert_eq!(selection.runs().len(), 1);
        assert_eq!(selection.runs()[0], 0..64);
        assert_eq!(selection.len(), 64);
        check(&selection);
    }

    #[test]
    fn removing_from_the_middle_of_a_run_splits_it() {
        let mut selection = Selection::new();
        selection.insert(0..100);
        selection.remove(40..60);
        assert_eq!(selection.runs(), &[0..40, 60..100]);
        assert_eq!(selection.len(), 80);
        assert!(!selection.contains(40) && !selection.contains(59));
        assert!(selection.contains(39) && selection.contains(60));
        check(&selection);
    }

    #[test]
    fn removing_a_span_that_covers_several_runs_takes_all_of_them() {
        let mut selection = Selection::new();
        selection.insert(0..10);
        selection.insert(20..30);
        selection.insert(40..50);
        selection.remove(5..45);
        assert_eq!(selection.runs(), &[0..5, 45..50]);
        assert_eq!(selection.len(), 10);
        check(&selection);
    }

    #[test]
    fn inserting_a_span_that_bridges_two_runs_joins_them() {
        let mut selection = Selection::new();
        selection.insert(0..10);
        selection.insert(20..30);
        selection.insert(10..20);
        assert_eq!(selection.runs().len(), 1);
        assert_eq!(selection.runs().first(), Some(&(0..30)));
        assert_eq!(selection.len(), 30);
        check(&selection);
    }

    #[test]
    fn invert_is_its_own_undo() {
        let mut selection = Selection::new();
        selection.insert(3..7);
        selection.insert(20..25);
        let before = indices(&selection, 40);

        selection.invert(40);
        check(&selection);
        assert_eq!(selection.len(), 40 - 9);
        assert!(!selection.contains(3) && selection.contains(2));

        selection.invert(40);
        assert_eq!(indices(&selection, 40), before);
        check(&selection);
    }

    #[test]
    fn inverting_an_empty_selection_selects_everything_and_the_reverse() {
        let mut selection = Selection::new();
        selection.invert(1_000_000);
        assert_eq!(selection.len(), 1_000_000);
        assert_eq!(selection.runs().len(), 1);

        selection.invert(1_000_000);
        assert!(selection.is_empty());
        assert_eq!(selection.anchor(), None, "an anchor outlived its selection");
        check(&selection);
    }

    #[test]
    fn a_shorter_listing_cannot_leave_a_phantom_selected() {
        // A file that is no longer there must not be counted by the shelf, the a11y tree,
        // or -- once it exists -- an operation.
        let mut selection = Selection::new();
        selection.select_all(100);
        selection.clamp_to(10);
        assert_eq!(selection.len(), 10);
        assert!(!selection.contains(10));
        assert_eq!(
            selection.lead(),
            None,
            "the lead outlived the entry it named"
        );
        check(&selection);
    }

    #[test]
    fn iterating_a_visible_window_costs_the_window_and_not_the_corpus() {
        let mut selection = Selection::new();
        selection.select_all(1_000_000);
        let visible: Vec<u64> = selection.iter_in(40..80).collect();
        assert_eq!(visible, (40..80).collect::<Vec<_>>());

        selection.remove(50..60);
        let visible: Vec<u64> = selection.iter_in(40..80).collect();
        assert_eq!(visible.len(), 30);
        assert!(!visible.contains(&55));
    }

    #[test]
    fn iterating_a_window_past_every_run_yields_nothing() {
        let mut selection = Selection::new();
        selection.insert(0..10);
        assert_eq!(selection.iter_in(100..200).count(), 0);
        assert_eq!(selection.iter_in(0..0).count(), 0);
    }

    #[test]
    fn the_morph_target_is_the_single_selection_and_nothing_else() {
        // The motion system animates one region between two rows. Several selected rows are
        // several regions, and asking the morph to represent them would be a lie the
        // renderer then has to draw.
        let mut selection = Selection::new();
        assert_eq!(selection.morph_target(), None);
        selection.select_only(7);
        assert_eq!(selection.morph_target(), Some(7));
        selection.toggle(9);
        assert_eq!(selection.morph_target(), None);
    }

    #[test]
    fn the_generation_changes_only_when_the_set_does() {
        // What keeps the shelf's size total off the per-frame path: re-clicking the row
        // that is already the whole selection must not invalidate a cached summary.
        let mut selection = Selection::new();
        selection.select_only(3);
        let generation = selection.generation();
        selection.select_only(3);
        assert_eq!(selection.generation(), generation);
        selection.select_only(4);
        assert_ne!(selection.generation(), generation);
    }

    #[test]
    fn clearing_an_empty_selection_is_not_a_change() {
        let mut selection = Selection::new();
        let generation = selection.generation();
        selection.clear();
        assert_eq!(selection.generation(), generation);
    }

    #[test]
    fn the_shared_empty_selection_selects_nothing() {
        assert!(NOTHING.is_empty());
        assert!(!NOTHING.contains(0));
        assert_eq!(NOTHING.morph_target(), None);
    }
}
