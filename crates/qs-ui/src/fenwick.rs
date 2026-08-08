//! Row-height prefix sums, with a uniform-height fast path.
//!
//! Research R6 settles the structure: a Fenwick tree (binary indexed tree) giving O(log n)
//! offset→index and index→offset. For a million rows that is ~8 MB of `u32` prefix sums and
//! roughly 20 memory accesses per lookup -- negligible against an 8.33 ms budget.
//!
//! # The fast path is not an optimization, it is the common case
//!
//! Every corpus M0 gates on has uniform row heights, and so does every real directory
//! listing in the flat view. [`Heights::Uniform`] short-circuits the tree entirely: the
//! mapping becomes a division, the 8 MB is never allocated, and `flat-1m` never touches
//! this file's interesting code at all.
//!
//! The tree still has to exist and has to be correct, because M1's grouped and grid views
//! need variable heights, and building the spike on a uniform-height assumption would make
//! the M0 measurement describe a renderer nobody is going to ship (research R6's rejected
//! alternative).
//!
//! # Heights are integers
//!
//! Row heights are stored as `u32` in physical pixels, not `f32` in logical ones. Summing a
//! million `f32` heights accumulates error at exactly the scale that matters -- see
//! research R5 -- and an integer prefix sum is exact by construction. Fractional logical
//! heights are resolved to whole physical pixels once, when the density or scale changes.

/// Prefix sums over per-row heights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fenwick {
    /// 1-indexed internal tree; `tree[0]` is unused.
    tree: Vec<u64>,
    len: usize,
    total: u64,
    /// Shortest row, recorded at construction.
    ///
    /// This is what actually bounds virtualization when heights vary. FR-002 states the
    /// bound as `viewport_height / row_height + 2`, which is unambiguous only while every
    /// row is the same height. With variable heights the worst case is set by the
    /// *shortest* row -- that is how many can fit on screen at once -- and computing the
    /// bound from a nominal or average height would make the assertion pass while the
    /// recycler laid out several times more rows than intended.
    min_height: u32,
}

impl Fenwick {
    /// Build from per-row heights in physical pixels. O(n).
    pub fn from_heights(heights: &[u32]) -> Self {
        let len = heights.len();
        let mut tree = vec![0u64; len + 1];
        for (i, &h) in heights.iter().enumerate() {
            if let Some(slot) = tree.get_mut(i + 1) {
                *slot += u64::from(h);
            }
            // The linear build: each node absorbs its child's accumulated sum. O(n) rather
            // than the O(n log n) of n successive `add` calls, which matters when a density
            // change rebuilds a million-row tree while the user is watching.
            let parent = i + 1 + ((i + 1) & (i + 1).wrapping_neg());
            if parent <= len {
                let value = tree.get(i + 1).copied().unwrap_or(0);
                if let Some(slot) = tree.get_mut(parent) {
                    *slot += value;
                }
            }
        }
        let total = heights.iter().map(|&h| u64::from(h)).sum();
        let min_height = heights.iter().copied().min().unwrap_or(1).max(1);
        Self {
            tree,
            len,
            total,
            min_height,
        }
    }

    /// Shortest row in the tree. See the field docs for why this is the bound that matters.
    pub fn min_height(&self) -> u32 {
        self.min_height
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total content height in physical pixels.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Sum of heights of rows `0..index`, i.e. the top edge of row `index`.
    pub fn offset_of(&self, index: usize) -> u64 {
        let mut i = index.min(self.len);
        let mut sum = 0u64;
        while i > 0 {
            sum += self.tree.get(i).copied().unwrap_or(0);
            i -= i & i.wrapping_neg();
        }
        sum
    }

    /// Height of one row.
    pub fn height_of(&self, index: usize) -> u32 {
        if index >= self.len {
            return 0;
        }
        (self.offset_of(index + 1) - self.offset_of(index)) as u32
    }

    /// The row containing content-space `offset`, and the offset's position within it.
    ///
    /// An offset past the end returns the last row, not `None`: the scroll clamp is the
    /// caller's job, and returning an option here would push a `unwrap` into the frame path
    /// for a case that cannot occur once the clamp is applied.
    pub fn index_at(&self, offset: u64) -> (usize, u64) {
        if self.len == 0 {
            return (0, 0);
        }
        if offset >= self.total {
            let last = self.len - 1;
            return (last, offset - self.offset_of(last));
        }

        // Binary lifting over the tree: descend from the highest power of two, taking a
        // step whenever it does not overshoot. O(log n) with no division.
        let mut position = 0usize;
        let mut remaining = offset;
        let mut step = self.len.next_power_of_two();
        while step > 0 {
            let next = position + step;
            if next <= self.len {
                let value = self.tree.get(next).copied().unwrap_or(0);
                if value <= remaining {
                    position = next;
                    remaining -= value;
                }
            }
            step /= 2;
        }
        (position.min(self.len - 1), remaining)
    }
}

/// How a source reports its row heights.
///
/// The two variants are not merely an optimization detail -- they are what lets the
/// recycler skip the tree entirely for the common case. See the `RowSource` contract.
#[derive(Clone, Debug, PartialEq)]
pub enum Heights {
    /// Every row is this tall, in physical pixels.
    Uniform(u32),
    Variable(std::sync::Arc<Fenwick>),
}

impl Heights {
    pub fn total(&self, row_count: u64) -> u64 {
        match self {
            Self::Uniform(h) => u64::from(*h) * row_count,
            Self::Variable(tree) => tree.total(),
        }
    }

    pub fn offset_of(&self, index: u64) -> u64 {
        match self {
            Self::Uniform(h) => u64::from(*h) * index,
            Self::Variable(tree) => tree.offset_of(index as usize),
        }
    }

    pub fn height_of(&self, index: u64) -> u32 {
        match self {
            Self::Uniform(h) => *h,
            Self::Variable(tree) => tree.height_of(index as usize),
        }
    }

    /// The shortest row, which is what bounds how many can be visible at once.
    pub fn min_height(&self) -> u32 {
        match self {
            Self::Uniform(h) => (*h).max(1),
            Self::Variable(tree) => tree.min_height(),
        }
    }

    /// The row containing `offset`, and how far into it the offset falls.
    pub fn index_at(&self, offset: u64, row_count: u64) -> (u64, u64) {
        match self {
            Self::Uniform(h) => {
                let h = u64::from((*h).max(1));
                let index = (offset / h).min(row_count.saturating_sub(1));
                (index, offset - index * h)
            }
            Self::Variable(tree) => {
                let (index, within) = tree.index_at(offset);
                (index as u64, within)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    fn reference_offsets(heights: &[u32]) -> Vec<u64> {
        let mut out = vec![0u64];
        for &h in heights {
            let last = *out.last().unwrap_or(&0);
            out.push(last + u64::from(h));
        }
        out
    }

    #[test]
    fn an_empty_tree_is_usable() {
        let tree = Fenwick::from_heights(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.total(), 0);
        assert_eq!(tree.index_at(0), (0, 0));
        assert_eq!(tree.height_of(0), 0);
    }

    #[test]
    fn offsets_match_a_naive_prefix_sum() {
        let heights: Vec<u32> = (0..1000).map(|i| 20 + (i % 13) as u32).collect();
        let tree = Fenwick::from_heights(&heights);
        let reference = reference_offsets(&heights);

        for (i, &expected) in reference.iter().enumerate() {
            assert_eq!(tree.offset_of(i), expected, "offset_of({i})");
        }
        assert_eq!(tree.total(), *reference.last().unwrap());
    }

    #[test]
    fn heights_round_trip() {
        let heights: Vec<u32> = (0..500).map(|i| 1 + (i * 7 % 40) as u32).collect();
        let tree = Fenwick::from_heights(&heights);
        for (i, &expected) in heights.iter().enumerate() {
            assert_eq!(tree.height_of(i), expected, "height_of({i})");
        }
    }

    #[test]
    fn index_at_inverts_offset_of() {
        let heights: Vec<u32> = (0..777).map(|i| 18 + (i % 11) as u32).collect();
        let tree = Fenwick::from_heights(&heights);

        for (i, &height) in heights.iter().enumerate() {
            let top = tree.offset_of(i);
            assert_eq!(tree.index_at(top), (i, 0), "top edge of row {i}");

            // A point in the middle of the row must map back to the same row.
            let middle = top + u64::from(height) / 2;
            let (index, within) = tree.index_at(middle);
            assert_eq!(index, i);
            assert_eq!(within, middle - top);
        }
    }

    #[test]
    fn an_offset_past_the_end_clamps_to_the_last_row() {
        let tree = Fenwick::from_heights(&[10, 10, 10]);
        let (index, _) = tree.index_at(999_999);
        assert_eq!(index, 2, "past the end must clamp, not wrap or panic");
    }

    #[test]
    fn a_million_rows_stays_exact() {
        // The reason heights are integers. A million rows at 28px is 28,000,000 content
        // pixels -- well past f32's exact-integer range of 2^24 -- and the bottom of the
        // corpus is precisely where an f32 accumulation would drift.
        let heights = vec![28u32; 1_000_000];
        let tree = Fenwick::from_heights(&heights);
        assert_eq!(tree.total(), 28_000_000);
        assert_eq!(tree.offset_of(999_999), 27_999_972);
        assert_eq!(tree.index_at(27_999_972), (999_999, 0));
        assert_eq!(tree.index_at(27_999_999), (999_999, 27));
    }

    #[test]
    fn the_uniform_fast_path_agrees_with_the_tree() {
        // The fast path exists because it is provably the same answer. If it ever is not,
        // `flat-1m` and `deep-40` stop being comparable measurements.
        let row_count = 5000u64;
        let height = 28u32;
        let uniform = Heights::Uniform(height);
        let variable = Heights::Variable(std::sync::Arc::new(Fenwick::from_heights(&vec![
            height;
            row_count as usize
        ])));

        assert_eq!(uniform.total(row_count), variable.total(row_count));
        for index in [0u64, 1, 42, 2499, row_count - 1] {
            assert_eq!(uniform.offset_of(index), variable.offset_of(index));
            assert_eq!(uniform.height_of(index), variable.height_of(index));
        }
        for offset in [0u64, 27, 28, 29, 1000, 139_999] {
            assert_eq!(
                uniform.index_at(offset, row_count),
                variable.index_at(offset, row_count),
                "offset {offset}"
            );
        }
    }

    #[test]
    fn a_zero_height_row_does_not_divide_by_zero() {
        let uniform = Heights::Uniform(0);
        assert_eq!(uniform.index_at(100, 10), (9, 100 - 9));
    }

    #[test]
    fn power_of_two_and_off_by_one_lengths_are_all_correct() {
        // Fenwick indexing is where off-by-one bugs live, and they only appear at specific
        // lengths -- exactly at, one below, and one above a power of two.
        for len in [
            1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            let heights: Vec<u32> = (0..len).map(|i| (i as u32 % 5) + 1).collect();
            let tree = Fenwick::from_heights(&heights);
            let reference = reference_offsets(&heights);
            for (i, &expected) in reference.iter().enumerate() {
                assert_eq!(tree.offset_of(i), expected, "len {len}, offset_of({i})");
            }
            for (i, &offset) in reference.iter().take(len).enumerate() {
                assert_eq!(tree.index_at(offset), (i, 0), "len {len}, row {i}");
            }
        }
    }
}
