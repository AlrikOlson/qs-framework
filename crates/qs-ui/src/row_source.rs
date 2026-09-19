//! The row-data interface consumed by layout and accessibility.
//!
//! Sources fill a reusable [`RowBuf`] with borrowed name bytes and metadata.
//! The renderer can use in-memory rows, directory listings or search results
//! without depending on how they are stored.

use std::ops::Range;

use crate::density::Density;
use crate::fenwick::Heights;

/// Dense ordinal within a source. Stable for the source's lifetime.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct RowId(pub u64);

/// Interned semantic type; drives icon selection.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct KindId(pub u16);

/// How much of a row has been loaded.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum LoadState {
    /// Only `name` and `IS_DIR` are meaningful. The renderer MUST NOT read `size`, `mtime`
    /// or `kind` -- they are not merely stale, they are undefined.
    #[default]
    Stub = 0,
    /// Name, size, mtime, kind.
    Basic = 1,
    /// All row fields are available.
    Full = 2,
}

/// Boolean row attributes.
///
/// Hand-written rather than pulled from `bitflags`: four constants and three operators do
/// not justify a dependency that every one of the eight cross-compilation targets has to
/// build and `cargo-deny` has to license-check.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RowFlags(pub u32);

impl RowFlags {
    pub const EMPTY: Self = Self(0);
    pub const IS_DIR: Self = Self(1 << 0);
    pub const IS_HIDDEN: Self = Self(1 << 1);
    pub const IS_SYMLINK: Self = Self(1 << 2);
    pub const IS_SELECTED: Self = Self(1 << 3);
    /// The pointer is over this row.
    pub const IS_HOVERED: Self = Self(1 << 4);
    /// Keyboard focus. Distinct from selection on purpose: a row can be focused without
    /// being selected while the keyboard moves through the list, and collapsing the two
    /// makes keyboard navigation invisible.
    pub const IS_FOCUSED: Self = Self(1 << 5);
    /// **Known** read-only. Absent means "not known to be read-only", never "writable".
    ///
    /// `Attrs::readonly` is an `Option` precisely so that "nobody looked" stays distinct
    /// from "it is false", and a bit cannot carry three states. So the bit is set only for
    /// `Some(true)`, and every consumer reads its absence as *no claim* — which for the
    /// material encoding means the ordinary surface, the same one an unfilled stub gets.
    /// Collapsing `None` into `false` here would reintroduce the lie `Attrs` exists to
    /// prevent, one layer up.
    pub const IS_READONLY: Self = Self(1 << 6);

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    #[inline]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for RowFlags {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

/// One row supplied to the renderer.
///
/// `name` is a byte range into the buffer's arena. Raw bytes allow filenames
/// that are not valid UTF-8.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RowView {
    pub id: RowId,
    /// Range into the name arena in [`RowBuf`].
    pub name: Range<u32>,
    pub size: u64,
    /// Modification time in Unix seconds, as expected by [`crate::format_mtime`].
    pub mtime: i64,
    pub kind: KindId,
    pub flags: RowFlags,
    pub state: LoadState,
    /// Nesting depth, for indentation. Zero in a flat listing.
    pub depth: u16,
}

impl Default for RowView {
    fn default() -> Self {
        Self {
            id: RowId(0),
            name: 0..0,
            size: 0,
            mtime: 0,
            kind: KindId(0),
            flags: RowFlags::EMPTY,
            state: LoadState::Stub,
            depth: 0,
        }
    }
}

/// The buffer a source fills.
///
/// Reused across frames by the recycler, so a steady-state scroll allocates nothing. RC-3
/// forbids retaining one *across* frames, which is what lets a source reclaim arena space
/// on a version bump -- the buffer is cleared and refilled, never held.
#[derive(Clone, Debug, Default)]
pub struct RowBuf {
    rows: Vec<RowView>,
    names: Vec<u8>,
}

impl RowBuf {
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty the buffer, keeping both allocations.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.names.clear();
    }

    /// Append a row whose name is `name`.
    pub fn push(&mut self, mut row: RowView, name: &[u8]) {
        let start = self.names.len() as u32;
        self.names.extend_from_slice(name);
        row.name = start..self.names.len() as u32;
        self.rows.push(row);
    }

    pub fn rows(&self) -> &[RowView] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Raw name bytes for a row.
    pub fn name_bytes(&self, row: &RowView) -> &[u8] {
        self.names
            .get(row.name.start as usize..row.name.end as usize)
            .unwrap_or(&[])
    }

    /// Name as UTF-8, lossily.
    ///
    /// Lossy rather than fallible because a filename that is not valid UTF-8 must still
    /// render -- refusing to draw a row because its name is malformed would make the file
    /// invisible, which is strictly worse than drawing it with a replacement character.
    pub fn name(&self, row: &RowView) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(self.name_bytes(row))
    }
}

/// Source of rows for layout and accessibility.
///
/// `rows()` must return immediately and fill the entire requested range.
/// Use [`LoadState::Stub`] for rows whose data has not loaded.
pub trait RowSource: Send + Sync {
    /// Exact row count, used for scroll extent and accessibility `set_size`.
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `out` with rows for `range`. MUST NOT block.
    fn rows(&self, range: Range<u64>, out: &mut RowBuf);

    /// Row heights at `density`, in physical pixels.
    fn heights(&self, density: Density, scale: f32) -> Heights;

    /// Monotonic version that increases when content changes, not on each frame.
    fn version(&self) -> u64;
}

/// A source with no rows.
///
/// Exists so the application can open a window before any source is attached, and so tests
/// of the recycler do not need a corpus.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptySource;

impl RowSource for EmptySource {
    fn len(&self) -> u64 {
        0
    }
    fn rows(&self, _range: Range<u64>, out: &mut RowBuf) {
        out.clear();
    }
    fn heights(&self, density: Density, scale: f32) -> Heights {
        Heights::Uniform(density.row_height_px(scale, 1.0))
    }
    fn version(&self) -> u64 {
        0
    }
}

/// A source that returns [`LoadState::Stub`] for everything.
///
/// Named in the contract's test obligations: it proves placeholder rendering works and that
/// the golden image of the placeholder state is a real state the renderer can reach, not a
/// hypothetical one.
#[derive(Debug, Clone)]
pub struct StubbornSource {
    pub count: u64,
}

impl RowSource for StubbornSource {
    fn len(&self) -> u64 {
        self.count
    }

    fn rows(&self, range: Range<u64>, out: &mut RowBuf) {
        out.clear();
        for id in range.start..range.end.min(self.count) {
            out.push(
                RowView {
                    id: RowId(id),
                    state: LoadState::Stub,
                    ..Default::default()
                },
                b"",
            );
        }
    }

    fn heights(&self, density: Density, scale: f32) -> Heights {
        Heights::Uniform(density.row_height_px(scale, 1.0))
    }

    fn version(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn flags_compose_and_test() {
        let f = RowFlags::IS_DIR | RowFlags::IS_SELECTED;
        assert!(f.contains(RowFlags::IS_DIR));
        assert!(f.contains(RowFlags::IS_SELECTED));
        assert!(!f.contains(RowFlags::IS_HIDDEN));
        assert!(f.without(RowFlags::IS_DIR).contains(RowFlags::IS_SELECTED));
        assert!(RowFlags::EMPTY.is_empty());
    }

    #[test]
    fn names_are_stored_in_the_arena_not_per_row() {
        let mut buf = RowBuf::new();
        buf.push(RowView::default(), b"alpha.txt");
        buf.push(RowView::default(), b"beta.rs");

        assert_eq!(buf.len(), 2);
        assert_eq!(buf.name(&buf.rows()[0]), "alpha.txt");
        assert_eq!(buf.name(&buf.rows()[1]), "beta.rs");
        // One contiguous buffer, not two allocations.
        assert_eq!(buf.names.len(), 16);
    }

    #[test]
    fn clearing_keeps_capacity_so_a_scroll_does_not_allocate() {
        let mut buf = RowBuf::new();
        for i in 0..100 {
            buf.push(RowView::default(), format!("file-{i}.txt").as_bytes());
        }
        let (rows, names) = (buf.rows.capacity(), buf.names.capacity());
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.rows.capacity(), rows);
        assert_eq!(buf.names.capacity(), names);
    }

    #[test]
    fn an_invalid_utf8_name_still_renders() {
        // M1 must draw files whose names are not valid UTF-8. Refusing would make the file
        // invisible, which is worse than a replacement character.
        let mut buf = RowBuf::new();
        buf.push(RowView::default(), &[0x66, 0x6f, 0xff, 0x6f]);
        let name = buf.name(&buf.rows()[0]);
        assert!(name.contains('\u{FFFD}'));
        assert_eq!(buf.name_bytes(&buf.rows()[0]).len(), 4);
    }

    #[test]
    fn the_stubborn_source_fills_the_whole_range() {
        // RS-2: a short return would force the recycler to handle gaps.
        let source = StubbornSource { count: 1000 };
        let mut buf = RowBuf::new();
        source.rows(100..140, &mut buf);
        assert_eq!(buf.len(), 40);
        assert!(buf.rows().iter().all(|r| r.state == LoadState::Stub));
        assert_eq!(buf.rows()[0].id, RowId(100));
    }

    #[test]
    fn a_range_past_the_end_is_truncated_not_padded_with_garbage() {
        let source = StubbornSource { count: 10 };
        let mut buf = RowBuf::new();
        source.rows(5..50, &mut buf);
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn the_empty_source_is_usable_as_a_starting_state() {
        let source = EmptySource;
        let mut buf = RowBuf::new();
        source.rows(0..10, &mut buf);
        assert!(source.is_empty());
        assert!(buf.is_empty());
    }
}
