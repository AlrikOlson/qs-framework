//! Draw lists, the UI→Render handoff, and the thread-affinity guards.
//!
//! # The handoff is the architecture
//!
//! Constitution II forbids `Arc<Mutex<AppState>>` and requires frames to render from
//! immutable snapshots. This module is where that stops being a principle and becomes a
//! data structure: the UI thread builds a [`DrawList`] and publishes it; the render thread
//! takes the most recent one. Neither ever waits for the other.
//!
//! A triple buffer is the right shape for exactly this. Two buffers would force the
//! producer to wait whenever the consumer holds one. Three means the producer always owns
//! a slot it can write, the consumer always owns a slot it can read, and the third is
//! wherever the handoff currently sits. The cost is one extra draw list of memory; the
//! benefit is that a slow frame on the GPU can never stall input handling, which is the
//! entire point.
//!
//! A queue was the alternative and is worse here: if the renderer falls behind, a queue
//! grows and the user sees stale frames drain out after they stop scrolling. A triple
//! buffer drops intermediate frames instead, which is what you want -- nobody wants to
//! watch the scroll they already finished.
//!
//! The ordering claims in [`DrawListChannel`] are checked by `tests/loom_handoff.rs` under
//! `loom`, not asserted in a comment and hoped for.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

#[cfg(loom)]
use loom::sync::atomic::{AtomicU8, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(loom)]
use loom::cell::UnsafeCell;
#[cfg(not(loom))]
use std::cell::UnsafeCell;

use bytemuck::{Pod, Zeroable};

use crate::color::Srgba;

// -- primitives --------------------------------------------------------------------

/// Which shader branch draws this instance.
///
/// One pipeline handles every primitive, discriminated here rather than by switching
/// pipelines. A pipeline switch costs a state change per batch; a branch on a value that
/// is uniform across a batch's worth of instances costs the GPU essentially nothing,
/// because every invocation in a warp takes the same side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PrimKind {
    /// Pass 0: filled rounded rectangle, signed-distance antialiased.
    Rect = 0,
    /// Pass 1: rounded-rectangle stroke. `param` is the stroke width in pixels.
    Stroke = 1,
    /// Pass 3: a glyph, sampling R8 coverage from the atlas.
    Glyph = 2,
}

impl PrimKind {
    /// Every primitive the pipeline can draw.
    ///
    /// Declared, not derived -- Rust has no stable way to enumerate an enum's variants.
    /// Two independent things keep the list from going stale, and neither is a comment
    /// asking people to remember:
    ///
    /// - [`PrimKind::index`] below is an *exhaustive* match, so adding a variant is a
    ///   compile error three lines from this array.
    /// - `the_shader_and_prim_kind_declare_the_same_primitives` in `tier_parity`
    ///   parses the `KIND_` constants out of `shaders/instance.wgsl` and asserts that set
    ///   equals this one. A primitive the shader can draw but this array omits fails the
    ///   test suite, which is the case the compile error above cannot see.
    ///
    /// It exists so the `tier_parity` suite can assert that *every* kind has a parity
    /// fixture, rather than asserting it about whichever kinds someone remembered.
    pub const ALL: [PrimKind; 3] = [Self::Rect, Self::Stroke, Self::Glyph];

    /// This variant's position in [`PrimKind::ALL`].
    ///
    /// The match is exhaustive on purpose. It is the compile-time half of the guard
    /// described on `ALL`, and its only job is to fail to build when a variant appears.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Rect => 0,
            Self::Stroke => 1,
            Self::Glyph => 2,
        }
    }

    /// The name this primitive answers to in `shaders/instance.wgsl`.
    ///
    /// Also an exhaustive match, and the bridge that lets a test compare the Rust enum
    /// against the WGSL constants by name rather than by discriminant alone -- a swapped
    /// pair of discriminants would otherwise compare equal as a set.
    #[must_use]
    pub const fn shader_const(self) -> &'static str {
        match self {
            Self::Rect => "KIND_RECT",
            Self::Stroke => "KIND_STROKE",
            Self::Glyph => "KIND_GLYPH",
        }
    }
}

/// One primitive. Exactly 48 bytes, asserted below.
///
/// The size is not arbitrary. At the fastest fling a frame carries roughly 45 rows of
/// perhaps 40 glyphs plus chrome -- call it 2,000 instances, or 96 KB. That fits in the
/// per-frame staging allocation with room to spare and streams to the GPU in one upload.
/// Growing the struct to 64 bytes would cost a third more bandwidth for fields that would
/// be zero on almost every instance.
///
/// Clipping is deliberately **not** a field. It lives on [`Batch`] as a scissor rect,
/// because clip regions are uniform across long runs of instances (a row's name column
/// clips identically for every glyph in it) and a per-instance copy would be 16 wasted
/// bytes repeated thousands of times.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, Default, Pod, Zeroable)]
pub struct Instance {
    /// `[x, y, width, height]` in physical pixels, already rebased against the scroll
    /// offset on the CPU (research R5 -- the rebasing is what makes `f32` safe here).
    pub rect: [f32; 4],
    /// `[u0, v0, u1, v1]` in normalized atlas coordinates. Zero for untextured primitives.
    pub uv: [f32; 4],
    /// Premultiplied linear RGBA8 -- see [`crate::color`].
    pub color: u32,
    /// Corner radius in physical pixels.
    pub radius: f32,
    /// Kind-specific scalar: stroke width for [`PrimKind::Stroke`], unused otherwise.
    pub param: f32,
    /// A [`PrimKind`].
    pub kind: u32,
}

const _: () = assert!(
    size_of::<Instance>() == 48,
    "the instance stride is a bandwidth budget, not an accident -- if this changes, \
     re-derive the per-frame upload size before accepting it"
);

impl Instance {
    pub fn rect(x: f32, y: f32, w: f32, h: f32, radius: f32, color: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [0.0; 4],
            color: color.to_premul_linear_rgba8(),
            radius,
            param: 0.0,
            kind: PrimKind::Rect as u32,
        }
    }

    pub fn stroke(x: f32, y: f32, w: f32, h: f32, radius: f32, width: f32, color: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [0.0; 4],
            color: color.to_premul_linear_rgba8(),
            radius,
            param: width,
            kind: PrimKind::Stroke as u32,
        }
    }

    pub fn glyph(x: f32, y: f32, w: f32, h: f32, uv: [f32; 4], color: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv,
            color: color.to_premul_linear_rgba8(),
            radius: 0.0,
            param: 0.0,
            kind: PrimKind::Glyph as u32,
        }
    }
}

/// A run of instances sharing a scissor rect and a texture binding.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Batch {
    pub range: Range<u32>,
    /// `[x, y, w, h]` in physical pixels. `None` means the whole surface.
    pub scissor: Option<[u32; 4]>,
    /// Whether this batch samples the glyph atlas.
    pub textured: bool,
}

// -- draw list ---------------------------------------------------------------------

/// Per-frame counters, mirrored into `FrameSample` by the harness.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DrawStats {
    pub rows_laid_out: u32,
    pub glyphs_rasterized: u32,
    /// Glyphs wanted this frame that the atlas could not supply, almost always because the
    /// per-frame upload bound deferred them. A non-zero value is text that is missing from
    /// the screen right now (SC-006), so it is a counter that has to be *shown* somewhere;
    /// the diagnostics overlay is where.
    pub glyphs_dropped: u32,
    /// Icons wanted this frame that the atlas could not supply.
    ///
    /// Separate from [`DrawStats::glyphs_dropped`] on purpose. The two share a meaning --
    /// something that should be on screen is not -- but not a cause: a dropped glyph sends
    /// you to the font stack, a dropped icon to [`crate::icon`] or to a size that fell
    /// outside the legible range. One number would send every investigation to the wrong
    /// place half the time.
    pub icons_dropped: u32,
    pub shaped_runs_new: u32,
    pub instances: u32,
}

/// Everything the render thread needs for one frame, and nothing it has to look up.
///
/// This owns its data rather than borrowing from UI state. That is the point: once
/// published, the render thread can take arbitrarily long without the UI thread having to
/// care what it is still reading.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct DrawList {
    pub instances: Vec<Instance>,
    pub batches: Vec<Batch>,
    /// Physical pixels.
    pub viewport: [u32; 2],
    /// Background clear colour.
    pub clear: Srgba,
    /// Monotonic; the render thread uses it to tell a re-presented frame from a new one.
    pub generation: u64,
    pub stats: DrawStats,
    /// Time from the input event that caused this frame to the moment the list was
    /// committed. This is the quantity SC-002 gates (FR-023) and it is measured here, at
    /// the handoff, because that is the last instant that is unambiguously ours -- past it
    /// the numbers belong to the driver and the compositor.
    ///
    /// `None` for a frame not caused by input (a resize, a first paint).
    pub input_to_commit: Option<Duration>,
}

impl DrawList {
    /// Drop the contents but keep the allocations.
    ///
    /// Called on the slot the producer is about to write. Steady-state frame building must
    /// not allocate, and the only way to guarantee that is to reuse the same `Vec`s frame
    /// after frame rather than building fresh ones and trusting the allocator.
    pub fn reset(&mut self, viewport: [u32; 2], clear: Srgba, generation: u64) {
        self.instances.clear();
        self.batches.clear();
        self.viewport = viewport;
        self.clear = clear;
        self.generation = generation;
        self.stats = DrawStats::default();
        self.input_to_commit = None;
    }

    /// Push a batch covering every instance added since the last batch ended.
    pub fn end_batch(&mut self, scissor: Option<[u32; 4]>, textured: bool) {
        let start = self.batches.last().map_or(0, |b| b.range.end);
        let end = self.instances.len() as u32;
        if end > start {
            self.batches.push(Batch {
                range: start..end,
                scissor,
                textured,
            });
        }
    }

    /// Instances past the end of the last batch.
    ///
    /// These are uploaded to the instance buffer and covered by no draw call, so they cost
    /// bandwidth and render nothing. Both consumers ([`crate::batcher`] and the CPU
    /// rasterizer) iterate `batches`, never `instances`, which makes the failure completely
    /// silent -- it looks exactly like code that did not run. Anything appending to a list
    /// after someone else has closed the final batch should assert this is zero.
    pub fn unbatched(&self) -> u32 {
        let covered = self.batches.last().map_or(0, |b| b.range.end);
        (self.instances.len() as u32).saturating_sub(covered)
    }

    pub fn finish(&mut self) {
        self.stats.instances = self.instances.len() as u32;
    }
}

// -- the triple buffer -------------------------------------------------------------

/// Bit 0..=1 of the state byte: which slot is ready for the consumer.
const IDX_MASK: u8 = 0b11;
/// Bit 2: the ready slot has not been consumed yet.
const FRESH: u8 = 0b100;

/// Lock-free single-producer / single-consumer handoff of three draw lists.
///
/// # Safety argument
///
/// At any instant the three slot indices are partitioned into exactly three roles --
/// producer-owned, consumer-owned, and "ready" -- and the partition is maintained by a
/// single atomic swap on each side. The producer only ever dereferences the index it holds
/// in `Producer::write`; the consumer only ever dereferences `Consumer::read`. A swap
/// hands an index across and takes a different one back in the same operation, so no index
/// is ever held by both sides. `Producer` and `Consumer` are separate, non-`Clone` types,
/// which is how single-producer/single-consumer is enforced rather than documented.
///
/// The `AcqRel` ordering on both swaps is what makes the *contents* of a slot visible: the
/// producer's writes happen-before its swap, which synchronizes-with the consumer's swap,
/// which happens-before its reads.
pub struct DrawListChannel {
    slots: [UnsafeCell<DrawList>; 3],
    state: AtomicU8,
}

// SAFETY: see the type-level safety argument. The `UnsafeCell`s are only ever accessed
// through `Producer`/`Consumer`, which hold disjoint indices by construction.
unsafe impl Send for DrawListChannel {}
unsafe impl Sync for DrawListChannel {}

impl std::fmt::Debug for DrawListChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawListChannel").finish_non_exhaustive()
    }
}

/// Build a connected producer/consumer pair.
pub fn draw_list_channel() -> (Producer, Consumer) {
    let channel = Arc::new(DrawListChannel {
        slots: [
            UnsafeCell::new(DrawList::default()),
            UnsafeCell::new(DrawList::default()),
            UnsafeCell::new(DrawList::default()),
        ],
        // Producer starts on slot 0, consumer on slot 1, slot 2 is ready-but-stale.
        state: AtomicU8::new(2),
    });
    (
        Producer {
            channel: Arc::clone(&channel),
            write: 0,
        },
        Consumer { channel, read: 1 },
    )
}

/// The UI-thread half.
#[derive(Debug)]
pub struct Producer {
    channel: Arc<DrawListChannel>,
    write: u8,
}

impl Producer {
    /// The slot to build this frame into. Retains last frame's allocations.
    pub fn slot(&mut self) -> &mut DrawList {
        let index = self.write as usize;
        // SAFETY: `self.write` is owned exclusively by this `Producer` until `publish`
        // swaps it away, and `Producer` is neither `Clone` nor `Sync`.
        unsafe { self.channel.slot_mut(index) }
    }

    /// Publish the current slot and take ownership of another.
    ///
    /// Never blocks and never fails. If the consumer has not picked up the previous frame,
    /// this overwrites it -- deliberately. See the module docs on why dropping an
    /// intermediate frame beats queueing it.
    pub fn publish(&mut self) {
        let old = self
            .channel
            .state
            .swap(self.write | FRESH, Ordering::AcqRel);
        self.write = old & IDX_MASK;
    }
}

/// The render-thread half.
#[derive(Debug)]
pub struct Consumer {
    channel: Arc<DrawListChannel>,
    read: u8,
}

impl Consumer {
    /// Take the newest published draw list, if there is one that has not been taken.
    ///
    /// `None` means nothing new since the last call -- the correct response is to
    /// re-present the previous frame, or to do nothing at all, which is what keeps an idle
    /// window at 0 Hz (SC-003).
    pub fn acquire(&mut self) -> Option<&DrawList> {
        if self.channel.state.load(Ordering::Acquire) & FRESH == 0 {
            return None;
        }
        let old = self.channel.state.swap(self.read, Ordering::AcqRel);
        self.read = old & IDX_MASK;
        let index = self.read as usize;
        // SAFETY: the swap transferred exclusive ownership of `old`'s index to this
        // consumer; the producer can no longer reach it.
        Some(unsafe { self.channel.slot_ref(index) })
    }

    /// The most recently acquired list, whether or not it is new.
    pub fn current(&self) -> &DrawList {
        // SAFETY: `self.read` is consumer-owned.
        unsafe { self.channel.slot_ref(self.read as usize) }
    }
}

impl DrawListChannel {
    /// # Safety
    /// The caller must own `index` per the type-level argument.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slot_mut(&self, index: usize) -> &mut DrawList {
        let cell = self.slot_cell(index);
        #[cfg(loom)]
        {
            cell.with_mut(|p| unsafe { &mut *p })
        }
        #[cfg(not(loom))]
        {
            unsafe { &mut *cell.get() }
        }
    }

    /// # Safety
    /// The caller must own `index` per the type-level argument.
    unsafe fn slot_ref(&self, index: usize) -> &DrawList {
        let cell = self.slot_cell(index);
        #[cfg(loom)]
        {
            cell.with(|p| unsafe { &*p })
        }
        #[cfg(not(loom))]
        {
            unsafe { &*cell.get() }
        }
    }

    fn slot_cell(&self, index: usize) -> &UnsafeCell<DrawList> {
        // The index always comes from a 2-bit field, so it is 0..=3; state is only ever
        // seeded with 0..=2. Clamping rather than indexing keeps a corrupted state byte
        // from becoming an out-of-bounds access.
        match index {
            0 => &self.slots[0],
            1 => &self.slots[1],
            _ => &self.slots[2],
        }
    }
}

// -- thread affinity ---------------------------------------------------------------

/// Debug-build guards asserting which thread is running.
///
/// FR-016 and Constitution I say no blocking work on the UI or Render thread. That is
/// unenforceable as a code review rule at any real size -- the violation is usually three
/// call levels below the function anyone reviewed. These guards make the *thread* checkable
/// at the boundary, so a helper that grew a file read shows up as a failing debug assertion
/// rather than as a p99 someone notices two milestones later.
///
/// Compiled out entirely in release: the frame path pays nothing.
pub mod affinity {
    use std::sync::OnceLock;
    use std::thread::ThreadId;

    static UI_THREAD: OnceLock<ThreadId> = OnceLock::new();
    static RENDER_THREAD: OnceLock<ThreadId> = OnceLock::new();

    /// Called once, from the thread that will build draw lists.
    pub fn register_ui_thread() {
        let _ = UI_THREAD.set(std::thread::current().id());
    }

    /// Called once, from the thread that will submit to the GPU.
    pub fn register_render_thread() {
        let _ = RENDER_THREAD.set(std::thread::current().id());
    }

    pub fn is_ui_thread() -> bool {
        UI_THREAD
            .get()
            .is_some_and(|&id| id == std::thread::current().id())
    }

    pub fn is_render_thread() -> bool {
        RENDER_THREAD
            .get()
            .is_some_and(|&id| id == std::thread::current().id())
    }

    /// True when the current thread must not block: it is the UI or the Render thread.
    pub fn is_frame_thread() -> bool {
        is_ui_thread() || is_render_thread()
    }

    /// Assert the caller is on the UI thread. No-op in release.
    #[inline]
    pub fn assert_ui_thread(what: &str) {
        debug_assert!(
            UI_THREAD.get().is_none() || is_ui_thread(),
            "{what} must run on the UI thread"
        );
    }

    /// Assert the caller is on the render thread. No-op in release.
    #[inline]
    pub fn assert_render_thread(what: &str) {
        debug_assert!(
            RENDER_THREAD.get().is_none() || is_render_thread(),
            "{what} must run on the render thread"
        );
    }

    /// Assert that a blocking operation is *not* happening on a frame thread.
    ///
    /// Call this from anything that does file I/O, takes a contended lock, or waits. It is
    /// the one guard that catches the class of bug FR-016 is actually about.
    #[inline]
    pub fn assert_may_block(what: &str) {
        debug_assert!(
            !is_frame_thread(),
            "{what} blocks and must not run on the UI or Render thread (Constitution I)"
        );
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn instances_appended_after_the_last_batch_are_reported_as_uncovered() {
        // The silent-render bug in one assertion: appending to a list whose final batch is
        // already closed produces instances that no draw call ever reaches.
        let mut list = DrawList::default();
        list.instances
            .push(Instance::rect(0.0, 0.0, 4.0, 4.0, 0.0, Srgba::default()));
        list.end_batch(None, false);
        assert_eq!(list.unbatched(), 0);

        list.instances
            .push(Instance::rect(4.0, 0.0, 4.0, 4.0, 0.0, Srgba::default()));
        assert_eq!(list.unbatched(), 1, "the appended instance is in no batch");

        list.end_batch(None, true);
        assert_eq!(list.unbatched(), 0);
    }

    #[test]
    fn the_instance_stride_is_48_bytes() {
        assert_eq!(size_of::<Instance>(), 48);
        assert_eq!(align_of::<Instance>(), 4);
    }

    #[test]
    fn nothing_is_available_before_the_first_publish() {
        let (_p, mut c) = draw_list_channel();
        assert!(c.acquire().is_none());
    }

    #[test]
    fn a_published_list_is_acquired_exactly_once() {
        let (mut p, mut c) = draw_list_channel();
        p.slot().reset([100, 100], Srgba::TRANSPARENT, 7);
        p.publish();

        assert_eq!(c.acquire().map(|d| d.generation), Some(7));
        assert!(
            c.acquire().is_none(),
            "acquiring twice must not hand back the same frame as though it were new"
        );
    }

    #[test]
    fn the_consumer_sees_only_the_newest_of_several_frames() {
        // This is the behaviour that keeps a slow renderer from replaying a stale fling.
        let (mut p, mut c) = draw_list_channel();
        for generation in 1..=5 {
            p.slot().reset([100, 100], Srgba::TRANSPARENT, generation);
            p.publish();
        }
        assert_eq!(c.acquire().map(|d| d.generation), Some(5));
        assert!(c.acquire().is_none());
    }

    #[test]
    fn the_producer_never_writes_the_slot_the_consumer_holds() {
        let (mut p, mut c) = draw_list_channel();
        p.slot().reset([1, 1], Srgba::TRANSPARENT, 1);
        p.publish();
        let held = c.acquire().unwrap().generation;

        // Publish twice more while the consumer conceptually holds its slot.
        for generation in 2..=3 {
            p.slot().reset([1, 1], Srgba::TRANSPARENT, generation);
            p.publish();
        }
        assert_eq!(held, 1);
        assert_eq!(
            c.current().generation,
            1,
            "the held frame must not be mutated"
        );
    }

    #[test]
    fn reset_keeps_capacity_so_steady_state_does_not_allocate() {
        let mut list = DrawList::default();
        for _ in 0..1000 {
            list.instances
                .push(Instance::rect(0.0, 0.0, 1.0, 1.0, 0.0, Srgba::TRANSPARENT));
        }
        let capacity = list.instances.capacity();
        list.reset([0, 0], Srgba::TRANSPARENT, 0);
        assert!(list.instances.is_empty());
        assert_eq!(list.instances.capacity(), capacity);
    }

    #[test]
    fn batches_cover_every_instance_exactly_once() {
        let mut list = DrawList::default();
        let push = |n: usize, list: &mut DrawList| {
            for _ in 0..n {
                list.instances
                    .push(Instance::rect(0.0, 0.0, 1.0, 1.0, 0.0, Srgba::TRANSPARENT));
            }
        };
        push(3, &mut list);
        list.end_batch(None, false);
        push(4, &mut list);
        list.end_batch(Some([0, 0, 10, 10]), true);
        // An empty batch must not be recorded -- a zero-length draw call is pure overhead.
        list.end_batch(None, false);

        assert_eq!(list.batches.len(), 2);
        assert_eq!(list.batches[0].range, 0..3);
        assert_eq!(list.batches[1].range, 3..7);
        let covered: u32 = list
            .batches
            .iter()
            .map(|b| b.range.end - b.range.start)
            .sum();
        assert_eq!(covered as usize, list.instances.len());
    }

    #[test]
    fn prim_kind_all_is_self_consistent() {
        // `ALL` is a hand-written list, so something has to check it against the
        // exhaustive match that is the compile-time guard. Every entry must sit at its
        // own index, and the discriminant must match the position: a variant added to
        // `index` but forgotten in `ALL` shifts one of these and fails here.
        for (i, kind) in PrimKind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i, "{kind:?} is not at index {i} of ALL");
            assert_eq!(
                *kind as u32 as usize, i,
                "{kind:?}'s discriminant is not {i}"
            );
        }
        // Names are distinct, so `shader_const` cannot collapse two variants onto one
        // WGSL constant and make the cross-check in tier_parity vacuously pass.
        let mut names: Vec<&str> = PrimKind::ALL.iter().map(|k| k.shader_const()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PrimKind::ALL.len());
    }

    #[test]
    fn affinity_guards_pass_when_no_thread_is_registered() {
        // Tests and the bench harness run without a registered UI thread; the guards must
        // not fire there, or every test would need a setup ritual.
        affinity::assert_ui_thread("test");
        affinity::assert_may_block("test");
    }
}
