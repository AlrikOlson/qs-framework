//! Kind-based vector icons, rasterized into the glyph atlas.
//!
//! UXDD 10.4: a 20 px grid, a 1.5 px stroke, "drawn as vectors and rasterized at exact
//! device resolution -- never scaled bitmaps". Both halves of that sentence decide the
//! design here.
//!
//! # Why icons live in the *glyph* atlas
//!
//! An icon is a glyph that isn't from a font. It is an 8-bit coverage mask that gets tinted
//! at draw time and sampled through the same UV rectangle, so it wants exactly what
//! [`crate::atlas::GlyphAtlas`] already provides: shelf packing, CLOCK eviction, the rule
//! that an entry used this frame is never evicted, and the per-frame upload bound. Giving
//! icons their own texture would duplicate all four and add a second bind group to the
//! batcher; instead the atlas *key* widened to [`crate::atlas::AtlasKey`] and nothing else
//! changed. The payoff is that icons ride the existing `Glyph` primitive, which is what
//! makes them render identically on all three tiers without a single tier-specific line.
//!
//! # Why the key is device pixels, not (size, scale)
//!
//! A 20 px icon at 150 % and a 30 px icon at 100 % are the *same rasterization*. Keying on
//! the pair would store both. Keying on the rounded device pixel size collapses them, which
//! is also the only key that can honestly claim "rasterized at exact device resolution":
//! there is one entry per distinct pixel grid, by construction.
//!
//! # Why no blank memoization
//!
//! [`GlyphAtlas::get_or_insert`](crate::atlas::GlyphAtlas::get_or_insert) memoizes glyphs
//! that render with no ink, because a space is text that is *supposed* to be invisible.
//! An icon that renders with no ink is a bug in this file. [`rasterize`] therefore returns
//! `None` rather than an empty bitmap, so the failure reaches `glyphs_dropped` instead of
//! being absorbed as "correctly blank".

use qs_text::RasterizedGlyph;
use tiny_skia::{FillRule, LineCap, LineJoin, Mask, Path, PathBuilder, Stroke, Transform};

/// The design grid every path in this file is drawn on. UXDD 10.4.
pub const GRID: f32 = 20.0;

/// Stroke weight, in grid units. UXDD 10.4.
pub const STROKE: f32 = 1.5;

/// Icons below this many device pixels are refused: a 1.5-unit stroke on a grid this small
/// is thinner than a pixel and every icon rasterizes to the same grey smudge, which is
/// worse than the placeholder square it replaces.
pub const MIN_PX: u16 = 8;

/// Icons above this are refused. 20 grid units at 300 % is 60; the cap is the guard against
/// a caller that computed its size wrongly, not a limit any real display reaches.
pub const MAX_PX: u16 = 96;

/// The fraction of the icon box one emblem occupies.
///
/// Arrived at by rendering the set at 20 px and looking, which is the only way this number can
/// be chosen. At 0.5 the emblem is not a modifier -- it eats the left bracket off `Code`, the
/// bottom two rules off `Text` and a third of the `Config` hex, so the row loses the kind it
/// was reading and gains only "this is a link". The criterion is that a symlink stays
/// distinguishable *from its target's kind*, and 0.5 fails it for four of the nine.
pub const EMBLEM_FRACTION: f32 = 0.42;

/// Emblem stroke weight, in grid units. Heavier than [`STROKE`] because an emblem is drawn at
/// half the icon's pixel size: 1.5 units there is 0.4 device pixels at a 20 px icon, which is
/// the grey smudge [`MIN_PX`] exists to refuse.
pub const EMBLEM_STROKE: f32 = 3.0;

/// State-icon stroke weight, in grid units. Between [`STROKE`] and [`EMBLEM_STROKE`], for the
/// same reason both of those exist: a state icon is asked for at the size chrome has room for
/// -- around 12 device pixels in a tab -- where 1.5 units is 0.9 of a pixel. See
/// [`state_path`].
pub const STATE_STROKE: f32 = 2.1;

/// The icon set. One variant per visually distinct silhouette, not one per file extension:
/// mapping many extensions onto one icon is the *point* of an icon set, and is done by the
/// caller (see `qs_ui::row::icon_for`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub enum IconKind {
    /// A file whose type is unknown, or known but unremarkable.
    #[default]
    Generic,
    Folder,
    /// Source code.
    Code,
    /// Configuration and manifests.
    Config,
    /// Prose documents.
    Document,
    Image,
    /// Plain text and logs.
    Text,
    /// Structured data.
    Data,
    Archive,
}

impl IconKind {
    /// Every kind, in declaration order. The index into this slice is
    /// [`IconKind::index`], which callers use to size a per-frame cache.
    pub const ALL: [Self; 9] = [
        Self::Generic,
        Self::Folder,
        Self::Code,
        Self::Config,
        Self::Document,
        Self::Image,
        Self::Text,
        Self::Data,
        Self::Archive,
    ];

    /// What to call this kind in a sentence a human reads.
    ///
    /// Here rather than at the one call site that needs words, because the words and the set
    /// have to change together: a tenth kind added to [`IconKind::ALL`] without a label is a
    /// compile error, and a tenth kind whose label lives in a table somewhere else is a blank
    /// bar in the multi-selection histogram that nobody notices.
    ///
    /// Plural, because every use of it so far counts things.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Generic => "Other",
            Self::Folder => "Folders",
            Self::Code => "Code",
            Self::Config => "Config",
            Self::Document => "Documents",
            Self::Image => "Images",
            Self::Text => "Text",
            Self::Data => "Data",
            Self::Archive => "Archives",
        }
    }

    /// Position in [`IconKind::ALL`]. Stable, and the reason a caller can hold a fixed-size
    /// array instead of a map.
    pub fn index(self) -> usize {
        match self {
            Self::Generic => 0,
            Self::Folder => 1,
            Self::Code => 2,
            Self::Config => 3,
            Self::Document => 4,
            Self::Image => 5,
            Self::Text => 6,
            Self::Data => 7,
            Self::Archive => 8,
        }
    }
}

/// A mark drawn *over* a kind icon rather than instead of one.
///
/// # Why an emblem is its own shape and not a field on [`IconKey`]
///
/// The tempting key is `{ kind, emblem: Option<Emblem>, px }`, and it is the same mistake as
/// giving icons their own texture: it makes an entry's identity the *pair*, so the atlas
/// stores one rasterization per (kind, emblem) and the count is `K * (E + 1)` per size --
/// 36 here against 13 for keying them separately. The cold frame is the one that cannot
/// afford it (see `qs_ui::row::ListRenderer::render`, which resolves icons in a prepass
/// precisely because the CPU tier's per-frame upload bound is contested there).
///
/// Keying separately also keeps the two tints independent, which is not a nicety: an emblem
/// is a different colour from the icon it modifies, and one coverage mask carries one colour.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Emblem {
    /// The flat well an emblem's mark sits in, drawn first in the surface colour.
    ///
    /// Without it the mark is a stroke laid over another stroke: the silhouette's lines show
    /// through the gaps and both read as noise. One of the module's two *filled* shapes; the
    /// other is [`StateIcon::Running`], and they are filled for opposite reasons — this one to
    /// knock a hole in what is under it, that one because it is the quietest mark in its set.
    Plate,
    /// The link arrow. Drawn over [`Emblem::Plate`].
    Symlink,
}

impl Emblem {
    /// Every emblem, in declaration order. The index into this slice is [`Emblem::index`],
    /// which callers use to size a per-frame cache -- the same contract as [`IconKind::ALL`].
    pub const ALL: [Self; 2] = [Self::Plate, Self::Symlink];

    /// Position in [`Emblem::ALL`].
    pub fn index(self) -> usize {
        match self {
            Self::Plate => 0,
            Self::Symlink => 1,
        }
    }
}

/// The state of something that is running, as a shape rather than as a codepoint.
///
/// # Why these are vectors, and why that is a fix rather than a feature
///
/// `qs::terminal::Status` carried its state in a `mark: &'static str` — U+25CF, U+25CB,
/// U+25B2 — and its own doc records what that cost: the first build used U+26A0 for the
/// failure states, and U+26A0 came out as a notdef box on this machine's fallback stack. The
/// response at the time was to retreat to Geometric Shapes (U+25xx), "the one block a shot has
/// confirmed this renderer resolves", which is an honest promise and a small one. It is still
/// a promise about somebody else's font.
///
/// A shape in this module is drawn from a `PathBuilder` and rasterized by `tiny-skia`. There is
/// no font in the path at all, so there is no machine on which it fails to resolve. That makes
/// the *drawn* state channel unconditional, which is what FR-021 wanted and what a codepoint
/// could not give it. The words stay: a reader who is not looking at the screen needs
/// [`crate::icon`] to be silent and `Status::words` to carry the state, which is why the icon
/// replaces the mark on the surface and never in the accessible name.
///
/// # Why the harness states are here and not in the harness crate
///
/// `qs-term` is a leaf that knows nothing about drawing, and `qs-gpu` knows nothing about
/// terminals. The shapes are the renderer's; the *mapping* from a session's state to one of
/// them is `qs`'s, at the one place that already turns a `Lifecycle` into ink
/// (`qs::terminal::status`). A state enum in this module that named PTY conditions would be
/// the renderer holding an opinion about processes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum StateIcon {
    /// Alive, and nothing better is known. The floor's live state.
    Running,
    /// Stopped, having succeeded.
    Finished,
    /// Stopped, having failed — or lost, which is the same news to a reader.
    Failed,
    /// An identified harness that is doing something.
    Working,
    /// An identified harness that is blocked on a person. The state ADR 014 calls the one a
    /// supervisor most needs to see, which is why it is the loudest silhouette in the set.
    AwaitingApproval,
    /// An adapter that was disbelieved. The session fell back to the floor, and this says so
    /// rather than showing the last thing the adapter claimed.
    Degraded,
    /// A record of a session a previous run had. **Nothing is running.**
    ///
    /// The one state in this set that is not about a process at all, which is why its shape is
    /// the *trace* of one: [`StateIcon::Running`]'s circle with most of it missing. Every other
    /// candidate silhouette said something the state does not mean — a clock says "recently",
    /// an archive box says "put away deliberately", a play triangle says "press me" — and this
    /// one says the only true thing, which is that a session was here and is not now.
    Remembered,
}

impl StateIcon {
    /// Every state icon, in declaration order. The index into this slice is
    /// [`StateIcon::index`] — the same per-frame-cache contract as [`IconKind::ALL`].
    pub const ALL: [Self; 7] = [
        Self::Running,
        Self::Finished,
        Self::Failed,
        Self::Working,
        Self::AwaitingApproval,
        Self::Degraded,
        Self::Remembered,
    ];

    /// Position in [`StateIcon::ALL`].
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Running => 0,
            Self::Finished => 1,
            Self::Failed => 2,
            Self::Working => 3,
            Self::AwaitingApproval => 4,
            Self::Degraded => 5,
            Self::Remembered => 6,
        }
    }
}

/// What one rasterization is a picture of: a file kind, a mark drawn over one, or the state of
/// something that is running.
///
/// This is the discriminator that keeps the atlas additive. See [`Emblem`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum IconShape {
    Kind(IconKind),
    Emblem(Emblem),
    State(StateIcon),
}

/// Identity of one rasterized icon. This is the atlas key.
///
/// `px` is the *device* pixel size of the square the shape occupies -- scale is already
/// folded in. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IconKey {
    pub shape: IconShape,
    pub px: u16,
}

impl IconKey {
    /// Build a key for a kind at a logical grid size and a device scale.
    ///
    /// Rounding here rather than at the draw site is deliberate: the rounded value is both
    /// the cache key and the drawn size, so they cannot drift apart and leave the icon
    /// resampled by a fraction of a pixel.
    pub fn new(kind: IconKind, logical_px: f32, scale: f32) -> Self {
        Self {
            shape: IconShape::Kind(kind),
            px: device_px(logical_px, scale),
        }
    }

    /// Build a key for a state icon at a logical box size and a device scale.
    ///
    /// Rounding happens here for the reason it does in [`IconKey::new`]: the rounded value is
    /// both the cache key and the drawn size.
    #[must_use]
    pub fn state(state: StateIcon, logical_px: f32, scale: f32) -> Self {
        Self {
            shape: IconShape::State(state),
            px: device_px(logical_px, scale),
        }
    }

    /// Build a key for an emblem. `logical_px` is the emblem's own box, not the icon's --
    /// the caller applies [`EMBLEM_FRACTION`], because the caller is also the one that has to
    /// decide whether the result is large enough to ask for at all. See [`emblem_px`].
    pub fn emblem(emblem: Emblem, logical_px: f32, scale: f32) -> Self {
        Self {
            shape: IconShape::Emblem(emblem),
            px: device_px(logical_px, scale),
        }
    }
}

/// The emblem size for an icon of `icon_px` device pixels, or `None` when that is below
/// [`MIN_PX`].
///
/// `None` means **do not draw an emblem**, and the caller must honour that by not asking:
/// [`rasterize`] would refuse the same size, and a refusal is counted as a dropped icon.
/// Conflating "too small to be worth drawing" with "the atlas could not supply it" would put
/// a design decision into a telemetry counter that exists to detect an upload-budget failure.
pub fn emblem_px(icon_px: u16) -> Option<u16> {
    let px = (f32::from(icon_px) * EMBLEM_FRACTION).round();
    let px = if px.is_finite() {
        px.clamp(0.0, f32::from(u16::MAX)) as u16
    } else {
        0
    };
    (MIN_PX..=MAX_PX).contains(&px).then_some(px)
}

/// The device pixel size a logical icon box resolves to. Saturating rather than wrapping,
/// and zero for a non-finite scale, so a bad layout produces a refused icon rather than a
/// cast that wraps into a plausible-looking size.
pub fn device_px(logical_px: f32, scale: f32) -> u16 {
    let px = (logical_px * scale).round();
    if px.is_finite() {
        px.clamp(0.0, f32::from(u16::MAX)) as u16
    } else {
        0
    }
}

/// Rasterize one icon into an R8 coverage bitmap.
///
/// `None` means the size is out of range or the path could not be stroked -- both are
/// failures the caller should count, not absorb. A successful result is always
/// `px * px` bytes with at least one non-zero sample.
pub fn rasterize(key: IconKey) -> Option<RasterizedGlyph> {
    if key.px < MIN_PX || key.px > MAX_PX {
        return None;
    }
    let px = u32::from(key.px);
    let scale = f32::from(key.px) / GRID;

    let (path, ink) = path_for(key.shape)?;
    // Transform *then* stroke, not the reverse. Stroking on the 20-unit grid and scaling the
    // resulting outline would flatten the curves at grid resolution and then magnify the
    // flattening error; this way every curve is flattened against the pixel grid it will
    // actually be sampled on, which is what "rasterized at exact device resolution" means.
    let path = path.transform(Transform::from_scale(scale, scale))?;
    let outline = match ink {
        Ink::Stroke(width) => {
            let stroke = Stroke {
                width: width * scale,
                line_cap: LineCap::Round,
                line_join: LineJoin::Round,
                ..Stroke::default()
            };
            path.stroke(&stroke, 1.0)?
        }
        // A filled shape is already its own outline. Stroking it would produce a ring, and a
        // ring knocks nothing out -- which is the entire job of `Emblem::Plate`.
        Ink::Fill => path,
    };

    let mut mask = Mask::new(px, px)?;
    // Winding, not even-odd: a stroker emits an outer contour and an inner contour with
    // opposite orientation, and two strokes that cross (the code icon's slash over its
    // brackets) must stay filled where they overlap rather than punching a hole.
    mask.fill_path(&outline, FillRule::Winding, true, Transform::identity());

    let coverage = mask.take();
    if !coverage.iter().any(|&c| c > 0) {
        // See the module docs: a blank icon is a bug, so refuse rather than memoize.
        return None;
    }

    Some(RasterizedGlyph {
        width: px,
        height: px,
        // Icons have no pen and no baseline. They are placed by their top-left corner, so
        // both offsets are zero by definition rather than by omission.
        left: 0,
        top: 0,
        coverage,
        was_color: false,
    })
}

/// How a path becomes coverage. Carried next to the geometry rather than inferred from the
/// variant, because the two are chosen together: the plate's inset assumes a fill and would
/// be wrong by half a stroke if anything decided to stroke it instead.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Ink {
    /// Stroke at this width, in grid units.
    Stroke(f32),
    Fill,
}

/// The path for one shape, on the 20-unit grid, and how to ink it.
///
/// Every stroked path stays inside `[1.25, 18.75]` on both axes so that half of the 1.5-unit
/// stroke cannot cross the mask edge and get clipped. Silhouettes are chosen to differ at the
/// *shape* level rather than by a badge on a shared page outline, because a badge is three
/// pixels wide at 20 px and every icon would read as "a document".
///
/// An emblem is the one case where a badge *is* the right answer, and it does not contradict
/// that rule: it is drawn on its own grid at its own size, and it modifies an icon the reader
/// has already identified rather than being what distinguishes one icon from another.
fn path_for(shape: IconShape) -> Option<(Path, Ink)> {
    match shape {
        IconShape::Kind(kind) => kind_path(kind).map(|p| (p, Ink::Stroke(STROKE))),
        IconShape::Emblem(emblem) => emblem_path(emblem),
        IconShape::State(state) => state_path(state),
    }
}

/// The state marks, each on its own 20-unit grid.
///
/// # These are drawn heavier than a kind icon, and the reason is where they appear
///
/// A kind icon sits in a 20 px column of its own. A state icon sits inside a tab, beside a
/// breadcrumb, or on a row already carrying a kind icon and a name — so it is asked for at
/// around 12 px, where a 1.5-unit stroke is 0.9 device pixels and every one of these reads as
/// the same grey speck. [`STATE_STROKE`] is what makes the set survive the sizes it is
/// actually drawn at, and it is the same argument [`EMBLEM_STROKE`] makes one level down.
///
/// The silhouettes are chosen to differ from each other *first*: this is a set a person scans
/// across a tab strip to find the one session that needs them, so a bell must not be a
/// question mark at a glance. They are checked pairwise against the kind icons too, because a
/// directory row draws one of each.
fn state_path(state: StateIcon) -> Option<(Path, Ink)> {
    let mut b = PathBuilder::new();
    match state {
        StateIcon::Running => {
            // A filled disc: the one state that is not an event but a condition. Filled rather
            // than stroked because it is the quietest thing in the set and a ring at 12 px
            // competes with `Working`, which is the one distinction that has to survive.
            circle(&mut b, 10.0, 10.0, 4.5);
            return b.finish().map(|p| (p, Ink::Fill));
        }
        StateIcon::Finished => {
            // A tick. Nothing else in the set is a single open polyline, which is what makes
            // it readable at 12 px where the closed shapes start to fill in.
            b.move_to(4.0, 10.5);
            b.line_to(8.5, 15.0);
            b.line_to(16.0, 5.5);
        }
        StateIcon::Failed => {
            // The warning triangle -- U+26A0, the codepoint that came back as a notdef box and
            // sent the whole channel to Geometric Shapes. Drawn rather than looked up, it is
            // available on every machine, which is the point of this module.
            //
            // A bar and no dot, and a render decided that. The exclamation mark's point sat
            // between the bar and the base with 2.1 units of stroke on each of the three, and
            // at 20 px the three merged into one vertical smear -- the triangle read as filled
            // rather than as carrying a mark. There is no arrangement that fits all three:
            // the base's inner edge is at 15.35 and a point needs a radius plus a gap on both
            // sides of it. The bar alone is unambiguous at every size in `SIZES`, and it is
            // what an icon set does at 16 px anyway.
            b.move_to(10.0, 3.0);
            b.line_to(17.2, 16.4);
            b.line_to(2.8, 16.4);
            b.close();
            b.move_to(10.0, 6.6);
            b.line_to(10.0, 12.4);
        }
        StateIcon::Working => {
            // Three quarters of a ring. The gap is what tells it from `Running` at any size --
            // a closed ring and a filled disc are the same silhouette once antialiasing has
            // had its way with 12 px, and an open one is not.
            let (cx, cy, r) = (10.0, 10.0, 6.2);
            let o = r * KAPPA;
            b.move_to(cx, cy - r);
            b.cubic_to(cx + o, cy - r, cx + r, cy - o, cx + r, cy);
            b.cubic_to(cx + r, cy + o, cx + o, cy + r, cx, cy + r);
            b.cubic_to(cx - o, cy + r, cx - r, cy + o, cx - r, cy);
        }
        StateIcon::AwaitingApproval => {
            // A bell: dome, rim, clapper. Chosen over a question mark and over a raised hand
            // because this is the one state in the set that is a *summons* rather than a
            // report, and a bell is the only shape in common use that says so without words.
            // It is also the widest silhouette here, which is what makes it findable when a
            // strip of tabs is scanned rather than read.
            let (cx, base, r) = (10.0, 14.0, 5.4);
            let o = r * KAPPA;
            b.move_to(cx - r, base);
            b.line_to(cx - r, 10.0);
            b.cubic_to(cx - r, 10.0 - o, cx - o, 4.6, cx, 4.6);
            b.cubic_to(cx + o, 4.6, cx + r, 10.0 - o, cx + r, 10.0);
            b.line_to(cx + r, base);
            // The rim runs wider than the dome, so the bell has a foot rather than ending in
            // two loose verticals.
            b.move_to(3.2, base);
            b.line_to(16.8, base);
            b.move_to(8.6, 16.6);
            b.line_to(11.4, 16.6);
        }
        StateIcon::Degraded => {
            // A question mark: what is running here is no longer something anything can claim
            // to know. Deliberately *not* a crossed-out anything -- a prohibition sign reads as
            // "this session was stopped", and the session is fine; it is the adapter that was
            // disbelieved.
            b.move_to(6.4, 7.6);
            b.cubic_to(6.4, 4.4, 13.6, 4.4, 13.6, 7.6);
            b.cubic_to(13.6, 10.4, 10.0, 10.4, 10.0, 13.2);
            b.move_to(10.0, 16.2);
            b.line_to(10.0, 16.2);
        }
        StateIcon::Remembered => {
            // `Working`'s circle, broken into four dashes: the trace of a session rather than a
            // session. Three things about the geometry are load-bearing and none is decoration.
            //
            // The radius is `Working`'s, not a smaller one, because the pair a reader has to
            // separate fastest here is *alive* against *not alive* — a smaller ring reads as a
            // quieter version of the same thing, and a broken one at the same size reads as the
            // same thing with pieces missing, which is the claim.
            //
            // **Three dashes and 50-degree gaps, and a render decided both numbers.** The first
            // build was four 60-degree dashes with 30-degree gaps, which reads correctly at 16
            // and 20 px and closes up completely at 12. The arithmetic says why, and it is the
            // round line cap rather than the gap: at 12 px this circle is 3.7 device px in
            // radius, so a 30-degree gap is 1.95 px of arc, and `STATE_STROKE`'s round caps
            // extend half a stroke — 0.63 px — into it from each side. What is left is 0.7 px of
            // background, which antialiasing fills in. The shape became `Working` at exactly the
            // size a tab draws it, and the pairwise test passed anyway because a nearly-closed
            // ring is still numerically far from a three-quarter one.
            //
            // A gap has to be about 3.3 px of arc for 2 px of it to survive the caps, which is
            // 50 degrees. Three dashes rather than four because at 40 degrees each the set reads
            // as four ticks that happen to lie on a circle rather than as a circle with pieces
            // out of it, and "the same ring, interrupted" is the entire claim.
            //
            // One dash sits in the upper left, where `Working`'s own gap is. That puts the two
            // shapes' difference where a scanning reader's eye already is rather than spreading
            // it evenly around a ring, and it is worth about a third of their pairwise distance.
            let (cx, cy, r) = (10.0, 10.0, 6.2);
            for start in [270.0f32, 30.0, 150.0] {
                arc(&mut b, cx, cy, r, start, 70.0);
            }
        }
    }
    b.finish().map(|p| (p, Ink::Stroke(STATE_STROKE)))
}

/// One arc of a circle, beginning at `start` degrees clockwise from twelve o'clock and sweeping
/// `sweep` degrees. Callers keep `sweep` at or under 90.
///
/// A single cubic, which approximates an arc of up to a quarter turn to well inside the
/// sub-pixel error the rasterizer can express at [`MAX_PX`]. The control offset is the general
/// form of [`KAPPA`] — `(4/3)·tan(θ/4)` — rather than a second magic number: at 90 degrees it
/// *is* `KAPPA`, which is what keeps the dashes on the same circle every other round shape in
/// this module is drawn on.
fn arc(b: &mut PathBuilder, cx: f32, cy: f32, r: f32, start: f32, sweep: f32) {
    let k = r * (4.0 / 3.0) * (sweep.to_radians() / 4.0).tan();
    // Screen coordinates: y grows downward, and the angle is measured from twelve o'clock, so
    // the point at angle a is (sin a, -cos a) and its forward tangent is (cos a, sin a).
    let point = |a: f32| {
        let a = a.to_radians();
        (cx + r * a.sin(), cy - r * a.cos())
    };
    let tangent = |a: f32| {
        let a = a.to_radians();
        (a.cos(), a.sin())
    };
    let (x0, y0) = point(start);
    let (x1, y1) = point(start + sweep);
    let (tx0, ty0) = tangent(start);
    let (tx1, ty1) = tangent(start + sweep);
    b.move_to(x0, y0);
    b.cubic_to(
        x0 + k * tx0,
        y0 + k * ty0,
        x1 - k * tx1,
        y1 - k * ty1,
        x1,
        y1,
    );
}

/// The plate and the marks that sit on it, each on its own 20-unit grid.
fn emblem_path(emblem: Emblem) -> Option<(Path, Ink)> {
    let mut b = PathBuilder::new();
    match emblem {
        Emblem::Plate => {
            // A disc, not a rounded square. Both knock the silhouette out; the disc removes
            // about a fifth less of it for the same mark, which at 20 px is the difference
            // between the `Code` brackets surviving and not. Half a unit of inset so the
            // antialiased edge has somewhere to land instead of being clipped square.
            circle(&mut b, 10.0, 10.0, 9.5);
            b.finish().map(|p| (p, Ink::Fill))
        }
        Emblem::Symlink => {
            // An arrow leaving to the upper right: shaft on the diagonal, two barbs at the
            // head. Two barbs rather than a filled triangle because a triangle at eight device
            // pixels is a blob, and the open head keeps the diagonal readable. Held inside a
            // radius of about 7 so the round cap on every end stays within the disc.
            b.move_to(6.0, 14.0);
            b.line_to(13.5, 6.5);
            b.move_to(13.5, 6.5);
            b.line_to(13.5, 11.0);
            b.move_to(13.5, 6.5);
            b.line_to(9.0, 6.5);
            b.finish().map(|p| (p, Ink::Stroke(EMBLEM_STROKE)))
        }
    }
}

fn kind_path(kind: IconKind) -> Option<Path> {
    let mut b = PathBuilder::new();
    match kind {
        IconKind::Generic => page(&mut b),
        IconKind::Folder => {
            // Body with a raised back tab, one closed contour.
            b.move_to(2.5, 16.5);
            b.line_to(2.5, 4.5);
            b.line_to(8.0, 4.5);
            b.line_to(9.8, 7.0);
            b.line_to(17.5, 7.0);
            b.line_to(17.5, 16.5);
            b.close();
        }
        IconKind::Code => {
            // Angle brackets around a slash: `</>`. No page outline -- the brackets alone
            // are the most legible "this is source" mark at 20 px.
            b.move_to(7.0, 5.5);
            b.line_to(2.5, 10.0);
            b.line_to(7.0, 14.5);
            b.move_to(13.0, 5.5);
            b.line_to(17.5, 10.0);
            b.line_to(13.0, 14.5);
            b.move_to(11.5, 4.5);
            b.line_to(8.5, 15.5);
        }
        IconKind::Config => {
            // A hex nut: a pointy-top hexagon with a bore. Round-ish and closed, so it is
            // unmistakable against every line-based and page-based icon in the set.
            b.move_to(10.0, 2.5);
            b.line_to(16.0, 6.25);
            b.line_to(16.0, 13.75);
            b.line_to(10.0, 17.5);
            b.line_to(4.0, 13.75);
            b.line_to(4.0, 6.25);
            b.close();
            circle(&mut b, 10.0, 10.0, 2.6);
        }
        IconKind::Document => {
            page(&mut b);
            for (y, x1) in [(9.5, 13.5), (12.5, 13.5), (15.5, 11.0)] {
                b.move_to(6.5, y);
                b.line_to(x1, y);
            }
        }
        IconKind::Image => {
            // Frame, sun, ridgeline. The interior marks stop a full unit short of the frame
            // so the two never merge into a filled block at small sizes.
            rounded_rect(&mut b, 2.5, 4.0, 15.0, 12.0, 2.0);
            circle(&mut b, 7.0, 8.0, 1.4);
            b.move_to(3.8, 14.0);
            b.line_to(8.0, 9.5);
            b.line_to(11.0, 12.5);
            b.line_to(13.5, 10.0);
            b.line_to(16.2, 14.0);
        }
        IconKind::Text => {
            // Ruled lines with no container. The absence of an outline is the signal.
            for (y, x1) in [(5.0, 17.0), (8.5, 17.0), (12.0, 17.0), (15.5, 12.0)] {
                b.move_to(3.0, y);
                b.line_to(x1, y);
            }
        }
        IconKind::Data => {
            // A database cylinder: full ellipse on top, two sides, one lower arc.
            ellipse(&mut b, 10.0, 5.6, 6.5, 2.6);
            b.move_to(3.5, 5.6);
            b.line_to(3.5, 14.4);
            b.move_to(16.5, 5.6);
            b.line_to(16.5, 14.4);
            lower_arc(&mut b, 10.0, 14.4, 6.5, 2.6);
            lower_arc(&mut b, 10.0, 10.0, 6.5, 2.6);
        }
        IconKind::Archive => {
            // A shipping box: lid band, body, clasp.
            b.move_to(2.0, 8.0);
            b.line_to(2.0, 4.5);
            b.line_to(18.0, 4.5);
            b.line_to(18.0, 8.0);
            b.close();
            b.move_to(3.5, 8.0);
            b.line_to(3.5, 17.0);
            b.line_to(16.5, 17.0);
            b.line_to(16.5, 8.0);
            b.move_to(8.5, 8.0);
            b.line_to(8.5, 11.5);
            b.line_to(11.5, 11.5);
            b.line_to(11.5, 8.0);
        }
    }
    b.finish()
}

/// A page with a folded corner. Shared by [`IconKind::Generic`] and
/// [`IconKind::Document`], which differ only by the ruled lines the latter adds.
fn page(b: &mut PathBuilder) {
    b.move_to(4.5, 2.5);
    b.line_to(12.0, 2.5);
    b.line_to(15.5, 6.0);
    b.line_to(15.5, 17.5);
    b.line_to(4.5, 17.5);
    b.close();
    b.move_to(12.0, 2.5);
    b.line_to(12.0, 6.0);
    b.line_to(15.5, 6.0);
}

/// Circular-arc constant: the cubic control-point offset that approximates a quarter
/// ellipse to within about 0.02 % of its radius.
const KAPPA: f32 = 0.552_284_8;

fn circle(b: &mut PathBuilder, cx: f32, cy: f32, r: f32) {
    ellipse(b, cx, cy, r, r);
}

fn ellipse(b: &mut PathBuilder, cx: f32, cy: f32, rx: f32, ry: f32) {
    let (ox, oy) = (rx * KAPPA, ry * KAPPA);
    b.move_to(cx - rx, cy);
    b.cubic_to(cx - rx, cy - oy, cx - ox, cy - ry, cx, cy - ry);
    b.cubic_to(cx + ox, cy - ry, cx + rx, cy - oy, cx + rx, cy);
    b.cubic_to(cx + rx, cy + oy, cx + ox, cy + ry, cx, cy + ry);
    b.cubic_to(cx - ox, cy + ry, cx - rx, cy + oy, cx - rx, cy);
    b.close();
}

/// The bottom half of an ellipse, left to right. This is the cylinder's belly.
fn lower_arc(b: &mut PathBuilder, cx: f32, cy: f32, rx: f32, ry: f32) {
    let (ox, oy) = (rx * KAPPA, ry * KAPPA);
    b.move_to(cx - rx, cy);
    b.cubic_to(cx - rx, cy + oy, cx - ox, cy + ry, cx, cy + ry);
    b.cubic_to(cx + ox, cy + ry, cx + rx, cy + oy, cx + rx, cy);
}

fn rounded_rect(b: &mut PathBuilder, x: f32, y: f32, w: f32, h: f32, r: f32) {
    let r = r.min(w * 0.5).min(h * 0.5);
    let o = r * KAPPA;
    let (x1, y1) = (x + w, y + h);
    b.move_to(x + r, y);
    b.line_to(x1 - r, y);
    b.cubic_to(x1 - r + o, y, x1, y + r - o, x1, y + r);
    b.line_to(x1, y1 - r);
    b.cubic_to(x1, y1 - r + o, x1 - r + o, y1, x1 - r, y1);
    b.line_to(x + r, y1);
    b.cubic_to(x + r - o, y1, x, y1 - r + o, x, y1 - r);
    b.line_to(x, y + r);
    b.cubic_to(x, y + r - o, x + r - o, y, x + r, y);
    b.close();
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use super::*;

    /// Mean coverage, 0.0..=1.0. The share of the icon's box that is ink.
    fn ink(bitmap: &RasterizedGlyph) -> f32 {
        let sum: u64 = bitmap.coverage.iter().map(|&c| u64::from(c)).sum();
        sum as f32 / (bitmap.coverage.len() as f32 * 255.0)
    }

    fn kind_key(kind: IconKind, px: u16) -> IconKey {
        IconKey {
            shape: IconShape::Kind(kind),
            px,
        }
    }

    fn emblem_key(emblem: Emblem, px: u16) -> IconKey {
        IconKey {
            shape: IconShape::Emblem(emblem),
            px,
        }
    }

    fn state_key(state: StateIcon, px: u16) -> IconKey {
        IconKey {
            shape: IconShape::State(state),
            px,
        }
    }

    /// Mean absolute difference in coverage between two bitmaps of the same size, as a
    /// fraction of full scale. The measure both distinctness tests are written against.
    fn distance(a: &RasterizedGlyph, b: &RasterizedGlyph) -> f32 {
        let diff: u64 = a
            .coverage
            .iter()
            .zip(&b.coverage)
            .map(|(&x, &y)| u64::from(x.abs_diff(y)))
            .sum();
        diff as f32 / (a.coverage.len() as f32 * 255.0)
    }

    #[test]
    fn every_kind_rasterizes_to_a_square_of_the_requested_size() {
        for kind in IconKind::ALL {
            for px in [16u16, 20, 30, 40] {
                let bitmap = rasterize(kind_key(kind, px))
                    .unwrap_or_else(|| panic!("{kind:?} at {px}px produced nothing"));
                assert_eq!(
                    (bitmap.width, bitmap.height),
                    (u32::from(px), u32::from(px))
                );
                assert_eq!(bitmap.coverage.len(), (px as usize) * (px as usize));
            }
        }
    }

    #[test]
    fn no_icon_is_blank_and_none_is_a_solid_block() {
        // Both ends matter. Nothing means the path is wrong; everything means the strokes
        // merged into a smudge and the icon carries no shape.
        for kind in IconKind::ALL {
            let bitmap = rasterize(kind_key(kind, 20)).unwrap();
            let ink = ink(&bitmap);
            assert!(
                ink > 0.04,
                "{kind:?} is nearly empty at 20px (ink {ink:.3})"
            );
            assert!(
                ink < 0.55,
                "{kind:?} is nearly solid at 20px (ink {ink:.3})"
            );
        }
    }

    #[test]
    fn no_icon_is_clipped_by_the_edge_of_its_box() {
        // Half of a 1.5-unit stroke is 0.75, so a path drawn to the grid boundary would lose
        // ink to the mask edge. A border row with meaningful coverage is that bug.
        for kind in IconKind::ALL {
            let px = 40usize;
            let bitmap = rasterize(kind_key(kind, px as u16)).unwrap();
            let at = |x: usize, y: usize| bitmap.coverage[y * px + x];
            for i in 0..px {
                for (x, y, edge) in [
                    (i, 0, "top"),
                    (i, px - 1, "bottom"),
                    (0, i, "left"),
                    (px - 1, i, "right"),
                ] {
                    assert_eq!(
                        at(x, y),
                        0,
                        "{kind:?} has ink on the {edge} edge at {i}, so it is clipped"
                    );
                }
            }
        }
    }

    #[test]
    fn every_pair_of_icons_is_visibly_different_at_twenty_pixels() {
        // The acceptance criterion is "distinguishable at 20px", and the only honest way to
        // test it is against the real rasterization at that exact size. Mean absolute
        // difference in coverage, as a fraction of full scale.
        let rendered: Vec<_> = IconKind::ALL
            .iter()
            .map(|&kind| (kind, rasterize(kind_key(kind, 20)).unwrap()))
            .collect();

        let mut worst = (f32::MAX, IconKind::Generic, IconKind::Generic);
        for (i, (ka, a)) in rendered.iter().enumerate() {
            for (kb, b) in rendered.iter().skip(i + 1) {
                let diff: u64 = a
                    .coverage
                    .iter()
                    .zip(&b.coverage)
                    .map(|(&x, &y)| u64::from(x.abs_diff(y)))
                    .sum();
                let d = diff as f32 / (a.coverage.len() as f32 * 255.0);
                if d < worst.0 {
                    worst = (d, *ka, *kb);
                }
            }
        }
        assert!(
            worst.0 > 0.05,
            "{:?} and {:?} differ by only {:.3} of full coverage at 20px",
            worst.1,
            worst.2,
            worst.0
        );
    }

    #[test]
    fn every_state_icon_is_legible_at_the_size_chrome_actually_asks_for() {
        // Twelve device pixels, not twenty: a tab is where this set lives, and a set that is
        // only checked at the list icon's size is checked at a size it is never drawn at.
        // `MIN_PX` is 8, so 12 is inside the legible range by construction -- what this
        // asserts is that the shapes hold up there, which `STATE_STROKE` is the reason for.
        for state in StateIcon::ALL {
            for px in [12u16, 16, 20, 32] {
                let bitmap = rasterize(state_key(state, px))
                    .unwrap_or_else(|| panic!("{state:?} at {px}px produced nothing"));
                assert_eq!(
                    (bitmap.width, bitmap.height),
                    (u32::from(px), u32::from(px))
                );
                let ink = ink(&bitmap);
                assert!(
                    ink > 0.04,
                    "{state:?} is nearly empty at {px}px (ink {ink:.3})"
                );
                assert!(
                    ink < 0.55,
                    "{state:?} is nearly solid at {px}px (ink {ink:.3})"
                );
            }
        }
    }

    #[test]
    fn no_state_icon_is_clipped_by_the_edge_of_its_box() {
        // `STATE_STROKE` is 2.1 units, so half of it is 1.05 -- more than the 0.75 the kind
        // icons budget for, and the reason these paths are held inside a tighter box. A state
        // icon clipped on one side is a bell with no rim or a triangle with no base, which
        // reads as a different state rather than as a rendering fault.
        for state in StateIcon::ALL {
            let px = 40usize;
            let bitmap = rasterize(state_key(state, px as u16)).unwrap();
            let at = |x: usize, y: usize| bitmap.coverage[y * px + x];
            for i in 0..px {
                for (x, y, edge) in [
                    (i, 0, "top"),
                    (i, px - 1, "bottom"),
                    (0, i, "left"),
                    (px - 1, i, "right"),
                ] {
                    assert_eq!(
                        at(x, y),
                        0,
                        "{state:?} has ink on the {edge} edge at {i}, so it is clipped"
                    );
                }
            }
        }
    }

    #[test]
    fn every_shape_the_module_draws_is_visibly_different_from_every_other_one() {
        // Widened from the kind-only version when `StateIcon` landed, and the widening is the
        // point rather than tidiness: a directory row draws a kind icon *and* a session state
        // icon, side by side, so a `Working` ring that reads as a `Config` hex nut is a defect
        // no test over one set alone can see.
        //
        // Judged at 12px, the smallest size anything in the union is asked for, because that
        // is where two shapes are closest. The kind icons are drawn at 20 and pass at 12 with
        // room to spare, so nothing is being weakened to accommodate the new set.
        let px = 12u16;
        let rendered: Vec<(String, RasterizedGlyph)> = IconKind::ALL
            .iter()
            .map(|&k| (format!("{k:?}"), rasterize(kind_key(k, px)).unwrap()))
            .chain(
                StateIcon::ALL
                    .iter()
                    .map(|&s| (format!("{s:?}"), rasterize(state_key(s, px)).unwrap())),
            )
            .collect();

        let mut worst = (f32::MAX, String::new(), String::new());
        for (i, (na, a)) in rendered.iter().enumerate() {
            for (nb, b) in rendered.iter().skip(i + 1) {
                let d = distance(a, b);
                if d < worst.0 {
                    worst = (d, na.clone(), nb.clone());
                }
            }
        }
        assert!(
            worst.0 > 0.05,
            "{} and {} differ by only {:.3} of full coverage at {px}px",
            worst.1,
            worst.2,
            worst.0
        );
    }

    #[test]
    fn the_two_live_states_do_not_collapse_into_each_other() {
        // `Running` and `Working` are the pair a person reads fastest and the pair that costs
        // most when it is wrong: one means a shell is sitting at a prompt, the other means an
        // agent is mid-task. They are a filled disc and an open ring for exactly this reason,
        // and the gap in the ring is what survives antialiasing at a tab's size.
        //
        // Held to a bar four times the set's general one, because "distinguishable if you
        // compare them" is not the property -- "distinguishable while scanning" is.
        for px in [12u16, 20] {
            let running = rasterize(state_key(StateIcon::Running, px)).unwrap();
            let working = rasterize(state_key(StateIcon::Working, px)).unwrap();
            let d = distance(&running, &working);
            assert!(
                d > 0.20,
                "running and working differ by only {d:.3} at {px}px"
            );
        }
    }

    #[test]
    fn a_remembered_session_does_not_read_as_a_live_one() {
        // The pair `session-persistence` exists to keep apart. Its chunk says in as many words
        // that a remembered session must never look running, and this set is one of the three
        // channels that promise carries -- so `Remembered` is held to the same scanning bar
        // `Running` and `Working` are, against *both* live states rather than against the set's
        // general 0.05.
        //
        // The bar is 0.08 rather than the 0.20 `Running` and `Working` are held to, and the
        // difference is not a concession: those two are a filled disc against an open ring, so
        // almost none of their coverage coincides, while `Remembered` is deliberately drawn on
        // `Working`'s own circle and most of its ink is *supposed* to land where `Working`'s is.
        // Mean coverage difference is the wrong instrument for the pair — it measures how much
        // ink moved, and what a reader sees is where the gaps are. The measured margin is 0.098
        // at 12 px against `Working` and far more against `Running`; the silhouette itself is
        // held by `the_gaps_in_the_remembered_ring_survive_the_size_a_tab_draws_it_at`, which is
        // the test that actually catches this pair collapsing.
        for px in [12u16, 20] {
            let remembered = rasterize(state_key(StateIcon::Remembered, px)).unwrap();
            for live in [StateIcon::Running, StateIcon::Working] {
                let other = rasterize(state_key(live, px)).unwrap();
                let d = distance(&remembered, &other);
                assert!(
                    d > 0.08,
                    "remembered and {live:?} differ by only {d:.3} at {px}px"
                );
            }
        }
    }

    #[test]
    fn the_gaps_in_the_remembered_ring_survive_the_size_a_tab_draws_it_at() {
        // **The assertion the distance metric could not make, and a render is why it exists.**
        // The first `Remembered` was four 60-degree dashes, and at 12 px `STATE_STROKE`'s round
        // caps ate all but 0.7 px of every gap: the shape came out as a closed ring, which is
        // the one silhouette in this set that must not appear, because it is `Working` with its
        // gap filled in. Every numeric test passed — a nearly-closed ring is still far from a
        // three-quarter one by mean coverage, so `distance` had nothing to say.
        //
        // *Ink* is the obvious second instrument and it is also too blunt to use: a round cap
        // adds back most of the area the gap it sits beside removed, so three 70-degree dashes
        // ink 0.147 at 12 px against a 270-degree arc's 0.152 — a 4% difference for a shape
        // that is drawing 60 fewer degrees of circle. A threshold there would be a coin toss.
        //
        // So this counts the thing the eye actually reads: walking the circle the dashes lie on
        // and counting how many separate runs of ink it passes through. A closed ring is one
        // run whatever its coverage says, `Working` is one run, and this shape's whole claim is
        // that it is three. Checked at every size chrome asks for, because the failure is a
        // function of the ratio between the cap and the arc and a future stroke change moves it.
        for px in [12u16, 16, 20, 32] {
            assert_eq!(
                ink_runs_around(
                    &rasterize(state_key(StateIcon::Remembered, px)).unwrap(),
                    6.2
                ),
                3,
                "at {px}px the remembered ring is not three dashes: its gaps have closed up \
                 under the round caps and it is drawing a ring"
            );
            // The control. If this ever stops being one run, the instrument has drifted rather
            // than the shape — a three-quarter arc is one dash by construction.
            assert_eq!(
                ink_runs_around(&rasterize(state_key(StateIcon::Working, px)).unwrap(), 6.2),
                1,
                "the instrument is wrong: a three-quarter arc is one run at {px}px"
            );
        }
    }

    /// How many separate runs of ink a walk around the circle of radius `grid_r` passes through.
    ///
    /// `grid_r` is in the module's 20-unit grid; the walk is done in device pixels at whatever
    /// size the bitmap is. Nearest-neighbour sampling with a coverage floor rather than
    /// interpolation: the question is "is there ink here", and a threshold of a third of full
    /// coverage is well below a stroke's centre and well above the antialiased fringe that a
    /// closed gap leaves behind.
    fn ink_runs_around(bitmap: &RasterizedGlyph, grid_r: f32) -> usize {
        let px = bitmap.width as f32;
        let (c, r) = (px / 2.0, grid_r * px / GRID);
        // Four samples per device pixel of circumference, so a one-pixel gap cannot fall
        // between two samples and be missed.
        let steps = ((2.0 * std::f32::consts::PI * r) * 4.0).ceil() as usize;
        let lit: Vec<bool> = (0..steps)
            .map(|i| {
                let a = i as f32 / steps as f32 * std::f32::consts::TAU;
                let (x, y) = ((c + r * a.cos()) as usize, (c + r * a.sin()) as usize);
                bitmap
                    .coverage
                    .get(y * bitmap.width as usize + x)
                    .is_some_and(|&v| v > 85)
            })
            .collect();
        // Rotations of the same ring must give the same answer, so a run that straddles the
        // start of the walk is one run and not two: count rising edges against the previous
        // sample, wrapping.
        (0..steps)
            .filter(|&i| lit[i] && !lit[(i + steps - 1) % steps])
            .count()
    }

    #[test]
    fn a_state_icon_costs_one_entry_per_state_and_not_one_per_session() {
        // The `IconShape` argument, one set further on. A key that carried the session id --
        // the obvious way to let a tab tint its own icon -- would store one rasterization per
        // open session, which is unbounded by construction. The tint is applied at draw time
        // to one shared coverage mask, exactly as it is for kinds.
        let mut keys = std::collections::BTreeSet::new();
        for state in StateIcon::ALL {
            for _session in 0..50 {
                keys.insert(state_key(state, 12));
            }
        }
        assert_eq!(keys.len(), StateIcon::ALL.len());
    }

    #[test]
    fn the_same_pixel_grid_is_one_key_however_it_was_reached() {
        // 20 logical at 150% and 30 logical at 100% are the same rasterization, and the key
        // has to say so or the atlas stores both.
        let a = IconKey::new(IconKind::Folder, 20.0, 1.5);
        let b = IconKey::new(IconKind::Folder, 30.0, 1.0);
        assert_eq!(a, b);
        assert_eq!(a.px, 30);
    }

    #[test]
    fn sizes_outside_the_legible_range_are_refused_rather_than_smudged() {
        assert!(rasterize(IconKey::new(IconKind::Folder, 20.0, 0.2)).is_none());
        assert!(rasterize(IconKey::new(IconKind::Folder, 20.0, 8.0)).is_none());
        // A non-finite scale must not panic its way through the cast.
        assert_eq!(IconKey::new(IconKind::Folder, 20.0, f32::NAN).px, 0);
    }

    #[test]
    fn rasterizing_is_deterministic() {
        // The atlas caches on the assumption that a key names one bitmap forever.
        let key = IconKey::new(IconKind::Archive, 20.0, 2.0);
        assert_eq!(rasterize(key), rasterize(key));
    }

    #[test]
    fn the_plate_is_solid_and_the_mark_is_not() {
        // Two different jobs, and the numbers are the jobs. A plate that is not nearly solid
        // knocks nothing out and the mark goes back to sitting on the silhouette's strokes; a
        // mark that *is* nearly solid is a blob rather than an arrow.
        let px = emblem_px(20).expect("the 20px list icon carries an emblem");
        let bitmap = rasterize(emblem_key(Emblem::Plate, px)).unwrap();
        // The plate is a disc inset by half a unit, so its mean coverage cannot exceed
        // (pi/4) * (19/20)^2 = 0.709 however opaque it is. The bar is set against that
        // ceiling. It was 0.80 while the plate was a rounded square, and rewriting it here
        // rather than loosening it is the point: the number describes the shape.
        let plate = ink(&bitmap);
        assert!(
            plate > 0.65,
            "the plate is not opaque enough (ink {plate:.3})"
        );
        // Mean coverage alone would be satisfied by a uniformly translucent disc, which knocks
        // nothing out. The middle -- where the mark goes -- has to be *fully* opaque.
        let mid = usize::from(px) / 2;
        assert_eq!(
            bitmap.coverage[mid * usize::from(px) + mid],
            255,
            "the plate is translucent"
        );

        let mark = ink(&rasterize(emblem_key(Emblem::Symlink, px)).unwrap());
        assert!(
            (0.10..0.60).contains(&mark),
            "the symlink mark is a blob or a ghost at {px}px (ink {mark:.3})"
        );
    }

    #[test]
    fn an_emblem_costs_one_entry_per_emblem_and_not_one_per_pair() {
        // The whole reason `IconShape` exists. Drawing every kind with the symlink emblem must
        // need `kinds + emblems` distinct atlas keys, not `kinds * (emblems + 1)`.
        let mut keys = std::collections::BTreeSet::new();
        for kind in IconKind::ALL {
            keys.insert(kind_key(kind, 20));
            for emblem in Emblem::ALL {
                keys.insert(emblem_key(emblem, 10));
            }
        }
        assert_eq!(
            keys.len(),
            IconKind::ALL.len() + Emblem::ALL.len(),
            "the key is keying the pair, so the atlas will store one entry per combination"
        );
    }

    #[test]
    fn emblem_px_never_asks_for_a_size_rasterize_would_refuse() {
        // The hazard this pair of functions exists to prevent: `icon_entry` counts a refused
        // rasterization as a dropped icon, and a dropped icon is the CPU tier's upload-budget
        // failure. "Too small to draw" must be answered *before* the atlas is asked, or a
        // design decision shows up in telemetry as a rendering fault.
        for icon_px in 0..=MAX_PX {
            if let Some(px) = emblem_px(icon_px) {
                for emblem in Emblem::ALL {
                    assert!(
                        rasterize(emblem_key(emblem, px)).is_some(),
                        "emblem_px({icon_px}) offered {px}px, which rasterize refuses for {emblem:?}"
                    );
                }
            }
        }
        // And the boundary is where MIN_PX puts it, not somewhere rounding drifted to.
        // Stated as a property rather than as arithmetic, so tuning EMBLEM_FRACTION against a
        // rendered image does not silently move it.
        let smallest = (0..=MAX_PX)
            .find(|&px| emblem_px(px).is_some())
            .expect("no icon size at all carries an emblem");
        assert_eq!(
            emblem_px(smallest),
            Some(MIN_PX),
            "the first emblem offered is not at the legibility floor"
        );
        assert_eq!(emblem_px(smallest - 1), None);
        // And the size the emblem was actually judged at has to be one of them.
        assert!(
            emblem_px(GRID as u16).is_some(),
            "the 20px list icon carries no emblem"
        );
    }

    #[test]
    fn the_emblem_changes_every_icon_it_sits_on_at_twenty_pixels() {
        // Criterion 1 is settled by looking, not by this. But a symlink that stopped altering
        // its target's picture would be a silent regression, and this is the cheap guard: the
        // emblem is composited over each kind exactly as the renderer layers it, and the
        // result has to differ from the bare icon by more than antialiasing noise.
        let px = 20usize;
        let e_px = emblem_px(px as u16).expect("a 20px icon carries an emblem");
        let plate = rasterize(emblem_key(Emblem::Plate, e_px)).unwrap();
        let mark = rasterize(emblem_key(Emblem::Symlink, e_px)).unwrap();

        // Bottom-left corner, the same origin `qs_ui::row` uses.
        let (ox, oy) = (0usize, px - e_px as usize);
        for kind in IconKind::ALL {
            let base = rasterize(kind_key(kind, px as u16)).unwrap();
            let mut over = base.coverage.clone();
            for layer in [&plate, &mark] {
                for y in 0..layer.height as usize {
                    for x in 0..layer.width as usize {
                        let c = layer.coverage[y * layer.width as usize + x];
                        let dst = &mut over[(oy + y) * px + ox + x];
                        *dst = (*dst).max(c);
                    }
                }
            }
            let diff: u64 = base
                .coverage
                .iter()
                .zip(&over)
                .map(|(&a, &b)| u64::from(a.abs_diff(b)))
                .sum();
            let d = diff as f32 / (base.coverage.len() as f32 * 255.0);
            assert!(
                d > 0.05,
                "the symlink emblem barely changes {kind:?} at 20px (differs by {d:.3})"
            );
        }
    }
}
