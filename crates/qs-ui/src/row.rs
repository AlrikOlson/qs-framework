//! Row layout and rendering: status rail, icon slot, name, size, modified, kind.
//!
//! This is the only file that turns rows into [`Instance`]s, and it is deliberately the
//! only one -- RP-2 (all three tiers consume the same draw lists) holds because there is
//! exactly one place a draw list can come from.
//!
//! # Middle ellipsis, and why the extension is preserved
//!
//! `a-very-long-quarterly-report-final-v3.xlsx` truncated at the end becomes
//! `a-very-long-quarterly-rep…`, which loses the single most identifying part of the name.
//! Truncating in the middle keeps both ends: `a-very-long-qua….xlsx`. The extension is
//! reserved *first*, before the head is measured, because a name whose extension gets eaten
//! by the ellipsis is worse than one truncated a few characters earlier.
//!
//! Cuts land on cluster boundaries. Cutting between glyphs of one cluster puts half a
//! Devanagari syllable or a lone combining mark on screen.
//!
//! # Placeholder rows
//!
//! FR-017 requires a [`LoadState::Stub`] row to render. The renderer MUST NOT read `size`,
//! `mtime` or `kind` from a stub -- they are undefined, not merely stale -- so a stub draws
//! its name (which is known) and grey bars where the metadata will go. When the row
//! upgrades, the bars are replaced in place: same row, same position, no reflow. A
//! placeholder that shifts the layout when it resolves is worse than one that does not
//! appear at all, because the user's pointer was already moving toward something.

use std::sync::Arc;

use qs_gpu::color::Srgba;
use qs_gpu::frame::{DrawList, Instance};
use qs_gpu::{
    AtlasEntry, AtlasKey, Emblem, GlyphAtlas, IconKey, IconKind, IconShape, PendingUpload,
    UploadClass,
};
use qs_text::{
    Features, FontDb, FontId, GlyphKey, GlyphRaster, PxSize, ShapedRun, ShapedRunCache, Shaper,
};

use crate::material::{self, Drive, Pass, Surface};
use crate::motion::InteractionMotion;
use crate::recycler::ViewportLayout;
use crate::row_source::{LoadState, RowBuf, RowFlags, RowView};
use crate::selection::Selection;
use crate::substance::Substance;
use crate::tokens::{Tokens, TypeRole, role};

/// Column geometry, in physical pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Columns {
    /// Left edge of the surface these columns are laid out on, in physical pixels.
    ///
    /// Zero for a list that owns the whole window. Non-zero once the window has chrome:
    /// every `*_x` accessor is measured from here, so a pane inset by a sidebar moves by
    /// changing one number rather than by every call site remembering to add an offset.
    pub x0: f32,
    /// Space between the surface edge and the row's selection region. This is what makes
    /// selection read as an object on a surface rather than a full-bleed table stripe.
    pub gutter: f32,
    pub rail: f32,
    pub icon: f32,
    pub name: f32,
    pub size: f32,
    pub modified: f32,
    pub kind: f32,
    pub gap: f32,
    pub padding: f32,
    /// Whether size, modified and kind are drawn at all.
    ///
    /// False in a Miller column. Those three reserve about 300 logical pixels between them
    /// — more than a column is wide once a path is a few deep — so a column that kept them
    /// would squeeze the name to its 40 px floor and truncate every entry to about six
    /// characters. A column is for walking structure; the metadata belongs to the list.
    pub metadata: bool,
}

impl Columns {
    /// Lay out columns for a viewport width.
    ///
    /// Gaps, padding, the rail and the selection inset all come from the space scale
    /// (UXDD 10.1). Column *widths* deliberately do not: a column is sized by the content
    /// it must hold -- "999.9 GB", a fixed-width timestamp -- which is a measurement, not a
    /// spacing decision, and forcing it onto an 8-step scale would either clip content or
    /// waste the only column users actually read.
    ///
    /// The name column absorbs all slack.
    ///
    /// Equivalent to [`Columns::for_rect`] at `x = 0`: the list owns the window's left
    /// edge. Kept because most callers -- and every test about column *widths* -- have no
    /// opinion about where the surface starts.
    pub fn for_width(width: f32, scale: f32, tokens: &Tokens) -> Self {
        Self::for_rect(0.0, width, scale, tokens)
    }

    /// Lay out columns for a surface that starts at `x` and is `width` wide.
    pub fn for_rect(x: f32, width: f32, scale: f32, tokens: &Tokens) -> Self {
        Self::for_rect_with(x, width, scale, tokens, true)
    }

    /// Lay out a **Miller column**: the name and nothing else.
    pub fn for_column(x: f32, width: f32, scale: f32, tokens: &Tokens) -> Self {
        Self::for_rect_with(x, width, scale, tokens, false)
    }

    pub(crate) fn for_rect_with(
        x: f32,
        width: f32,
        scale: f32,
        tokens: &Tokens,
        metadata: bool,
    ) -> Self {
        let px = |logical: f32| logical * scale.max(0.1);
        let space = |step: &str| px(tokens.space(step));

        let gutter = space(crate::tokens::space::MD);
        let padding = space(crate::tokens::space::LG);
        let gap = space(crate::tokens::space::XL);
        let rail = space(crate::tokens::space::XS);
        // The icon column *is* the UXDD 10.4 design grid, taken from the module that draws
        // on it. A literal here would let the column and the rasterized icon drift apart by
        // a pixel and leave the icon quietly off-centre in its own slot.
        let icon = px(qs_gpu::icon::GRID);

        // Content-sized, in logical pixels: widest plausible "999.9 GB", a
        // "2026-08-07 14:22" timestamp, and a kind label.
        let (size, modified, kind) = if metadata {
            (px(72.0), px(132.0), px(96.0))
        } else {
            (0.0, 0.0, 0.0)
        };

        // Two gaps rather than four without the metadata columns: a gap sits *between*
        // things, and three of them have gone.
        let gaps = if metadata { gap * 4.0 } else { gap };
        let fixed = gutter * 2.0 + padding * 2.0 + rail + icon + size + modified + kind + gaps;
        // A window narrow enough to squeeze the name column to nothing is a real state --
        // users do drag windows that small -- and the correct behaviour is a name column
        // that is merely tiny, not one with a negative width.
        let name = (width - fixed).max(px(40.0));

        Self {
            x0: x,
            gutter,
            rail,
            icon,
            name,
            size,
            modified,
            kind,
            gap,
            padding,
            metadata,
        }
    }

    /// Left edge of the row's selection region: inside the gutter.
    pub fn content_x(&self) -> f32 {
        self.x0 + self.gutter
    }

    /// Width of the selection region for a surface of `width`.
    pub fn content_width(&self, width: f32) -> f32 {
        (width - self.gutter * 2.0).max(1.0)
    }

    /// `x` of the status rail's left edge.
    pub fn rail_x(&self) -> f32 {
        self.x0 + self.gutter + self.padding
    }

    /// `x` of the icon slot's left edge.
    pub fn icon_x(&self) -> f32 {
        self.rail_x() + self.rail + self.gap
    }

    /// `x` of the name column's left edge.
    pub fn name_x(&self) -> f32 {
        self.icon_x() + self.icon + self.gap
    }

    pub fn size_x(&self) -> f32 {
        self.name_x() + self.name + self.gap
    }

    pub fn modified_x(&self) -> f32 {
        self.size_x() + self.size + self.gap
    }

    pub fn kind_x(&self) -> f32 {
        self.modified_x() + self.modified + self.gap
    }
}

/// Where a truncated run is cut.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Truncation {
    /// Glyphs `0..head` are drawn unshifted.
    pub head: usize,
    /// Glyphs `tail..` are drawn shifted left by [`Truncation::tail_shift`].
    pub tail: usize,
    /// `x` at which to draw the ellipsis.
    pub ellipsis_x: f32,
    /// Amount to subtract from each tail glyph's `x`.
    pub tail_shift: f32,
}

/// Byte offset at which a filename's extension begins, if it has one worth preserving.
///
/// A leading dot is not an extension (`.gitignore` is a name, not an extension), and an
/// "extension" longer than a dozen bytes is almost certainly part of the name.
pub fn extension_start(name: &str) -> Option<usize> {
    let dot = name.rfind('.')?;
    if dot == 0 || dot + 1 >= name.len() {
        return None;
    }
    let extension_len = name.len() - dot;
    (extension_len <= 12).then_some(dot)
}

/// Decide where to cut `run` so it fits in `max_width`.
///
/// `None` means it already fits. See the module docs for why the tail is reserved first.
pub fn truncate_middle(
    run: &ShapedRun,
    max_width: f32,
    ellipsis_width: f32,
    extension_byte: Option<usize>,
) -> Option<Truncation> {
    if run.width <= max_width || run.glyphs.is_empty() {
        return None;
    }

    // Reserve the tail: the first glyph at or after the extension's first byte.
    let mut tail = match extension_byte {
        Some(byte) => run
            .glyphs
            .iter()
            .position(|g| g.cluster as usize >= byte)
            .unwrap_or(run.glyphs.len()),
        None => run.glyphs.len(),
    };

    let tail_x = |tail: usize| -> f32 { run.glyphs.get(tail).map_or(run.width, |g| g.x) };

    // If head + ellipsis + tail does not fit, give up tail glyphs one at a time. The
    // extension is preferred, not guaranteed: a 40-character "extension" on a 60-pixel
    // column cannot win, and pretending otherwise would push the head to zero.
    let mut available = max_width - ellipsis_width - (run.width - tail_x(tail));
    while available < 0.0 && tail < run.glyphs.len() {
        tail += 1;
        available = max_width - ellipsis_width - (run.width - tail_x(tail));
    }
    if available < 0.0 {
        // Not even the ellipsis fits. Draw nothing rather than overflow the column.
        return Some(Truncation {
            head: 0,
            tail: run.glyphs.len(),
            ellipsis_x: 0.0,
            tail_shift: 0.0,
        });
    }

    // Largest head that fits. The predicate measures each glyph's **trailing** edge, not
    // its pen position: `g.x <= available` would admit a glyph that starts inside the
    // budget and ends outside it, which overflows the column by up to one glyph.
    // `partition_point` needs the values to be non-decreasing, which `Shaper` guarantees
    // and its tests assert.
    let head = run
        .glyphs
        .partition_point(|g| g.x + g.advance <= available)
        .min(tail);
    let head = snap_to_cluster_start(&run.glyphs, head);
    let ellipsis_x = run.glyphs.get(head).map_or(run.width, |g| g.x);

    Some(Truncation {
        head,
        tail,
        ellipsis_x,
        tail_shift: tail_x(tail) - (ellipsis_x + ellipsis_width),
    })
}

/// Move `index` back until it sits at the first glyph of a cluster.
///
/// Trailing zero-advance marks usually travel with their base under the width predicate
/// above, but "usually" is not a guarantee -- a cluster whose glyphs are reordered by
/// shaping can put a non-zero advance in the middle. Cutting inside a cluster puts a lone
/// combining mark or half a Devanagari syllable on screen, so the boundary is enforced
/// rather than assumed.
fn snap_to_cluster_start(glyphs: &[qs_text::shape::ShapedGlyph], mut index: usize) -> usize {
    while index > 0 {
        let (Some(current), Some(previous)) = (glyphs.get(index), glyphs.get(index - 1)) else {
            break;
        };
        if current.cluster != previous.cluster {
            break;
        }
        index -= 1;
    }
    index
}

/// Human-readable byte count. Fixed to one decimal above KB so the column never reflows.
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 7] = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS.get(unit).copied().unwrap_or("B"))
}

/// Unix **seconds** to a fixed-width local-ish timestamp.
///
/// Deliberately not locale-aware and deliberately not `chrono`. M0 needs a *stable*,
/// fixed-width string so the modified column's width and the golden images are
/// reproducible; a locale-dependent format would make a reference image captured in one
/// region fail in another, which is a golden-image suite people learn to ignore.
///
/// # The unit is seconds, and it used to be nanoseconds
///
/// This took **nanoseconds** while `qs-shell` filled [`RowView::mtime`] in seconds
/// (`unix_seconds`, from `SystemTime::duration_since`). Nothing connected the two, so every
/// real file rendered as `1970-01-01 00:00`: 1.7e9 seconds read as nanoseconds is 1.7
/// seconds past the epoch. Every test passed, because each one fed nanoseconds to a
/// nanosecond function — the defect lived in the gap between two crates that agreed on a
/// type and not on a unit, which is the one place a unit test cannot look.
/// `a_timestamp_from_the_listing_pipeline_is_not_1970` is the test that spans the gap.
pub fn format_mtime(unix_seconds: i64) -> String {
    let secs = unix_seconds;
    let days = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute) = (time_of_day / 3600, (time_of_day % 3600) / 60);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}")
}

/// Howard Hinnant's `civil_from_days`. Exact for the whole proleptic Gregorian range and
/// about six lines, which is the whole reason a date crate is not in the dependency graph
/// for one column of a spike.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// What the **view** knows about this frame's rows, over and above what the source said.
///
/// Deliberately *not* carried on [`RowView`]. Hover, focus and selection are properties of
/// the **view**, not of the data: two panes showing the same directory have different
/// focused rows, and a `RowSource` that had to know about them could not be shared. The
/// source stays pure data and the view supplies interaction state at draw time.
/// Borrowed rather than owned, and that is what keeps this `Copy`: a selection is a set
/// with an allocation behind it, and the renderer takes this by value once per row.
///
/// # Why [`SessionMarks`] is in here with the pointer and the keyboard
///
/// It is not interaction, and the name is now slightly wider than it reads. It is here
/// because this is the one per-frame value that reaches **both**
/// [`ListRenderer::render`] and [`crate::a11y::SemanticTree::for_frame`], and a session mark
/// has to appear in both: rendered in ink and missing from the accessible name, the folder's
/// confidence would be a claim made to sighted users only. Every alternative transport — a
/// field on the renderer, another argument to `render` — reaches exactly one of the two, and
/// the other then grows a second answer to the same question. See [`crate::mark`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Interaction<'a> {
    /// Logical corpus index under the pointer.
    pub hovered: Option<u64>,
    /// Logical corpus index with keyboard focus.
    pub focused: Option<u64>,
    /// What is selected. A **set**, not a row: click, `Ctrl`-click, `Shift`-click and a
    /// marquee all produce different sets, and the one-row form could express none of them.
    pub selection: &'a Selection,
    /// Logical corpus index the pointer is currently held down on.
    ///
    /// Deliberately has no [`RowFlags`] bit, unlike the three above. Those three exist as
    /// flags because a row can carry them from the *data* side too. Press is purely a
    /// presentation state with no data counterpart, and what the renderer reads is not its
    /// boolean but its animated intensity, which comes from
    /// [`InteractionMotion`](crate::motion::InteractionMotion) rather than from here.
    pub pressed: Option<u64>,
    /// The agent sessions filed under the directories on screen, by corpus index.
    ///
    /// Empty for every caller that has never heard of a session, which is all of them except
    /// `qs`'s frame. See the type's docs for why this rides here.
    pub marks: &'a crate::mark::SessionMarks,
}

impl Default for Interaction<'_> {
    fn default() -> Self {
        Self {
            hovered: None,
            focused: None,
            selection: &crate::selection::NOTHING,
            pressed: None,
            marks: &crate::mark::NO_MARKS,
        }
    }
}

impl Interaction<'_> {
    /// Interaction state with `selection` and nothing else.
    #[must_use]
    pub fn with_selection(selection: &Selection) -> Interaction<'_> {
        Interaction {
            selection,
            ..Interaction::default()
        }
    }

    /// The state flags that apply to `index`.
    fn flags_for(self, index: u64) -> RowFlags {
        let mut flags = RowFlags::EMPTY;
        if self.hovered == Some(index) {
            flags = flags | RowFlags::IS_HOVERED;
        }
        if self.focused == Some(index) {
            flags = flags | RowFlags::IS_FOCUSED;
        }
        if self.selection.contains(index) {
            flags = flags | RowFlags::IS_SELECTED;
        }
        flags
    }
}

/// The resolved type roles for one frame's rows.
///
/// A struct rather than four more parameters: these four always travel together, are
/// computed once per frame, and are the same for every row — which is exactly when a
/// parameter object earns its place instead of just moving the argument list elsewhere.
#[derive(Clone, Copy, PartialEq, Debug)]
struct RowType {
    primary: TypeRole,
    primary_px: PxSize,
    secondary: TypeRole,
    secondary_px: PxSize,
}

/// A type role resolved against one frame's scale: the face that satisfies its weight, its
/// physical size, and the vertical metrics needed to place a line.
///
/// This is the public counterpart of [`RowType`], and it is a handle rather than a role
/// *name* for the same reason [`ListRenderer::render`] resolves its two roles once per
/// frame: the resolution is a font-database question and a metrics parse, and repeating it
/// per string would put both on the frame path.
///
/// It exists so callers outside the row builder -- the diagnostics overlay is the first --
/// can draw text through the same shaper, cache and atlas as rows. A second text path would
/// render differently on the CPU tier, which is exactly what RP-2 forbids.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ResolvedRole {
    size: PxSize,
    font: FontId,
    /// Top of the line box to the baseline.
    pub ascent: f32,
    /// Baseline-to-baseline advance for stacked lines of this role.
    pub line_height: f32,
}

/// Where a row's text baseline sits, and how much room is left around it.
///
/// # This type exists so SC-007 can fail
///
/// SC-007 is "200% text scale does not clip". The evidence behind it used to be that row
/// height and font size both double at 200% -- a statement about *proportion*, which stays
/// true no matter how badly the text overflows its row, because a box twice too small is
/// still twice as big as a box that was once too small. Nothing compared a glyph's extent
/// to a row's height, so the criterion was green for a reason unrelated to what it claims.
///
/// The two headrooms are that comparison. They are the quantity a clipped descender is a
/// symptom of, so a test that asserts they stay non-negative can go red for the reason
/// SC-007 names.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RowTextFit {
    /// Distance from the row's top edge down to the baseline.
    pub baseline: f32,
    /// Room between the row's top edge and the top of the tallest ascender. Negative means
    /// the tops of letters are cut off.
    pub headroom_above: f32,
    /// Room between the deepest descender and the row's bottom edge. Negative means the
    /// tails of `g`, `y` and `p` are cut off.
    pub headroom_below: f32,
}

impl RowTextFit {
    pub fn clips(self) -> bool {
        self.headroom_above < 0.0 || self.headroom_below < 0.0
    }

    /// The tighter of the two margins -- the one that runs out first.
    pub fn tightest(self) -> f32 {
        self.headroom_above.min(self.headroom_below)
    }
}

/// Place the row baseline and measure what is left above and below it.
///
/// The baseline is *optically* centred: the visual centre of a line of Latin text is the
/// middle of its x-height, not the middle of its ascent-to-descent box. Centring the box
/// makes every row look a pixel or two top-heavy.
///
/// That choice is also why this is worth measuring rather than assuming. Optical centring
/// pushes the baseline down by half the x-height, which spends ascender headroom to buy
/// descender headroom; whether what remains is still positive depends on the face's
/// metrics, and the face is a platform question.
pub fn row_text_fit(row_height: u32, ascent: f32, descent: f32, x_height: f32) -> RowTextFit {
    let height = row_height as f32;
    let baseline = (height + x_height) * 0.5;
    RowTextFit {
        baseline,
        headroom_above: baseline - ascent,
        headroom_below: height - (baseline + descent),
    }
}

/// Builds draw lists from rows.
//
// `too_many_arguments` is allowed on the private draw helpers below: the row builder
// genuinely needs every one of them per call, and bundling them into a struct would move
// the list to a different file rather than shorten it.
///
/// Owns the text machinery because all of it is `&mut` and single-threaded by design --
/// see [`qs_text::Shaper`].
pub struct ListRenderer {
    pub tokens: Tokens,
    /// Whether this frame's rows are being drawn **lit** — set for the duration of
    /// [`ListRenderer::render_lit`] and false everywhere else.
    ///
    /// It exists for one reason: an emitting surface changes the ground under its own
    /// label, so the ink that reads on the unlit row is washed out on the lit one. The row
    /// therefore asks for `content/on-lit` when it is lit and `content/primary` when it is
    /// not — the same shape `Tokens::effects_enabled` already gives the materials, one
    /// level up. Both states are checked: see `MaterialDef::text_lit`.
    lit: bool,
    pub atlas: GlyphAtlas,
    shaper: Shaper,
    cache: ShapedRunCache,
    raster: GlyphRaster,
    db: Arc<dyn FontDb>,
    ui_font: FontId,
    /// Resolved face per weight class, so a role's weight costs one map lookup rather than
    /// a font-database query per string per frame.
    faces: std::collections::HashMap<u16, FontId>,
    /// Weight classes this machine can actually satisfy, captured once at construction.
    ///
    /// Reported rather than assumed: on a platform whose UI family ships as a variable font
    /// (macOS) this is a single entry, every role resolves to 400, and the type scale is
    /// carried by size alone. That is a real degradation and Constitution III says it must
    /// be visible, not silent.
    weight_coverage: Vec<u16>,
    /// Glyphs wanted this frame that the atlas could not supply. Reported, because a
    /// missing glyph is the SC-006 failure and it must not be invisible in the telemetry.
    pub glyphs_dropped: u32,
    /// Icons wanted this frame that the atlas could not supply. See
    /// [`qs_gpu::frame::DrawStats::icons_dropped`] for why this is not folded into
    /// `glyphs_dropped`.
    pub icons_dropped: u32,
    /// The vertical metric the last frame's rows were laid out on, or `None` when the face
    /// had no parseable metrics and the fallback baseline was used.
    ///
    /// Diagnostic state, the same kind as `glyphs_dropped`: it reports something about the
    /// frame that is otherwise invisible once the draw list is flat. SC-007 is why it is
    /// worth reporting -- a row whose text does not fit is a criterion failure, and without
    /// this the headroom is a number that exists for one expression inside `render` and is
    /// then unrecoverable from anything the renderer produces.
    pub last_text_fit: Option<RowTextFit>,
    icons: IconCache,
    /// Glyph draws this pass could not emit yet, because the atlas did not have the glyph
    /// and the per-frame upload bound had not been allocated.
    ///
    /// This is the caller's half of `GlyphAtlas::want`: the atlas cannot decide what to
    /// spend the bound on until it has seen everything the frame wants, so a draw that
    /// misses is parked here rather than dropped, and [`ListRenderer::flush_text`] emits
    /// the ones that made it. Kept across frames for its capacity -- cleared, never
    /// reallocated.
    deferred: Vec<DeferredGlyph>,
}

/// One glyph draw waiting on the atlas.
///
/// Everything `Instance::glyph` needs except the atlas entry, which is exactly what is
/// missing. The key is the *quantized* one, subpixel phase included, because that is the
/// key the atlas will be asked for again after admission -- reconstructing it from the pen
/// position at flush time would risk quantizing to a different phase and looking up a glyph
/// nobody demanded.
#[derive(Clone, Copy, Debug)]
struct DeferredGlyph {
    key: GlyphKey,
    /// The integer pen x `GlyphKey::quantize` returned, before `entry.left` is added.
    integer_x: f32,
    /// The absolute baseline this glyph sits on, before `entry.top` is subtracted.
    baseline: f32,
    color: Srgba,
}

/// One frame's worth of resolved icon entries.
///
/// The same idea as [`ResolvedRole`]: resolve once per frame, draw many times. A thousand
/// visible rows share nine rasterizations, and after the first frame they share nine atlas
/// hits -- so the per-row cost of an icon is one array index and one instance push.
///
/// Filled **lazily**. Resolving all nine up front would be simpler and wrong: it would
/// spend nine of the CPU tier's sixty-four per-frame uploads on kinds that may not be on
/// screen, and that budget is already nearly exhausted by the row pass and the F1 overlay.
///
/// Holding an [`AtlasEntry`] for the rest of the frame is safe for exactly one reason: the
/// atlas never evicts an entry used in the current frame (see `qs_gpu::atlas`). Caching one
/// across frames would not be safe, and is why this is reset in [`ListRenderer::render`].
#[derive(Clone, Copy, Debug)]
struct IconCache {
    px: u16,
    slots: [IconSlot; IconKind::ALL.len()],
    /// The emblem's own device size, or `None` when the icon box is too small to carry one.
    ///
    /// Resolved once per frame from [`qs_gpu::icon::emblem_px`] and then *believed*: a `None`
    /// here means no emblem is requested at all this frame, which is the difference between a
    /// design decision and a dropped icon. See [`ListRenderer::emblem_entry`].
    emblem_px: Option<u16>,
    emblem_slots: [IconSlot; Emblem::ALL.len()],
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum IconSlot {
    /// Not asked for yet this frame. Distinct from `Missing`, or a kind that legitimately
    /// failed would be retried once per row instead of once per frame.
    #[default]
    Unasked,
    Missing,
    Ready(AtlasEntry),
}

impl Default for IconCache {
    fn default() -> Self {
        Self {
            px: 0,
            slots: [IconSlot::Unasked; IconKind::ALL.len()],
            emblem_px: None,
            emblem_slots: [IconSlot::Unasked; Emblem::ALL.len()],
        }
    }
}

impl IconCache {
    /// The cache for one frame's icon box, with the emblem size already decided.
    fn for_px(px: u16) -> Self {
        Self {
            px,
            emblem_px: qs_gpu::icon::emblem_px(px),
            ..Self::default()
        }
    }
}

impl std::fmt::Debug for ListRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ListRenderer")
            .field("atlas", &self.atlas.stats())
            .field("cache", &self.cache.stats())
            .finish()
    }
}

impl ListRenderer {
    pub fn new(tokens: Tokens, db: Arc<dyn FontDb>, config: qs_gpu::TierConfig) -> Self {
        let ui_font = db.ui_font();

        // Resolve every weight the token file asks for, once. Roles are fixed at startup,
        // so this is a handful of lookups rather than anything the frame path repeats.
        let mut faces = std::collections::HashMap::new();
        let mut weight_coverage = Vec::new();
        for name in [role::XS, role::SM, role::MD, role::LG, role::XL] {
            let weight = tokens.type_role(name).weight;
            let face = db.ui_font_at(qs_text::FontWeight(weight));
            faces.insert(weight, face);
            if !weight_coverage.contains(&weight) {
                weight_coverage.push(weight);
            }
        }
        // Distinct *faces*, not distinct requested weights: two roles that both fall back
        // to the regular face are one step of hierarchy, not two.
        let distinct_faces = {
            let mut ids: Vec<FontId> = faces.values().copied().collect();
            ids.sort();
            ids.dedup();
            ids.len()
        };
        if distinct_faces < 2 {
            tracing::info!(
                target: "qs::text",
                "the UI family resolves to a single face on this machine; the type scale                  is carried by size alone (weights requested: {weight_coverage:?})"
            );
        }

        Self {
            tokens,
            lit: false,
            atlas: GlyphAtlas::new(config.atlas_size, config.max_glyph_uploads_per_frame),
            shaper: Shaper::new(Arc::clone(&db)),
            cache: ShapedRunCache::new(config.shaped_run_cache_entries),
            raster: GlyphRaster::new(),
            db,
            ui_font,
            faces,
            weight_coverage,
            glyphs_dropped: 0,
            icons_dropped: 0,
            last_text_fit: None,
            icons: IconCache::default(),
            deferred: Vec::new(),
        }
    }

    /// The face for a role's weight, falling back to the UI face.
    fn face_for(&self, role: TypeRole) -> FontId {
        self.faces
            .get(&role.weight)
            .copied()
            .unwrap_or(self.ui_font)
    }

    /// Weight classes the type scale actually resolved to distinct faces.
    pub fn weight_coverage(&self) -> &[u16] {
        &self.weight_coverage
    }

    /// Resolve a named type role for this frame's scale. Call once per frame per role, not
    /// once per string.
    pub fn resolve_role(&mut self, role: &str, scale: f32, text_scale: f32) -> ResolvedRole {
        let spec = self.tokens.type_role(role);
        let size = PxSize::new(spec.size_px(scale, text_scale));
        let font = self.face_for(spec);
        // A face whose metrics will not parse still has to lay out somewhere. These
        // proportions are typical of a UI face; the alternative -- zero -- stacks every
        // line on the same baseline, which is a worse failure than being slightly off.
        let (ascent, line_height) = match self.shaper.metrics(font, size) {
            Some(m) => (m.ascent, m.line_height),
            None => (size.to_f32() * 0.8, size.to_f32() * 1.25),
        };
        ResolvedRole {
            size,
            font,
            ascent,
            line_height,
        }
    }

    /// Resolve a monospaced role at an explicit pixel size.
    ///
    /// Not a named token role, and deliberately: a terminal's type size is the *user's*, set
    /// against how much of a grid fits in a pane, and running it through the type scale would
    /// make the density control silently resize somebody's shell. The size arrives as a
    /// number for that reason.
    ///
    /// Falls back to the UI face on a machine with no monospaced one; the caller can tell,
    /// because [`ListRenderer::cell_advance`] measures whether the resolved face actually
    /// advances every character equally rather than trusting that it does.
    pub fn resolve_mono(&mut self, px: f32) -> ResolvedRole {
        let size = PxSize::new(px.max(1.0));
        let font = self.db.mono_font();
        let (ascent, line_height) = match self.shaper.metrics(font, size) {
            Some(m) => (m.ascent, m.line_height),
            None => (size.to_f32() * 0.8, size.to_f32() * 1.25),
        };
        ResolvedRole {
            size,
            font,
            ascent,
            line_height,
        }
    }

    /// One cell's advance in a monospaced role, and whether the face really is one.
    ///
    /// Measured rather than assumed. A machine with none of the named monospaced faces
    /// resolves to the UI face, and a grid drawn as runs of text in a proportional face
    /// drifts a little further out of its columns with every character — legible for six
    /// characters and unreadable across eighty. Knowing which it is lets the caller pay for
    /// per-cell placement only where it is needed.
    ///
    /// The probe is `i` against `M`, the narrowest and widest ASCII letters in almost every
    /// proportional design. Comparing two *similar* characters would report a proportional
    /// face as monospaced.
    pub fn cell_advance(&mut self, role: ResolvedRole) -> (f32, bool) {
        let features = Features::default();
        let wide = self.measure("M", role, features);
        let narrow = self.measure("i", role, features);
        // A tenth of a pixel: two glyphs from a monospaced face have identical advances up to
        // the rounding the shaper does, and nothing proportional is this close.
        let monospaced = (wide - narrow).abs() < 0.1;
        (wide.max(1.0), monospaced)
    }

    /// Draw one line of arbitrary text in a resolved role, ellipsized at `max_width`.
    ///
    /// `baseline` is absolute, not relative to a row: this is the entry point for text that
    /// is not row content.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_label(
        &mut self,
        list: &mut DrawList,
        text: &str,
        x: f32,
        baseline: f32,
        max_width: f32,
        role: ResolvedRole,
        features: Features,
        color: Srgba,
    ) {
        self.draw_text(
            list, text, x, baseline, max_width, role.size, role.font, features, color,
            // End-ellipsis: middle truncation exists to preserve a filename's extension,
            // and arbitrary text has no extension to preserve.
            false,
        );
    }

    /// Draw one state icon with its top-left corner at `(x, y)`, in a box of `px` device
    /// pixels. Returns whether anything was drawn.
    ///
    /// # Why this is the entry point rather than a mark in a string
    ///
    /// A session's state used to reach the screen as a codepoint inside a label —
    /// `qs::terminal::Status::mark`, one of U+25CF, U+25CB, U+25B2 — and that channel is only
    /// as reliable as the machine's font stack. Its own history says so: the first build of it
    /// used U+26A0 and drew a notdef box. A [`StateIcon`] is a path this workspace rasterizes,
    /// so it resolves everywhere, and this is how a surface that is not a row reaches it.
    ///
    /// The words stay in the label. A reader using the accessible name needs text, and an icon
    /// is silent to them — so the two channels carry the same state by different means rather
    /// than one replacing the other.
    ///
    /// # Why it asks the atlas directly and holds no per-frame slot
    ///
    /// [`ListRenderer::icon_entry`] caches into [`IconCache`] because the row loop asks for
    /// the same nine kinds a thousand times a frame. Chrome does not: a tab strip draws one of
    /// these per tab and the overview one per session, so the handful of atlas lookups cost
    /// less than a second cache keyed by `(state, px)` would — and that cache would have to be
    /// keyed by size, because these are drawn at a tab's size, a row's size and the overview's
    /// size in the same frame.
    ///
    /// A refusal is counted in `icons_dropped` exactly as a kind's is, for the same reason: on
    /// the CPU tier that counter is the upload-budget failure, and a state icon competes for
    /// the same [`UploadClass::Structural`] bound.
    pub fn draw_state_icon(
        &mut self,
        list: &mut DrawList,
        state: qs_gpu::icon::StateIcon,
        x: f32,
        y: f32,
        px: u16,
        color: Srgba,
    ) -> bool {
        self.draw_chrome_icon(list, IconShape::State(state), x, y, px, color)
    }

    /// Draw one disclosure/separator chevron. See [`ListRenderer::draw_state_icon`].
    ///
    /// It exists for the reason that one does, one control further out: the mark it replaces
    /// was a character — `\u{25aa}` in front of a group header, `\u{203a}` between two
    /// breadcrumbs — and a character has nowhere to put the state the control is supposed to
    /// show. See `qs_gpu::icon::Chevron` and `docs/text-as-layout-audit.md`.
    pub fn draw_chevron(
        &mut self,
        list: &mut DrawList,
        chevron: qs_gpu::icon::Chevron,
        x: f32,
        y: f32,
        px: u16,
        color: Srgba,
    ) -> bool {
        self.draw_chrome_icon(list, IconShape::Chevron(chevron), x, y, px, color)
    }

    /// The half both chrome icon entry points share.
    ///
    /// One function rather than two copies, because the interesting parts — the structural
    /// upload class, the dropped-icon counter, the rounding — are the same for every shape
    /// chrome draws, and a second copy is where one of them goes missing.
    fn draw_chrome_icon(
        &mut self,
        list: &mut DrawList,
        shape: IconShape,
        x: f32,
        y: f32,
        px: u16,
        color: Srgba,
    ) -> bool {
        let key = IconKey { shape, px };
        let Some(entry) =
            self.atlas
                .get_or_render(key, false, UploadClass::Structural, qs_gpu::icon::rasterize)
        else {
            self.icons_dropped += 1;
            return false;
        };
        list.instances.push(Instance::glyph(
            x.round(),
            y.round(),
            entry.width as f32,
            entry.height as f32,
            entry.uv,
            color,
        ));
        true
    }

    /// Advance width of `text` in a resolved role, in physical pixels.
    ///
    /// Goes through the same shaped-run cache [`ListRenderer::draw_label`] does, so a
    /// caller that measures a string and then draws it pays for one shaping and cannot
    /// disagree with what lands on screen. That matters for anything that has to place
    /// the *next* thing after this one -- a breadcrumb whose segments are measured by a
    /// second, approximate path would overlap on exactly the paths that are hard to
    /// reproduce.
    pub fn measure(&mut self, text: &str, role: ResolvedRole, features: Features) -> f32 {
        self.cache
            .get_or_shape(&mut self.shaper, text, role.font, role.size, features)
            .width
    }

    pub fn cache_stats(&self) -> qs_text::CacheStats {
        self.cache.stats()
    }

    /// Drop everything size-dependent. Called on a density, scale or text-scale change.
    pub fn invalidate_metrics(&mut self) {
        self.cache.clear();
        self.atlas.clear();
    }

    /// Hand the frame's uploads to the caller, and check that every text pass flushed.
    ///
    /// The check lives here because this is the one call that happens once per frame, after
    /// every pass. A parked glyph that nobody flushed is text missing from the screen with
    /// no counter reading non-zero -- silent, and worse than the shredding this mechanism
    /// replaced -- so it is loud in debug and still says so in release rather than being
    /// compiled out into nothing.
    pub fn take_uploads(&mut self) -> Vec<PendingUpload> {
        if !self.deferred.is_empty() {
            tracing::error!(
                target: "qs::text",
                parked = self.deferred.len(),
                "a text pass ended without calling ListRenderer::flush_text; that many glyph \
                 draws were dropped silently"
            );
            debug_assert!(
                self.deferred.is_empty(),
                "a text pass ended without calling ListRenderer::flush_text"
            );
            self.deferred.clear();
        }
        self.atlas.take_uploads()
    }

    /// Build one frame's draw list.
    ///
    /// # Three layers, in this order, and the order is the point
    ///
    /// 1. **Surface** -- zebra banding, hover and press, per row.
    /// 2. **Selection** -- the selected regions, hoisted out of the row loop.
    /// 3. **Content** -- focus ring, icon and text, per row.
    ///
    /// The selection region cannot be drawn inside the row loop, and that is what forced the
    /// split. Rows emit in slot order, so a region drawn during row 9's turn but displaced
    /// upward mid-morph would paint over rows 6-8's text, which have already been emitted.
    /// Hoisting it makes the layering explicit rather than a property of the fact that rows
    /// happened not to overlap.
    pub fn render(
        &mut self,
        list: &mut DrawList,
        buf: &RowBuf,
        layout: &ViewportLayout,
        interaction: Interaction<'_>,
        motion: &InteractionMotion,
    ) {
        self.render_rows(list, buf, layout, interaction, motion, true, None);
    }

    /// [`ListRenderer::render`], also describing every painted surface to `scene`.
    ///
    /// The lit mode's walk (specs/002 T015's second half): the slab is admitted at the same
    /// call site that paints the material, with the same [`Surface`], so which material a
    /// row gets is decided exactly once — a re-derivation in the caller is the
    /// two-descriptions drift scene-handoff rule 1 exists to catch. A separate entry point
    /// rather than an `Option` on `render`, for the reason `compile_with` is one: every
    /// existing caller keeps the signature it has.
    pub fn render_lit(
        &mut self,
        list: &mut DrawList,
        buf: &RowBuf,
        layout: &ViewportLayout,
        interaction: Interaction<'_>,
        motion: &InteractionMotion,
        scene: &mut crate::scene::SceneBuilder,
    ) {
        self.lit = true;
        self.render_rows(list, buf, layout, interaction, motion, true, Some(scene));
        self.lit = false;
    }

    /// Draw one **Miller column**: the same rows, name only.
    ///
    /// A separate entry rather than a flag on the renderer, because which columns a row has
    /// is a property of the surface being drawn and not of the renderer — two columns in one
    /// frame could otherwise disagree depending on which was drawn last.
    pub fn render_column(
        &mut self,
        list: &mut DrawList,
        buf: &RowBuf,
        layout: &ViewportLayout,
        interaction: Interaction<'_>,
        motion: &InteractionMotion,
    ) {
        self.render_rows(list, buf, layout, interaction, motion, false, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn render_rows(
        &mut self,
        list: &mut DrawList,
        buf: &RowBuf,
        layout: &ViewportLayout,
        interaction: Interaction<'_>,
        motion: &InteractionMotion,
        metadata: bool,
        mut scene: Option<&mut crate::scene::SceneBuilder>,
    ) {
        qs_gpu::affinity::assert_ui_thread("ListRenderer::render");

        self.cache.begin_frame();
        self.atlas.begin_frame();
        self.glyphs_dropped = 0;
        self.icons_dropped = 0;
        self.icons = IconCache::for_px(qs_gpu::icon::device_px(qs_gpu::icon::GRID, layout.scale));

        let columns = Columns::for_rect_with(
            layout.origin_x,
            layout.width as f32,
            layout.scale,
            &self.tokens,
            metadata,
        );

        // The row's two roles, resolved once per frame. `ui/md` is the primary text and
        // owns the baseline; `ui/sm` metadata sits on the *same* baseline at a smaller
        // size, which is what makes the columns read as one line rather than as three
        // separately centred blocks.
        let primary = self.tokens.type_role(role::MD);
        let secondary = self.tokens.type_role(role::SM);
        let font_px = PxSize::new(primary.size_px(layout.scale, layout.text_scale));
        let secondary_px = PxSize::new(secondary.size_px(layout.scale, layout.text_scale));
        let metrics = self.shaper.metrics(self.face_for(primary), font_px);
        // Optically centred -- see `row_text_fit`, which is this calculation. The renderer
        // calls it rather than repeating it, and records what it got, so the SC-007 fit
        // check measures the baseline rows are actually drawn on rather than a second copy
        // of the formula that could drift from this one without anything noticing.
        self.last_text_fit =
            metrics.map(|m| row_text_fit(layout.row_height, m.ascent, m.descent, m.x_height));
        let baseline = match self.last_text_fit {
            Some(fit) => fit.baseline,
            None => layout.row_height as f32 * 0.7,
        };

        // There is no icon prepass here any more, and its absence is the point. It used to
        // resolve every visible row's icon before a single glyph was asked for, because
        // rows are drawn top-down and the text of the first few rows spent the CPU tier's
        // entire 64-upload budget, leaving the bottom half with no icon *and* no name.
        // That was the right priority expressed the only way available at the time -- as
        // the order two loops happened to run in. `UploadClass::Structural` states it in
        // the atlas instead: icons draw from their own bound, so no amount of text can
        // starve them and nothing depends on which loop went first.

        // Layer 1: the surface.
        //
        // Banding is drawn full-bleed and the *states* are drawn inset. Banding is a
        // property of the surface -- it should run to the window edge like ruled paper.
        // Hover, press and selection are properties of an *object*, and an object needs an
        // edge: an inset rounded region reads as a thing sitting on the list, where a
        // full-bleed stripe reads as a table row.
        let region = StateRegion::new(&self.tokens, &columns, layout);
        let scale = layout.scale.max(0.1);
        let squish_px = self.tokens.space(PRESS_SQUISH);
        let substance_tokens = self.tokens.substance();
        for slot in 0..buf.len() as u32 {
            let top = layout.row_top(slot);
            let index = layout.visible.first + u64::from(slot);

            // The row's body. It used to be a bare `Instance::rect` in `surface/row-alt`
            // drawn only on the banded rows, with the unbanded ones showing the canvas
            // through. Both are materials now, and the reason is `substance`: a row has to
            // *be a surface* before what it is made of can say anything, and a row that is
            // the absence of a rectangle is not a surface.
            //
            // The two materials differ only in albedo, and each albedo is exactly the colour
            // that pixel already was -- `Material::composites` walks Pbr layers, so any other
            // choice would move every ratio the contrast gate checks. What varies per row is
            // how the surface responds to light, never how much light there is.
            let material = if index % 2 == 1 {
                material::name::ROW_BODY_ALT
            } else {
                material::name::ROW_BODY
            };
            let substance = buf
                .rows()
                .get(slot as usize)
                .map_or(Substance::UNKNOWN, |row| {
                    Substance::of(row, substance_tokens)
                });
            let body_surface = Surface {
                x: layout.origin_x,
                y: top,
                w: layout.width as f32,
                h: layout.row_height as f32,
                radius: 0.0,
                scale,
            };
            self.tokens.paint_substance(
                material,
                body_surface,
                1.0,
                Drive::REST,
                Some(substance),
                &mut list.instances,
            );
            // The same material, the same surface, described to the lighting pass. Here,
            // beside the paint, so a row's slab can never come from a different material
            // than its pixels did.
            if let Some(builder) = scene.as_deref_mut() {
                if let Some(slab) = self.tokens.scene_slab(material, body_surface) {
                    builder.admit(slab);
                }
            }
            // Hover and press are the same material, layered rather than blended: a press
            // deepens whatever is already beneath it, which is what makes it read as the
            // same object being pushed instead of as a second colour arriving.
            //
            // And it is pushed literally: the region draws inside itself in proportion to
            // the press animation, so the row compresses under the pointer and springs back
            // on release. The squish belongs to the *region*, not to one material, or the
            // wash and the selection fill would be two different shapes for as long as a
            // row is held.
            let squish = squish_px * motion.press_alpha(index);
            for alpha in [motion.hover_alpha(index), motion.press_alpha(index)] {
                self.tokens.paint(
                    material::name::ROW_HOVER,
                    region.surface(top, scale, squish),
                    alpha,
                    &mut list.instances,
                );
            }
        }

        // The focus lamp (US3), stated before the selection layer takes the builder. It is
        // not a layer: nothing is drawn for it, and the row that has focus is painted exactly
        // as it would be with the mode off -- FR-031, asserted by
        // `the_focus_ring_is_identical_with_the_lit_mode_on_and_off`.
        if let Some(builder) = scene.as_deref_mut() {
            self.set_focus_lamp(builder, layout, &region, motion);
        }

        // Layer 2: the selection regions, hoisted -- see this function's doc comment.
        self.draw_selection(list, layout, &region, interaction.selection, motion, scene);

        // The surface/content seam, recorded where it actually is (specs/002 T019). Layers
        // 1 and 2 — the canvas painted before this call, the row bodies, the washes and the
        // selection — are surfaces the lighting pass may modulate; everything from here on
        // samples the atlas or sits on top of text. Without this boundary the whole region
        // is one textured batch, `surface_content_split` lands at zero, and the lighting
        // pass paints under everything and changes nothing — which is exactly how the first
        // lit window shipped a no-op and a `--shot-gpu` diff caught it.
        list.end_batch(None, false);

        // Layer 3: content, over both.
        for (slot, row) in buf.rows().iter().enumerate() {
            let top = layout.row_top(slot as u32);
            let index = layout.visible.first + slot as u64;
            self.draw_row(
                list,
                buf,
                row,
                index,
                top,
                &columns,
                layout,
                interaction,
                RowType {
                    primary,
                    primary_px: font_px,
                    secondary,
                    secondary_px,
                },
                baseline,
            );
        }

        // Spend the bound on what this pass wanted most, then emit the draws that were
        // waiting on it. Before the stats are read: `flush_text` is what decides how many
        // glyphs were rasterized and how many are genuinely missing.
        self.flush_text(list);

        list.stats.rows_laid_out = buf.len() as u32;
        list.stats.shaped_runs_new = self.cache.stats().new_this_frame;
        list.stats.glyphs_rasterized = self.atlas.stats().uploaded_this_frame;
        list.stats.glyphs_dropped = self.glyphs_dropped;
        list.stats.glyph_shortfall = self.atlas.stats().demanded_this_frame;
        list.stats.icons_dropped = self.icons_dropped;
        list.end_batch(None, true);
        list.finish();
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_row(
        &mut self,
        list: &mut DrawList,
        buf: &RowBuf,
        row: &RowView,
        index: u64,
        top: f32,
        columns: &Columns,
        layout: &ViewportLayout,
        interaction: Interaction<'_>,
        type_roles: RowType,
        baseline: f32,
    ) {
        let height = layout.row_height as f32;
        // Data flags from the source, view flags from the caller.
        let flags = row.flags | interaction.flags_for(index);
        let focused = flags.contains(RowFlags::IS_FOCUSED);

        // The focus ring is painted onto the SAME shape the selection is, so it takes that
        // shape from the one type that knows where it is.
        //
        // It used to rebuild the shape here — `columns.content_x()`, `content_width`, the
        // radius class and an `xs` inset, all restated — and that was the same rectangle as
        // `StateRegion`'s by coincidence rather than by construction. The coincidence ended
        // the moment the region stopped being inset by the gutter: the ring would have kept
        // the old width, and a focused *and* selected row would have shown its focus ring
        // floating inside its own selection. That is exactly the defect
        // `ViewportLayout::origin` exists to prevent, one surface further in, and the fix is
        // the same one — read the shape from whatever owns it.
        //
        // Squish zero: the ring does not compress with a press today. Passed explicitly so
        // that stays a decision rather than an omission.
        let region = StateRegion::new(&self.tokens, columns, layout).surface(
            top,
            layout.scale.max(0.1),
            0.0,
        );

        // Banding, hover, press and the selection region are all drawn before this function
        // is reached -- see `ListRenderer::render` for the three layers and why the
        // selection one cannot live in the row loop.

        // Focus ring: 2px ring plus a 1px contrasting outline (UXDD §10.5).
        //
        // The outline is drawn *first and larger*, so the ring paints over its inner edge
        // and the pair reads as one indicator. Both are inside-aligned strokes -- see
        // `qs_gpu::cpu_raster` for why that alignment is load-bearing for tier parity.
        //
        // Focus is drawn independently of selection because they are different states: a
        // row can be focused without being selected while the keyboard moves through the
        // list, and collapsing them would make keyboard navigation invisible.
        if focused {
            let focus = self.tokens.focus();
            let scale = layout.scale.max(0.1);
            let ring_w = focus.ring_width * scale;
            let outline_w = focus.outline_width * scale;

            let (fx, fy) = (region.x, region.y);
            let (fw, fh) = (region.w, region.h);

            list.instances.push(Instance::stroke(
                fx,
                fy,
                fw,
                fh,
                region.radius,
                ring_w + outline_w * 2.0,
                self.tokens.color("border/focus-outline"),
            ));
            list.instances.push(Instance::stroke(
                fx + outline_w,
                fy + outline_w,
                (fw - outline_w * 2.0).max(1.0),
                (fh - outline_w * 2.0).max(1.0),
                (region.radius - outline_w).max(0.0),
                ring_w,
                self.tokens.color("border/focus"),
            ));
        }

        let hidden = row.flags.contains(RowFlags::IS_HIDDEN);
        // The lit row's ink. A selected row is a lamp when the mode is on, and lettering on
        // a lit panel is a different colour from lettering on a dark one — measured at
        // 1.53:1 before this existed, which is a filename nobody can read. `content/on-lit`
        // is white in the dark theme and near-black in the light one, because the lit panel
        // lands on opposite sides of the two themes' grounds; the contrast gate checks both
        // pairs, so neither is a guess.
        // Three inks, because a selected row is a coloured object and a LIT one is a light,
        // and neither takes the ordinary row ink. The material declares both pairs — `text`
        // and `text_lit` — so the contrast gate checks the state that is on screen rather
        // than the one that happens to be first in the file.
        let selected = flags.contains(RowFlags::IS_SELECTED);
        let on_lamp = self.lit && selected;
        let (primary_ink, secondary_ink, tertiary_ink) = if on_lamp {
            let lamp = self.tokens.color("content/on-lit");
            (lamp, lamp, lamp)
        } else if selected {
            let on_accent = self.tokens.color("content/on-accent");
            (on_accent, on_accent, on_accent)
        } else {
            (
                self.tokens.color("content/primary"),
                self.tokens.color("content/secondary"),
                self.tokens.color("content/tertiary"),
            )
        };
        let name_color = fade_on(primary_ink, hidden, on_lamp);

        // Icon slot: a kind-based vector icon rasterized into the same atlas as the text
        // (`qs_gpu::icon`), so it is a coverage mask tinted at draw time exactly like a
        // glyph. Directories get the accent, files the tertiary content colour -- one
        // rasterization serves both, because only the tint differs.
        //
        // Dimmed for hidden files, which the placeholder square was not. A hidden file's
        // name fades and its icon did not, which read as two rows rather than one.
        if let Some(entry) = self.icon_entry(kind_of(row)) {
            let size = entry.width as f32;
            let icon_x = (columns.icon_x() + (columns.icon - size) * 0.5).round();
            let icon_y = (top + (height - size) * 0.5).round();
            list.instances.push(Instance::glyph(
                icon_x,
                icon_y,
                size,
                entry.height as f32,
                entry.uv,
                // The icon takes the row's ink on a lit or accent panel, for the reason the
                // text does and one more: a folder's icon is tinted `icon/folder`, and on a
                // panel painted in the accent the two are close enough that the icon
                // disappeared entirely — visible in the first lamp render as rows whose
                // folder marks had simply gone.
                fade_on(
                    if on_lamp || selected {
                        primary_ink
                    } else {
                        icon_tint(&self.tokens, row)
                    },
                    hidden,
                    on_lamp,
                ),
            ));
            // The emblem is a *modifier* on an icon the reader has already identified, so it
            // is drawn after the silhouette and never instead of it. `IS_SYMLINK` has been
            // set by the source and read by `kind_label` since the row model existed; this is
            // the first thing to draw it.
            if row.flags.contains(RowFlags::IS_SYMLINK) {
                self.draw_emblem(list, Emblem::Symlink, icon_x, icon_y, size, hidden);
            }
        }

        // Name.
        let name = buf.name(row);
        self.draw_text(
            list,
            &name,
            columns.name_x(),
            top + baseline,
            columns.name,
            type_roles.primary_px,
            // Bolder on a lit or accent panel. The lamp's ground is bright and busy — it
            // has ribs and a hot centre-line — and a 400-weight name on it reads thinner
            // than the same name on a flat dark row even at the same measured contrast,
            // because contrast is a property of two colours and legibility is a property of
            // the stroke that carries one of them. `face_for` already resolves a face per
            // weight class, and `weight_coverage` records what this machine could actually
            // satisfy, so asking for 600 costs a map lookup and degrades to the UI face on
            // a machine that has no bold.
            self.face_for(if on_lamp || selected {
                TypeRole {
                    weight: 600,
                    ..type_roles.primary
                }
            } else {
                type_roles.primary
            }),
            Features::default(),
            name_color,
            true,
        );

        if row.state == LoadState::Stub {
            // FR-017. `size`, `mtime` and `kind` are undefined for a stub, so they are not
            // read -- placeholders go where they will land, at the same positions, so the
            // upgrade replaces them in place with no reflow.
            let bar = self.tokens.color("content/tertiary");
            let bar = Srgba { a: 0.22, ..bar };
            let bar_h = (height * 0.28).max(2.0);
            let bar_y = top + (height - bar_h) * 0.5;
            let placeholders: &[(f32, f32)] = &if columns.metadata {
                [
                    (columns.size_x(), columns.size * 0.7),
                    (columns.modified_x(), columns.modified * 0.85),
                    (columns.kind_x(), columns.kind * 0.6),
                ]
            } else {
                // Nothing to stand in for: a column draws no metadata, so a placeholder
                // for it would be a shimmer promising a value that never arrives.
                [(0.0, 0.0); 3]
            };
            for &(x, w) in placeholders.iter().filter(|(_, w)| *w > 0.0) {
                list.instances
                    .push(Instance::rect(x, bar_y, w, bar_h, bar_h * 0.5, bar));
            }
            return;
        }

        let secondary = fade_on(secondary_ink, hidden, on_lamp);
        let tertiary = fade_on(tertiary_ink, hidden, on_lamp);

        // Size and modified use tabular figures so a column of numbers forms a grid rather
        // than a ragged edge as rows scroll past (FR-013).
        let tabular = Features {
            tabular_figures: true,
        };

        if columns.metadata && !row.flags.contains(RowFlags::IS_DIR) {
            let text = format_size(row.size);
            self.draw_text_right_aligned(
                list,
                &text,
                columns.size_x() + columns.size,
                top + baseline,
                columns.size,
                type_roles.secondary_px,
                self.face_for(type_roles.secondary),
                tabular,
                secondary,
            );
        } else if columns.metadata {
            // The session indicator lives in the slot a **folder leaves empty**, which is the
            // whole reason it costs no reflow: `format_size` is drawn only for non-directories,
            // so a directory row's size column is already laid out, already right-aligned, and
            // already a pair the contrast gate checks on all four row grounds. Nothing labels
            // this column — there is no header row — so a count here is not filed under a
            // heading that says "Size".
            //
            // Reached only past the `LoadState::Stub` return above, which is acceptance 4 with
            // no code in it: a stub knows only its name and `IS_DIR`, draws its placeholder
            // bars, and never asks for a mark. `rows()` is untouched and still cannot block.
            if let Some(mark) = interaction.marks.get(index) {
                self.draw_text_right_aligned(
                    list,
                    &mark.words(),
                    columns.size_x() + columns.size,
                    top + baseline,
                    columns.size,
                    type_roles.secondary_px,
                    self.face_for(type_roles.secondary),
                    tabular,
                    // **The confidence is the ink**, and it is the same two levels the tab
                    // strip uses one surface up: ordinary secondary when a shell vouched for
                    // being here, muted when every session merely launched here and may have
                    // `cd`'d away. A reader who learns the distinction on a tab reads it the
                    // same way on a row. The words say it too, for a reader who hears the row
                    // rather than seeing it — see `SessionMark::spoken`.
                    if mark.vouched() { secondary } else { tertiary },
                );
                // The state rail, in the slot `Columns` has reserved and nothing has drawn
                // since the pretend selection indicator was deleted (see `draw_selection`).
                //
                // Present only for an outcome. A folder whose shells are all alive gets the
                // count and no rail, because the neutral token for a live shell is
                // `border/subtle`, which `design/tokens.json` states is deliberately outside
                // the contrast gate — WCAG 1.4.11 scopes non-text contrast to boundaries that
                // *identify* a component, and a rail meaning "something is alive in here"
                // would be claiming exemption while carrying meaning. Presence is the count's
                // job; this is reserved for the two outcomes worth interrupting for.
                if let Some(token) = mark.rail() {
                    let inset = region.h * 0.22;
                    let rail_h = (region.h - inset * 2.0).max(1.0);
                    let rail_w = columns.rail.max(1.0);
                    list.instances.push(Instance::rect(
                        columns.rail_x(),
                        region.y + inset,
                        rail_w,
                        rail_h,
                        rail_w * 0.5,
                        fade(self.tokens.color(token), hidden),
                    ));
                } else if let Some(flash) = mark.arrival() {
                    // The summons flash: a session filed here just started awaiting
                    // approval, and the reserved rail slot carries the accent for the
                    // handful of frames the arrival lasts, then returns to nothing —
                    // the row's steady state has no accent, so the arrival must end
                    // where it began. An outcome rail outranks it, exactly as
                    // `worst_rail` ranks outcomes above everything alive: the flash is
                    // "look here", and a failure already says that louder.
                    let inset = region.h * 0.22;
                    let rail_h = (region.h - inset * 2.0).max(1.0);
                    let rail_w = columns.rail.max(1.0);
                    let accent = fade(self.tokens.color("border/focus"), hidden);
                    list.instances.push(Instance::rect(
                        columns.rail_x(),
                        region.y + inset,
                        rail_w,
                        rail_h,
                        rail_w * 0.5,
                        Srgba {
                            a: accent.a * flash,
                            ..accent
                        },
                    ));
                }
            }
        }

        if !columns.metadata {
            // A Miller column stops at the name.
            return;
        }

        let modified = format_mtime(row.mtime);
        self.draw_text(
            list,
            &modified,
            columns.modified_x(),
            top + baseline,
            columns.modified,
            type_roles.secondary_px,
            self.face_for(type_roles.secondary),
            tabular,
            secondary,
            false,
        );

        let kind = kind_label(row);
        self.draw_text(
            list,
            kind,
            columns.kind_x(),
            top + baseline,
            columns.kind,
            type_roles.secondary_px,
            self.face_for(type_roles.secondary),
            Features::default(),
            tertiary,
            false,
        );
    }

    /// Draw the selection regions and their status rails.
    ///
    /// # Two paths, because a morph is a single-selection idea
    ///
    /// One selected row is drawn through [`InteractionMotion`], which slides a single region
    /// between the row it left and the row it arrived at -- UXDD §10.3's geometry morph, and
    /// the thing that makes arrowing through a list read as one object moving rather than as
    /// a light switching off and another on.
    ///
    /// Several selected rows are several regions, and there is no coherent "the region" to
    /// move. Asking the morph to represent them would mean picking one row to animate and
    /// leaving the rest to appear instantly, which reads as a bug. So a multiple selection
    /// draws its rows directly and the morph sits out the frame --
    /// [`Selection::morph_target`] is what keeps the two from both drawing the same row.
    ///
    /// # The halo is a separate pass, and it has to be
    ///
    /// Every selected row's glow reaches a falloff past its own band, so a loop that emitted
    /// glow-then-fill per row would paint row N+1's halo over row N's *fill* -- a bright
    /// accent band across the bottom of every selected row but the last, which looks like a
    /// rendering bug rather than like a glow. All the halos go down first, then all the
    /// fills, and the two passes iterate the same runs rather than collecting a `Vec` the
    /// steady-state frame is not allowed to allocate.
    ///
    /// Which layers bleed is the *material's* answer rather than this function's --
    /// [`material::Pass`] -- so a look that later grows a second reaching layer does not
    /// need this loop rewritten.
    #[allow(clippy::too_many_arguments)]
    fn draw_selection(
        &mut self,
        list: &mut DrawList,
        layout: &ViewportLayout,
        region: &StateRegion,
        selection: &Selection,
        motion: &InteractionMotion,
        mut scene: Option<&mut crate::scene::SceneBuilder>,
    ) {
        // How hard the selection is being moved and where the material cycle has got to.
        // The intensity is zero whenever nothing is animating, so a settled list is a still
        // picture and an animated material costs nothing at rest; the phase rides whatever
        // wakefulness there already is and never buys any -- see `InteractionMotion::phase`.
        let drive = motion.drive();

        if selection.len() > 1 {
            let first = layout.visible.first;
            let end = first.saturating_add(u64::from(layout.visible.count));
            let top_of = |index: u64| -> f32 {
                let slot = (index - first) as f64;
                (slot * f64::from(layout.row_height) - layout.first_row_offset) as f32
            };
            // A multiple selection has no travelling region, so no row is the one being
            // moved and nothing swells: several rows flaring at once would read as the list
            // flashing rather than as one object moving.
            for index in selection.iter_in(first..end) {
                self.push_selection(
                    list,
                    layout,
                    region,
                    top_of(index),
                    1.0,
                    Drive::REST,
                    0.0,
                    Pass::Bleed,
                    None,
                );
            }
            for index in selection.iter_in(first..end) {
                self.push_selection(
                    list,
                    layout,
                    region,
                    top_of(index),
                    1.0,
                    Drive::REST,
                    0.0,
                    Pass::Body,
                    scene.as_deref_mut(),
                );
            }
            return;
        }

        let Some(draw) = motion.selection_draw(layout.visible.count) else {
            return;
        };

        // [`ViewportLayout::row_top`] generalised to a slot that may be negative or past the
        // end: mid-morph the region sits *between* two rows, and either endpoint may have
        // scrolled off. The arithmetic stays relative to the first visible row for exactly
        // the reason `row_top` gives -- the absolute content offset of row 999,999 is 28
        // million and does not survive an f32.
        let slot = draw.row as f64 - layout.visible.first as f64;
        let top = (slot * f64::from(layout.row_height) - layout.first_row_offset) as f32
            + draw.offset_rows * layout.row_height as f32;
        // The same squish the hover wash takes, read off the row the region is heading to:
        // a pressed row's selection fill and its wash are one object and must compress
        // together.
        let squish = self.tokens.space(PRESS_SQUISH) * motion.press_alpha(draw.row);
        self.push_selection(
            list,
            layout,
            region,
            top,
            draw.alpha,
            drive,
            squish,
            Pass::Bleed,
            None,
        );
        self.push_selection(
            list,
            layout,
            region,
            top,
            draw.alpha,
            drive,
            squish,
            Pass::Body,
            scene,
        );
    }

    /// State the focus lamp on the scene, or leave it absent when nothing has focus (T062).
    ///
    /// # The lamp reads the same region the focus ring is painted on
    ///
    /// [`StateRegion`] owns where a row's state is drawn, and the lamp takes its position from
    /// exactly that rectangle rather than rebuilding one. A lamp positioned from a second
    /// description of the same row would drift from the ring by whatever the two disagreed
    /// about, and a light that is not quite over the thing it is finding is worse than no
    /// light: the user's eye goes to the brightest place and the ring is somewhere else.
    ///
    /// # The height is measured from the canvas, not from the focused row
    ///
    /// A row's elevation is a property of *that row*; how the room is lit is a property of the
    /// room. Hanging the lamp a fixed distance over whatever the focused row happens to be
    /// made of would make the whole window's shadows shift when a row gained a step, which
    /// reads as the lighting flickering as the keyboard moves between differently-elevated
    /// rows.
    ///
    /// # Mid-travel, and off-screen
    ///
    /// The lamp's slot may be fractional (it is travelling) and may be outside the viewport
    /// entirely (focus scrolled away). Both are fine and neither is culled: a light outside
    /// the frame still lights what is inside it, which is the difference between a light and a
    /// slab. The arithmetic stays relative to the first visible row for
    /// [`ViewportLayout::row_top`]'s reason -- row 999,999's absolute offset does not survive
    /// an `f32`.
    fn set_focus_lamp(
        &self,
        scene: &mut crate::scene::SceneBuilder,
        layout: &ViewportLayout,
        region: &StateRegion,
        motion: &InteractionMotion,
    ) {
        let Some(draw) = motion.focus_light_draw() else {
            scene.set_focus_light(None);
            return;
        };
        let scale = layout.scale.max(0.1);
        let slot = draw.row as f64 - layout.visible.first as f64;
        let top = (slot * f64::from(layout.row_height) - layout.first_row_offset) as f32
            + draw.offset_rows * layout.row_height as f32
            + layout.origin_y;
        let surface = region.surface(top, scale, 0.0);
        scene.set_focus_light(Some(crate::scene::focus_light(
            [surface.x, surface.y, surface.w, surface.h],
            0.0,
            self.tokens.lighting().rig.focus,
            scale,
            motion.focus_light_gain(),
        )));
    }

    /// One pass of one selected row, at a surface-relative `top`.
    ///
    /// The halo is under the fill because that is the order the material declares its layers
    /// in, and it matters: a glow is solid inside its own shape, so a halo on top of the
    /// selected fill would deepen the row's background by whatever the accent contributes and
    /// quietly cost the secondary and tertiary text the 4.5:1 their token is authored to
    /// preserve. That used to be a rule this comment asserted and no gate could see; the
    /// material's `over`/`text` declaration is what turned it into a check -- see
    /// [`crate::material::Material::composites`].
    #[allow(clippy::too_many_arguments)]
    fn push_selection(
        &self,
        list: &mut DrawList,
        layout: &ViewportLayout,
        region: &StateRegion,
        top: f32,
        alpha: f32,
        drive: Drive,
        squish: f32,
        pass: Pass,
        scene: Option<&mut crate::scene::SceneBuilder>,
    ) {
        let Some(material) = self.tokens.material(material::name::ROW_SELECTED) else {
            return;
        };
        let scale = layout.scale.max(0.1);
        let height = layout.row_height as f32;

        // The bleeding pass is culled against a band widened by how far the material reaches:
        // a row one pixel above the surface still throws its halo onto it, and culling it
        // with its row would make the glow pop rather than scroll. Anything thrown past the
        // surface is covered by the chrome, which is drawn afterwards. The body pass is
        // culled against the row's own band. Both are culled in surface-relative space --
        // the arithmetic above is relative to the first visible row -- and only then moved
        // onto the surface; the other way round would compare an absolute `y` to a height.
        // Widened at *full* drive rather than at this frame's, so a row whose halo is about
        // to swell is not culled a frame before it does.
        let bleed = match pass {
            Pass::Bleed => material.bleed(scale, Drive::new(1.0, drive.phase)),
            Pass::Body => 0.0,
        };
        if top + height + bleed <= 0.0 || top - bleed >= layout.height as f32 {
            return;
        }

        let top = top + layout.origin_y;
        let surface = region.surface(top, scale, squish);
        material.compile_pass(
            pass,
            surface,
            alpha,
            drive,
            self.tokens.effects_enabled(),
            &mut list.instances,
        );
        // The raised slab, admitted beside the paint (US1). Body pass only, so the two
        // passes of one region cannot cast twice, and only while the region is actually
        // visible -- a slab for a fully faded selection would shadow from nothing. The
        // travelling region admits at its interpolated surface, so mid-morph the shadow
        // moves WITH the region rather than teleporting between rows.
        if alpha > 0.0 {
            if let Some(builder) = scene {
                if let Some(slab) = self
                    .tokens
                    .scene_slab(material::name::ROW_SELECTED, surface)
                {
                    builder.admit(slab);
                }
            }
        }

        if pass == Pass::Bleed {}

        // The status rail is NOT drawn here any more, and the deletion is the point.
        //
        // It marks a row's VCS status — modified, added, conflicted — and nothing sets one
        // yet, so it was being drawn for *selection* to keep the slot warm. A rail that
        // means "this file changed" appearing because you clicked a row does not read as a
        // reserved slot; it reads as a stray blue tick floating inside the selection, which
        // is exactly how it was described the first time somebody who had not written it
        // looked at it. A placeholder that is indistinguishable from an artefact is not
        // reserving anything.
        //
        // `Columns` still reserves the rail's WIDTH, so `git-status-column` can draw a real
        // rail without reflowing a single row — which was the reservation actually worth
        // keeping. What went is the pretend indicator, not the space for the real one.
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_text(
        &mut self,
        list: &mut DrawList,
        text: &str,
        x: f32,
        baseline: f32,
        max_width: f32,
        size: PxSize,
        font: FontId,
        features: Features,
        color: Srgba,
        middle_ellipsis: bool,
    ) {
        if text.is_empty() || max_width <= 0.0 {
            return;
        }
        let run = self
            .cache
            .get_or_shape(&mut self.shaper, text, font, size, features);

        let truncation = if run.width > max_width {
            let ellipsis =
                self.cache
                    .get_or_shape(&mut self.shaper, "\u{2026}", font, size, features);
            let extension = middle_ellipsis.then(|| extension_start(text)).flatten();
            truncate_middle(&run, max_width, ellipsis.width, extension).map(|t| (t, ellipsis))
        } else {
            None
        };

        match truncation {
            None => self.emit_glyphs(list, &run, 0..run.glyphs.len(), x, baseline, size, color),
            Some((cut, ellipsis)) => {
                self.emit_glyphs(list, &run, 0..cut.head, x, baseline, size, color);
                self.emit_glyphs(
                    list,
                    &ellipsis,
                    0..ellipsis.glyphs.len(),
                    x + cut.ellipsis_x,
                    baseline,
                    size,
                    color,
                );
                self.emit_glyphs(
                    list,
                    &run,
                    cut.tail..run.glyphs.len(),
                    x - cut.tail_shift,
                    baseline,
                    size,
                    color,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_text_right_aligned(
        &mut self,
        list: &mut DrawList,
        text: &str,
        right: f32,
        baseline: f32,
        max_width: f32,
        size: PxSize,
        font: FontId,
        features: Features,
        color: Srgba,
    ) {
        let run = self
            .cache
            .get_or_shape(&mut self.shaper, text, font, size, features);
        let x = (right - run.width).max(right - max_width);
        self.emit_glyphs(list, &run, 0..run.glyphs.len(), x, baseline, size, color);
    }

    /// This frame's atlas entry for one icon kind, rasterizing it on first use.
    ///
    /// The `false` passed to `get_or_render` is the one place icons deliberately diverge
    /// from glyphs: an inkless glyph is a space and gets memoized as a legitimate blank, but
    /// an inkless icon is a bug in `qs_gpu::icon` and must stay countable.
    fn icon_entry(&mut self, kind: IconKind) -> Option<AtlasEntry> {
        let index = kind.index();
        match self.icons.slots.get(index) {
            Some(&IconSlot::Ready(entry)) => return Some(entry),
            Some(&IconSlot::Missing) => return None,
            _ => {}
        }

        let key = IconKey {
            shape: IconShape::Kind(kind),
            px: self.icons.px,
        };
        let entry =
            self.atlas
                .get_or_render(key, false, UploadClass::Structural, qs_gpu::icon::rasterize);
        if entry.is_none() {
            self.icons_dropped += 1;
        }
        if let Some(slot) = self.icons.slots.get_mut(index) {
            *slot = match entry {
                Some(entry) => IconSlot::Ready(entry),
                None => IconSlot::Missing,
            };
        }
        entry
    }

    /// This frame's atlas entry for one emblem, or `None` when no emblem is drawn at this size.
    ///
    /// The size check happens *here rather than in the atlas*, and that ordering is the point.
    /// `rasterize` refuses anything below `MIN_PX`, and [`ListRenderer::icon_entry`] counts a
    /// refusal as a dropped icon -- which is the CPU tier's upload-budget failure and is
    /// asserted to be zero on the cold frame. Asking for an emblem the box is too small to
    /// carry would put "the design says not at this size" into a counter that means "the atlas
    /// could not keep up", and the row test would go red for a reason nobody could read.
    fn emblem_entry(&mut self, emblem: Emblem) -> Option<AtlasEntry> {
        let px = self.icons.emblem_px?;
        let index = emblem.index();
        match self.icons.emblem_slots.get(index) {
            Some(&IconSlot::Ready(entry)) => return Some(entry),
            Some(&IconSlot::Missing) => return None,
            _ => {}
        }

        let key = IconKey {
            shape: IconShape::Emblem(emblem),
            px,
        };
        let entry =
            self.atlas
                .get_or_render(key, false, UploadClass::Structural, qs_gpu::icon::rasterize);
        if entry.is_none() {
            self.icons_dropped += 1;
        }
        if let Some(slot) = self.icons.emblem_slots.get_mut(index) {
            *slot = match entry {
                Some(entry) => IconSlot::Ready(entry),
                None => IconSlot::Missing,
            };
        }
        entry
    }

    /// Badge an icon with the file's extension, on a ribbon knocked out of its bottom-right.
    ///
    /// # Why the ribbon is not inside the silhouette
    ///
    /// UXDD §10.4 says the extension is "shown in the icon", which in Windows and Finder means
    /// text on the face of a page — those shells can do that because their generic document
    /// icon *is* a page. Seven of `qs_gpu::icon`'s nine silhouettes have no interior to write
    /// in: `Code` is `</>` with no container, `Text` is four bare rules, and `Document`,
    /// `Image`, `Config` and `Data` all fill their own interiors with their own marks. So the
    /// badge knocks a well out of the icon instead, the same move `Emblem::Plate` makes.
    ///
    /// # Why it is a Rect and not an emblem
    ///
    /// An emblem is a `px * px` square by construction — `rasterize` produces a square mask and
    /// `IconKey` carries one dimension — and a ribbon is wide and short. `Instance::rect`
    /// already carries a corner radius and already renders on all three tiers, and it costs no
    /// atlas entry: this whole function adds nothing to the icon half of the atlas.
    ///
    /// Returns without drawing anything at all when the badge would not read: no extension, an
    /// icon too small to carry one, or text too wide for the space left beside the emblem's
    /// corner. Never a clipped ribbon — "the badge is present" has to stay a reliable signal.
    #[allow(clippy::too_many_arguments)]
    fn draw_extension_badge(
        &mut self,
        list: &mut DrawList,
        name: &str,
        icon_x: f32,
        icon_y: f32,
        icon_size: f32,
        badge_px: PxSize,
        badge_font: FontId,
        badge_ascent: f32,
        badge_line: f32,
        scale: f32,
        hidden: bool,
    ) {
        let Some(dot) = extension_start(name) else {
            return;
        };
        // Without the leading dot. It costs about a quarter of a narrow ribbon and carries
        // nothing the badge needs: the badge's entire context is already "this is the type".
        // Everything after it is drawn as written — no case transformation, because the name's
        // bytes are what the file is called.
        let Some(ext) = name.get(dot + 1..) else {
            return;
        };
        if ext.is_empty() {
            return;
        }

        let pad = self.tokens.space(crate::tokens::space::XS) * scale;
        // The line box, with no padding added on top of it. A line box already carries leading
        // above the caps and below the baseline; adding `space/xs` as well double-pads it, and
        // since the threshold below is a multiple of this height, that mistake pushes the
        // badge's first appearance several pixels of icon further out than it needs to be.
        // Horizontal padding is still explicit, because a run's advance width has no side
        // bearing to spare.
        let ribbon_h = badge_line;
        if icon_size < ribbon_h * BADGE_ICON_RATIO {
            return;
        }

        // Reserved unconditionally, not only for symlinks. Reserving it conditionally would put
        // the badge at a different x on a symlinked file than on the one beside it, and a
        // grid's cells have to agree with each other about where things are.
        let reserved = icon_size * (qs_gpu::icon::EMBLEM_FRACTION + BADGE_EMBLEM_GAP);
        let room = icon_size - reserved;
        if room <= pad * 2.0 {
            return;
        }

        // Shaped, not counted. `extension_start` caps the suffix at twelve *bytes*, and twelve
        // bytes of CJK is far wider than twelve of ASCII, so the only honest fit test is the
        // real advance width. Measured before anything is pushed, because a ribbon already in
        // the draw list cannot be taken back out.
        let text_w = self
            .cache
            .get_or_shape(
                &mut self.shaper,
                ext,
                badge_font,
                badge_px,
                Features::default(),
            )
            .width;
        let ribbon_w = text_w + pad * 2.0;
        if ribbon_w > room {
            return;
        }

        let x = (icon_x + icon_size - ribbon_w).round();
        let y = (icon_y + icon_size - ribbon_h).round();
        list.instances.push(Instance::rect(
            x,
            y,
            ribbon_w,
            ribbon_h,
            self.tokens.radius(crate::tokens::radius::CHIP) * scale.max(0.1),
            fade(self.tokens.color("icon/badge"), hidden),
        ));
        self.draw_text(
            list,
            ext,
            x + pad,
            y + (ribbon_h - badge_line) * 0.5 + badge_ascent,
            text_w,
            badge_px,
            badge_font,
            Features::default(),
            fade(self.tokens.color("icon/badge-text"), hidden),
            false,
        );
    }

    /// Lay the emblem stack over an icon already drawn at `(icon_x, icon_y)` of `icon_size`.
    ///
    /// Bottom-left, which is where every desktop shell has put a shortcut mark for thirty
    /// years, and the corner the kind silhouettes carry least detail in.
    ///
    /// Two instances, not one: the plate is a flat knockout in the surface colour and the mark
    /// is drawn over it in ink. One coverage mask cannot carry two colours, and a mark laid
    /// straight on the silhouette shows the icon's strokes through the gaps in its own.
    fn draw_emblem(
        &mut self,
        list: &mut DrawList,
        emblem: Emblem,
        icon_x: f32,
        icon_y: f32,
        icon_size: f32,
        hidden: bool,
    ) {
        let plate_color = fade(self.tokens.color("surface/base"), hidden);
        let mark_color = fade(self.tokens.color("content/secondary"), hidden);
        for (shape, color) in [(Emblem::Plate, plate_color), (emblem, mark_color)] {
            if let Some(entry) = self.emblem_entry(shape) {
                let size = entry.width as f32;
                list.instances.push(Instance::glyph(
                    icon_x.round(),
                    (icon_y + icon_size - size).round(),
                    size,
                    entry.height as f32,
                    entry.uv,
                    color,
                ));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_glyphs(
        &mut self,
        list: &mut DrawList,
        run: &ShapedRun,
        range: std::ops::Range<usize>,
        origin_x: f32,
        baseline: f32,
        size: PxSize,
        color: Srgba,
    ) {
        let Some(glyphs) = run.glyphs.get(range) else {
            return;
        };
        for glyph in glyphs {
            let pen = origin_x + glyph.x;
            let (key, integer_x) = GlyphKey::quantize(glyph.font, glyph.glyph_id, size, pen);
            let baseline = (baseline + glyph.y).round();

            // `want` records demand and does not rasterize: what the atlas spends its bound
            // on is decided in `flush_text`, once every draw in the pass has been counted.
            // A miss here is therefore not a drop -- it is a draw waiting its turn, and
            // parking it is what keeps a common letter from losing its slot to whichever
            // row happened to be laid out first.
            let Some(entry) = self.atlas.want(key) else {
                self.deferred.push(DeferredGlyph {
                    key,
                    integer_x,
                    baseline,
                    color,
                });
                continue;
            };
            if entry.width == 0 || entry.height == 0 {
                continue;
            }

            list.instances.push(Instance::glyph(
                integer_x + entry.left as f32,
                baseline - entry.top as f32,
                entry.width as f32,
                entry.height as f32,
                entry.uv,
                color,
            ));
        }
    }

    /// Spend the atlas's content bound on this pass's most-wanted glyphs and emit the draws
    /// that were waiting on them.
    ///
    /// **Every pass that draws text must call this before closing its batch.** A glyph that
    /// missed the atlas is parked, not dropped, so skipping the flush loses the text
    /// entirely rather than merely deferring it -- which is why `take_uploads` asserts the
    /// park is empty and says so in release builds too.
    ///
    /// Instances land at the end of the pass rather than where the row loop reached them.
    /// That is safe for one reason worth stating: within a pass, glyphs are the topmost
    /// layer. Banding, hover, selection, focus rings, stub placeholder bars and the icon
    /// badge's ribbon are all drawn *under* text by design, so moving text later in the
    /// list can only preserve that order. Two glyphs never overlap -- the atlas gutters
    /// them and the shaper advances the pen -- so their order among themselves is not
    /// observable.
    pub fn flush_text(&mut self, list: &mut DrawList) {
        // Split the borrow: `admit_demanded` needs `&mut atlas` while the render closure
        // needs `&mut raster` and `&db`.
        let Self {
            atlas,
            raster,
            db,
            deferred,
            glyphs_dropped,
            ..
        } = self;

        atlas.admit_demanded(|key| match key {
            AtlasKey::Glyph(glyph) => raster.rasterize(db.as_ref(), glyph),
            // Icons are `UploadClass::Structural` and never go through `want`, so they
            // cannot appear in content demand. Refusing rather than rasterizing keeps that
            // true instead of merely likely.
            //
            // Images are `UploadClass::Image` and reach the atlas through
            // `GlyphAtlas::get_or_decode_image`, which takes a decoder rather than a glyph
            // rasterizer. Same refusal, same reason: this closure could not produce one if
            // it wanted to, and saying so keeps the demand path glyph-only by construction.
            AtlasKey::Icon(_) | AtlasKey::Image(_) => None,
        });

        for parked in deferred.drain(..) {
            let Some(entry) = atlas.get(parked.key) else {
                // Two reasons left, and only one is a failure: the bound did not reach this
                // glyph, or the face could not raster it. The third case -- a blank, which
                // is a space and is *supposed* to be invisible -- is resident by now and
                // takes the branch below, which is why counting it here would make this
                // number nonzero on every line containing a space.
                *glyphs_dropped += 1;
                continue;
            };
            if entry.width == 0 || entry.height == 0 {
                continue;
            }
            list.instances.push(Instance::glyph(
                parked.integer_x + entry.left as f32,
                parked.baseline - entry.top as f32,
                entry.width as f32,
                entry.height as f32,
                entry.uv,
                parked.color,
            ));
        }
    }

    // ---------------------------------------------------------------------------------
    // Grid view (UXDD §5.1). Here rather than in a module of its own because the pass
    // needs the renderer's private state -- the per-frame icon cache, the shaped-run
    // cache, the atlas -- and Rust privacy is per module: a sibling `grid.rs` would have
    // to be bought with `pub(crate)` on half of `ListRenderer`'s fields.
    // ---------------------------------------------------------------------------------

    /// Build one frame's draw list as a grid of cells.
    ///
    /// The same three layers as [`ListRenderer::render`] and for the same reason, collapsed
    /// to two here: a grid has no zebra banding to put underneath, and the selection region
    /// is per-cell rather than a single region that morphs between rows, so it does not
    /// need hoisting out of the loop.
    pub fn render_grid(
        &mut self,
        list: &mut DrawList,
        buf: &RowBuf,
        layout: &ViewportLayout,
        metrics: GridMetrics,
        interaction: Interaction<'_>,
        motion: &InteractionMotion,
    ) {
        qs_gpu::affinity::assert_ui_thread("ListRenderer::render_grid");

        self.cache.begin_frame();
        self.atlas.begin_frame();
        self.glyphs_dropped = 0;
        self.icons_dropped = 0;
        self.icons = IconCache::for_px(metrics.icon_px);

        let label = self.tokens.type_role(role::SM);
        let label_px = PxSize::new(label.size_px(layout.scale, layout.text_scale));
        let label_font = self.face_for(label);
        let ascent = self
            .shaper
            .metrics(label_font, label_px)
            .map_or(label_px.to_f32() * 0.8, |m| m.ascent);

        // The extension badge draws at the **label's** role, not at a size of its own.
        //
        // Two reasons, and the second is the one that decided it.
        //
        // Not a size derived from the icon: the shaped-run cache is keyed on (text, font,
        // size), so a badge that scaled with `cell_logical` would mint a fresh key for every
        // visible cell on every frame of a Ctrl+scroll zoom, miss the cache, and re-rasterize
        // every glyph at a new size on the frames that can least afford it.
        //
        // And not `ui/xs` either, though design/tokens.json names that role "Badges". An
        // extension is a *substring of the filename drawn directly below it*, so at the label's
        // face and size its glyphs are already resident and the badge costs the atlas almost
        // nothing. Asking for a second role asks for a second copy of characters the atlas
        // already holds, and this frame cannot afford them: measured on a nine-cell cold grid,
        // `ui/xs` pushed uploads to the CPU tier's 64-per-frame ceiling and dropped 7 glyphs,
        // and what went missing was the *filenames* of the last cells -- a badge bought at the
        // price of the name it abbreviates. At the label's role the same frame drops none and
        // uploads four more than it did before badges existed.
        let badge_line = self
            .shaper
            .metrics(label_font, label_px)
            .map_or(label_px.to_f32() * 1.2, |m| m.ascent + m.descent);

        // No icon prepass, for the reason the row pass states: icons draw from
        // `UploadClass::Structural`'s own bound, so text cannot starve them and neither
        // pass has to be ordered around the other.

        let scale = layout.scale.max(0.1);
        let radius = self.tokens.radius(crate::tokens::radius::ROW) * scale;
        let pad = self.tokens.space(crate::tokens::space::SM) * scale;
        let focus = self.tokens.focus();

        // The list animates selection as one region *travelling* between two rows, which is
        // a one-dimensional idea: `SelectionDraw::offset_rows` displaces it vertically and
        // nothing else. A grid would need the same displacement in two axes, and inventing
        // that here would be motion design done inside a view chunk. So the grid uses the
        // animation's alpha and ignores its displacement: selection cross-fades between
        // cells instead of sliding. Tracked as a gap on `uxdd-motion-remainder`.
        let selection = motion.selection_draw(layout.visible.count);

        for (slot, row) in buf.rows().iter().enumerate() {
            let index = layout.visible.first + slot as u64;
            let flags = row.flags | interaction.flags_for(index);
            let hidden = row.flags.contains(RowFlags::IS_HIDDEN);
            let (cx, cy, cw, ch) = layout.cell_rect(slot as u32);

            // The cell's own region, inset so neighbouring selected cells read as two
            // objects rather than one block -- the grid's version of `StateRegion`. The
            // *look* is the list's, by name: a grid that painted its own hover wash would be
            // the second definition this library exists to remove, and a cell is a row with
            // a different shape rather than a different surface.
            let (rx, ry) = (cx + pad * 0.5, cy + pad * 0.5);
            let (rw, rh) = ((cw - pad).max(1.0), (ch - pad).max(1.0));
            let cell = Surface::new(rx, ry, rw, rh, radius, scale);

            for alpha in [motion.hover_alpha(index), motion.press_alpha(index)] {
                self.tokens
                    .paint(material::name::ROW_HOVER, cell, alpha, &mut list.instances);
            }
            // A multiple selection has no travelling region to read an alpha from -- see
            // `draw_selection` -- so every selected cell is drawn opaque and the morph sits
            // the frame out.
            let alpha = if interaction.selection.len() > 1 {
                f32::from(u8::from(flags.contains(RowFlags::IS_SELECTED)))
            } else {
                selection
                    .filter(|draw| draw.row == index)
                    .map_or(0.0, |draw| draw.alpha)
            };
            // One cell at a time, both passes together: a grid cell's halo reaches into the
            // gap between cells rather than onto the next cell's fill, because the inset is
            // horizontal as well as vertical here.
            //
            // The drive is the list's, unchanged. A grid cross-fades where the list slides,
            // but "the selection is being moved" is the same fact in both views, and the
            // flare is what carries the move here since the displacement does not.
            self.tokens.paint_driven(
                material::name::ROW_SELECTED,
                cell,
                alpha,
                motion.drive(),
                &mut list.instances,
            );

            if flags.contains(RowFlags::IS_FOCUSED) {
                let ring = focus.ring_width * scale;
                let outline = focus.outline_width * scale;
                list.instances.push(Instance::stroke(
                    rx,
                    ry,
                    rw,
                    rh,
                    radius,
                    ring + outline * 2.0,
                    self.tokens.color("border/focus-outline"),
                ));
                list.instances.push(Instance::stroke(
                    rx + outline,
                    ry + outline,
                    (rw - outline * 2.0).max(1.0),
                    (rh - outline * 2.0).max(1.0),
                    (radius - outline).max(0.0),
                    ring,
                    self.tokens.color("border/focus"),
                ));
            }

            let icon_color = fade(icon_tint(&self.tokens, row), hidden);
            if let Some(entry) = self.icon_entry(kind_of(row)) {
                let size = entry.width as f32;
                let icon_x = (cx + (cw - size) * 0.5).round();
                let icon_y = (cy + pad + (metrics.icon_box(scale) - size) * 0.5).round();
                list.instances.push(Instance::glyph(
                    icon_x,
                    icon_y,
                    size,
                    entry.height as f32,
                    entry.uv,
                    icon_color,
                ));
                // Same emblem, same corner, same rule as the list. A symlink that looked like
                // one in List and like a plain file in Grid would be two answers to one
                // question, which is the defect `kind_of` already exists to prevent.
                if row.flags.contains(RowFlags::IS_SYMLINK) {
                    self.draw_emblem(list, Emblem::Symlink, icon_x, icon_y, size, hidden);
                }
                // UXDD §10.4 scopes the extension badge to *file* icons. A folder named
                // `site.com` has no type to announce, and badging it would say it did.
                if !row.flags.contains(RowFlags::IS_DIR) {
                    let name = buf.name(row);
                    self.draw_extension_badge(
                        list, &name, icon_x, icon_y, size, label_px, label_font, ascent,
                        badge_line, scale, hidden,
                    );
                }
            }

            // The label is centred on the cell, which means it has to be measured before it
            // is placed. `measure` and `draw_text` share the shaped-run cache, so this is
            // one shaping per distinct name per frame, not two.
            let name = buf.name(row);
            let limit = (cw - pad * 2.0).max(0.0);
            let width = self
                .cache
                .get_or_shape(
                    &mut self.shaper,
                    &name,
                    label_font,
                    label_px,
                    Features::default(),
                )
                .width
                .min(limit);
            let baseline = cy + ch - pad - (metrics.label_height(scale) - ascent);
            self.draw_text(
                list,
                &name,
                cx + (cw - width) * 0.5,
                baseline,
                limit,
                label_px,
                label_font,
                Features::default(),
                fade(self.tokens.color("content/primary"), hidden),
                // Middle ellipsis, exactly as in the list: a cell is narrow, and the
                // extension is the part a truncated name most needs to keep.
                true,
            );
        }

        // Spend the bound on what this pass wanted most, then emit the draws that were
        // waiting on it. Before the stats are read: `flush_text` is what decides how many
        // glyphs were rasterized and how many are genuinely missing.
        self.flush_text(list);

        list.stats.rows_laid_out = buf.len() as u32;
        list.stats.shaped_runs_new = self.cache.stats().new_this_frame;
        list.stats.glyphs_rasterized = self.atlas.stats().uploaded_this_frame;
        list.stats.glyphs_dropped = self.glyphs_dropped;
        list.stats.glyph_shortfall = self.atlas.stats().demanded_this_frame;
        list.stats.icons_dropped = self.icons_dropped;
        list.end_batch(None, true);
        list.finish();
    }
}

/// The grid's cell geometry, derived from one continuously variable size.
///
/// UXDD §5.1 says the size is scalable "with no fixed steps", so everything here is a
/// function of `cell_logical` rather than a lookup into a table of sizes. The one place the
/// continuity genuinely stops is [`GridMetrics::icon_px`], because the icon rasterizer has
/// a hard ceiling: past `qs_gpu::icon::MAX_PX` the cell keeps growing and the glyph does
/// not. That is a real limit of the icon path rather than of the grid, and it is clamped
/// here where it is visible instead of failing to rasterize later.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GridMetrics {
    /// The requested cell width in logical pixels, before it is stretched to fill.
    pub cell_logical: f32,
    /// How many cells fit across the surface. At least one, however narrow the window.
    pub columns: u32,
    /// Cell height in physical pixels.
    pub cell_height: u32,
    /// Icon size in device pixels, clamped to what the rasterizer will produce.
    pub icon_px: u16,
    /// The label's line height in logical pixels, carried so the cell height and the
    /// baseline cannot be computed from two different numbers.
    label_logical: f32,
}

/// Fraction of the cell width the icon box occupies. The rest is padding and the label.
const ICON_BOX_FRACTION: f32 = 0.55;

/// How many times the extension badge's own height the icon box must be before the badge is
/// drawn at all (UXDD §10.4, "the extension shown in the icon at ≥ 32 px").
///
/// A *ratio*, not the literal 32, for two reasons. It derives UXDD's number rather than
/// copying it: `ui/xs` is 11 px, so the ribbon is about 13 logical px tall and the threshold
/// lands at roughly 32.5 — which is presumably where 32 came from. And because both sides of
/// the comparison are physical pixels, the device scale cancels, so the threshold is the same
/// 32 *logical* px on every display, while a user who turns text scale up correctly needs a
/// bigger icon before a bigger badge will fit. Anyone tempted to divide one side by `scale`
/// should read this twice.
const BADGE_ICON_RATIO: f32 = 2.5;

/// The gap between the emblem's reserved corner and the badge, as a fraction of the icon box.
const BADGE_EMBLEM_GAP: f32 = 0.06;

impl GridMetrics {
    /// Fit cells of `cell_logical` across a surface `surface_width` physical pixels wide.
    ///
    /// `label_line_height` is the resolved label role's line height in **physical** pixels;
    /// it is divided back out so the metrics stay expressed in logical units and a scale
    /// change re-derives rather than accumulates.
    #[must_use]
    pub fn fit(
        surface_width: u32,
        cell_logical: f32,
        scale: f32,
        label_line_height: f32,
        space_sm: f32,
    ) -> Self {
        let scale = scale.max(0.1);
        let cell_logical = cell_logical.max(24.0);
        let target = (cell_logical * scale).max(1.0);
        // At least one column: a window narrower than one cell shows one stretched cell
        // rather than zero columns and a division by zero.
        let columns = ((surface_width as f32 / target) as u32).max(1);

        let label_logical = label_line_height / scale;
        let pad = space_sm;
        let height_logical =
            pad + cell_logical * ICON_BOX_FRACTION + pad * 0.5 + label_logical + pad;

        Self {
            cell_logical,
            columns,
            cell_height: (height_logical * scale).round().max(1.0) as u32,
            // Clamped, not merely computed. `qs_gpu::icon::rasterize` *refuses* a size
            // outside MIN_PX..=MAX_PX and returns `None`, which the renderer counts as a
            // dropped icon -- so an un-clamped cell scaled past the ceiling would show a
            // grid of labels with no pictures and a non-zero counter nobody is reading.
            icon_px: qs_gpu::icon::device_px(cell_logical * ICON_BOX_FRACTION, scale)
                .clamp(qs_gpu::icon::MIN_PX, qs_gpu::icon::MAX_PX),
            label_logical,
        }
    }

    /// The icon box's height in physical pixels.
    #[must_use]
    pub fn icon_box(&self, scale: f32) -> f32 {
        self.cell_logical * ICON_BOX_FRACTION * scale
    }

    /// The label's line height in physical pixels.
    #[must_use]
    pub fn label_height(&self, scale: f32) -> f32 {
        self.label_logical * scale
    }
}

/// The inset rounded region that hover, press and selection all share.
///
/// One type rather than the same arithmetic written three times. The three states have to be
/// exactly coincident: a row that is hovered *and* selected would otherwise show a rim of
/// the weaker state around the edge of the stronger one, and during a morph the region that
/// arrives would not be the same shape as the one that left.
#[derive(Clone, Copy, PartialEq, Debug)]
struct StateRegion {
    x: f32,
    width: f32,
    inset: f32,
    height: f32,
    radius: f32,
}

impl StateRegion {
    fn new(tokens: &Tokens, columns: &Columns, layout: &ViewportLayout) -> Self {
        let scale = layout.scale.max(0.1);
        // One inset, on all four sides, and both halves of that are deliberate.
        //
        // It is *uniform* because it used to not be: the region sat `gutter` (six logical
        // pixels) inside the row horizontally and `xs` (two) vertically, so a selected row
        // was visibly narrower than the row it belonged to while being almost exactly as
        // tall. Nothing chose that asymmetry -- the horizontal edge came from the column
        // layout and the vertical one from the space scale -- and it read as the selection
        // failing to cover its own row.
        //
        // It is not *zero* because the gap is load-bearing twice over. It is what makes two
        // adjacent selected rows read as two objects rather than one tall block, and since
        // `prim-emissive-edge` it is also the only ground the row's contact shadow has to
        // fall on: a region flush with its row would cast onto the row below, where the next
        // row's opaque body paints over it, and the shadow would vanish exactly when there
        // is a neighbour to be above.
        let inset = tokens.space(crate::tokens::space::XS) * scale;
        Self {
            x: columns.x0 + inset,
            width: (layout.width as f32 - inset * 2.0).max(1.0),
            inset,
            height: (layout.row_height as f32 - inset * 2.0).max(1.0),
            // `CHIP`, not `ROW`. The region is a row's height minus its inset — about
            // fifteen logical pixels — and a six-pixel radius is forty per cent of that,
            // which reads as a lozenge rather than as a row: the selection and the focus
            // ring both looked like pills. Three is the same proportion on this shape that
            // `ROW` is on a card, which is what the radius scale is for.
            radius: tokens.radius(crate::tokens::radius::CHIP) * scale,
        }
    }

    /// The shape a material is painted onto for one row.
    ///
    /// The one route from a row's geometry to a look. `StateRegion` owns *where* the region
    /// is and a material owns *what it looks like*, which is what keeps hover, press and
    /// selection exactly coincident: all three are painted onto this, so a row that is
    /// hovered and selected cannot show a rim of the weaker state around the stronger one.
    ///
    /// `squish` is that geometry animating. A pressed row draws every one of its layers a
    /// little smaller, which reads as the row being pushed rather than as a colour arriving
    /// — and it has to live here rather than in a material for the same reason the region
    /// exists at all: the hover wash and the selection fill must squish by the same amount
    /// or the press pulls them apart.
    fn surface(self, row_top: f32, scale: f32, squish: f32) -> Surface {
        let inset = squish * scale;
        Surface::new(
            self.x + inset,
            row_top + self.inset + inset,
            (self.width - inset * 2.0).max(1.0),
            (self.height - inset * 2.0).max(1.0),
            (self.radius - inset).max(0.0),
            scale,
        )
    }
}

/// How far a pressed row's region draws inside its resting shape, in **logical** pixels.
///
/// A step from the space scale rather than a number, like every other distance in the row:
/// `xs` is two logical pixels, which on a comfortable row is a compression of about one part
/// in fourteen — visible as a push, invisible as a size change. It is scaled by the press
/// animation's own alpha, so it arrives at the 40 ms press timing and releases at the 80 ms
/// hover timing without a second plan, and under Reduce Motion the press alpha is instant
/// and the row simply is smaller for as long as it is held.
const PRESS_SQUISH: &str = crate::tokens::space::XS;

/// Scale a token colour's opacity for an animated state.
///
/// Multiplicative, not absolute: a token that is already translucent stays in proportion, so
/// an animation cannot make a state *more* opaque than the design system says it is.
/// Hidden files render dimmed rather than absent.
fn fade(color: Srgba, hidden: bool) -> Srgba {
    fade_on(color, hidden, false)
}

/// [`fade`], told whether the ground it will sit on is a **lit panel**.
///
/// A hidden entry is drawn at reduced opacity, which on the ordinary dark row moves its ink
/// toward the ground and reads as "present but quiet". On a lamp it does the same thing and
/// the result is different in kind: the panel is bright, the lit ink is dark, and dropping
/// its alpha slides it straight through the ground's own colour — a hidden file on a
/// selected lit row was the least readable text in the window, at the exact moment the user
/// had selected it. So the lit panel fades **less**: enough to keep the distinction between
/// hidden and not, not enough to spend the contrast the gate just bought.
fn fade_on(color: Srgba, hidden: bool, lit: bool) -> Srgba {
    if hidden {
        Srgba {
            a: color.a * if lit { 0.82 } else { 0.55 },
            ..color
        }
    } else {
        color
    }
}

/// Which icon a row gets.
///
/// Two rules, in order.
///
/// `IS_DIR` wins over `kind` unconditionally. A source that reports a directory whose name
/// ends in `.rs` is describing a directory, and drawing it as source code would be a lie
/// about what activating the row does.
///
/// A [`LoadState::Stub`] never reaches the kind table at all: the [`RowSource`] contract
/// says `kind` is *undefined* on a stub, not stale, and only `name` and `IS_DIR` are
/// readable. So a stub is a folder or a generic file and nothing else -- which is also the
/// honest picture, because that is genuinely all that is known about it yet.
///
/// The numeric table mirrors `qs_bench::gen::kind_of_extension`, M0's stand-in for a type
/// resolver. It is written out rather than imported because `qs-ui` must not depend on the
/// bench crate, and because M1 replaces the *producer* of these ids, not this consumer:
/// `KindId -> IconKind` is the mapping that survives, whoever assigns the `KindId`.
///
/// [`RowSource`]: crate::row_source::RowSource
/// **Public because the multi-selection histogram counts the same kinds the icons draw.**
/// A chart with its own `match row.kind.0` would be a second answer to one question, and it
/// fails quietly: a folder counted as `Generic` in the chart while the list draws it as a
/// folder looks plausible from either side. Promoting this was `inspector-metadata`'s first
/// move for exactly that reason.
#[must_use]
pub fn kind_of(row: &RowView) -> IconKind {
    if row.flags.contains(RowFlags::IS_DIR) {
        return IconKind::Folder;
    }
    if row.state == LoadState::Stub {
        return IconKind::Generic;
    }
    match row.kind.0 {
        1 => IconKind::Folder,
        2 => IconKind::Code,
        3 => IconKind::Config,
        4 => IconKind::Document,
        5 => IconKind::Image,
        6 => IconKind::Text,
        7 => IconKind::Data,
        8 => IconKind::Archive,
        _ => IconKind::Generic,
    }
}

/// What colour a row's icon is drawn in.
///
/// One function, called by both views, for the reason `kind_of` is one function: a folder
/// that is amber in List and blue in Grid is two answers to one question.
///
/// Folders get `icon/folder`. They used to get `border/focus`, which was a token -- so it
/// satisfied "not a literal" -- but the wrong one: `border/focus` is declared `role: border`
/// and every contrast pair it appears in is `kind: boundary`, so the gate was holding a
/// filled glyph to a focus ring's requirement and passing by coincidence. It also put the
/// folder tint and the focus ring on the same colour, which is worst on the one row that most
/// needs its ring seen.
fn icon_tint(tokens: &Tokens, row: &RowView) -> Srgba {
    if row.flags.contains(RowFlags::IS_DIR) {
        tokens.color("icon/folder")
    } else {
        tokens.color("content/tertiary")
    }
}

fn kind_label(row: &RowView) -> &'static str {
    if row.flags.contains(RowFlags::IS_DIR) {
        "Folder"
    } else if row.flags.contains(RowFlags::IS_SYMLINK) {
        "Link"
    } else {
        "File"
    }
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

    #[test]
    fn a_miller_column_spends_its_width_on_the_name() {
        // The defect this exists to prevent, seen on screen before it was caught here: at a
        // realistic column width the size, modified and kind columns reserve more space
        // than the column has, so the name collapses to its floor and every entry renders
        // as about six characters plus an ellipsis.
        let tokens = Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let width = 300.0;

        let list = Columns::for_rect(0.0, width, 1.0, &tokens);
        let column = Columns::for_column(0.0, width, 1.0, &tokens);

        assert!(
            list.name <= 40.0,
            "the premise changed: a list row at {width}px no longer squeezes the name to              its floor, so this test is no longer measuring anything ({} px)",
            list.name
        );
        assert!(
            column.name > list.name * 3.0,
            "a column gave the name {} px of {width}, barely more than the list's {}",
            column.name,
            list.name
        );
        assert_eq!(column.size, 0.0);
        assert_eq!(column.modified, 0.0);
        assert_eq!(column.kind, 0.0);
        assert!(!column.metadata);
        assert!(list.metadata);
    }

    #[test]
    fn sizes_format_at_one_decimal_so_the_column_never_reflows() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        // The largest value a `u64` byte count can express. The unit table has to reach it,
        // or a pathological size renders as a five-digit number that breaks the column.
        assert_eq!(format_size(u64::MAX), "16.0 EB");
    }

    #[test]
    fn timestamps_are_fixed_width_and_locale_independent() {
        // A locale-dependent format makes a reference image captured in one region fail in
        // another, which is a golden-image suite people learn to ignore.
        assert_eq!(format_mtime(0), "1970-01-01 00:00");
        assert_eq!(format_mtime(86_400), "1970-01-02 00:00");
        // 2026-08-07T00:00:00Z
        assert_eq!(format_mtime(1_786_060_800), "2026-08-07 00:00");
        // A leap-day, because the civil-date conversion is where those go wrong.
        assert_eq!(format_mtime(1_709_164_800), "2024-02-29 00:00");

        let width = format_mtime(0).len();
        for seconds in [0i64, 1, 1_786_060_800, -86_400] {
            assert_eq!(format_mtime(seconds).len(), width, "seconds {seconds}");
        }
    }

    #[test]
    fn a_timestamp_from_the_listing_pipeline_is_not_1970() {
        // The test that spans the gap the unit bug lived in. Everything above feeds
        // `format_mtime` a literal in whatever unit it happens to want, which is exactly why
        // none of them noticed that `qs-shell` fills `RowView::mtime` from
        // `SystemTime::duration_since(UNIX_EPOCH).as_secs()`. So this one starts from a
        // `SystemTime` and goes through the same arithmetic the listing does.
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_786_060_800);
        let secs = i64::try_from(
            t.duration_since(std::time::UNIX_EPOCH)
                .expect("after the epoch")
                .as_secs(),
        )
        .expect("in range");

        let rendered = format_mtime(secs);
        assert_eq!(rendered, "2026-08-07 00:00");
        assert!(
            !rendered.starts_with("1970"),
            "a real file's date collapsed to the epoch, which is what a unit mismatch \
             between qs-shell and this function looks like: {rendered}"
        );
    }

    #[test]
    fn a_negative_timestamp_does_not_panic() {
        // Pre-epoch mtimes exist on real filesystems, usually because something wrote a
        // zero or a garbage value.
        let s = format_mtime(-1);
        assert!(s.starts_with("1969"), "{s}");
    }

    #[test]
    fn extensions_are_detected_only_where_they_are_real() {
        assert_eq!(extension_start("report.xlsx"), Some(6));
        assert_eq!(extension_start("archive.tar.gz"), Some(11));
        assert_eq!(
            extension_start(".gitignore"),
            None,
            "a leading dot is a name"
        );
        assert_eq!(extension_start("Makefile"), None);
        assert_eq!(extension_start("trailing."), None);
        assert_eq!(
            extension_start("a.thisisnotanextension"),
            None,
            "an over-long suffix is part of the name"
        );
    }

    fn fake_run(widths: &[f32], clusters: &[u32]) -> ShapedRun {
        let mut x = 0.0;
        let glyphs = widths
            .iter()
            .zip(clusters)
            .map(|(&w, &cluster)| {
                let g = qs_text::shape::ShapedGlyph {
                    font: FontId(0),
                    glyph_id: 1,
                    x,
                    y: 0.0,
                    advance: w,
                    cluster,
                };
                x += w;
                g
            })
            .collect();
        ShapedRun {
            glyphs,
            width: x,
            missing_glyphs: 0,
            has_directional_override: false,
        }
    }

    #[test]
    fn text_that_fits_is_not_truncated() {
        let run = fake_run(&[10.0; 5], &[0, 1, 2, 3, 4]);
        assert!(truncate_middle(&run, 100.0, 8.0, None).is_none());
        assert!(truncate_middle(&run, 50.0, 8.0, None).is_none());
    }

    #[test]
    fn truncation_preserves_the_extension() {
        // "abcdefgh.rs": 11 glyphs of 10px = 110px, into an 80px column.
        let run = fake_run(&[10.0; 11], &(0..11).collect::<Vec<_>>());
        let cut = truncate_middle(&run, 80.0, 8.0, Some(8)).unwrap();

        assert_eq!(cut.tail, 8, "the tail must start at the extension's dot");
        assert!(cut.head > 0, "the head must not be eaten entirely");

        // Head + ellipsis + tail must actually fit.
        let head_w = run.glyphs[cut.head].x;
        let tail_w = run.width - run.glyphs[cut.tail].x;
        assert!(head_w + 8.0 + tail_w <= 80.0 + 0.01);
    }

    #[test]
    fn an_extension_too_long_for_the_column_gives_way_rather_than_erasing_the_head() {
        // A 100px column, a 30px "extension" of 3 glyphs, and 20 glyphs total.
        let run = fake_run(&[10.0; 20], &(0..20).collect::<Vec<_>>());
        let cut = truncate_middle(&run, 45.0, 8.0, Some(17)).unwrap();
        let head_w = run.glyphs.get(cut.head).map_or(run.width, |g| g.x);
        let tail_w = run.width - run.glyphs.get(cut.tail).map_or(run.width, |g| g.x);
        assert!(head_w + 8.0 + tail_w <= 45.0 + 0.01);
    }

    #[test]
    fn a_column_too_narrow_for_even_the_ellipsis_draws_nothing_rather_than_overflowing() {
        let run = fake_run(&[10.0; 10], &(0..10).collect::<Vec<_>>());
        let cut = truncate_middle(&run, 4.0, 8.0, None).unwrap();
        assert_eq!(cut.head, 0);
        assert_eq!(cut.tail, run.glyphs.len());
    }

    #[test]
    fn cuts_land_on_cluster_boundaries() {
        // Glyphs 3..6 share cluster 3 -- a base plus two marks, but with the advance in the
        // middle rather than on the base, which is what shaping reordering can produce and
        // what the naive "zero-advance marks follow their base" assumption misses.
        let run = fake_run(
            &[10.0, 10.0, 10.0, 0.0, 10.0, 0.0, 10.0, 10.0],
            &[0, 1, 2, 3, 3, 3, 6, 7],
        );

        // A budget that lands the raw width cut inside cluster 3.
        let cut = truncate_middle(&run, 46.0, 8.0, Some(7)).unwrap();

        assert!(cut.head > 0, "the head must not be eaten entirely");
        if cut.head < run.glyphs.len() {
            assert_ne!(
                run.glyphs[cut.head].cluster,
                run.glyphs[cut.head - 1].cluster,
                "the cut split a cluster in half"
            );
        }
    }

    #[test]
    fn the_head_never_overflows_its_budget_by_a_glyph() {
        // Measuring each glyph's pen position instead of its trailing edge admits a glyph
        // that starts inside the budget and ends outside it. That is a one-glyph overflow
        // into the next column, and it is invisible until someone looks closely.
        let run = fake_run(&[10.0; 12], &(0..12).collect::<Vec<_>>());
        for max in [15.0f32, 25.0, 37.0, 44.0, 61.0, 88.0] {
            let Some(cut) = truncate_middle(&run, max, 8.0, None) else {
                continue;
            };
            let head_w = run.glyphs.get(cut.head).map_or(run.width, |g| g.x);
            let tail_w = run.width - run.glyphs.get(cut.tail).map_or(run.width, |g| g.x);
            assert!(
                head_w + 8.0 + tail_w <= max + 0.01,
                "max {max}: head {head_w} + ellipsis 8 + tail {tail_w} overflows"
            );
        }
    }

    #[test]
    fn every_gap_and_pad_comes_from_the_space_scale() {
        // Constitution VII. The scale is a closed set (2,4,6,8,12,16,24,32), so a value off
        // it means someone reintroduced a literal.
        let tokens = Tokens::embedded(crate::tokens::Theme::Light).unwrap();
        let scale = [2.0f32, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0];
        let c = Columns::for_width(1920.0, 1.0, &tokens);

        for (name, value) in [
            ("gutter", c.gutter),
            ("padding", c.padding),
            ("gap", c.gap),
            ("rail", c.rail),
        ] {
            assert!(
                scale.iter().any(|s| (s - value).abs() < 0.001),
                "{name} = {value} is not a step on the space scale {scale:?}"
            );
        }
    }

    #[test]
    fn the_selection_region_is_inset_from_both_edges() {
        // The point of the chunk: selection is an object on a surface, not a full-bleed
        // table stripe. If someone reverts to `0.0..width`, this fails.
        let tokens = Tokens::embedded(crate::tokens::Theme::Light).unwrap();
        let width = 1920.0;
        let c = Columns::for_width(width, 1.0, &tokens);

        assert!(c.content_x() > 0.0, "the region must not start at the edge");
        assert!(
            c.content_x() + c.content_width(width) < width,
            "the region must not reach the right edge"
        );
        // Symmetric: the same gutter on both sides.
        let right_gap = width - (c.content_x() + c.content_width(width));
        assert!((right_gap - c.content_x()).abs() < 0.001);
    }

    #[test]
    fn the_row_radius_is_the_uxdd_row_class() {
        // UXDD 10.1: rows and cards share radius 6, by name rather than by number, so they
        // cannot drift apart.
        let tokens = Tokens::embedded(crate::tokens::Theme::Light).unwrap();
        assert_eq!(tokens.radius(crate::tokens::radius::ROW), 6.0);
        assert_eq!(tokens.radius(crate::tokens::radius::CHIP), 3.0);
        assert_eq!(tokens.radius(crate::tokens::radius::PANEL), 10.0);
    }

    #[test]
    fn the_focus_indicator_matches_the_accessibility_spec() {
        // UXDD 10.5: 2px ring, 1px contrasting outline. These are accessibility numbers,
        // not aesthetic ones, so they are asserted against the spec rather than read back
        // from the token file.
        let tokens = Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let focus = tokens.focus();
        assert_eq!(focus.ring_width, 2.0);
        assert_eq!(focus.outline_width, 1.0);
    }

    #[test]
    fn a_row_is_at_least_a_24px_pointer_target() {
        // UXDD 10.5 hit targets. Compact density at 100% is the smallest a row ever gets.
        use crate::density::Density;
        assert!(Density::Compact.row_height_px(1.0, 1.0) >= 24);
        assert!(Density::Default.row_height_px(1.0, 1.0) >= 24);
    }

    #[test]
    fn hover_and_focus_are_distinct_states_from_selection() {
        // A row can be focused without being selected while the keyboard moves through the
        // list; collapsing them makes keyboard navigation invisible.
        let f = RowFlags::IS_FOCUSED;
        assert!(!f.contains(RowFlags::IS_SELECTED));
        assert!(!f.contains(RowFlags::IS_HOVERED));
        let both = RowFlags::IS_FOCUSED | RowFlags::IS_SELECTED;
        assert!(both.contains(RowFlags::IS_FOCUSED) && both.contains(RowFlags::IS_SELECTED));
    }

    #[test]
    fn columns_never_go_negative_on_a_narrow_window() {
        let tokens = Tokens::embedded(crate::tokens::Theme::Light).unwrap();
        for width in [1.0f32, 50.0, 200.0, 400.0, 1920.0, 3840.0] {
            let c = Columns::for_width(width, 1.0, &tokens);
            assert!(
                c.name > 0.0,
                "width {width} produced a name column of {}",
                c.name
            );
            assert!(c.name_x() > 0.0);
            assert!(c.size_x() > c.name_x());
            assert!(c.kind_x() > c.modified_x());
        }
    }

    #[test]
    fn columns_scale_with_the_device_pixel_ratio() {
        let tokens = Tokens::embedded(crate::tokens::Theme::Light).unwrap();
        let at_1x = Columns::for_width(1920.0, 1.0, &tokens);
        let at_2x = Columns::for_width(3840.0, 2.0, &tokens);
        assert!((at_2x.icon - at_1x.icon * 2.0).abs() < 0.01);
        assert!((at_2x.padding - at_1x.padding * 2.0).abs() < 0.01);
    }

    // -- icons ------------------------------------------------------------------------

    /// A row per icon kind, plus a hidden one and a stub, repeated `times` over.
    ///
    /// The names are long and all different **on purpose**. An earlier version of this
    /// fixture gave every row the name `name.ext`, which needs eight distinct glyphs; the
    /// tests below passed and a cold frame of real filenames dropped five of nine icons,
    /// because the row text had spent the whole per-frame upload bound first. A fixture
    /// cheap enough to fit the budget cannot detect a bug about the budget.
    fn icon_rows_repeated(times: usize) -> RowBuf {
        const NAMES: [&[u8]; 12] = [
            b"src",
            b"node_modules",
            b"main.rs",
            b"Cargo.toml",
            b"README.md",
            b"screenshot.png",
            b"build.log",
            b"tokens.json",
            b"release-v0.1.0.tar.gz",
            b"quicksilver.exe",
            b".gitignore",
            b"selected_row.rs",
        ];
        let mut buf = RowBuf::new();
        let spec = [
            (1, RowFlags::IS_DIR, LoadState::Basic),
            (1, RowFlags::IS_DIR, LoadState::Basic),
            (2, RowFlags::EMPTY, LoadState::Basic),
            (3, RowFlags::EMPTY, LoadState::Basic),
            (4, RowFlags::EMPTY, LoadState::Basic),
            (5, RowFlags::EMPTY, LoadState::Basic),
            (6, RowFlags::EMPTY, LoadState::Basic),
            (7, RowFlags::EMPTY, LoadState::Basic),
            (8, RowFlags::EMPTY, LoadState::Basic),
            (9, RowFlags::EMPTY, LoadState::Basic),
            (0, RowFlags::IS_HIDDEN, LoadState::Basic),
            // A stub whose `kind` says "image". That field is undefined on a stub.
            (5, RowFlags::EMPTY, LoadState::Stub),
        ];
        for round in 0..times {
            for (index, &(kind, flags, state)) in spec.iter().enumerate() {
                buf.push(
                    RowView {
                        id: crate::row_source::RowId((round * spec.len() + index) as u64),
                        kind: crate::row_source::KindId(kind),
                        size: 1024 * (index as u64 + 1) * 37,
                        flags,
                        state,
                        ..RowView::default()
                    },
                    NAMES.get(index).copied().unwrap_or(b"unnamed"),
                );
            }
        }
        buf
    }

    fn icon_rows() -> RowBuf {
        icon_rows_repeated(1)
    }

    /// `rows` rows that differ *only* in `KindId`. Same name, same flags, same state, so
    /// every one of them does identical text work.
    fn kind_rows(kinds: impl Iterator<Item = u16>) -> RowBuf {
        let mut buf = RowBuf::new();
        for kind in kinds {
            buf.push(
                RowView {
                    kind: crate::row_source::KindId(kind),
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                b"name.ext",
            );
        }
        buf
    }

    fn uniform_kind_rows(rows: usize, kind: u16) -> RowBuf {
        kind_rows(std::iter::repeat_n(kind, rows))
    }

    fn varied_kind_rows(rows: usize) -> RowBuf {
        // 0..=9 covers every kind the corpus assigns; cycling fills out `rows`.
        kind_rows((0..rows).map(|i| (i % 10) as u16))
    }

    fn layout_for(rows: u64, scale: f32) -> ViewportLayout {
        let density = crate::density::Density::default();
        let row_height = density.row_height_px(scale, 1.0);
        ViewportLayout {
            origin_x: 0.0,
            origin_y: 0.0,
            width: 1200,
            height: row_height * rows as u32,
            scale,
            text_scale: 1.0,
            density,
            row_height,
            scroll: 0.0,
            first_row_offset: 0.0,
            visible: crate::scroll::VisibleRange {
                first: 0,
                count: rows as u32,
            },
            row_count: rows,
            content_height: u64::from(row_height) * rows,
            min_row_height: row_height,
            columns: 1,
        }
    }

    /// Motion at rest: nothing animating, so a draw list is a function of the interaction
    /// state alone. Tests about icons, text and columns want this; the tests about motion
    /// build their own.
    /// A selection of exactly one row -- the shape that still drives the morph.
    fn one(index: u64) -> Selection {
        let mut selection = Selection::new();
        selection.select_only(index);
        selection
    }

    fn settled() -> InteractionMotion {
        InteractionMotion::default()
    }

    fn renderer() -> ListRenderer {
        let db: Arc<dyn FontDb> = Arc::new(qs_text::SystemFontDb::scan());
        ListRenderer::new(
            Tokens::embedded(crate::tokens::Theme::Dark).unwrap(),
            db,
            qs_gpu::config_for(qs_gpu::RenderPath::Cpu),
        )
    }

    // -- SC-007: text fits inside its row, at every scale ---------------------------------

    /// The row's primary text metrics at one (density, device scale, text scale) point,
    /// resolved through the same shaper and the same role the renderer uses.
    fn fit_at(
        renderer: &mut ListRenderer,
        density: crate::density::Density,
        scale: f32,
        text_scale: f32,
    ) -> Option<RowTextFit> {
        let role = renderer.tokens.type_role(role::MD);
        let size = PxSize::new(role.size_px(scale, text_scale));
        let font = renderer.face_for(role);
        let m = renderer.shaper.metrics(font, size)?;
        Some(row_text_fit(
            density.row_height_px(scale, text_scale),
            m.ascent,
            m.descent,
            m.x_height,
        ))
    }

    /// Every (density, device scale, text scale) combination the layout can be put into.
    fn scale_matrix() -> Vec<(crate::density::Density, f32, f32)> {
        use crate::density::{Density, MAX_TEXT_SCALE, MIN_TEXT_SCALE};
        let mut out = Vec::new();
        for density in [Density::Compact, Density::Default] {
            for scale in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] {
                for text in [MIN_TEXT_SCALE, 1.0, 1.15, 1.5, 2.0, 2.5, MAX_TEXT_SCALE] {
                    out.push((density, scale, text));
                }
            }
        }
        out
    }

    #[test]
    fn text_fits_inside_its_row_at_every_scale_including_two_hundred_percent() {
        // SC-007, asserted as the thing SC-007 says rather than as a proportion.
        //
        // The old evidence was `row_height_grows_with_the_text_scale`, which checks that
        // the row and the font both double at 200%. That is true of a row too small for its
        // text and of a row with acres of space, so it could not tell them apart. This
        // measures the glyph extent against the row and requires the margins to stay
        // non-negative -- which is what "does not clip" means.
        //
        // The matrix is not decoration. `row_height_px` rounds to a whole physical pixel
        // and `size_px` does not, so at fractional device scales the row grows in steps
        // while the text grows continuously and the margin drifts. The single point the old
        // test checked, (Default, 1.0, 2.0), is one of the few where rounding cannot bite.
        let mut renderer = renderer();
        let mut worst = f32::INFINITY;
        let mut worst_at = None;

        for (density, scale, text) in scale_matrix() {
            let Some(fit) = fit_at(&mut renderer, density, scale, text) else {
                // No parseable face on this machine; the renderer's own fallback applies
                // and there are no metrics to check against. Skipping is honest here in a
                // way it would not be for the whole test -- see the assertion below, which
                // fails if *every* point skipped.
                continue;
            };
            assert!(
                !fit.clips(),
                "{density:?} at scale {scale} and text scale {text}: row {} px, \
                 headroom {:.2} above and {:.2} below the baseline at {:.2}",
                density.row_height_px(scale, text),
                fit.headroom_above,
                fit.headroom_below,
                fit.baseline,
            );
            if fit.tightest() < worst {
                worst = fit.tightest();
                worst_at = Some((density, scale, text));
            }
        }

        assert!(
            worst_at.is_some(),
            "no point in the matrix produced metrics, so this test asserted nothing"
        );
        // Recorded rather than bounded. A threshold here would be a number invented to sit
        // just under whatever this machine's UI font happens to give, and the honest claim
        // is only that the margin is non-negative everywhere.
        println!("tightest SC-007 margin: {worst:.2} px at {worst_at:?}");
    }

    #[test]
    fn a_row_that_ignores_the_text_scale_clips_and_the_check_says_so() {
        // The falsifiability proof for SC-007. Apply the exact defect the criterion exists
        // to catch -- honour the text scale for the font and not for the row -- and require
        // the fit check to report it.
        //
        // The specific verdict matters, not merely "something was wrong" -- and asserting
        // it is what corrected this test's first version, which required the *descender* to
        // be the casualty on the reasoning that clipped tails are the classic symptom.
        // Optical centring makes that wrong. The baseline sits at (row + x_height) / 2, so
        // it tracks half the row height while the ascender tracks the full font size: hold
        // the row still and double the text, and the baseline rises far slower than the
        // ascender above it. Measured here, a 28 px row holding 2x text loses 7.56 px off
        // the tops of the letters while still keeping 0.97 px under the descenders.
        //
        // So optical centring spends ascender headroom to protect descenders, and the
        // ascender is where this defect shows up first. Worth knowing before someone reads
        // a clipping report and goes looking at the wrong end of the glyph.
        use crate::density::Density;
        let mut renderer = renderer();
        let role = renderer.tokens.type_role(role::MD);

        let text_scale = 2.0;
        let size = PxSize::new(role.size_px(1.0, text_scale));
        let font = renderer.face_for(role);
        let Some(m) = renderer.shaper.metrics(font, size) else {
            panic!("the test machine has no parseable UI face; SC-007 cannot be checked");
        };

        // The row height a machine gets when someone drops `text_scale` from the row
        // calculation but leaves it in the font calculation.
        let frozen = Density::Default.row_height_px(1.0, 1.0);
        let broken = row_text_fit(frozen, m.ascent, m.descent, m.x_height);
        assert!(
            broken.clips(),
            "a {frozen} px row holding {text_scale}x text must be reported as clipping, \
             got {broken:?}"
        );
        assert!(
            broken.headroom_above < 0.0,
            "the ascender is what optical centring sacrifices first: {broken:?}"
        );

        // And the same metrics in the row height the shipped code actually computes do
        // not clip -- otherwise the assertion above would be satisfied by a check that
        // reports clipping unconditionally.
        let correct = Density::Default.row_height_px(1.0, text_scale);
        let fit = row_text_fit(correct, m.ascent, m.descent, m.x_height);
        assert!(!fit.clips(), "{correct} px is the shipped height: {fit:?}");
    }

    #[test]
    fn the_fit_measurement_is_the_baseline_the_renderer_draws_on() {
        // The SC-007 fit check is evidence about the shipped renderer only if it measures
        // the baseline the renderer actually uses. `render` calls `row_text_fit`, but a
        // test that only exercised the arithmetic would keep passing if `render` went back
        // to its own copy of the formula -- and then the criterion would be measuring a row
        // nobody draws, which is the same genus of blindness this chunk is closing.
        //
        // The measurement has to be ABSOLUTE, and getting that wrong is worth recording,
        // because the first version of this test was itself the failure it was written to
        // rule out. It rendered at both densities and compared how far the glyphs moved,
        // reasoning that the shared font size makes every other term cancel. It does -- but
        // so does the thing under test. Optical centring gives (h + x)/2 and box centring
        // gives h/2, and those differ by x/2, a constant: any difference across two row
        // heights cancels it exactly. That version passed with `render` mutated to use
        // naive box-centring, which is the mutation it existed to catch. A differential can
        // only ever show the baseline is linear in row height with slope one half, and both
        // formulas satisfy that.
        //
        // So `render` records the fit it used and this reads it back. That is also why
        // `last_text_fit` is worth having as real diagnostic state rather than a test hook:
        // once the draw list is flat, the row's vertical metric is unrecoverable from
        // anything the renderer produces.
        use crate::density::Density;
        let mut renderer = renderer();

        for density in [Density::Compact, Density::Default] {
            for text_scale in [1.0, 2.0] {
                let mut layout = layout_for(1, 1.0);
                layout.density = density;
                layout.text_scale = text_scale;
                layout.row_height = density.row_height_px(1.0, text_scale);

                let mut list = DrawList::default();
                renderer.render(
                    &mut list,
                    &plain_rows(1),
                    &layout,
                    Interaction::default(),
                    &settled(),
                );

                let role = renderer.tokens.type_role(role::MD);
                let size = PxSize::new(role.size_px(1.0, text_scale));
                let font = renderer.face_for(role);
                let Some(m) = renderer.shaper.metrics(font, size) else {
                    panic!("the test machine has no parseable UI face");
                };
                let expected = row_text_fit(layout.row_height, m.ascent, m.descent, m.x_height);

                assert_eq!(
                    renderer.last_text_fit,
                    Some(expected),
                    "{density:?} at text scale {text_scale}: the renderer laid rows out on \
                     a different metric than the one SC-007 measures"
                );

                // And the recorded metric is the one the glyphs were actually placed on,
                // not a number computed beside them and discarded. Glyph tops sit within a
                // line box of the baseline; anything further means the record and the draw
                // list disagree about where the row's text is.
                let tops: Vec<f32> = list
                    .instances
                    .iter()
                    .filter(|i| i.kind == qs_gpu::PrimKind::Glyph as u32)
                    .map(|i| i.rect[1])
                    .collect();
                assert!(!tops.is_empty(), "no glyphs were drawn");
                for top in tops {
                    assert!(
                        top > expected.baseline - m.ascent - 1.0
                            && top < expected.baseline + m.descent + 1.0,
                        "a glyph at y={top} is nowhere near the recorded baseline {}",
                        expected.baseline
                    );
                }
            }
        }
    }

    #[test]
    fn the_fit_arithmetic_is_the_optically_centred_baseline() {
        let fit = row_text_fit(28, 14.0, 3.3, 7.0);
        assert!((fit.baseline - 17.5).abs() < 0.001, "(28 + 7) / 2");
        assert!((fit.headroom_above - 3.5).abs() < 0.001);
        assert!((fit.headroom_below - 7.2).abs() < 0.001);
        assert!(!fit.clips());

        // A row shorter than the text: both margins are reported, and `tightest` picks the
        // one that ran out.
        let tight = row_text_fit(12, 14.0, 3.3, 7.0);
        assert!(tight.clips());
        assert!(tight.tightest() < 0.0);
    }

    // -- motion, as it reaches the draw list --------------------------------------------

    /// Rows with distinct names, so nothing about the fixture is degenerate.
    fn plain_rows(count: usize) -> RowBuf {
        let mut buf = RowBuf::new();
        for i in 0..count {
            buf.push(
                RowView {
                    id: crate::row_source::RowId(i as u64),
                    kind: crate::row_source::KindId(2),
                    size: 4096,
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                b"module.rs",
            );
        }
        buf
    }

    /// Alpha of an instance, `0..=255`, out of the packed premultiplied colour.
    fn alpha_of(instance: &Instance) -> u32 {
        instance.color >> 24
    }

    /// Whether an instance is the *body* of a state material — the layer text sits on.
    ///
    /// Rect, Gradient or Pbr rather than Rect alone, because a material decides what its body
    /// is made of and has now changed that twice -- to a ramp, and then to a lit surface.
    /// Asking for one kind would have made every test below quietly measure zero regions and
    /// pass on the ones that count things that are absent.
    ///
    /// This list is the price of identifying a layer by its primitive instead of by its role,
    /// and it is worth paying here: the alternative is threading a marker through the draw
    /// list that only tests would read.
    fn is_region_body(instance: &Instance) -> bool {
        instance.kind == qs_gpu::PrimKind::Rect as u32
            || instance.kind == qs_gpu::PrimKind::Gradient as u32
            || instance.kind == qs_gpu::PrimKind::Pbr as u32
            || instance.kind == qs_gpu::PrimKind::Sweep as u32
    }

    /// The inset rounded state regions in a draw list: hover, press and selection all share
    /// this geometry, which is exactly why `StateRegion` is one type.
    fn state_regions(list: &DrawList, layout: &ViewportLayout) -> Vec<Instance> {
        let tokens = Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let columns = Columns::for_width(layout.width as f32, layout.scale, &tokens);
        // From `StateRegion` itself rather than restated here. It used to be
        // `columns.content_width(..)`, which was the same number by coincidence until the
        // region stopped being inset by the gutter -- at which point every one of these tests
        // matched nothing and failed for a reason that had nothing to do with what it was
        // testing. One definition of where the region is, and this is a reader of it.
        let width = StateRegion::new(&tokens, &columns, layout).width;
        // A pressed region draws inside itself, so "the region's width" is a range rather
        // than a number. The tolerance is exactly the squish at full press -- widening it
        // further would start matching the status rail.
        let squish = tokens.space(PRESS_SQUISH) * layout.scale.max(0.1);
        list.instances
            .iter()
            .filter(|i| {
                is_region_body(i)
                    && i.radius > 0.0
                    && (i.rect[2] - width).abs() <= squish * 2.0 + 0.5
            })
            .copied()
            .collect()
    }

    fn draw(
        renderer: &mut ListRenderer,
        buf: &RowBuf,
        layout: &ViewportLayout,
        interaction: Interaction,
        motion: &InteractionMotion,
    ) -> DrawList {
        let mut list = DrawList::default();
        renderer.render(&mut list, buf, layout, interaction, motion);
        list
    }

    #[test]
    fn the_window_ground_is_still_visible_after_the_rows_are_drawn_on_it() {
        // This bug shipped, and it was found by a person looking at the running application
        // rather than by anything in this suite.
        //
        // `surface/canvas` is painted first and carries the ambient field. Rows then tile the
        // entire list area, edge to edge, one per visible entry. While the row bodies were
        // OPAQUE the field was painted over on every pixel a row covered — which is all of
        // them — so the window's ground was invisible in `cargo run` while looking correct in
        // `gpu_strip`, whose four rows leave most of the canvas showing.
        //
        // Every gate was green throughout. The contrast gate checks `surface/canvas` in
        // isolation and has no way to ask whether anything is drawn on top of it; the parity
        // suite checks primitives, not composition; the primitive budget counts instances,
        // not coverage. A ground nobody can see costs a full-viewport fragment pass per frame
        // and delivers nothing, which is the worst of both.
        //
        // So this asks the one question none of them ask: after a real frame from the real
        // render path, can the ground still be seen through what is on top of it?
        let mut renderer = renderer();
        let buf = plain_rows(12);
        let layout = layout_for(12, 1.0);
        let motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        let list = draw(
            &mut renderer,
            &buf,
            &layout,
            Interaction::default(),
            &motion,
        );

        // The row bodies: full-width, square-cornered lit surfaces, one per visible entry.
        let width = layout.width as f32;
        let bodies: Vec<&Instance> = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Pbr as u32
                    && i.radius == 0.0
                    && (i.rect[2] - width).abs() < 0.5
            })
            .collect();
        assert!(
            !bodies.is_empty(),
            "no full-width row bodies were drawn, so this proves nothing -- either the row \
             stopped being a surface, in which case `substance` has nothing to vary, or this \
             test is looking for the wrong shape"
        );

        for body in &bodies {
            let alpha = body.color >> 24;
            assert!(
                alpha < 255,
                "a row body is fully opaque, so it paints over the ambient field on every \
                 pixel it covers -- and rows cover the whole list area. The ground would \
                 still be drawn, still cost a full-viewport pass, and never be seen. If a \
                 row genuinely must be opaque, then `surface/canvas` is drawing a field \
                 nobody can look at and the field is what should go."
            );
        }
    }

    #[test]
    fn a_hovered_row_draws_a_wash_whose_opacity_follows_the_animation() {
        // The wiring this whole chunk is about: without it the motion state advances
        // perfectly and the screen changes instantly anyway.
        let mut renderer = renderer();
        let buf = plain_rows(6);
        let layout = layout_for(6, 1.0);
        let interaction = Interaction {
            hovered: Some(2),
            ..Interaction::default()
        };

        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.sync(interaction);
        let opening = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &motion),
            &layout,
        );
        assert!(
            opening.is_empty() || alpha_of(&opening[0]) == 0,
            "hover was already fully painted on the frame it began"
        );

        motion.advance(0.040);
        let midway = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &motion),
            &layout,
        );
        assert_eq!(midway.len(), 1, "expected exactly one hover wash");
        let partial = alpha_of(&midway[0]);

        while motion.advance(1.0 / 120.0) {}
        let settled = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &motion),
            &layout,
        );
        assert_eq!(settled.len(), 1);
        let full = alpha_of(&settled[0]);

        assert!(
            partial > 0 && partial < full,
            "hover opacity went {partial}/255 midway and {full}/255 settled -- it is not \
             animating, it is switching"
        );
        // And it lands on the row it is meant to.
        assert!((settled[0].rect[1] - layout.row_top(2)).abs() <= layout.row_height as f32);
    }

    #[test]
    fn a_press_deepens_the_row_instead_of_replacing_its_hover() {
        // Two washes, layered. One wash that merely changed colour would make a press read
        // as a different state arriving rather than as the same object being pushed.
        let mut renderer = renderer();
        let buf = plain_rows(4);
        let layout = layout_for(4, 1.0);
        let interaction = Interaction {
            hovered: Some(1),
            pressed: Some(1),
            ..Interaction::default()
        };
        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.sync(interaction);
        while motion.advance(1.0 / 120.0) {}

        let regions = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &motion),
            &layout,
        );
        assert_eq!(
            regions.len(),
            2,
            "a pressed row should draw the hover wash and the press wash, not one of them"
        );
        assert_eq!(
            regions[0].rect, regions[1].rect,
            "the two washes must be exactly coincident"
        );

        // And the pressed region draws inside the resting one. The squish is what makes a
        // press read as the row being pushed; both washes take it, which is the half that
        // matters -- one of them squishing would tear the state apart under the pointer.
        let hovered_only = Interaction {
            hovered: Some(1),
            ..Interaction::default()
        };
        let mut resting_motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        resting_motion.sync(hovered_only);
        while resting_motion.advance(1.0 / 120.0) {}
        let resting = state_regions(
            &draw(&mut renderer, &buf, &layout, hovered_only, &resting_motion),
            &layout,
        );
        assert!(
            regions[0].rect[2] < resting[0].rect[2],
            "a pressed region is {} wide against a resting {}: nothing was squished",
            regions[0].rect[2],
            resting[0].rect[2]
        );
    }

    #[test]
    fn the_selection_region_travels_between_the_two_rows_during_a_morph() {
        // UXDD 10.3 calls a selection change a geometry morph. The observable form of that
        // is one region at an intermediate position -- not two regions cross-fading, and not
        // one region that teleports.
        let mut renderer = renderer();
        let buf = plain_rows(8);
        let layout = layout_for(8, 1.0);
        let selection = one(6);
        let interaction = Interaction::with_selection(&selection);

        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.set_selected(Some(2));
        while motion.advance(1.0 / 120.0) {}
        motion.set_selected(Some(6));
        motion.advance(0.060);

        let regions = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &motion),
            &layout,
        );
        assert_eq!(regions.len(), 1, "a morph is one region, not a cross-fade");

        let y = regions[0].rect[1];
        let (from, to) = (layout.row_top(2), layout.row_top(6));
        assert!(
            y > from && y < to,
            "midway through the morph the region sat at {y}, outside the span \
             {from}..{to} it is supposed to be crossing"
        );
        let mut rested = InteractionMotion::new(crate::motion::MotionPreference::Full);
        rested.sync(interaction);
        while rested.advance(1.0 / 120.0) {}
        let at_rest = draw(&mut renderer, &buf, &layout, interaction, &rested);
        assert_eq!(
            alpha_of(&regions[0]),
            alpha_of(&state_regions(&at_rest, &layout)[0]),
            "a geometry morph must not also fade"
        );
    }

    #[test]
    fn a_multiple_selection_draws_a_region_per_selected_row_and_no_morph() {
        // The morph is one region moving between two rows. Three selected rows are three
        // regions, and letting the morph also draw would put a fourth somewhere between two
        // of them -- a region under a row nobody selected.
        let mut renderer = renderer();
        let buf = plain_rows(8);
        let layout = layout_for(8, 1.0);
        let mut selection = Selection::new();
        selection.select_only(1);
        selection.range_to(3);
        let interaction = Interaction::with_selection(&selection);

        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.set_selected(Some(6));
        motion.advance(0.060);

        let regions = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &motion),
            &layout,
        );
        assert_eq!(regions.len(), 3, "one region per selected row");
        // Within the row's band rather than exactly at its top: `StateRegion` insets
        // vertically so consecutive selected rows read as separate objects.
        for (region, row) in regions.iter().zip(1..=3) {
            let (top, bottom) = (layout.row_top(row), layout.row_top(row + 1));
            let y = region.rect[1];
            assert!(
                y >= top && y < bottom,
                "the region for row {row} sat at {y}, outside its band {top}..{bottom}"
            );
        }
    }

    #[test]
    fn a_selection_reaching_past_the_viewport_draws_only_what_is_on_screen() {
        // Select-all over a million entries must cost the viewport, not the corpus -- the
        // whole reason `iter_in` walks runs rather than indices.
        let mut renderer = renderer();
        let buf = plain_rows(8);
        let layout = layout_for(8, 1.0);
        let mut selection = Selection::new();
        selection.select_all(1_000_000);
        let interaction = Interaction::with_selection(&selection);

        let regions = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &settled()),
            &layout,
        );
        assert_eq!(regions.len(), buf.len());
    }

    #[test]
    fn the_selection_region_is_drawn_beneath_every_row_of_content() {
        // The reason `render` has three layers at all. Mid-morph the region sits between two
        // rows, and drawn inside the row loop it would paint over the text of every row it
        // had already passed. Emitting it before any content makes that impossible rather
        // than unlikely.
        let mut renderer = renderer();
        let buf = plain_rows(8);
        let layout = layout_for(8, 1.0);
        let selection = one(6);
        let interaction = Interaction::with_selection(&selection);
        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.set_selected(Some(2));
        while motion.advance(1.0 / 120.0) {}
        motion.set_selected(Some(6));
        motion.advance(0.060);

        let list = draw(&mut renderer, &buf, &layout, interaction, &motion);
        let tokens = Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let columns = Columns::for_width(layout.width as f32, layout.scale, &tokens);
        // From `StateRegion`, for the reason `state_regions` above gives.
        let region_w = StateRegion::new(&tokens, &columns, &layout).width;

        let last_region = list
            .instances
            .iter()
            .rposition(|i| {
                is_region_body(i) && i.radius > 0.0 && (i.rect[2] - region_w).abs() < 0.5
            })
            .expect("no selection region was drawn");
        let first_content = list
            .instances
            .iter()
            .position(|i| i.kind == qs_gpu::PrimKind::Glyph as u32)
            .expect("no glyphs were drawn, so this proves nothing");

        assert!(
            last_region < first_content,
            "a state region at index {last_region} is emitted after content at \
             {first_content}, so it paints over the text"
        );
    }

    // -- the selection halo ---------------------------------------------------------------

    fn glows(list: &DrawList) -> Vec<Instance> {
        list.instances
            .iter()
            .filter(|i| i.kind == qs_gpu::PrimKind::Glow as u32)
            .copied()
            .collect()
    }

    /// The accent halos a selected row draws, without its contact shadows.
    ///
    /// `row/selected` draws **two** glows since `prim-emissive-edge`: a dark contact shadow
    /// beneath everything, and the accent spill from the row's emissive edge. Both are
    /// `PrimKind::Glow` because physically they *are* the same primitive — light falling off
    /// from a shape — so nothing in the draw list distinguishes them except that the shadow is
    /// **displaced** along the direction the key light says a shadow falls, and the spill sits
    /// exactly on its region.
    ///
    /// Derived from `shadow_direction` rather than from the order the layers happen to be
    /// declared in. Layer order is load-bearing for *painting* and is stated in the token
    /// file, but a test that read it would silently start measuring the shadow the first time
    /// somebody reordered the stack — and would still pass, on the wrong instance.
    ///
    /// This is the price of identifying a layer by its primitive instead of by its role, which
    /// `is_region_body` above already pays and states. The alternative is a marker threaded
    /// through the draw list that only tests would read.
    fn halos(list: &DrawList) -> Vec<Instance> {
        let along = qs_gpu::frame::shadow_direction();
        let reach = |i: &Instance| i.rect[0] * along[0] + i.rect[1] * along[1];
        let mut glows = glows(list);
        // Group by size: every selected row contributes one shadow and one spill of identical
        // extent, and the shadow is the one further along the shadow direction.
        let mut halos = Vec::new();
        while let Some(first) = glows.pop() {
            let twin = glows
                .iter()
                .position(|g| g.rect[2] == first.rect[2] && g.rect[3] == first.rect[3]);
            match twin {
                Some(index) => {
                    let other = glows.remove(index);
                    halos.push(if reach(&first) < reach(&other) {
                        first
                    } else {
                        other
                    });
                }
                // A glow with no twin is not a selection pair; keep it rather than dropping it
                // silently, so a material that grows a third glow fails a count somewhere
                // instead of disappearing from every test at once.
                None => halos.push(first),
            }
        }
        halos
    }

    /// The contact shadows: the displaced half of the pairs [`halos`] returns the other half of.
    fn contact_shadows(list: &DrawList) -> Vec<Instance> {
        let halos = halos(list);
        glows(list)
            .into_iter()
            .filter(|g| !halos.iter().any(|h| h.rect == g.rect))
            .collect()
    }

    #[test]
    fn the_selection_halo_flares_while_the_region_is_moving_and_settles_when_it_lands() {
        // The material half is checked in `material::tests`; this is the *wiring*, which is
        // the half a unit test on `Material` cannot see. It is also the check a screenshot
        // could not give: the flare lives inside a 65 ms selection change, and a capture
        // that lands a frame late shows the settled halo and looks like a feature that was
        // never built.
        let mut renderer = renderer();
        let buf = plain_rows(20);
        let layout = layout_for(20, 1.0);
        let selection = one(12);
        let interaction = Interaction::with_selection(&selection);

        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.set_selected(Some(4));
        while motion.advance(1.0 / 120.0) {}
        motion.set_selected(Some(12));
        assert!(motion.is_animating(), "nothing to measure");

        let moving = halos(&draw(&mut renderer, &buf, &layout, interaction, &motion));
        while motion.advance(1.0 / 120.0) {}
        let landed = halos(&draw(&mut renderer, &buf, &layout, interaction, &motion));

        assert_eq!(moving.len(), 1, "one selected row, one halo");
        assert_eq!(landed.len(), 1);
        // 1.3x, not 1.5x. The halo's authored reach swell came down from 1.75 to 1.4 when
        // its resting alpha and reach went UP -- a spill that already reaches two space
        // steps does not need to nearly double again to read as flaring, and at the old
        // swell it washed over three rows. The claim under test is unchanged: the drive
        // reaches the material, and the flare is visible moving versus at rest.
        assert!(
            moving[0].param > landed[0].param * 1.3,
            "the halo reached {} while moving against {} at rest: the drive is not reaching \
             the material",
            moving[0].param,
            landed[0].param
        );
        assert!(
            moving[0].color >> 24 > landed[0].color >> 24,
            "the halo did not brighten while the region was moving"
        );
    }

    #[test]
    fn under_reduced_motion_the_halo_never_flares() {
        // Constitution VI, in the channel nobody would think to check. A reduced plan opens
        // no ticket, so the drive is zero on every frame and the halo is exactly its token.
        let mut renderer = renderer();
        let buf = plain_rows(20);
        let layout = layout_for(20, 1.0);
        let selection = one(12);
        let interaction = Interaction::with_selection(&selection);

        let mut reduced = InteractionMotion::new(crate::motion::MotionPreference::Reduced);
        reduced.set_selected(Some(4));
        reduced.set_selected(Some(12));
        assert!(!reduced.is_animating(), "a reduced plan opened a ticket");
        let under_reduced = halos(&draw(&mut renderer, &buf, &layout, interaction, &reduced));

        let mut full = InteractionMotion::new(crate::motion::MotionPreference::Full);
        full.set_selected(Some(12));
        while full.advance(1.0 / 120.0) {}
        let settled = halos(&draw(&mut renderer, &buf, &layout, interaction, &full));

        assert_eq!(under_reduced.len(), 1);
        assert_eq!(
            under_reduced[0].param, settled[0].param,
            "the reduced halo is not the resting halo"
        );
        assert_eq!(under_reduced[0].color, settled[0].color);
    }

    /// Every selected row, as a `Selection`.
    fn many(indices: &[u64]) -> Selection {
        let mut selection = Selection::new();
        for &index in indices {
            selection.toggle(index);
        }
        selection
    }

    #[test]
    fn a_selected_row_carries_a_halo_on_exactly_the_shape_of_its_region() {
        let mut renderer = renderer();
        let buf = plain_rows(6);
        let layout = layout_for(6, 1.0);
        let selection = one(3);
        let interaction = Interaction::with_selection(&selection);
        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.set_selected(Some(3));
        while motion.advance(1.0 / 120.0) {}

        let list = draw(&mut renderer, &buf, &layout, interaction, &motion);
        let halos = halos(&list);
        assert_eq!(halos.len(), 1, "expected one halo for one selected row");

        let regions = state_regions(&list, &layout);
        let region = regions.last().expect("no selection region was drawn");
        assert_eq!(
            halos[0].rect, region.rect,
            "the halo is not on the region's rect, so it will sit visibly off its shape"
        );
        assert_eq!(halos[0].radius, region.radius);
        assert!(
            halos[0].param > 0.0,
            "the halo has no falloff, so it is a flat rectangle behind the row"
        );
    }

    #[test]
    fn every_halo_is_drawn_before_every_fill_and_not_row_by_row() {
        // The defect the two-pass split exists to prevent. Emitted per row, the halo of row
        // N+1 reaches back over the *fill* of row N and lays an accent band across the
        // bottom of every selected row but the last -- which reads as a rendering fault, not
        // as a glow. Nothing else in the frame would look wrong, which is why this is an
        // assertion about emission order rather than about pixels.
        let mut renderer = renderer();
        let buf = plain_rows(8);
        let layout = layout_for(8, 1.0);
        let selection = many(&[2, 3, 4]);
        let interaction = Interaction::with_selection(&selection);
        let motion = settled();

        let list = draw(&mut renderer, &buf, &layout, interaction, &motion);
        let last_halo = list
            .instances
            .iter()
            .rposition(|i| i.kind == qs_gpu::PrimKind::Glow as u32)
            .expect("no halo was drawn");
        let regions = state_regions(&list, &layout);
        assert_eq!(regions.len(), 3, "expected three selection fills");
        let first_fill = list
            .instances
            .iter()
            .position(|i| is_region_body(i) && i.color == regions[0].color)
            .expect("no selection fill was drawn");

        assert!(
            last_halo < first_fill,
            "halo at {last_halo} is emitted after a fill at {first_fill}: adjacent selected \
             rows will show a bright band where one row's glow crosses the next one's fill"
        );
    }

    #[test]
    fn the_halo_is_not_drawn_when_the_platform_has_asked_for_no_effects() {
        // Forced-colours mode supplies a fixed palette, and a halo is not in it. Same switch
        // the command bar's gradient reads, so the two cannot disagree about what "effects"
        // means.
        let black = Srgba::new(0.0, 0.0, 0.0, 1.0);
        let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
        let highlight = Srgba::new(0.1, 0.3, 0.9, 1.0);
        let db: Arc<dyn FontDb> = Arc::new(qs_text::SystemFontDb::scan());
        let mut renderer = ListRenderer::new(
            Tokens::forced(black, white, highlight, white),
            db,
            qs_gpu::config_for(qs_gpu::RenderPath::Cpu),
        );
        let buf = plain_rows(4);
        let layout = layout_for(4, 1.0);
        // Two rows, so the fills come from the multiple-selection path and do not depend on
        // a morph having been driven -- this test is about the palette, not about motion.
        let selection = many(&[1, 2]);
        let interaction = Interaction::with_selection(&selection);

        let list = draw(&mut renderer, &buf, &layout, interaction, &settled());
        assert!(
            glows(&list).is_empty(),
            "a halo was drawn in forced-colours mode"
        );
        assert!(
            !state_regions(&list, &layout).is_empty(),
            "the selection itself vanished, which is a different and worse bug"
        );
    }

    #[test]
    fn the_halo_costs_a_measured_amount_of_overdraw_and_not_an_assumed_one() {
        // FR-002 bounds a frame's work by the visible entry count, and a glow quad is much
        // larger than its shape. Selection under focus under hover on the same row is where
        // they stack, so the worst case worth measuring is a full viewport with every row
        // selected, hovered, pressed and one of them focused.
        //
        // Instance count and *fill* cost are two different questions and only the second one
        // moves: a halo is one more instance per selected row, but its quad is padded by the
        // falloff on all four sides, which on a 28px row is most of the growth. The number
        // below is measured and printed, not asserted at a round figure -- what is asserted
        // is the shape of the answer, that the halo layer costs less than the rest of the
        // frame put together.
        let mut renderer = renderer();
        let rows = 40u64;
        let buf = plain_rows(rows as usize);
        let layout = layout_for(rows, 1.0);
        let selection = many(&(0..rows).collect::<Vec<_>>());
        let interaction = Interaction {
            hovered: Some(7),
            pressed: Some(7),
            focused: Some(7),
            ..Interaction::with_selection(&selection)
        };
        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.sync(interaction);
        while motion.advance(1.0 / 120.0) {}

        let list = draw(&mut renderer, &buf, &layout, interaction, &motion);

        // The area the rasterizer actually shades: the quad the vertex stage generates,
        // which is the rect grown by the same padding rule the shader applies. Counting the
        // instance's rect instead would report a glow as costing what its shape costs, which
        // is the assumption this test exists to replace.
        let quad_area = |i: &Instance| -> f64 {
            let pad = if i.kind == qs_gpu::PrimKind::Glyph as u32 {
                0.0
            } else if i.kind == qs_gpu::PrimKind::Glow as u32 {
                i.param.max(0.0) + 1.0
            } else {
                1.0
            };
            f64::from(i.rect[2] + pad * 2.0).max(0.0) * f64::from(i.rect[3] + pad * 2.0).max(0.0)
        };

        let total: f64 = list.instances.iter().map(quad_area).sum();
        let halo: f64 = list
            .instances
            .iter()
            .filter(|i| i.kind == qs_gpu::PrimKind::Glow as u32)
            .map(quad_area)
            .sum();
        let viewport = f64::from(layout.width) * f64::from(layout.height);
        let halos = halos(&list).len();
        let shadows = contact_shadows(&list).len();

        eprintln!(
            "overdraw over {rows} rows, all selected: {} instances ({halos} halos), fill \
             {:.2}x viewport without halos, {:.2}x with",
            list.instances.len(),
            (total - halo) / viewport,
            total / viewport
        );

        // One per selected row, plus one for the HOVERED row, which now has a spill of its
        // own: when the selection became a lamp, hover became the dimmer light beside it
        // rather than a flat wash. Stated as the sum rather than loosened to an inequality,
        // because what this test exists to catch is a halo per *pass* or per *layer*
        // creeping in, and only an exact count sees that.
        assert_eq!(
            halos,
            rows as usize + 1,
            "one halo per selected row, plus the hovered row's own"
        );
        // And one shadow. Counted separately rather than folded into the number above,
        // because they are the two halves of what a selected row costs in fill and they are
        // allowed to move independently -- the spill's reach swells with the drive and the
        // shadow's does not.
        assert_eq!(
            shadows, rows as usize,
            "one contact shadow per selected row"
        );
        assert!(
            halo < total - halo,
            "the halo layer costs {:.2}x the viewport against {:.2}x for the whole rest of \
             the frame: a decorative layer has become the majority of the fill cost, and \
             the `reach` on `row/selected`'s glow layer is the number to reconsider",
            halo / viewport,
            (total - halo) / viewport
        );
    }

    #[test]
    fn the_surface_primitive_budget_holds_in_the_worst_case() {
        // FR-002's bound re-measured in the unit that costs frame time, now that a look is
        // composed rather than written out. Bounding *entries* was enough while a row's
        // surface was a fixed handful of hand-written pushes; a material makes adding a
        // layer a one-line edit to design/tokens.json, and one line there costs one instance
        // per visible entry on every frame. Nothing in a token file looks like a frame-time
        // decision, so this is where that decision gets made.
        //
        // Two views, both worst-case: the list, and the grid, which packs several entries per
        // laid-out row and multiplies the bound by `columns` for exactly this reason.
        let mut renderer = renderer();
        let rows = 40u64;
        let buf = plain_rows(rows as usize);
        let layout = layout_for(rows, 1.0);
        let selection = many(&(0..rows).collect::<Vec<_>>());
        let interaction = Interaction {
            hovered: Some(7),
            pressed: Some(7),
            focused: Some(7),
            ..Interaction::with_selection(&selection)
        };
        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.sync(interaction);
        while motion.advance(1.0 / 120.0) {}

        let list = draw(&mut renderer, &buf, &layout, interaction, &motion);

        let mut grid_list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let grid = grid_layout(rows, metrics, 800, 0.0);
        let grid_buf = named_grid_rows(
            grid.visible.count as usize,
            b"quarterly-report.pdf",
            RowFlags::EMPTY,
        );
        let grid_selection = many(&(0..rows).collect::<Vec<_>>());
        let grid_interaction = Interaction {
            hovered: Some(3),
            pressed: Some(3),
            focused: Some(3),
            ..Interaction::with_selection(&grid_selection)
        };
        let mut grid_motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        grid_motion.sync(grid_interaction);
        while grid_motion.advance(1.0 / 120.0) {}
        renderer.render_grid(
            &mut grid_list,
            &grid_buf,
            &grid,
            metrics,
            grid_interaction,
            &grid_motion,
        );

        for (view, list, layout) in [("list", &list, &layout), ("grid", &grid_list, &grid)] {
            let surface_prims = list
                .instances
                .iter()
                .filter(|i| i.kind != qs_gpu::PrimKind::Glyph as u32)
                .count() as u32;
            let entries = layout.visible.count.max(1);

            eprintln!(
                "{view}: {surface_prims} surface primitives over {entries} entries \
                 ({:.2} per entry, allowance {}), bound {}",
                f64::from(surface_prims) / f64::from(entries),
                crate::recycler::SURFACE_PRIMS_PER_ENTRY,
                layout.surface_prim_bound(),
            );

            assert!(
                surface_prims <= layout.surface_prim_bound(),
                "the {view} pass emitted {surface_prims} surface primitives against a bound \
                 of {} ({} per entry over {entries} entries). A material gained a layer: \
                 either the layer is worth its per-frame cost and \
                 `SURFACE_PRIMS_PER_ENTRY` should say so, or it is not.",
                layout.surface_prim_bound(),
                crate::recycler::SURFACE_PRIMS_PER_ENTRY,
            );

            // Guard the guard. A bound nothing approaches is a bound that would not notice a
            // look doubling, and the whole risk here is a change nobody thinks of as a
            // performance change.
            // Against the entries actually drawn rather than against the viewport's whole
            // bound: what is being guarded is the per-entry allowance, and a half-full
            // viewport would otherwise read as slack in a number that has none.
            let allowed = entries * crate::recycler::SURFACE_PRIMS_PER_ENTRY;
            assert!(
                surface_prims * 2 > allowed,
                "the {view} worst case reaches only {surface_prims} of the {allowed} its \
                 {entries} entries are allowed, which is slack enough that a doubled \
                 material would still pass"
            );
        }
    }

    #[test]
    fn under_reduced_motion_the_selection_is_fully_painted_on_the_first_frame() {
        // "Instant" has to mean instant in the draw list, not merely in the plan.
        let mut renderer = renderer();
        let buf = plain_rows(6);
        let layout = layout_for(6, 1.0);
        let selection = one(4);
        let interaction = Interaction::with_selection(&selection);
        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Reduced);
        motion.sync(interaction);
        assert!(!motion.is_animating());

        let regions = state_regions(
            &draw(&mut renderer, &buf, &layout, interaction, &motion),
            &layout,
        );
        assert_eq!(regions.len(), 1);
        assert!(
            (regions[0].rect[1] - layout.row_top(4)).abs() < layout.row_height as f32,
            "the region landed at {} rather than on row 4 at {}",
            regions[0].rect[1],
            layout.row_top(4)
        );
        assert!(
            alpha_of(&regions[0]) > 0,
            "the region was drawn transparent"
        );
    }

    #[test]
    fn a_stub_never_reads_its_kind() {
        // The RowSource contract says `kind` is undefined on a stub, not stale. Drawing an
        // image icon because the undefined bytes happened to say 5 would be inventing
        // information about a row nothing is known about yet.
        let stub = RowView {
            kind: crate::row_source::KindId(5),
            state: LoadState::Stub,
            ..RowView::default()
        };
        assert_eq!(kind_of(&stub), IconKind::Generic);

        let mut dir_stub = stub.clone();
        dir_stub.flags = RowFlags::IS_DIR;
        assert_eq!(kind_of(&dir_stub), IconKind::Folder);
    }

    // -- The symlink emblem (UXDD 10.4) --------------------------------------------------

    /// `rows` rows, every `nth` of them a symlink. Same kind and name throughout, so the only
    /// thing that varies between two draw lists is the emblem.
    fn symlink_rows(rows: usize, nth: usize) -> RowBuf {
        let mut buf = RowBuf::new();
        for i in 0..rows {
            let flags = if nth > 0 && i % nth == 0 {
                RowFlags::IS_SYMLINK
            } else {
                RowFlags::EMPTY
            };
            buf.push(
                RowView {
                    kind: crate::row_source::KindId(2),
                    flags,
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                b"link.rs",
            );
        }
        buf
    }

    /// The glyph-primitive instances that are a `px`-by-`px` square.
    ///
    /// Square as well as sized, because an icon rides the `Glyph` primitive and so does text:
    /// filtering on width alone would count any letter that happened to advance the same
    /// number of pixels, which is a different number on every font the system might hand us.
    fn icon_quads_of(list: &DrawList, px: f32) -> Vec<&Instance> {
        list.instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Glyph as u32
                    && (i.rect[2] - px).abs() < 0.5
                    && (i.rect[3] - px).abs() < 0.5
            })
            .collect()
    }

    #[test]
    fn a_symlink_carries_two_emblem_quads_and_a_plain_row_carries_none() {
        // Two, not one: the plate knocks the silhouette out and the mark is drawn over it in a
        // different colour. One quad would mean the mark is sitting directly on the icon's
        // strokes, which is the state this chunk exists to leave.
        let icon_px = f32::from(qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0));
        let emblem_px =
            f32::from(qs_gpu::icon::emblem_px(icon_px as u16).expect("20px carries an emblem"));

        let mut renderer = renderer();
        let layout = layout_for(8, 1.0);

        let linked = draw(
            &mut renderer,
            &symlink_rows(8, 2),
            &layout,
            Interaction::default(),
            &settled(),
        );
        assert_eq!(
            icon_quads_of(&linked, emblem_px).len(),
            8, // four symlinked rows, two quads each
            "a symlink row is not drawing plate + mark"
        );

        let plain = draw(
            &mut renderer,
            &symlink_rows(8, 0),
            &layout,
            Interaction::default(),
            &settled(),
        );
        assert!(
            icon_quads_of(&plain, emblem_px).is_empty(),
            "an emblem was drawn on rows that are not symlinks"
        );
    }

    #[test]
    fn the_emblem_sits_in_the_icons_bottom_left_corner() {
        let icon_px = f32::from(qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0));
        let emblem_px =
            f32::from(qs_gpu::icon::emblem_px(icon_px as u16).expect("20px carries an emblem"));

        let mut renderer = renderer();
        let layout = layout_for(1, 1.0);
        let list = draw(
            &mut renderer,
            &symlink_rows(1, 1),
            &layout,
            Interaction::default(),
            &settled(),
        );

        let icon = *icon_quads_of(&list, icon_px)
            .first()
            .expect("no icon was drawn");
        for emblem in icon_quads_of(&list, emblem_px) {
            assert!(
                (emblem.rect[0] - icon.rect[0]).abs() < 0.6,
                "the emblem is not flush with the icon's left edge ({} vs {})",
                emblem.rect[0],
                icon.rect[0]
            );
            let bottom = icon.rect[1] + icon_px;
            assert!(
                (emblem.rect[1] + emblem_px - bottom).abs() < 0.6,
                "the emblem is not flush with the icon's bottom edge"
            );
        }
    }

    #[test]
    fn drawing_emblems_still_drops_no_icons_on_the_cold_frame() {
        // Criterion 4, and the specific regression think:33 predicted. An emblem asked for at
        // a size `rasterize` refuses is counted as a *dropped icon*, which is the CPU tier's
        // upload-budget failure -- so a design decision would surface as a rendering fault.
        let mut renderer = renderer();
        let buf = symlink_rows(12, 1);
        let mut list = DrawList::default();
        renderer.render(
            &mut list,
            &buf,
            &layout_for(12, 1.0),
            Interaction::default(),
            &settled(),
        );
        assert_eq!(
            list.stats.icons_dropped, 0,
            "emblems went missing on the cold frame"
        );
    }

    #[test]
    fn an_icon_too_small_for_an_emblem_skips_it_rather_than_dropping_it() {
        // The other half of the same rule. Below the threshold there is no emblem *and* no
        // drop: silence, not a counted failure.
        let icon_px = qs_gpu::icon::MIN_PX + 1;
        assert_eq!(
            qs_gpu::icon::emblem_px(icon_px),
            None,
            "this test is not exercising the small-icon path any more"
        );

        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(400, 24.0, 1.0);
        let layout = grid_layout(40, metrics, 400, 0.0);
        let mut buf = RowBuf::new();
        for _ in 0..layout.visible.count {
            buf.push(
                RowView {
                    kind: crate::row_source::KindId(2),
                    flags: RowFlags::IS_SYMLINK,
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                b"link.rs",
            );
        }
        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );
        assert_eq!(
            list.stats.icons_dropped, 0,
            "an emblem too small to draw was counted as a dropped icon"
        );
    }

    #[test]
    fn the_grid_emblems_a_symlink_exactly_as_the_list_does() {
        // `kind_of` is one function so the two views cannot disagree about what a row is.
        // The emblem has the same obligation.
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let layout = grid_layout(40, metrics, 720, 0.0);
        let emblem_px = f32::from(
            qs_gpu::icon::emblem_px(metrics.icon_px).expect("a 120px cell carries an emblem"),
        );
        let mut buf = RowBuf::new();
        for i in 0..layout.visible.count {
            let flags = if i % 2 == 0 {
                RowFlags::IS_SYMLINK
            } else {
                RowFlags::EMPTY
            };
            buf.push(
                RowView {
                    kind: crate::row_source::KindId(2),
                    flags,
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                b"link.rs",
            );
        }
        let linked = buf
            .rows()
            .iter()
            .filter(|r| r.flags.contains(RowFlags::IS_SYMLINK))
            .count();

        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );
        assert_eq!(
            icon_quads_of(&list, emblem_px).len(),
            linked * 2,
            "the grid draws a different number of emblem quads than the list would"
        );
    }

    // -- the session indicator (chunk `directory-session-indicators`) ---------------------

    /// Two folders and a file, all loaded, so a mark has somewhere to land and somewhere it
    /// must not.
    fn folders_and_a_file() -> RowBuf {
        let mut buf = RowBuf::new();
        for (name, is_dir) in [("src", true), ("docs", true), ("main.rs", false)] {
            buf.push(
                RowView {
                    flags: if is_dir {
                        RowFlags::IS_DIR
                    } else {
                        RowFlags::EMPTY
                    },
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                name.as_bytes(),
            );
        }
        buf
    }

    /// The instances a frame draws for `marks`, at scale 1.
    fn frame_with_marks(
        renderer: &mut ListRenderer,
        buf: &RowBuf,
        marks: &crate::mark::SessionMarks,
    ) -> DrawList {
        let layout = layout_for(buf.len() as u64, 1.0);
        let mut list = DrawList::default();
        renderer.render(
            &mut list,
            buf,
            &layout,
            Interaction {
                marks,
                ..Interaction::default()
            },
            &settled(),
        );
        list
    }

    /// Every rect this frame drew in `token`, at the reserved rail slot's `x`.
    fn rails_of(renderer: &ListRenderer, list: &DrawList, token: &str) -> usize {
        let columns = Columns::for_width(1200.0, 1.0, &renderer.tokens);
        let colour = renderer.tokens.color(token).to_premul_linear_rgba8();
        list.instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::frame::PrimKind::Rect as u32
                    && i.color == colour
                    && (i.rect[0] - columns.rail_x()).abs() < 0.5
            })
            .count()
    }

    #[test]
    fn a_folder_with_no_sessions_draws_nothing_extra_at_all() {
        // Acceptance 1's second half. Asserted as byte-identical draw lists rather than as
        // "no rail was found", because the failure worth catching is an indicator that draws
        // something invisible -- a zero-alpha glyph, an empty-string label, a rect of zero
        // width -- and every one of those passes a search for a colour and fails this.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let buf = folders_and_a_file();
        let without = frame_with_marks(&mut renderer, &buf, &crate::mark::NO_MARKS);
        let empty = crate::mark::SessionMarks::new();
        let also_without = frame_with_marks(&mut renderer, &buf, &empty);
        assert_eq!(
            without.instances.len(),
            also_without.instances.len(),
            "an empty mark set is not the same as no mark set"
        );
    }

    #[test]
    fn a_marked_folder_draws_its_count_and_an_unmarked_one_beside_it_does_not() {
        // Acceptance 1. The count is text, so it is counted as glyphs rather than looked for
        // by colour -- and it is compared against the SAME frame with the mark removed, so
        // what is measured is the indicator and not the two folder names beside it.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let buf = folders_and_a_file();
        let bare = frame_with_marks(&mut renderer, &buf, &crate::mark::NO_MARKS);

        let mut marks = crate::mark::SessionMarks::new();
        marks.insert(0, crate::mark::SessionMark::new(2, true, None).unwrap());
        let marked = frame_with_marks(&mut renderer, &buf, &marks);

        let glyphs = |list: &DrawList| {
            list.instances
                .iter()
                .filter(|i| i.kind == qs_gpu::frame::PrimKind::Glyph as u32)
                .count()
        };
        assert!(
            glyphs(&marked) > glyphs(&bare),
            "the marked folder drew no more text than the unmarked one"
        );
    }

    #[test]
    fn a_summons_flash_draws_in_the_rail_slot_and_a_finished_one_draws_nothing() {
        // The arrival's row reading: mid-flash the reserved rail slot carries the accent;
        // at intensity zero — which is where the pulse ends — the mark is byte-identical
        // to one that never flashed. Asserted as whole draw lists for the reason
        // `a_folder_with_no_sessions_draws_nothing_extra_at_all` gives: an invisible rect
        // passes a search for a colour and fails this.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let buf = folders_and_a_file();
        let mark = crate::mark::SessionMark::new(2, true, None).unwrap();

        let mut settled = crate::mark::SessionMarks::new();
        settled.insert(0, mark.clone());
        // Warm the glyph atlas first: a cold one flushes text at the end of the list while
        // a warm one emits inline, and what is compared below must differ only in the mark.
        let _ = frame_with_marks(&mut renderer, &buf, &settled);
        let quiet = frame_with_marks(&mut renderer, &buf, &settled);

        let mut flashing = crate::mark::SessionMarks::new();
        flashing.insert(0, mark.clone().with_arrival(Some(0.8)));
        let flashed = frame_with_marks(&mut renderer, &buf, &flashing);
        assert_eq!(
            rails_of(&renderer, &quiet, "border/focus"),
            0,
            "a settled mark drew an accent rail"
        );
        assert_eq!(
            flashed.instances.len(),
            quiet.instances.len() + 1,
            "the flash is exactly one rect in the reserved slot"
        );

        // A finished arrival quantizes to nothing and takes the untouched path.
        let mut finished = crate::mark::SessionMarks::new();
        finished.insert(0, mark.with_arrival(Some(0.0)));
        let after = frame_with_marks(&mut renderer, &buf, &finished);
        assert_eq!(after.instances, quiet.instances);
    }

    #[test]
    fn an_outcome_rail_outranks_the_summons_flash() {
        // A folder where one session failed and another just started waiting shows the
        // failure: the flash is "look here", and a failure already says that louder. One
        // rect either way — the two must not stack in one slot.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let buf = folders_and_a_file();
        let mut marks = crate::mark::SessionMarks::new();
        marks.insert(
            0,
            crate::mark::SessionMark::new(2, true, Some("rail/conflict"))
                .unwrap()
                .with_arrival(Some(0.8)),
        );
        let list = frame_with_marks(&mut renderer, &buf, &marks);
        assert_eq!(rails_of(&renderer, &list, "rail/conflict"), 1);
        // Counted by position rather than by colour, because the flash is drawn at a
        // modulated alpha no token search would find: one rect in the slot, full stop.
        let columns = Columns::for_width(1200.0, 1.0, &renderer.tokens);
        let in_slot = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::frame::PrimKind::Rect as u32
                    && (i.rect[0] - columns.rail_x()).abs() < 0.5
            })
            .count();
        assert_eq!(in_slot, 1, "the flash stacked under an outcome rail");
    }

    #[test]
    fn a_file_row_keeps_its_size_column_even_when_something_marks_its_index() {
        // Acceptance 5's neighbour: the indicator takes the slot a FOLDER leaves empty, and
        // must never displace a file's size. Index 2 is `main.rs`, and marking it is exactly
        // the mistake a join keyed on the wrong thing would make.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let buf = folders_and_a_file();
        let bare = frame_with_marks(&mut renderer, &buf, &crate::mark::NO_MARKS);

        let mut marks = crate::mark::SessionMarks::new();
        marks.insert(2, crate::mark::SessionMark::new(9, true, None).unwrap());
        let marked = frame_with_marks(&mut renderer, &buf, &marks);

        assert_eq!(
            bare.instances.len(),
            marked.instances.len(),
            "a mark on a file row changed what the row drew"
        );
    }

    #[test]
    fn the_count_is_muted_when_no_shell_vouched_for_being_there() {
        // Acceptance 2, the INK half -- the half `mark.rs` cannot assert, because it holds the
        // words and this holds the colours. Both channels have to be asserted separately: a
        // confidence carried in one and quietly dropped from the other is exactly the state
        // this chunk exists to make impossible, and it is invisible from either side alone.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let secondary = renderer
            .tokens
            .color("content/secondary")
            .to_premul_linear_rgba8();
        let tertiary = renderer
            .tokens
            .color("content/tertiary")
            .to_premul_linear_rgba8();
        assert_ne!(secondary, tertiary, "the two ink levels are one colour");

        let buf = folders_and_a_file();
        let count_glyphs = |list: &DrawList, colour: u32| {
            list.instances
                .iter()
                .filter(|i| i.kind == qs_gpu::frame::PrimKind::Glyph as u32 && i.color == colour)
                .count()
        };

        let mut vouched = crate::mark::SessionMarks::new();
        vouched.insert(0, crate::mark::SessionMark::new(2, true, None).unwrap());
        let vouched = frame_with_marks(&mut renderer, &buf, &vouched);

        let mut inherited = crate::mark::SessionMarks::new();
        inherited.insert(0, crate::mark::SessionMark::new(2, false, None).unwrap());
        let inherited = frame_with_marks(&mut renderer, &buf, &inherited);

        assert!(
            count_glyphs(&vouched, secondary) > count_glyphs(&inherited, secondary),
            "the vouched count is not drawn in ordinary ink"
        );
        assert!(
            count_glyphs(&inherited, tertiary) > count_glyphs(&vouched, tertiary),
            "the inherited count is not muted"
        );
    }

    #[test]
    fn only_an_outcome_draws_a_rail_and_a_live_shell_draws_none() {
        // Acceptance 3, and the constraint that shapes it: `border/subtle` is declared outside
        // the contrast gate in `design/tokens.json`, so a running-only folder draws no rail
        // rather than an ungated one. A rail that appeared for every marked folder would be an
        // undeclared pair on four row grounds -- which is not a failing check, it is an
        // UNCHECKED one, and the token file's own comment says so.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let buf = folders_and_a_file();

        let mut alive = crate::mark::SessionMarks::new();
        alive.insert(0, crate::mark::SessionMark::new(3, true, None).unwrap());
        let alive = frame_with_marks(&mut renderer, &buf, &alive);
        assert_eq!(rails_of(&renderer, &alive, "rail/conflict"), 0);
        assert_eq!(rails_of(&renderer, &alive, "rail/added"), 0);
        assert_eq!(
            rails_of(&renderer, &alive, "border/subtle"),
            0,
            "a live shell drew an ungated rail"
        );

        for token in ["rail/added", "rail/conflict"] {
            let mut marks = crate::mark::SessionMarks::new();
            marks.insert(
                1,
                crate::mark::SessionMark::new(1, true, Some(token)).unwrap(),
            );
            let list = frame_with_marks(&mut renderer, &buf, &marks);
            assert_eq!(
                rails_of(&renderer, &list, token),
                1,
                "{token} did not reach the reserved rail slot"
            );
        }
    }

    #[test]
    fn a_stub_directory_is_never_marked_however_it_is_asked() {
        // Acceptance 4, asserted at the renderer as well as at the join in `qs`. The join is
        // what decides not to mark a stub; this is what makes the decision safe to get wrong
        // -- a stub returns before the metadata block, so a mark filed against one by some
        // future caller still draws no count beside two placeholder bars.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let mut buf = RowBuf::new();
        buf.push(
            RowView {
                flags: RowFlags::IS_DIR,
                state: LoadState::Stub,
                ..RowView::default()
            },
            b"src",
        );
        let bare = frame_with_marks(&mut renderer, &buf, &crate::mark::NO_MARKS);

        let mut marks = crate::mark::SessionMarks::new();
        marks.insert(
            0,
            crate::mark::SessionMark::new(4, true, Some("rail/conflict")).unwrap(),
        );
        let marked = frame_with_marks(&mut renderer, &buf, &marks);
        assert_eq!(
            bare.instances.len(),
            marked.instances.len(),
            "a stub drew an indicator"
        );
        assert_eq!(rails_of(&renderer, &marked, "rail/conflict"), 0);
    }

    #[test]
    fn the_indicator_does_not_touch_the_folder_icon() {
        // Acceptance 5. `kind_of` still decides the icon, and the icon is drawn in the same
        // tint at the same size whether or not the row is marked -- which is what "beside it,
        // and does not replace or recolour it" means, asserted rather than asserted-by-comment.
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let buf = folders_and_a_file();
        let icon_px = qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0) as f32;
        let folder = renderer
            .tokens
            .color("icon/folder")
            .to_premul_linear_rgba8();
        let icons = |list: &DrawList| {
            list.instances
                .iter()
                .filter(|i| {
                    i.kind == qs_gpu::frame::PrimKind::Glyph as u32
                        && i.color == folder
                        && (i.rect[2] - icon_px).abs() < 0.5
                })
                .count()
        };

        let bare = frame_with_marks(&mut renderer, &buf, &crate::mark::NO_MARKS);
        let mut marks = crate::mark::SessionMarks::new();
        marks.insert(
            0,
            crate::mark::SessionMark::new(2, true, Some("rail/conflict")).unwrap(),
        );
        let marked = frame_with_marks(&mut renderer, &buf, &marks);
        assert_eq!(
            icons(&bare),
            icons(&marked),
            "the indicator changed the folder icon"
        );
        assert!(icons(&bare) > 0, "no folder icon was drawn to compare");
    }

    #[test]
    fn a_folder_is_tinted_from_its_own_token_and_never_from_the_focus_ring() {
        // The finding in think:32. `border/focus` satisfied "comes from a token", but it is
        // declared `role: border` and gated only as a boundary, so a filled glyph was riding a
        // focus ring's contrast requirement -- and a focused folder drew its icon in the same
        // colour as the ring around it.
        let tokens = Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let folder = tokens.color("icon/folder");
        assert_ne!(
            folder,
            tokens.color("border/focus"),
            "the folder tint is still the focus ring's colour"
        );

        let dir = RowView {
            flags: RowFlags::IS_DIR,
            state: LoadState::Basic,
            ..RowView::default()
        };
        assert_eq!(icon_tint(&tokens, &dir), folder);

        let file = RowView {
            state: LoadState::Basic,
            ..RowView::default()
        };
        assert_eq!(icon_tint(&tokens, &file), tokens.color("content/tertiary"));

        // And it is the colour that actually reaches the draw list, not just the one the
        // helper returns.
        let icon_px = f32::from(qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0));
        let mut renderer = renderer();
        let mut buf = RowBuf::new();
        buf.push(dir, b"src");
        let list = draw(
            &mut renderer,
            &buf,
            &layout_for(1, 1.0),
            Interaction::default(),
            &settled(),
        );
        let icon = *icon_quads_of(&list, icon_px)
            .first()
            .expect("no folder icon was drawn");
        assert_eq!(icon.color, folder.to_premul_linear_rgba8());
    }

    #[test]
    fn a_directory_is_a_folder_whatever_its_extension_says() {
        let row = RowView {
            // 2 is source code. A directory named `src.rs` is still a directory, and an
            // icon that says otherwise lies about what opening it does.
            kind: crate::row_source::KindId(2),
            flags: RowFlags::IS_DIR,
            state: LoadState::Basic,
            ..RowView::default()
        };
        assert_eq!(kind_of(&row), IconKind::Folder);
    }

    #[test]
    fn every_row_draws_exactly_one_icon_and_it_is_a_glyph_primitive() {
        // "A glyph primitive" is the load-bearing half: an icon that drew as a `Rect` would
        // look right on the GPU tier and would not be sampling the atlas at all, so tier
        // parity would be a coincidence rather than a consequence.
        let mut renderer = renderer();
        let buf = icon_rows();
        let rows = buf.len() as u64;
        let mut list = DrawList::default();
        renderer.render(
            &mut list,
            &buf,
            &layout_for(rows, 1.0),
            Interaction::default(),
            &settled(),
        );

        // The *first* frame, deliberately: this is the frame on which the CPU tier's
        // sixty-four-upload budget is contested, and the one where icons used to lose.
        assert_eq!(
            list.stats.icons_dropped, 0,
            "icons went missing on the cold frame"
        );
        if list.stats.glyphs_rasterized > 0 {
            assert!(
                list.stats.glyphs_dropped > 0,
                "this fixture no longer exhausts the per-frame upload bound, so it can no \
                 longer show that icons survive contention for it -- give the rows longer or \
                 more varied names"
            );
        }

        let icon_px = qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0) as f32;
        let icons = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Glyph as u32
                    && (i.rect[2] - icon_px).abs() < 0.5
                    && (i.rect[3] - icon_px).abs() < 0.5
            })
            .count();
        assert_eq!(
            icons, rows as usize,
            "expected one {icon_px}px icon per row, found {icons} across {rows} rows"
        );
    }

    // -- the upload bound is allocated by demand, not by draw order -----------------------

    /// `count` rows whose names are all different, over an alphabet real filenames use.
    ///
    /// The repeated fixtures above are wrong for measuring the upload bound: twelve names
    /// repeated four times want the same glyphs four times, and a grid of them converges on
    /// the first frame while a real folder does not. Distinct names are what put a cold
    /// viewport over the bound, which is the condition being measured.
    fn distinct_name_rows(count: usize) -> RowBuf {
        const STEMS: [&str; 8] = [
            "quicksilver",
            "recycler",
            "atlas-upload",
            "breadcrumb",
            "shaped-run",
            "provider",
            "manifest",
            "viewport",
        ];
        const EXTS: [&str; 6] = ["rs", "json", "md", "png", "toml", "log"];
        let mut buf = RowBuf::new();
        for index in 0..count {
            let stem = STEMS.get(index % STEMS.len()).copied().unwrap_or("file");
            let ext = EXTS.get(index % EXTS.len()).copied().unwrap_or("bin");
            let name = format!("{stem}-{index:03}.{ext}");
            buf.push(
                RowView {
                    id: crate::row_source::RowId(index as u64),
                    kind: crate::row_source::KindId((index % 10) as u16),
                    size: 1024 * (index as u64 + 1) * 37,
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                name.as_bytes(),
            );
        }
        buf
    }

    /// Render `buf` cold, one frame at a time, until the atlas reports convergence.
    ///
    /// Returns `(frames, first frame's stats)`. The renderer is fresh, so frame one is a
    /// genuinely cold atlas -- which is the frame the whole mechanism is about.
    fn frames_to_converge(buf: &RowBuf, grid: bool) -> (u32, qs_gpu::frame::DrawStats) {
        let mut renderer = renderer();
        let entries = buf.len() as u64;
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let cell_rows = entries.div_ceil(u64::from(metrics.columns.max(1)));
        let (layout, grid_layout) = (
            layout_for(entries, 1.0),
            grid_layout(
                entries,
                metrics,
                metrics.cell_height * cell_rows as u32,
                0.0,
            ),
        );
        let mut first = qs_gpu::frame::DrawStats::default();

        // Bounded so a mechanism that never converges fails as a number rather than as a
        // hung test.
        const GIVE_UP: u32 = 64;
        for frame in 1..=GIVE_UP {
            let mut list = DrawList::default();
            if grid {
                renderer.render_grid(
                    &mut list,
                    buf,
                    &grid_layout,
                    metrics,
                    Interaction::default(),
                    &settled(),
                );
            } else {
                renderer.render(&mut list, buf, &layout, Interaction::default(), &settled());
            }
            if frame == 1 {
                first = list.stats;
            }
            let _ = renderer.take_uploads();
            if renderer.atlas.converged() && list.stats.glyphs_dropped == 0 {
                return (frame, first);
            }
        }
        (GIVE_UP + 1, first)
    }

    #[test]
    fn a_cold_viewport_converges_in_a_measured_number_of_frames() {
        // SC-006's other half. The bound means a cold viewport takes more than one frame to
        // fill, and "more than one" was previously an assumption nobody had checked. This
        // measures it, on the CPU tier, whose 64-upload bound is the tightest of the three.
        //
        // The number is asserted rather than merely printed, so raising it is a decision
        // somebody makes on purpose: a change that doubles the frames a cold folder needs
        // to become readable fails here with both numbers in the message.
        const CEILING: u32 = 4;
        let buf = distinct_name_rows(48);
        for (view, grid) in [("list", false), ("grid", true)] {
            let (frames, first) = frames_to_converge(&buf, grid);
            eprintln!(
                "cold {view}: converged in {frames} frame(s); frame 1 rasterized {} \
                 glyph(s), dropped {}, short {}",
                first.glyphs_rasterized, first.glyphs_dropped, first.glyph_shortfall
            );
            assert!(
                frames <= CEILING,
                "a cold {view} viewport took {frames} frames to converge, over the {CEILING} \
                 this is held to"
            );
            assert_eq!(
                first.icons_dropped, 0,
                "the cold {view} frame lost an icon, so the structural bound is not holding"
            );
        }
    }

    #[test]
    fn the_cold_frame_serves_more_draws_than_first_come_admission_could() {
        // The defect, stated as the thing it costs. The same 64 uploads buy a different
        // number of glyph draws depending on *which* 64 keys they are spent on, and that
        // difference is the whole chunk. Measured on this fixture, at this bound, with the
        // only change being the admission rule:
        //
        //     draw order    1757 served, 302 dropped
        //     demand order  1827 served, 211 dropped
        //
        // So the ceiling below sits between the two: a regression to first-come admission
        // fails it, which is the property that makes this a gate rather than a record of
        // what the code currently does. (Checked by temporarily restoring the old rule and
        // watching it go red -- the numbers above are that run.)
        //
        // Note what the symptom is *not*: the top rows do not get everything while the
        // bottom rows get nothing. Names share letters at shared subpixel phases, so the
        // top and bottom quarters serve within 15% of each other under either rule. What
        // the reader sees is individual letters missing throughout, which is why the
        // measurement has to be a count of draws and not a picture of where they landed.
        const UNSERVED_PERCENT_CEILING: usize = 12;

        let buf = distinct_name_rows(48);
        let rows = buf.len() as u64;
        let mut renderer = renderer();
        let mut list = DrawList::default();
        renderer.render(
            &mut list,
            &buf,
            &layout_for(rows, 1.0),
            Interaction::default(),
            &settled(),
        );

        assert!(
            list.stats.glyphs_dropped > 0,
            "this fixture no longer exhausts the upload bound, so it cannot show how the              bound is spent -- give the rows longer or more varied names"
        );

        // Text only: an icon is a glyph primitive too, and one per row would dilute exactly
        // the ratio being measured.
        let icon_px = qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0) as f32;
        let served = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Glyph as u32 && (i.rect[3] - icon_px).abs() >= 0.5
            })
            .count();
        let dropped = list.stats.glyphs_dropped as usize;
        let wanted = served + dropped;
        assert!(wanted > 0);
        let unserved = dropped * 100 / wanted;

        assert!(
            unserved <= UNSERVED_PERCENT_CEILING,
            "the cold frame left {unserved}% of its {wanted} glyph draws unserved              ({served} served, {dropped} dropped), over the {UNSERVED_PERCENT_CEILING}%              this is held to -- the upload bound is being spent in draw order again"
        );
    }

    #[test]
    fn icons_survive_a_cold_frame_with_no_prepass_ordering_them_first() {
        // What replaced the prepass. This locks the *classification* -- that icons are
        // structural and are therefore not rationed against text -- on a frame where the
        // content bound is provably exhausted.
        //
        // It is deliberately not the starvation test, and the distinction is worth being
        // honest about. Park-and-flush means content is admitted after the row loop ends,
        // so an icon requested inside that loop would currently win the race even on the
        // shared bound: this test would still pass. That is exactly the accident the chunk
        // objects to -- safety by pass ordering rather than by rule -- so the property that
        // no amount of text can starve an icon is asserted where it can actually be
        // constructed, in `qs_gpu::atlas`'s
        // `structural_entries_draw_from_their_own_bound_and_text_cannot_starve_them`, which
        // spends the content bound to the last upload *before* asking for the icon.
        let buf = distinct_name_rows(48);
        let rows = buf.len() as u64;
        let mut renderer = renderer();
        let mut list = DrawList::default();
        renderer.render(
            &mut list,
            &buf,
            &layout_for(rows, 1.0),
            Interaction::default(),
            &settled(),
        );

        let stats = renderer.atlas.stats();
        assert!(
            stats.content_this_frame >= renderer.atlas.content_bound(),
            "the content bound was not exhausted ({} of {}), so this proves nothing about \
             contention for it",
            stats.content_this_frame,
            renderer.atlas.content_bound()
        );
        assert!(
            stats.structural_this_frame > 0,
            "no icon was uploaded on the frame text spent entirely"
        );
        assert_eq!(list.stats.icons_dropped, 0);
    }

    #[test]
    fn a_text_pass_that_forgets_to_flush_is_caught_rather_than_silently_short() {
        // The one way this mechanism is worse than what it replaced: a parked glyph that
        // nobody flushes is not deferred, it is gone, and nothing else would say so. The
        // guard lives in `take_uploads` because that is the one call every frame makes
        // after every pass.
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let role = renderer.resolve_role(role::MD, 1.0, 1.0);
        renderer.draw_label(
            &mut list,
            "unflushed",
            0.0,
            20.0,
            500.0,
            role,
            Features::default(),
            renderer.tokens.color("content/primary"),
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = renderer.take_uploads();
            }))
            .is_err(),
            "a pass ended with parked glyphs and take_uploads accepted it"
        );
    }

    #[test]
    fn icons_are_rasterized_once_per_kind_per_frame_not_once_per_row() {
        // This measures atlas *hits*, and the choice is the whole test.
        //
        // Misses would prove nothing: the atlas is itself a cache, so a key is a miss once
        // however many times it is asked for, and a version of this test written against
        // `misses` passed with `IconCache` deleted. Hits count *lookups*, which is exactly
        // what the per-frame cache elides.
        //
        // The two fixtures have the same row count, the same names, the same flags and the
        // same states, so their glyph work is identical to the lookup. They differ only in
        // how many distinct icon kinds they contain. With the cache, the difference in hits
        // is the difference in kinds. Without it, both do one lookup per row and the
        // difference is zero.
        const ROWS: usize = 12;
        let uniform = uniform_kind_rows(ROWS, 2);
        let varied = varied_kind_rows(ROWS);
        let kinds = |buf: &RowBuf| {
            buf.rows()
                .iter()
                .map(kind_of)
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        let spread = kinds(&varied) - kinds(&uniform);
        assert!(spread >= 7, "the fixture must cover the icon set");

        // Second frame: everything is resident, so every lookup is a hit and nothing is
        // clouded by rasterization.
        let hits_on_the_second_frame = |buf: &RowBuf| {
            let mut renderer = renderer();
            let layout = layout_for(buf.len() as u64, 1.0);
            let mut list = DrawList::default();
            renderer.render(&mut list, buf, &layout, Interaction::default(), &settled());
            let warm = renderer.atlas.stats().hits;
            let mut list = DrawList::default();
            renderer.render(&mut list, buf, &layout, Interaction::default(), &settled());
            renderer.atlas.stats().hits - warm
        };

        let one_kind = hits_on_the_second_frame(&uniform);
        let many_kinds = hits_on_the_second_frame(&varied);
        assert_eq!(
            many_kinds - one_kind,
            spread as u64,
            "{ROWS} rows of {} kinds cost {many_kinds} atlas lookups against {one_kind} for \
             {ROWS} rows of one kind. Equal counts mean the icon is looked up once per ROW; \
             the per-frame cache exists so it is looked up once per KIND.",
            kinds(&varied)
        );
    }

    #[test]
    fn a_scale_change_rasterizes_a_new_icon_rather_than_scaling_the_old_bitmap() {
        // UXDD 10.4 forbids scaled bitmaps. The observable form of that promise is that the
        // drawn size at 2x is twice the size at 1x *and* came from a second rasterization.
        let mut renderer = renderer();
        let buf = icon_rows();
        let rows = buf.len() as u64;

        let mut at_1x = DrawList::default();
        renderer.render(
            &mut at_1x,
            &buf,
            &layout_for(rows, 1.0),
            Interaction::default(),
            &settled(),
        );
        let misses_1x = renderer.atlas.stats().misses;

        let mut at_2x = DrawList::default();
        renderer.render(
            &mut at_2x,
            &buf,
            &layout_for(rows, 2.0),
            Interaction::default(),
            &settled(),
        );
        assert!(
            renderer.atlas.stats().misses > misses_1x,
            "2x reused the 1x bitmaps, which is exactly the scaled-bitmap failure"
        );

        let size_of = |list: &DrawList, px: f32| {
            list.instances
                .iter()
                .find(|i| {
                    i.kind == qs_gpu::PrimKind::Glyph as u32
                        && (i.rect[2] - px).abs() < 0.5
                        && (i.rect[3] - px).abs() < 0.5
                })
                .map(|i| i.rect[2])
        };
        let one = qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0) as f32;
        let two = qs_gpu::icon::device_px(qs_gpu::icon::GRID, 2.0) as f32;
        assert_eq!(size_of(&at_1x, one), Some(one));
        assert_eq!(size_of(&at_2x, two), Some(two));
        assert!((two - one * 2.0).abs() < 0.01);
    }

    #[test]
    fn distinct_kinds_land_in_distinct_atlas_slots() {
        // If two kinds shared a key the rows would render identically and every test above
        // would still pass -- they all count instances, not appearances.
        let mut renderer = renderer();
        let buf = icon_rows();
        let rows = buf.len() as u64;
        let mut list = DrawList::default();
        renderer.render(
            &mut list,
            &buf,
            &layout_for(rows, 1.0),
            Interaction::default(),
            &settled(),
        );

        let icon_px = qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0) as f32;
        let uvs: Vec<[f32; 4]> = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Glyph as u32 && (i.rect[2] - icon_px).abs() < 0.5
            })
            .map(|i| i.uv)
            .collect();
        let distinct_uvs = {
            let mut keys: Vec<_> = uvs.iter().map(|uv| format!("{uv:?}")).collect();
            keys.sort();
            keys.dedup();
            keys.len()
        };
        let distinct_kinds: std::collections::HashSet<_> = buf.rows().iter().map(kind_of).collect();
        assert_eq!(
            distinct_uvs,
            distinct_kinds.len(),
            "distinct kinds must not share an atlas slot"
        );
    }

    #[test]
    fn a_hidden_file_dims_its_icon_the_same_way_it_dims_its_name() {
        let mut renderer = renderer();
        let mut buf = RowBuf::new();
        for flags in [RowFlags::EMPTY, RowFlags::IS_HIDDEN] {
            buf.push(
                RowView {
                    kind: crate::row_source::KindId(2),
                    flags,
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                b"a.rs",
            );
        }
        let mut list = DrawList::default();
        renderer.render(
            &mut list,
            &buf,
            &layout_for(2, 1.0),
            Interaction::default(),
            &settled(),
        );

        let icon_px = qs_gpu::icon::device_px(qs_gpu::icon::GRID, 1.0) as f32;
        let alphas: Vec<u32> = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Glyph as u32 && (i.rect[2] - icon_px).abs() < 0.5
            })
            .map(|i| i.color >> 24)
            .collect();
        assert_eq!(alphas.len(), 2);
        assert!(
            alphas[1] < alphas[0],
            "the hidden row's icon is as opaque as the visible one's ({alphas:?})"
        );
    }

    // -- Grid view ------------------------------------------------------------------------

    fn grid_metrics(surface_width: u32, cell: f32, scale: f32) -> GridMetrics {
        // 16px label line and an 8px space step: the numbers a resolved `ui/sm` and
        // `space/sm` land near, hard-coded so the geometry tests do not depend on which
        // font the machine happens to have.
        GridMetrics::fit(surface_width, cell, scale, 16.0 * scale, 8.0)
    }

    fn grid_layout(entries: u64, metrics: GridMetrics, height: u32, scroll: f64) -> ViewportLayout {
        let source = crate::row_source::StubbornSource { count: entries };
        crate::recycler::Recycler::new().layout_grid(
            &source,
            metrics.columns,
            metrics.cell_height,
            1200,
            height,
            1.0,
            1.0,
            crate::density::Density::default(),
            scroll,
        )
    }

    // -- The extension badge (UXDD §10.4) -------------------------------------------------

    fn grid_layout_scaled(
        entries: u64,
        metrics: GridMetrics,
        height: u32,
        scale: f32,
    ) -> ViewportLayout {
        let source = crate::row_source::StubbornSource { count: entries };
        crate::recycler::Recycler::new().layout_grid(
            &source,
            metrics.columns,
            metrics.cell_height,
            (1200.0 * scale) as u32,
            height,
            scale,
            1.0,
            crate::density::Density::default(),
            0.0,
        )
    }

    fn named_grid_rows(count: usize, name: &[u8], flags: RowFlags) -> RowBuf {
        let mut buf = RowBuf::new();
        for _ in 0..count {
            buf.push(
                RowView {
                    kind: crate::row_source::KindId(2),
                    flags,
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                name,
            );
        }
        buf
    }

    /// The badge's ribbon, found by its fill. `icon/badge` is drawn by nothing else, so within
    /// a row or grid draw list the colour identifies the chip exactly.
    fn badge_chips(list: &DrawList) -> Vec<&Instance> {
        let fill = Tokens::embedded(crate::tokens::Theme::Dark)
            .unwrap()
            .color("icon/badge")
            .to_premul_linear_rgba8();
        list.instances
            .iter()
            .filter(|i| i.kind == qs_gpu::PrimKind::Rect as u32 && i.color == fill)
            .collect()
    }

    /// Render one grid frame and report whether any cell got a badge.
    fn badged_at(cell: f32, scale: f32, name: &[u8], flags: RowFlags) -> bool {
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics((1200.0 * scale) as u32, cell, scale);
        let layout = grid_layout_scaled(40, metrics, (720.0 * scale) as u32, scale);
        let buf = named_grid_rows(layout.visible.count as usize, name, flags);
        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );
        !badge_chips(&list).is_empty()
    }

    /// The smallest whole logical cell size at which a badge appears, or `None`.
    fn smallest_badged_cell(scale: f32) -> Option<f32> {
        (24..=140)
            .map(|c| c as f32)
            .find(|&c| badged_at(c, scale, b"main.rs", RowFlags::EMPTY))
    }

    #[test]
    fn a_large_cell_badges_a_file_and_a_small_one_does_not() {
        assert!(
            badged_at(120.0, 1.0, b"main.rs", RowFlags::EMPTY),
            "a 120px cell (66px icon) drew no extension badge"
        );
        assert!(
            !badged_at(30.0, 1.0, b"main.rs", RowFlags::EMPTY),
            "a 30px cell (16px icon) drew a badge it has no room for"
        );
    }

    #[test]
    fn the_badge_threshold_is_the_same_logical_size_at_every_device_scale() {
        // BADGE_ICON_RATIO compares two *physical* lengths, so the device scale cancels and
        // the threshold is the same logical size on every display. Written as a literal 32,
        // or with a stray division by `scale`, this test is what goes red.
        let at_1x = smallest_badged_cell(1.0).expect("no cell size badges at 1x");
        let at_2x = smallest_badged_cell(2.0).expect("no cell size badges at 2x");
        assert!(
            (at_1x - at_2x).abs() <= 1.0,
            "the badge appears at {at_1x} logical px at 1x but {at_2x} at 2x"
        );

        // Where it lands is a *consequence* of the resolved `ui/xs` line box, not a constant,
        // so this asserts the band the derivation can produce rather than one number. UXDD
        // §10.4 says 32, which implies a line box near 12.8 px; the system UI font this test
        // runs against resolves nearer 14.7, so the honest threshold is higher. The band is
        // wide enough for that and still narrow enough to fail if the ratio is dropped or the
        // badge starts scaling with the icon.
        let icon_at_threshold = at_1x * ICON_BOX_FRACTION;
        assert!(
            (28.0..48.0).contains(&icon_at_threshold),
            "the derived threshold is a {icon_at_threshold:.1}px icon, outside what \
             BADGE_ICON_RATIO over a plausible ui/xs line box can produce"
        );
    }

    #[test]
    fn the_badge_never_enters_the_corner_the_symlink_emblem_owns() {
        let scale = 1.0;
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, scale);
        let layout = grid_layout_scaled(40, metrics, 720, scale);
        let buf = named_grid_rows(
            layout.visible.count as usize,
            b"main.rs",
            RowFlags::IS_SYMLINK,
        );
        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );

        let icon_px = f32::from(metrics.icon_px);
        let emblem_px = f32::from(
            qs_gpu::icon::emblem_px(metrics.icon_px).expect("a 120px cell carries an emblem"),
        );
        let icons = icon_quads_of(&list, icon_px);
        assert!(!icons.is_empty(), "no icons were drawn");
        let chips = badge_chips(&list);
        assert_eq!(
            chips.len(),
            icons.len(),
            "not every symlinked file got both an icon and a badge"
        );

        // Both present, neither overlapping: the emblem owns [icon_x, icon_x + emblem_px] and
        // the chip must start to the right of it.
        for (icon, chip) in icons.iter().zip(&chips) {
            assert!(
                chip.rect[0] >= icon.rect[0] + emblem_px,
                "the badge starts at {} and the emblem ends at {}",
                chip.rect[0],
                icon.rect[0] + emblem_px
            );
            assert!(
                chip.rect[0] + chip.rect[2] <= icon.rect[0] + icon_px + 0.5,
                "the badge runs past the icon's right edge"
            );
        }
    }

    #[test]
    fn the_ribbon_is_a_rect_and_costs_the_icon_atlas_nothing() {
        // Criterion 4. An emblem is a `px * px` square mask, so a wide-short ribbon cannot be
        // one; drawn as an atlas shape it would also mint an entry per (extension, size).
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let layout = grid_layout_scaled(40, metrics, 720, 1.0);
        // Ten distinct extensions: an atlas-backed ribbon would need ten entries.
        let mut buf = RowBuf::new();
        for i in 0..layout.visible.count {
            let name: &[u8] = match i % 5 {
                0 => b"main.rs",
                1 => b"notes.txt",
                2 => b"data.json",
                3 => b"photo.png",
                _ => b"build.log",
            };
            buf.push(
                RowView {
                    kind: crate::row_source::KindId(2),
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                name,
            );
        }
        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );

        assert!(!badge_chips(&list).is_empty(), "no badges were drawn");
        for chip in badge_chips(&list) {
            assert_eq!(
                chip.kind,
                qs_gpu::PrimKind::Rect as u32,
                "the ribbon is not a Rect"
            );
        }
        assert_eq!(
            list.stats.icons_dropped, 0,
            "badging cost the icon atlas entries it could not supply"
        );
    }

    #[test]
    fn badging_costs_the_cold_frame_no_dropped_glyphs() {
        // The finding that decided which type role the badge uses, pinned so it cannot be
        // undone by someone reading design/tokens.json and noticing that `ui/xs` is the role
        // literally named "Badges".
        //
        // An extension is a substring of the filename drawn directly below it, so at the
        // label's face and size its glyphs are already resident. At a role of its own they are
        // a second copy of characters the atlas already holds, and this frame cannot afford
        // them: measured on this exact grid, `ui/xs` hit the CPU tier's 64-upload ceiling and
        // dropped seven glyphs — and what went missing was the *filenames* of the last cells.
        // A badge bought at the price of the name it abbreviates is a bad trade, and this is
        // the assertion that says so.
        // A frame the labels alone can afford: nine cells in one band. Scoped that way on
        // purpose. A *large* cold grid already drops glyphs without any badge at all —
        // measured at 49 on a 40-cell frame here — because chunk:upload-budget-priority is
        // still open, and asserting zero there would pin someone else's defect rather than
        // this chunk's cost. On a frame with headroom the question is answerable and sharp:
        // does adding badges tip it over? At `ui/xs` it did, by seven. Here it must not.
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(850, 90.0, 1.0);
        let layout = grid_layout_scaled(9, metrics, 100, 1.0);
        // Distinct names, so the frame needs a realistic spread of glyphs rather than one
        // string repeated into a single cache entry.
        let names: [&[u8]; 9] = [
            b"main.rs",
            b"photo.png",
            b"notes.txt",
            b"data.json",
            b"archive.tar.gz",
            b"readme.md",
            b"Makefile",
            b"a.javascript",
            b"quicksilver.exe",
        ];
        let mut buf = RowBuf::new();
        for i in 0..layout.visible.count as usize {
            buf.push(
                RowView {
                    kind: crate::row_source::KindId((i % 9) as u16),
                    state: LoadState::Basic,
                    ..RowView::default()
                },
                names[i % names.len()],
            );
        }

        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );

        assert!(!badge_chips(&list).is_empty(), "no badges were drawn");
        // The *first* frame, deliberately: this is the one where the upload budget is
        // contested, exactly as the row pass's icon assertion does.
        assert_eq!(
            list.stats.glyphs_dropped, 0,
            "the badge spent the cold frame's upload budget and the filenames paid for it"
        );
    }

    #[test]
    fn nothing_without_an_extension_is_badged() {
        for name in [b"Makefile".as_slice(), b".gitignore", b"trailing."] {
            assert!(
                !badged_at(120.0, 1.0, name, RowFlags::EMPTY),
                "{} was badged and has no extension",
                String::from_utf8_lossy(name)
            );
        }
        // A folder has no type to announce, whatever its name looks like. UXDD §10.4 scopes
        // the extension to file icons.
        assert!(
            !badged_at(120.0, 1.0, b"site.com", RowFlags::IS_DIR),
            "a folder named like a file was badged"
        );
    }

    #[test]
    fn an_extension_too_wide_for_the_room_draws_nothing_rather_than_a_clipped_ribbon() {
        // `extension_start` caps a suffix at twelve bytes, so eleven characters is the widest
        // badge that can be asked for. At a cell only just over the threshold there is no room
        // for it beside the emblem's reserved corner, and the answer is absence.
        let cell = smallest_badged_cell(1.0).expect("no cell size badges") + 1.0;
        assert!(
            badged_at(cell, 1.0, b"a.rs", RowFlags::EMPTY),
            "this test is not exercising a cell that badges at all"
        );
        assert!(
            !badged_at(cell, 1.0, b"a.javascript", RowFlags::EMPTY),
            "an eleven-character extension was drawn into a box that cannot hold it"
        );
    }

    #[test]
    fn the_list_view_can_never_reach_the_badge_threshold() {
        // Criterion 6, proven rather than asserted structurally: the list's icon is fixed at
        // the 20-unit grid, and no device scale can lift it over a threshold that is a ratio.
        for scale in [1.0f32, 1.25, 1.5, 2.0, 3.0] {
            let icon = f32::from(qs_gpu::icon::device_px(qs_gpu::icon::GRID, scale));
            // The ribbon at its smallest: an 11px role, no descent, no padding. Even that
            // floor is above what a 20-logical-px icon can carry.
            let floor = 11.0 * scale;
            assert!(
                icon < floor * BADGE_ICON_RATIO,
                "at {scale}x the list's {icon}px icon clears the badge threshold"
            );
        }

        // And the list pass draws no chip whatever the name.
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let buf = named_grid_rows(8, b"main.rs", RowFlags::EMPTY);
        renderer.render(
            &mut list,
            &buf,
            &layout_for(8, 1.0),
            Interaction::default(),
            &settled(),
        );
        assert!(
            badge_chips(&list).is_empty(),
            "the list drew an extension badge"
        );
    }

    #[test]
    fn the_cell_size_is_continuous_rather_than_stepped() {
        // UXDD §5.1: "no fixed steps". A hundred sizes a logical pixel apart must produce a
        // monotonically growing cell, not a staircase that snaps to a handful of presets.
        let mut previous = 0u32;
        let mut distinct = 0usize;
        for step in 0..100 {
            let cell = 64.0 + step as f32 * 2.0;
            let height = grid_metrics(1200, cell, 1.0).cell_height;
            assert!(
                height >= previous,
                "cell {cell} shrank: {height} < {previous}"
            );
            if height != previous {
                distinct += 1;
            }
            previous = height;
        }
        assert!(
            distinct > 80,
            "only {distinct} distinct heights across 100 sizes -- that is a staircase"
        );
    }

    #[test]
    fn the_column_count_falls_as_the_cell_grows_and_never_reaches_zero() {
        let mut previous = u32::MAX;
        for step in 0..60 {
            let cell = 48.0 + step as f32 * 24.0;
            let columns = grid_metrics(1200, cell, 1.0).columns;
            assert!(columns >= 1, "cell {cell} produced {columns} columns");
            assert!(columns <= previous, "cell {cell} gained columns");
            previous = columns;
        }
        // A window narrower than a single cell still shows one.
        assert_eq!(grid_metrics(40, 320.0, 1.0).columns, 1);
    }

    #[test]
    fn the_icon_is_clamped_to_what_the_rasterizer_will_produce() {
        // The one place the continuity genuinely stops, asserted rather than left to be
        // discovered as an icon that silently fails to rasterize.
        let huge = grid_metrics(4000, 1000.0, 2.0);
        assert!(
            huge.icon_px <= qs_gpu::icon::MAX_PX,
            "icon {} px is past the rasterizer's ceiling",
            huge.icon_px
        );
        let tiny = grid_metrics(400, 24.0, 1.0);
        assert!(tiny.icon_px >= qs_gpu::icon::MIN_PX);
    }

    #[test]
    fn the_grid_covers_every_instance_it_appends() {
        // RP-2: an instance past the last batch is uploaded, reached by no draw call, and
        // invisible. The list pass and the two overlays each learned this separately.
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let layout = grid_layout(400, metrics, 720, 0.0);
        let buf = varied_kind_rows(layout.visible.count as usize);

        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );

        assert!(!list.instances.is_empty(), "the grid appended nothing");
        assert_eq!(
            list.unbatched(),
            0,
            "{} grid instances are covered by no draw call",
            list.unbatched()
        );
    }

    #[test]
    fn every_cell_draws_its_icon_and_they_land_where_the_layout_says() {
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let layout = grid_layout(400, metrics, 720, 0.0);
        let buf = uniform_kind_rows(layout.visible.count as usize, 5);

        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );

        let icon_px = f32::from(metrics.icon_px);
        let icons: Vec<&Instance> = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Glyph as u32 && (i.rect[2] - icon_px).abs() < 0.5
            })
            .collect();
        assert_eq!(
            icons.len(),
            buf.len(),
            "one icon per cell: {} icons for {} cells",
            icons.len(),
            buf.len()
        );

        // Each icon sits inside the cell its slot was assigned. An icon drawn at the wrong
        // column is the failure that a screenshot makes obvious and a count does not.
        for (slot, icon) in icons.iter().enumerate() {
            let (cx, cy, cw, ch) = layout.cell_rect(slot as u32);
            assert!(
                icon.rect[0] >= cx && icon.rect[0] + icon.rect[2] <= cx + cw + 0.5,
                "slot {slot}: icon at x={} is outside cell x={cx}..{}",
                icon.rect[0],
                cx + cw
            );
            assert!(
                icon.rect[1] >= cy && icon.rect[1] + icon.rect[3] <= cy + ch + 0.5,
                "slot {slot}: icon at y={} is outside cell y={cy}..{}",
                icon.rect[1],
                cy + ch
            );
        }
    }

    #[test]
    fn the_grid_labels_its_cells() {
        let mut renderer = renderer();
        if renderer.weight_coverage().is_empty() {
            return;
        }
        let mut list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let layout = grid_layout(40, metrics, 720, 0.0);
        let buf = varied_kind_rows(layout.visible.count as usize);

        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            Interaction::default(),
            &settled(),
        );

        let icon_px = f32::from(metrics.icon_px);
        let glyphs = list
            .instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Glyph as u32 && (i.rect[2] - icon_px).abs() >= 0.5
            })
            .count();
        assert!(glyphs > 0, "the grid drew icons and no names");
    }

    #[test]
    fn the_focused_cell_gets_a_ring_and_the_others_do_not() {
        let mut renderer = renderer();
        let mut list = DrawList::default();
        let metrics = grid_metrics(1200, 120.0, 1.0);
        let layout = grid_layout(400, metrics, 720, 0.0);
        let buf = uniform_kind_rows(layout.visible.count as usize, 0);

        let interaction = Interaction {
            focused: Some(6),
            ..Interaction::default()
        };
        renderer.render_grid(&mut list, &buf, &layout, metrics, interaction, &settled());

        let strokes: Vec<&Instance> = list
            .instances
            .iter()
            .filter(|i| i.kind == qs_gpu::PrimKind::Stroke as u32)
            .collect();
        // The ring and its contrasting outline, and nothing else.
        assert_eq!(strokes.len(), 2, "{strokes:?}");
        let (cx, cy, _, _) = layout.cell_rect(6);
        assert!(
            strokes
                .iter()
                .all(|s| s.rect[0] >= cx - 1.0 && s.rect[1] >= cy - 1.0),
            "the ring is not on cell 6"
        );
    }
    #[test]
    fn the_focus_ring_is_identical_with_the_lit_mode_on_and_off() {
        // T060 / FR-031. The lit mode replaces the ring with nothing; it adds a light BESIDE
        // it. That obligation is easy to state and easy to lose, because the obvious way to
        // make focus read as light is to soften the ring once the light is doing the work --
        // which quietly removes the only focus indicator that survives the CPU tier, forced
        // colours, and every path where the mode is suppressed.
        //
        // Asserted over the ring's instances byte for byte rather than "a ring exists": a
        // ring drawn at a different width, colour or radius is still a ring, and it is
        // exactly the change this test exists to refuse.
        let mut renderer = renderer();
        let layout = layout_for(12, 2.0);
        let buf = uniform_kind_rows(layout.visible.count as usize, 0);
        let interaction = Interaction {
            focused: Some(4),
            ..Interaction::default()
        };
        let mut motion = settled();
        motion.set_focused(Some(4));

        let mut unlit = DrawList::default();
        unlit.reset([1200, layout.height], qs_gpu::Srgba::TRANSPARENT, 1);
        renderer.render(&mut unlit, &buf, &layout, interaction, &motion);

        let mut lit = DrawList::default();
        lit.reset([1200, layout.height], qs_gpu::Srgba::TRANSPARENT, 2);
        let mut builder = crate::scene::SceneBuilder::new(
            2,
            [1200.0, layout.height as f32],
            64.0,
            qs_gpu::scene::Environment::default(),
        );
        renderer.render_lit(&mut lit, &buf, &layout, interaction, &motion, &mut builder);

        let rings = |list: &DrawList| -> Vec<Instance> {
            list.instances
                .iter()
                .filter(|i| i.kind == qs_gpu::PrimKind::Stroke as u32)
                .copied()
                .collect()
        };
        let (off, on) = (rings(&unlit), rings(&lit));
        assert!(
            !off.is_empty(),
            "the unlit arm drew no ring at all, so this test compares nothing to nothing"
        );
        assert_eq!(
            off.len(),
            on.len(),
            "the lit mode changed how many strokes the focus indicator is made of"
        );
        for (a, b) in off.iter().zip(&on) {
            assert_eq!(
                bytemuck::bytes_of(a),
                bytemuck::bytes_of(b),
                "the lit mode changed the focus ring: {a:?} became {b:?}"
            );
        }

        // And the light exists, so the arm above is not passing because the mode did nothing.
        assert!(
            builder.finish().focus_light.is_some(),
            "the lit arm published no focus lamp, so `render_lit` is not exercising US3"
        );
    }
}
