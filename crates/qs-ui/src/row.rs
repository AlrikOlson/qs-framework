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
use qs_gpu::{AtlasEntry, GlyphAtlas, IconKey, IconKind, PendingUpload};
use qs_text::{
    Features, FontDb, FontId, GlyphKey, GlyphRaster, PxSize, ShapedRun, ShapedRunCache, Shaper,
};

use crate::motion::InteractionMotion;
use crate::recycler::ViewportLayout;
use crate::row_source::{LoadState, RowBuf, RowFlags, RowView};
use crate::tokens::{Tokens, TypeRole, role};

/// Column geometry, in physical pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Columns {
    /// Space between the viewport edge and the row's selection region. This is what makes
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
    pub fn for_width(width: f32, scale: f32, tokens: &Tokens) -> Self {
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
        let size = px(72.0);
        let modified = px(132.0);
        let kind = px(96.0);

        let fixed = gutter * 2.0 + padding * 2.0 + rail + icon + size + modified + kind + gap * 4.0;
        // A window narrow enough to squeeze the name column to nothing is a real state --
        // users do drag windows that small -- and the correct behaviour is a name column
        // that is merely tiny, not one with a negative width.
        let name = (width - fixed).max(px(40.0));

        Self {
            gutter,
            rail,
            icon,
            name,
            size,
            modified,
            kind,
            gap,
            padding,
        }
    }

    /// Left edge of the row's selection region: inside the gutter.
    pub fn content_x(&self) -> f32 {
        self.gutter
    }

    /// Width of the selection region for a viewport of `width`.
    pub fn content_width(&self, width: f32) -> f32 {
        (width - self.gutter * 2.0).max(1.0)
    }

    /// `x` of the name column's left edge.
    pub fn name_x(&self) -> f32 {
        self.gutter + self.padding + self.rail + self.gap + self.icon + self.gap
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

/// Unix nanoseconds to a fixed-width local-ish timestamp.
///
/// Deliberately not locale-aware and deliberately not `chrono`. M0 needs a *stable*,
/// fixed-width string so the modified column's width and the golden images are
/// reproducible; a locale-dependent format would make a reference image captured in one
/// region fail in another, which is a golden-image suite people learn to ignore.
pub fn format_mtime(unix_nanos: i64) -> String {
    let secs = unix_nanos.div_euclid(1_000_000_000);
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

/// Which row the pointer and keyboard are on.
///
/// Deliberately *not* carried on [`RowView`]. Hover, focus and selection are properties of
/// the **view**, not of the data: two panes showing the same directory have different
/// focused rows, and a `RowSource` that had to know about them could not be shared. The
/// source stays pure data and the view supplies interaction state at draw time.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Interaction {
    /// Logical corpus index under the pointer.
    pub hovered: Option<u64>,
    /// Logical corpus index with keyboard focus.
    pub focused: Option<u64>,
    /// Logical corpus index that is selected. M0 has single selection only; real selection
    /// semantics (surviving navigation and filtering) are M1.
    pub selected: Option<u64>,
    /// Logical corpus index the pointer is currently held down on.
    ///
    /// Deliberately has no [`RowFlags`] bit, unlike the three above. Those three exist as
    /// flags because a row can carry them from the *data* side too. Press is purely a
    /// presentation state with no data counterpart, and what the renderer reads is not its
    /// boolean but its animated intensity, which comes from
    /// [`InteractionMotion`](crate::motion::InteractionMotion) rather than from here.
    pub pressed: Option<u64>,
}

impl Interaction {
    /// The state flags that apply to `index`.
    fn flags_for(self, index: u64) -> RowFlags {
        let mut flags = RowFlags::EMPTY;
        if self.hovered == Some(index) {
            flags = flags | RowFlags::IS_HOVERED;
        }
        if self.focused == Some(index) {
            flags = flags | RowFlags::IS_FOCUSED;
        }
        if self.selected == Some(index) {
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

    pub fn cache_stats(&self) -> qs_text::CacheStats {
        self.cache.stats()
    }

    /// Drop everything size-dependent. Called on a density, scale or text-scale change.
    pub fn invalidate_metrics(&mut self) {
        self.cache.clear();
        self.atlas.clear();
    }

    pub fn take_uploads(&mut self) -> Vec<PendingUpload> {
        self.atlas.take_uploads()
    }

    /// Build one frame's draw list.
    ///
    /// # Three layers, in this order, and the order is the point
    ///
    /// 1. **Surface** -- zebra banding, hover and press, per row.
    /// 2. **Selection** -- one region, hoisted out of the row loop.
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
        interaction: Interaction,
        motion: &InteractionMotion,
    ) {
        qs_gpu::affinity::assert_ui_thread("ListRenderer::render");

        self.cache.begin_frame();
        self.atlas.begin_frame();
        self.glyphs_dropped = 0;
        self.icons_dropped = 0;
        self.icons = IconCache {
            px: qs_gpu::icon::device_px(qs_gpu::icon::GRID, layout.scale),
            ..IconCache::default()
        };

        let columns = Columns::for_width(layout.width as f32, layout.scale, &self.tokens);

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

        // Resolve the icons the visible rows actually need, before a single glyph is asked
        // for. This is a priority decision about the per-frame upload bound, and it was
        // made by rendering a cold frame and looking at it: rows are drawn top-down, each
        // asking for its icon and then its text, so the text of the first few rows spent the
        // CPU tier's entire 64-upload budget and the bottom half of the list came up with no
        // icon *and* no name. Icons win that race on merit -- there are at most nine of
        // them, each serves every row of its kind, and once resident they never cost
        // anything again, whereas glyphs keep arriving as the user scrolls.
        //
        // Still only the kinds that are actually on screen: resolving all nine unconditionally
        // would spend uploads on icons nobody can see.
        for row in buf.rows() {
            self.icon_entry(icon_for(row));
        }

        // Layer 1: the surface.
        //
        // Banding is drawn full-bleed and the *states* are drawn inset. Banding is a
        // property of the surface -- it should run to the window edge like ruled paper.
        // Hover, press and selection are properties of an *object*, and an object needs an
        // edge: an inset rounded region reads as a thing sitting on the list, where a
        // full-bleed stripe reads as a table row.
        let region = StateRegion::new(&self.tokens, &columns, layout);
        let band = self.tokens.color("surface/row-alt");
        let wash = self.tokens.color("surface/row-hover");
        for slot in 0..buf.len() as u32 {
            let top = layout.row_top(slot);
            let index = layout.visible.first + u64::from(slot);
            if index % 2 == 1 {
                list.instances.push(Instance::rect(
                    0.0,
                    top,
                    layout.width as f32,
                    layout.row_height as f32,
                    0.0,
                    band,
                ));
            }
            // Hover and press are the same wash, layered rather than blended: a press
            // deepens whatever is already beneath it, which is what makes it read as the
            // same object being pushed instead of as a second colour arriving.
            for alpha in [motion.hover_alpha(index), motion.press_alpha(index)] {
                if alpha > 0.0 {
                    list.instances
                        .push(region.instance(top, at_alpha(wash, alpha)));
                }
            }
        }

        // Layer 2: the selection region, hoisted -- see this function's doc comment.
        self.draw_selection(list, layout, &columns, &region, motion);

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

        list.stats.rows_laid_out = buf.len() as u32;
        list.stats.shaped_runs_new = self.cache.stats().new_this_frame;
        list.stats.glyphs_rasterized = self.atlas.stats().uploaded_this_frame;
        list.stats.glyphs_dropped = self.glyphs_dropped;
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
        interaction: Interaction,
        type_roles: RowType,
        baseline: f32,
    ) {
        let height = layout.row_height as f32;
        // Data flags from the source, view flags from the caller.
        let flags = row.flags | interaction.flags_for(index);
        let focused = flags.contains(RowFlags::IS_FOCUSED);

        let radius = self.tokens.radius(crate::tokens::radius::ROW) * layout.scale.max(0.1);
        let region_x = columns.content_x();
        let region_w = columns.content_width(layout.width as f32);

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
            let inset = self.tokens.space(crate::tokens::space::XS) * scale;

            let (fx, fy) = (region_x, top + inset);
            let (fw, fh) = (region_w, (height - inset * 2.0).max(1.0));

            list.instances.push(Instance::stroke(
                fx,
                fy,
                fw,
                fh,
                radius,
                ring_w + outline_w * 2.0,
                self.tokens.color("border/focus-outline"),
            ));
            list.instances.push(Instance::stroke(
                fx + outline_w,
                fy + outline_w,
                (fw - outline_w * 2.0).max(1.0),
                (fh - outline_w * 2.0).max(1.0),
                (radius - outline_w).max(0.0),
                ring_w,
                self.tokens.color("border/focus"),
            ));
        }

        let hidden = row.flags.contains(RowFlags::IS_HIDDEN);
        let name_color = fade(self.tokens.color("content/primary"), hidden);

        // Icon slot: a kind-based vector icon rasterized into the same atlas as the text
        // (`qs_gpu::icon`), so it is a coverage mask tinted at draw time exactly like a
        // glyph. Directories get the accent, files the tertiary content colour -- one
        // rasterization serves both, because only the tint differs.
        //
        // Dimmed for hidden files, which the placeholder square was not. A hidden file's
        // name fades and its icon did not, which read as two rows rather than one.
        if let Some(entry) = self.icon_entry(icon_for(row)) {
            let size = entry.width as f32;
            list.instances.push(Instance::glyph(
                (columns.gutter
                    + columns.padding
                    + columns.rail
                    + columns.gap
                    + (columns.icon - size) * 0.5)
                    .round(),
                (top + (height - size) * 0.5).round(),
                size,
                entry.height as f32,
                entry.uv,
                fade(
                    if row.flags.contains(RowFlags::IS_DIR) {
                        self.tokens.color("border/focus")
                    } else {
                        self.tokens.color("content/tertiary")
                    },
                    hidden,
                ),
            ));
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
            self.face_for(type_roles.primary),
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
            for (x, w) in [
                (columns.size_x(), columns.size * 0.7),
                (columns.modified_x(), columns.modified * 0.85),
                (columns.kind_x(), columns.kind * 0.6),
            ] {
                list.instances
                    .push(Instance::rect(x, bar_y, w, bar_h, bar_h * 0.5, bar));
            }
            return;
        }

        let secondary = fade(self.tokens.color("content/secondary"), hidden);
        let tertiary = fade(self.tokens.color("content/tertiary"), hidden);

        // Size and modified use tabular figures so a column of numbers forms a grid rather
        // than a ragged edge as rows scroll past (FR-013).
        let tabular = Features {
            tabular_figures: true,
        };

        if !row.flags.contains(RowFlags::IS_DIR) {
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

    /// Draw the selection region and its status rail, wherever the morph currently puts it.
    fn draw_selection(
        &mut self,
        list: &mut DrawList,
        layout: &ViewportLayout,
        columns: &Columns,
        region: &StateRegion,
        motion: &InteractionMotion,
    ) {
        let Some(draw) = motion.selection_draw(layout.visible.count) else {
            return;
        };

        // [`ViewportLayout::row_top`] generalised to a slot that may be negative or past the
        // end: mid-morph the region sits *between* two rows, and either endpoint may have
        // scrolled off. The arithmetic stays relative to the first visible row for exactly
        // the reason `row_top` gives -- the absolute content offset of row 999,999 is 28
        // million and does not survive an f32.
        let height = layout.row_height as f32;
        let slot = draw.row as f64 - layout.visible.first as f64;
        let top = (slot * f64::from(layout.row_height) - layout.first_row_offset) as f32
            + draw.offset_rows * height;

        if top + height <= 0.0 || top >= layout.height as f32 {
            return;
        }

        list.instances.push(region.instance(
            top,
            at_alpha(self.tokens.color("surface/row-selected"), draw.alpha),
        ));

        // Status rail. Nothing at M0 sets a VCS status, so this draws only for selection --
        // but the slot is reserved so M1's rail does not reflow every row. It travels with
        // the region because it is one indicator, not two.
        list.instances.push(Instance::rect(
            columns.gutter + columns.padding,
            top + height * 0.15,
            columns.rail,
            height * 0.7,
            columns.rail * 0.5,
            at_alpha(self.tokens.color("border/focus"), draw.alpha),
        ));
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
            kind,
            px: self.icons.px,
        };
        let entry = self
            .atlas
            .get_or_render(key, false, qs_gpu::icon::rasterize);
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

            let Some(entry) = self
                .atlas
                .get_or_insert(self.db.as_ref(), &mut self.raster, key)
            else {
                // Three reasons the atlas can decline, and only two of them are failures:
                // deferred by the per-frame upload bound, or the face could not raster the
                // glyph. The third is a blank -- a space -- which is text that is supposed
                // to be invisible. Counting blanks here made this number nonzero on every
                // line that contains a space, which is why it has to ask.
                if !self.atlas.is_blank(key) {
                    self.glyphs_dropped += 1;
                }
                continue;
            };
            if entry.width == 0 || entry.height == 0 {
                continue;
            }

            list.instances.push(Instance::glyph(
                integer_x + entry.left as f32,
                (baseline + glyph.y).round() - entry.top as f32,
                entry.width as f32,
                entry.height as f32,
                entry.uv,
                color,
            ));
        }
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
        // Inset vertically by one space step so consecutive selected rows read as separate
        // objects rather than one tall block.
        let inset = tokens.space(crate::tokens::space::XS) * scale;
        Self {
            x: columns.content_x(),
            width: columns.content_width(layout.width as f32),
            inset,
            height: (layout.row_height as f32 - inset * 2.0).max(1.0),
            radius: tokens.radius(crate::tokens::radius::ROW) * scale,
        }
    }

    fn instance(self, row_top: f32, color: Srgba) -> Instance {
        Instance::rect(
            self.x,
            row_top + self.inset,
            self.width,
            self.height,
            self.radius,
            color,
        )
    }
}

/// Scale a token colour's opacity for an animated state.
///
/// Multiplicative, not absolute: a token that is already translucent stays in proportion, so
/// an animation cannot make a state *more* opaque than the design system says it is.
fn at_alpha(color: Srgba, factor: f32) -> Srgba {
    Srgba {
        a: color.a * factor.clamp(0.0, 1.0),
        ..color
    }
}

/// Hidden files render dimmed rather than absent.
fn fade(color: Srgba, hidden: bool) -> Srgba {
    if hidden {
        Srgba {
            a: color.a * 0.55,
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
fn icon_for(row: &RowView) -> IconKind {
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
        assert_eq!(format_mtime(86_400 * 1_000_000_000), "1970-01-02 00:00");
        // 2026-08-07T00:00:00Z
        assert_eq!(
            format_mtime(1_786_060_800 * 1_000_000_000),
            "2026-08-07 00:00"
        );
        // A leap-day, because the civil-date conversion is where those go wrong.
        assert_eq!(
            format_mtime(1_709_164_800 * 1_000_000_000),
            "2024-02-29 00:00"
        );

        let width = format_mtime(0).len();
        for nanos in [
            0i64,
            1_000_000_000,
            1_786_060_800_000_000_000,
            -86_400_000_000_000,
        ] {
            assert_eq!(format_mtime(nanos).len(), width, "nanos {nanos}");
        }
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
        }
    }

    /// Motion at rest: nothing animating, so a draw list is a function of the interaction
    /// state alone. Tests about icons, text and columns want this; the tests about motion
    /// build their own.
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

    /// The inset rounded state regions in a draw list: hover, press and selection all share
    /// this geometry, which is exactly why `StateRegion` is one type.
    fn state_regions(list: &DrawList, layout: &ViewportLayout) -> Vec<Instance> {
        let tokens = Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let columns = Columns::for_width(layout.width as f32, layout.scale, &tokens);
        let width = columns.content_width(layout.width as f32);
        list.instances
            .iter()
            .filter(|i| {
                i.kind == qs_gpu::PrimKind::Rect as u32
                    && i.radius > 0.0
                    && (i.rect[2] - width).abs() < 0.5
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
    }

    #[test]
    fn the_selection_region_travels_between_the_two_rows_during_a_morph() {
        // UXDD 10.3 calls a selection change a geometry morph. The observable form of that
        // is one region at an intermediate position -- not two regions cross-fading, and not
        // one region that teleports.
        let mut renderer = renderer();
        let buf = plain_rows(8);
        let layout = layout_for(8, 1.0);
        let interaction = Interaction {
            selected: Some(6),
            ..Interaction::default()
        };

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
    fn the_selection_region_is_drawn_beneath_every_row_of_content() {
        // The reason `render` has three layers at all. Mid-morph the region sits between two
        // rows, and drawn inside the row loop it would paint over the text of every row it
        // had already passed. Emitting it before any content makes that impossible rather
        // than unlikely.
        let mut renderer = renderer();
        let buf = plain_rows(8);
        let layout = layout_for(8, 1.0);
        let interaction = Interaction {
            selected: Some(6),
            ..Interaction::default()
        };
        let mut motion = InteractionMotion::new(crate::motion::MotionPreference::Full);
        motion.set_selected(Some(2));
        while motion.advance(1.0 / 120.0) {}
        motion.set_selected(Some(6));
        motion.advance(0.060);

        let list = draw(&mut renderer, &buf, &layout, interaction, &motion);
        let tokens = Tokens::embedded(crate::tokens::Theme::Dark).unwrap();
        let columns = Columns::for_width(layout.width as f32, layout.scale, &tokens);
        let region_w = columns.content_width(layout.width as f32);

        let last_region = list
            .instances
            .iter()
            .rposition(|i| {
                i.kind == qs_gpu::PrimKind::Rect as u32
                    && i.radius > 0.0
                    && (i.rect[2] - region_w).abs() < 0.5
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

    #[test]
    fn under_reduced_motion_the_selection_is_fully_painted_on_the_first_frame() {
        // "Instant" has to mean instant in the draw list, not merely in the plan.
        let mut renderer = renderer();
        let buf = plain_rows(6);
        let layout = layout_for(6, 1.0);
        let interaction = Interaction {
            selected: Some(4),
            ..Interaction::default()
        };
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
        assert_eq!(icon_for(&stub), IconKind::Generic);

        let mut dir_stub = stub.clone();
        dir_stub.flags = RowFlags::IS_DIR;
        assert_eq!(icon_for(&dir_stub), IconKind::Folder);
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
        assert_eq!(icon_for(&row), IconKind::Folder);
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
                .map(icon_for)
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
        let distinct_kinds: std::collections::HashSet<_> =
            buf.rows().iter().map(icon_for).collect();
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
}
