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
    /// through the gaps and both read as noise. This is the only *filled* shape in the module.
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

/// What one rasterization is a picture of: a file kind, or a mark drawn over one.
///
/// This is the discriminator that keeps the atlas additive. See [`Emblem`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum IconShape {
    Kind(IconKind),
    Emblem(Emblem),
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
    }
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
