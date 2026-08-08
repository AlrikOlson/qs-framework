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

/// Identity of one rasterized icon. This is the atlas key.
///
/// `px` is the *device* pixel size of the square the icon occupies -- scale is already
/// folded in. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IconKey {
    pub kind: IconKind,
    pub px: u16,
}

impl IconKey {
    /// Build a key for a logical grid size at a device scale.
    ///
    /// Rounding here rather than at the draw site is deliberate: the rounded value is both
    /// the cache key and the drawn size, so they cannot drift apart and leave the icon
    /// resampled by a fraction of a pixel.
    pub fn new(kind: IconKind, logical_px: f32, scale: f32) -> Self {
        Self {
            kind,
            px: device_px(logical_px, scale),
        }
    }
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

    let path = path_for(key.kind)?;
    // Transform *then* stroke, not the reverse. Stroking on the 20-unit grid and scaling the
    // resulting outline would flatten the curves at grid resolution and then magnify the
    // flattening error; this way every curve is flattened against the pixel grid it will
    // actually be sampled on, which is what "rasterized at exact device resolution" means.
    let path = path.transform(Transform::from_scale(scale, scale))?;
    let stroke = Stroke {
        width: STROKE * scale,
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        ..Stroke::default()
    };
    let outline = path.stroke(&stroke, 1.0)?;

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

/// The path for one kind, on the 20-unit grid.
///
/// Every path stays inside `[1.25, 18.75]` on both axes so that half of the 1.5-unit stroke
/// cannot cross the mask edge and get clipped. Silhouettes are chosen to differ at the
/// *shape* level rather than by a badge on a shared page outline, because a badge is three
/// pixels wide at 20 px and every icon would read as "a document".
fn path_for(kind: IconKind) -> Option<Path> {
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
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    /// Mean coverage, 0.0..=1.0. The share of the icon's box that is ink.
    fn ink(bitmap: &RasterizedGlyph) -> f32 {
        let sum: u64 = bitmap.coverage.iter().map(|&c| u64::from(c)).sum();
        sum as f32 / (bitmap.coverage.len() as f32 * 255.0)
    }

    #[test]
    fn every_kind_rasterizes_to_a_square_of_the_requested_size() {
        for kind in IconKind::ALL {
            for px in [16u16, 20, 30, 40] {
                let bitmap = rasterize(IconKey { kind, px })
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
            let bitmap = rasterize(IconKey { kind, px: 20 }).unwrap();
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
            let bitmap = rasterize(IconKey {
                kind,
                px: px as u16,
            })
            .unwrap();
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
            .map(|&kind| (kind, rasterize(IconKey { kind, px: 20 }).unwrap()))
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
}
