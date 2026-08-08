//! Glyph rasterization into R8 coverage bitmaps.
//!
//! Research R8 fixes the format: an R8 coverage atlas with **three horizontal subpixel
//! positions**, glyphs up to 48 px rasterized on the CPU and uploaded. Both halves of that
//! matter.
//!
//! *Coverage, not subpixel RGB.* A three-channel LCD-filtered mask would look sharper on a
//! desktop LCD and would be wrong everywhere else -- rotated displays, OLED subpixel
//! layouts, and any composited surface with non-opaque background. One channel is the
//! honest choice and it is also a third of the atlas bandwidth.
//!
//! *Three subpixel positions, not one and not sixteen.* Snapping every glyph to a whole
//! pixel makes text visibly shimmer during a slow scroll, because each glyph jumps a full
//! pixel at a different moment. Quantizing to 1/3 px removes the shimmer at 3x the atlas
//! entries; going finer buys nothing a reader can see and multiplies atlas pressure, which
//! is the exact resource the R8 thrash cliff is about.

use swash::FontRef;
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::zeno::{Format, Vector};

use crate::PxSize;
use crate::fontdb::{FontDb, FontId};

/// Horizontal subpixel quantization. See the module docs.
pub const SUBPIXEL_POSITIONS: u8 = 3;

/// Glyphs larger than this are not atlased -- they would evict a meaningful fraction of
/// the atlas for one character. M0 never produces one (row text tops out near 28 px even
/// at 200% scale), so this is a guard against a future caller, not a live path.
pub const MAX_ATLAS_PX: f32 = 48.0;

/// Identity of one rasterized glyph. This is the atlas key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct GlyphKey {
    pub font: FontId,
    pub glyph_id: u16,
    pub size: PxSize,
    /// Which of the [`SUBPIXEL_POSITIONS`] horizontal phases this is.
    pub subpixel: u8,
}

impl GlyphKey {
    /// Build a key from a glyph's fractional pen position.
    ///
    /// Returns the key together with the **integer** pen position the caller should draw
    /// at. Splitting the position this way is what keeps the fractional part inside the
    /// rasterized bitmap instead of in the vertex data, which is the whole point of having
    /// subpixel variants at all.
    pub fn quantize(font: FontId, glyph_id: u16, size: PxSize, x: f32) -> (Self, f32) {
        let floor = x.floor();
        let fract = x - floor;
        let phase = (fract * f32::from(SUBPIXEL_POSITIONS)).floor() as u8;
        let phase = phase.min(SUBPIXEL_POSITIONS - 1);
        (
            Self {
                font,
                glyph_id,
                size,
                subpixel: phase,
            },
            floor,
        )
    }

    /// The horizontal offset, in pixels, this key's phase represents.
    pub fn subpixel_offset(self) -> f32 {
        f32::from(self.subpixel) / f32::from(SUBPIXEL_POSITIONS)
    }
}

/// An R8 coverage bitmap plus where to put it relative to the pen.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct RasterizedGlyph {
    pub width: u32,
    pub height: u32,
    /// Offset from the pen position to the left edge of the bitmap.
    pub left: i32,
    /// Offset from the baseline to the **top** edge of the bitmap, positive upward.
    pub top: i32,
    /// `width * height` coverage bytes, row-major, no padding.
    pub coverage: Vec<u8>,
    /// The source was a colour bitmap or colour outline and has been flattened to
    /// coverage. Emoji hit this. Reported so the emoji suite's "reported, not gating"
    /// status (decision A-2) has something concrete behind it rather than a shrug.
    pub was_color: bool,
}

impl RasterizedGlyph {
    /// A glyph with no ink -- a space, or a mark that rendered empty. Not an error, and
    /// not something to store in the atlas.
    pub fn is_blank(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Owns `swash`'s scaler cache.
///
/// `&mut self` for the same reason as [`crate::shape::Shaper`]: this holds a mutable cache
/// and belongs to one thread.
pub struct GlyphRaster {
    ctx: ScaleContext,
    /// Glyphs whose rasterization was attempted and produced nothing. Diagnostic only.
    pub blanks: u64,
    pub rasterized: u64,
}

impl std::fmt::Debug for GlyphRaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlyphRaster")
            .field("rasterized", &self.rasterized)
            .field("blanks", &self.blanks)
            .finish()
    }
}

impl Default for GlyphRaster {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphRaster {
    pub fn new() -> Self {
        Self {
            ctx: ScaleContext::new(),
            blanks: 0,
            rasterized: 0,
        }
    }

    /// Rasterize one glyph.
    ///
    /// `None` means the face could not be read or the size is out of range. A glyph that
    /// legitimately has no ink returns `Some` with a zero-sized bitmap -- the caller needs
    /// to tell "nothing to draw" apart from "could not draw", because only one of those is
    /// a problem.
    pub fn rasterize(&mut self, db: &dyn FontDb, key: GlyphKey) -> Option<RasterizedGlyph> {
        let px = key.size.to_f32();
        if !(px.is_finite() && px > 0.0 && px <= MAX_ATLAS_PX) {
            return None;
        }

        let data = db.face_data(key.font)?;
        let font = FontRef::from_index(data.bytes, data.index as usize)?;

        let mut scaler = self
            .ctx
            .builder(font)
            .size(px)
            // Hinting at UI sizes keeps stems on pixel boundaries vertically, which is what
            // stops rows of text looking blurry. It is applied vertically only by swash,
            // so it does not fight the horizontal subpixel positioning above.
            .hint(true)
            .build();

        // Source order is the fallback order swash walks. Colour first so emoji render as
        // something rather than as an empty outline, then plain outlines, then any
        // monochrome strike an old bitmap font might carry.
        let image = Render::new(&[
            Source::ColorOutline(0),
            Source::ColorBitmap(StrikeWith::BestFit),
            Source::Outline,
            Source::Bitmap(StrikeWith::BestFit),
        ])
        .format(Format::Alpha)
        .offset(Vector::new(key.subpixel_offset(), 0.0))
        .render(&mut scaler, key.glyph_id)?;

        let width = image.placement.width;
        let height = image.placement.height;

        if width == 0 || height == 0 {
            self.blanks += 1;
            return Some(RasterizedGlyph::default());
        }

        let expected = (width as usize).checked_mul(height as usize)?;
        // `Format::Alpha` is one byte per pixel. Trusting that without checking would turn
        // a swash behaviour change into an out-of-bounds read in the atlas uploader.
        if image.data.len() < expected {
            return None;
        }

        self.rasterized += 1;
        Some(RasterizedGlyph {
            width,
            height,
            left: image.placement.left,
            top: image.placement.top,
            coverage: image.data,
            was_color: matches!(
                image.content,
                swash::scale::image::Content::Color | swash::scale::image::Content::SubpixelMask
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;
    use crate::fontdb::SystemFontDb;
    use crate::shape::Shaper;
    use crate::{Features, PxSize};
    use std::sync::Arc;

    #[test]
    fn quantization_splits_position_into_integer_and_phase() {
        let f = FontId(0);
        let size = PxSize::new(14.0);

        let (k, x) = GlyphKey::quantize(f, 7, size, 10.0);
        assert_eq!((k.subpixel, x), (0, 10.0));

        let (k, x) = GlyphKey::quantize(f, 7, size, 10.5);
        assert_eq!((k.subpixel, x), (1, 10.0));

        let (k, x) = GlyphKey::quantize(f, 7, size, 10.9);
        assert_eq!((k.subpixel, x), (2, 10.0));

        // The boundary case: a value that rounds up to exactly 3 must clamp, or it becomes
        // a fourth atlas variant that nothing else knows about.
        let (k, _) = GlyphKey::quantize(f, 7, size, 10.99999);
        assert!(k.subpixel < SUBPIXEL_POSITIONS);
    }

    #[test]
    fn quantization_never_exceeds_the_phase_count() {
        let f = FontId(0);
        let size = PxSize::new(14.0);
        for i in 0..10_000 {
            let x = i as f32 * 0.37;
            let (k, floor) = GlyphKey::quantize(f, 1, size, x);
            assert!(k.subpixel < SUBPIXEL_POSITIONS);
            assert!(floor <= x && x - floor < 1.0);
        }
    }

    #[test]
    fn a_real_glyph_rasterizes_to_non_empty_coverage() {
        let db = SystemFontDb::scan();
        if db.is_empty() {
            return;
        }
        let ui = db.ui_font();
        let db: Arc<dyn FontDb> = Arc::new(db);
        let mut shaper = Shaper::new(Arc::clone(&db));
        let run = shaper.shape("A", ui, PxSize::new(24.0), Features::default());
        let Some(glyph) = run.glyphs.first() else {
            return;
        };

        let mut raster = GlyphRaster::new();
        let (key, _) = GlyphKey::quantize(glyph.font, glyph.glyph_id, PxSize::new(24.0), glyph.x);
        let bitmap = raster.rasterize(db.as_ref(), key).unwrap();

        assert!(!bitmap.is_blank(), "a capital A must have ink");
        assert_eq!(
            bitmap.coverage.len(),
            (bitmap.width * bitmap.height) as usize
        );
        assert!(bitmap.coverage.iter().any(|&c| c > 0));
    }

    #[test]
    fn an_oversized_request_is_refused_rather_than_atlased() {
        let db = SystemFontDb::scan();
        if db.is_empty() {
            return;
        }
        let ui = db.ui_font();
        let mut raster = GlyphRaster::new();
        let key = GlyphKey {
            font: ui,
            glyph_id: 1,
            size: PxSize::new(MAX_ATLAS_PX + 1.0),
            subpixel: 0,
        };
        assert!(raster.rasterize(&db, key).is_none());
    }
}
