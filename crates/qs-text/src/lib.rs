//! Text shaping, shaped-run caching, glyph rasterization and bidirectional resolution.
//!
//! The boundary this crate defends is that **nothing above it knows what a glyph is**.
//! `qs-ui` asks for a shaped run and gets back positioned glyph ids plus a cluster map;
//! `qs-gpu` asks for a coverage bitmap and gets back an R8 buffer. Neither of them ever
//! touches `rustybuzz` or `swash` directly, which is what makes research R2's "held in
//! reserve: swap in HarfBuzz proper" a real option rather than a comforting sentence.

pub mod bidi;
pub mod cache;
pub mod fontdb;
pub mod raster;
pub mod shape;

#[cfg(target_os = "windows")]
pub mod win_fontdb;

#[cfg(target_os = "macos")]
pub mod mac_fontdb;

#[cfg(all(unix, not(target_os = "macos")))]
pub mod linux_fontdb;

pub use bidi::{BidiLevel, VisualRun, resolve_paragraph};
pub use cache::{CacheStats, RunKey, ShapedRunCache};
pub use fontdb::{FaceData, FaceInfo, FontDb, FontId, FontStyle, FontWeight, SystemFontDb};
pub use raster::{GlyphKey, GlyphRaster, RasterizedGlyph, SUBPIXEL_POSITIONS};
pub use shape::{ShapedGlyph, ShapedRun, Shaper};

/// Font size in pixels, quantized to 1/64 px so it can be a hash key.
///
/// Sizes arrive as `f32` from the density model multiplied by an OS text scale, which
/// means they are almost never exactly representable. Hashing the raw float would make
/// the shaped-run cache miss on values that differ in the last mantissa bit and render
/// identically -- a cache that silently stops working is worse than no cache, because
/// the frame-time regression has no obvious cause.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PxSize(u32);

impl PxSize {
    pub fn new(px: f32) -> Self {
        Self((px.max(0.0) * 64.0).round() as u32)
    }

    pub fn to_f32(self) -> f32 {
        self.0 as f32 / 64.0
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Shaping features that change the output and therefore participate in the cache key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Features {
    /// `tnum` -- fixed-advance digits. The size and modified columns need it so numbers
    /// do not jitter horizontally as rows scroll past (FR-013).
    pub tabular_figures: bool,
}
