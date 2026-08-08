//! Colour, converted once at draw-list build time.
//!
//! # Why the conversion happens here and not in the shader
//!
//! Tokens are authored as sRGB hex, because that is what a designer can reason about and
//! what every design tool emits. GPUs blend correctly only in linear space, and correct
//! blending of a premultiplied source is `dst * (1 - a) + src`, which requires the colour
//! to already carry its alpha.
//!
//! Doing that conversion per fragment would mean a `pow` per channel per pixel for a value
//! that is constant across the whole primitive. Doing it per instance, on the CPU, at the
//! moment the draw list is built, costs one conversion per *primitive* -- roughly 200 per
//! frame rather than roughly two million. The instance buffer therefore carries
//! premultiplied linear RGBA8 and the shader does no colour maths at all.
//!
//! # Why RGBA8 and not RGBA16F
//!
//! 8 bits of *linear* precision is visibly insufficient in dark greys -- banding shows up
//! exactly where a dark theme lives. The mitigation is that the framebuffer is sRGB
//! (`Bgra8UnormSrgb`), so the hardware converts back on write and the 8-bit quantization
//! happens in perceptual space where it is invisible. The instance colour is only ever an
//! input to blending, never a storage format, so 4 bytes is the right size and the
//! precision argument does not apply.

/// A colour as authored: sRGB components, straight (non-premultiplied) alpha.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Srgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Srgba {
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Parse `#rgb`, `#rrggbb` or `#rrggbbaa`. The leading `#` is optional.
    ///
    /// Returns `None` rather than a default colour: a token file with a typo in it should
    /// fail the build (Principle VII), and silently substituting magenta would let it
    /// through.
    pub fn parse_hex(text: &str) -> Option<Self> {
        let hex = text.strip_prefix('#').unwrap_or(text);
        let byte = |i: usize| -> Option<f32> {
            let s = hex.get(i..i + 2)?;
            Some(f32::from(u8::from_str_radix(s, 16).ok()?) / 255.0)
        };
        let nibble = |i: usize| -> Option<f32> {
            let s = hex.get(i..i + 1)?;
            let v = u8::from_str_radix(s, 16).ok()?;
            Some(f32::from(v * 17) / 255.0)
        };

        match hex.len() {
            3 => Some(Self::new(nibble(0)?, nibble(1)?, nibble(2)?, 1.0)),
            4 => Some(Self::new(nibble(0)?, nibble(1)?, nibble(2)?, nibble(3)?)),
            6 => Some(Self::new(byte(0)?, byte(2)?, byte(4)?, 1.0)),
            8 => Some(Self::new(byte(0)?, byte(2)?, byte(4)?, byte(6)?)),
            _ => None,
        }
    }

    /// Relative luminance per WCAG 2.x. Used by the build-time contrast gate.
    pub fn relative_luminance(self) -> f32 {
        let l = |c: f32| srgb_to_linear(c);
        0.2126 * l(self.r) + 0.7152 * l(self.g) + 0.0722 * l(self.b)
    }

    /// WCAG contrast ratio against `other`, in `1.0..=21.0`.
    ///
    /// Both colours are assumed opaque. A translucent foreground has no single contrast
    /// ratio -- it depends what is behind it -- so the token pair list composites first
    /// and passes opaque colours here.
    pub fn contrast_ratio(self, other: Self) -> f32 {
        let a = self.relative_luminance();
        let b = other.relative_luminance();
        let (hi, lo) = if a > b { (a, b) } else { (b, a) };
        (hi + 0.05) / (lo + 0.05)
    }

    /// Composite `self` over an opaque `background`, yielding an opaque colour.
    pub fn over(self, background: Self) -> Self {
        let a = self.a.clamp(0.0, 1.0);
        Self {
            r: self.r * a + background.r * (1.0 - a),
            g: self.g * a + background.g * (1.0 - a),
            b: self.b * a + background.b * (1.0 - a),
            a: 1.0,
        }
    }

    /// Convert to the instance-buffer representation: premultiplied, linear, RGBA8,
    /// packed little-endian as `0xAABBGGRR` so it maps to a WGSL `u32` unpacked with
    /// `unpack4x8unorm`.
    pub fn to_premul_linear_rgba8(self) -> u32 {
        let a = self.a.clamp(0.0, 1.0);
        let q = |c: f32| -> u32 {
            // Convert to linear, then premultiply. The order matters: premultiplying in
            // sRGB and then converting would apply the transfer function to the alpha as
            // well, which darkens every translucent edge. It is a subtle enough error that
            // it usually ships.
            let linear = srgb_to_linear(c.clamp(0.0, 1.0)) * a;
            (linear * 255.0 + 0.5) as u32 & 0xFF
        };
        q(self.r) | (q(self.g) << 8) | (q(self.b) << 16) | (((a * 255.0 + 0.5) as u32 & 0xFF) << 24)
    }
}

/// The sRGB electro-optical transfer function.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Its inverse. Used by the CPU rasterizer, which composites in linear and writes sRGB.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn hex_parsing_covers_the_authored_forms() {
        assert_eq!(
            Srgba::parse_hex("#fff"),
            Some(Srgba::new(1.0, 1.0, 1.0, 1.0))
        );
        assert_eq!(
            Srgba::parse_hex("ffffff"),
            Some(Srgba::new(1.0, 1.0, 1.0, 1.0))
        );
        assert_eq!(
            Srgba::parse_hex("#000000"),
            Some(Srgba::new(0.0, 0.0, 0.0, 1.0))
        );
        let half = Srgba::parse_hex("#00000080").unwrap();
        assert!((half.a - 0.502).abs() < 0.01);
    }

    #[test]
    fn a_malformed_token_is_rejected_not_defaulted() {
        assert_eq!(Srgba::parse_hex("#gg0000"), None);
        assert_eq!(Srgba::parse_hex("#12345"), None);
        assert_eq!(Srgba::parse_hex(""), None);
    }

    #[test]
    fn contrast_matches_the_wcag_reference_values() {
        let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
        let black = Srgba::new(0.0, 0.0, 0.0, 1.0);
        assert!((white.contrast_ratio(black) - 21.0).abs() < 0.01);
        assert!((white.contrast_ratio(white) - 1.0).abs() < 0.01);
        // The ratio is symmetric -- a foreground/background swap must not change it, or
        // the contrast gate would depend on the order pairs happen to be listed in.
        assert_eq!(white.contrast_ratio(black), black.contrast_ratio(white));
    }

    #[test]
    fn premultiplication_happens_after_linearization() {
        // A 50%-alpha mid-grey. If the implementation premultiplied in sRGB space, the
        // RGB channels would come out at roughly half of 0.5^2.2 -- noticeably darker.
        let c = Srgba::new(0.5, 0.5, 0.5, 0.5);
        let packed = c.to_premul_linear_rgba8();
        let r = packed & 0xFF;
        let expected = (srgb_to_linear(0.5) * 0.5 * 255.0 + 0.5) as u32;
        assert_eq!(r, expected);
    }

    #[test]
    fn a_fully_transparent_colour_premultiplies_to_zero() {
        assert_eq!(Srgba::TRANSPARENT.to_premul_linear_rgba8(), 0);
        let invisible_white = Srgba::new(1.0, 1.0, 1.0, 0.0);
        assert_eq!(
            invisible_white.to_premul_linear_rgba8(),
            0,
            "premultiplied transparent must be all zero, or additive blending shows ghosts"
        );
    }

    #[test]
    fn transfer_functions_round_trip() {
        for i in 0..=100 {
            let c = i as f32 / 100.0;
            assert!((linear_to_srgb(srgb_to_linear(c)) - c).abs() < 1e-4);
        }
    }

    #[test]
    fn compositing_over_an_opaque_background_stays_opaque() {
        let fg = Srgba::new(1.0, 0.0, 0.0, 0.5);
        let bg = Srgba::new(0.0, 0.0, 1.0, 1.0);
        let out = fg.over(bg);
        assert_eq!(out.a, 1.0);
        assert!((out.r - 0.5).abs() < 1e-6);
        assert!((out.b - 0.5).abs() < 1e-6);
    }
}
