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

    /// The same premultiplied linear RGBA the instance buffer carries, but as the four
    /// floats a shader would see after `unpack4x8unorm`.
    ///
    /// Quantized through 8 bits on purpose, even though the destination is `f32`. A
    /// gradient carries one stop in [`Instance::color`](crate::frame::Instance::color) and
    /// the other in [`Instance::uv`](crate::frame::Instance::uv), and the first of those is
    /// an RGBA8. Storing the second at full float precision would make the two ends of one
    /// ramp disagree about how finely a colour can be expressed — an asymmetry with no
    /// visual benefit that both tiers would then have to reproduce identically to stay in
    /// parity. Quantizing here means there is only one precision in the system.
    pub fn to_premul_linear_f32(self) -> [f32; 4] {
        let packed = self.to_premul_linear_rgba8();
        let byte = |shift: u32| f32::from(((packed >> shift) & 0xFF) as u8) / 255.0;
        [byte(0), byte(8), byte(16), byte(24)]
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

/// Cube root that keeps the sign, which is what `pow` does not.
///
/// Written this way rather than as `f32::cbrt` because `shaders/instance.wgsl` has no
/// `cbrt` and reaches the same value as `sign(x) * pow(abs(x), 1/3)`. The two agree
/// mathematically; spelling them the same way keeps the WGSL a transcription of this
/// rather than a second opinion about it.
fn signed_cbrt(x: f32) -> f32 {
    x.signum() * x.abs().powf(1.0 / 3.0)
}

/// Linear-light sRGB to Oklab: perceptual lightness plus two opponent axes.
///
/// Public because it is the space a **gradient** interpolates in. The palette is authored
/// in OKLCH (see [`Oklch`]) precisely so hue and chroma stay perceptually stable, and a
/// ramp between two of those stops that lerps in linear sRGB drifts through a duller hue
/// at its midpoint — the endpoints stay right, which is what makes it hard to notice.
/// Oklab rather than OKLCH because interpolation wants Cartesian axes: lerping a hue
/// *angle* has to choose a direction round the circle, and for a two-stop UI ramp the
/// short way round is not always the one that avoids leaving the gamut.
pub fn linear_rgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let l = signed_cbrt(0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b);
    let m = signed_cbrt(0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b);
    let s = signed_cbrt(0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b);
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

/// Oklab back to linear-light sRGB, **unclamped**.
///
/// Unclamped because the caller decides what an out-of-gamut result means. A gradient
/// clamps, because the Oklab straight line between two in-gamut stops can leave sRGB in
/// the middle when the stops are far apart in hue, and the alternative — rescaling chroma
/// per fragment — is a gamut-mapping bisection no fragment shader should run.
pub fn oklab_to_linear_rgb(lab: [f32; 3]) -> [f32; 3] {
    let [lightness, a, b] = lab;
    let l = (lightness + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
    let m = (lightness - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
    let s = (lightness - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

// -- dither ---------------------------------------------------------------------------
//
// A ramp is quantized twice: once into the 8-bit framebuffer, and once by the eye, which
// is very good at finding a contour where a large region shares one value. `material-library`
// shipped that failure and had to revert two row ramps to flat fills over it -- four output
// levels spread across three hundred and sixty pixels read as four hard vertical edges.
//
// The fix is not more bits. It is to spend the bits differently: offset each pixel by less
// than half a level before it is quantized, so the contour becomes grain instead of an edge
// and the *average* over any small neighbourhood is the value the ramp actually asked for.
//
// Three properties are load-bearing, and each one rules out the obvious alternative:
//
// - The seed is the **framebuffer pixel**, not the shape-local position. Local position is
//   exact only when the rect lands on an integer, and off-grid geometry is exactly what the
//   parity fixtures carry; a `floor` near a boundary would pick different pixels on the two
//   tiers and the dither would disagree where nothing is wrong. It is also the right reading
//   of the effect: dither belongs to the display grid, where film grain would belong to the
//   surface.
// - The hash is **integer**. The usual `fract(52.98 * fract(...))` is one float rounding away
//   from landing on the other side of a discontinuity, and the two tiers would then disagree
//   by a whole step on scattered pixels. `u32` wrapping multiplies and shifts are exact and
//   identical in WGSL and in Rust.
// - The amplitude follows the **sRGB encode**, not linear light. The framebuffer is
//   `Bgra8UnormSrgb`, so the hardware encodes before it quantizes and half a level lives at
//   `d(linear)/d(srgb) / 2 / 255`. A constant linear amplitude would over-dither the brights
//   and under-dither the darks, which is precisely where a dark theme lives.

/// A 32-bit hash of one framebuffer pixel.
///
/// Exact on both tiers by construction: every operation is a wrapping `u32` multiply or a
/// shift, and WGSL defines unsigned overflow to wrap the same way [`u32::wrapping_mul`] does.
/// The constants are the usual odd 32-bit mixers; nothing here depends on which ones.
#[must_use]
pub fn dither_hash(x: u32, y: u32) -> u32 {
    let mut h = x.wrapping_mul(0x27d4_eb2d) ^ y.wrapping_mul(0x1656_67b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    h
}

/// The dither offset for one framebuffer pixel, in `[-0.5, 0.5)`.
///
/// Twenty-four bits over two to the twenty-fourth: both the conversion and the division are
/// exact in `f32`, so the shader and this reach the same float rather than nearly the same
/// one. Taking the *high* bits matters -- the low bits of a multiply-shift mixer are the
/// weakest, and a dither that correlates with pixel parity draws a checkerboard.
#[must_use]
pub fn dither_noise(x: u32, y: u32) -> f32 {
    (dither_hash(x, y) >> 8) as f32 / 16_777_216.0 - 0.5
}

/// `2.4 / 1.055` -- the slope of the sRGB encode, factored out of [`dither_amplitude`].
pub const DITHER_SLOPE: f32 = 2.274_881_5;

/// `1.4 / 2.4` -- the exponent that turns a linear value into that slope's argument.
pub const DITHER_EXPONENT: f32 = 0.583_333_3;

/// Where the sRGB encode stops being a power law, in linear light.
pub const DITHER_TOE: f32 = 0.003_130_8;

/// `1 / 12.92` -- the encode's constant slope below [`DITHER_TOE`].
pub const DITHER_TOE_SLOPE: f32 = 0.077_399_38;

/// One least-significant bit of the *output* encoding, expressed in linear light.
///
/// The full amplitude, not half: [`dither_noise`] supplies the `±0.5`.
///
/// The sRGB transfer function is `l = ((s + 0.055) / 1.055)^2.4` above the toe, so
/// `dl/ds = (2.4 / 1.055) * l^(1.4/2.4)`; below it the encode is the straight line `s / 12.92`
/// and the slope is constant. The branch is the same one [`linear_to_srgb`] carries, and it
/// is here rather than approximated away because the power law keeps falling below the toe
/// while the truth does not: at `l = 0.002` a single expression reaches 78% of one level, and
/// the darkest few per cent of the range is exactly where a dark theme's ramps live.
#[must_use]
pub fn dither_amplitude(linear: f32) -> f32 {
    let linear = linear.max(0.0);
    let slope = if linear <= DITHER_TOE {
        DITHER_TOE_SLOPE
    } else {
        DITHER_SLOPE * linear.powf(DITHER_EXPONENT)
    };
    slope / 255.0
}

/// Straight linear-light RGB, offset by less than half an output level.
///
/// Alpha is deliberately not dithered. Alpha is coverage, and a gradient's coverage is what
/// antialiases its edge -- noise on it would fray every boundary the shape has, to fix
/// banding that lives in the interior.
#[must_use]
pub fn dithered(rgb: [f32; 3], x: u32, y: u32) -> [f32; 3] {
    let noise = dither_noise(x, y);
    let one = |c: f32| (c + noise * dither_amplitude(c)).clamp(0.0, 1.0);
    [one(rgb[0]), one(rgb[1]), one(rgb[2])]
}

/// A colour in OKLCH: perceptual lightness, chroma, and hue angle in degrees.
///
/// This is the space the token file authors in, and the reason is that its three axes are
/// separable in a way sRGB's are not. Two greys with the same `h` are the same grey at
/// different lightnesses; two accents at the same `c` are equally colourful. Neither
/// statement is true of hex, which is why a hand-picked palette drifts in hue without
/// anybody being able to point at the value that is wrong.
///
/// # Gamut
///
/// OKLCH is a cylinder and sRGB is not. For most `(l, h)` there is a chroma beyond which no
/// sRGB colour exists, and that ceiling varies by a factor of five across the lightness
/// range of a single hue. [`Oklch::to_srgb`] handles that by reducing chroma, never by
/// moving lightness or hue: an out-of-gamut accent should become a less colourful version
/// of the same colour, not a different one. That is the simple form of the CSS Color 4
/// gamut-mapping algorithm — the full one minimizes ΔE against the clipped colour, which
/// buys a fraction of a JND on colours a UI palette has no reason to ask for.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Oklch {
    /// Perceptual lightness, `0.0` (black) to `1.0` (white).
    pub l: f32,
    /// Chroma. Unbounded in principle; sRGB tops out near `0.33`.
    pub c: f32,
    /// Hue angle in degrees.
    pub h: f32,
}

impl Oklch {
    pub const fn new(l: f32, c: f32, h: f32) -> Self {
        Self { l, c, h }
    }

    /// Linear-light sRGB primaries, **unclamped**: a component outside `0.0..=1.0` means
    /// this colour is outside the sRGB gamut. Kept private because an unclamped triple is
    /// only meaningful as an input to the gamut test.
    fn to_linear_rgb(self) -> [f32; 3] {
        // OKLCH is Oklab in polar form, so the conversion is one coordinate change away
        // from the shared matrix. Keeping a second copy of the matrix here is how the two
        // would drift.
        let (sin, cos) = self.h.to_radians().sin_cos();
        oklab_to_linear_rgb([self.l, self.c * cos, self.c * sin])
    }

    /// Whether an exact sRGB colour exists for this lightness, chroma and hue.
    ///
    /// The tolerance absorbs the round trip's own float error, so a colour sitting exactly
    /// on the gamut boundary does not read as outside it depending on the last bit.
    pub fn in_srgb_gamut(self) -> bool {
        self.to_linear_rgb()
            .iter()
            .all(|v| (-1e-6..=1.0 + 1e-6).contains(v))
    }

    /// The largest chroma that stays inside sRGB at this lightness and hue.
    ///
    /// Found by bisection rather than solved, because the boundary is the intersection of
    /// three cubics with the unit cube and has no useful closed form. Thirty iterations
    /// takes the bracket below `f32`'s resolution, and the whole palette is resolved once
    /// at load time.
    pub fn max_chroma(l: f32, h: f32) -> f32 {
        let (mut lo, mut hi) = (0.0_f32, 0.5_f32);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if Self::new(l, mid, h).in_srgb_gamut() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Convert to opaque sRGB, reducing chroma until the colour fits the gamut.
    pub fn to_srgb(self) -> Srgba {
        let mapped = if self.in_srgb_gamut() {
            self
        } else {
            Self::new(self.l, Self::max_chroma(self.l, self.h), self.h)
        };
        let [r, g, b] = mapped.to_linear_rgb();
        // The clamp catches the residual float error at the boundary only; the chroma
        // reduction above is what actually keeps this from being a hue-shifting clip.
        let f = |v: f32| linear_to_srgb(v.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        Srgba::new(f(r), f(g), f(b), 1.0)
    }
}

impl Srgba {
    /// Measure this colour in OKLCH. Alpha is dropped: OKLCH describes a colour, not a
    /// compositing operation.
    ///
    /// Hue is meaningless as chroma approaches zero, and near-neutral colours in 8-bit sRGB
    /// have a lot of hue noise for that reason — one least-significant bit is an 18° hue
    /// error at `c = 0.003` and a 1° error at `c = 0.014`. Read `h` only where `c` is large
    /// enough to carry it.
    pub fn to_oklch(self) -> Oklch {
        let (r, g, b) = (
            srgb_to_linear(self.r),
            srgb_to_linear(self.g),
            srgb_to_linear(self.b),
        );

        let [lightness, a, bb] = linear_rgb_to_oklab([r, g, b]);
        Oklch {
            l: lightness,
            c: a.hypot(bb),
            h: bb.atan2(a).to_degrees().rem_euclid(360.0),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn the_dither_offset_stays_inside_half_a_level() {
        // The bound the whole effect rests on. A dither wider than half a level stops
        // being invisible and starts being grain, and a dither narrower than half a level
        // does not reach the next value and so changes nothing.
        let mut lowest = f32::MAX;
        let mut highest = f32::MIN;
        for y in 0..64 {
            for x in 0..64 {
                let n = dither_noise(x, y);
                lowest = lowest.min(n);
                highest = highest.max(n);
            }
        }
        assert!(
            lowest >= -0.5 && highest < 0.5,
            "the offset left [-0.5, 0.5): {lowest} to {highest}"
        );
        assert!(
            lowest < -0.45 && highest > 0.45,
            "the offset never reaches the ends it is allowed: {lowest} to {highest}"
        );
    }

    #[test]
    fn the_dither_averages_to_nothing() {
        // What licenses leaving the contrast gate alone. The gate checks a composite of
        // token colours; if the dither had a mean, every dithered surface would sit at a
        // different colour from the one the gate cleared.
        let (mut sum, mut count) = (0.0_f64, 0.0_f64);
        for y in 0..256 {
            for x in 0..256 {
                sum += f64::from(dither_noise(x, y));
                count += 1.0;
            }
        }
        let mean = sum / count;
        assert!(mean.abs() < 0.002, "the offset has a bias of {mean}");
    }

    #[test]
    fn the_dither_does_not_correlate_with_pixel_parity() {
        // The failure taking the *low* bits of the mixer would produce: a hash that tracks
        // x or y parity draws a checkerboard rather than grain, and a checkerboard over a
        // ramp is a worse artefact than the banding it replaced.
        for (name, of) in [
            (
                "x parity",
                (|x: u32, y: u32| (x, y)) as fn(u32, u32) -> (u32, u32),
            ),
            ("y parity", |x: u32, y: u32| (y, x)),
        ] {
            let (mut even, mut odd) = (0.0_f64, 0.0_f64);
            for a in 0..256 {
                for b in 0..256 {
                    let (x, y) = of(a, b);
                    let n = f64::from(dither_noise(x, y));
                    if a % 2 == 0 { even += n } else { odd += n }
                }
            }
            let split = (even - odd).abs() / (256.0 * 256.0 / 2.0);
            assert!(split < 0.01, "{name}: the two halves differ by {split}");
        }
    }

    #[test]
    fn the_amplitude_follows_the_srgb_encode_rather_than_linear_light() {
        // One output level, expressed in linear light, is not one number: it is 0.077 of a
        // per-cent near black and 0.9 of a per-cent near white. Getting this backwards --
        // a constant linear amplitude -- is what under-dithers exactly the dark greys a
        // dark theme is made of.
        for probe in [0.002_f32, 0.02, 0.2, 0.9] {
            let amplitude = dither_amplitude(probe);
            // What the sRGB encode says one level is worth here, measured through the real
            // transfer function rather than through the constants under test. Centred, so
            // the curve's own convexity does not bias the comparison.
            let s = linear_to_srgb(probe);
            let truth = srgb_to_linear(s + 0.5 / 255.0) - srgb_to_linear(s - 0.5 / 255.0);
            let ratio = amplitude / truth;
            assert!(
                (0.97..1.03).contains(&ratio),
                "at {probe} the amplitude is {ratio} times one output level"
            );
        }
        // Black is on the toe, where a level is worth the same everywhere. It is not zero:
        // the encode has a slope there, and a ramp leaving black crosses levels like any
        // other.
        assert!((dither_amplitude(0.0) - DITHER_TOE_SLOPE / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_ramp_too_shallow_for_eight_bits_dissolves_into_grain() {
        // The failure this whole thing exists for, taken from the material-library close:
        // a selection ramp crossing four output levels over three hundred and sixty pixels
        // drew four hard edges the full height of the row, and both ramps were reverted to
        // flat fills over it.
        //
        // The write is modelled the way the framebuffer performs it -- linear in, sRGB out,
        // eight bits -- because that is where the quantization happens. Counting *edges*
        // rather than run lengths is deliberate: a run says how wide a band is, and an
        // undithered ramp is allowed wide bands near the ends where the value really is
        // constant. What makes it banding is that there are exactly four places it changes.
        const ROWS: u32 = 360;
        let from = srgb_to_linear(0.160);
        let to = srgb_to_linear(0.176);
        let at = |y: u32| from + (to - from) * y as f32 / (ROWS - 1) as f32;
        let quantise = |l: f32| (linear_to_srgb(l.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;

        let edges =
            |sample: &dyn Fn(u32) -> u8| (1..ROWS).filter(|&y| sample(y) != sample(y - 1)).count();
        let banded = edges(&|y| quantise(at(y)));
        let grained = edges(&|y| quantise(dithered([at(y); 3], 0, y)[0]));

        assert!(
            banded <= 6,
            "the fixture is not shallow enough to band: {banded} edges"
        );
        assert!(
            grained > 60,
            "the dither left the contours intact: {grained} edges against {banded}"
        );
    }

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
    fn oklch_round_trips_through_srgb() {
        // Both directions, over the range a token file actually uses. A transposed matrix
        // row survives a one-way test and dies here.
        for &l in &[0.05_f32, 0.2, 0.45, 0.7, 0.95] {
            for &asked in &[0.0_f32, 0.02, 0.08] {
                for &h in &[27.0_f32, 73.0, 149.0, 260.0, 268.0] {
                    // Only in-gamut colours round-trip: asking for more chroma than sRGB
                    // holds is answered with less, which is `to_srgb`'s job and is tested
                    // by `the_gamut_map_reduces_chroma_and_leaves_hue_alone`.
                    let c = asked.min(Oklch::max_chroma(l, h));
                    let back = Oklch::new(l, c, h).to_srgb().to_oklch();
                    assert!((back.l - l).abs() < 0.01, "lightness {l} -> {}", back.l);
                    assert!((back.c - c).abs() < 0.01, "chroma {c} -> {}", back.c);
                    if c > 0.01 {
                        let err = ((back.h - h + 180.0).rem_euclid(360.0) - 180.0).abs();
                        assert!(err < 5.0, "hue {h} -> {} at c={c}", back.h);
                    }
                }
            }
        }
    }

    #[test]
    fn the_achromatic_axis_is_grey_in_both_directions() {
        for i in 0..=10 {
            let l = i as f32 / 10.0;
            let grey = Oklch::new(l, 0.0, 268.0).to_srgb();
            assert!(
                (grey.r - grey.g).abs() < 1e-3 && (grey.g - grey.b).abs() < 1e-3,
                "c = 0 must be neutral, got {grey:?}"
            );
            assert!(grey.to_oklch().c < 1e-3);
        }
        assert!((Oklch::new(1.0, 0.0, 0.0).to_srgb().r - 1.0).abs() < 1e-3);
        assert_eq!(Oklch::new(0.0, 0.0, 0.0).to_srgb().r, 0.0);
    }

    #[test]
    fn the_gamut_map_reduces_chroma_and_leaves_hue_alone() {
        // The whole reason to gamut-map rather than clip per channel: clipping a too-
        // colourful blue turns it a different colour, which is exactly the drift a
        // perceptual palette exists to prevent.
        let asked = Oklch::new(0.5, 0.4, 260.0);
        assert!(!asked.in_srgb_gamut(), "0.4 chroma must be outside sRGB");

        let got = asked.to_srgb().to_oklch();
        assert!(got.c < asked.c, "chroma must come down");
        assert!((got.l - asked.l).abs() < 0.01, "lightness must not move");
        let err = ((got.h - asked.h + 180.0).rem_euclid(360.0) - 180.0).abs();
        assert!(err < 3.0, "hue must not move, got {}", got.h);
    }

    #[test]
    fn max_chroma_brackets_the_gamut_boundary() {
        for &h in &[27.0_f32, 73.0, 149.0, 260.0] {
            for &l in &[0.3_f32, 0.5, 0.75] {
                let max = Oklch::max_chroma(l, h);
                assert!(max > 0.0, "some chroma must fit at l={l} h={h}");
                assert!(Oklch::new(l, max, h).in_srgb_gamut(), "the max must fit");
                assert!(
                    !Oklch::new(l, max + 0.01, h).in_srgb_gamut(),
                    "it must be the *maximum*: l={l} h={h} max={max}"
                );
            }
        }
        // The ceiling is not a constant -- this is why "the same chroma in both themes"
        // and "the same fraction of what is available" are different requests, and why
        // the token file needs both spellings.
        let mid = Oklch::max_chroma(0.5, 260.0);
        let light = Oklch::max_chroma(0.85, 260.0);
        assert!(
            mid > light * 2.0,
            "sRGB is not a cylinder: {mid} at l=0.5 vs {light} at l=0.85"
        );
    }

    #[test]
    fn the_ends_of_a_ramp_cannot_carry_colour() {
        // White admits none: at l = 1.0 the only sRGB colour is #ffffff, so a ramp whose
        // chroma curve does not taper to zero there is asking for something that cannot
        // exist. The bound is the bisection's residual bracket rather than a literal zero,
        // and it is four orders of magnitude below the chroma one 8-bit step can carry.
        assert!(Oklch::max_chroma(1.0, 268.0) < 1e-4);
        let white = Oklch::new(1.0, 0.3, 268.0).to_srgb();
        assert!(
            white.r > 0.999 && white.g > 0.999 && white.b > 0.999,
            "l = 1 must be white at any chroma, got {white:?}"
        );

        // Black is the asymmetric one, and the asymmetry is worth stating rather than
        // asserting away. `max_chroma(0.0, ..)` is about 0.011 rather than zero, because
        // at l = 0 a small chroma still maps to linear components within `0.0..=1.0` --
        // they are simply all indistinguishable from zero. What matters is the colour that
        // comes out, and that is black whatever chroma is asked for.
        for &c in &[0.0_f32, 0.02, 0.08, 0.3] {
            let black = Oklch::new(0.0, c, 268.0).to_srgb();
            assert!(
                black.r < 1e-3 && black.g < 1e-3 && black.b < 1e-3,
                "l = 0 must be black at any chroma, got {black:?} for c = {c}"
            );
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
