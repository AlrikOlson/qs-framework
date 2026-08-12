//! Render a ramp too shallow for eight bits, with and without the dither, and write a PNG
//! to look at.
//!
//! `cargo run -p qs-gpu --example dither_strip -- out.png`
//!
//! chunk:prim-noise-dither's numeric criteria are all checkable and all green: the offset
//! stays inside half a level, it averages to nothing, it tracks the sRGB encode's slope
//! within three per cent at four probes, and the two tiers now produce identical bytes on
//! every sharp gradient fixture. None of that answers whether the banding is gone, and this
//! project has shipped a defect that only looking found in every visual chunk so far --
//! including the one that produced this chunk, where two row ramps were authored, measured,
//! passed every gate, and drew four hard edges.
//!
//! # Why this does not go through `CpuRasterizer`
//!
//! The other looking harnesses render a draw list, which is the right thing when the
//! question is about a shape. The question here is about a *write*: the framebuffer is
//! `Bgra8UnormSrgb`, so the hardware encodes to sRGB and quantizes to eight bits, and that
//! is the only step at which banding exists. `CpuRasterizer` composites into a **linear**
//! eight-bit pixmap and encodes once at the end, so its quantization is a different one --
//! coarser in the darks, which is worth knowing but is not what this picture is about.
//! Modelling the GPU's write directly is what makes the two halves comparable.
//!
//! Four bands, each split down the middle -- plain on the left, dithered on the right --
//! because the honest comparison is side by side rather than remembered:
//!
//! - a dark ramp over four output levels, which is the case from the material-library close;
//! - the same interval in the midtones, where a level is worth more in linear light;
//! - a near-white ramp, where the encode's slope is steepest and the offset is largest;
//! - a flat fill at the dark ramp's midpoint, which must come out **flat**: the dither is
//!   supposed to dissolve a contour, not to add grain to a surface that has none.

// A looking harness, not shipped code: a panic here is a developer seeing a stack trace
// instead of a picture. Same set the other examples allow.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use qs_gpu::color::{dithered, linear_to_srgb, srgb_to_linear};

const WIDTH: u32 = 720;
const BAND: u32 = 120;
const GAP: u32 = 8;

/// One band: a vertical ramp between two sRGB values, plain on the left of `split` and
/// dithered on the right.
struct Band {
    from: f32,
    to: f32,
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "dither.png".to_string());

    let bands = [
        // Four output levels, in the dark greys a dark theme's surfaces live in. This is the
        // interval that shipped as four hard edges.
        Band {
            from: 0.160,
            to: 0.176,
        },
        Band {
            from: 0.480,
            to: 0.496,
        },
        Band {
            from: 0.880,
            to: 0.896,
        },
        // Flat: the control. Any grain visible here is the dither doing something it was
        // not asked to do.
        Band {
            from: 0.168,
            to: 0.168,
        },
    ];

    let height = bands.len() as u32 * (BAND + GAP) + GAP;
    let mut rgba = vec![0u8; (WIDTH * height * 4) as usize];
    // A mid grey surround, so neither half of a band is being judged against black.
    for px in rgba.chunks_exact_mut(4) {
        px.copy_from_slice(&[0x18, 0x1b, 0x22, 0xff]);
    }

    let split = WIDTH / 2;
    for (index, band) in bands.iter().enumerate() {
        let top = GAP + index as u32 * (BAND + GAP);
        for row in 0..BAND {
            let y = top + row;
            // The ramp runs down the band, which is the axis a row's ramp runs across and
            // the axis the contours were perpendicular to.
            let t = row as f32 / (BAND - 1) as f32;
            let linear = srgb_to_linear(band.from + (band.to - band.from) * t);
            for x in 0..WIDTH {
                let rgb = if x < split {
                    [linear; 3]
                } else {
                    dithered([linear; 3], x, y)
                };
                // What `Bgra8UnormSrgb` does on write: encode, then quantize to eight bits.
                let byte = |c: f32| (linear_to_srgb(c.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                let at = ((y * WIDTH + x) * 4) as usize;
                rgba[at] = byte(rgb[0]);
                rgba[at + 1] = byte(rgb[1]);
                rgba[at + 2] = byte(rgb[2]);
                rgba[at + 3] = 0xff;
            }
        }
    }

    let mut pixmap = tiny_skia::Pixmap::new(WIDTH, height).expect("pixmap");
    pixmap.data_mut().copy_from_slice(&rgba);
    pixmap.save_png(&out).expect("save");
    println!("wrote {out}");
    println!("left half of each band: plain. right half: dithered.");
}
