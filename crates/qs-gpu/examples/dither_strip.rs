//! Save shallow gradients with and without dithering.
//!
//! `cargo run -p qs-gpu --example dither_strip -- out.png`
//!
//! Each band shows plain output on the left and dithered output on the right.
//! The bands cover dark, midtone and near-white gradients, plus a flat fill.
//! The calculation models the GPU's sRGB output conversion directly; it does
//! not use the CPU renderer's linear eight-bit buffer.

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
