//! Save palette swatches in both themes.
//!
//! `cargo run -p qs-ui --example palette_strip -- out.png`
//!
//! The image places related row colors side by side, including hover and
//! selection. It also shows the focus outline and status colors.

// A developer tool that renders a picture and exits; the crate's production lints are about
// code that runs inside a frame.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use qs_gpu::cpu_raster::CpuRasterizer;
use qs_gpu::{DrawList, Instance};
use qs_ui::tokens::{Theme, Tokens};

/// Encode already-sRGB, opaque RGBA8 bytes.
///
/// A fresh pixmap rather than the rasterizer's own: that one is the linear working buffer.
/// Everything drawn here is opaque, so straight alpha and premultiplied coincide.
fn write_png(path: &str, width: u32, height: u32, srgb: &[u8]) {
    let size = tiny_skia::IntSize::from_wh(width, height).expect("size");
    let pixmap = tiny_skia::Pixmap::from_vec(srgb.to_vec(), size).expect("pixmap");
    pixmap.save_png(path).expect("save");
}

/// Each row of the sheet: a caption position and the tokens laid out left to right.
const SHEET: [&[&str]; 4] = [
    // The neutral surface ramp, in lightness order. An uneven step shows up here as a band
    // that is too wide or too narrow relative to its neighbours.
    &[
        "surface/base",
        "surface/row-alt",
        "surface/row-hover",
        "border/subtle",
        "surface/overlay",
    ],
    // Hover against selected. THE comparison.
    &["surface/row-hover", "surface/row-selected"],
    // The three text emphases and the badge, each drawn on the base surface.
    &[
        "content/primary",
        "content/secondary",
        "content/tertiary",
        "icon/badge",
        "content/on-overlay",
    ],
    // Chromatic tokens. The three rails must be distinguishable from each other.
    &[
        "border/focus",
        "icon/folder",
        "rail/modified",
        "rail/added",
        "rail/conflict",
    ],
];

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "palette.png".to_string());

    let (swatch, gap, pad) = (150u32, 8u32, 24u32);
    let width = pad * 2 + swatch * 5 + gap * 4;
    let height = pad * 2 + (swatch + gap) * SHEET.len() as u32 + swatch;

    for (theme, label) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
        let tokens = Tokens::embedded(theme).unwrap();
        let base = tokens.color("surface/base");

        let mut list = DrawList::default();
        list.reset([width, height], base, 1);

        for (row, names) in SHEET.iter().enumerate() {
            let y = pad + row as u32 * (swatch + gap);
            for (col, name) in names.iter().enumerate() {
                let x = pad + col as u32 * (swatch + gap);
                list.instances.push(Instance::rect(
                    x as f32,
                    y as f32,
                    swatch as f32,
                    swatch as f32,
                    0.0,
                    tokens.color(name),
                ));
            }
        }

        // The focus ring over a selected row, with its 1px outline, at the bottom. Drawn as
        // three nested rects because that is what the row renderer draws, and the point is
        // that the ring stays visible against a surface close to its own colour.
        let y = (pad + SHEET.len() as u32 * (swatch + gap)) as f32;
        let focus = tokens.focus();
        let scale = 2.0; // draw the ring at 2x so a 2px ring is actually inspectable
        let radius = tokens.radius(qs_ui::tokens::radius::ROW) * scale;

        for (i, under) in ["surface/row-selected", "surface/base", "surface/row-hover"]
            .into_iter()
            .enumerate()
        {
            // The ring over three different surfaces, because the failure it is designed
            // against is surface-dependent: a single-colour ring vanishes on the one
            // surface closest to it, which is the selected row -- the row most likely to
            // have focus.
            let x = pad as f32 + i as f32 * (swatch + gap) as f32;
            let (w, h) = (swatch as f32, swatch as f32);
            list.instances
                .push(Instance::rect(x, y, w, h, 0.0, tokens.color(under)));

            let (rx, ry) = (x + 24.0, y + 24.0);
            let (rw, rh) = (w - 48.0, h - 48.0);
            list.instances.push(Instance::stroke(
                rx - focus.outline_width * scale,
                ry - focus.outline_width * scale,
                rw + focus.outline_width * scale * 2.0,
                rh + focus.outline_width * scale * 2.0,
                radius,
                focus.outline_width * scale,
                tokens.color("border/focus-outline"),
            ));
            list.instances.push(Instance::stroke(
                rx,
                ry,
                rw,
                rh,
                radius,
                focus.ring_width * scale,
                tokens.color("border/focus"),
            ));
        }

        list.end_batch(None, false);
        list.finish();

        let mut raster = CpuRasterizer::new(width, height, 256).expect("rasterizer");
        raster.render(&list);

        // `render` leaves a **linear** premultiplied buffer; `to_srgb_rgba8` is the step
        // that makes it displayable. Saving the pixmap directly writes linear values into
        // an sRGB file and every colour comes out too dark -- measured, #4f5563 lands as
        // #141720, which is exactly srgb_to_linear applied once. That is easy to miss
        // because a uniformly darkened palette still looks like a palette, and the two
        // sibling harnesses in this tree both do it.
        let path = out.replace(".png", &format!("-{label}.png"));
        write_png(&path, width, height, &raster.to_srgb_rgba8());
        println!("wrote {path} ({label})");
    }
}
