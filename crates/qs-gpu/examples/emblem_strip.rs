//! Save the icon set with and without symlink emblems in both themes.
//!
//! `cargo run -p qs-gpu --example emblem_strip -- out.png`
//!
//! This example rasterizes and composites the icon masks directly for visual
//! inspection at their intended size.

// A developer tool that renders a picture and exits. The crate's production lints are about
// code that runs inside a frame: an out-of-range pixmap or an unwritable path here should stop
// with a message, and adding error plumbing would only obscure the geometry this file is for.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use qs_gpu::icon::{self, Emblem, IconKey, IconKind, IconShape};
use tiny_skia::{Pixmap, PremultipliedColorU8};

const ICON: u16 = 20;
const PAD: u16 = 8;
const ZOOM: u32 = 6;

struct Theme {
    bg: [u8; 3],
    file: [u8; 3],
    folder: [u8; 3],
    plate: [u8; 3],
    mark: [u8; 3],
}

fn main() {
    let themes = [
        Theme {
            bg: [0x1b, 0x1b, 0x1f],
            file: [0x9a, 0xa0, 0xb0],
            folder: [0xe0, 0xa5, 0x4a],
            plate: [0x1b, 0x1b, 0x1f],
            mark: [0xb3, 0xb8, 0xc6],
        },
        Theme {
            bg: [0xff, 0xff, 0xff],
            file: [0x5c, 0x62, 0x72],
            folder: [0x8a, 0x5a, 0x00],
            plate: [0xff, 0xff, 0xff],
            mark: [0x4f, 0x55, 0x63],
        },
    ];

    let cols = IconKind::ALL.len() as u32;
    let cell = u32::from(ICON + PAD);
    let w = cols * cell + u32::from(PAD);
    // Two bands per theme: plain, then emblemed.
    let band = cell;
    let h = themes.len() as u32 * band * 2 + u32::from(PAD);

    let mut pm = Pixmap::new(w, h).expect("pixmap");

    for (t, theme) in themes.iter().enumerate() {
        let y0 = t as u32 * band * 2;
        fill(&mut pm, 0, y0, w, band * 2, theme.bg);
        for (c, &kind) in IconKind::ALL.iter().enumerate() {
            let x = u32::from(PAD) + c as u32 * cell;
            for (r, emblemed) in [false, true].into_iter().enumerate() {
                let y = y0 + r as u32 * band + u32::from(PAD) / 2;
                let tint = if kind == IconKind::Folder {
                    theme.folder
                } else {
                    theme.file
                };
                blit(&mut pm, shape(IconShape::Kind(kind), ICON), x, y, tint);
                if emblemed {
                    let e = icon::emblem_px(ICON).expect("20px carries an emblem");
                    let ey = y + u32::from(ICON) - u32::from(e);
                    blit(&mut pm, shape(em(Emblem::Plate), e), x, ey, theme.plate);
                    blit(&mut pm, shape(em(Emblem::Symlink), e), x, ey, theme.mark);
                }
            }
        }
    }

    let out = std::env::args()
        .nth(1)
        .expect("usage: emblem_strip <out.png>");
    zoom(&pm, ZOOM).save_png(&out).expect("save");
    pm.save_png(out.replace(".png", "-1x.png"))
        .expect("save 1x");
    println!("wrote {out}");
}

fn em(e: Emblem) -> IconShape {
    IconShape::Emblem(e)
}

fn shape(shape: IconShape, px: u16) -> qs_text::RasterizedGlyph {
    icon::rasterize(IconKey { shape, px }).unwrap_or_else(|| panic!("{shape:?} at {px}px"))
}

fn fill(pm: &mut Pixmap, x0: u32, y0: u32, w: u32, h: u32, c: [u8; 3]) {
    let px = PremultipliedColorU8::from_rgba(c[0], c[1], c[2], 255).expect("opaque");
    let stride = pm.width();
    for y in y0..(y0 + h).min(pm.height()) {
        for x in x0..(x0 + w).min(stride) {
            pm.pixels_mut()[(y * stride + x) as usize] = px;
        }
    }
}

/// Source-over of a coverage mask tinted `c`, which is what `Instance::glyph` does on the GPU.
fn blit(pm: &mut Pixmap, g: qs_text::RasterizedGlyph, x0: u32, y0: u32, c: [u8; 3]) {
    let stride = pm.width();
    for gy in 0..g.height {
        for gx in 0..g.width {
            let a = u32::from(g.coverage[(gy * g.width + gx) as usize]);
            if a == 0 {
                continue;
            }
            let (x, y) = (x0 + gx, y0 + gy);
            if x >= stride || y >= pm.height() {
                continue;
            }
            let dst = pm.pixels_mut()[(y * stride + x) as usize];
            let mix = |s: u8, d: u8| ((u32::from(s) * a + u32::from(d) * (255 - a)) / 255) as u8;
            pm.pixels_mut()[(y * stride + x) as usize] = PremultipliedColorU8::from_rgba(
                mix(c[0], dst.red()),
                mix(c[1], dst.green()),
                mix(c[2], dst.blue()),
                255,
            )
            .expect("opaque");
        }
    }
}

fn zoom(src: &Pixmap, n: u32) -> Pixmap {
    let mut out = Pixmap::new(src.width() * n, src.height() * n).expect("pixmap");
    let (ow, oh) = (out.width(), out.height());
    for y in 0..oh {
        for x in 0..ow {
            let s = src.pixels()[((y / n) * src.width() + x / n) as usize];
            out.pixels_mut()[(y * ow + x) as usize] = s;
        }
    }
    out
}
