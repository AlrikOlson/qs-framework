//! Save material samples beside their flat fills in both themes.
//!
//! `cargo run -p qs-ui --example material_strip -- out.png`
//!
//! This uses the CPU renderer. GPU-only layers use their declared fallbacks,
//! so effects such as the selection halo are absent.

// A looking harness, not shipped code: a panic here is a developer seeing a stack trace
// instead of a picture. Same set the other examples allow.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use qs_gpu::cpu_raster::CpuRasterizer;
use qs_gpu::frame::{DrawList, Instance, PrimKind};
use qs_ui::Surface;
use qs_ui::material::name;
use qs_ui::tokens::{Theme, Tokens};

const WIDTH: u32 = 760;
const BAND: u32 = 56;
const GAP: u32 = 10;
const MARGIN: u32 = 16;

fn write_png(path: &str, width: u32, height: u32, srgb: &[u8]) {
    let mut pixmap = tiny_skia::Pixmap::new(width, height).expect("pixmap");
    pixmap.data_mut().copy_from_slice(srgb);
    pixmap.save_png(path).expect("save");
}

/// Every band, in the order they are drawn. The second field is the ground each is painted
/// over, so a translucent material is seen on what it will actually sit on rather than on
/// the page.
const BANDS: [(&str, &str); 7] = [
    (name::CHROME_BAR, "surface/base"),
    (name::CHROME_SHELF, "surface/base"),
    (name::CHROME_CHIP_HOVER, "surface/overlay-lift"),
    (name::ROW_HOVER, "surface/base"),
    (name::ROW_HOVER, "surface/row-alt"),
    (name::ROW_SELECTED, "surface/base"),
    (name::ROW_SELECTED, "surface/row-hover"),
];

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "materials.png".to_string());

    let height = MARGIN * 2 + BANDS.len() as u32 * BAND + (BANDS.len() as u32 - 1) * GAP;

    for (theme, label) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
        let tokens = Tokens::embedded(theme).expect("tokens");
        let mut raster = CpuRasterizer::new(WIDTH, height, 64).expect("rasterizer");
        let mut list = DrawList::default();
        list.reset([WIDTH, height], tokens.color("surface/base"), 1);

        let x = MARGIN as f32;
        // Two columns: the material on the left, the flat token it replaced on the right,
        // adjacent so the comparison is seen rather than remembered.
        let w = ((WIDTH - MARGIN * 2) as f32 - GAP as f32) * 0.5;
        let mut y = MARGIN as f32;
        let radius = tokens.radius(qs_ui::tokens::radius::ROW);

        for (material, ground) in BANDS {
            let base = tokens.color(ground);
            for column in 0..2 {
                let cx = x + column as f32 * (w + GAP as f32);
                list.instances
                    .push(Instance::rect(cx, y, w, BAND as f32, 0.0, base));
            }

            // Left: the material, painted onto a region the shape of a row.
            tokens.paint(
                material,
                Surface::new(x, y, w, BAND as f32, radius, 1.0),
                1.0,
                &mut list.instances,
            );

            // Right: the same material with effects switched off, which is what the code
            // drew before this chunk and what forced-colours mode draws now.
            //
            // Compiled rather than reconstructed from a layer's colour. The first version of
            // this harness read `layers.first().flat`, which for `row/selected` is the
            // *halo's* stop -- so the comparison column painted a 45%-opacity focus blue and
            // the selection appeared to have got four times darker. A picture that is wrong
            // in a way the code is not is worse than no picture.
            if let Some(stack) = tokens.material(material) {
                stack.compile(
                    Surface::new(x + w + GAP as f32, y, w, BAND as f32, radius, 1.0),
                    1.0,
                    qs_ui::material::Drive::REST,
                    false,
                    &mut list.instances,
                );
            }

            y += (BAND + GAP) as f32;
        }

        list.end_batch(None, false);
        list.finish();

        // The CPU tier resolves fidelity on the way in, so this count is the honest report
        // of what a machine with no usable GPU is not shown.
        let halos = list
            .instances
            .iter()
            .filter(|i| i.kind == PrimKind::Glow as u32)
            .count();

        raster.render(&list);
        let path = out.replace(".png", &format!("-{label}.png"));
        write_png(&path, WIDTH, height, &raster.to_srgb_rgba8());
        println!(
            "wrote {path}: {} instances, {halos} of them halos the CPU tier drew as nothing",
            list.instances.len()
        );
    }
}
