//! Render the gradient primitive against the real tokens, in both themes, and write two
//! PNGs to look at.
//!
//! `cargo run -p qs-ui --example gradient_strip -- out.png`
//!
//! chunk:prim-gradient's numeric criteria are all checkable and all green: the two tiers
//! agree to one 8-bit level where the ramp is isolated, the midpoint lands on the
//! perceptual midpoint of its stops, and the contrast gate covers both ends. None of that
//! answers whether the ramp *looks* like anything, and this project has shipped a defect
//! that only looking found in every visual chunk so far. It found one here too: at the
//! dark theme's first values the bar's ramp was indistinguishable from the flat fill it
//! replaced, because `surface/overlay` is pinned at `l = 0` and a small lift off pure
//! black is a handful of 8-bit levels. See `surface/overlay-lift`'s description.
//!
//! # Why this lives in qs-ui and not qs-gpu
//!
//! It was written in `qs-gpu` first, next to the primitive, and had to hand-resolve the
//! two stops out of `design/tokens.json` as literals -- `qs-gpu` is below the token layer.
//! Two copies of a colour is the drift this token file exists to prevent, and a *looking*
//! harness showing something other than what ships is worse than no harness. `qs-ui` reads
//! the real tokens, so the picture is the shipped ramp by construction.
//!
//! Four bands per theme, adjacent so the comparisons are seen rather than remembered:
//!
//! - the command bar's ramp, at roughly the height it is drawn -- the only band that
//!   answers "is the lift visible, and is it too much";
//! - the flat fill it replaced, directly beneath, because the honest question about a
//!   subtle gradient is whether it is doing anything at all;
//! - a wide-interval ramp, steel blue to amber, in Oklab;
//! - the same pair walked in **linear light**, which is what the shader would do without
//!   `linear_rgb_to_oklab`. It reaches the bright end early and spends most of its length
//!   near the top, crushing the dark half into a sliver. That is the same defect
//!   `the_ramp_is_walked_in_oklab_and_not_in_linear_srgb` measures as an `l` of 0.647
//!   where the perceptual midpoint is 0.551, and it is far more obvious as a picture.

// A looking harness, not shipped code: a panic here is a developer seeing a stack trace
// instead of a picture. Same set the other two examples allow.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use qs_gpu::color::Srgba;
use qs_gpu::cpu_raster::CpuRasterizer;
use qs_gpu::frame::{DrawList, Instance};
use qs_ui::tokens::{Theme, Tokens};

const WIDTH: u32 = 720;
const BAND: u32 = 44;
const GAP: u32 = 12;
const MARGIN: u32 = 16;
const BANDS: u32 = 4;

fn write_png(path: &str, width: u32, height: u32, srgb: &[u8]) {
    let mut pixmap = tiny_skia::Pixmap::new(width, height).expect("pixmap");
    pixmap.data_mut().copy_from_slice(srgb);
    pixmap.save_png(path).expect("save");
}

/// The linear-light walk, as a run of flat sub-rectangles.
///
/// Built from `Instance::rect` rather than from a second gradient kind on purpose: the
/// band's job is to show what the primitive is *not* doing, and drawing that with the
/// primitive itself would be circular.
fn linear_light_band(list: &mut DrawList, x: f32, y: f32, w: f32, h: f32, a: Srgba, b: Srgba) {
    const STEPS: u32 = 240;
    let lin = qs_gpu::color::srgb_to_linear;
    let to_srgb = qs_gpu::color::linear_to_srgb;
    let slice = w / STEPS as f32;
    for i in 0..STEPS {
        let t = i as f32 / (STEPS - 1) as f32;
        let mix = |from: f32, to: f32| to_srgb(lin(from) + (lin(to) - lin(from)) * t);
        list.instances.push(Instance::rect(
            x + slice * i as f32,
            y,
            // A hair of overlap, or the seams read as 240 hairlines and the band looks
            // like a defect in the comparison rather than in what is being compared.
            slice + 1.0,
            h,
            0.0,
            Srgba::new(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b), 1.0),
        ));
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gradient.png".to_string());

    // The same pair `the_ramp_is_walked_in_oklab_and_not_in_linear_srgb` measures, so the
    // picture and the assertion are about one thing.
    let steel = Srgba::new(0.145, 0.176, 0.278, 1.0);
    let amber = Srgba::new(0.925, 0.706, 0.196, 1.0);

    let height = MARGIN * 2 + BANDS * BAND + (BANDS - 1) * GAP;

    for (theme, label) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
        let tokens = Tokens::embedded(theme).expect("tokens");
        let lift = tokens.color("surface/overlay-lift");
        let deep = tokens.color("surface/overlay");

        let mut raster = CpuRasterizer::new(WIDTH, height, 64).expect("rasterizer");
        let mut list = DrawList::default();
        list.reset([WIDTH, height], tokens.color("surface/base"), 1);

        let x = MARGIN as f32;
        let w = (WIDTH - MARGIN * 2) as f32;
        let mut y = MARGIN as f32;
        let step = (BAND + GAP) as f32;

        // 1. The bar's ramp, top-lit, exactly as `chrome::draw_command_bar` emits it.
        list.instances.push(Instance::gradient(
            x,
            y,
            w,
            BAND as f32,
            0.0,
            std::f32::consts::FRAC_PI_2,
            lift,
            deep,
        ));
        y += step;

        // 2. The flat fill it replaced.
        list.instances
            .push(Instance::rect(x, y, w, BAND as f32, 0.0, deep));
        y += step;

        // 3. The wide-interval pair in Oklab.
        list.instances.push(Instance::gradient(
            x,
            y,
            w,
            BAND as f32,
            0.0,
            0.0,
            steel,
            amber,
        ));
        y += step;

        // 4. The same pair in linear light, for the comparison.
        linear_light_band(&mut list, x, y, w, BAND as f32, steel, amber);

        list.end_batch(None, false);
        list.finish();

        raster.render(&list);
        let path = out.replace(".png", &format!("-{label}.png"));
        write_png(&path, WIDTH, height, &raster.to_srgb_rgba8());
        println!("wrote {path}");
    }
}
