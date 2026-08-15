//! What the bloom costs per frame, per tier — and whether it can be seen at all.
//!
//! `cargo run --release -p qs-gpu --example bloom_cost`
//!
//! # The four numbers
//!
//! `blur_cost`'s structure, with one column added, because the bloom's cost has to separate
//! from the target's exactly as the blur's did:
//!
//! - **one pass** — the frame as it renders with no target at all.
//! - **two pass** — `force_offscreen`, so the whole frame goes through the target and the
//!   resolve. The difference from one pass is the *target's* cost, taken here on the same
//!   machine in the same run rather than read off another report.
//! - **bloomed** — the same list with an active [`Bloom`]. The difference from two pass is the
//!   bloom itself: one bright-pass downsample and two Gaussian passes at a sixteenth of the
//!   viewport's fragment count, plus a resolve that samples a second texture.
//! - **bloomed + blurred** — both effects in one frame, which is the case the third chain
//!   texture exists for. The difference from *bloomed* is the blur's own three passes, and it
//!   should land near `blur_cost`'s chain figure — if it does not, the two effects are
//!   interfering rather than sharing.
//!
//! # The visibility check, and why it is not "bloom on versus bloom off"
//!
//! It is that, but measured **outside the bright element**. A bloom that only brightened the
//! emitter's own pixels would pass an on-versus-off comparison over the whole frame and be
//! invisible as an effect — the whole claim is that light *leaves* the bright thing. So the
//! reported figure is the largest change at a distance from the source, inside the blur's
//! reach, with the source's own rect excluded.
//!
//! This is `specs/002-ray-traced-mode`'s R15 lesson applied one effect further out: a guard
//! that compares the effect against its own absence measures that *something* happened, not
//! that the thing the effect is for happened.
//!
//! # What it measures, and what it does not
//!
//! `blur_cost`'s caveats, unchanged and all of them: wall clock from submit to the queue going
//! idle, not a GPU timestamp; treat the absolute figures as an upper bound and the differences
//! as the answer. The Reduced tier is reached by asking for its backends directly, so on a
//! machine whose GL driver is a translation layer over the same hardware "the low tier" means
//! the low *API*, not low-end hardware.
//!
//! The threshold below is this harness's own, not the shipped one. The shipped number comes
//! from `qs_ui::Tokens::bloom` and `qs-gpu` cannot see a palette; what is reproduced here is
//! the *rule* — a threshold above every flat colour in the scene — so the emitter is the only
//! thing that blooms.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use qs_gpu::batcher::Renderer;
use qs_gpu::color::Srgba;
use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::frame::{Bloom, DrawList, Instance};
use qs_gpu::path::RenderPath;
use qs_gpu::target::{BlurChain, OffscreenTarget, blur_reach_pixels};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FRAMES: u32 = 120;
const WARMUP: u32 = 20;

/// Above every flat colour the scene below uses, and below the emitter. See the module docs
/// for why this is not the shipped number.
const THRESHOLD: f32 = 0.60;
/// The shipped strength: `lighting.allowance.receiver.addition_max` in `design/tokens.json`.
/// Duplicated here rather than read, because this crate has no palette — a mismatch would
/// misreport the *cost* not at all (the passes run either way) and the visibility a little.
const STRENGTH: f32 = 0.35;

/// Where the one bright element sits. Small, because a small bright thing is what bloom is
/// for: a focus ring, a lamp, the peak of a sweep.
const EMITTER: [u32; 4] = [900, 500, 120, 80];

/// `blur_cost`'s scene, so the one-pass numbers are comparable, plus one element above the
/// threshold. Every other colour here is well below it.
fn scene(emitter: bool, panel: bool) -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.09, 0.09, 0.11, 1.0), 1);
    list.instances.push(Instance::rect(
        0.0,
        0.0,
        WIDTH as f32,
        HEIGHT as f32,
        0.0,
        Srgba::new(0.10, 0.10, 0.13, 1.0),
    ));
    let row_h = 28.0;
    let rows = (HEIGHT as f32 / row_h) as u32;
    for i in 0..rows {
        let y = i as f32 * row_h;
        if i % 2 == 1 {
            list.instances.push(Instance::rect(
                0.0,
                y,
                WIDTH as f32,
                row_h,
                0.0,
                Srgba::new(0.12, 0.12, 0.15, 1.0),
            ));
        }
        for c in 0..20 {
            list.instances.push(Instance::rect(
                40.0 + c as f32 * 36.0,
                y + 8.0,
                26.0,
                12.0,
                2.0,
                // Deliberately below THRESHOLD: chrome-shaped marks must not bloom, which is
                // the shipped rule (no authored colour is above the palette's own ceiling).
                Srgba::new(0.72, 0.74, 0.80, 0.9),
            ));
        }
    }
    if emitter {
        list.instances.push(Instance::rect(
            EMITTER[0] as f32,
            EMITTER[1] as f32,
            EMITTER[2] as f32,
            EMITTER[3] as f32,
            6.0,
            Srgba::new(1.0, 0.98, 0.90, 1.0),
        ));
    }
    if panel {
        list.instances.push(Instance::blur(
            WIDTH as f32 * 0.66,
            48.0,
            WIDTH as f32 * 0.32,
            HEIGHT as f32 - 96.0,
            14.0,
            Srgba::new(0.196, 0.196, 0.212, 0.90),
            Srgba::new(0.196, 0.196, 0.212, 1.0),
        ));
    }
    list.end_batch(None, false);
    list
}

fn surface(ctx: &GpuContext, readable: bool) -> (wgpu::Texture, wgpu::TextureView) {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    if readable {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom_cost surface"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ctx.capabilities.surface_format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn measure(ctx: &GpuContext, list: &DrawList, force: bool) -> (f64, Option<u64>) {
    let (_texture, view) = surface(ctx, false);
    let mut renderer = Renderer::new(ctx, 512);
    renderer.force_offscreen(force);

    for _ in 0..WARMUP {
        let cmd = renderer.render(ctx, &view, list, None, None);
        ctx.queue.submit([cmd]);
    }
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let start = std::time::Instant::now();
    for _ in 0..FRAMES {
        let cmd = renderer.render(ctx, &view, list, None, None);
        ctx.queue.submit([cmd]);
    }
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let elapsed = start.elapsed();

    let chain = renderer.blur().map(BlurChain::bytes);
    (elapsed.as_secs_f64() * 1000.0 / f64::from(FRAMES), chain)
}

/// Render one frame and read the surface back as RGBA8 rows, unpadded.
fn shoot(ctx: &GpuContext, list: &DrawList) -> Vec<u8> {
    let (texture, view) = surface(ctx, true);
    let mut renderer = Renderer::new(ctx, 512);
    let cmd = renderer.render(ctx, &view, list, None, None);
    ctx.queue.submit([cmd]);

    let row = WIDTH * 4;
    let padded = row.div_ceil(256) * 256;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bloom_cost readback"),
        size: u64::from(padded) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((row * HEIGHT) as usize);
    for y in 0..HEIGHT {
        let start = (y * padded) as usize;
        out.extend_from_slice(&mapped[start..start + row as usize]);
    }
    drop(mapped);
    buffer.unmap();
    out
}

/// The largest per-channel change between two frames, restricted to pixels **outside** the
/// emitter and within the blur's reach of it. See the module docs for why the restriction is
/// the whole point.
fn light_that_left_the_source(off: &[u8], on: &[u8]) -> (u8, u32) {
    let reach = blur_reach_pixels();
    let x0 = EMITTER[0].saturating_sub(reach);
    let y0 = EMITTER[1].saturating_sub(reach);
    let x1 = (EMITTER[0] + EMITTER[2] + reach).min(WIDTH);
    let y1 = (EMITTER[1] + EMITTER[3] + reach).min(HEIGHT);
    let mut worst = 0u8;
    let mut changed = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let inside_source = x >= EMITTER[0]
                && x < EMITTER[0] + EMITTER[2]
                && y >= EMITTER[1]
                && y < EMITTER[1] + EMITTER[3];
            if inside_source {
                continue;
            }
            let i = ((y * WIDTH + x) * 4) as usize;
            let mut pixel = 0u8;
            for c in 0..3 {
                pixel = pixel.max(on[i + c].abs_diff(off[i + c]));
            }
            worst = worst.max(pixel);
            if pixel > 0 {
                changed += 1;
            }
        }
    }
    (worst, changed)
}

/// What blooming a *mark* costs the contrast ratio between it and its own ground.
///
/// This is the measurement that decides where the shipped threshold may sit, and it exists
/// because the argument against a lower one was a citation rather than a number. Rule 1a of
/// `specs/002-ray-traced-mode/contracts/lit-contrast.md` says a meaning-bearing element may
/// emit but its ground may never receive; bloom from a glyph lands on that glyph's own ground,
/// so a threshold below the ink ceiling puts the two in conflict. How much conflict is a
/// quantity, and `cargo xtask contrast` cannot see it — it reads tokens, not frames.
///
/// The scene is the shipped dark theme's worst realistic pair: body ink at 0.817 relative
/// luminance on a ground at 0.011, which clears 4.5:1 by a wide margin *before* the bloom.
/// Returns (ratio before, ratio after) with the ground sampled just outside the mark.
fn contrast_cost_of_blooming_a_mark(ctx: &GpuContext) -> (f64, f64) {
    // Dark-theme `surface/base` and `content/primary`, as sRGB, to two decimal places. Close
    // enough: the question is the size of the change, not the third digit of the ratio.
    let ground = Srgba::new(0.11, 0.11, 0.13, 1.0);
    let ink = Srgba::new(0.92, 0.93, 0.95, 1.0);
    let mark = [900.0_f32, 500.0, 120.0, 24.0];

    let build = |bloom: Option<Bloom>| {
        let mut list = DrawList::default();
        list.reset([WIDTH, HEIGHT], ground, 1);
        list.instances.push(Instance::rect(
            0.0,
            0.0,
            WIDTH as f32,
            HEIGHT as f32,
            0.0,
            ground,
        ));
        list.instances
            .push(Instance::rect(mark[0], mark[1], mark[2], mark[3], 0.0, ink));
        if let Some(bloom) = bloom {
            list.set_bloom(bloom);
        }
        list.end_batch(None, false);
        list
    };

    // A threshold below the ink, so the mark blooms. This is the configuration the shipped
    // palette refuses, rendered so the refusal has a number behind it.
    let blooming = Bloom {
        threshold: 0.70,
        strength: STRENGTH,
    };
    let off = shoot(ctx, &build(None));
    let on = shoot(ctx, &build(Some(blooming)));

    // Sample the ground four pixels below the mark: close enough to catch the bloom, outside
    // the mark itself. Averaged along its width so one stray texel cannot carry the answer.
    let sample = |px: &[u8]| -> f64 {
        let y = (mark[1] + mark[3] + 4.0) as u32;
        let mut total = 0.0;
        let mut n = 0.0;
        for x in (mark[0] as u32)..((mark[0] + mark[2]) as u32) {
            let i = ((y * WIDTH + x) * 4) as usize;
            let l = |c: u8| {
                let v = f64::from(c) / 255.0;
                if v <= 0.040_45 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            total += 0.2126 * l(px[i]) + 0.7152 * l(px[i + 1]) + 0.0722 * l(px[i + 2]);
            n += 1.0;
        }
        total / n
    };

    let ink_l = f64::from(ink.relative_luminance());
    let ratio = |bg: f64| (ink_l + 0.05) / (bg + 0.05);
    (ratio(sample(&off)), ratio(sample(&on)))
}

fn main() {
    let target = OffscreenTarget::bytes_at([WIDTH, HEIGHT]);
    let chain = BlurChain::bytes_at([WIDTH, HEIGHT]);
    println!("at {WIDTH}x{HEIGHT}:");
    println!(
        "  colour target  {:.1} MB ({target} bytes)",
        target as f64 / 1_000_000.0
    );
    println!(
        "  chain          {:.1} MB ({chain} bytes, {} halves, {:.1}% of the target)",
        chain as f64 / 1_000_000.0,
        BlurChain::HALVES,
        chain as f64 / target as f64 * 100.0
    );
    println!(
        "  bloom's share  {:.1} MB (the third half)\n",
        chain as f64 / BlurChain::HALVES as f64 / 1_000_000.0
    );

    let bloom = Bloom {
        threshold: THRESHOLD,
        strength: STRENGTH,
    };
    let plain = scene(true, false);
    let mut bloomed = scene(true, false);
    bloomed.set_bloom(bloom);
    let mut both = scene(true, true);
    both.set_bloom(bloom);

    println!(
        "{} instances, {FRAMES} frames per run, threshold {THRESHOLD}, strength {STRENGTH}\n",
        plain.instances.len()
    );

    for tier in [RenderPath::Primary, RenderPath::Reduced] {
        let instance = new_instance(tier);
        let Ok(ctx) = GpuContext::new(tier, instance, None) else {
            println!("{tier:?}: no usable adapter, skipped");
            continue;
        };
        let name = ctx.capabilities.adapter_name.clone();
        let backend = ctx.capabilities.backend;

        let (one, _) = measure(&ctx, &plain, false);
        let (two, _) = measure(&ctx, &plain, true);
        let (lit, allocated) = measure(&ctx, &bloomed, false);
        let (mixed, _) = measure(&ctx, &both, false);

        assert!(
            allocated.is_some(),
            "the bloomed run allocated no chain, so its number is not the bloom's cost"
        );

        let (worst, changed) =
            light_that_left_the_source(&shoot(&ctx, &plain), &shoot(&ctx, &bloomed));

        println!("{tier:?} — {name} via {backend:?}");
        println!("  one pass          {one:.3} ms/frame");
        println!(
            "  two pass          {two:.3} ms/frame  ({:+.3} target + resolve)",
            two - one
        );
        println!("  bloomed           {lit:.3} ms/frame");
        println!(
            "  bloom             {:+.3} ms/frame ({:+.1}% of the one-pass frame)",
            lit - two,
            (lit - two) / one * 100.0
        );
        println!("  bloomed + blurred {mixed:.3} ms/frame");
        println!(
            "  blur on top       {:+.3} ms/frame (compare blur_cost's chain figure)",
            mixed - lit
        );
        println!(
            "  total             {:+.3} ms/frame against the 8.33 ms budget ({:.1}% of it)",
            lit - one,
            (lit - one) / 8.33 * 100.0
        );
        println!("  visible           {worst}/255 worst change, {changed} px, outside the source");

        let (before, after) = contrast_cost_of_blooming_a_mark(&ctx);
        println!(
            "  if ink bloomed    {before:.2}:1 -> {after:.2}:1 against its own ground \
             ({:+.1}%)\n",
            (after - before) / before * 100.0
        );
    }
}
