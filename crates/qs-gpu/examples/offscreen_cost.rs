//! What the offscreen target costs, per tier, before anything is built on top of it.
//!
//! `cargo run --release -p qs-gpu --example offscreen_cost`
//!
//! # Why this runs before the first effect
//!
//! The roadmap chunk's acceptance asks for "a cost measurement on the low tier before
//! anything is built on top", and the order is the point. Measured afterwards, the number is
//! the blur's cost and the target's cost added together with no way to separate them, and the
//! decision it would inform — whether a neighbourhood effect is affordable on the Reduced
//! tier at all — has already been made.
//!
//! # What it measures, and what it does not
//!
//! Wall clock from `submit` to the queue going idle, over many frames, for the same draw list
//! rendered both ways. The difference is one full-viewport render-target write plus one
//! full-viewport textured triangle.
//!
//! It is **not** a GPU timestamp. `GpuTimer` reports the instance pass only, and the resolve
//! deliberately does not carry a second timestamp pair — see `batcher.rs`. Wall clock around a
//! synchronous poll includes submit overhead and driver scheduling, so treat the absolute
//! numbers as an upper bound and the *difference between the two paths* as the answer, which
//! is the quantity the decision actually turns on.
//!
//! The Reduced tier is reached by asking for its backends directly rather than through a
//! forced-tier switch, because `QS_FORCE_TIER` does not exist yet (spec 002 T008). On a
//! machine whose GL driver is a translation layer over the same hardware, "the low tier" here
//! means the low *API*, not low-end hardware — an honest limit, and stated rather than
//! buried.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::cast_precision_loss
)]

use qs_gpu::batcher::Renderer;
use qs_gpu::color::Srgba;
use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::frame::{DrawList, Instance};
use qs_gpu::path::RenderPath;
use qs_gpu::target::OffscreenTarget;

/// 1080p, because that is what most of these machines actually run at, and the target's cost
/// is a function of area.
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FRAMES: u32 = 120;
const WARMUP: u32 = 20;

/// A frame with roughly what a scrolling list puts on screen: a ground, banded rows, a
/// selection and a bar's worth of chrome.
fn scene() -> DrawList {
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
        // A little text-shaped chrome per row, so the instance count is in the right order of
        // magnitude rather than a dozen.
        for c in 0..20 {
            list.instances.push(Instance::rect(
                40.0 + c as f32 * 36.0,
                y + 8.0,
                26.0,
                12.0,
                2.0,
                Srgba::new(0.80, 0.82, 0.88, 0.9),
            ));
        }
    }
    list.end_batch(None, false);
    list
}

fn measure(ctx: &GpuContext, list: &DrawList, force: bool) -> f64 {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen_cost surface"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ctx.capabilities.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = Renderer::new(ctx, 512);
    renderer.force_offscreen(force);

    for _ in 0..WARMUP {
        let cmd = renderer.render(ctx, &view, list, None);
        ctx.queue.submit([cmd]);
    }
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let start = std::time::Instant::now();
    for _ in 0..FRAMES {
        let cmd = renderer.render(ctx, &view, list, None);
        ctx.queue.submit([cmd]);
    }
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let elapsed = start.elapsed();

    if force {
        assert_eq!(
            renderer.offscreen_allocations(),
            1,
            "the target must be allocated once, not once per frame"
        );
    }
    elapsed.as_secs_f64() * 1000.0 / f64::from(FRAMES)
}

fn main() {
    println!(
        // Decimal MB, matching `OffscreenTarget`'s doc comments and the roadmap. Printing
        // MiB here and MB there is how two correct numbers start looking like a discrepancy.
        "offscreen target at {WIDTH}x{HEIGHT}: {:.1} MB ({} bytes)\n",
        OffscreenTarget::bytes_at([WIDTH, HEIGHT]) as f64 / 1_000_000.0,
        OffscreenTarget::bytes_at([WIDTH, HEIGHT])
    );

    let list = scene();
    println!(
        "{} instances per frame, {FRAMES} frames per run\n",
        list.instances.len()
    );

    for tier in [RenderPath::Primary, RenderPath::Reduced] {
        let instance = new_instance(tier);
        let Ok(ctx) = GpuContext::new(tier, instance, None) else {
            println!("{tier:?}: no usable adapter, skipped");
            continue;
        };
        let name = ctx.capabilities.adapter_name.clone();
        let backend = ctx.capabilities.backend;

        let one = measure(&ctx, &list, false);
        let two = measure(&ctx, &list, true);

        println!("{tier:?} — {name} via {backend:?}");
        println!("  one pass  {one:.3} ms/frame");
        println!("  two pass  {two:.3} ms/frame");
        println!(
            "  resolve   {:+.3} ms/frame ({:+.1}%)\n",
            two - one,
            (two - one) / one * 100.0
        );
    }
}
