//! What the backdrop blur costs per frame with the panel open, per tier.
//!
//! `cargo run --release -p qs-gpu --example blur_cost`
//!
//! # The three numbers, and why all three are needed
//!
//! `offscreen_cost` already measured the target and its resolve against a one-pass frame. This
//! measures the same list three ways, so the blur's own cost separates from the target's:
//!
//! - **one pass** — the frame as it renders today, no target at all.
//! - **two pass** — `force_offscreen`, so the whole frame goes through the target and the
//!   resolve. The difference from one pass is the *target's* cost, which is `offscreen_cost`'s
//!   answer, repeated here so the third number has a control taken on the same machine in the
//!   same run rather than one read off a different report.
//! - **blurred** — a real panel in the list. The difference from two pass is the **chain**:
//!   one downsample and two Gaussian passes at a sixteenth of the viewport's fragment count,
//!   plus a second instance pass for the batches above the cut.
//!
//! Reporting only "blurred minus one pass" would fold the target's cost into the blur's, and
//! the target's was already paid and already argued for.
//!
//! # What it measures, and what it does not
//!
//! `offscreen_cost`'s caveats, unchanged and all of them: wall clock from submit to the queue
//! going idle, not a GPU timestamp; treat the absolute figures as an upper bound and the
//! differences as the answer. The Reduced tier is reached by asking for its backends directly,
//! so on a machine whose GL driver is a translation layer over the same hardware "the low tier"
//! means the low *API*, not low-end hardware.

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
use qs_gpu::target::{BlurChain, OffscreenTarget};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FRAMES: u32 = 120;
const WARMUP: u32 = 20;

/// `offscreen_cost`'s scene, so the two examples' one-pass numbers are comparable: a ground,
/// banded rows, and text-shaped chrome at the right order of magnitude.
fn scene(panel: bool) -> DrawList {
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
                Srgba::new(0.80, 0.82, 0.88, 0.9),
            ));
        }
    }
    if panel {
        // An inspector-sized panel: a third of the window, full height less the chrome. The
        // chain's cost does not depend on the panel's area -- it blurs the whole viewport
        // either way -- but the second instance pass and the panel's own fragments do, and a
        // popover-sized panel would understate both.
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

fn measure(ctx: &GpuContext, list: &DrawList, force: bool) -> (f64, Option<u64>) {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blur_cost surface"),
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

fn main() {
    let target = OffscreenTarget::bytes_at([WIDTH, HEIGHT]);
    let chain = BlurChain::bytes_at([WIDTH, HEIGHT]);
    println!("at {WIDTH}x{HEIGHT}:");
    println!(
        "  colour target  {:.1} MB ({target} bytes)",
        target as f64 / 1_000_000.0
    );
    println!(
        "  blur chain     {:.1} MB ({chain} bytes, both halves, {:.1}% of the target)\n",
        chain as f64 / 1_000_000.0,
        chain as f64 / target as f64 * 100.0
    );

    let plain = scene(false);
    let panelled = scene(true);
    println!(
        "{} instances, {FRAMES} frames per run\n",
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
        let (blurred, allocated) = measure(&ctx, &panelled, false);

        assert!(
            allocated.is_some(),
            "the panelled run allocated no chain, so its number is not the blur's cost"
        );

        println!("{tier:?} — {name} via {backend:?}");
        println!("  one pass  {one:.3} ms/frame");
        println!(
            "  two pass  {two:.3} ms/frame  ({:+.3} target + resolve)",
            two - one
        );
        println!("  blurred   {blurred:.3} ms/frame");
        println!(
            "  chain     {:+.3} ms/frame ({:+.1}% of the one-pass frame)",
            blurred - two,
            (blurred - two) / one * 100.0
        );
        println!(
            "  total     {:+.3} ms/frame against the 8.33 ms budget ({:.1}% of it)\n",
            blurred - one,
            (blurred - one) / 8.33 * 100.0
        );
    }
}
