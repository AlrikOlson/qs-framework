//! What a full-viewport fragment pass costs, per tier.
//!
//! `cargo run --release -p qs-gpu --example field_cost`
//!
//! # Why this exists
//!
//! `prim-field-wash`'s acceptance is explicit: *"frame time measured against the M0 budget on
//! the low tier before this is called done, because a full-viewport fragment pass is the first
//! effect here that could plausibly cost a frame"*. Every primitive before it was bounded by
//! the component that asked for it — a row's halo costs a row's worth of fragments. The field
//! covers the window, so its cost is fragment work, and fragment work at 1080p is two million
//! invocations per pass whatever the instance count says.
//!
//! # What it measures
//!
//! The same scene four ways: with a flat ground, with the field's ground, with the conic sweep
//! over it, and with both — which is what `surface/canvas` actually ships. The difference
//! between the first and the last is the number the decision turns on, and it is reported
//! against the 8.33 ms budget SC-001 gates rather than as a bare millisecond count, because
//! "0.3 ms" means nothing without the frame it is a fraction of.
//!
//! Wall clock from `submit` to the queue going idle, over many frames, exactly as
//! `offscreen_cost` does it and with the same honest limit: this includes submit overhead and
//! driver scheduling, so treat the absolute numbers as an upper bound and the *differences*
//! as the answer.
//!
//! The Reduced tier is reached by asking for its backends directly rather than through a
//! forced-tier switch, for the reason `offscreen_cost` states — on a machine whose GL driver
//! is a translation layer over the same hardware, "the low tier" here means the low *API* and
//! not low-end hardware. The CPU tier is not measured and does not need to be: the field is
//! `Fidelity::Enhanced` and `tiny-skia` draws its floor, a flat rectangle, which is what the
//! ground cost before any of this.

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
use qs_gpu::frame::{DrawList, FieldCentre, FieldWash, Instance};
use qs_gpu::path::RenderPath;

/// 1080p, because the cost of a full-viewport pass is a function of area and this is what most
/// of these machines actually run at.
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FRAMES: u32 = 240;
const WARMUP: u32 = 40;
/// Timed passes over the whole set. See [`measure_all`] for why more than one.
const ROUNDS: u32 = 3;

/// The frame budget SC-001 gates, in milliseconds: 120 Hz.
const BUDGET_MS: f64 = 8.33;

/// Roughly what the shipped `surface/canvas` carries, so the numbers are about the palette
/// that ships rather than about a stress test nobody will see.
fn shipped_field() -> FieldWash {
    let centre = |at: [f32; 2], drift: [f32; 2], reach: f32, phase: f32, tint: Srgba| FieldCentre {
        at,
        drift,
        reach,
        phase,
        tint,
    };
    FieldWash::new(&[
        centre(
            [0.14, 0.10],
            [0.07, 0.05],
            0.62,
            0.0,
            Srgba::new(0.93, 0.94, 0.99, 1.0),
        ),
        centre(
            [0.88, 0.20],
            [0.06, 0.06],
            0.56,
            0.31,
            Srgba::new(0.99, 0.96, 0.90, 1.0),
        ),
        centre(
            [0.22, 0.88],
            [0.05, 0.07],
            0.58,
            0.62,
            Srgba::new(0.90, 0.97, 0.93, 1.0),
        ),
        centre(
            [0.80, 0.78],
            [0.06, 0.04],
            0.54,
            0.85,
            Srgba::new(0.89, 0.91, 0.99, 1.0),
        ),
    ])
}

/// Which full-viewport layers the ground is built from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ground {
    /// What the window had before any of this: one flat rectangle.
    Flat,
    /// The field alone.
    Field,
    /// The conic sweep alone, which is the other full-viewport layer the canvas carries.
    Sweep,
    /// Both, which is what `surface/canvas` ships.
    Shipped,
}

impl Ground {
    fn label(self) -> &'static str {
        match self {
            Self::Flat => "flat rect (the old ground)",
            Self::Field => "field only",
            Self::Sweep => "sweep only",
            Self::Shipped => "field + sweep (shipped)",
        }
    }
}

/// A frame with roughly what a scrolling list puts on screen, over the given ground.
fn scene(ground: Ground) -> DrawList {
    let base = Srgba::new(0.10, 0.10, 0.13, 1.0);
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], base, 1);
    list.set_field(shipped_field());

    let (w, h) = (WIDTH as f32, HEIGHT as f32);
    match ground {
        Ground::Flat => list
            .instances
            .push(Instance::rect(0.0, 0.0, w, h, 0.0, base)),
        Ground::Field => list
            .instances
            .push(Instance::field(0.0, 0.0, w, h, 0.6, 0.0, base)),
        Ground::Sweep => list.instances.push(Instance::sweep(
            0.0,
            0.0,
            w,
            h,
            0.0,
            0.0,
            base,
            Srgba::new(0.12, 0.12, 0.15, 0.55),
        )),
        Ground::Shipped => {
            list.instances
                .push(Instance::field(0.0, 0.0, w, h, 0.6, 0.0, base));
            list.instances.push(Instance::sweep(
                0.0,
                0.0,
                w,
                h,
                0.0,
                0.0,
                base,
                Srgba::new(0.12, 0.12, 0.15, 0.55),
            ));
        }
    }

    // The list on top of it, so the ground is measured under the overdraw it actually sits
    // under rather than on an empty screen.
    let row_h = 28.0;
    let rows = (h / row_h) as u32;
    for i in 0..rows {
        let y = i as f32 * row_h;
        if i % 2 == 1 {
            list.instances.push(Instance::rect(
                0.0,
                y,
                w,
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
    list.end_batch(None, false);
    list
}

fn run(
    ctx: &GpuContext,
    renderer: &mut Renderer,
    view: &wgpu::TextureView,
    list: &DrawList,
) -> f64 {
    for _ in 0..WARMUP {
        let cmd = renderer.render(ctx, view, list, None);
        ctx.queue.submit([cmd]);
    }
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let start = std::time::Instant::now();
    for _ in 0..FRAMES {
        let cmd = renderer.render(ctx, view, list, None);
        ctx.queue.submit([cmd]);
    }
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    start.elapsed().as_secs_f64() * 1000.0 / f64::from(FRAMES)
}

/// Every ground on one tier, as the **minimum** over several interleaved rounds.
///
/// Both halves of that are load-bearing, and the first version of this harness had neither.
/// It built a fresh `Renderer` per ground and measured each one once, in order — so the first
/// ground measured paid for a pipeline compile and whatever else the driver does once, and on
/// the GL tier that one-off cost was larger than the thing being measured. The flat ground
/// came out *slower* than the field, and a full-viewport pass appeared to cost -0.18 ms.
///
/// A negative cost is not a small cost, it is a broken measurement, and it would have been
/// very easy to read as "the field is free on the low tier" — which is the conclusion this
/// chunk's acceptance most wanted checked. So: one renderer shared by every ground, so the
/// pipeline is compiled before any timing starts; round-robin, so a drift in clocks or thermals
/// lands on all four rather than on whichever went first; and the minimum rather than the mean,
/// because the quantity of interest is what the frame costs when nothing else interferes and
/// every source of noise here is additive.
fn measure_all(ctx: &GpuContext, scenes: &[(Ground, DrawList)], rounds: u32) -> Vec<f64> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("field_cost surface"),
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

    // One untimed pass over every scene first, so nothing below is the first of anything.
    for (_, list) in scenes {
        run(ctx, &mut renderer, &view, list);
    }

    let mut best = vec![f64::MAX; scenes.len()];
    for _ in 0..rounds {
        for (slot, (_, list)) in best.iter_mut().zip(scenes) {
            *slot = slot.min(run(ctx, &mut renderer, &view, list));
        }
    }
    best
}

fn main() {
    println!("full-viewport ground at {WIDTH}x{HEIGHT}, {FRAMES} frames per run");
    println!("budget: {BUDGET_MS} ms/frame (SC-001, 120 Hz)\n");

    // `Flat` first, and the reporting below depends on it: it is the baseline every other
    // ground is quoted against.
    let grounds = [Ground::Flat, Ground::Field, Ground::Sweep, Ground::Shipped];
    let scenes: Vec<(Ground, DrawList)> = grounds.iter().map(|g| (*g, scene(*g))).collect();
    let Some((_, first)) = scenes.first() else {
        return;
    };
    println!("{} instances per frame\n", first.instances.len());

    for tier in [RenderPath::Primary, RenderPath::Reduced] {
        let instance = new_instance(tier);
        let Ok(ctx) = GpuContext::new(tier, instance, None) else {
            println!("{tier:?}: no usable adapter, skipped");
            continue;
        };
        println!(
            "{tier:?} — {} via {:?}",
            ctx.capabilities.adapter_name, ctx.capabilities.backend
        );

        let measured = measure_all(&ctx, &scenes, ROUNDS);
        let flat = measured.first().copied().unwrap_or(0.0);
        for ((ground, _), ms) in scenes.iter().zip(&measured) {
            if *ground == Ground::Flat {
                println!("  {:<26} {ms:.3} ms/frame", ground.label());
            } else {
                println!(
                    "  {:<26} {ms:.3} ms/frame  {:+.3} vs flat, {:.1}% of budget",
                    ground.label(),
                    ms - flat,
                    (ms - flat) / BUDGET_MS * 100.0
                );
            }
        }
        println!();
    }
}
