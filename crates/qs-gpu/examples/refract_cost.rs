//! What refraction costs per frame, and whether it costs what it claims to.
//!
//! `cargo run --release -p qs-gpu --example refract_cost`
//!
//! # What it answers, and what it turned out not to be able to
//!
//! The acceptance criterion is *"frame cost measured with a full-width panel"*, written against
//! a specific fear: an unbounded march in a UI fragment shader is a frame-time cliff waiting for
//! a wide panel. **The first run answers that** — a full-width refracting panel, against two
//! controls taken on the same machine in the same run rather than read off another report:
//! **one pass** (no target at all) and **two pass** (`force_offscreen`, so the whole frame goes
//! through the target and the resolve). Their difference is the target's cost, already paid and
//! already argued for by `offscreen-render-target`; folding it in would overstate this effect by
//! whatever the target costs.
//!
//! The other three runs were built to answer a second question — whether the cost follows the
//! **bevel band** (which is where the march actually runs, since it early-outs where the rim
//! profile is zero) or the **panel's area**. `deep` holds area exactly fixed and grows the band
//! 4.4x; `broad` holds the bevel fixed and doubles the area while barely moving the band. Cost
//! following the area column would be the cliff; cost following the band column would be the
//! shape `PrimKind::Refract` originally claimed in prose.
//!
//! **It is neither, because the question is below this harness's floor.** The differences are
//! 0.004 ms against per-configuration spreads of 0.004 ms, and single-sample versions of this
//! example reported the two ratios as 1.28/1.66 and then 1.81/1.99 from the identical run
//! immediately afterwards. The example now prints its own spread and says so rather than
//! offering a ratio; the absolute column is stable and is what the budget cares about. See
//! `docs/refraction/cost.md`, and `PrimKind::Refract`, whose documentation was corrected to
//! match this rather than the other way round.
//!
//! # What it measures, and what it does not
//!
//! `offscreen_cost`'s caveats, unchanged: wall clock from submit to the queue going idle, not a
//! GPU timestamp. Treat the absolute figures as an upper bound and the differences as the
//! answer. The Reduced tier is reached by asking for its backends directly, so on a machine
//! whose GL driver is a translation layer over the same hardware "the low tier" means the low
//! *API*, not low-end hardware.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::indexing_slicing,
    clippy::cast_precision_loss
)]

use qs_gpu::batcher::Renderer;
use qs_gpu::color::Srgba;
use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::frame::{DrawList, Instance};
use qs_gpu::path::RenderPath;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
/// Frames per measurement run.
///
/// Refraction changes frame time by only a few microseconds on the measured
/// hardware. Longer runs reduce the scheduling noise that dominated runs of
/// 120 frames.
const FRAMES: u32 = 2000;
const WARMUP: u32 = 200;
/// How many interleaved sweeps. Odd, so the median is a sample rather than a mean of two.
const REPS: u32 = 5;

/// A grid of `cols` by `rows` panels, each `w` by `h` with the given bevel.
///
/// Multiple panels make small differences easier to measure above timing noise.
#[derive(Clone, Copy)]
struct Grid {
    cols: u32,
    rows: u32,
    w: f32,
    h: f32,
    bevel: f32,
}

impl Grid {
    /// Fragments inside the panels, and fragments inside their bevel bands.
    ///
    /// Computed rather than measured, and deliberately crude -- the corners are left as
    /// squares. What it is for is the two ratios in the report: the runs are chosen so that
    /// area and band move independently, and the milliseconds have to follow one of them.
    fn areas(self) -> (f64, f64) {
        let (w, h, b) = (f64::from(self.w), f64::from(self.h), f64::from(self.bevel));
        let n = f64::from(self.cols * self.rows);
        let area = w * h;
        let band = area - (w - 2.0 * b).max(0.0) * (h - 2.0 * b).max(0.0);
        (area * n, band * n)
    }
}

/// `blur_cost`'s scene, so the one-pass numbers are comparable across the two examples: a
/// ground, banded rows, and text-shaped chrome at the right order of magnitude.
fn scene(panels: Option<Grid>) -> DrawList {
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
    if let Some(g) = panels {
        // Spread over the viewport rather than stacked, so no panel is occluded by another and
        // every one of them shades every fragment it covers. Overlapping them would let early
        // depth rejection -- which this pipeline does not have, but a driver might -- decide
        // how much of the experiment actually runs.
        let gap_x = (WIDTH as f32 - g.cols as f32 * g.w) / (g.cols + 1) as f32;
        let gap_y = (HEIGHT as f32 - g.rows as f32 * g.h) / (g.rows + 1) as f32;
        for r in 0..g.rows {
            for c in 0..g.cols {
                list.instances.push(Instance::refract(
                    gap_x + c as f32 * (g.w + gap_x),
                    gap_y + r as f32 * (g.h + gap_y),
                    g.w,
                    g.h,
                    14.0,
                    g.bevel,
                    0.32,
                    0.0,
                    0.55,
                    1.0,
                    Srgba::new(0.196, 0.196, 0.212, 1.0),
                ));
            }
        }
    }
    list.end_batch(None, false);
    list
}

fn measure(ctx: &GpuContext, list: &DrawList, force: bool) -> f64 {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("refract_cost surface"),
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

    assert!(
        force || renderer.blur().is_none(),
        "a refracting frame allocated the blur chain; the chain is the blur's and nothing here \
         samples it, so this number would be measuring three passes nobody asked for"
    );
    elapsed.as_secs_f64() * 1000.0 / f64::from(FRAMES)
}

fn main() {
    // What ships: one inspector-sized panel, full height less the chrome. This run answers the
    // acceptance criterion -- what a full-width panel costs -- and nothing else; the three
    // below answer where that cost comes from.
    let shipped = Grid {
        cols: 1,
        rows: 1,
        w: WIDTH as f32 - 96.0,
        h: HEIGHT as f32 - 96.0,
        bevel: 18.0,
    };
    // The discriminating trio. `deep` holds the area EXACTLY fixed against `base` and grows the
    // band 4.4x; `broad` holds the bevel fixed and doubles the area while barely moving the
    // band. Cost following the band column is the shape this primitive claims; cost following
    // the area column is the frame-time cliff a full-face effect would have.
    let base = Grid {
        cols: 8,
        rows: 3,
        w: 200.0,
        h: 200.0,
        bevel: 6.0,
    };
    let deep = Grid {
        bevel: 30.0,
        ..base
    };
    let broad = Grid { w: 400.0, ..base };

    let plain = scene(None);
    println!(
        "{} instances, {FRAMES} frames per run, {WIDTH}x{HEIGHT}\n",
        plain.instances.len()
    );

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

        let one = measure(&ctx, &plain, false);
        let two = measure(&ctx, &plain, true);
        println!("  one pass                 {one:.3} ms/frame");
        println!(
            "  two pass                 {two:.3} ms/frame  ({:+.3} target + resolve, the control)",
            two - one
        );

        // Measured ONCE per configuration and reused for both the table and the claim. The
        // first version of this example re-measured for the ratio line, and the two numbers
        // for the same scene disagreed by more than the effect -- which is its own reading of
        // the resolution problem, and the reason `FRAMES` is what it is.
        // Every configuration measured REPS times, interleaved -- one full sweep, then the
        // next -- rather than all the reps of one configuration together. Two reasons, both
        // recorded elsewhere in this repository. `findings.md` 2.7: a long-lived process drifts
        // roughly 40% over twenty scenario runs, so consecutive reps of one scene sample a
        // different machine from the reps of the next. And the first version of this example
        // took one sample each and reported ratios of 1.28 and 1.66; the identical run
        // immediately afterwards reported 1.81 and 1.99, on the same machine with nothing
        // changed. A single sample here is not a small measurement, it is none.
        let configs = [
            ("full-width panel", shipped),
            ("24 panels, bevel  6", base),
            ("24 panels, bevel 30", deep),
            ("24 wide panels, bevel 6", broad),
        ];
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
        for _ in 0..REPS {
            for (i, (_, g)) in configs.iter().enumerate() {
                samples[i].push(measure(&ctx, &scene(Some(*g)), false) - two);
            }
        }
        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN from a timer"));
            s[s.len() / 2]
        };
        let spread = |v: &[f64]| {
            let (lo, hi) = v
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
            hi - lo
        };

        let runs: Vec<(&str, Grid, f64)> = configs
            .iter()
            .enumerate()
            .map(|(i, (label, g))| (*label, *g, median(&samples[i])))
            .collect();

        for (i, (label, g, cost)) in runs.iter().enumerate() {
            let (area, band) = g.areas();
            println!(
                "  {label:<24} {:+.3} ms median, spread {:.3}   area {:>5.2} Mpx, \
                 band {:>5.2} Mpx",
                cost,
                spread(&samples[i]),
                area / 1e6,
                band / 1e6,
            );
        }

        // Whether the ratios below mean anything at all. The two comparisons differ by 0.005
        // to 0.010 ms; if any configuration's own spread is that large, a ratio between two of
        // them is a ratio between two samples of the same noise.
        let worst_spread = (0..configs.len())
            .map(|i| spread(&samples[i]))
            .fold(0.0_f64, f64::max);
        let smallest_difference = (runs[2].2 - runs[1].2)
            .abs()
            .min((runs[3].2 - runs[1].2).abs());
        let resolved = worst_spread < smallest_difference;

        let (base_area, base_band) = base.areas();
        let (deep_area, deep_band) = deep.areas();
        let (broad_area, broad_band) = broad.areas();
        let (shipped_cost, base_cost) = (runs[0].2, runs[1].2);
        let (deep_cost, broad_cost) = (runs[2].2, runs[3].2);
        println!(
            "  deeper bevel: area x{:.2}, band x{:.2}  -> cost x{:.2}",
            deep_area / base_area,
            deep_band / base_band,
            deep_cost / base_cost.max(1e-9),
        );
        println!(
            "  wider panels: area x{:.2}, band x{:.2}  -> cost x{:.2}",
            broad_area / base_area,
            broad_band / base_band,
            broad_cost / base_cost.max(1e-9),
        );
        println!(
            "  the shipped full-width panel costs {:+.3} ms against the 8.33 ms budget \
             ({:.2}% of it)",
            shipped_cost,
            shipped_cost / 8.33 * 100.0
        );
        if !resolved {
            println!(
                "  !! the two ratios above are NOT RESOLVED: the worst per-configuration \
                 spread is {worst_spread:.3} ms and the difference they are built from is \
                 {smallest_difference:.3} ms. Read the absolute column, which is stable, and \
                 do not attribute the cost to area or to band from this run."
            );
        }
        // A run whose numbers cannot be told apart says so, rather than letting a reader take
        // a ratio of two samples of the same noise as a finding.
        // Two impossibilities, and the second is the one the first version of this example
        // missed: every cost must be positive (a panel cannot be cheaper than no panel), AND
        // strictly more work cannot cost strictly less. Both comparison runs are supersets of
        // `base` -- same panels, plus band or plus area -- so a ratio below one is the harness
        // reporting on itself. Checking only the first left the GL tier printing "band x4.38 ->
        // cost x0.43" as though it were a finding.
        if base_cost <= 0.0
            || deep_cost <= 0.0
            || broad_cost <= 0.0
            || deep_cost < base_cost
            || broad_cost < base_cost
        {
            println!(
                "  !! UNRESOLVED: a run with strictly more work came out no dearer than one \
                 with less, which no model allows. These differences are below this harness's \
                 floor -- raise FRAMES, or read the absolute column and ignore the ratios."
            );
        }
        println!();
    }
}
