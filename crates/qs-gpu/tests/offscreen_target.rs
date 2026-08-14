//! The offscreen target's one load-bearing claim: taking the two-pass path changes nothing.
//!
//! # Why this test is the chunk
//!
//! `crates/qs-gpu/src/target.rs` adds a second render target and a resolve pass. Every effect
//! that needs neighbouring pixels — blur, bloom, refraction — is built on it, and every one of
//! them will be authored by looking at the result. So if the resolve is not *exactly* a copy,
//! all three are authored against an image that differs from the single-pass one by an amount
//! nobody measured, and the difference is attributed to the effect.
//!
//! The failure modes this catches are all silent ones. A linear sampler with a half-texel
//! offset blurs the frame by a hair. A target in a non-sRGB format makes the resolve a gamma
//! conversion. A blend state on the resolve pipeline composites every translucent pixel twice.
//! An unflipped `v` axis renders the frame upside down — which is the one that is *not* silent,
//! and is here for completeness rather than for fear.
//!
//! # Why it forces the path
//!
//! No shipped primitive returns `true` from `PrimKind::needs_backdrop` yet: the infrastructure
//! deliberately landed before the first effect that uses it, so its memory cost, resize
//! behaviour and tier answer were decided in the open. `Renderer::force_offscreen` is what
//! keeps the path from being unreachable, and therefore untested, in the meantime. Without it
//! the blur chunk would be debugging the target and the blur at the same time.

// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the vocabulary of a
// test, not a hazard in one.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use qs_gpu::batcher::Renderer;
use qs_gpu::color::Srgba;
use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::frame::{DrawList, Environment, Instance, PrimKind};
use qs_gpu::path::RenderPath;
use qs_gpu::target::{OffscreenTarget, tier_can_hold_target};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

/// A context, or `None` on a machine with no usable adapter.
///
/// Skipped rather than failed. A headless CI runner without a GPU is a real configuration,
/// and a test that fails there teaches people to ignore the suite. The skip prints, so a run
/// that silently covered nothing is distinguishable from one that passed.
fn context() -> Option<GpuContext> {
    let instance = new_instance(RenderPath::Primary);
    match GpuContext::new(RenderPath::Primary, instance, None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("offscreen_target: SKIPPED, no usable adapter ({e})");
            None
        }
    }
}

/// A draw list with enough different primitives that a colour-space or blend mistake shows.
///
/// Opaque fills alone would survive a doubled composite unchanged, and that is one of the
/// mistakes worth catching — so there are translucent overlaps here on purpose, and a
/// gradient, because a ramp is where a gamma error is most visible.
fn scene() -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.09, 0.09, 0.11, 1.0), 1);
    list.set_environment(Environment {
        horizon: Srgba::new(0.12, 0.12, 0.15, 1.0),
        zenith: Srgba::new(0.30, 0.31, 0.36, 1.0),
    });

    list.instances.push(Instance::rect(
        16.0,
        16.0,
        120.0,
        80.0,
        8.0,
        Srgba::new(0.85, 0.32, 0.20, 1.0),
    ));
    // Translucent, and overlapping the one above: a resolve that blends rather than replaces
    // composites this twice and darkens exactly here.
    list.instances.push(Instance::rect(
        80.0,
        48.0,
        120.0,
        80.0,
        12.0,
        Srgba::new(0.20, 0.55, 0.90, 0.55),
    ));
    list.instances.push(Instance::gradient(
        24.0,
        120.0,
        200.0,
        50.0,
        6.0,
        0.7,
        Srgba::new(0.95, 0.90, 0.30, 1.0),
        Srgba::new(0.10, 0.70, 0.55, 1.0),
    ));
    list.end_batch(None, false);
    list
}

fn draw(ctx: &GpuContext, renderer: &mut Renderer, list: &DrawList) -> Vec<u8> {
    draw_into(ctx, renderer, list, [WIDTH, HEIGHT])
}

fn draw_into(
    ctx: &GpuContext,
    renderer: &mut Renderer,
    list: &DrawList,
    [width, height]: [u32; 2],
) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen_target test surface"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ctx.capabilities.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let commands = renderer.render(ctx, &view, list, None, None);

    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen_target readback"),
        size: u64::from(padded) * u64::from(height),
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
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([commands, encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    buffer.unmap();
    pixels
}

#[test]
fn the_two_pass_path_is_pixel_identical_to_the_one_pass_path() {
    let Some(ctx) = context() else { return };
    let list = scene();

    let mut direct = Renderer::new(&ctx, 512);
    let one_pass = draw(&ctx, &mut direct, &list);

    let mut through_target = Renderer::new(&ctx, 512);
    through_target.force_offscreen(true);
    let two_pass = draw(&ctx, &mut through_target, &list);

    assert_eq!(one_pass.len(), two_pass.len());

    // Exact, not "close". A tolerance here would be a tolerance every effect built on this
    // target inherits, and the whole reason the resolve was written as a nearest-sampled
    // unblended 1:1 blit is so that no tolerance is needed.
    let differing: Vec<usize> = one_pass
        .iter()
        .zip(&two_pass)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();

    assert!(
        differing.is_empty(),
        "{} of {} bytes differ (first at byte {}, texel {},{}): the resolve is not a copy. \
         A linear sampler offset, a blend state on the resolve pipeline, a mismatched target \
         format and an unflipped v axis all land here.",
        differing.len(),
        one_pass.len(),
        differing[0],
        (differing[0] / 4) % WIDTH as usize,
        (differing[0] / 4) / WIDTH as usize,
    );
}

#[test]
fn the_target_is_not_allocated_until_a_frame_needs_it() {
    // Acceptance: "the existing single-pass path stays the path when no effect asks for the
    // target". Stated as memory rather than as a code path, because that is the part a user
    // pays for: an installation that never enables a neighbourhood effect must not be
    // carrying 33 MB at 4K on the chance that it might.
    let Some(ctx) = context() else { return };
    let list = scene();

    let mut renderer = Renderer::new(&ctx, 512);
    assert!(renderer.offscreen().is_none(), "nothing has rendered yet");

    draw(&ctx, &mut renderer, &list);
    assert!(
        renderer.offscreen().is_none(),
        "no shipped primitive needs a backdrop, so a frame must not have allocated one"
    );

    renderer.force_offscreen(true);
    draw(&ctx, &mut renderer, &list);
    let allocated = renderer.offscreen().expect("the forced path allocates");
    assert_eq!(allocated.size(), [WIDTH, HEIGHT]);
    assert_eq!(allocated.bytes(), u64::from(WIDTH) * u64::from(HEIGHT) * 4);
}

#[test]
fn a_repeat_frame_reuses_the_target_and_a_resize_replaces_it() {
    // The resize half of the stated lifetime, in both directions. Reallocating every frame
    // is invisible in a screenshot and expensive in a profile; reallocating never means
    // sampling outside the texture after a window grows.
    let Some(ctx) = context() else { return };

    let mut renderer = Renderer::new(&ctx, 512);
    renderer.force_offscreen(true);

    let list = scene();
    draw(&ctx, &mut renderer, &list);
    assert_eq!(renderer.offscreen_allocations(), 1);

    for _ in 0..5 {
        draw(&ctx, &mut renderer, &list);
    }
    assert_eq!(
        renderer.offscreen_allocations(),
        1,
        "an unchanged viewport must reuse the target, not reallocate it every frame"
    );

    let mut bigger = DrawList::default();
    bigger.reset([WIDTH * 2, HEIGHT], Srgba::new(0.09, 0.09, 0.11, 1.0), 2);
    bigger.end_batch(None, false);
    draw_into(&ctx, &mut renderer, &bigger, [WIDTH * 2, HEIGHT]);
    assert_eq!(renderer.offscreen_allocations(), 2, "a resize reallocates");
    assert_eq!(
        renderer.offscreen().expect("allocated").size(),
        [WIDTH * 2, HEIGHT]
    );
}

#[test]
fn the_target_survives_a_frame_that_does_not_need_it() {
    // Freeing on the first frame with no blurred surface would reallocate on the next one
    // that has one -- once per scroll past a popover, which is the worst possible cadence for
    // a 33 MB allocation.
    let Some(ctx) = context() else { return };
    let list = scene();

    let mut renderer = Renderer::new(&ctx, 512);
    renderer.force_offscreen(true);
    draw(&ctx, &mut renderer, &list);
    assert!(renderer.offscreen().is_some());

    renderer.force_offscreen(false);
    draw(&ctx, &mut renderer, &list);
    assert!(
        renderer.offscreen().is_some(),
        "the target is kept across frames that do not use it"
    );
}

#[test]
fn the_tier_answer_is_a_floor_and_not_a_slower_target() {
    // No GPU needed. Restated in the integration suite because it is the decision the chunk
    // exists to make in the open, and `tiny-skia` acquiring a target chain is exactly the
    // kind of change that would be made locally inside whichever effect chunk wanted it.
    assert!(tier_can_hold_target(RenderPath::Primary));
    assert!(tier_can_hold_target(RenderPath::Reduced));
    assert!(!tier_can_hold_target(RenderPath::Cpu));

    // And the obligation that falls on any effect built on the target: the CPU tier cannot
    // draw it, so it must name what it becomes there.
    for kind in PrimKind::ALL {
        if kind.needs_backdrop() {
            assert!(
                !matches!(kind.fidelity(), qs_gpu::frame::Fidelity::Exact),
                "{kind:?} samples the backdrop, which the CPU tier has no way to do, so it \
                 cannot claim the CPU tier reproduces it exactly"
            );
        }
    }
}

#[test]
fn the_memory_cost_is_the_one_that_was_stated() {
    // Acceptance: "memory cost stated at the resolutions the application actually runs at,
    // not asymptotically". 8.3 MB at 1080p and 33.2 MB at 4K, counted against the same GPU
    // allocation ceiling as everything else.
    assert_eq!(OffscreenTarget::bytes_at([1920, 1080]), 8_294_400);
    assert_eq!(OffscreenTarget::bytes_at([2560, 1440]), 14_745_600);
    assert_eq!(OffscreenTarget::bytes_at([3840, 2160]), 33_177_600);
}
