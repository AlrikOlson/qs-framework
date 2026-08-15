//! The lit mode's three adapter claims, on real hardware (specs/002 T034, FR-004, SC-007).
//!
//! 1. **Mode off is byte-identical to the pre-feature frame.** With no scene, `render`
//!    encodes the command stream it always did — the lighting pipeline exists but never
//!    draws — so two unlit renders of one list are identical, and that pair *is* the
//!    pre-feature baseline: the baseline is not a recorded image from an old binary, it is
//!    the unlit path itself, which this test holds still.
//! 2. **A shadowless scene changes nothing.** A scene whose only slabs lie flat (the canvas
//!    at elevation zero) has nothing to cast and nothing to occlude, so the lighting draw
//!    runs and multiplies every pixel by one. Byte-identical to mode off — which is what
//!    makes "the mode is on but nothing is elevated" indistinguishable from off, exactly as
//!    designed.
//! 3. **An elevated caster changes pixels.** The proof the pass can go red: without it,
//!    claims 1 and 2 would also pass for a pipeline that never drew at all.

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
use qs_gpu::frame::{DrawList, Instance};
use qs_gpu::path::RenderPath;
use qs_gpu::scene::{Environment, Light, SceneList, Slab};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

fn context() -> Option<GpuContext> {
    let instance = new_instance(RenderPath::Primary);
    match GpuContext::new(RenderPath::Primary, instance, None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("lit_mode: SKIPPED, no usable adapter ({e})");
            None
        }
    }
}

fn list() -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.09, 0.09, 0.11, 1.0), 1);
    list.instances.push(Instance::rect(
        16.0,
        16.0,
        160.0,
        60.0,
        6.0,
        Srgba::new(0.20, 0.22, 0.27, 1.0),
    ));
    list.instances.push(Instance::rect(
        16.0,
        90.0,
        160.0,
        60.0,
        6.0,
        Srgba::new(0.16, 0.18, 0.22, 1.0),
    ));
    list.end_batch(None, false);
    list
}

fn ground() -> Slab {
    Slab {
        rect: [0.0, 0.0, WIDTH as f32, HEIGHT as f32],
        thickness: 1.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    }
}

fn scene(slabs: Vec<Slab>) -> SceneList {
    let mut scene = SceneList::default();
    scene.reset(1, Environment::default());
    for slab in slabs {
        scene.push(slab);
    }
    scene.key_light = Some(Light {
        vector: qs_gpu::frame::LIGHT_DIR,
        colour: [1.0, 1.0, 1.0],
        intensity: 0.72,
        size: 5.0,
    });
    scene
}

fn draw(
    ctx: &GpuContext,
    renderer: &mut Renderer,
    list: &DrawList,
    s: Option<&SceneList>,
) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lit_mode test surface"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
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
    // A validation error inside `render` drops the whole command buffer and reads back a
    // blank frame with nothing on stderr — the silent failure the batcher's own docs warn
    // about. The error scope turns it into a panic with the actual message.
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let commands = renderer.render(ctx, &view, list, s, None);
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("validation error while encoding the lit frame: {error}");
    }

    let unpadded = WIDTH * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lit_mode readback"),
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
    ctx.queue.submit([commands, encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for row in 0..HEIGHT {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    buffer.unmap();
    pixels
}

#[test]
fn the_mode_off_frame_is_the_frame_it_always_was() {
    // T034 / FR-004 / SC-007. Determinism first: the unlit path is the pre-feature path,
    // and holding it still across renders is what "byte-identical to the baseline" means
    // when the baseline is the path itself rather than a stale recording.
    let Some(ctx) = context() else { return };
    let list = list();
    let mut renderer = Renderer::new(&ctx, 512);
    let a = draw(&ctx, &mut renderer, &list, None);
    let b = draw(&ctx, &mut renderer, &list, None);
    assert_eq!(a, b, "two unlit renders of one list diverged");
}

#[test]
fn a_scene_with_nothing_elevated_changes_no_pixel() {
    // The lighting draw RUNS here — the scene is renderable — and multiplies every pixel
    // by exactly one, because a flat world casts nothing and occludes nothing. This is the
    // half that catches a pass that darkens unconditionally (a wrong AO ambient, a blend
    // state that double-composites).
    let Some(ctx) = context() else { return };
    let list = list();
    let mut renderer = Renderer::new(&ctx, 512);
    let unlit = draw(&ctx, &mut renderer, &list, None);
    let flat = scene(vec![ground()]);
    let lit = draw(&ctx, &mut renderer, &list, Some(&flat));
    assert_eq!(
        unlit, lit,
        "a shadowless scene changed pixels: the pass is adding or removing light a flat \
         world does not have"
    );
}

#[test]
fn an_elevated_caster_darkens_the_frame_it_stands_over() {
    // The can-go-red proof: without this, the two tests above also pass for a pipeline
    // that never draws. A slab standing 12 px over the ground must cast.
    let Some(ctx) = context() else { return };
    let list = list();
    let mut renderer = Renderer::new(&ctx, 512);
    let unlit = draw(&ctx, &mut renderer, &list, None);
    let raised = scene(vec![
        ground(),
        Slab {
            rect: [16.0, 16.0, 160.0, 60.0],
            radius: 6.0,
            elevation: 12.0,
            thickness: 12.0,
            attenuation_floor: 0.0,
            ..Slab::default()
        },
    ]);
    let lit = draw(&ctx, &mut renderer, &list, Some(&raised));
    assert_ne!(
        unlit, lit,
        "an elevated caster changed nothing: the lighting pass is not drawing"
    );

    // And the change is a darkening, not a repaint: no pixel got brighter (US1 has no
    // additive term), and at least one got darker.
    let mut darker = 0usize;
    for (a, b) in unlit.iter().zip(&lit) {
        assert!(
            b <= a,
            "US1's pass may only attenuate, but a byte increased"
        );
        if b < a {
            darker += 1;
        }
    }
    assert!(darker > 0);
}

#[test]
fn the_lighting_pipeline_compiles() {
    // The pipeline is created in `Renderer::new`, where a failure is an uncaptured error
    // that logs to a subscriber nobody installs — this scope makes it a red test with the
    // compiler's actual message instead of a blank frame two tests later.
    let Some(ctx) = context() else { return };
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _renderer = Renderer::new(&ctx, 512);
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("Renderer::new produced a validation error: {error}");
    }
}

#[test]
fn an_app_shaped_scene_leaves_unshadowed_rows_alone() {
    // The window regression probe: the real frame is a canvas at thickness zero under a
    // column of CONTIGUOUS row slabs at elevation 2 — no gaps, nothing tall. Nothing in it
    // casts onto anything, so the frame must be byte-identical to unlit, exactly like the
    // flat scene. The first lit window rendered black; whatever did that must fail here.
    let Some(ctx) = context() else { return };
    let list = list();
    let mut renderer = Renderer::new(&ctx, 512);
    let unlit = draw(&ctx, &mut renderer, &list, None);

    let mut slabs = vec![Slab {
        rect: [0.0, 0.0, WIDTH as f32, HEIGHT as f32],
        elevation: 0.0,
        thickness: 0.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    }];
    for i in 0..6 {
        slabs.push(Slab {
            rect: [0.0, i as f32 * 28.0, WIDTH as f32, 28.0],
            elevation: 2.0,
            thickness: 2.0,
            attenuation_floor: 0.0,
            ..Slab::default()
        });
    }
    let app_shaped = scene(slabs);
    let lit = draw(&ctx, &mut renderer, &list, Some(&app_shaped));

    // What MAY change: the thin penumbra band the row block's 2 px boundary genuinely
    // casts onto the canvas just past it (a 2 px step shadows ~2 px of ground — that is
    // "seams darken" working). What may NOT: row interiors, which have nothing above them.
    let px = |buf: &Vec<u8>, x: u32, y: u32| -> [u8; 4] {
        let i = ((y * WIDTH + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    };
    for (x, y) in [(64u32, 14u32), (128, 42), (200, 100), (30, 150)] {
        assert_eq!(
            px(&unlit, x, y),
            px(&lit, x, y),
            "a row interior at ({x}, {y}) changed with nothing standing over it"
        );
    }
    let differing = unlit.iter().zip(&lit).filter(|(a, b)| a != b).count();
    assert!(
        differing < (WIDTH as usize) * 4 * 6,
        "the boundary band should be a few pixel-rows of bytes, not {differing}"
    );
}

#[test]
fn the_exact_window_scene_replayed() {
    // The slab dump from the black window, replayed byte for byte: canvas 1920x1200 at
    // elevation 0/thickness 0, 28 contiguous 1274x42 rows at elevation 1.5. If this is
    // clean the shader is exonerated and the fault is in the app's invocation.
    let Some(ctx) = context() else { return };
    let mut list = DrawList::default();
    list.reset([1920, 1200], Srgba::new(0.09, 0.09, 0.11, 1.0), 1);
    list.instances.push(Instance::rect(
        100.0,
        100.0,
        1000.0,
        800.0,
        0.0,
        Srgba::new(0.20, 0.22, 0.27, 1.0),
    ));
    list.end_batch(None, false);

    let mut slabs = vec![Slab {
        rect: [0.0, 0.0, 1920.0, 1200.0],
        elevation: 0.0,
        thickness: 0.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    }];
    for i in 0..28 {
        slabs.push(Slab {
            rect: [0.0, 50.0 + i as f32 * 42.0, 1274.0, 42.0],
            elevation: 1.5,
            thickness: 1.5,
            attenuation_floor: 0.0,
            ..Slab::default()
        });
    }
    let replay = scene(slabs);

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("replay"),
        size: wgpu::Extent3d {
            width: 1920,
            height: 1200,
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
    let mut renderer = Renderer::new(&ctx, 512);
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let commands = renderer.render(&ctx, &view, &list, Some(&replay), None);
    ctx.queue.submit([commands]);
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("replay validation error: {error}");
    }
    // Read one interior row pixel back via a 1x1 copy.
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("replay 1px"),
        size: 256,
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
            origin: wgpu::Origin3d {
                x: 500,
                y: 500,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("range");
    let px = [mapped[0], mapped[1], mapped[2], mapped[3]];
    drop(mapped);
    assert!(
        px[0] > 10 || px[1] > 10 || px[2] > 10,
        "the replayed window scene rendered near-black at an interior pixel: {px:?}"
    );
}

#[test]
fn the_content_half_still_draws_after_the_lit_seam() {
    // The restoration path: after the lit draw switches pipelines, the content half must
    // draw exactly as it would have. Exercised with an untextured batch AFTER a textured
    // one (the cut keeps it in the content half), carrying a bright rect whose absence is
    // unmissable. The black window's symptom — dark surfaces, no content — is this test's
    // subject.
    let Some(ctx) = context() else { return };
    let mut lit_list = DrawList::default();
    lit_list.reset([WIDTH, HEIGHT], Srgba::new(0.09, 0.09, 0.11, 1.0), 1);
    lit_list.instances.push(Instance::rect(
        16.0,
        16.0,
        160.0,
        60.0,
        6.0,
        Srgba::new(0.20, 0.22, 0.27, 1.0),
    ));
    lit_list.end_batch(None, false);
    // An empty textured batch marks the cut, as the first glyph batch does in the app.
    lit_list.end_batch(None, true);
    lit_list.instances.push(Instance::rect(
        60.0,
        100.0,
        80.0,
        40.0,
        0.0,
        Srgba::new(1.0, 1.0, 1.0, 1.0),
    ));
    lit_list.end_batch(None, false);

    let mut renderer = Renderer::new(&ctx, 512);
    let raised = scene(vec![
        ground(),
        Slab {
            rect: [16.0, 16.0, 160.0, 60.0],
            radius: 6.0,
            elevation: 12.0,
            thickness: 12.0,
            attenuation_floor: 0.0,
            ..Slab::default()
        },
    ]);
    let lit = draw(&ctx, &mut renderer, &lit_list, Some(&raised));

    // The white rect's centre pixel, drawn after the seam, must be white.
    let i = ((120 * WIDTH + 100) * 4) as usize;
    let px = [lit[i], lit[i + 1], lit[i + 2]];
    assert!(
        px[0] > 200 && px[1] > 200 && px[2] > 200,
        "the content half vanished after the lit seam: pixel at (100, 120) is {px:?}"
    );
}
