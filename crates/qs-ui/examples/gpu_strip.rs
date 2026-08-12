//! Render a mock of the application through the **real** shader, on a real adapter, and
//! write two PNGs to look at.
//!
//! `cargo run -p qs-ui --example gpu_strip -- out.png`
//!
//! # Why this exists when `material_strip` already does
//!
//! `material_strip` rasterizes on the CPU tier, and [`qs_gpu::frame::Instance::cpu_floor`]
//! drops every [`qs_gpu::frame::Fidelity::Enhanced`] primitive before `tiny-skia` is
//! reached. That is correct — it is what the fallback tier genuinely looks like — and it
//! means the harness structurally cannot show a glow or a rim. Its own doc comment says so
//! about the halo. Every enhanced primitive added since has inherited the same blind spot,
//! so the effects the design leans hardest on were the only ones nobody could look at.
//!
//! This renders through `shaders/instance.wgsl` itself: a headless `wgpu` device, an
//! offscreen `Bgra8UnormSrgb` texture, and a read-back. Not a transcription of the shader
//! and not an approximation of it — the same code the window runs.
//!
//! # What it draws
//!
//! A window, not a swatch board. Rows at every state the list can be in, under the command
//! bar and above the status shelf, because the question a material has to answer is whether
//! it reads *in place* next to the other things it will always be next to. A palette strip
//! answers a different and easier question, which is why the previous visual chunks kept
//! shipping looks that measured correctly and sat wrong.

// A looking harness, not shipped code: a panic here is a developer seeing a stack trace
// instead of a picture. Same set the other examples allow.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use qs_gpu::batcher::Renderer;
use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::frame::DrawList;
use qs_gpu::path::RenderPath;
use qs_ui::material::{Drive, Surface, name};
use qs_ui::tokens::{Theme, Tokens};

const WIDTH: u32 = 900;
const HEIGHT: u32 = 700;
const SCALE: f32 = 2.0;

/// The mock window's furniture, in logical pixels.
const BAR_H: f32 = 44.0;
const SHELF_H: f32 = 26.0;
const ROW_H: f32 = 32.0;
const PAD: f32 = 12.0;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gpu-strip.png".to_string());

    let instance = new_instance(RenderPath::Primary);
    let ctx = GpuContext::new(RenderPath::Primary, instance, None)
        .expect("no usable adapter; this harness needs a GPU because that is the point of it");
    eprintln!(
        "gpu_strip: {} via {:?}, {}",
        ctx.capabilities.adapter_name, ctx.capabilities.backend, ctx.capabilities.driver
    );

    for (theme, label) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
        let tokens = Tokens::embedded(theme).expect("tokens");
        let mut list = DrawList::default();
        list.reset([WIDTH, HEIGHT], tokens.color("surface/base"), 1);
        // The same sky `App::render` sets, or a lit surface here reflects a room the
        // application does not have.
        list.set_environment(qs_gpu::frame::Environment {
            horizon: tokens.color("surface/base"),
            zenith: tokens.color("surface/row-hover"),
        });
        build(&tokens, &mut list);
        list.end_batch(None, false);

        let pixels = render(&ctx, &list);
        let path = out.replace(".png", &format!("-{label}.png"));
        write_png(&path, &pixels);
        eprintln!(
            "gpu_strip: wrote {path} ({} instances)",
            list.instances.len()
        );
    }
}

/// One mock window's worth of materials.
///
/// Painted in the order the application paints them, because order is a material decision
/// that `Material::composites` gates and a harness that reordered it would be checking a
/// stack nobody ships.
fn build(tokens: &Tokens, list: &mut DrawList) {
    // Where in the material cycle to draw this frame. The application advances the phase only
    // while some other animation holds the loop awake, so a still picture of it is always a
    // picture of one instant -- and the only way to judge a drifting light is to look at more
    // than one. `QS_PHASE=0.5 cargo run --example gpu_strip` is that.
    let phase: f32 = std::env::var("QS_PHASE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let ambient = Drive::new(0.0, phase);
    let px = |logical: f32| logical * SCALE;
    let radius = tokens.radius("row") * SCALE;

    // The ground, first, exactly as `App::render` paints it. A harness that skipped it would
    // be judging every row against a flat field the application no longer has.
    tokens.paint_driven(
        name::SURFACE_CANVAS,
        Surface::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, SCALE),
        1.0,
        ambient,
        &mut list.instances,
    );

    // The command bar, across the top.
    tokens.paint_driven(
        name::CHROME_BAR,
        Surface::new(0.0, 0.0, WIDTH as f32, px(BAR_H), 0.0, SCALE),
        1.0,
        ambient,
        &mut list.instances,
    );

    // A hovered chip beside the space a resting one occupies, so the hover state is judged
    // against its own absence rather than against nothing.
    let chip_w = px(78.0);
    let chip_h = px(26.0);
    let chip_y = (px(BAR_H) - chip_h) * 0.5;
    tokens.paint(
        name::CHROME_CHIP_HOVER,
        Surface::new(
            px(PAD) + chip_w + px(8.0),
            chip_y,
            chip_w,
            chip_h,
            radius,
            SCALE,
        ),
        1.0,
        &mut list.instances,
    );

    let list_top = px(BAR_H) + px(PAD);
    let row_w = WIDTH as f32 - px(PAD) * 2.0;
    let row_x = px(PAD);
    let row_of = |i: usize| list_top + i as f32 * px(ROW_H);
    let surface_of = |i: usize| Surface::new(row_x, row_of(i), row_w, px(ROW_H), radius, SCALE);

    // Alternating banding, which is a token rather than a material and is drawn by the row
    // renderer. Included because "near-invisible" is a claim about how it sits next to the
    // states below it, not about the swatch.
    for i in [1usize, 3, 5, 7] {
        let alt = tokens.color("surface/row-alt");
        let s = surface_of(i);
        list.instances
            .push(qs_gpu::frame::Instance::rect(s.x, s.y, s.w, s.h, 0.0, alt));
    }

    // Every state the list can be in, in one column, at rest and at full drive. The selection
    // is the only one that swells, and judging it only at rest hides the loudest frame it has
    // -- which is the frame a viewer's eye is actually drawn to.
    let selected_at = |i: usize, drive: Drive, list: &mut DrawList| {
        let Some(material) = tokens.material(name::ROW_SELECTED) else {
            return;
        };
        for pass in [qs_ui::material::Pass::Bleed, qs_ui::material::Pass::Body] {
            material.compile_pass(
                pass,
                surface_of(i),
                1.0,
                drive,
                tokens.effects_enabled(),
                &mut list.instances,
            );
        }
    };

    tokens.paint(name::ROW_HOVER, surface_of(2), 1.0, &mut list.instances);
    selected_at(4, Drive::REST, list);
    selected_at(6, Drive::new(1.0, 0.0), list);

    // The status shelf, along the bottom.
    tokens.paint_driven(
        name::CHROME_SHELF,
        Surface::new(
            0.0,
            HEIGHT as f32 - px(SHELF_H),
            WIDTH as f32,
            px(SHELF_H),
            0.0,
            SCALE,
        ),
        1.0,
        ambient,
        &mut list.instances,
    );
}

/// Submit one draw list to an offscreen target and read the texture back.
fn render(ctx: &GpuContext, list: &DrawList) -> Vec<u8> {
    let format = ctx.capabilities.surface_format;
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gpu_strip target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = Renderer::new(ctx, 512);
    let commands = renderer.render(ctx, &view, list, None);

    // `copy_texture_to_buffer` requires the row stride to be a multiple of 256, so the
    // read-back buffer is padded and unpadded again below. Getting this wrong shifts every
    // row by a few pixels and produces a sheared picture that looks like a shader bug.
    let unpadded = WIDTH * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_strip readback"),
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
        let row_bytes = &mapped[start..start + unpadded as usize];
        // The target is `Bgra8UnormSrgb`; a PNG wants RGBA. Swapping here rather than in the
        // shader keeps the harness the only thing that knows about the surface's byte order.
        for texel in row_bytes.chunks_exact(4) {
            pixels.extend_from_slice(&[texel[2], texel[1], texel[0], texel[3]]);
        }
    }
    drop(mapped);
    buffer.unmap();
    pixels
}

fn write_png(path: &str, rgba: &[u8]) {
    let mut pixmap = tiny_skia::Pixmap::new(WIDTH, HEIGHT).expect("pixmap");
    for (dst, src) in pixmap.pixels_mut().iter_mut().zip(rgba.chunks_exact(4)) {
        // The framebuffer holds premultiplied sRGB, which is exactly what a `PremultipliedColorU8`
        // is. Round-tripping through straight colour here would double-apply the alpha.
        *dst = tiny_skia::PremultipliedColorU8::from_rgba(src[0], src[1], src[2], src[3])
            .expect("the framebuffer produced a colour brighter than its own alpha");
    }
    pixmap.save_png(path).expect("save");
}
