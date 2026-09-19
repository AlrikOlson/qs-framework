//! Save the ambient field at four phases in both themes.
//!
//! `cargo run --release -p qs-ui --example field_strip -- out.png`
//!
//! The field fills each tile without foreground content, making movement and
//! falloff visible. Use `gpu_strip` to inspect it behind rows.

// A looking harness, not shipped code: a panic here is a developer seeing a stack trace
// instead of a picture. Same set the other examples allow.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
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

const TILE: u32 = 440;
const GAP: u32 = 8;
const COLS: u32 = 2;
const ROWS: u32 = 2;
const WIDTH: u32 = TILE * COLS + GAP * (COLS + 1);
const HEIGHT: u32 = TILE * ROWS + GAP * (ROWS + 1);

fn scene(tokens: &Tokens) -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], tokens.color("surface/base"), 1);
    // The centres, from the material that declares them -- the same route `qs::main` takes,
    // so this harness cannot show a field the application would not draw.
    if let Some(field) = tokens.field(name::SURFACE_CANVAS) {
        list.set_field(field);
    }

    for i in 0..(COLS * ROWS) {
        let col = i % COLS;
        let row = i / COLS;
        let x = (GAP + col * (TILE + GAP)) as f32;
        let y = (GAP + row * (TILE + GAP)) as f32;
        // A quarter turn between tiles, so the four together are one full cycle.
        let phase = i as f32 / (COLS * ROWS) as f32;
        tokens.paint_driven(
            name::SURFACE_CANVAS,
            Surface::new(x, y, TILE as f32, TILE as f32, 0.0, 1.0),
            1.0,
            Drive::new(0.0, phase),
            &mut list.instances,
        );
    }
    list.end_batch(None, false);
    list
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "field_strip.png".to_string());

    let instance = new_instance(RenderPath::Primary);
    let ctx = match GpuContext::new(RenderPath::Primary, instance, None) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("field_strip: no usable adapter ({e:?})");
            return;
        }
    };
    println!(
        "field_strip: {} via {:?}",
        ctx.capabilities.adapter_name, ctx.capabilities.backend
    );

    for (theme, suffix) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
        let tokens = Tokens::embedded(theme).unwrap();
        let list = scene(&tokens);
        let pixels = render(&ctx, &list);
        let path = out.replace(".png", &format!("-{suffix}.png"));
        write_png(&path, &pixels);
        println!(
            "field_strip: wrote {path} ({} instances)",
            list.instances.len()
        );
    }
}

/// Submit one draw list to an offscreen target and read the texture back.
///
/// The same shape as `gpu_strip`'s, including the 256-byte row-stride padding: getting that
/// wrong shears every row by a few pixels and looks exactly like a shader bug.
fn render(ctx: &GpuContext, list: &DrawList) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("field_strip target"),
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

    let mut renderer = Renderer::new(ctx, 512);
    let commands = renderer.render(ctx, &view, list, None, None);

    let unpadded = WIDTH * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("field_strip readback"),
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
        // The target is `Bgra8UnormSrgb`; a PNG wants RGBA.
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
        *dst = tiny_skia::PremultipliedColorU8::from_rgba(src[0], src[1], src[2], src[3])
            .expect("the framebuffer produced a colour brighter than its own alpha");
    }
    pixmap.save_png(path).expect("save");
}
