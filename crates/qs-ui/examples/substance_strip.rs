//! Save row-material comparisons using the GPU renderer.
//!
//! `cargo run -p qs-ui --example substance_strip -- out.png`
//!
//! Each strip varies one row property while holding the others fixed, with
//! a control group for comparison in both themes.

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
use qs_gpu::frame::{DrawList, Environment};
use qs_gpu::path::RenderPath;
use qs_ui::material::{Drive, Surface, name};
use qs_ui::row_source::{KindId, LoadState, RowFlags, RowId, RowView};
use qs_ui::substance::Substance;
use qs_ui::tokens::{Theme, Tokens};

const WIDTH: u32 = 820;
const HEIGHT: u32 = 1000;
const SCALE: f32 = 2.0;

const ROW_H: f32 = 26.0;
const PAD: f32 = 14.0;
const GAP: f32 = 22.0;

const NOW: i64 = 1_786_060_800;
const DAY: i64 = 86_400;

fn row(size: u64, mtime: i64, flags: RowFlags) -> RowView {
    RowView {
        id: RowId(0),
        name: 0..0,
        size,
        mtime,
        kind: KindId(0),
        flags,
        state: LoadState::Basic,
        depth: 0,
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "substance-strip.png".to_string());

    let instance = new_instance(RenderPath::Primary);
    let ctx = GpuContext::new(RenderPath::Primary, instance, None)
        .expect("no usable adapter; this harness needs a GPU because that is the point of it");
    eprintln!(
        "substance_strip: {} via {:?}",
        ctx.capabilities.adapter_name, ctx.capabilities.backend
    );

    for (theme, label) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
        let tokens = Tokens::embedded(theme).expect("tokens");
        let mut list = DrawList::default();
        list.reset([WIDTH, HEIGHT], tokens.color("surface/base"), 1);
        // The same sky `App::render` sets. Without it a lit surface reflects a room the
        // application does not have, and the environment encoding would be judged against
        // a light nobody ships.
        list.set_environment(Environment {
            horizon: tokens.color("surface/base"),
            zenith: tokens.color("surface/row-hover"),
        });
        build(&tokens, &mut list);
        list.end_batch(None, false);

        let pixels = render(&ctx, &list);
        let path = out.replace(".png", &format!("-{label}.png"));
        write_png(&path, &pixels);
        eprintln!("substance_strip: wrote {path}");
    }
}

fn build(tokens: &Tokens, list: &mut DrawList) {
    let px = |logical: f32| logical * SCALE;
    let response = tokens.substance();
    let mut y = px(PAD);

    // Held at their mid-values while the other fact sweeps. A picture with three things
    // varying at once cannot say which of them is doing the work.
    let mid_age = NOW - 90 * DAY;

    let sweeps: [(&str, Vec<RowView>); 1] = [(
        // Size, into the bevel. Powers of a thousand, because that is how file sizes are
        // spaced and a linear sweep would show five identical rows and one different one.
        "size",
        [
            0u64,
            4_096,
            262_144,
            16_777_216,
            1_073_741_824,
            68_719_476_736,
        ]
        .iter()
        .map(|&s| row(s, mid_age, RowFlags::EMPTY))
        .collect(),
    )];

    for (_fact, rows) in &sweeps {
        for (i, r) in rows.iter().enumerate() {
            // Alternating body materials, exactly as the list paints them, so the encoding is
            // judged over both grounds rather than over a convenient one.
            let material = if i % 2 == 1 {
                name::ROW_BODY_ALT
            } else {
                name::ROW_BODY
            };
            tokens.paint_substance(
                material,
                Surface::new(
                    px(PAD),
                    y,
                    WIDTH as f32 - px(PAD) * 2.0,
                    px(ROW_H),
                    0.0,
                    SCALE,
                ),
                1.0,
                Drive::REST,
                Some(Substance::of(r, response)),
                &mut list.instances,
            );
            y += px(ROW_H);
        }
        y += px(GAP);
    }

    // And the control: the same rows with no substance at all, which is what the list looked
    // like before this chunk. If a sweep above is indistinguishable from this block, the
    // encoding is not carrying anything.
    for i in 0..3 {
        let material = if i % 2 == 1 {
            name::ROW_BODY_ALT
        } else {
            name::ROW_BODY
        };
        tokens.paint_substance(
            material,
            Surface::new(
                px(PAD),
                y,
                WIDTH as f32 - px(PAD) * 2.0,
                px(ROW_H),
                0.0,
                SCALE,
            ),
            1.0,
            Drive::REST,
            None,
            &mut list.instances,
        );
        y += px(ROW_H);
    }
}

fn render(ctx: &GpuContext, list: &DrawList) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("substance_strip target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = Renderer::new(ctx, 512);
    let commands = renderer.render(ctx, &view, list, None, None);

    let unpadded = WIDTH * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("substance_strip readback"),
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
