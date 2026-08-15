//! A popover over a list, through the real pipeline on a real adapter, as a PNG.
//!
//! `cargo run --release -p qs-gpu --example blur_panel -- out.png`
//!
//! # Why this exists rather than a test
//!
//! Everything about the blur that a test can hold, a test holds: the floor is asserted in
//! `tier_parity`, the kernel and the memory in `qs_gpu::target`, the contrast in `qs_ui`. What
//! none of them can see is whether the effect is *visible* — research R15 is this repository's
//! recorded case of a correct, bounded, four-ways-tested effect that moved 446,580 pixels by a
//! peak of 7/255 and was worth nothing. Every gate here asks whether a claim is true; none asks
//! whether an effect can be seen.
//!
//! So this draws the two frames side by side — the same panel with the blur and with the floor
//! it degrades to — and leaves the answer to a person looking at it. It also prints the mean
//! absolute difference between the halves, which is the cheap version of the R15 question: a
//! number near zero means the chain ran and changed nothing, which is the failure that looks
//! exactly like success.
//!
//! # What is in the frame
//!
//! Deliberately high-frequency content behind the panel — thin bright bars at row pitch, a
//! saturated ramp, hard edges — because a blur over a flat ground is indistinguishable from a
//! tint. If the bars are still countable through the glass, the kernel is too narrow; if the
//! panel is a flat slab, the glass is too opaque.

// Same set the other examples allow: a harness that panics on a malformed readback is
// reporting a defect rather than hiding one.
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
use qs_gpu::frame::{DrawList, Instance};
use qs_gpu::path::RenderPath;
use qs_gpu::target::{BlurChain, blur_reach_pixels};

/// One half. The full image is two of these side by side.
const HALF: u32 = 420;
const HEIGHT: u32 = 360;
const WIDTH: u32 = HALF * 2;

/// The glass and the panel, as `design/tokens.json` authors them for `chrome/popover` in the
/// dark theme. Repeated here rather than read, because this example belongs to `qs-gpu` and
/// the token file is `qs-ui`'s — the dependency points the other way.
const GLASS: Srgba = Srgba {
    r: 0.196,
    g: 0.196,
    b: 0.212,
    a: 0.90,
};
const PANEL: Srgba = Srgba {
    r: 0.196,
    g: 0.196,
    b: 0.212,
    a: 1.0,
};

/// The content behind the panel, drawn into `x0..x0 + HALF`.
fn backdrop(list: &mut DrawList, x0: f32) {
    list.instances.push(Instance::rect(
        x0,
        0.0,
        HALF as f32,
        HEIGHT as f32,
        0.0,
        Srgba::new(0.07, 0.07, 0.09, 1.0),
    ));
    // Rows at a realistic pitch, alternating, with a bright bar of "text" on each. Thin and
    // high-contrast on purpose: this is the signal the blur has to destroy.
    let row_h = 24.0;
    for i in 0..(HEIGHT as f32 / row_h) as u32 {
        let y = i as f32 * row_h;
        if i % 2 == 1 {
            list.instances.push(Instance::rect(
                x0,
                y,
                HALF as f32,
                row_h,
                0.0,
                Srgba::new(0.11, 0.11, 0.14, 1.0),
            ));
        }
        for c in 0..7 {
            list.instances.push(Instance::rect(
                x0 + 16.0 + c as f32 * 56.0,
                y + 8.0,
                40.0,
                8.0,
                2.0,
                Srgba::new(0.86, 0.88, 0.94, 1.0),
            ));
        }
    }
    // A saturated ramp under the LEFT half of where the panel goes, and nothing under the
    // right half. That placement is the measurement: the transmitted backdrop is
    // `(1 - glass alpha)` of whatever is behind, so an effect is visible exactly to the extent
    // the backdrop VARIES beneath it. A panel floated over uniform content shows nothing at
    // any opacity, and a demo built that way concludes the blur is broken.
    list.instances.push(Instance::gradient(
        x0,
        0.0,
        HALF as f32 * 0.55,
        HEIGHT as f32,
        0.0,
        std::f32::consts::FRAC_PI_2,
        Srgba::new(0.20, 0.45, 0.95, 1.0),
        Srgba::new(0.95, 0.60, 0.15, 1.0),
    ));
    // And the bars again on top of it, so the ramp is not a smooth field the blur cannot be
    // seen working on: what a blur destroys is high frequency, and only a backdrop that HAS
    // some can show that it did.
    let row_h = 24.0;
    for i in 0..(HEIGHT as f32 / row_h) as u32 {
        let y = i as f32 * row_h;
        for c in 0..4 {
            list.instances.push(Instance::rect(
                x0 + 16.0 + c as f32 * 56.0,
                y + 8.0,
                40.0,
                8.0,
                2.0,
                Srgba::new(0.06, 0.06, 0.08, 1.0),
            ));
        }
    }
}

/// Where the panel sits inside a half.
fn panel_rect(x0: f32) -> (f32, f32, f32, f32) {
    (x0 + 60.0, 70.0, HALF as f32 - 120.0, HEIGHT as f32 - 160.0)
}

fn scene() -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.07, 0.07, 0.09, 1.0), 1);

    // Both backdrops first, and identical. They have to be in the same batch run as each
    // other and BEFORE the blur, or the right half's panel would have a different backdrop
    // from the left's and the comparison would be measuring the scene rather than the effect.
    backdrop(&mut list, 0.0);
    backdrop(&mut list, HALF as f32);

    // Left: the floor. What UXDD 10.7 says a machine without the effect shows, drawn as the
    // plain rect it degrades to rather than as a blur instance with the chain disabled -- the
    // point is to picture what actually ships, not to picture the same code path twice.
    let (x, y, w, h) = panel_rect(0.0);
    list.instances.push(Instance::rect(x, y, w, h, 14.0, PANEL));

    // Right: the effect.
    let (x, y, w, h) = panel_rect(HALF as f32);
    list.instances
        .push(Instance::blur(x, y, w, h, 14.0, GLASS, PANEL));

    // A hairline on each, so the two panels are separated from their grounds the same way and
    // the difference between them is only the fill.
    for x0 in [0.0, HALF as f32] {
        let (x, y, w, h) = panel_rect(x0);
        list.instances.push(Instance::stroke(
            x,
            y,
            w,
            h,
            14.0,
            1.0,
            Srgba::new(0.35, 0.35, 0.40, 1.0),
        ));
    }

    list.end_batch(None, false);
    list
}

fn render(ctx: &GpuContext) -> (Vec<u8>, Option<u64>) {
    let format = ctx.capabilities.surface_format;
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blur_panel target"),
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

    let list = scene();
    let mut renderer = Renderer::new(ctx, 512);
    let commands = renderer.render(ctx, &view, &list, None, None);

    let unpadded = WIDTH * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blur_panel readback"),
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
    let bgra =
        format == wgpu::TextureFormat::Bgra8UnormSrgb || format == wgpu::TextureFormat::Bgra8Unorm;
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for row in 0..HEIGHT {
        let start = (row * padded) as usize;
        let row_bytes = &mapped[start..start + unpadded as usize];
        for texel in row_bytes.chunks_exact(4) {
            if bgra {
                pixels.extend_from_slice(&[texel[2], texel[1], texel[0], texel[3]]);
            } else {
                pixels.extend_from_slice(texel);
            }
        }
    }
    drop(mapped);
    buffer.unmap();

    let chain = renderer.blur().map(BlurChain::bytes);
    (pixels, chain)
}

/// Mean absolute channel difference between the two panels, over the pixels inside them.
///
/// The R15 question in one number. Both panels sit at the same offset inside their halves and
/// over identical content, so a difference here is the effect and nothing else.
fn panel_difference(rgba: &[u8]) -> f64 {
    let (lx, ly, w, h) = panel_rect(0.0);
    let mut total = 0.0_f64;
    let mut count = 0_u64;
    // Inset by the radius so the comparison is over the panel's interior rather than over its
    // rounded corners, where one side is antialiased against a different neighbour.
    for y in (ly as u32 + 16)..(ly + h) as u32 - 16 {
        for x in (lx as u32 + 16)..(lx + w) as u32 - 16 {
            let a = ((y * WIDTH + x) * 4) as usize;
            let b = ((y * WIDTH + x + HALF) * 4) as usize;
            for c in 0..3 {
                total += f64::from(rgba[a + c].abs_diff(rgba[b + c]));
                count += 1;
            }
        }
    }
    total / count as f64
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "blur_panel.png".to_string());

    let tier = RenderPath::Primary;
    let instance = new_instance(tier);
    let ctx = GpuContext::new(tier, instance, None).expect("no usable adapter");
    println!(
        "{tier:?} — {} via {:?}",
        ctx.capabilities.adapter_name, ctx.capabilities.backend
    );

    let (rgba, chain) = render(&ctx);

    let mut pixmap = tiny_skia::Pixmap::new(WIDTH, HEIGHT).expect("pixmap");
    for (dst, src) in pixmap.pixels_mut().iter_mut().zip(rgba.chunks_exact(4)) {
        *dst = tiny_skia::PremultipliedColorU8::from_rgba(src[0], src[1], src[2], src[3])
            .expect("the framebuffer produced a colour brighter than its own alpha");
    }
    pixmap.save_png(&out).expect("save");

    println!("left: the floor (opaque surface/raised).  right: the effect.");
    println!(
        "kernel: sigma {} downsampled texels, reach {} physical px",
        qs_gpu::target::BLUR_SIGMA_TEXELS,
        blur_reach_pixels()
    );
    match chain {
        Some(bytes) => println!("chain: allocated, {} bytes at {WIDTH}x{HEIGHT}", bytes),
        None => println!(
            "chain: NOT ALLOCATED — the blur instance did not reach the renderer, so the \
             right half is whatever the placeholder holds"
        ),
    }
    let difference = panel_difference(&rgba);
    println!("mean |difference| between the panels: {difference:.2}/255");
    if difference < 1.0 {
        println!(
            "  ...which is nothing. A correct, bounded, fully tested effect that changes no \
             pixel is research R15's finding, and this is what it looks like."
        );
    }
    println!("wrote {out}");
}
