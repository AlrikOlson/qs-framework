//! A glass panel over a list, through the real pipeline on a real adapter, as a PNG.
//!
//! `cargo run --release -p qs-gpu --example refract_panel -- out.png`
//!
//! # Why this exists rather than a test
//!
//! `blur_panel`'s reason, one primitive later: every gate in this repository asks whether a
//! claim is *true*, and none asks whether an effect is *visible*. Research R15 is the recorded
//! case — a correct, bounded, four-ways-tested effect that moved 446,580 pixels by a peak of
//! 7/255 and was worth nothing.
//!
//! But this example also does something `blur_panel` does not, because refraction has a claim
//! blur does not have. `PrimKind::Refract` transmits **only within the bevel**, and the middle
//! of the panel is asserted to be bit-identical to the PBR surface — that is the contrast
//! argument that lets a glass panel carry a label at all. So the two halves are measured
//! twice, in two regions, and the two numbers point in opposite directions:
//!
//! - the **interior** difference must be exactly `0.00`, or the effect is under a label;
//! - the **bevel band** difference must be large, or the effect is invisible.
//!
//! A build where both are near zero is R15 again. A build where the interior is non-zero is a
//! moving ground under a green `cargo xtask contrast`, which is the failure
//! `qs_ui::substance` refused three separate encodings to avoid.
//!
//! # What is in the frame
//!
//! The two halves are the same panel at refraction strength 0 and 1. Strength 0 is the PBR
//! surface exactly — `Instance::refract` documents that as byte-identical — so the difference
//! between the halves is the glass and nothing else: not a different primitive, not a different
//! shading path, not a different set of surface parameters.
//!
//! The content behind is high-frequency and runs **through** where the bevel falls, because a
//! displacement is only visible to the extent the thing displaced varies. A panel floated over
//! a flat ground refracts perfectly and shows nothing.

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

/// One half. The full image is two of these side by side.
const HALF: u32 = 420;
const HEIGHT: u32 = 360;
const WIDTH: u32 = HALF * 2;

/// The panel's geometry. The bevel is generous on purpose: this is an instrument, and a 4 px
/// lip would leave the reader deciding whether they can see something rather than what they
/// can see. `chrome/popover` authors a smaller one; `refract_cost` measures both.
const RADIUS: f32 = 24.0;
const BEVEL: f32 = 18.0;

/// The panel's albedo. `chrome/popover`'s dark-theme surface, repeated here rather than read
/// because this example belongs to `qs-gpu` and `design/tokens.json` is `qs-ui`'s — the
/// dependency points the other way.
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
    // A saturated ramp across the whole half. Colour rather than grey, because dispersion is
    // half of what this effect does and a monochrome backdrop cannot show a prism doing
    // anything at all — the three channels would land on the same value however far apart the
    // three rays are.
    list.instances.push(Instance::gradient(
        x0,
        0.0,
        HALF as f32,
        HEIGHT as f32,
        0.0,
        std::f32::consts::FRAC_PI_4,
        Srgba::new(0.16, 0.38, 0.92, 1.0),
        Srgba::new(0.95, 0.55, 0.12, 1.0),
    ));
    // Rows at a realistic pitch with a bright bar of "text" on each, running edge to edge so
    // that they cross the panel's bevel rather than stopping short of it. What a refraction
    // does is *bend* a straight line, and a straight line has to be there to be bent.
    let row_h = 20.0;
    for i in 0..(HEIGHT as f32 / row_h) as u32 {
        let y = i as f32 * row_h;
        list.instances.push(Instance::rect(
            x0,
            y + 7.0,
            HALF as f32,
            6.0,
            0.0,
            if i % 2 == 0 {
                Srgba::new(0.92, 0.94, 0.98, 1.0)
            } else {
                Srgba::new(0.04, 0.04, 0.06, 1.0)
            },
        ));
    }
    // And verticals, so the horizontal displacement at the left and right bevels has something
    // to bend too. Without these the effect is only legible along two of the four edges, which
    // reads as an asymmetry in the normal rather than as the fixture's own fault.
    for c in 0..9 {
        list.instances.push(Instance::rect(
            x0 + 8.0 + c as f32 * 46.0,
            0.0,
            5.0,
            HEIGHT as f32,
            0.0,
            Srgba::new(0.02, 0.02, 0.03, 0.85),
        ));
    }
}

/// Where the panel sits inside a half.
fn panel_rect(x0: f32) -> (f32, f32, f32, f32) {
    (x0 + 60.0, 70.0, HALF as f32 - 120.0, HEIGHT as f32 - 160.0)
}

fn scene() -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.07, 0.07, 0.09, 1.0), 1);

    // Both backdrops first, and identical. They must precede the panels and be in the same run,
    // or the right half's panel would refract a different scene from the left's and the
    // comparison would be measuring the fixture rather than the effect.
    backdrop(&mut list, 0.0);
    backdrop(&mut list, HALF as f32);

    // Left: strength 0. NOT a `PrimKind::Pbr` instance that happens to have the same
    // parameters — the same kind, at zero strength, so anything the refracting branch does to
    // the surface *other* than refract it shows up as a difference here and is attributed to
    // the effect. Using Pbr on the left would silently forgive exactly that.
    let (x, y, w, h) = panel_rect(0.0);
    list.instances.push(Instance::refract(
        x, y, w, h, RADIUS, BEVEL, 0.32, 0.0, 0.55, 0.0, PANEL,
    ));

    // Right: the glass.
    let (x, y, w, h) = panel_rect(HALF as f32);
    list.instances.push(Instance::refract(
        x, y, w, h, RADIUS, BEVEL, 0.32, 0.0, 0.55, 1.0, PANEL,
    ));

    list.end_batch(None, false);
    list
}

/// The signed distance to the panel's rounded boundary, negative inside.
///
/// A transcription of `sd_rounded_box` from `shaders/instance.wgsl`, for the reason
/// `tier_parity` transcribes rather than approximates: the regions measured below are defined
/// by the same function the shader uses to decide where the bevel is, so "inside the bevel"
/// here means what it means there.
fn sd_rounded_box(px: f32, py: f32, rect: (f32, f32, f32, f32), radius: f32) -> f32 {
    let (x, y, w, h) = rect;
    let (hx, hy) = (w * 0.5, h * 0.5);
    let r = radius.min(hx.min(hy));
    let (lx, ly) = (px - (x + hx), py - (y + hy));
    let qx = lx.abs() - hx + r;
    let qy = ly.abs() - hy + r;
    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
}

/// Mean absolute channel difference between the two panels, over one region.
///
/// `region` decides which fragments count, from the signed distance to the boundary. Both
/// panels sit at the same offset inside their halves and over identical content, so anything
/// here is the effect and nothing else.
fn difference(rgba: &[u8], region: impl Fn(f32) -> bool) -> (f64, u64) {
    let rect = panel_rect(0.0);
    let (x, y, w, h) = rect;
    let mut total = 0.0_f64;
    let mut count = 0_u64;
    for py in (y as u32)..(y + h) as u32 {
        for px in (x as u32)..(x + w) as u32 {
            // The pixel centre, which is what the fragment shader is handed.
            let d = sd_rounded_box(px as f32 + 0.5, py as f32 + 0.5, rect, RADIUS);
            if !region(d) {
                continue;
            }
            let a = ((py * WIDTH + px) * 4) as usize;
            let b = ((py * WIDTH + px + HALF) * 4) as usize;
            for c in 0..3 {
                total += f64::from(rgba[a + c].abs_diff(rgba[b + c]));
            }
            count += 1;
        }
    }
    (total / (count * 3).max(1) as f64, count)
}

/// The largest single-channel difference anywhere in the band, and where it is.
fn peak(rgba: &[u8]) -> (u8, u32, u32) {
    let rect = panel_rect(0.0);
    let (x, y, w, h) = rect;
    let mut best = (0_u8, 0_u32, 0_u32);
    for py in (y as u32)..(y + h) as u32 {
        for px in (x as u32)..(x + w) as u32 {
            if sd_rounded_box(px as f32 + 0.5, py as f32 + 0.5, rect, RADIUS) > 0.0 {
                continue;
            }
            let a = ((py * WIDTH + px) * 4) as usize;
            let b = ((py * WIDTH + px + HALF) * 4) as usize;
            for c in 0..3 {
                let d = rgba[a + c].abs_diff(rgba[b + c]);
                if d > best.0 {
                    best = (d, px, py);
                }
            }
        }
    }
    best
}

/// A nearest-neighbour magnification of one corner of each panel, side by side.
///
/// Written because looking at an 18 px band in an 840 px image is not looking at it. Nearest
/// rather than filtered on purpose: the question this answers is whether the refracted content
/// is *sharp* — whether the glass magnifies or merely smears — and a smooth upscale would
/// answer it by softening both sides equally.
fn magnified(rgba: &[u8], scale: u32) -> (Vec<u8>, u32, u32) {
    let (x, y, _, _) = panel_rect(0.0);
    // The corner plus enough of the flank to show the bevel turning into flat interior, which
    // is the transition the whole effect lives in.
    let (cw, ch) = (150_u32, 110_u32);
    let (ox, oy) = (x as u32 - 8, y as u32 - 8);
    let (w, h) = (cw * 2 * scale, ch * scale);
    let mut out = vec![0_u8; (w * h * 4) as usize];
    for py in 0..h {
        for px in 0..w {
            // Which half, and where inside it.
            let half = u32::from(px >= cw * scale);
            let sx = ox + half * HALF + (px - half * cw * scale) / scale;
            let sy = oy + py / scale;
            let src = ((sy * WIDTH + sx) * 4) as usize;
            let dst = ((py * w + px) * 4) as usize;
            out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
        }
    }
    (out, w, h)
}

fn save(path: &str, rgba: &[u8], w: u32, h: u32) {
    let mut pixmap = tiny_skia::Pixmap::new(w, h).expect("pixmap");
    for (dst, src) in pixmap.pixels_mut().iter_mut().zip(rgba.chunks_exact(4)) {
        *dst = tiny_skia::PremultipliedColorU8::from_rgba(src[0], src[1], src[2], src[3])
            .expect("the framebuffer produced a colour brighter than its own alpha");
    }
    pixmap.save_png(path).expect("save");
}

fn render(ctx: &GpuContext) -> Vec<u8> {
    let format = ctx.capabilities.surface_format;
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("refract_panel target"),
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
        label: Some("refract_panel readback"),
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
    pixels
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "refract_panel.png".to_string());

    let tier = RenderPath::Primary;
    let instance = new_instance(tier);
    let ctx = GpuContext::new(tier, instance, None).expect("no usable adapter");
    println!(
        "{tier:?} — {} via {:?}",
        ctx.capabilities.adapter_name, ctx.capabilities.backend
    );

    let rgba = render(&ctx);

    save(&out, &rgba, WIDTH, HEIGHT);
    let zoom_path = out.replace(".png", "-zoom.png");
    let (zoom, zw, zh) = magnified(&rgba, 4);
    save(&zoom_path, &zoom, zw, zh);

    println!("left: strength 0 (the PBR surface).  right: strength 1 (glass).");
    println!("bevel {BEVEL} px, radius {RADIUS} px");

    // The band, and the interior, and they are the two halves of the same claim.
    let (band, band_px) = difference(&rgba, |d| d < 0.0 && d > -BEVEL);
    let (inside, inside_px) = difference(&rgba, |d| d <= -BEVEL);
    let (peak_v, peak_x, peak_y) = peak(&rgba);

    println!("bevel band : mean |difference| {band:>6.2}/255 over {band_px} px");
    println!("interior   : mean |difference| {inside:>6.2}/255 over {inside_px} px");
    println!("peak       : {peak_v}/255 at ({peak_x}, {peak_y})");

    if inside != 0.0 {
        println!(
            "  !! THE INTERIOR MOVED. Refraction is supposed to be identically zero deeper \
             than the bevel — that is what lets a label sit on this panel while the contrast \
             gate reads only the albedo. A non-zero number here is a moving ground under a \
             green gate, which is qs_ui::substance's refused encoding arriving by another door."
        );
    }
    if band < 1.0 {
        println!(
            "  !! THE BAND DID NOT MOVE. A correct, bounded, fully tested effect that changes \
             no pixel is research R15's finding, and this is what it looks like."
        );
    }
    println!("wrote {out} and {zoom_path} (4x, nearest)");
}
