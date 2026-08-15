//! The two claims `PrimKind::Refract` makes that only a running GPU can settle.
//!
//! # Why these are here and not in `tier_parity`
//!
//! `tier_parity` compares the CPU rasterizer against a Rust transcription of `instance.wgsl`.
//! Both halves are Rust and the WGSL is never executed, so what it can hold about this
//! primitive is exactly its floor: a refracting panel degrades to its albedo. That is worth
//! holding and it is held there. It says nothing about either claim below, because both are
//! properties of the shader's arithmetic on a real adapter.
//!
//! ## The contrast claim
//!
//! `refraction_never_reaches_the_middle_of_a_surface` is the load-bearing one. `qs_ui::substance`
//! records the rule three refused encodings established: only **edge-localised** material
//! properties are free against the contrast budget, because `cargo xtask contrast` checks the
//! authored albedo and cannot see what a shader does to the middle of a surface. Refraction is
//! allowed to exist at all because it is identically zero deeper than `bevel` inward — so the
//! ground under a label on a glass panel is the ground the gate checked.
//!
//! Every part of that is invisible to every other gate in this repository. A `share` term that
//! leaked a thousandth into the interior would leave the contrast gate green, `tier_parity`
//! green, and the shipped inspector showing the file list faintly through its own labels. So
//! this asserts **byte identity**, not a tolerance: a tolerance here is a budget for exactly
//! the defect the rule exists to forbid, and the two frames are the same shader on the same
//! adapter with one scalar different, so there is no rounding to allow for.
//!
//! ## The visibility claim
//!
//! `the_ray_reaches_content_the_fragment_cannot_see` is the inverse, and it exists because of
//! research R15: a correct, bounded, four-ways-tested effect that moved 446,580 pixels by a peak
//! of 7/255 and was worth nothing. A refraction that satisfied the contrast claim by doing
//! nothing anywhere would pass the test above perfectly.
//!
//! **Its first version was itself an example of the failure it guards against, and the mutation
//! run is what caught it.** It compared the band at strength 1 against strength 0 and asked
//! whether the pixels differed — which they do enormously, because a transmitting panel shows
//! the backdrop where an opaque one showed the panel. That difference is *transmission*, not
//! *refraction*. Setting `GLASS_DEPTH` to 0.02, so the ray is bent exactly as before and then
//! travels almost nowhere, left the test green: the assertion could not see the difference
//! between glass and a hole.
//!
//! What it does instead is remove the alternative explanation. The only coloured thing behind
//! the panel sits in its **interior**, inset by the full bevel, so every fragment of the band
//! is over black and an undisplaced sample can only return black. Colour in the band is then
//! proof that the ray reached content the fragment sits nowhere near, which is the one thing
//! only a refraction does. See `masked` for which direction that turned out to be, and for the
//! fixture that measured zero because it assumed the other one.

// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the vocabulary of a
// test, not a hazard in one.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use qs_gpu::batcher::Renderer;
use qs_gpu::color::Srgba;
use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::frame::{DrawList, Instance};
use qs_gpu::path::RenderPath;

const WIDTH: u32 = 192;
const HEIGHT: u32 = 144;

/// The panel, in the middle of the viewport with room for its backdrop around it.
const PANEL: (f32, f32, f32, f32) = (32.0, 24.0, 128.0, 96.0);
const RADIUS: f32 = 16.0;
const BEVEL: f32 = 14.0;

/// A context, or `None` on a machine with no usable adapter.
///
/// Skipped rather than failed, for `offscreen_target.rs`'s reason: a headless runner with no
/// GPU is a real configuration, and a test that fails there teaches people to ignore the
/// suite. The skip prints, so a run that covered nothing is distinguishable from a pass.
fn context() -> Option<GpuContext> {
    let instance = new_instance(RenderPath::Primary);
    match GpuContext::new(RenderPath::Primary, instance, None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("refraction_stops_at_the_bevel: SKIPPED, no usable adapter ({e})");
            None
        }
    }
}

/// The same frame at a given refraction strength.
///
/// The backdrop is deliberately high-frequency and high-contrast, and it runs edge to edge
/// **under** the panel rather than around it. A refraction displaces what is behind it, so a
/// panel over a flat ground refracts perfectly and produces an identical frame at every
/// strength — which would make the contrast assertion below pass for the wrong reason and the
/// visibility assertion fail for no reason.
fn frame(strength: f32) -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.05, 0.05, 0.07, 1.0), 1);
    for i in 0..(HEIGHT / 6) {
        list.instances.push(Instance::rect(
            0.0,
            (i * 6) as f32,
            WIDTH as f32,
            3.0,
            0.0,
            if i % 2 == 0 {
                Srgba::new(0.95, 0.97, 1.0, 1.0)
            } else {
                Srgba::new(0.02, 0.02, 0.03, 1.0)
            },
        ));
    }
    for c in 0..(WIDTH / 9) {
        list.instances.push(Instance::rect(
            (c * 9) as f32,
            0.0,
            4.0,
            HEIGHT as f32,
            0.0,
            Srgba::new(0.10, 0.55, 0.95, 1.0),
        ));
    }
    panel(&mut list, strength);
    list
}

/// A frame where the only coloured thing behind the panel is somewhere no undisplaced sample
/// can reach: a green rectangle covering the panel's **interior**, with black everywhere the
/// bevel band actually sits.
///
/// # Which way the ray goes, which this fixture got wrong first
///
/// The obvious mask is the other way round — colour outside the panel, black under it — on the
/// intuition that a bevel's normal tilts outward and so the ray must reach outward. It does
/// not, and the fixture built that way measured a peak green bias of exactly 0.
///
/// Light entering a **denser** medium bends *toward* the normal, so the refracted ray turns
/// back across the surface and travels **inward**. That is why a glass bevel magnifies rather
/// than shrinking what is behind it, and it is visible in `docs/refraction/refract_panel.png`
/// once you know to look: the content inside the lip is the content from further in, pulled
/// out to the edge. With `GLASS_IOR` 1.52 and this bevel the displacement reaches the
/// `REFRACT_MAX_PX` clamp at 24 px, which is comfortably past the 14 px band.
///
/// # Why green rather than bright
///
/// Brightness cannot discriminate. A bevel's own specular lip reaches 613/765 at the boundary
/// because Fresnel goes to one there — that is the whole reason an edge catches light — so
/// "brighter than the surface can manage" is satisfied by the surface. Hue can: the panel's
/// albedo is neutral, its metalness is zero and the environment is the palette's neutral sky,
/// so nothing in this frame returns a green-dominant pixel except a sample that landed on the
/// rectangle below.
fn masked(strength: f32) -> DrawList {
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.0, 0.0, 0.0, 1.0), 1);
    let (x, y, w, h) = PANEL;
    // Inset by the bevel, so no fragment of the band sits over it. Square corners and no
    // radius: a rounded inset would leave the four corners' samples landing on black and the
    // test would be measuring the corners' geometry rather than the ray's reach.
    list.instances.push(Instance::rect(
        x + BEVEL,
        y + BEVEL,
        w - 2.0 * BEVEL,
        h - 2.0 * BEVEL,
        0.0,
        Srgba::new(0.0, 1.0, 0.0, 1.0),
    ));
    panel(&mut list, strength);
    list
}

fn panel(list: &mut DrawList, strength: f32) {
    let (x, y, w, h) = PANEL;
    list.instances.push(Instance::refract(
        x,
        y,
        w,
        h,
        RADIUS,
        BEVEL,
        0.32,
        0.0,
        0.55,
        strength,
        Srgba::new(0.196, 0.196, 0.212, 1.0),
    ));
    list.end_batch(None, false);
}

/// Render one draw list and read the surface back.
fn draw(ctx: &GpuContext, list: &DrawList) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("refraction test surface"),
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
    let mut renderer = Renderer::new(ctx, 256);
    let commands = renderer.render(ctx, &view, list, None, None);

    let unpadded = WIDTH * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("refraction readback"),
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

/// The signed distance to the panel's rounded boundary, negative inside.
///
/// A transcription of `sd_rounded_box` from `shaders/instance.wgsl`, for the reason
/// `tier_parity` transcribes rather than approximates: "deeper than the bevel" has to mean here
/// what it means there, or the region this test guards is not the region the shader guards.
fn distance(px: u32, py: u32) -> f32 {
    let (x, y, w, h) = PANEL;
    let (hx, hy) = (w * 0.5, h * 0.5);
    let r = RADIUS.min(hx.min(hy));
    let lx = (px as f32 + 0.5) - (x + hx);
    let ly = (py as f32 + 0.5) - (y + hy);
    let qx = lx.abs() - hx + r;
    let qy = ly.abs() - hy + r;
    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
}

/// Every pixel where `region` holds, as `(x, y, index)`.
fn region(keep: impl Fn(f32) -> bool) -> Vec<(u32, u32, usize)> {
    let mut out = Vec::new();
    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            if keep(distance(px, py)) {
                out.push((px, py, ((py * WIDTH + px) * 4) as usize));
            }
        }
    }
    out
}

#[test]
fn refraction_at_zero_strength_is_the_pbr_surface() {
    let Some(ctx) = context() else { return };

    // The same geometry and the same four surface parameters through the two kinds. The only
    // fields that differ are `kind` and the scalar in `param`, which carries emission on one
    // and refraction on the other -- both zero here.
    let mut refracting = frame(0.0);
    let mut lit = DrawList::default();
    lit.reset([WIDTH, HEIGHT], Srgba::new(0.05, 0.05, 0.07, 1.0), 1);
    // Everything except the panel, copied rather than rebuilt: a second transcription of the
    // backdrop is a second thing that can drift, and this comparison is worthless if the two
    // frames differ anywhere but the panel.
    lit.instances.extend(
        refracting
            .instances
            .iter()
            .take(refracting.instances.len() - 1),
    );
    let (x, y, w, h) = PANEL;
    lit.instances.push(Instance::pbr(
        x,
        y,
        w,
        h,
        RADIUS,
        BEVEL,
        0.32,
        0.0,
        0.55,
        0.0,
        Srgba::new(0.196, 0.196, 0.212, 1.0),
    ));
    lit.end_batch(None, false);
    refracting.instances.shrink_to_fit();

    let a = draw(&ctx, &refracting);
    let b = draw(&ctx, &lit);

    // Byte identity, not a tolerance. This is the claim `Instance::refract` makes in prose --
    // "zero is not 'off with rounding', it is the PBR surface exactly" -- and it is what makes
    // the no-backdrop degradation a property of the arithmetic rather than a second code path
    // somebody has to keep in step. It also crosses the one-pass/two-pass boundary: the
    // refracting frame takes the offscreen path because `needs_backdrop` is true for its kind
    // and the lit one does not, so a difference here would also be
    // `the_two_pass_path_is_pixel_identical_to_the_one_pass_path` failing at one remove.
    let differing = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| p != q)
        .count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} pixels differ between a refraction at strength 0 and the PBR \
         surface it claims to be. The floor, the shader's no-backdrop degradation and the \
         `refract/zero-strength` parity fixture all rest on those two being the same frame.",
        a.len() / 4
    );
}

#[test]
fn refraction_never_reaches_the_middle_of_a_surface() {
    let Some(ctx) = context() else { return };
    let glass = draw(&ctx, &frame(1.0));
    let plain = draw(&ctx, &frame(0.0));

    // Deeper than the bevel, and one pixel of margin so the assertion is about the interior
    // rather than about which side of the boundary a half-covered fragment landed on.
    let interior = region(|d| d <= -(BEVEL + 1.0));
    assert!(
        interior.len() > 2_000,
        "the interior region is {} px, which is too small to be evidence of anything",
        interior.len()
    );

    for (px, py, i) in interior {
        assert_eq!(
            glass[i..i + 4],
            plain[i..i + 4],
            "({px}, {py}) is {:.1} px inside the boundary and moved when the refraction was \
             turned on. Refraction must be identically zero deeper than the bevel: the contrast \
             gate reads the authored albedo and cannot see this, so a panel whose middle moves \
             ships a shifting ground under a label under a green build. See qs_ui::substance.",
            -distance(px, py)
        );
    }
}

#[test]
fn the_ray_reaches_content_the_fragment_cannot_see() {
    let Some(ctx) = context() else { return };
    let glass = draw(&ctx, &masked(1.0));
    let opaque = draw(&ctx, &masked(0.0));

    // Inside the shape and within the bevel, with a pixel of margin at each end so neither the
    // antialiased boundary nor the interior's exact zero is counted.
    let band = region(|d| d < -1.0 && d > -(BEVEL - 1.0));
    assert!(
        band.len() > 1_000,
        "the bevel band is {} px, which is too small to be evidence of anything",
        band.len()
    );

    // How green a pixel is, beyond what a neutral surface could account for. The surface's own
    // shading — albedo, key light, sky, the specular lip — is neutral, so `ceiling` measures
    // what this frame can produce with no ray going anywhere, rather than assuming it is zero.
    // The channels are normalised out of the surface format before comparison; reading raw
    // bytes and calling the second one green is how a test misses a swizzle.
    let green_bias = |px: &[u8], i: usize| {
        let p = if matches!(
            ctx.capabilities.surface_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            [px[i + 2], px[i + 1], px[i]]
        } else {
            [px[i], px[i + 1], px[i + 2]]
        };
        i32::from(p[1]) - i32::from(p[0]).max(i32::from(p[2]))
    };
    let ceiling = band
        .iter()
        .map(|(_, _, i)| green_bias(&opaque, *i))
        .max()
        .expect("the band is non-empty");

    let reached = band
        .iter()
        .filter(|(_, _, i)| green_bias(&glass, *i) > ceiling + 40)
        .count();
    let peak = band
        .iter()
        .map(|(_, _, i)| green_bias(&glass, *i))
        .max()
        .expect("the band is non-empty");

    assert!(
        reached > band.len() / 20,
        "only {reached} of {} bevel-band pixels are more green than this frame can produce \
         without a ray travelling (ceiling {ceiling}/255, peak {peak}/255). The backdrop is \
         black everywhere the band sits and green only in the panel's interior, so a green \
         fragment is the only evidence that a sample left the place it started, and the ray \
         bends INWARD -- see `masked`. Without it the effect is \
         transmission, not refraction — which is exactly what a bent ray that travels nowhere \
         looks like, and it is research R15's finding wearing a different hat.",
        band.len()
    );
}
