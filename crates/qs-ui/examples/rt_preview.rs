//! Render the ray-tracing proposal against the real palette, and write PNGs to look at.
//!
//! `cargo run -p qs-ui --example rt_preview -- out.png`
//! `QS_RT=shadow,ao,bounce,focus cargo run -p qs-ui --example rt_preview -- out.png`
//!
//! A LOOKING SPIKE. It shares no code with the shipped renderer and is not meant to: the
//! question it answers is whether an orthographic 3D scene is worth building, and the honest
//! way to answer that is to look at one rather than to describe it.
//!
//! What it is faithful about: the palette (every colour comes from `design/tokens.json`), the
//! microfacet model (the same GGX/Smith/Schlick the shipped shader runs), and the geometry
//! (the same rectangles, at the same screen positions, because the camera is orthographic).
//!
//! What it is not: fast, tiled, or hardware-traced. See the shader's header.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use bytemuck::{Pod, Zeroable};

use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::path::RenderPath;
use qs_ui::tokens::{Theme, Tokens};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 900;
const HEIGHT: u32 = 640;

/// Catalog tiles are small and wide: each shows one idea, not a whole window.
const TILE_W: u32 = 460;
const TILE_H: u32 = 250;
const SCALE: f32 = 2.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Slab {
    centre_radius: [f32; 4],
    half_rough: [f32; 4],
    albedo_metal: [f32; 4],
    emissive: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Scene {
    viewport: [f32; 2],
    count: u32,
    flags: u32,
    light: [f32; 4],
    env_horizon: [f32; 4],
    env_zenith: [f32; 4],
    focus: [f32; 4],
}

const FLAG_SHADOW: u32 = 1;
const FLAG_AO: u32 = 2;
const FLAG_BOUNCE: u32 = 4;
const FLAG_FOCUS: u32 = 8;

/// Straight linear RGB for a token, which is the space the shader lights in.
fn linear(tokens: &Tokens, name: &str) -> [f32; 3] {
    let c = tokens.color(name);
    [
        qs_gpu::color::srgb_to_linear(c.r),
        qs_gpu::color::srgb_to_linear(c.g),
        qs_gpu::color::srgb_to_linear(c.b),
    ]
}

/// The key light as this shader spells it, from the rig: direction toward the light and
/// the penumbra hardness `1 / tan(size / 2)` — one light for the whole product.
fn rig_light(tokens: &Tokens) -> [f32; 4] {
    let key = tokens.lighting().rig.key;
    let d = key.direction;
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    [
        d[0] / len,
        d[1] / len,
        d[2] / len,
        1.0 / (key.size_deg.to_radians() * 0.5).tan().max(1e-4),
    ]
}

fn slab(
    centre: [f32; 3],
    half: [f32; 3],
    radius: f32,
    albedo: [f32; 3],
    rough: f32,
    metal: f32,
) -> Slab {
    Slab {
        centre_radius: [centre[0], centre[1], centre[2], radius],
        half_rough: [half[0], half[1], half[2], rough],
        albedo_metal: [albedo[0], albedo[1], albedo[2], metal],
        emissive: [0.0; 4],
    }
}

/// One window's worth of interface, from the REAL scene (T005).
///
/// Promoted from the hand-built spike: the slabs now come from `qs_ui::scene::SceneBuilder`
/// over the shipped materials, so the elevations are the authored scale in
/// `design/tokens.json` and the allowances ride along — the preview can no longer drift
/// from what the application actually publishes. What is still local: the conversion into
/// this shader's own slab layout, and one **US2 preview** override marked below.
fn build(tokens: &Tokens) -> Vec<Slab> {
    use qs_ui::material::{Surface, name};
    use qs_ui::scene::SceneBuilder;

    let px = |logical: f32| logical * SCALE;
    let (w, h) = (WIDTH as f32, HEIGHT as f32);
    let mut builder = SceneBuilder::new(1, [w, h], 64.0, Default::default());
    let mut admit = |material: &str, surface: Surface| {
        if let Some(slab) = tokens.scene_slab(material, surface) {
            builder.admit(slab);
        }
    };

    // The same arrangement the spike drew, painted with the shipped materials.
    admit(
        name::SURFACE_CANVAS,
        Surface::new(0.0, 0.0, w, h, 0.0, SCALE),
    );
    let bar_h = px(44.0);
    admit(
        name::CHROME_BAR,
        Surface::new(0.0, 0.0, w, bar_h, 0.0, SCALE),
    );
    admit(
        name::CHROME_CHIP_HOVER,
        Surface::new(
            px(70.0),
            bar_h * 0.5 - px(13.0),
            px(78.0),
            px(26.0),
            px(6.0),
            SCALE,
        ),
    );

    let row_x = px(14.0);
    let row_w = w - row_x * 2.0;
    let row_h = px(30.0);
    let top = bar_h + px(26.0);
    for i in 0..7u32 {
        let y = top + i as f32 * (row_h + px(5.0));
        let surface = Surface::new(row_x, y, row_w, row_h, px(6.0), SCALE);
        let material = match i {
            2 => name::ROW_HOVER,
            4 => name::ROW_SELECTED,
            _ if i % 2 == 1 => name::ROW_BODY_ALT,
            _ => name::ROW_BODY,
        };
        admit(material, surface);
    }
    let shelf_h = px(26.0);
    admit(
        name::CHROME_SHELF,
        Surface::new(0.0, h - shelf_h, w, shelf_h, 0.0, SCALE),
    );

    let scene = builder.finish();
    let accent = linear(tokens, "border/focus");
    let selected = tokens
        .material(name::ROW_SELECTED)
        .map(|m| m.elevation * SCALE)
        .unwrap_or(0.0);
    scene
        .slabs
        .iter()
        .map(|s| {
            let half_thick = (s.thickness * 0.5).max(0.5);
            let mut converted = slab(
                [
                    s.rect[0] + s.rect[2] * 0.5,
                    s.rect[1] + s.rect[3] * 0.5,
                    s.elevation - half_thick,
                ],
                [s.rect[2] * 0.5, s.rect[3] * 0.5, half_thick],
                s.radius,
                s.albedo,
                s.roughness,
                s.metalness,
            );
            // A US2 PREVIEW, marked as one: emission is not authored on any material yet
            // (T050 is US2's task), and the emissive selection is the idea this preview
            // exists to look at. The override names the elevation it keys on so it breaks
            // loudly if the selected row's step changes.
            if (s.elevation - selected).abs() < 0.01 && selected > 0.0 && s.rect[3] < 100.0 {
                converted.emissive = [accent[0], accent[1], accent[2], 1.5];
            }
            converted
        })
        .collect()
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rt.png".to_string());

    // Which secondary rays to trace. Off by default so the first image is the baseline the
    // others have to earn their cost against.
    let requested = std::env::var("QS_RT").unwrap_or_default();
    let mut flags = 0u32;
    for name in requested.split(',') {
        match name.trim() {
            "shadow" => flags |= FLAG_SHADOW,
            "ao" => flags |= FLAG_AO,
            "bounce" => flags |= FLAG_BOUNCE,
            "focus" => flags |= FLAG_FOCUS,
            "all" => flags |= FLAG_SHADOW | FLAG_AO | FLAG_BOUNCE | FLAG_FOCUS,
            _ => {}
        }
    }

    let instance = new_instance(RenderPath::Primary);
    let ctx = GpuContext::new(RenderPath::Primary, instance, None).expect("no usable adapter");
    eprintln!(
        "rt_preview: {} via {:?} | flags = {flags:#06b}",
        ctx.capabilities.adapter_name, ctx.capabilities.backend
    );

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_preview"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rt_preview.wgsl").into()),
        });

    if std::env::var("QS_TILES").is_ok() {
        // The catalog contact sheet. Dark theme only: it is where the light reads hardest, and a
        // catalog that showed both would be thirty more images nobody compares.
        // Both themes. Material ideas read far better in the light theme, where surfaces have
        // the reflectance to show a roughness difference at all; emissive ideas read better in
        // the dark theme, where there is contrast for light to make. Rendering both and choosing
        // per idea is more honest than picking one and calling the weak half subtle.
        for (theme, suffix) in [(Theme::Dark, "dark"), (Theme::Light, "light")] {
            let tokens = Tokens::embedded(theme).expect("tokens");
            let lift = linear(&tokens, "surface/row-hover");
            let horizon = linear(&tokens, "surface/base");
            let sky = |c: f32| c * 0.85 + 0.15;
            for (name, slabs) in tiles(&tokens) {
                let scene = Scene {
                    viewport: [TILE_W as f32, TILE_H as f32],
                    count: slabs.len() as u32,
                    flags: FLAG_SHADOW | FLAG_AO | FLAG_BOUNCE,
                    light: rig_light(&tokens),
                    env_horizon: [horizon[0], horizon[1], horizon[2], 1.0],
                    env_zenith: [sky(lift[0]), sky(lift[1]), sky(lift[2]), 1.0],
                    focus: [0.0, 0.0, 0.0, 0.0],
                };
                let pixels = render(&ctx, &shader, &scene, &slabs, TILE_W, TILE_H);
                let path = format!("target/preview/tile-{name}-{suffix}.png");
                write_png(&path, &pixels, TILE_W, TILE_H);
                eprintln!("tile: {path}");
            }
        }
        return;
    }

    for (theme, label) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
        let tokens = Tokens::embedded(theme).expect("tokens");
        let slabs = build(&tokens);

        // The room the interface is standing in, from the palette rather than from the
        // shader -- the same rule the shipped `Environment` follows.
        let horizon = linear(&tokens, "surface/base");
        let lift = linear(&tokens, "surface/row-hover");
        // The ceiling is brighter than the floor. Authoring the sky from a surface token
        // alone gave the dark theme an environment darker than its own ground, and every
        // shadow fell to black -- see the note in the report.
        // Brighten TOWARD white rather than by a factor. Multiplying worked in the dark theme
        // and sent the light theme -- whose ground is already near white -- to 2.8x, so every
        // surface clipped. A room is lighter than its floor; it is not three times its floor.
        let sky = |c: f32| c * 0.85 + 0.15;
        let zenith = [sky(lift[0]), sky(lift[1]), sky(lift[2])];
        let scene = Scene {
            viewport: [WIDTH as f32, HEIGHT as f32],
            count: slabs.len() as u32,
            flags,
            // Toward the light, from the rig — the same direction the PBR bevels, the
            // contact shadows and the shipped raymarch use. `w` is the penumbra hardness.
            light: rig_light(&tokens),
            env_horizon: [horizon[0], horizon[1], horizon[2], 1.0],
            env_zenith: [zenith[0], zenith[1], zenith[2], 1.0],
            // The focus lamp, parked over the selected row.
            focus: [
                WIDTH as f32 * 0.5,
                88.0 + 26.0 + 4.0 * (60.0 + 10.0) + 30.0,
                120.0,
                2.4,
            ],
        };

        let pixels = render(&ctx, &shader, &scene, &slabs, WIDTH, HEIGHT);
        let path = out.replace(".png", &format!("-{label}.png"));
        write_png(&path, &pixels, WIDTH, HEIGHT);
        eprintln!("rt_preview: wrote {path} ({} slabs)", slabs.len());
    }
}

// -- catalog tiles -------------------------------------------------------------------------
//
// One small scene per idea in the catalog. These are renders, not illustrations: the same
// shader, the same palette, the same microfacet model. An idea this harness cannot express is
// absent here rather than faked, and the catalog says which.

/// The ground every tile stands on.
fn tile_ground(tokens: &Tokens) -> Slab {
    slab(
        [TILE_W as f32 * 0.5, TILE_H as f32 * 0.5, -60.0],
        [TILE_W as f32 * 0.5, TILE_H as f32 * 0.5, 60.0],
        0.0,
        linear(tokens, "surface/base"),
        0.92,
        0.0,
    )
}

/// Row `i` of `n` in a tile: centre, half-extent, radius.
fn tile_row(i: usize, n: usize, elevation: f32, thickness: f32) -> ([f32; 3], [f32; 3], f32) {
    let margin = 26.0;
    let gap = 10.0;
    let usable = TILE_H as f32 - margin * 2.0;
    let h = (usable - gap * (n as f32 - 1.0)) / n as f32;
    let cy = margin + i as f32 * (h + gap) + h * 0.5;
    let w = TILE_W as f32 - margin * 2.0;
    (
        [TILE_W as f32 * 0.5, cy, elevation - thickness],
        [w * 0.5, h * 0.5, thickness],
        10.0,
    )
}

/// Every tile the harness can honestly render, as (file stem, slabs).
#[allow(clippy::too_many_lines)]
fn tiles(tokens: &Tokens) -> Vec<(&'static str, Vec<Slab>)> {
    let base = linear(tokens, "surface/row-alt");
    let sel = linear(tokens, "surface/row-selected");
    let accent = linear(tokens, "border/focus");
    let amber = linear(tokens, "rail/modified");
    let green = linear(tokens, "rail/added");
    let red = linear(tokens, "rail/conflict");

    let mut out: Vec<(&'static str, Vec<Slab>)> = Vec::new();

    // 01 - roughness as age. Same colour, same height; only the polish changes.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..5 {
        let (c, hf, r) = tile_row(i, 5, 12.0, 5.0);
        v.push(slab(c, hf, r, base, 0.06 + i as f32 * 0.22, 0.35));
    }
    out.push(("01-roughness-as-age", v));

    // 02 - thickness as size. Same material; the slab deepens and casts further.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..5 {
        let thick = 2.0 + i as f32 * 7.0;
        let (c, hf, r) = tile_row(i, 5, 6.0 + thick, thick);
        v.push(slab(c, hf, r, base, 0.4, 0.1));
    }
    out.push(("02-thickness-as-size", v));

    // 03 - metalness as kind. Dielectric through to metal, one step at a time.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..5 {
        let (c, hf, r) = tile_row(i, 5, 12.0, 5.0);
        v.push(slab(c, hf, r, base, 0.25, i as f32 / 4.0));
    }
    out.push(("03-metalness-as-kind", v));

    // 04 - reflectivity as permission. The middle row reflects nothing at all.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..5 {
        let (c, hf, r) = tile_row(i, 5, 12.0, 5.0);
        if i == 2 {
            v.push(slab(c, hf, r, [0.008, 0.008, 0.010], 0.95, 0.0));
        } else {
            v.push(slab(c, hf, r, base, 0.22, 0.5));
        }
    }
    out.push(("04-reflectivity-as-permission", v));

    // 05 - contact hardening. Three heights, three shadows, one light.
    let mut v = vec![tile_ground(tokens)];
    for (i, elev) in [4.0f32, 16.0, 44.0].into_iter().enumerate() {
        let (c, hf, r) = tile_row(i, 3, elev, 4.0);
        v.push(slab(c, hf, r, base, 0.5, 0.05));
    }
    out.push(("05-contact-hardening", v));

    // 06 - occlusion as density. Sparse rows against a stack of thin ones.
    let mut v = vec![tile_ground(tokens)];
    for i in [0usize, 2] {
        let (c, hf, r) = tile_row(i, 3, 10.0, 4.0);
        v.push(slab(c, hf, r, base, 0.6, 0.0));
    }
    let (c, hf, r) = tile_row(1, 3, 10.0, 4.0);
    for k in 0..7 {
        let dy = (k as f32 - 3.0) * (hf[1] * 2.0 / 7.0);
        v.push(slab(
            [c[0], c[1] + dy, c[2]],
            [hf[0], hf[1] / 7.0 * 0.7, hf[2]],
            r * 0.4,
            base,
            0.6,
            0.0,
        ));
    }
    out.push(("06-occlusion-as-density", v));

    // 07 - search as a wavefront, frozen at one instant.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..6 {
        let (c, hf, r) = tile_row(i, 6, 14.0, 5.0);
        let mut sl = slab(c, hf, r, base, 0.45, 0.0);
        let front = 1.0 - (i as f32 - 1.6).abs() / 2.2;
        if front > 0.0 {
            sl.emissive = [accent[0], accent[1], accent[2], front * 1.6];
        }
        v.push(sl);
    }
    out.push(("07-search-wavefront", v));

    // 08 - radiance as relevance. One emitter; the neighbours read its distance.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..6 {
        let (c, hf, r) = tile_row(i, 6, if i == 2 { 32.0 } else { 8.0 }, 5.0);
        let mut sl = slab(c, hf, r, if i == 2 { sel } else { base }, 0.4, 0.0);
        if i == 2 {
            sl.emissive = [accent[0], accent[1], accent[2], 1.8];
        }
        v.push(sl);
    }
    out.push(("08-radiance-as-relevance", v));

    // 09 - shadow as a drop target. Held above the list, aimed by its shadow.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..5 {
        let (c, hf, r) = tile_row(i, 5, 5.0, 3.0);
        v.push(slab(c, hf, r, base, 0.6, 0.0));
    }
    let (c, hf, r) = tile_row(2, 5, 78.0, 6.0);
    v.push(slab(
        [c[0] + 30.0, c[1] - 12.0, c[2]],
        [hf[0] * 0.55, hf[1], hf[2]],
        r,
        sel,
        0.3,
        0.1,
    ));
    out.push(("09-shadow-as-drop-target", v));

    // 10 - progress as a moving shadow. One occluder; half the list is behind its edge.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..6 {
        let (c, hf, r) = tile_row(i, 6, 5.0, 3.0);
        v.push(slab(c, hf, r, base, 0.6, 0.0));
    }
    v.push(slab(
        [TILE_W as f32 * 0.28, TILE_H as f32 * 0.5, 56.0],
        [TILE_W as f32 * 0.28, TILE_H as f32 * 0.66, 8.0],
        0.0,
        [0.02, 0.02, 0.03],
        0.9,
        0.0,
    ));
    out.push(("10-progress-as-moving-shadow", v));

    // 11 - the room takes the folder colour. Three emitters, three hues, one ground.
    let mut v = vec![tile_ground(tokens)];
    for (i, tint) in [amber, green, accent].into_iter().enumerate() {
        let (c, hf, r) = tile_row(i, 3, 36.0, 8.0);
        let mut sl = slab(
            [c[0], c[1], c[2]],
            [hf[0] * 0.40, hf[1] * 0.75, hf[2]],
            r,
            base,
            0.35,
            0.0,
        );
        sl.emissive = [tint[0], tint[1], tint[2], 1.9];
        v.push(sl);
    }
    out.push(("11-room-takes-folder-colour", v));

    // 12 - energy as heat. The same row at four operation costs.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..4 {
        let (c, hf, r) = tile_row(i, 4, 16.0, 5.0);
        let mut sl = slab(c, hf, r, base, 0.4, 0.0);
        sl.emissive = [amber[0], amber[1], amber[2], i as f32 * 0.75];
        v.push(sl);
    }
    out.push(("12-energy-as-heat", v));

    // 13 - conflict as interference. Two emitters overlapping on one ground.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..5 {
        let (c, hf, r) = tile_row(i, 5, 5.0, 3.0);
        v.push(slab(c, hf, r, base, 0.6, 0.0));
    }
    for (dx, tint) in [(-62.0f32, red), (62.0, amber)] {
        let mut sl = slab(
            [TILE_W as f32 * 0.5 + dx, TILE_H as f32 * 0.5, 48.0],
            [44.0, 17.0, 7.0],
            8.0,
            base,
            0.3,
            0.0,
        );
        sl.emissive = [tint[0], tint[1], tint[2], 2.1];
        v.push(sl);
    }
    out.push(("13-conflict-as-interference", v));

    // 14 - depth as tree depth. Each step down the tree sits further down the box.
    let mut v = vec![tile_ground(tokens)];
    for i in 0..5 {
        let (c, hf, r) = tile_row(i, 5, 48.0 - i as f32 * 10.0, 4.0);
        v.push(slab(
            [c[0] + i as f32 * 17.0, c[1], c[2]],
            [hf[0] - i as f32 * 17.0, hf[1], hf[2]],
            r,
            base,
            0.5,
            0.05,
        ));
    }
    out.push(("14-depth-as-tree-depth", v));

    // 15 - the thesis in one tile: five facts, five materials, one list.
    let mut v = vec![tile_ground(tokens)];
    let spec: [(f32, f32, f32, f32); 5] = [
        (0.08, 0.55, 4.0, 0.0),
        (0.30, 0.20, 10.0, 0.0),
        (0.62, 0.05, 5.0, 0.0),
        (0.92, 0.00, 3.0, 0.0),
        (0.22, 0.85, 15.0, 1.5),
    ];
    for (i, (rough, metal, thick, emit)) in spec.into_iter().enumerate() {
        let (c, hf, r) = tile_row(i, 5, 8.0 + thick, thick);
        let mut sl = slab(c, hf, r, if emit > 0.0 { sel } else { base }, rough, metal);
        if emit > 0.0 {
            sl.emissive = [accent[0], accent[1], accent[2], emit];
        }
        v.push(sl);
    }
    out.push(("15-thesis-one-list", v));

    out
}

fn render(
    ctx: &GpuContext,
    shader: &wgpu::ShaderModule,
    scene: &Scene,
    slabs: &[Slab],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let format = ctx.capabilities.surface_format;

    let scene_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rt scene"),
            contents: bytemuck::bytes_of(scene),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let slab_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rt slabs"),
            contents: bytemuck::cast_slice(slabs),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: slab_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rt_preview"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
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

    let unpadded = width * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
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
    ctx.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    let mapped = slice.get_mapped_range().expect("mapped");
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        for texel in mapped[start..start + unpadded as usize].chunks_exact(4) {
            pixels.extend_from_slice(&[texel[2], texel[1], texel[0], texel[3]]);
        }
    }
    drop(mapped);
    readback.unmap();
    pixels
}

fn write_png(path: &str, rgba: &[u8], width: u32, height: u32) {
    let mut pixmap = tiny_skia::Pixmap::new(width, height).expect("pixmap");
    for (dst, src) in pixmap.pixels_mut().iter_mut().zip(rgba.chunks_exact(4)) {
        *dst = tiny_skia::PremultipliedColorU8::from_rgba(src[0], src[1], src[2], src[3])
            .expect("premultiplied");
    }
    pixmap.save_png(path).expect("save");
}
