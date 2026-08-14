//! Per-primitive agreement between the CPU tier and the shader, with no GPU device.
//!
//! # Why this exists
//!
//! RP-2 says all three tiers consume the same draw lists. Consuming the same list is not
//! the same as producing the same pixels, and the gap between those two statements is
//! where [`PrimKind::Stroke`] lived for a while: the shader strokes *inside* the shape,
//! `tiny-skia` strokes *centred* on the path, and the two disagreed by a pixel on every
//! edge. Nothing caught it, because nothing had ever emitted a stroke.
//!
//! The thing that *would* have caught it is the cross-tier reference-image suite, which
//! needs a GPU device and a perceptual threshold that is still uncalibrated (research
//! R10). So this module takes the cheap half of that job and does it now: it transcribes
//! `shaders/instance.wgsl` into Rust, evaluates it per pixel, and compares the result
//! against [`CpuRasterizer`]'s output. No device, milliseconds, and a failure names one
//! primitive instead of one frame.
//!
//! It does **not** replace the reference-image suite. It cannot see anything that happens
//! above the draw list -- wrong colour, wrong position, a row that forgot to emit -- and
//! it cannot see driver-specific behaviour, because there is no driver. What it sees is
//! exactly one class of bug: the two tiers implementing the same primitive differently.
//!
//! # Why a transcription is possible at all
//!
//! `fs_main` has no `fwidth`, no `smoothstep` and no derivative of any kind -- the comment
//! at line 124 of the shader explains that `0.5 - d` is exact one-pixel coverage for a
//! distance field measured in pixels, and says so *because* it is what keeps the CPU tier
//! correct. That same property is what makes the shader reproducible in ordinary Rust: a
//! fragment's output depends only on its own interpolated inputs.
//!
//! # What the numbers can and cannot mean
//!
//! Two things make the tiers differ for reasons that are not bugs, and both were found by
//! building this suite rather than assumed beforehand.
//!
//! The first is definitional. `tiny-skia` computes analytic *area* coverage; the shader
//! computes `0.5` minus a *distance*. On a straight edge those are the same number. On a
//! curve they are two different exact answers to slightly different questions, and they
//! diverge in proportion to curvature -- which is why the rounded fixtures carry a larger
//! bound than the sharp ones.
//!
//! The second is [`CPU_EDGE_QUANTUM`]: `tiny-skia` resolves an edge to the nearest quarter
//! pixel, and the shader's distance field does not resolve it to anything, because it is
//! continuous. Geometry landing on a quarter therefore agrees to the last bit, and
//! geometry landing on an eighth -- a 1.875px stroke at 125% display scale, say -- does
//! not. That is a property of the fallback tier, measured here and bounded here.
//!
//! Every bound below is derived from one of those two numbers and stated in the fixture
//! that uses it. None was raised until a test went green.
//!
//! # What the bounds were checked against
//!
//! Twelve mutations, each applied for real and re-run: the CPU tier shifted a whole pixel;
//! shifted a *quarter* pixel; the stroke inset deleted (the original bug); the glyph blit
//! reading one texel across; the transcription's stroke band turned into a fill; its
//! radius clamp deleted; a `PrimKind` dropped from `ALL`; a `cases_for` arm stubbed out to
//! nothing; and `CPU_EDGE_QUANTUM` sharpened. Those nine all fail the suite. The radius clamp
//! was the one that initially did *not* -- no fixture asked for a radius large enough to
//! clamp, so deleting the clamp changed nothing anywhere. `rect/over-large-radius` and
//! `stroke/over-large-radius` exist because of that survivor.
//!
//! Three more arrived with the gradient, and they matter more than the count suggests
//! because a ramp can be wrong in ways that still look like a ramp. Making the CPU tier
//! interpolate in linear light rather than Oklab takes `gradient/vertical-sharp` to 33
//! channels and also reddens `the_ramp_is_walked_in_oklab_and_not_in_linear_srgb` -- two
//! independent failures, which is the point of having both. Normalizing the gradient axis
//! against the box's half-width instead of its support takes it to 29. And dropping the
//! dither from the CPU tier while the shader keeps it takes it to 1, which is the whole
//! bound now that the three sharp gradient fixtures measure zero.
//!
//! Those fixtures used to allow 1, and the 1 was the CPU tier approximating an Oklab curve
//! with 33 chords. `prim-noise-dither` had to delete that approximation -- a stop list has
//! nowhere to put a per-pixel offset -- and the tiers came out bit-identical on a ramp, so
//! the bound was tightened to nothing rather than left with slack nobody was using.
//!
//! # Two questions, chosen by the primitive
//!
//! Everything above describes the [`Fidelity::Exact`] question: *do the two tiers agree?*
//! It is the right question for a rounded rect and an impossible one for an effect
//! `tiny-skia` cannot draw at any tolerance, so a primitive picks which question it is
//! asked. An [`Fidelity::Enhanced`] kind is asked the other one -- *did the CPU tier draw
//! the floor its `PrimKind` declares?* -- and it is asked exactly, with no tolerance at all,
//! because both sides of that comparison are the same rasterizer.
//!
//! What the second question buys is not leniency. It is that the fallback stops being
//! whatever fell out and becomes something somebody chose, wrote down, and can be held to.
//!
//! [`PrimKind::Glow`] is the one `Enhanced` kind, and its fixtures ask the floor question
//! instead. The synthetic declarations in `the_floor_check_*` are still here and still run
//! in both directions: they are what shows the check can go red, which a real `Enhanced`
//! kind cannot demonstrate about itself.
//!
//! One thing the floor route structurally cannot see is whether the *shader* draws anything
//! -- a branch returning zero satisfies "the CPU tier drew nothing" perfectly. The four
//! tests under "the glow's own geometry" are that half, and they hold the transcription to
//! the halo's measured profile rather than to a snapshot of it.
//!
//! A tolerance on its own is a weak assertion, so the comparison does not rest on one.
//! Alongside the channel difference it pins the ink's bounding box *exactly*, its
//! coverage-weighted centroid to a fraction of a pixel, and the total ink laid down to a
//! fraction of a percent. That last one carries the weight where the others cannot: the
//! quarter-pixel step takes from one edge exactly what it gives the opposite edge, so it
//! moves the centroid and cannot move the total.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use tiny_skia::Pixmap;

use crate::atlas::PendingUpload;
use crate::color::Srgba;
use crate::cpu_raster::CpuRasterizer;
use crate::frame::{DrawList, Fidelity, Floor, Instance, PrimKind};

/// `shaders/instance.wgsl`, transcribed function for function.
///
/// The transcription is the whole point, so it is kept boring and literal: same names,
/// same order, same operations, one Rust function per WGSL function, with the shader's
/// line numbers in the doc comments. If it were rewritten into idiomatic Rust it would
/// stop being evidence about the shader and start being a second opinion about geometry.
///
/// **If `instance.wgsl` changes, this must change with it.** Nothing enforces that
/// automatically; what the suite does enforce is that the *set of primitives* on both
/// sides matches, which is the drift that has actually happened.
pub(crate) mod shader {
    /// `const KIND_STROKE: u32 = 1u;` -- shader line 15.
    pub const KIND_STROKE: u32 = 1;
    /// `const KIND_GLYPH: u32 = 2u;` -- shader line 16.
    pub const KIND_GLYPH: u32 = 2;
    /// `const KIND_GRADIENT: u32 = 3u;` -- shader line 17.
    pub const KIND_GRADIENT: u32 = 3;
    /// `const KIND_GLOW: u32 = 4u;` -- shader line 18.
    pub const KIND_GLOW: u32 = 4;
    /// `const KIND_RIM: u32 = 5u;` -- shader line 19.
    pub const KIND_RIM: u32 = 5;
    /// `const KIND_PBR: u32 = 6u;` -- shader line 20.
    pub const KIND_PBR: u32 = 6;
    /// `const KIND_SWEEP: u32 = 7u;` -- shader line 21.
    pub const KIND_SWEEP: u32 = 7;
    /// `const KIND_FIELD: u32 = 8u;` -- shader line 22.
    pub const KIND_FIELD: u32 = 8;

    pub const PI: f32 = std::f32::consts::PI;
    pub const TAU: f32 = std::f32::consts::TAU;

    /// `fn sd_rounded_box_grad(p, b, r)` -- the distance field's analytic gradient, i.e. the
    /// direction of the nearest edge.
    ///
    /// Transcribed rather than approximated by finite differences, which is the whole reason
    /// the CPU side can reproduce a normal at all: a difference would need neighbouring
    /// fragments, and this tier has none.
    pub fn sd_rounded_box_grad(p: [f32; 2], b: [f32; 2], r: f32) -> [f32; 2] {
        let q = [p[0].abs() - b[0] + r, p[1].abs() - b[1] + r];
        let s = [sign(p[0]), sign(p[1])];
        if q[0].max(q[1]) > 0.0 {
            let m = [q[0].max(0.0), q[1].max(0.0)];
            let len = (m[0] * m[0] + m[1] * m[1]).sqrt();
            return [m[0] / len * s[0], m[1] / len * s[1]];
        }
        if q[0] > q[1] {
            return [s[0], 0.0];
        }
        [0.0, s[1]]
    }

    /// WGSL `sign`, which returns 0 for 0 -- not Rust's `f32::signum`, which returns 1.0.
    /// The difference is exactly one pixel wide and sits on the shape's centre lines.
    fn sign(v: f32) -> f32 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// `fn bevel_normal(distance, grad, width)` -- a quarter-round profile: vertical at the
    /// boundary, flat `width` pixels inward.
    pub fn bevel_normal(distance: f32, grad: [f32; 2], width: f32) -> [f32; 3] {
        if width <= 0.0 {
            return [0.0, 0.0, 1.0];
        }
        let t = (-distance / width).clamp(0.0, 1.0);
        let theta = (1.0 - t) * PI * 0.5;
        let (sin, cos) = theta.sin_cos();
        [grad[0] * sin, grad[1] * sin, cos]
    }

    /// `fn environment(ray)` -- an infinite sky, so the ray's elevation is the whole answer.
    pub fn environment(ray: [f32; 3], horizon: [f32; 3], zenith: [f32; 3]) -> [f32; 3] {
        let t = (ray[2] * 0.5 + 0.5).clamp(0.0, 1.0);
        [
            horizon[0] + (zenith[0] - horizon[0]) * t,
            horizon[1] + (zenith[1] - horizon[1]) * t,
            horizon[2] + (zenith[2] - horizon[2]) * t,
        ]
    }

    /// `fn distribution_ggx(n_dot_h, roughness)`.
    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let a = roughness * roughness;
        let a2 = a * a;
        let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        a2 / (PI * d * d).max(1e-7)
    }

    /// `fn visibility_smith(n_dot_v, n_dot_l, roughness)` -- height-correlated, with the
    /// BRDF's `1 / (4 (N.L)(N.V))` already folded in.
    pub fn visibility_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let a = roughness * roughness;
        let a2 = a * a;
        let lv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
        let ll = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
        0.5 / (lv + ll).max(1e-5)
    }

    /// `fn fresnel_schlick(cos_theta, f0)`.
    pub fn fresnel_schlick(cos_theta: f32, f0: [f32; 3]) -> [f32; 3] {
        let f = (1.0 - cos_theta).clamp(0.0, 1.0).powi(5);
        [
            f0[0] + (1.0 - f0[0]) * f,
            f0[1] + (1.0 - f0[1]) * f,
            f0[2] + (1.0 - f0[2]) * f,
        ]
    }

    /// `fn fresnel_roughness(cos_theta, f0, roughness)`.
    pub fn fresnel_roughness(cos_theta: f32, f0: [f32; 3], roughness: f32) -> [f32; 3] {
        let f = (1.0 - cos_theta).clamp(0.0, 1.0).powi(5);
        let ceiling = 1.0 - roughness;
        [
            f0[0] + (ceiling.max(f0[0]) - f0[0]) * f,
            f0[1] + (ceiling.max(f0[1]) - f0[1]) * f,
            f0[2] + (ceiling.max(f0[2]) - f0[2]) * f,
        ]
    }

    /// `const LIGHT_DIR` -- shader line, above and slightly to the left.
    ///
    /// Taken from `crate::frame` rather than re-typed here. It used to be a literal, and it
    /// stopped being one when a material gained the ability to cast a contact shadow: the
    /// direction a shadow falls and the direction a surface is shaded from have to be the same
    /// vector, and two copies of it are two chances for a window whose shadows point one way
    /// and whose highlights point the other.
    pub const LIGHT_DIR: [f32; 3] = crate::frame::LIGHT_DIR;
    /// `const LIGHT_RADIANCE` -- the exposure that puts a flat dielectric back at its albedo.
    pub const LIGHT_RADIANCE: f32 = 4.0757;

    /// `fn edge_emission(distance, bevel)` -- how much of a surface's emission reaches one
    /// fragment: all of it at the boundary, none of it `bevel` pixels inward.
    ///
    /// The rim's profile, and reusing `rim_t` rather than restating it is the point: the whole
    /// contrast argument is that emission stops exactly where the bevel does, so there must be
    /// one answer to "how far inside the edge am I" and not two.
    /// `const LAMP_FLOOR` / rib constants -- shader constants.
    pub const LAMP_FLOOR: f32 = 0.34;
    pub const LAMP_RIB_PERIOD: f32 = 17.0;
    pub const LAMP_RIB_DEPTH: f32 = 0.16;

    /// `fn lamp_emission(distance, bevel, local, half_size)` -- the tube profile: hot along
    /// the surface's own centre-line, a lip at the housing, ribs and grain over both.
    pub fn lamp_emission(distance: f32, bevel: f32, local: [f32; 2], half_size: [f32; 2]) -> f32 {
        let v = (local[1] / half_size[1].max(1.0)).clamp(-1.0, 1.0);
        let tube = (1.0 - v.abs()).max(0.0).powf(0.55);
        let body = LAMP_FLOOR + (1.0 - LAMP_FLOOR) * tube;
        let rim = 1.0 - rim_t(distance, bevel.max(1.0));
        let lip = rim * rim * 0.30;
        let ribs = 1.0
            - LAMP_RIB_DEPTH
                * (0.5 + 0.5 * (local[0] * (std::f32::consts::TAU / LAMP_RIB_PERIOD)).cos());
        // The literal is the shader's, digit for digit. A hash constant truncated on one
        // side of the transcription and not the other is a grain that differs between
        // tiers, which is precisely what this module exists to prevent.
        #[allow(clippy::excessive_precision)]
        const SCATTER: f32 = 43758.5453;
        let grain = 0.93 + 0.07 * ((local[0] * 12.9898 + local[1] * 78.233).sin() * SCATTER).fract();
        (body + lip) * ribs * grain
    }

    pub fn edge_emission(distance: f32, bevel: f32) -> f32 {
        let edge = 1.0 - rim_t(distance, bevel);
        edge * edge
    }

    /// `fn shade_pbr(...)` -- Cook-Torrance with an orthographic viewer.
    #[allow(clippy::too_many_arguments)]
    pub fn shade_pbr(
        normal: [f32; 3],
        albedo: [f32; 3],
        roughness: f32,
        metallic: f32,
        env_strength: f32,
        emissive: f32,
        horizon: [f32; 3],
        zenith: [f32; 3],
    ) -> [f32; 3] {
        let rough = roughness.clamp(0.045, 1.0);
        let metal = metallic.clamp(0.0, 1.0);

        let v = [0.0, 0.0, 1.0];
        let l = normalize3(LIGHT_DIR);
        let h = normalize3([l[0] + v[0], l[1] + v[1], l[2] + v[2]]);

        let n_dot_v = normal[2].max(1e-4);
        let n_dot_l = dot3(normal, l).max(0.0);
        let n_dot_h = dot3(normal, h).max(0.0);
        let v_dot_h = dot3(v, h).max(0.0);

        let f0 = [
            0.04 + (albedo[0] - 0.04) * metal,
            0.04 + (albedo[1] - 0.04) * metal,
            0.04 + (albedo[2] - 0.04) * metal,
        ];

        let d = distribution_ggx(n_dot_h, rough);
        let vis = visibility_smith(n_dot_v, n_dot_l, rough);
        let f = fresnel_schlick(v_dot_h, f0);

        let reflected = [
            2.0 * normal[0] * n_dot_v - v[0],
            2.0 * normal[1] * n_dot_v - v[1],
            2.0 * normal[2] * n_dot_v - v[2],
        ];
        let env_spec_tint = fresnel_roughness(n_dot_v, f0, rough);
        let env_spec = environment(reflected, horizon, zenith);
        let env_diff = environment(normal, horizon, zenith);

        // See the shader: a rig, not a gain.
        let sky_mix = env_strength.clamp(0.0, 1.0);
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let specular = d * vis * f[i];
            let kd = (1.0 - f[i]) * (1.0 - metal);
            let diffuse = kd * albedo[i] / PI;
            let direct = (diffuse + specular) * n_dot_l * LIGHT_RADIANCE * (1.0 - sky_mix);
            let ambient = (env_diff[i] * albedo[i] * (1.0 - metal)
                + env_spec[i] * env_spec_tint[i])
                * sky_mix;
            // Added, not mixed: a light is this surface plus light. `emissive` arrives already
            // weighted by `edge_emission`, so it is zero everywhere deeper than the bevel.
            out[i] = direct + ambient + albedo[i] * emissive.max(0.0);
        }
        out
    }

    fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn normalize3(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }
    // `KIND_RECT` (line 14) has no constant here on purpose: the shader reaches it as the
    // `else` of both branches, and a constant nothing compares against is dead weight
    // that would still have to be kept in sync.

    /// WGSL `unpack4x8unorm`: byte 0 of the word becomes component `x`.
    ///
    /// Worth transcribing rather than assuming -- the packing order is the one place a
    /// silent red/blue swap between the tiers could hide, since both tiers would still
    /// produce a plausible picture.
    pub fn unpack4x8unorm(v: u32) -> [f32; 4] {
        [
            f32::from(((v) & 0xFF) as u8) / 255.0,
            f32::from(((v >> 8) & 0xFF) as u8) / 255.0,
            f32::from(((v >> 16) & 0xFF) as u8) / 255.0,
            f32::from(((v >> 24) & 0xFF) as u8) / 255.0,
        ]
    }

    /// `fn sd_rounded_box(p, b, r)` -- shader line 96. Inigo Quilez's formulation,
    /// negative inside.
    pub fn sd_rounded_box(p: [f32; 2], b: [f32; 2], r: f32) -> f32 {
        let q = [p[0].abs() - b[0] + r, p[1].abs() - b[1] + r];
        let m = [q[0].max(0.0), q[1].max(0.0)];
        (m[0] * m[0] + m[1] * m[1]).sqrt() + q[0].max(q[1]).min(0.0) - r
    }

    /// The quad expansion from `vs_main` -- shader line 76. Shapes grow by one pixel so
    /// the antialiased edge has somewhere to live; glyphs do not, because padding would
    /// shear their UV mapping; a glow grows by its whole falloff, because the fragment
    /// stage only runs where the quad reaches and a halo cut off at one pixel is a halo
    /// with a square edge.
    ///
    /// Takes `param` for that last case, which is why this is not a function of `kind`
    /// alone any more. A rim is deliberately not a fourth case: it lives entirely inside its
    /// shape, so the one-pixel margin every other fill gets is exactly what it needs.
    pub fn quad_pad(kind: u32, param: f32) -> f32 {
        if kind == KIND_GLYPH {
            0.0
        } else if kind == KIND_GLOW {
            param.max(0.0) + 1.0
        } else {
            1.0
        }
    }

    /// `fn glow_t(distance, falloff)` -- how far along the falloff a fragment is: 0 at the
    /// shape's edge and inside it, 1 at the limit.
    pub fn glow_t(distance: f32, falloff: f32) -> f32 {
        if falloff <= 0.0 {
            return if distance <= 0.0 { 0.0 } else { 1.0 };
        }
        (distance / falloff).clamp(0.0, 1.0)
    }

    /// `fn rim_t(distance, width)` -- how far *inside* the shape a fragment is: 0 at the
    /// boundary and outside it, 1 at `width` inward.
    ///
    /// The same field as `glow_t`, read with the opposite sign, and answering a zero width
    /// rather than dividing by it for the same reason.
    pub fn rim_t(distance: f32, width: f32) -> f32 {
        if width <= 0.0 {
            return 1.0;
        }
        (-distance / width).clamp(0.0, 1.0)
    }

    /// `fn halo(near, far, t)` -- the tint, mixed in premultiplied linear.
    ///
    /// Not Oklab, and the shader says why at length: a zero-alpha stop has no hue to
    /// unpremultiply, so a perceptual walk fades the halo through black.
    pub fn halo(near: [f32; 4], far: [f32; 4], t: f32) -> [f32; 4] {
        [
            near[0] + (far[0] - near[0]) * t,
            near[1] + (far[1] - near[1]) * t,
            near[2] + (far[2] - near[2]) * t,
            near[3] + (far[3] - near[3]) * t,
        ]
    }

    /// The signed-distance half of `fs_main` -- shader lines 113 to 128.
    ///
    /// `local` is `VsOut::local`: the fragment's position relative to the rect centre, in
    /// pixels. Evaluating in that space is what makes one unit of distance one pixel on
    /// screen, which is what makes the coverage below correct without a derivative.
    /// The two lines of `fs_main` that every non-glyph branch shares: clamp the radius,
    /// then take the signed distance.
    ///
    /// Its own function because the glow needs the distance *twice* -- once for coverage
    /// and once for the tint -- exactly as the shader does, where it is one `let`.
    /// Recomputing it at the second call site would be a place for the two to drift.
    pub fn fs_distance(local: [f32; 2], half_size: [f32; 2], radius: f32) -> f32 {
        // Line 113: an unclamped radius larger than half the shorter side inverts the SDF
        // and renders a bow-tie.
        let radius = radius.clamp(0.0, half_size[0].min(half_size[1]));
        sd_rounded_box(local, half_size, radius)
    }

    pub fn fs_alpha(
        kind: u32,
        local: [f32; 2],
        half_size: [f32; 2],
        radius: f32,
        param: f32,
    ) -> f32 {
        let distance = fs_distance(local, half_size, radius);

        if kind == KIND_STROKE {
            // Lines 120-121: distance to the centre-line of a band of width `param`.
            let half_width = param * 0.5;
            (0.5 - ((distance + half_width).abs() - half_width)).clamp(0.0, 1.0)
        } else if kind == KIND_GLOW {
            // Quadratic, continuous across the shape's boundary: solid inside, falling from
            // the edge outward. There is no edge here to antialias, so no `0.5 - d`.
            let fade = 1.0 - glow_t(distance, param);
            fade * fade
        } else if kind == KIND_PBR {
            // The fill's coverage, unchanged: shading changes what is inside a shape, never
            // which pixels the shape covers. That separation is what makes the floor exact.
            (0.5 - distance).clamp(0.0, 1.0)
        } else if kind == KIND_RIM {
            // Quadratic inward, multiplied by the fill's own coverage rather than replacing
            // it: that factor is what keeps the light inside the shape and antialiases its
            // outer edge, which is why a rim needs no quad padding.
            let fade = 1.0 - rim_t(distance, param);
            (0.5 - distance).clamp(0.0, 1.0) * fade * fade
        } else {
            // Line 127.
            (0.5 - distance).clamp(0.0, 1.0)
        }
    }

    /// `fn unpremultiply(c)` -- the shader's helper, transcribed.
    pub fn unpremultiply(c: [f32; 4]) -> [f32; 3] {
        if c[3] <= 0.0 {
            return [0.0; 3];
        }
        [c[0] / c[3], c[1] / c[3], c[2] / c[3]]
    }

    /// `fn ramp(near, far, local, half_size, angle, pixel)` -- the gradient's colour at one
    /// fragment, premultiplied.
    ///
    /// The Oklab conversions themselves are *not* re-transcribed: they come from
    /// `crate::color`, which is the module the shader's own copy is a transcription of.
    /// Writing a third copy here would mean the suite could only ever catch the shader
    /// disagreeing with this file, not the shader disagreeing with the palette. The dither
    /// comes from the same module for the same reason, and it takes the **framebuffer**
    /// pixel rather than anything derived from `local` -- see `qs_gpu::color::dithered`.
    pub fn ramp(
        near: [f32; 4],
        far: [f32; 4],
        local: [f32; 2],
        half_size: [f32; 2],
        angle: f32,
        pixel: [u32; 2],
    ) -> [f32; 4] {
        ramp_at(near, far, ramp_t(local, half_size, angle), pixel)
    }

    /// `fn ramp_t(local, half_size, angle)` -- how far along a linear ramp a fragment is.
    pub fn ramp_t(local: [f32; 2], half_size: [f32; 2], angle: f32) -> f32 {
        let (sin, cos) = angle.sin_cos();
        let extent = (half_size[0] * cos).abs() + (half_size[1] * sin).abs();
        if extent <= 0.0 {
            return 0.5;
        }
        ((local[0] * cos + local[1] * sin) / extent * 0.5 + 0.5).clamp(0.0, 1.0)
    }

    /// `fn sweep_t(local, half_size, phase)` -- how far *around* the shape a fragment is, on
    /// the same two-stop ramp.
    ///
    /// The mirror is what makes this continuous across the wrap, and the continuity is the
    /// property `a_sweep_has_no_seam_where_the_angle_wraps` measures rather than assumes.
    /// `atan2(0, 0)` is indeterminate in WGSL, so the exact centre is answered explicitly on
    /// both sides.
    pub fn sweep_t(local: [f32; 2], half_size: [f32; 2], phase: f32) -> f32 {
        let nx = local[0] / half_size[0].max(1e-4);
        let ny = local[1] / half_size[1].max(1e-4);
        let angle = if nx == 0.0 && ny == 0.0 {
            0.0
        } else {
            ny.atan2(nx)
        };
        let turns = (angle - phase) / TAU + 0.5;
        let f = turns - turns.floor();
        1.0 - (f * 2.0 - 1.0).abs()
    }

    /// `fn field_weight(delta, reach)` -- how strongly one centre reaches a point.
    ///
    /// Quartic with **bounded support**: zero at the reach and beyond, with zero slope at both
    /// ends. The bounded part is the cost argument the whole primitive rests on, so it is
    /// transcribed rather than approximated -- a tail that merely got small would still be
    /// four evaluations per fragment and would put a faint ring where it was truncated.
    pub fn field_weight(delta: [f32; 2], reach: f32) -> f32 {
        if reach <= 0.0 {
            return 0.0;
        }
        let d2 = delta[0] * delta[0] + delta[1] * delta[1];
        let t = (d2 / (reach * reach)).clamp(0.0, 1.0);
        let falloff = 1.0 - t;
        falloff * falloff
    }

    /// `fn field(base, uv01, aspect, phase, amplitude, pixel)` -- the whole field at one
    /// fragment, composited over the base and dithered.
    ///
    /// The Oklab conversions and the dither come from `crate::color` for the reason `ramp_at`
    /// gives: a third copy here could only ever catch the shader disagreeing with this file,
    /// never with the palette.
    pub fn field(
        base: [f32; 4],
        uv01: [f32; 2],
        aspect: f32,
        phase: f32,
        amplitude: f32,
        centres: &crate::frame::FieldWash,
        pixel: [u32; 2],
    ) -> [f32; 4] {
        let mut total = 0.0_f32;
        let mut lab = [0.0_f32; 3];

        for centre in &centres.centres {
            let angle = phase + centre.phase * TAU;
            let at = [
                centre.at[0] + centre.drift[0] * angle.cos(),
                centre.at[1] + centre.drift[1] * angle.sin(),
            ];
            let delta = [uv01[0] - at[0], (uv01[1] - at[1]) * aspect];
            let tint = centre.tint.to_premul_linear_f32();
            let weight = field_weight(delta, centre.reach) * tint[3];
            total += weight;
            let c = crate::color::linear_rgb_to_oklab(unpremultiply(tint));
            for (slot, channel) in lab.iter_mut().zip(c) {
                *slot += channel * weight;
            }
        }

        if total <= 0.0 {
            return base;
        }

        let mixed =
            crate::color::oklab_to_linear_rgb([lab[0] / total, lab[1] / total, lab[2] / total]);
        let coverage = total.clamp(0.0, 1.0) * amplitude.clamp(0.0, 1.0);
        let mut out = [0.0_f32; 4];
        for ((slot, wash), under) in out.iter_mut().zip(mixed).zip(base) {
            *slot = wash.clamp(0.0, 1.0) * coverage + under * (1.0 - coverage);
        }
        out[3] = coverage + base[3] * (1.0 - coverage);

        let straight = unpremultiply(out);
        let dithered = crate::color::dithered(straight, pixel[0], pixel[1]);
        [
            dithered[0] * out[3],
            dithered[1] * out[3],
            dithered[2] * out[3],
            out[3],
        ]
    }

    /// `fn ramp_at(near, far, t, pixel)` -- the two-stop ramp's colour at one fragment, shared
    /// by both the linear gradient and the conic sweep.
    pub fn ramp_at(near: [f32; 4], far: [f32; 4], t: f32, pixel: [u32; 2]) -> [f32; 4] {
        let from = crate::color::linear_rgb_to_oklab(unpremultiply(near));
        let to = crate::color::linear_rgb_to_oklab(unpremultiply(far));
        let lab = [
            from[0] + (to[0] - from[0]) * t,
            from[1] + (to[1] - from[1]) * t,
            from[2] + (to[2] - from[2]) * t,
        ];
        let alpha = near[3] + (far[3] - near[3]) * t;
        let rgb = crate::color::oklab_to_linear_rgb(lab);
        let rgb = crate::color::dithered(
            [
                rgb[0].clamp(0.0, 1.0),
                rgb[1].clamp(0.0, 1.0),
                rgb[2].clamp(0.0, 1.0),
            ],
            pixel[0],
            pixel[1],
        );
        [rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha, alpha]
    }

    /// `textureSample(atlas_texture, atlas_sampler, uv).r` -- shader line 107.
    ///
    /// The sampler is `FilterMode::Linear` with `AddressMode::ClampToEdge` (see
    /// `batcher.rs`), so this is bilinear, not nearest. That distinction is invisible in
    /// the pipeline as it stands, because glyph quads land on integer pixels and the
    /// bilinear weights collapse to `1, 0, 0, 0` -- but it is exactly why a quad at a
    /// *fractional* origin does not agree between the tiers, and it has to be modelled
    /// honestly for that measurement to mean anything.
    pub fn texture_sample_r(atlas: &[u8], size: u32, uv: [f32; 2]) -> f32 {
        let last = i64::from(size) - 1;
        // Texel centres sit at +0.5, so the filter grid is offset by half a texel.
        let x = uv[0] * size as f32 - 0.5;
        let y = uv[1] * size as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);

        let texel = |ix: f32, iy: f32| -> f32 {
            let cx = (ix as i64).clamp(0, last) as usize;
            let cy = (iy as i64).clamp(0, last) as usize;
            f32::from(atlas[cy * size as usize + cx]) / 255.0
        };

        let top = texel(x0, y0) * (1.0 - fx) + texel(x0 + 1.0, y0) * fx;
        let bottom = texel(x0, y0 + 1.0) * (1.0 - fx) + texel(x0 + 1.0, y0 + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    // -- the lighting pass (T039), transcribed from `shaders/lighting.wgsl` -----------
    //
    // Function for function, constant for constant. If `lighting.wgsl` changes, this must
    // change with it — the same standing rule the header states for `instance.wgsl`.

    use crate::scene::Slab;

    /// `SHADOW_STEPS` — shader constant.
    pub const SHADOW_STEPS: u32 = 12;
    /// `AO_SAMPLES` — shader constant.
    pub const AO_SAMPLES: u32 = 4;
    /// `AO_STRENGTH` — shader constant.
    pub const AO_STRENGTH: f32 = 0.35;
    /// `BOUNCE_REACH` — shader constant. How far bounced light reaches before it has fallen
    /// to a quarter, in physical pixels.
    pub const BOUNCE_REACH: f32 = 90.0;
    /// `FOCUS_REACH` — shader constant. How far the focus lamp's influence reaches before it
    /// has fallen to a quarter, in physical pixels.
    pub const FOCUS_REACH: f32 = 220.0;

    /// The focus lamp as the shader reads it: the uniform's `focus`/`focus_mix` pair.
    ///
    /// A type rather than three loose arguments so a caller cannot supply a position and
    /// forget the share, which would be a lamp that is somewhere and does nothing — and
    /// would silently make every focus-light assertion vacuous.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct FocusLamp {
        /// The focused row's rect, `[x, y, w, h]`, physical pixels. A **strip**, not a point:
        /// a bulb over a long row lights its middle third, the same finding `bounce` records
        /// one light earlier.
        pub rect: [f32; 4],
        /// How high the strip hangs above the canvas, physical pixels.
        pub height: f32,
        /// `1 / tan(size / 2)`, from [`crate::lighting::hardness`].
        pub hardness: f32,
        /// The lamp's share of the shading directly beneath it. Zero is the identity.
        pub share: f32,
        /// How far the room dims at the edge of the lamp's reach. The half a person sees.
        pub ambient: f32,
    }

    /// `fn slab_distance` — the 2D rounded box extruded from `elevation - thickness` up to
    /// `elevation`.
    pub fn slab_distance(q: [f32; 3], slab: &Slab) -> f32 {
        let centre = [
            slab.rect[0] + slab.rect[2] * 0.5,
            slab.rect[1] + slab.rect[3] * 0.5,
        ];
        let half = [slab.rect[2] * 0.5, slab.rect[3] * 0.5];
        let d2 = sd_rounded_box([q[0] - centre[0], q[1] - centre[1]], half, slab.radius);
        let half_thick = (slab.thickness * 0.5).max(0.5);
        let dz = (q[2] - (slab.elevation - half_thick)).abs() - half_thick;
        let outside = [d2.max(0.0), dz.max(0.0)];
        d2.max(dz).min(0.0) + (outside[0] * outside[0] + outside[1] * outside[1]).sqrt()
    }

    /// `fn scene_distance` — the least distance to any slab.
    pub fn scene_distance(q: [f32; 3], slabs: &[Slab]) -> f32 {
        slabs
            .iter()
            .map(|slab| slab_distance(q, slab))
            .fold(1e9, f32::min)
    }

    /// `fn scene_distance_excluding` — the scene with the receiver left out.
    ///
    /// A surface does not shadow itself. Without this a ray to a *place* cannot leave a
    /// large flat surface at all: most of the segment runs nearly parallel to the receiver,
    /// so the receiver's own face is the nearest surface for most of the march and
    /// `k * h / t` reads it as a near-miss occluder. It bites harder the SOFTER the light,
    /// which is why the focus lamp found it and the key light's 5-degree source never did.
    pub fn scene_distance_excluding(q: [f32; 3], slabs: &[Slab], skip: usize) -> f32 {
        slabs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != skip)
            .map(|(_, slab)| slab_distance(q, slab))
            .fold(1e9, f32::min)
    }

    /// `fn soft_shadow` — `min(k * h / t)` along one ray toward the light (research R4).
    pub fn soft_shadow(origin: [f32; 3], toward: [f32; 3], k: f32, slabs: &[Slab]) -> f32 {
        let mut res = 1.0_f32;
        let mut t = 0.35_f32;
        for _ in 0..SHADOW_STEPS {
            let q = [
                origin[0] + toward[0] * t,
                origin[1] + toward[1] * t,
                origin[2] + toward[2] * t,
            ];
            let h = scene_distance(q, slabs);
            res = res.min((k * h / t).clamp(0.0, 1.0));
            t += h.clamp(0.5, 24.0);
            if res < 0.005 || t > 400.0 {
                break;
            }
        }
        res.clamp(0.0, 1.0)
    }

    /// `fn bounce_shadow` — visibility along a segment of known, finite length.
    ///
    /// Fixed steps across the segment rather than [`soft_shadow`]'s distance-driven march:
    /// a bounce ray skims its own receiver, so the scene distance never grows and the
    /// adaptive march spends its whole budget inside ~17 px of a 90 px reach.
    pub fn bounce_shadow(
        origin: [f32; 3],
        toward: [f32; 3],
        d: f32,
        k: f32,
        slabs: &[Slab],
        skip: usize,
    ) -> f32 {
        let mut res = 1.0_f32;
        for i in 1..=SHADOW_STEPS {
            let t = d * i as f32 / (SHADOW_STEPS + 1) as f32;
            let q = [
                origin[0] + toward[0] * t,
                origin[1] + toward[1] * t,
                origin[2] + toward[2] * t,
            ];
            let h = scene_distance_excluding(q, slabs, skip);
            res = res.min((k * h / t).clamp(0.0, 1.0));
            if res < 0.005 {
                break;
            }
        }
        res.clamp(0.0, 1.0)
    }

    /// `fn focus_point` — the nearest point on the lamp's strip, clamped to the focused row's
    /// rect. Directly under the row this is the pixel's own column; past either end it is the
    /// nearer end.
    pub fn focus_point(origin: [f32; 3], lamp: FocusLamp) -> [f32; 3] {
        let [x, y, w, h] = lamp.rect;
        [
            origin[0].clamp(x, x + w),
            origin[1].clamp(y, y + h),
            lamp.height,
        ]
    }

    /// `fn focus_shadow` — visibility along the finite segment from the receiver to the lamp.
    pub fn focus_shadow(origin: [f32; 3], receiver: usize, lamp: FocusLamp, slabs: &[Slab]) -> f32 {
        let point = focus_point(origin, lamp);
        let to_light = [
            point[0] - origin[0],
            point[1] - origin[1],
            point[2] - origin[2],
        ];
        let d = (to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2])
            .sqrt();
        if d < 0.001 {
            return 1.0;
        }
        let toward = [to_light[0] / d, to_light[1] / d, to_light[2] / d];
        bounce_shadow(origin, toward, d, lamp.hardness, slabs, receiver)
    }

    /// `fn focus_weight` — how much of this pixel's shading the lamp owns.
    pub fn focus_weight(origin: [f32; 3], lamp: FocusLamp) -> f32 {
        if lamp.share <= 0.0 {
            return 0.0;
        }
        let point = focus_point(origin, lamp);
        let d = ((point[0] - origin[0]).powi(2)
            + (point[1] - origin[1]).powi(2)
            + (point[2] - origin[2]).powi(2))
        .sqrt();
        let falloff = 1.0 / (1.0 + (d / FOCUS_REACH) * (d / FOCUS_REACH));
        (lamp.share * falloff).clamp(0.0, 1.0)
    }

    /// `fn focus_ambient` — how bright the room is at `origin`: full under the lamp, falling
    /// to `1 - ambient` past its reach.
    ///
    /// The half of the lamp a person actually sees. [`focus_weight`] only changes a pixel
    /// where the two lights disagree, and on a list of rows at one elevation they agree
    /// almost everywhere — measured at a peak of 7/255 across a shipped 1200x700 window with
    /// focus moved eight rows.
    pub fn focus_ambient(origin: [f32; 3], lamp: FocusLamp) -> f32 {
        if lamp.ambient <= 0.0 {
            return 1.0;
        }
        let point = focus_point(origin, lamp);
        let d = ((point[0] - origin[0]).powi(2)
            + (point[1] - origin[1]).powi(2)
            + (point[2] - origin[2]).powi(2))
        .sqrt();
        let falloff = 1.0 / (1.0 + (d / FOCUS_REACH) * (d / FOCUS_REACH));
        (1.0 - lamp.ambient * (1.0 - falloff)).clamp(0.0, 1.0)
    }

    /// `fn occlusion` — bounded samples straight up, weights halving.
    pub fn occlusion(origin: [f32; 3], slabs: &[Slab]) -> f32 {
        let mut occ = 0.0_f32;
        let mut weight = 0.5_f32;
        for i in 1..=AO_SAMPLES {
            let up = i as f32 * 3.0;
            let d = scene_distance([origin[0], origin[1], origin[2] + up], slabs);
            occ += weight * ((up - d) / up).clamp(0.0, 1.0);
            weight *= 0.5;
        }
        occ.clamp(0.0, 1.0)
    }

    /// `fn bounce` — one-bounce light from every emitting slab onto `receiver`.
    ///
    /// The nearest point on the emitter's top face is what the receiver sees, so a full-width
    /// row lights like a strip rather than like a bulb over its centre. Falloff is the
    /// shader's bounded inverse-square, the receiver's upward normal gives the cosine term,
    /// and the shadow ray toward the emitter is what makes the light **occluded** — a
    /// surface hidden behind something taller receives nothing.
    pub fn bounce(origin: [f32; 3], receiver: usize, slabs: &[Slab], k: f32) -> [f32; 3] {
        let mut added = [0.0_f32; 3];
        for (i, slab) in slabs.iter().enumerate() {
            if slab.emission_strength <= 0.0 || i == receiver {
                continue;
            }
            let [x, y, w, h] = slab.rect;
            // Nearest point on the emitter's top face.
            let q = [origin[0].clamp(x, x + w), origin[1].clamp(y, y + h)];
            let to_light = [
                q[0] - origin[0],
                q[1] - origin[1],
                slab.elevation - origin[2],
            ];
            let d = (to_light[0] * to_light[0] + to_light[1] * to_light[1]
                + to_light[2] * to_light[2])
                .sqrt();
            if d > BOUNCE_REACH * 2.0 {
                continue;
            }
            let n_dot_l = (to_light[2] / d.max(0.001)).clamp(0.0, 1.0);
            let falloff = 1.0 / (1.0 + (d / BOUNCE_REACH) * (d / BOUNCE_REACH));
            let toward = [
                to_light[0] / d.max(0.001),
                to_light[1] / d.max(0.001),
                to_light[2] / d.max(0.001),
            ];
            let shade = bounce_shadow(origin, toward, d, k, slabs, receiver);
            let gain = slab.emission_strength * falloff * n_dot_l * shade;
            for (channel, emitted) in added.iter_mut().zip(slab.emission) {
                *channel += emitted * gain;
            }
        }
        added
    }

    /// `fs_lit`'s receiver scan: the topmost slab under the pixel, by rect containment.
    ///
    /// Deliberately ignoring the corner radius, exactly as the shader does. Shared by both
    /// halves of the fragment so the two cannot disagree about which surface is being lit —
    /// a second copy of this scan is how a bounce lands on one slab's allowance while the
    /// attenuation is clamped to another's.
    fn receiver_at(p: [f32; 2], slabs: &[Slab]) -> Option<(usize, f32, f32, f32)> {
        let mut found: Option<(usize, f32, f32, f32)> = None;
        let mut top = -1e9_f32;
        for (i, slab) in slabs.iter().enumerate() {
            let [x, y, w, h] = slab.rect;
            if p[0] >= x && p[0] <= x + w && p[1] >= y && p[1] <= y + h && slab.elevation >= top {
                top = slab.elevation;
                found = Some((i, top, slab.attenuation_floor, slab.addition_max));
            }
        }
        found
    }

    /// `fs_lit`, minus the blend: the attenuation the pass writes for pixel `p`.
    ///
    /// Includes the receiver scan (topmost slab under the pixel, rect containment), the focus
    /// lamp's mix, and the allowance clamp — the line that makes the contrast gate's closed
    /// form a bound.
    ///
    /// `lamp` is `None` when nothing has keyboard focus. A lamp with a zero share produces the
    /// same numbers, and that equality is asserted rather than assumed — see
    /// `a_lamp_with_no_share_is_the_frame_that_shipped_before_it`.
    pub fn lit_attenuation(
        p: [f32; 2],
        slabs: &[Slab],
        toward: [f32; 3],
        k: f32,
        lamp: Option<FocusLamp>,
    ) -> f32 {
        let Some((receiver, top, floor, _)) = receiver_at(p, slabs) else {
            return 1.0;
        };
        let origin = [p[0], p[1], top + 0.5];
        let mut atten = soft_shadow(origin, toward, k, slabs);
        if let Some(lamp) = lamp {
            let w = focus_weight(origin, lamp);
            if w > 0.0 {
                atten = atten + (focus_shadow(origin, receiver, lamp, slabs) - atten) * w;
            }
            atten *= focus_ambient(origin, lamp);
        }
        atten *= 1.0 - AO_STRENGTH * occlusion(origin, slabs);
        atten.clamp(floor.clamp(0.0, 1.0), 1.0)
    }

    /// `fs_lit`'s other half: the light the pass **adds** at pixel `p`, per channel.
    ///
    /// Clamped to the receiver's own `addition_max`, per channel, which is the arithmetic
    /// form of lit-contrast rule 1a: a text ground's allowance is zero, so a filename's
    /// background is untouched however hard the row beside it glows. Per channel rather than
    /// on the magnitude, so a coloured bounce cannot exceed the allowance by arriving as
    /// three components that individually fit and jointly do not.
    pub fn lit_addition(p: [f32; 2], slabs: &[Slab], k: f32) -> [f32; 3] {
        let Some((receiver, top, _, take)) = receiver_at(p, slabs) else {
            return [0.0; 3];
        };
        if take <= 0.0 {
            return [0.0; 3];
        }
        let origin = [p[0], p[1], top + 0.5];
        let added = bounce(origin, receiver, slabs, k);
        [
            added[0].clamp(0.0, take),
            added[1].clamp(0.0, take),
            added[2].clamp(0.0, take),
        ]
    }
}

// -- the reference tier --------------------------------------------------------------

/// A linear premultiplied RGBA surface, which is what the GPU's framebuffer holds before
/// the sRGB transfer function is applied on write. [`CpuRasterizer`]'s pixmap is in the
/// same space, so the comparison happens there rather than after `to_srgb_rgba8` -- a
/// difference of one in a dark linear value becomes a difference of twelve in sRGB, and
/// comparing after the transfer would make every threshold a statement about gamma.
struct Surface {
    pixels: Vec<[f32; 4]>,
    width: u32,
    height: u32,
}

impl Surface {
    fn at(&self, x: u32, y: u32) -> [f32; 4] {
        self.pixels[(y * self.width + x) as usize]
    }
}

/// Run the shader over every pixel of every instance, blending as the pipeline does.
///
/// This is a software GPU for exactly one shader. It reproduces the parts of the fixed
/// function pipeline the shader's output actually depends on -- the quad the vertex stage
/// generates, which fragments that quad covers, the interpolation of `local` and `uv`
/// across it, and the `One / OneMinusSrcAlpha` blend -- and nothing else.
fn render_reference(list: &DrawList, atlas: &[u8], atlas_size: u32) -> Surface {
    let (width, height) = (list.viewport[0], list.viewport[1]);
    let clear = {
        let (r, g, b, a) = (
            crate::color::srgb_to_linear(list.clear.r),
            crate::color::srgb_to_linear(list.clear.g),
            crate::color::srgb_to_linear(list.clear.b),
            list.clear.a,
        );
        // `Pixmap::fill` takes straight colour and premultiplies, which is what
        // `cpu_raster::linear_premul` relies on.
        [r * a, g * a, b * a, a]
    };
    let mut surface = Surface {
        pixels: vec![clear; (width as usize) * (height as usize)],
        width,
        height,
    };

    for batch in &list.batches {
        let range = batch.range.start as usize..batch.range.end as usize;
        for instance in &list.instances[range] {
            draw_reference_instance(
                &mut surface,
                instance,
                atlas,
                atlas_size,
                list.environment,
                list.field,
            );
        }
    }
    surface
}

fn draw_reference_instance(
    surface: &mut Surface,
    instance: &Instance,
    atlas: &[u8],
    atlas_size: u32,
    environment: crate::frame::Environment,
    field: crate::frame::FieldWash,
) {
    // Straight linear, which is what the shader's `unpremultiply` hands `environment`.
    let horizon = shader::unpremultiply(environment.horizon.to_premul_linear_f32());
    let zenith = shader::unpremultiply(environment.zenith.to_premul_linear_f32());
    let [x, y, w, h] = instance.rect;
    // The CPU tier's own early-outs, mirrored so the two tiers agree about *nothing*
    // being drawn as well as about something being drawn.
    if !(w > 0.0 && h > 0.0) {
        return;
    }
    let color = shader::unpack4x8unorm(instance.color);
    // Mirrors `CpuRasterizer::draw_instance`: for a gradient, `color` is only the near
    // stop, so an invisible one says nothing about the far end. A glow's two stops are the
    // same field pair and read the same way.
    let peak_alpha = if instance.kind == shader::KIND_GRADIENT
        || instance.kind == shader::KIND_GLOW
        || instance.kind == shader::KIND_SWEEP
    {
        color[3].max(instance.uv[3])
    } else {
        color[3]
    };
    if peak_alpha <= 0.0 {
        return;
    }

    let half_size = [w * 0.5, h * 0.5];
    let centre = [x + half_size[0], y + half_size[1]];
    let pad = shader::quad_pad(instance.kind, instance.param);

    // Which fragments the vertex stage's quad actually generates. Outside it the shader
    // never runs, so neither does this.
    let x0 = (centre[0] - half_size[0] - pad).floor().max(0.0) as u32;
    let y0 = (centre[1] - half_size[1] - pad).floor().max(0.0) as u32;
    let x1 = (centre[0] + half_size[0] + pad)
        .ceil()
        .min(surface.width as f32) as u32;
    let y1 = (centre[1] + half_size[1] + pad)
        .ceil()
        .min(surface.height as f32) as u32;

    for py in y0..y1 {
        for px in x0..x1 {
            // Fragment shading is at the pixel centre.
            let p = [px as f32 + 0.5, py as f32 + 0.5];
            let local = [p[0] - centre[0], p[1] - centre[1]];
            if local[0].abs() > half_size[0] + pad || local[1].abs() > half_size[1] + pad {
                continue;
            }

            let alpha = if instance.kind == shader::KIND_GLYPH {
                // `out.uv = mix(inst.uv.xy, inst.uv.zw, corner)`, interpolated: the
                // fragment's fraction across the quad picks the same fraction across the
                // atlas sub-rectangle.
                let fx = (p[0] - x) / w;
                let fy = (p[1] - y) / h;
                let uv = [
                    instance.uv[0] + (instance.uv[2] - instance.uv[0]) * fx,
                    instance.uv[1] + (instance.uv[3] - instance.uv[1]) * fy,
                ];
                shader::texture_sample_r(atlas, atlas_size, uv)
            } else {
                shader::fs_alpha(
                    instance.kind,
                    local,
                    half_size,
                    instance.radius,
                    instance.param,
                )
            };

            // Coverage and colour are separable in `fs_main`: a gradient takes the fill's
            // coverage and replaces only the tint being covered.
            let tint = if instance.kind == shader::KIND_GRADIENT {
                shader::ramp(
                    color,
                    instance.uv,
                    local,
                    half_size,
                    instance.param,
                    [px, py],
                )
            } else if instance.kind == shader::KIND_FIELD {
                let uv01 = [
                    local[0] / half_size[0].max(1e-4) * 0.5 + 0.5,
                    local[1] / half_size[1].max(1e-4) * 0.5 + 0.5,
                ];
                let aspect = half_size[1] / half_size[0].max(1e-4);
                shader::field(
                    color,
                    uv01,
                    aspect,
                    instance.param,
                    instance.uv[0],
                    &field,
                    [px, py],
                )
            } else if instance.kind == shader::KIND_SWEEP {
                shader::ramp_at(
                    color,
                    instance.uv,
                    shader::sweep_t(local, half_size, instance.param),
                    [px, py],
                )
            } else if instance.kind == shader::KIND_GLOW {
                let distance = shader::fs_distance(local, half_size, instance.radius);
                shader::halo(color, instance.uv, shader::glow_t(distance, instance.param))
            } else if instance.kind == shader::KIND_PBR {
                let radius = instance.radius.clamp(0.0, half_size[0].min(half_size[1]));
                let distance = shader::fs_distance(local, half_size, instance.radius);
                let grad = shader::sd_rounded_box_grad(local, half_size, radius);
                let normal = shader::bevel_normal(distance, grad, instance.uv[0]);
                let straight = shader::unpremultiply(color);
                let emissive = instance.param
                    * shader::lamp_emission(distance, instance.uv[0], local, half_size);
                let lit = shader::shade_pbr(
                    normal,
                    straight,
                    instance.uv[1],
                    instance.uv[2],
                    instance.uv[3],
                    emissive,
                    horizon,
                    zenith,
                );
                [
                    lit[0] * color[3],
                    lit[1] * color[3],
                    lit[2] * color[3],
                    color[3],
                ]
            } else {
                color
            };

            // `return tint * alpha` under `One / OneMinusSrcAlpha`. The colour is already
            // premultiplied, so scaling the whole vector keeps it that way.
            let src = [
                tint[0] * alpha,
                tint[1] * alpha,
                tint[2] * alpha,
                tint[3] * alpha,
            ];
            let inv = 1.0 - src[3];
            let dst = surface.pixels[(py * surface.width + px) as usize];
            surface.pixels[(py * surface.width + px) as usize] = [
                src[0] + dst[0] * inv,
                src[1] + dst[1] * inv,
                src[2] + dst[2] * inv,
                src[3] + dst[3] * inv,
            ];
        }
    }
}

// -- comparison ----------------------------------------------------------------------

/// Alpha below which a pixel is treated as empty when measuring extent. Deliberately
/// low: the point of the bounding box is to catch an edge that moved, so it must not
/// quietly forgive a faint extra row.
const INK_FLOOR: u8 = 8;

#[derive(Debug, PartialEq, Eq)]
struct Bounds {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

/// What the two tiers disagree about, in the five ways worth knowing.
///
/// Five rather than one because [`CPU_EDGE_QUANTUM`] blunts some of them and not others.
/// Snapping an edge to the nearest quarter pixel moves coverage around, so `max_channel`
/// and `centroid_shift` both have to carry slack for it; but it moves an equal amount off
/// one edge and onto the opposite one, so `total_ink` is untouched by it and stays sharp.
/// A metric that survives the known noise is worth more than four that do not.
#[derive(Debug)]
struct Divergence {
    /// Largest absolute difference in any channel of any pixel, in 0..=255 units.
    max_channel: u8,
    /// Mean absolute channel difference over the whole surface.
    mean_channel: f32,
    cpu_ink: Option<Bounds>,
    gpu_ink: Option<Bounds>,
    /// Distance between the two coverage-weighted centroids, in pixels.
    centroid_shift: f32,
    /// Total alpha, in whole-pixel units: how much ink each tier laid down.
    cpu_ink_area: f32,
    gpu_ink_area: f32,
}

impl Divergence {
    /// Difference in laid-down ink, relative to the larger of the two.
    fn area_error(&self) -> f32 {
        let scale = self.cpu_ink_area.max(self.gpu_ink_area);
        if scale <= 0.0 {
            return 0.0;
        }
        (self.cpu_ink_area - self.gpu_ink_area).abs() / scale
    }
}

fn ink_bounds(sample: impl Fn(u32, u32) -> [u8; 4], width: u32, height: u32) -> Option<Bounds> {
    let mut bounds: Option<Bounds> = None;
    for y in 0..height {
        for x in 0..width {
            if sample(x, y)[3] <= INK_FLOOR {
                continue;
            }
            match &mut bounds {
                None => {
                    bounds = Some(Bounds {
                        x0: x,
                        y0: y,
                        x1: x,
                        y1: y,
                    });
                }
                Some(b) => {
                    b.x0 = b.x0.min(x);
                    b.y0 = b.y0.min(y);
                    b.x1 = b.x1.max(x);
                    b.y1 = b.y1.max(y);
                }
            }
        }
    }
    bounds
}

/// Coverage-weighted centroid, and the total coverage it was weighted by (in whole-pixel
/// units). `None` when there is no ink to weigh.
///
/// Two measurements from one pass because they are complementary. The centroid moves
/// continuously with the shape, so it has sub-pixel resolution a whole-pixel bounding box
/// cannot reach -- but a thin ring levers a displacement of its *hole* by the ratio of
/// hole area to ring area, so on strokes it is sensitive to noise as well as to bugs. The
/// total is the opposite: blind to where the ink is, but immune to an edge sliding one
/// way while the opposite edge slides with it.
fn coverage(
    sample: impl Fn(u32, u32) -> [u8; 4],
    width: u32,
    height: u32,
) -> Option<((f32, f32), f32)> {
    let (mut sx, mut sy, mut total) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..height {
        for x in 0..width {
            let a = f64::from(sample(x, y)[3]);
            sx += f64::from(x) * a;
            sy += f64::from(y) * a;
            total += a;
        }
    }
    if total <= 0.0 {
        return None;
    }
    Some((
        ((sx / total) as f32, (sy / total) as f32),
        (total / 255.0) as f32,
    ))
}

fn quantise(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn compare(cpu: &Pixmap, gpu: &Surface) -> Divergence {
    let (width, height) = (cpu.width(), cpu.height());
    assert_eq!((width, height), (gpu.width, gpu.height));

    let cpu_at = |x: u32, y: u32| -> [u8; 4] {
        let p = cpu.pixels()[(y * width + x) as usize];
        [p.red(), p.green(), p.blue(), p.alpha()]
    };
    let gpu_at = |x: u32, y: u32| -> [u8; 4] {
        let p = gpu.at(x, y);
        [
            quantise(p[0]),
            quantise(p[1]),
            quantise(p[2]),
            quantise(p[3]),
        ]
    };

    let mut max_channel = 0u8;
    let mut total = 0u64;
    for y in 0..height {
        for x in 0..width {
            let (a, b) = (cpu_at(x, y), gpu_at(x, y));
            for c in 0..4 {
                let d = a[c].abs_diff(b[c]);
                max_channel = max_channel.max(d);
                total += u64::from(d);
            }
        }
    }

    let cpu_c = coverage(cpu_at, width, height);
    let gpu_c = coverage(gpu_at, width, height);
    let centroid_shift = match (cpu_c, gpu_c) {
        (Some((a, _)), Some((b, _))) => ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt(),
        // One tier drew nothing and the other did. Report it as an enormous shift so the
        // assertion fails rather than silently skipping the sharpest check it has.
        (None, Some(_)) | (Some(_), None) => f32::INFINITY,
        (None, None) => 0.0,
    };

    Divergence {
        max_channel,
        mean_channel: total as f32 / (width * height * 4) as f32,
        cpu_ink: ink_bounds(cpu_at, width, height),
        gpu_ink: ink_bounds(gpu_at, width, height),
        centroid_shift,
        cpu_ink_area: cpu_c.map_or(0.0, |c| c.1),
        gpu_ink_area: gpu_c.map_or(0.0, |c| c.1),
    }
}

// -- the fixtures --------------------------------------------------------------------

const SURFACE: u32 = 64;
const ATLAS: u32 = 32;

/// How finely the CPU tier can place an edge, in pixels.
///
/// `tiny-skia`'s antialiased fill resolves an edge to the nearest quarter pixel: a shape
/// whose left edge is at 12.375 rasterizes as though it were at 12.5. The shader's
/// distance field has no such step. Measured, not assumed -- see
/// `the_cpu_tier_resolves_an_edge_to_a_quarter_of_a_pixel`, which fails if `tiny-skia`
/// ever gets finer, because at that point every bound derived from this number is looser
/// than it needs to be and the suite has quietly stopped being sharp.
const CPU_EDGE_QUANTUM: f32 = 0.25;

/// The most an edge can be misplaced by that quantisation: half a step.
const MAX_EDGE_ERROR: f32 = CPU_EDGE_QUANTUM * 0.5;

/// The most a *single pixel* can therefore differ by, in 0..=255 units.
///
/// A pixel straddling a misplaced edge loses or gains exactly the coverage the edge
/// swept over it, so the worst case is [`MAX_EDGE_ERROR`] of coverage: 32 levels. Any
/// case tolerating more than this is tolerating something other than the quantisation,
/// and has to say what.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const QUANTISATION_CHANNEL_BUDGET: u8 = (MAX_EDGE_ERROR * 255.0) as u8;

/// One primitive drawn one way, plus the bounds its two tiers are allowed to differ by.
///
/// Every bound is measured and then justified in the fixture's comment. A bound with no
/// stated source is a bound someone raised until the test went green.
struct Case {
    name: &'static str,
    instance: Instance,
    /// Atlas content the case needs. Empty for everything but glyphs.
    uploads: Vec<PendingUpload>,
    /// Largest tolerated single-channel difference, in 0..=255 units.
    max_channel: u8,
    /// Largest tolerated mean channel difference over the surface.
    mean_channel: f32,
    /// Largest tolerated centroid drift, in pixels.
    centroid_shift: f32,
    /// Largest tolerated difference in total laid-down ink, as a fraction. This is the
    /// bound that stays tight even where the edge quantisation forces the others open.
    ink_area: f32,
}

/// A blob of coverage standing in for a glyph.
///
/// Deliberately not uniform: a solid block would agree between a bilinear sample and a
/// nearest blit no matter how badly the addressing was wrong, because every texel it
/// could possibly read holds the same value. The gradient plus the notch means a
/// half-texel addressing error changes the answer.
fn glyph_coverage(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let notch = x * 3 < w && y * 3 < h;
            let v = if notch {
                0
            } else {
                ((x * 37 + y * 61) % 256) as u8
            };
            out.push(v);
        }
    }
    out
}

/// Normalised atlas coordinates for a sub-rectangle of the atlas.
fn uv_for(x: u32, y: u32, w: u32, h: u32) -> [f32; 4] {
    let s = ATLAS as f32;
    [
        x as f32 / s,
        y as f32 / s,
        (x + w) as f32 / s,
        (y + h) as f32 / s,
    ]
}

/// The parity fixtures for one primitive.
///
/// **This match is the guard.** It is exhaustive over [`PrimKind`], so a new primitive
/// cannot compile until someone writes down how to check it, and
/// `no_prim_kind_can_ship_without_a_parity_test` refuses an empty list -- a stub arm
/// returning nothing fails the suite rather than passing it.
fn cases_for(kind: PrimKind) -> Vec<Case> {
    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    // Measured on the quarter-aligned rounded fixtures, where the edge quantisation
    // contributes nothing and the only thing left is the difference between an analytic
    // area and `0.5 - d`. Corner pixels are where the two part company.
    const CURVATURE: u8 = 24;
    // The two budgets are independent -- one is a rasterizer's placement step, the other
    // is a definitional difference about coverage -- so a fixture that suffers both is
    // allowed their sum, and nothing more.
    const ROUNDED_AND_OFF_GRID: u8 = QUANTISATION_CHANNEL_BUDGET + CURVATURE;

    match kind {
        PrimKind::Rect => vec![
            Case {
                // The baseline. Axis-aligned, integer bounds, no curvature: the analytic
                // area and the distance field are answering the same question, and the
                // edges are already on the quarter grid. Nothing may differ here at all,
                // and nothing does -- this fixture measures 0.
                name: "rect/sharp",
                instance: Instance::rect(16.0, 20.0, 32.0, 24.0, 0.0, white),
                uploads: Vec::new(),
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.001,
            },
            Case {
                // Curvature is where the two definitions of coverage part company: 22 of
                // the 24 allowed, all of it in the corner pixels.
                name: "rect/rounded",
                instance: Instance::rect(12.0, 10.0, 40.0, 28.0, 6.0, white),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.10,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
            Case {
                // 125% display scale. Every coordinate lands off the integer grid, though
                // still on quarters, so this one is bounded by curvature alone -- which is
                // itself the finding: a fill is only safe off-grid because 125% happens to
                // land on the CPU tier's quarter-pixel step.
                name: "rect/fractional-scale",
                instance: Instance::rect(15.0, 12.5, 37.5, 27.5, 7.5, white),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.12,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
            Case {
                // 150% display scale, where a fill does *not* land on quarters: the right
                // edge falls on an eighth and gets snapped. Bounded by the sum.
                // Displacing one edge of a `w` by `wide` fill by `d` moves the centroid by
                // at most `d / 2`, so an eighth of a pixel at the edge is a sixteenth at
                // the centroid. Measured 0.022 against that 0.0625 ceiling.
                name: "rect/eighth-grid",
                instance: Instance::rect(9.375, 11.0, 30.375, 24.0, 9.0, white),
                uploads: Vec::new(),
                max_channel: ROUNDED_AND_OFF_GRID,
                mean_channel: 0.20,
                centroid_shift: MAX_EDGE_ERROR * 0.5,
                ink_area: 0.006,
            },
            Case {
                // A radius larger than the shape can hold. Both tiers clamp -- the shader
                // to `min(half_size)`, `rounded_rect` to half the shorter side, which are
                // the same number -- and both produce a stadium rather than the bow-tie an
                // unclamped SDF would invert into.
                //
                // This fixture exists because mutation testing found the suite blind
                // without it: deleting the clamp from the transcription changed no
                // measured value anywhere, since nothing asked for a radius big enough to
                // clamp. A guard nothing exercises is a guard nobody is checking.
                // The bounds here are looser than `CURVATURE` and stated rather than
                // shared, because a clamped radius makes the shape a stadium -- arc all
                // the way round, where the radius-6 fixtures are arc at four short
                // corners. Measured 26 and 0.005. They are also not what this fixture
                // rests on: if either tier failed to clamp, the SDF would invert into a
                // bow-tie and the *bounding box* would change, which no tolerance forgives.
                name: "rect/over-large-radius",
                instance: Instance::rect(16.0, 20.0, 32.0, 24.0, 1000.0, white),
                uploads: Vec::new(),
                max_channel: 32,
                mean_channel: 0.25,
                centroid_shift: 0.01,
                ink_area: 0.008,
            },
            Case {
                // Translucent and coloured, to exercise the premultiplied blend rather
                // than just the coverage. The CPU tier un-premultiplies for tiny-skia and
                // tiny-skia premultiplies again in u8, so a couple of levels of round-trip
                // loss are real and expected on top of the curvature.
                name: "rect/translucent",
                instance: Instance::rect(8.0, 8.0, 30.0, 30.0, 4.0, Srgba::new(0.2, 0.6, 0.9, 0.5)),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.05,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
        ],
        PrimKind::Stroke => vec![
            Case {
                // The case that started the chunk: before the CPU tier inset its path,
                // this ring began one pixel outside the rect. Still exact.
                name: "stroke/sharp",
                instance: Instance::stroke(10.0, 10.0, 20.0, 20.0, 0.0, 2.0, white),
                uploads: Vec::new(),
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
            Case {
                // `ink_area` is looser on a ring than on a fill and it is not the
                // quantisation doing it: the corner arcs' curvature error is an absolute
                // number of square pixels, and a 1.5px ring has an eighth the area of the
                // fill it surrounds, so the same error is eight times the fraction.
                // Measured 0.0085 of 144 square pixels, which is 1.2 of them.
                name: "stroke/rounded",
                instance: Instance::stroke(8.0, 8.0, 32.0, 24.0, 6.0, 1.5, white),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.25,
                centroid_shift: 0.01,
                ink_area: 0.010,
            },
            Case {
                // 125% scale: a 1.5px stroke becomes 1.875px, which is the width the icon
                // set actually draws with and the one nobody has looked at. The ring's
                // inner contour lands on an eighth and snaps, and this is the fixture
                // where the centroid has to give ground: displacing a *hole* moves the
                // ring's centroid by the ratio of hole area to ring area, which here is
                // about 610 to 220 -- so an eighth of a pixel at the edge becomes a third
                // of a pixel at the centroid. The bound is that lever arm, not a fudge,
                // and `ink_area` stays tight underneath it because the quantisation takes
                // from one edge exactly what it gives the other.
                name: "stroke/fractional-scale",
                instance: Instance::stroke(10.5, 9.5, 33.75, 25.0, 7.5, 1.875, white),
                uploads: Vec::new(),
                max_channel: ROUNDED_AND_OFF_GRID,
                mean_channel: 1.00,
                centroid_shift: 0.40,
                ink_area: 0.010,
            },
            Case {
                // The clamp again, on the harder side: the two tiers reach it by different
                // routes. The shader clamps the *outer* radius and then offsets the
                // distance field inward; the CPU tier insets the path first and clamps the
                // *reduced* radius against the *reduced* rectangle. Those are only the
                // same number because insetting a rounded rect by `t` reduces its radius by
                // exactly `t`, which is an identity worth having a fixture for rather than
                // a comment.
                // Loosest bounds in the file, and the reason is geometric rather than
                // convenient: this ring has *two* fully-curved boundaries, and the inner
                // one is tighter than the outer, so it carries the stadium's curvature
                // error twice over on an eighth of the ink. Measured 49 and 0.021. As
                // above, the clamp itself is caught by the bounding box, not by these.
                name: "stroke/over-large-radius",
                instance: Instance::stroke(12.0, 14.0, 32.0, 24.0, 1000.0, 2.0, white),
                uploads: Vec::new(),
                max_channel: 56,
                mean_channel: 0.65,
                centroid_shift: 0.01,
                ink_area: 0.025,
            },
        ],
        PrimKind::Gradient => {
            // A wide hue interval on purpose. Two stops of the same hue would agree
            // between any two interpolation spaces, so a fixture built from them would be
            // green whether or not the CPU tier walked Oklab at all -- and walking Oklab
            // is the entire reason this primitive is not a two-instance crossfade.
            // `the_cpu_tier_walks_the_ramp_in_oklab_and_not_in_linear_srgb` is what makes
            // that non-vacuous; these measure how *well* it walks it.
            let steel = Srgba::new(0.145, 0.176, 0.278, 1.0);
            let cyan = Srgba::new(0.220, 0.792, 0.882, 1.0);
            let clear_cyan = Srgba::new(0.220, 0.792, 0.882, 0.0);
            vec![
                Case {
                    // The ramp alone: no curvature, no off-grid edge, axis on a cardinal
                    // direction. It measures **zero** -- the two tiers produce the same
                    // bytes -- which is what licenses treating the bounds below as
                    // curvature rather than as the gradient.
                    //
                    // It used to be 1, and the 1 was the CPU tier approximating a
                    // continuous Oklab curve with 33 chords. `prim-noise-dither` deleted
                    // that approximation, because a dither has to land on the same pixel
                    // on both tiers and a stop list has nowhere to put one. Tightened here
                    // rather than left at 1: a bound with slack nobody is using is a bound
                    // that stops reporting the next regression.
                    name: "gradient/vertical-sharp",
                    instance: Instance::gradient(
                        16.0,
                        12.0,
                        32.0,
                        40.0,
                        0.0,
                        std::f32::consts::FRAC_PI_2,
                        steel,
                        cyan,
                    ),
                    uploads: Vec::new(),
                    max_channel: 0,
                    mean_channel: 0.0001,
                    centroid_shift: 0.01,
                    ink_area: 0.001,
                },
                Case {
                    // The axis alone. A diagonal ramp normalizes t against the box's
                    // support rather than its half-width, and this is the fixture that
                    // would catch the two tiers using different normalizations: they
                    // would put the stops in different places and every interior pixel
                    // would shift. Still zero, so the axis costs nothing either.
                    name: "gradient/diagonal-sharp",
                    instance: Instance::gradient(
                        12.0,
                        16.0,
                        40.0,
                        32.0,
                        0.0,
                        std::f32::consts::FRAC_PI_4,
                        steel,
                        cyan,
                    ),
                    uploads: Vec::new(),
                    max_channel: 0,
                    mean_channel: 0.0001,
                    centroid_shift: 0.01,
                    ink_area: 0.001,
                },
                Case {
                    // Curvature, on the same footing as `rect/rounded`: a gradient is a
                    // rect fill with a varying tint, so it inherits that fixture's
                    // definitional difference between an analytic area and `0.5 - d`, and
                    // nothing more. Measured 22 of the 24 allowed -- the same 22.
                    name: "gradient/diagonal-rounded",
                    instance: Instance::gradient(
                        10.0,
                        10.0,
                        40.0,
                        32.0,
                        6.0,
                        std::f32::consts::FRAC_PI_4,
                        steel,
                        cyan,
                    ),
                    uploads: Vec::new(),
                    max_channel: CURVATURE,
                    mean_channel: 0.10,
                    centroid_shift: 0.01,
                    ink_area: 0.002,
                },
                Case {
                    // A stop that fades to nothing, sharp-edged so the measurement is
                    // about alpha and not about corners. This used to be the case where the
                    // two tiers could genuinely part company for a reason the others could
                    // not see: the shader interpolates straight colour and re-premultiplies,
                    // while `tiny-skia` interpolated between stops it had already
                    // premultiplied, and the two orders differ. It measured 1 across a broad
                    // region -- a whole area a least-significant bit out, rather than a few
                    // corner pixels out by twenty -- which is why `max_channel` and not
                    // `mean_channel` was the bound carrying it.
                    //
                    // Evaluating the ramp per pixel removed the second order entirely, so
                    // there is now nothing here for a bound to hold: both sides do the same
                    // arithmetic in the same sequence and produce the same bytes.
                    name: "gradient/fade-out-sharp",
                    instance: Instance::gradient(
                        12.0, 16.0, 36.0, 28.0, 0.0, 0.0, cyan, clear_cyan,
                    ),
                    uploads: Vec::new(),
                    max_channel: 0,
                    mean_channel: 0.0001,
                    centroid_shift: 0.02,
                    ink_area: 0.002,
                },
                Case {
                    // Both at once, which is the shape a real wash takes. Bounded by
                    // curvature, as the sum of the two isolations predicts.
                    name: "gradient/fade-out-rounded",
                    instance: Instance::gradient(
                        12.0, 16.0, 36.0, 28.0, 4.0, 0.0, cyan, clear_cyan,
                    ),
                    uploads: Vec::new(),
                    max_channel: CURVATURE,
                    mean_channel: 0.10,
                    centroid_shift: 0.02,
                    ink_area: 0.002,
                },
            ]
        }
        PrimKind::Glyph => vec![
            Case {
                // An integer-aligned quad, which is the only kind the pipeline emits: the
                // fractional pen position is baked into the subpixel variant instead. The
                // bilinear weights collapse to nearest and there is no edge to quantise,
                // so this is exact -- it measures 0, not 1.
                name: "glyph/integer-origin",
                instance: Instance::glyph(10.0, 12.0, 9.0, 11.0, uv_for(3, 5, 9, 11), white),
                uploads: vec![PendingUpload {
                    x: 3,
                    y: 5,
                    width: 9,
                    height: 11,
                    coverage: glyph_coverage(9, 11),
                }],
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.001,
            },
            Case {
                // The same glyph at 125%: the atlas entry is re-rasterized larger, so the
                // quad grows to 11x14 and still lands on integers. This is what "glyph at
                // a non-integer scale" means in the shipped pipeline -- the scale is
                // fractional, the geometry that results is not. The case where it is not
                // is pinned separately, by
                // `a_glyph_quad_at_a_fractional_origin_does_not_agree_across_tiers`.
                name: "glyph/fractional-scale",
                instance: Instance::glyph(14.0, 7.0, 11.0, 14.0, uv_for(6, 9, 11, 14), white),
                uploads: vec![PendingUpload {
                    x: 6,
                    y: 9,
                    width: 11,
                    height: 14,
                    coverage: glyph_coverage(11, 14),
                }],
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.001,
            },
        ],
        // A `Fidelity::Enhanced` kind, so `run_kind` asks these fixtures the floor
        // question rather than the parity one: does the CPU tier draw *nothing*. The four
        // tolerance fields below are therefore unread, and they are set to the tightest
        // values in the file rather than to something arbitrary -- if the fidelity
        // declaration is ever relaxed to `Exact`, these become live bounds that fail loudly
        // instead of a set of numbers nobody chose quietly passing.
        //
        // What the floor check cannot see is whether the *GPU* side draws anything at all:
        // a shader branch that returned zero would satisfy every assertion here. That is
        // what `a_glow_reaches_beyond_its_shape_and_fades_to_nothing` and the two tests
        // beside it are for.
        PrimKind::Glow => vec![
            Case {
                name: "glow/rounded",
                instance: Instance::glow(20.0, 22.0, 24.0, 20.0, 6.0, 8.0, white),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
            Case {
                // Two-tone, and translucent at both ends: the form a material will reach
                // for, and the one where a floor of `Nothing` could be satisfied by
                // accident if the rasterizer merely rounded a faint halo away. It does not
                // -- `cpu_floor` drops the instance before `tiny-skia` is reached -- but
                // the fixture is what says so.
                name: "glow/two-tone",
                instance: Instance::glow_two_tone(
                    14.0,
                    18.0,
                    36.0,
                    28.0,
                    10.0,
                    12.0,
                    Srgba::new(0.35, 0.72, 1.0, 0.55),
                    Srgba::new(0.60, 0.30, 1.0, 0.10),
                ),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
        ],
        // The third `Enhanced` kind and the first floored at a primitive rather than at
        // absence: unlit, a surface is its albedo. The tolerance fields are unread while the
        // declaration says `Enhanced`, and set tightest-in-file for the reason the glow's are.
        PrimKind::Pbr => vec![
            Case {
                name: "pbr/dielectric",
                instance: Instance::pbr(
                    20.0, 22.0, 24.0, 20.0, 6.0, 5.0, 0.35, 0.0, 1.0, 0.0, white,
                ),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
            Case {
                // Metal, and translucent: the form where a floor of `Plain(Rect)` could be
                // satisfied by accident if `cpu_floor` let `uv` through -- a bevel read as a
                // stroke width, or a roughness read as an angle, would still draw something.
                name: "pbr/metal-translucent",
                instance: Instance::pbr(
                    14.0,
                    18.0,
                    36.0,
                    28.0,
                    10.0,
                    8.0,
                    0.18,
                    1.0,
                    0.8,
                    0.0,
                    Srgba::new(0.90, 0.72, 0.36, 0.6),
                ),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
        ],
        // Enhanced too, and floored at `Nothing` for a different reason than the glow --
        // `PrimKind::fidelity` records it. The same note applies about the four tolerance
        // fields: unread while the declaration says `Enhanced`, tightest-in-file so that
        // relaxing it to `Exact` fails loudly rather than passing on numbers nobody chose.
        //
        // And the same hole applies: nothing here can see whether the GPU side draws
        // anything at all. `a_rim_is_brightest_inside_its_own_edge_and_gone_by_its_width`
        // and the two tests beside it are that half.
        PrimKind::Rim => vec![
            Case {
                name: "rim/rounded",
                instance: Instance::rim(20.0, 22.0, 24.0, 20.0, 6.0, RIM_WIDTH, white),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
            Case {
                // Translucent, and wide enough that the ramp reaches well into the shape:
                // the form a material actually authors, and the one where a floor of
                // `Nothing` could be satisfied by accident if the rasterizer merely rounded
                // a faint edge light away. It does not -- `cpu_floor` drops the instance
                // before `tiny-skia` is reached -- but the fixture is what says so.
                name: "rim/wide-translucent",
                instance: Instance::rim(
                    14.0,
                    18.0,
                    36.0,
                    28.0,
                    10.0,
                    9.0,
                    Srgba::new(0.85, 0.92, 1.0, 0.30),
                ),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
        ],
        // `Exact`, and the only primitive on this list that looks enhanced and is not.
        // The bounds are the gradient's, because a sweep *is* the gradient with a different
        // parameter: the same two stops, the same Oklab walk, the same dither, the same
        // `0.5 - d` coverage. Both tiers run the same arithmetic in the same order, so the
        // sharp cases measure zero and the rounded one is bounded by curvature alone --
        // exactly as `gradient/diagonal-rounded` is, and for the same reason.
        //
        // What no fixture here can see is the seam, because the seam is a property of the
        // *parameter* and both sides compute it identically: two tiers agreeing on a hue step
        // would measure zero. `a_sweep_has_no_seam_where_the_angle_wraps` is that half.
        PrimKind::Sweep => {
            // The same wide hue interval the gradient fixtures use, and for the same reason:
            // two stops of one hue would agree between any two interpolation spaces.
            let steel = Srgba::new(0.145, 0.176, 0.278, 1.0);
            let cyan = Srgba::new(0.220, 0.792, 0.882, 1.0);
            let clear_cyan = Srgba::new(0.220, 0.792, 0.882, 0.0);
            vec![
                Case {
                    // Square and sharp-edged, phase at zero: the angular parameter alone,
                    // with no curvature and no aspect normalization in play.
                    name: "sweep/square-sharp",
                    instance: Instance::sweep(16.0, 16.0, 32.0, 32.0, 0.0, 0.0, steel, cyan),
                    uploads: Vec::new(),
                    max_channel: 0,
                    mean_channel: 0.0001,
                    centroid_shift: 0.01,
                    ink_area: 0.001,
                },
                Case {
                    // Wide, and phased a third of a turn off. The aspect is what makes this
                    // fixture worth having: the angle is taken in the shape's normalized
                    // space, so a tier that forgot to divide by `half_size` would bunch the
                    // whole ramp onto the two short sides and every interior pixel would move.
                    name: "sweep/wide-phased",
                    instance: Instance::sweep(
                        8.0,
                        20.0,
                        48.0,
                        24.0,
                        0.0,
                        std::f32::consts::TAU / 3.0,
                        steel,
                        cyan,
                    ),
                    uploads: Vec::new(),
                    max_channel: 0,
                    mean_channel: 0.0001,
                    centroid_shift: 0.01,
                    ink_area: 0.001,
                },
                Case {
                    // Curvature, on the same footing as `gradient/diagonal-rounded`: a sweep
                    // is a rect fill with a varying tint and inherits that fixture's
                    // definitional difference between an analytic area and `0.5 - d`.
                    //
                    // Deliberately the *same* geometry as that fixture, down to the radius.
                    // `CURVATURE` is documented as measured on the quarter-aligned rounded
                    // fixtures, so it is a bound for that geometry class and not a general
                    // allowance; a fatter radius measured 25 against it here, which is a
                    // fixture outside the class rather than a regression, and would have
                    // needed its own derived bound rather than a widened shared one.
                    name: "sweep/rounded",
                    instance: Instance::sweep(
                        10.0,
                        10.0,
                        40.0,
                        32.0,
                        6.0,
                        std::f32::consts::FRAC_PI_4,
                        steel,
                        cyan,
                    ),
                    uploads: Vec::new(),
                    max_channel: CURVATURE,
                    mean_channel: 0.10,
                    centroid_shift: 0.01,
                    ink_area: 0.002,
                },
                Case {
                    // A stop that fades to nothing, which is the form a travelling highlight
                    // over an existing surface actually takes: the ramp has to be walked in
                    // straight colour and re-premultiplied on both tiers, or the fade drags
                    // its hue toward black on one of them.
                    name: "sweep/fade-out-sharp",
                    instance: Instance::sweep(14.0, 18.0, 36.0, 28.0, 0.0, 0.0, cyan, clear_cyan),
                    uploads: Vec::new(),
                    max_channel: 0,
                    mean_channel: 0.0001,
                    centroid_shift: 0.02,
                    ink_area: 0.002,
                },
            ]
        }
        // `Enhanced`, floored at the base colour. The four tolerance fields are unread while
        // the declaration says `Enhanced` -- `run_kind` asks `assert_floor` instead -- and are
        // set tightest-in-file for the reason the glow's are: relaxing the declaration to
        // `Exact` should fail loudly rather than pass on numbers nobody chose.
        //
        // The floor question is the sharp one for this primitive, and sharper than it was for
        // the halo. A field converges to its base wherever no centre reaches, so a rasterizer
        // that drew the *base* and called it the field would satisfy a careless check. What
        // `assert_floor` actually holds is that the CPU tier draws the floor **exactly**, and
        // the fixtures below put centres right in the middle of the shape so that "exactly the
        // base" is a claim with something to be wrong about.
        //
        // What no fixture here can see is whether the GPU side draws a field at all, since
        // both sides of a floor check are the same rasterizer. The four tests below the
        // fixtures are that half.
        PrimKind::Field => vec![
            Case {
                name: "field/full-viewport",
                instance: Instance::field(0.0, 0.0, 64.0, 64.0, 0.6, 0.0, white),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
            Case {
                // Off-square and phased, which is the form a real window takes: the aspect
                // correction and the drift both have to end up in the floor's *absence*
                // rather than in a differently-wrong rectangle.
                name: "field/wide-phased",
                instance: Instance::field(
                    6.0,
                    10.0,
                    52.0,
                    28.0,
                    1.0,
                    std::f32::consts::FRAC_PI_3,
                    Srgba::new(0.10, 0.11, 0.14, 1.0),
                ),
                uploads: Vec::new(),
                max_channel: 0,
                mean_channel: 0.0,
                centroid_shift: 0.0,
                ink_area: 0.0,
            },
        ],
    }
}

fn run_case(case: &Case) -> Divergence {
    let mut cpu = CpuRasterizer::new(SURFACE, SURFACE, ATLAS).unwrap();
    cpu.upload_glyphs(&case.uploads);

    // The reference's atlas mirror, filled by the same uploads. Both tiers read the same
    // bytes at the same coordinates, which is what makes the comparison about
    // rasterization rather than about atlas packing.
    let mut atlas = vec![0u8; (ATLAS as usize) * (ATLAS as usize)];
    for upload in &case.uploads {
        for row in 0..upload.height {
            let src = (row as usize) * (upload.width as usize);
            let dst = ((upload.y + row) as usize) * (ATLAS as usize) + upload.x as usize;
            atlas[dst..dst + upload.width as usize]
                .copy_from_slice(&upload.coverage[src..src + upload.width as usize]);
        }
    }

    let mut list = DrawList::default();
    // Transparent clear: with nothing behind it, every non-zero pixel is the primitive's
    // own contribution and a difference cannot be diluted by a shared background.
    list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
    list.instances.push(case.instance);
    // No scissor. `cpu_raster::clip_mask` documents that the CPU tier does not implement
    // one, so a scissored fixture would be measuring that known gap instead of parity.
    list.end_batch(None, !case.uploads.is_empty());

    let reference = render_reference(&list, &atlas, ATLAS);
    let pixmap = cpu.render(&list);
    compare(pixmap, &reference)
}

#[track_caller]
fn assert_parity(case: &Case) {
    let d = run_case(case);
    // Printed unconditionally: when a threshold fails, the first question is always "by
    // how much, and in which of the four ways", and cargo hides this on success anyway.
    eprintln!(
        "{}: max={} mean={:.4} centroid={:.4} area={:.5} ({:.3} vs {:.3}) ink={:?}",
        case.name,
        d.max_channel,
        d.mean_channel,
        d.centroid_shift,
        d.area_error(),
        d.cpu_ink_area,
        d.gpu_ink_area,
        d.cpu_ink
    );

    assert!(
        d.area_error() <= case.ink_area,
        "{}: the tiers laid down different amounts of ink -- {:.4} against {:.4}, a \
         relative difference of {:.5} over the bound of {:.5}. This is the one measure \
         the CPU tier's quarter-pixel edge step cannot move, so a failure here is a real \
         difference in how much of the primitive got drawn",
        case.name,
        d.cpu_ink_area,
        d.gpu_ink_area,
        d.area_error(),
        case.ink_area
    );
    assert_eq!(
        d.cpu_ink, d.gpu_ink,
        "{}: the two tiers disagree about where the primitive is, not merely about how it \
         is shaded -- an edge moved",
        case.name
    );
    assert!(
        d.centroid_shift <= case.centroid_shift,
        "{}: coverage centroid drifted {:.4}px, over the {:.4}px bound",
        case.name,
        d.centroid_shift,
        case.centroid_shift
    );
    assert!(
        d.max_channel <= case.max_channel,
        "{}: worst channel differs by {}, over the bound of {}",
        case.name,
        d.max_channel,
        case.max_channel
    );
    assert!(
        d.mean_channel <= case.mean_channel,
        "{}: mean channel difference {:.4}, over the bound of {:.4}",
        case.name,
        d.mean_channel,
        case.mean_channel
    );
}

// -- the floor comparison ---------------------------------------------------------------

/// The CPU tier's pixels for a draw list, as raw linear premultiplied bytes.
///
/// Bytes rather than a [`Divergence`]: the floor comparison has no tolerance to spend, so
/// there is nothing to measure. Both sides come out of the same rasterizer, and the only
/// interesting answer is whether they are the same.
fn cpu_pixels(instances: &[Instance], uploads: &[PendingUpload]) -> Vec<u8> {
    let mut cpu = CpuRasterizer::new(SURFACE, SURFACE, ATLAS).unwrap();
    cpu.upload_glyphs(uploads);

    let mut list = DrawList::default();
    list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
    list.instances.extend_from_slice(instances);
    list.end_batch(None, !uploads.is_empty());
    cpu.render(&list).data().to_vec()
}

/// Whether the CPU tier drew `instance` as `floor` says it should have.
///
/// The rewrite below restates [`Instance::cpu_floor`]'s rule rather than calling it, for the
/// same reason the `shader` module above restates `instance.wgsl` rather than linking
/// against it: a check that calls the thing it is checking is a check that agrees with any
/// answer. Stated here, a floor that quietly started carrying `param` through would show up
/// as a failure rather than as two functions changing together.
///
/// A predicate rather than an assertion because the guards on it need to watch it say
/// *false*. With every `PrimKind` currently [`Fidelity::Exact`], a declaration that is wrong
/// on purpose is the only way to show this check can go red at all.
fn cpu_output_is_the_floor(instance: &Instance, floor: Floor, uploads: &[PendingUpload]) -> bool {
    let drawn = cpu_pixels(std::slice::from_ref(instance), uploads);
    let expected = match floor {
        Floor::Nothing => cpu_pixels(&[], uploads),
        Floor::Plain(kind) => cpu_pixels(
            &[Instance {
                uv: [0.0; 4],
                param: 0.0,
                kind: kind as u32,
                ..*instance
            }],
            uploads,
        ),
    };
    drawn == expected
}

#[track_caller]
fn assert_floor(case: &Case, floor: Floor) {
    assert!(
        cpu_output_is_the_floor(&case.instance, floor, &case.uploads),
        "{}: the CPU tier drew something other than the {floor:?} its PrimKind declares. \
         The fallback tier's appearance is a design decision, and this assertion is what \
         keeps it one instead of whatever the rasterizer happened to do with an effect it \
         could not draw",
        case.name
    );
}

fn run_kind(kind: PrimKind) {
    for case in &cases_for(kind) {
        // The fixture list is the same length either way -- what changes is the question.
        // An enhanced primitive needs a *different* fixture, not fewer of them, which is
        // what `no_prim_kind_can_ship_without_a_parity_test` holds every kind to.
        match kind.fidelity() {
            Fidelity::Exact => assert_parity(case),
            Fidelity::Enhanced { floor } => assert_floor(case, floor),
        }
    }
}

// -- scene-effect floors (T022) -------------------------------------------------------

use crate::lighting::{SceneEffect, SceneFloor};
use crate::path::RenderPath;

/// One tier's rendering of one scene effect's contribution, isolated: what the lighting
/// pass would add to or remove from an otherwise-untouched `SURFACE` x `SURFACE` image.
///
/// A function value rather than a method so the mutation test can hand the harness a
/// deliberately wrong renderer and watch it go red — the same design that keeps
/// [`assert_floor`] honest for primitives.
type EffectContribution = fn(SceneEffect, RenderPath) -> Surface;

/// The zero contribution: the pass touched nothing.
fn zero_contribution() -> Surface {
    Surface {
        pixels: vec![[0.0; 4]; (SURFACE * SURFACE) as usize],
        width: SURFACE,
        height: SURFACE,
    }
}

/// The lighting pass's contribution for one effect on one tier, through the **real**
/// transcription rather than a stub.
///
/// This function used to return zero for every effect on every tier, which was honest while
/// no shader existed and became vacuous the moment one did: `scene_effect_holds_its_floor`
/// only compares when the floor is `Nothing`, so a contribution that is zero everywhere
/// compares zero against zero and the sweep passes without touching the pass. The fixture is
/// what makes the comparison mean something — a scene that genuinely produces the effect, so
/// a tier that declares `Nothing` is asserted to have dropped something that was there.
///
/// The tier gate is [`SceneEffect::draws`] rather than a match, so a tier's answer comes from
/// the same declaration the floor does and the two cannot drift apart.
fn lit_contribution(effect: SceneEffect, tier: RenderPath) -> Surface {
    let mut surface = zero_contribution();
    if !effect.draws(tier) {
        return surface;
    }
    let slabs = effect_fixture(effect);
    let k = crate::lighting::hardness(5.0);
    let toward = normalized_light();
    for y in 0..SURFACE {
        for x in 0..SURFACE {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let value = match effect {
                // The two that ride the attenuation channel. Recorded as the pass's
                // departure from 1.0, so an untouched pixel is zero and the floor's
                // expected image is the zero surface without a special case.
                SceneEffect::Shadow | SceneEffect::Occlusion => {
                    let atten = shader::lit_attenuation(p, &slabs, toward, k, None);
                    [1.0 - atten, 0.0, 0.0, 0.0]
                }
                // The additive one, per channel.
                SceneEffect::Bounce => {
                    let added = shader::lit_addition(p, &slabs, k);
                    [added[0], added[1], added[2], 0.0]
                }
                // Nothing draws refraction yet: US4's shader (T069) is unwritten, so the
                // honest contribution is zero and this arm is the seam it lands in. Stated
                // rather than silent, because a zero here is indistinguishable from a
                // dropped effect and the difference is the whole point of the sweep.
                SceneEffect::Refraction => [0.0; 4],
            };
            surface.pixels[(y * SURFACE + x) as usize] = value;
        }
    }
    surface
}

/// The shipped key light, normalized — the direction every lighting fixture marches along.
fn normalized_light() -> [f32; 3] {
    use crate::frame::LIGHT_DIR;
    let len = (LIGHT_DIR[0] * LIGHT_DIR[0]
        + LIGHT_DIR[1] * LIGHT_DIR[1]
        + LIGHT_DIR[2] * LIGHT_DIR[2])
        .sqrt();
    [
        LIGHT_DIR[0] / len,
        LIGHT_DIR[1] / len,
        LIGHT_DIR[2] / len,
    ]
}

/// A scene that genuinely produces `effect`, sized to the `SURFACE` x `SURFACE` fixture.
///
/// One per effect rather than one shared scene: a scene that casts a shadow does not
/// necessarily emit, and a fixture that produced nothing would make its floor assertion
/// vacuous in exactly the way this whole function exists to stop.
fn effect_fixture(effect: SceneEffect) -> Vec<crate::scene::Slab> {
    use crate::scene::Slab;
    let ground = Slab {
        rect: [0.0, 0.0, SURFACE as f32, SURFACE as f32],
        radius: 0.0,
        elevation: 0.0,
        thickness: 1.0,
        attenuation_floor: 0.0,
        // The ground is what receives the bounce; a zero allowance here would make the
        // additive fixture produce nothing and pass for the wrong reason.
        addition_max: 1.0,
        ..Slab::default()
    };
    let mut caster = Slab {
        rect: [
            SURFACE as f32 * 0.25,
            SURFACE as f32 * 0.25,
            SURFACE as f32 * 0.5,
            SURFACE as f32 * 0.25,
        ],
        radius: 2.0,
        elevation: 12.0,
        thickness: 12.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    };
    if effect == SceneEffect::Bounce {
        caster.emission = [1.0, 0.8, 0.4];
        caster.emission_strength = 1.0;
    }
    vec![ground, caster]
}

/// Whether `draw`'s output on `tier` is **exactly** the floor `effect` declares.
///
/// Scene-effect contract rule 2: zero tolerance, because both sides of the comparison run
/// the same rasterizer and there is nothing for a tolerance to forgive — it would only
/// ever hide a real difference. The scene-level mirror of [`cpu_output_is_the_floor`].
fn scene_effect_holds_its_floor(
    effect: SceneEffect,
    tier: RenderPath,
    draw: EffectContribution,
) -> bool {
    let drawn = draw(effect, tier);
    let expected = match effect.floor(tier) {
        // The tier draws the effect in full: parity rather than flooring is the question
        // there, and it is asked by the per-effect fixtures (T032, T041), not here.
        None => return true,
        Some(SceneFloor::Nothing) => zero_contribution(),
        // A bounded floor is the same effect at stated parameters, so its expected image
        // comes through the same contribution. No shipped effect declares one yet: shadow
        // and occlusion are drawn in full on their lowest tier, and everything else floors
        // to Nothing.
        Some(SceneFloor::Bounded { .. }) => draw(effect, tier),
    };
    drawn.pixels == expected.pixels
}

#[test]
fn every_declared_scene_floor_is_exercised_and_held_exactly() {
    // Contract rules 1, 2 and 5 in one sweep: every effect, every tier, every rung run on
    // every build. Today each declared floor is Nothing and the pass contributes nothing,
    // so the suite is green by the honest route — the declarations and the implementation
    // agree because both say "not yet".
    for effect in SceneEffect::ALL {
        for tier in [RenderPath::Cpu, RenderPath::Reduced, RenderPath::Primary] {
            assert!(
                scene_effect_holds_its_floor(effect, tier, lit_contribution),
                "{effect:?} on {tier:?} does not draw the floor it declares"
            );
        }
    }
}

// A little scene for the lighting tests: the canvas, and one caster standing over it.
fn lit_fixture(elevation: f32) -> Vec<crate::scene::Slab> {
    use crate::scene::Slab;
    let ground = Slab {
        rect: [0.0, 0.0, 512.0, 512.0],
        radius: 0.0,
        elevation: 0.0,
        thickness: 1.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    };
    let caster = Slab {
        rect: [96.0, 96.0, 128.0, 64.0],
        radius: 6.0,
        elevation,
        thickness: elevation,
        attenuation_floor: 0.0,
        ..Slab::default()
    };
    vec![ground, caster]
}

#[test]
fn a_caster_at_two_elevations_produces_penumbras_that_scale_with_height() {
    // T032, against the ANALYTIC curve rather than a recorded array. For an edge at height
    // `h` over a receiver, `min(k*h/t)` (research R4) makes the penumbra width at the
    // receiver proportional to the distance the ray travels before passing the edge —
    // which is proportional to `h`. That linearity IS SC-002: shadows must differ by
    // height or elevation is not readable. The shadow term is probed directly, because
    // `lit_attenuation` composes occlusion in and ambient darkening near the caster's wall
    // would pollute a threshold count.
    //
    // A 20-degree light rather than the shipped 5: the mechanism under test is linearity,
    // and a wider penumbra gives the fixed-step march more samples to resolve it with.
    use crate::frame::LIGHT_DIR;

    let k = crate::lighting::hardness(20.0);
    let len = (LIGHT_DIR[0] * LIGHT_DIR[0]
        + LIGHT_DIR[1] * LIGHT_DIR[1]
        + LIGHT_DIR[2] * LIGHT_DIR[2])
        .sqrt();
    let toward = [LIGHT_DIR[0] / len, LIGHT_DIR[1] / len, LIGHT_DIR[2] / len];

    // The shadow profile marching down-screen from under the caster's bottom edge
    // (y = 160), through the shadow it throws, out into the light. 0.125 px steps so a
    // narrow penumbra still spans many samples.
    let profile = |elevation: f32| -> Vec<f32> {
        let slabs = lit_fixture(elevation);
        (0..480)
            .map(|i| {
                let y = 158.0 + i as f32 * 0.125;
                shader::soft_shadow([160.0, y, 0.5], toward, k, &slabs)
            })
            .collect()
    };
    let width = |profile: &[f32]| -> usize {
        profile.iter().filter(|&&a| a > 0.02 && a < 0.98).count()
    };

    let near = width(&profile(8.0));
    let far = width(&profile(24.0));
    assert!(
        far > near && near > 0,
        "a higher caster must throw a softer shadow: {near} penumbra samples at 8 px,          {far} at 24 px"
    );
    let ratio = far as f32 / near as f32;
    assert!(
        (1.6..=5.0).contains(&ratio),
        "penumbra width should scale roughly linearly with elevation (analytic curve):          3x height gave {ratio:.2}x width ({near} -> {far} samples)"
    );
}

#[test]
fn the_shadow_lands_where_the_contact_shadow_offset_already_points() {
    // One light, everywhere: `GeometryDef::offset` displaces contact shadows along
    // `shadow_direction()`, and the raymarched shadow must fall the same way — down and to
    // the right — or the window's shadows point two ways at once. The rig test in qs-ui
    // holds the tokens to LIGHT_DIR; this holds the march to it.
    use crate::frame::{LIGHT_DIR, shadow_direction};

    let k = crate::lighting::hardness(5.0);
    let len = (LIGHT_DIR[0] * LIGHT_DIR[0]
        + LIGHT_DIR[1] * LIGHT_DIR[1]
        + LIGHT_DIR[2] * LIGHT_DIR[2])
        .sqrt();
    let toward = [LIGHT_DIR[0] / len, LIGHT_DIR[1] / len, LIGHT_DIR[2] / len];
    let slabs = lit_fixture(16.0);

    // A caster at 16 px reaches roughly 0.8 x 16 = 13 px past its edge. Sample inside that
    // reach on the shadow side (down-right of the bottom edge), and mirrored on the light
    // side (up-left of the top edge), where the same wall proximity gives the same
    // occlusion but no shadow.
    let direction = shadow_direction();
    assert!(
        direction[0] > 0.0 && direction[1] > 0.0,
        "the shipped light is up-and-left"
    );
    let shade = shader::soft_shadow([166.0, 168.0, 0.5], toward, k, &slabs);
    let lit = shader::soft_shadow([154.0, 88.0, 0.5], toward, k, &slabs);
    assert!(
        shade < lit,
        "the raymarched shadow falls opposite `shadow_direction()`: shadow side {shade},          light side {lit}"
    );
}

#[test]
fn the_allowance_floor_bounds_the_attenuation_per_slab() {
    // The clamp is the contrast gate's closed-form claim being true on real frames
    // (lit-contrast rules 1a/3a). A receiver directly under the caster, floored at 0.87 —
    // the light theme's text-ground allowance — may not darken past it, however hard the
    // geometry shadows it.
    use crate::frame::LIGHT_DIR;
    let k = crate::lighting::hardness(5.0);
    let len = (LIGHT_DIR[0] * LIGHT_DIR[0]
        + LIGHT_DIR[1] * LIGHT_DIR[1]
        + LIGHT_DIR[2] * LIGHT_DIR[2])
        .sqrt();
    let toward = [LIGHT_DIR[0] / len, LIGHT_DIR[1] / len, LIGHT_DIR[2] / len];
    let mut slabs = lit_fixture(24.0);
    slabs[0].attenuation_floor = 0.87;

    // Deep in the caster's shadow: a 24 px caster reaches ~20 px past its edge along
    // the shadow direction, so (170, 175) sits in the umbra.
    let atten = shader::lit_attenuation([170.0, 175.0], &slabs, toward, k, None);
    assert!(
        atten >= 0.87,
        "the shader may not exceed the allowance: floor 0.87, attenuation {atten}"
    );

    slabs[0].attenuation_floor = 0.0;
    let free = shader::lit_attenuation([170.0, 175.0], &slabs, toward, k, None);
    assert!(
        free <= atten,
        "with the floor released the same pixel should be at least as dark"
    );
}

#[test]
fn the_lighting_shader_and_its_transcription_state_the_same_bounds() {
    // The lighting half of `the_transcription_matches_the_shader_source`: every constant
    // and function the transcription mirrors must exist in the WGSL, and no derivative may
    // — the CPU side cannot reproduce one, and the header forbids them by name.
    let source = include_str!("shaders/lighting.wgsl");
    for needle in [
        "fn slab_distance",
        "fn scene_distance",
        "fn soft_shadow",
        "fn occlusion",
        "fn sd_rounded_box",
        "fn bounce",
        "fn bounce_shadow",
        "fn focus_shadow",
        "fn focus_weight",
        "fn focus_ambient",
        "fn focus_point",
        "fn scene_distance_excluding",
        "const LIT_SLABS: u32 = 192u",
        "const SHADOW_STEPS: u32 = 12u",
        "const AO_SAMPLES: u32 = 4u",
        "const AO_STRENGTH: f32 = 0.35",
        "const BOUNCE_REACH: f32 = 90.0",
        "const FOCUS_REACH: f32 = 220.0",
    ] {
        assert!(
            source.contains(needle),
            "lighting.wgsl has no `{needle}`; the transcription describes a shader that \
             does not exist"
        );
    }
    assert!(
        !source.contains("fwidth(") && !source.contains("dpdx(") && !source.contains("dpdy("),
        "a derivative reached lighting.wgsl, which the CPU transcription cannot reproduce"
    );
    assert_eq!(crate::lighting::LIT_SLABS, 192);
    assert_eq!(shader::SHADOW_STEPS, 12);
    assert_eq!(shader::AO_SAMPLES, 4);
    assert!((shader::AO_STRENGTH - 0.35).abs() < f32::EPSILON);
    assert!((shader::BOUNCE_REACH - 90.0).abs() < f32::EPSILON);
    assert!((shader::FOCUS_REACH - 220.0).abs() < f32::EPSILON);
}

#[test]
fn the_scene_floor_harness_can_go_red() {
    // The mutation proof T022 requires: a pass that draws SOMETHING where the effect's
    // floor says Nothing must fail the harness. A fidelity class that cannot go red is a
    // promise nobody is keeping.
    fn wrong(_effect: SceneEffect, _tier: RenderPath) -> Surface {
        let mut surface = zero_contribution();
        surface.pixels[0] = [1.0, 0.0, 0.0, 1.0];
        surface
    }
    assert!(
        !scene_effect_holds_its_floor(SceneEffect::Bounce, RenderPath::Reduced, wrong),
        "a contribution where the floor declares Nothing was accepted — the harness \
         cannot detect the failure it exists for"
    );
}

#[test]
fn a_floored_effect_drops_something_that_was_actually_there() {
    // T055's real content, and the assertion that stops the sweep above from passing for the
    // wrong reason. `scene_effect_holds_its_floor` compares nothing at all on a tier that
    // draws the effect in full, so a contribution function that returned zero everywhere
    // would satisfy every declared `Nothing` floor by never producing anything to drop. This
    // asserts the other half: on the lowest tier that DOES draw it, the fixture's
    // contribution is non-zero.
    //
    // Refraction is excluded by name rather than by a general skip, because its shader (T069)
    // is unwritten and a blanket "skip the empty ones" would silently re-admit exactly the
    // vacuity this test exists to close once bounce or shadow regressed to nothing.
    for effect in SceneEffect::ALL {
        if effect == SceneEffect::Refraction {
            continue;
        }
        let tier = effect.requires();
        assert!(
            effect.draws(tier),
            "{effect:?} does not draw on the tier it says it requires"
        );
        let drawn = lit_contribution(effect, tier);
        let energy: f32 = drawn.pixels.iter().map(|p| p[0] + p[1] + p[2]).sum();
        assert!(
            energy > 0.0,
            "{effect:?} contributes nothing on {tier:?}, the lowest tier that draws it — \
             every floor assertion for it is therefore comparing nothing against nothing"
        );

        // And the floor below it drops that contribution entirely.
        for lower in [RenderPath::Reduced, RenderPath::Cpu] {
            if effect.floor(lower) == Some(SceneFloor::Nothing) {
                assert!(
                    scene_effect_holds_its_floor(effect, lower, lit_contribution),
                    "{effect:?} on {lower:?} declares Nothing and drew something"
                );
            }
        }
    }
}

#[test]
fn bounce_is_occluded_by_what_stands_between_the_emitter_and_the_receiver() {
    // T047. The contract's word is "occluded", and the difference between an occluded bounce
    // and an unoccluded one is not cosmetic: an unoccluded bounce leaks the selected row's
    // light through the interface's own geometry, so a surface visibly behind something
    // taller glows anyway. That reads as a rendering fault rather than as light.
    //
    // The control is the same scene with the obstruction removed — measuring one number and
    // asserting it is small would pass on a scene that was too far away to be lit at all.
    use crate::scene::Slab;
    let k = crate::lighting::hardness(5.0);

    let ground = Slab {
        rect: [0.0, 0.0, 512.0, 512.0],
        elevation: 0.0,
        thickness: 1.0,
        attenuation_floor: 0.0,
        addition_max: 1.0,
        ..Slab::default()
    };
    let emitter = Slab {
        rect: [0.0, 100.0, 40.0, 40.0],
        elevation: 10.0,
        thickness: 10.0,
        attenuation_floor: 0.0,
        emission: [1.0, 1.0, 1.0],
        emission_strength: 1.0,
        ..Slab::default()
    };
    // Tall, thin, and directly between the emitter and the sample point.
    let wall = Slab {
        rect: [60.0, 90.0, 10.0, 60.0],
        elevation: 60.0,
        thickness: 60.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    };

    let sample = [100.0, 120.0];
    let open = shader::lit_addition(sample, &[ground, emitter], k);
    let blocked = shader::lit_addition(sample, &[ground, emitter, wall], k);

    assert!(
        open[0] > 0.0,
        "the control scene lights nothing, so the occlusion assertion below would hold \
         for a scene with no light in it: {open:?}"
    );
    assert!(
        blocked[0] < open[0] * 0.1,
        "a receiver behind an obstruction still took the emitter's light: {blocked:?} \
         behind the wall against {open:?} with it removed"
    );
}

// -- the tests -----------------------------------------------------------------------

#[test]
fn rect_agrees_across_tiers() {
    run_kind(PrimKind::Rect);
}

#[test]
fn stroke_agrees_across_tiers() {
    run_kind(PrimKind::Stroke);
}

#[test]
fn glyph_agrees_across_tiers() {
    run_kind(PrimKind::Glyph);
}

#[test]
fn gradient_agrees_across_tiers() {
    run_kind(PrimKind::Gradient);
}

#[test]
fn sweep_agrees_across_tiers() {
    run_kind(PrimKind::Sweep);
}

#[test]
fn field_stays_at_its_floor_on_the_cpu_tier() {
    run_kind(PrimKind::Field);
}

/// A field of one centre, dead centre, reaching a quarter of the width.
fn fixture_field(tint: Srgba) -> crate::frame::FieldWash {
    crate::frame::FieldWash::new(&[crate::frame::FieldCentre {
        at: [0.5, 0.5],
        drift: [0.0, 0.0],
        reach: 0.25,
        phase: 0.0,
        tint,
    }])
}

/// The reference tier's pixels for one field instance under one wash.
fn field_pixels(instance: Instance, field: crate::frame::FieldWash) -> Surface {
    let mut list = DrawList::default();
    list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
    list.set_field(field);
    list.instances.push(instance);
    list.end_batch(None, false);
    render_reference(
        &list,
        &vec![0u8; (ATLAS as usize) * (ATLAS as usize)],
        ATLAS,
    )
}

#[test]
fn a_centre_reaches_its_limit_and_stops() {
    // The property the whole cost argument rests on: bounded support. A tail that merely got
    // small would still be four evaluations per fragment, and the claim that a full-viewport
    // pass is affordable would be resting on nothing.
    let base = Srgba::new(0.10, 0.10, 0.13, 1.0);
    let tint = Srgba::new(0.30, 0.62, 0.95, 1.0);
    let field = fixture_field(tint);
    let surface = field_pixels(
        Instance::field(0.0, 0.0, SURFACE as f32, SURFACE as f32, 1.0, 0.0, base),
        field,
    );

    let mid = SURFACE / 2;
    let centre = surface.at(mid, mid);
    // A quarter of the width, from the middle: the reach lands here.
    let limit = mid + SURFACE / 4;

    let base_linear = base.to_premul_linear_f32();
    let differs = |p: [f32; 4]| -> f32 {
        (0..3)
            .map(|i| (p[i] - base_linear[i]).abs())
            .fold(0.0_f32, f32::max)
    };

    assert!(
        differs(centre) > 0.05,
        "the centre did not tint the ground at all: {centre:?}"
    );
    // Past the reach the field is the base *exactly*, up to the dither's own half-level.
    for x in (limit + 2)..SURFACE {
        let p = surface.at(x, mid);
        assert!(
            differs(p) <= 2.0 / 255.0,
            "the centre is still tinting at x={x}, which is past its reach: {p:?}"
        );
    }
}

#[test]
fn the_phase_carries_the_centres_across_the_window() {
    // A field that did not drift would satisfy every floor fixture above, because the floor
    // is the base either way. This is what says the phase reaches the centres.
    let base = Srgba::new(0.10, 0.10, 0.13, 1.0);
    let tint = Srgba::new(0.30, 0.62, 0.95, 1.0);
    let drifting = crate::frame::FieldWash::new(&[crate::frame::FieldCentre {
        at: [0.5, 0.5],
        drift: [0.30, 0.0],
        reach: 0.25,
        phase: 0.0,
        tint,
    }]);

    let brightest_column = |phase: f32| -> u32 {
        let surface = field_pixels(
            Instance::field(0.0, 0.0, SURFACE as f32, SURFACE as f32, 1.0, phase, base),
            drifting,
        );
        let row = SURFACE / 2;
        (0..SURFACE)
            .max_by(|a, b| {
                let at = |x: u32| surface.at(x, row)[2];
                at(*a)
                    .partial_cmp(&at(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    };

    // At phase 0 the drift term is `cos(0) = +1`, so the centre sits right of the middle; half
    // a turn later `cos(PI) = -1` puts it left of it. Asserting the *sides* rather than a
    // distance keeps this about the direction the phase carries a centre, which is the part a
    // sign error gets wrong.
    let at_rest = brightest_column(0.0);
    let half_turn = brightest_column(std::f32::consts::PI);
    assert!(
        at_rest > SURFACE / 2,
        "at phase zero the centre is not right of the middle: {at_rest}"
    );
    assert!(
        half_turn < SURFACE / 2,
        "half a turn later the centre is not left of the middle: {half_turn}"
    );
}

#[test]
fn the_worst_colour_a_field_reaches_is_one_of_its_own_centres() {
    // The claim the contrast gate rests on, and the reason it is measured here rather than
    // argued in a comment.
    //
    // `Material::composites` checks each centre's tint at full amplitude over the base. It
    // does NOT check the Oklab blend of two overlapping centres, because there are infinitely
    // many of those. The argument is that Oklab's L is monotone in luminance, so a weighted
    // mean of two centres has a luminance between theirs and cannot be darker than the darker
    // one or lighter than the lighter one -- so the extremes the gate checks bracket every
    // blend the shader can produce.
    //
    // This walks the weight simplex densely and checks that. A blend that escaped the bracket
    // would be a surface the gate measured and the renderer then exceeded, which is the exact
    // shape of failure the whole `over`/`text` machinery exists to prevent.
    use crate::color::{linear_rgb_to_oklab, oklab_to_linear_rgb};

    let tints = [
        Srgba::new(0.30, 0.62, 0.95, 1.0),
        Srgba::new(0.86, 0.42, 0.31, 1.0),
        Srgba::new(0.36, 0.80, 0.55, 1.0),
        Srgba::new(0.68, 0.44, 0.90, 1.0),
    ];
    // Relative luminance on colour that is ALREADY linear, which is what the shader is
    // holding at this point. `Srgba::relative_luminance` applies the sRGB decode first, so
    // handing it a linear value would decode it twice -- the weights are the shared part.
    let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

    let labs: Vec<[f32; 3]> = tints
        .iter()
        .map(|t| {
            let p = t.to_premul_linear_f32();
            linear_rgb_to_oklab(shader::unpremultiply(p))
        })
        .collect();
    let ends: Vec<f32> = tints
        .iter()
        .map(|t| {
            let p = t.to_premul_linear_f32();
            luminance([p[0], p[1], p[2]])
        })
        .collect();
    let (lo, hi) = ends
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));

    // Every combination of integer weights summing to 12 over four centres: 455 blends,
    // including every pair, triple and the even mix.
    const STEPS: i32 = 12;
    let mut checked = 0;
    for a in 0..=STEPS {
        for b in 0..=(STEPS - a) {
            for c in 0..=(STEPS - a - b) {
                let d = STEPS - a - b - c;
                let w = [a, b, c, d].map(|n| n as f32 / STEPS as f32);
                let mut lab = [0.0_f32; 3];
                for (centre, weight) in labs.iter().zip(w) {
                    for (slot, channel) in lab.iter_mut().zip(centre) {
                        *slot += channel * weight;
                    }
                }
                let rgb = oklab_to_linear_rgb(lab);
                let y = luminance([
                    rgb[0].clamp(0.0, 1.0),
                    rgb[1].clamp(0.0, 1.0),
                    rgb[2].clamp(0.0, 1.0),
                ]);
                assert!(
                    y >= lo - 1e-3 && y <= hi + 1e-3,
                    "the blend at weights {w:?} has luminance {y}, outside the [{lo}, {hi}] \
                     the gate checks -- the contrast gate would be measuring a surface the \
                     renderer can exceed"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(
        checked, 455,
        "the simplex walk did not cover what it claims"
    );
}

#[test]
fn the_shader_draws_a_field_and_not_only_the_transcription() {
    // Both sides of a floor fixture are the same rasterizer, so nothing above can see whether
    // the GPU tiers grew the branch at all. Same shape of check as the sweep's and the rim's.
    let source = include_str!("shaders/instance.wgsl");
    let fs_main = source
        .split_once("fn fs_main")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    assert!(
        fs_main.contains("KIND_FIELD") && fs_main.contains("field("),
        "instance.wgsl's fragment stage has no KIND_FIELD branch, so nothing on the GPU tiers \
         draws the ground however green the transcription is"
    );
    for needed in [
        "fn field_weight",
        "fn field(",
        "field_place",
        "field_tint",
        "field_form",
    ] {
        assert!(
            source.contains(needed),
            "instance.wgsl is missing `{needed}`, so the field is not reading the scene"
        );
    }
    // The dither reaches the field. This is the largest surface in the window and the interval
    // it moves across is a handful of 8-bit levels, so an undithered field bands -- which is
    // the one artefact here that is visible from across a room.
    let body = source
        .split_once("fn field(")
        .and_then(|(_, rest)| rest.split_once("\n}"))
        .map(|(body, _)| body)
        .unwrap_or_default();
    assert!(
        body.contains("dithered("),
        "the field does not dither, and a full-window ramp at 8 bits bands"
    );
}

#[test]
fn a_sweep_has_no_seam_where_the_angle_wraps() {
    // The defect this primitive is not allowed to ship with, and the one no parity fixture
    // can see: both tiers compute the parameter the same way, so two tiers agreeing on a hue
    // step would measure zero divergence and the suite would stay green over a visible line
    // down the middle of every swept surface.
    //
    // So this measures the parameter directly. Walk a full turn at a fine step and assert
    // that no adjacent pair moves further than a step's worth of the ramp. The mirrored form
    // has slope 2 in turns, so a step of `1/STEPS` of a turn can move `t` by at most
    // `2 / STEPS`; a raw `fract` would jump the whole way from 1 to 0 in one step, which is
    // `STEPS / 2` times the bound and cannot hide inside any tolerance worth writing.
    const STEPS: usize = 2048;
    let half = [1.0_f32, 1.0];
    let bound = 2.0 / STEPS as f32 + 1e-5;

    for phase in [0.0_f32, 0.7, -1.9, std::f32::consts::PI] {
        let mut previous = None;
        // Start at the wrap itself (-PI) so the discontinuity is inside the walk rather than
        // at one end of it, where a comparison against the next sample would never happen.
        for i in 0..=STEPS {
            let angle = -std::f32::consts::PI + std::f32::consts::TAU * (i as f32 / STEPS as f32);
            let local = [angle.cos(), angle.sin()];
            let t = shader::sweep_t(local, half, phase);
            assert!(
                (0.0..=1.0).contains(&t),
                "t left the ramp at angle {angle} phase {phase}: {t}"
            );
            if let Some(previous) = previous {
                let step: f32 = t - previous;
                assert!(
                    step.abs() <= bound,
                    "the sweep stepped {step} at angle {angle} phase {phase}, which is a hue \
                     seam and not a ramp; the bound is {bound}"
                );
            }
            previous = Some(t);
        }
    }

    // And the wrap closes: one full turn returns to where it started, so the seam is absent
    // rather than merely small.
    for phase in [0.0_f32, 0.7, -1.9] {
        let at = |angle: f32| shader::sweep_t([angle.cos(), angle.sin()], half, phase);
        assert!(
            (at(-std::f32::consts::PI) - at(std::f32::consts::PI)).abs() <= 1e-4,
            "the two sides of the wrap disagree at phase {phase}"
        );
    }
}

#[test]
fn the_phase_carries_the_highlight_around_the_shape() {
    // What the primitive is *for*: the far stop travels with the phase rather than the shape
    // being re-tinted in place. A sweep whose phase did nothing would satisfy every parity
    // fixture above, because both tiers would draw the same still picture.
    let half = [1.0_f32, 1.0];
    let at = |angle: f32, phase: f32| shader::sweep_t([angle.cos(), angle.sin()], half, phase);

    // At phase zero the far stop is at +x, and it is *the* far stop: t = 1 there and nowhere
    // else on the turn.
    assert!(
        (at(0.0, 0.0) - 1.0).abs() <= 1e-5,
        "the highlight is not at +x"
    );
    assert!(
        at(std::f32::consts::PI, 0.0) <= 1e-5,
        "the near stop is not opposite it"
    );

    // A quarter turn of phase moves the peak a quarter turn, in the same sense a gradient's
    // angle turns its axis: through +y, which on this surface is downward.
    let quarter = std::f32::consts::FRAC_PI_2;
    assert!(
        (at(quarter, quarter) - 1.0).abs() <= 1e-5,
        "a quarter turn of phase did not move the highlight a quarter turn"
    );
    // And +x is now a quarter of a turn from the peak, which on a mirrored ramp is exactly
    // the midpoint between the two stops. Asserting the exact value rather than "less than
    // the peak": a sweep that merely dimmed everywhere would also be less than the peak.
    assert!(
        (at(0.0, quarter) - 0.5).abs() <= 1e-5,
        "+x is not the ramp's midpoint a quarter turn after the highlight left it: {}",
        at(0.0, quarter)
    );
}

#[test]
fn the_shader_sweeps_and_not_only_the_transcription() {
    // The half `run_kind` cannot see: both sides of a parity fixture are Rust, so a shader
    // that never grew the branch would still measure zero. Same shape of check as
    // `the_shader_lights_the_inside_of_an_edge_and_not_only_the_transcription`.
    let source = include_str!("shaders/instance.wgsl");
    let fs_main = source
        .split_once("fn fs_main")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    assert!(
        fs_main.contains("KIND_SWEEP") && fs_main.contains("sweep_t"),
        "instance.wgsl's fragment stage has no KIND_SWEEP branch, so nothing on the GPU tiers \
         draws a sweep however green the transcription is"
    );
    assert!(
        source.contains("fn sweep_t") && source.contains("atan2"),
        "the shader has no angular parameter"
    );
    // The shared helper, which is the acceptance criterion that a second ramp did not grow.
    assert!(
        source.contains("fn ramp_at"),
        "instance.wgsl has no shared ramp_at, so the sweep and the gradient are two copies of \
         the same Oklab walk"
    );
    let sweep_arm = fs_main
        .split_once("KIND_SWEEP")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    assert!(
        sweep_arm
            .split_once("else if")
            .map(|(arm, _)| arm)
            .unwrap_or(sweep_arm)
            .contains("ramp_at"),
        "the sweep branch does not walk the shared ramp"
    );
}

/// The colour the CPU tier actually put at one pixel, back in straight sRGB.
fn sampled_srgb(pixmap: &Pixmap, x: u32, y: u32) -> Srgba {
    let p = pixmap.pixels()[(y * pixmap.width() + x) as usize];
    let a = f32::from(p.alpha()) / 255.0;
    let channel = |c: u8| -> f32 {
        let linear = f32::from(c) / 255.0;
        crate::color::linear_to_srgb(if a > 0.0 {
            (linear / a).clamp(0.0, 1.0)
        } else {
            0.0
        })
    };
    Srgba::new(channel(p.red()), channel(p.green()), channel(p.blue()), a)
}

#[test]
fn the_ramp_is_walked_in_oklab_and_not_in_linear_srgb() {
    // The one thing the parity fixtures structurally cannot check.
    //
    // Every `gradient/*` case above compares the CPU tier against a transcription of the
    // shader. If both sides lerped in linear sRGB they would agree exactly and every
    // fixture would be green -- so parity is evidence that the two tiers do the same
    // thing, and no evidence at all about *which* thing. chunk:oklch-color-ramps put the
    // palette in a perceptual space precisely so a ramp between two stops keeps its
    // chroma; a linear-sRGB lerp between the same two stops sags toward grey in the
    // middle while both endpoints stay correct, which is the failure that ships.
    //
    // The property being asserted is the one that *defines* a perceptual space: the
    // midpoint of the ramp is the perceptual midpoint of its endpoints, so `l` at the
    // centre is the mean of the two `l`s. A linear-light lerp has no such property --
    // averaging light rather than lightness, it reaches the bright end early and spends
    // most of the ramp's length near the top, which collapses the dark half of the
    // gradient into a narrow band.
    //
    // My first attempt asserted the wrong thing: that the Oklab midpoint would be *more*
    // chromatic, on the strength of the usual "blue to yellow goes through grey" story.
    // That story is about interpolating in *gamma-encoded* sRGB. In linear light the
    // midpoint of these two stops measures c = 0.103 against Oklab's 0.055 -- more
    // colourful, not less, because it is also much lighter. Left here because the bounds
    // below are the second attempt and the first one looked just as plausible.
    let from = Srgba::new(0.145, 0.176, 0.278, 1.0); // deep steel blue
    let to = Srgba::new(0.925, 0.706, 0.196, 1.0); // amber
    let (x, y, w, h) = (8.0_f32, 8.0_f32, 48.0_f32, 24.0_f32);

    let mut cpu = CpuRasterizer::new(SURFACE, SURFACE, ATLAS).unwrap();
    let mut list = DrawList::default();
    list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
    list.instances
        .push(Instance::gradient(x, y, w, h, 0.0, 0.0, from, to));
    list.end_batch(None, false);
    let measured = sampled_srgb(
        cpu.render(&list),
        (x + w * 0.5) as u32,
        (y + h * 0.5) as u32,
    );

    // The midpoint a linear-light lerp would have produced, which is what the shader would
    // do if `ramp` simply mixed the two premultiplied colours.
    let lerp = |a: f32, b: f32| crate::color::linear_to_srgb((a + b) * 0.5);
    let muddy = Srgba::new(
        lerp(
            crate::color::srgb_to_linear(from.r),
            crate::color::srgb_to_linear(to.r),
        ),
        lerp(
            crate::color::srgb_to_linear(from.g),
            crate::color::srgb_to_linear(to.g),
        ),
        lerp(
            crate::color::srgb_to_linear(from.b),
            crate::color::srgb_to_linear(to.b),
        ),
        1.0,
    );

    let got = measured.to_oklch();
    let linear = muddy.to_oklch();
    let perceptual_mid = (from.to_oklch().l + to.to_oklch().l) * 0.5;
    eprintln!(
        "ramp midpoint: measured l={:.3} c={:.3} / linear-sRGB l={:.3} c={:.3} / \
         perceptual mid l={:.3}",
        got.l, got.c, linear.l, linear.c, perceptual_mid
    );

    // Guard the guard first. For two stops of similar lightness the two spaces agree at
    // the midpoint, and this fixture would then be green on a ramp that never touched
    // Oklab. The stops are far apart in lightness on purpose, and this is what refuses to
    // let a later edit quietly bring them together.
    assert!(
        (linear.l - perceptual_mid).abs() > 0.05,
        "a linear-light lerp landed within {:.4} of the perceptual midpoint, so these two \
         stops cannot tell the two spaces apart and this fixture is proving nothing",
        (linear.l - perceptual_mid).abs()
    );

    // 0.012 is the 8-bit sampling floor, not a tolerance for being slightly wrong: the
    // pixel is read back through a linear premultiplied RGBA8 pixmap, and one step there
    // is worth roughly 0.004 of OKLCH `l` in this range. The linear-light answer misses by
    // 0.088, seven times the bound.
    assert!(
        (got.l - perceptual_mid).abs() < 0.012,
        "the ramp's midpoint measured l={:.4} where the perceptual midpoint of its two \
         stops is {:.4}. A ramp walked in Oklab lands on that number by construction; the \
         linear-light lerp lands on {:.4} instead",
        got.l,
        perceptual_mid,
        linear.l
    );
}

#[test]
fn the_cpu_tier_resolves_an_edge_to_a_quarter_of_a_pixel() {
    // The source of every loosened bound in this file, measured rather than asserted from
    // documentation -- `tiny-skia` does not document it.
    //
    // Slide a hard vertical edge across a pixel in sixteenths and read the coverage of the
    // pixel it crosses. The shader's answer is the offset itself. `tiny-skia`'s answer is
    // that offset snapped to the nearest quarter.
    //
    // This test is also the tripwire on the rest of the file. If `tiny-skia` ever
    // rasterizes more finely, `worst` collapses and this fails -- which is the signal that
    // `CPU_EDGE_QUANTUM` and every bound derived from it are now looser than the truth,
    // and the suite has gone quietly blunt. A tolerance nobody re-derives is a tolerance
    // that only ever grows.
    let mut worst = 0.0f32;
    for sixteenth in 0..16 {
        let offset = sixteenth as f32 / 16.0;
        let left = 20.0 + offset;

        let mut cpu = CpuRasterizer::new(SURFACE, SURFACE, ATLAS).unwrap();
        let mut list = DrawList::default();
        list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
        list.instances.push(Instance::rect(
            left,
            10.0,
            20.0,
            20.0,
            0.0,
            Srgba::new(1.0, 1.0, 1.0, 1.0),
        ));
        list.end_batch(None, false);
        let got =
            f32::from(cpu.render(&list).pixels()[(20 * SURFACE + 20) as usize].alpha()) / 255.0;

        // The pixel spans [20, 21] and the shape starts at `left`, so exact coverage is
        // `1 - offset`. That is also what the reference produces, which the parity
        // fixtures already prove; here only the CPU tier is under the microscope.
        //
        // What snaps is the *edge*, not the coverage -- the distinction matters, because
        // rounding a coverage of 0.875 to a quarter gives 1.0 while rounding the edge
        // gives 0.75, and only the second one is what happens.
        let exact = 1.0 - offset;
        let snapped = 1.0 - ((left / CPU_EDGE_QUANTUM).round() * CPU_EDGE_QUANTUM - 20.0);
        assert!(
            (got - snapped).abs() <= 2.0 / 255.0,
            "an edge at +{offset} produced coverage {got:.4}; snapping the edge to the \
             quarter-pixel grid predicts {snapped:.4}"
        );
        worst = worst.max((got - exact).abs());
    }

    assert!(
        worst >= MAX_EDGE_ERROR - 1.0 / 255.0,
        "the CPU tier placed every edge to within {worst:.4} of exact, better than the \
         {MAX_EDGE_ERROR} this file's bounds are derived from. That is good news and a \
         broken assumption: re-measure CPU_EDGE_QUANTUM and tighten the fixtures, because \
         they are now tolerating error that no longer exists"
    );
}

/// A white instance, for the floor checks below. Opaque on purpose: a faint one could
/// satisfy a `Nothing` floor by rounding to zero and the guard would prove nothing.
fn floor_probe(kind_of: impl Fn(Srgba) -> Instance) -> Instance {
    kind_of(Srgba::new(1.0, 1.0, 1.0, 1.0))
}

#[test]
fn the_floor_check_fails_when_the_cpu_tier_draws_more_than_its_floor() {
    // The mutation check for the fidelity route, and the reason `Enhanced` is a contract
    // rather than a permission slip.
    //
    // The failure it stands in for is concrete and well-meant: someone adds a glow to the
    // CPU tier as a hard rounded rect, or as a stack of translucent rings, because a halo
    // that vanishes on the fallback tier looks like a missing feature. The declared floor
    // said `Nothing`, and this is what notices.
    //
    // Both directions of wrong are covered. Drawing *something* where the floor says
    // nothing, and drawing the wrong *primitive* where the floor names one -- a ring is
    // not a fill, and a check that only compared "is there ink" would pass it.
    let solid = floor_probe(|c| Instance::rect(16.0, 20.0, 32.0, 24.0, 4.0, c));
    assert!(
        !cpu_output_is_the_floor(&solid, Floor::Nothing, &[]),
        "a rect the CPU tier plainly draws was accepted against a floor of Nothing: the \
         floor check cannot fail, so it is not checking anything"
    );

    let ring = floor_probe(|c| Instance::stroke(10.0, 10.0, 20.0, 20.0, 0.0, 2.0, c));
    assert!(
        !cpu_output_is_the_floor(&ring, Floor::Plain(PrimKind::Rect), &[]),
        "a 2px ring was accepted as a filled rect: the floor check is comparing something \
         weaker than the pixels"
    );
}

#[test]
fn the_floor_check_passes_when_the_cpu_tier_draws_exactly_its_floor() {
    // The other half of the mutation pair. A check that can only fail is as useless as one
    // that can only pass, and this is the one that would catch `cpu_pixels` rendering two
    // different surfaces for reasons that have nothing to do with fidelity -- a stray
    // atlas upload, a clear colour that drifted, an uninitialised pixmap.
    let solid = floor_probe(|c| Instance::rect(16.0, 20.0, 32.0, 24.0, 4.0, c));
    assert!(
        cpu_output_is_the_floor(&solid, Floor::Plain(PrimKind::Rect), &[]),
        "a rect was rejected as its own floor"
    );

    // A primitive that genuinely draws nothing satisfies a floor of nothing. This is the
    // shape a real `Enhanced` kind takes on this tier once `cpu_floor` has dropped it.
    let invisible = Instance::rect(16.0, 20.0, 32.0, 24.0, 4.0, Srgba::TRANSPARENT);
    assert!(
        cpu_output_is_the_floor(&invisible, Floor::Nothing, &[]),
        "an instance that laid down no ink was rejected against a floor of Nothing"
    );
}

// -- the glow's own geometry --------------------------------------------------------------
//
// Everything above this line compares two tiers. A glow has only one tier, so the floor
// check says the CPU side draws nothing and *nothing at all* says the GPU side draws
// something. These four tests are that half: they hold the reference -- the transcription of
// `instance.wgsl` that the whole suite is built on -- to the halo the design asked for.

/// The reference tier's pixels for one instance, alone on the fixture surface.
fn reference_pixels(instance: Instance) -> Surface {
    let mut list = DrawList::default();
    list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
    list.instances.push(instance);
    list.end_batch(None, false);
    render_reference(
        &list,
        &vec![0u8; (ATLAS as usize) * (ATLAS as usize)],
        ATLAS,
    )
}

/// The fixture glow: a rounded box whose right edge is at x = 44 and whose vertical centre
/// is y = 32, so a horizontal scan out of its right flank meets no corner.
const GLOW_RIGHT_EDGE: f32 = 44.0;
const GLOW_FALLOFF: f32 = 8.0;

fn fixture_glow(color: Srgba) -> Instance {
    Instance::glow(20.0, 22.0, 24.0, 20.0, 6.0, GLOW_FALLOFF, color)
}

#[test]
fn a_glow_reaches_beyond_its_shape_and_fades_to_nothing() {
    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    let surface = reference_pixels(fixture_glow(white));

    // Solid inside: the profile is 1 wherever the distance is negative, which is what makes
    // a glow usable on its own and not only as something to hide under a fill.
    assert!(
        (surface.at(32, 32)[3] - 1.0).abs() < 1e-3,
        "the glow is not opaque inside its own shape"
    );

    // The profile, measured against the curve it is supposed to be, one pixel at a time.
    // Comparing against `(1 - d/falloff)^2` rather than against a recorded array is what
    // makes this a statement about the design instead of a snapshot: a halo that changed
    // shape but stayed monotonic would pass a monotonicity check and fail this one.
    let mut previous = 1.0_f32;
    for step in 0..=9 {
        let px = GLOW_RIGHT_EDGE as u32 + step;
        let distance = (px as f32 + 0.5) - GLOW_RIGHT_EDGE;
        let fade = (1.0 - (distance / GLOW_FALLOFF).clamp(0.0, 1.0)).max(0.0);
        let expected = fade * fade;
        let measured = surface.at(px, 32)[3];
        assert!(
            (measured - expected).abs() < 2.0 / 255.0,
            "at {distance:.1}px outside the shape the halo measured {measured:.4}, not the \
             {expected:.4} a quadratic falloff over {GLOW_FALLOFF}px predicts"
        );
        assert!(
            measured <= previous + 1e-6,
            "the halo brightened at {distance:.1}px out -- a falloff has to be monotonic or \
             it has a visible ring in it"
        );
        previous = measured;
    }

    // And it is genuinely gone by the limit, rather than merely faint. A halo that stops at
    // a non-zero value has an edge, wherever the quad happens to end.
    let past = GLOW_RIGHT_EDGE as u32 + GLOW_FALLOFF as u32;
    assert_eq!(
        surface.at(past, 32)[3],
        0.0,
        "the halo still had ink at its own falloff limit"
    );
}

#[test]
fn the_glow_quad_is_padded_by_its_whole_falloff_and_not_by_one_pixel() {
    // The failure this chunk predicted, and the one that reads as clipping rather than as a
    // bug in the padding. Every non-glyph quad grows by exactly one pixel so its
    // antialiased edge has somewhere to live; a halo drawn into a one-pixel margin is a
    // halo with a square edge, and at a small radius nobody notices until somebody asks for
    // a big one.
    //
    // Reverting `quad_pad` to a flat 1.0 turns both halves of this red.
    assert_eq!(
        shader::quad_pad(shader::KIND_GLOW, GLOW_FALLOFF),
        GLOW_FALLOFF + 1.0,
        "the transcription pads a glow by something other than its falloff"
    );

    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    let surface = reference_pixels(fixture_glow(white));

    // Five pixels out is inside the falloff and four pixels outside any one-pixel margin.
    let far = surface.at(GLOW_RIGHT_EDGE as u32 + 5, 32)[3];
    assert!(
        far > 0.05,
        "5px outside the shape the halo measured {far:.4}: the quad is not reaching its own \
         falloff, so the effect is being cut off square"
    );

    // The corner diagonal, which catches padding applied on one axis only -- a mistake that
    // leaves the flanks perfect and the corners sheared.
    let corner = surface.at(GLOW_RIGHT_EDGE as u32 + 3, 22 + 20 + 3)[3];
    assert!(
        corner > 0.0,
        "the halo has no ink diagonally past its corner: the quad grew in one axis only"
    );

    // The shader itself, not only the transcription. The suite's transcription is checked
    // against `tiny-skia` everywhere else; here there is no CPU side to disagree with it, so
    // the WGSL has to be read directly or a correct transcription of a wrong shader passes.
    let source = include_str!("shaders/instance.wgsl");
    let vs_main = source
        .split_once("fn vs_main")
        .and_then(|(_, rest)| rest.split_once("fn sd_rounded_box"))
        .map(|(body, _)| body)
        .unwrap_or_default();
    assert!(
        vs_main.contains("KIND_GLOW") && vs_main.contains("inst.param"),
        "vs_main does not vary its quad padding with a glow's falloff; the transcription \
         above is describing a shader that no longer exists"
    );
}

#[test]
fn the_halo_fades_without_drifting_through_black() {
    // Why the glow mixes in premultiplied linear and the gradient does not.
    //
    // `unpremultiply` returns black for a zero-alpha stop, because a colour that is not
    // there has no hue to recover. Walking the halo's ramp in Oklab -- the obvious thing to
    // do, given the ramp function is right there -- would therefore fade every glow through
    // black and leave a dark rim just inside the falloff, on exactly the surfaces a glow is
    // for. Premultiplied, the same colour at decreasing alpha is bit-for-bit the same
    // colour, which is what this measures.
    let instance = fixture_glow(Srgba::new(0.2, 0.6, 1.0, 0.5));
    let surface = reference_pixels(instance);
    // Against the *packed* near stop, not against the `Srgba` it came from. The instance
    // buffer holds premultiplied linear RGBA8, so a translucent colour loses a bit on the
    // way in; measuring from the source would be measuring that quantisation, which has
    // nothing to do with whether the ramp drifts.
    let near = shader::unpremultiply(shader::unpack4x8unorm(instance.color));
    let expected = near[0];

    for step in 1..7 {
        let px = GLOW_RIGHT_EDGE as u32 + step;
        let p = surface.at(px, 32);
        assert!(p[3] > 0.01, "no ink to measure the hue of at +{step}px");
        let straight_r = p[0] / p[3];
        assert!(
            (straight_r - expected).abs() < 1e-5,
            "at +{step}px the halo's red channel unpremultiplied to {straight_r:.5} against \
             the {expected:.5} it started at -- the fade is dragging the colour somewhere"
        );
    }
}

#[test]
fn a_glow_with_no_falloff_is_the_shape_and_nothing_around_it() {
    // `glow_t` divides by the falloff, so zero is the input that has to be answered rather
    // than computed. The answer is the plain fill: solid inside, nothing outside. Returning
    // NaN would paint the whole padded quad, and returning 1 everywhere would paint nothing
    // at all -- both are silent, because a glow that renders wrong still renders.
    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    let surface = reference_pixels(Instance::glow(20.0, 22.0, 24.0, 20.0, 6.0, 0.0, white));

    assert!(
        (surface.at(32, 32)[3] - 1.0).abs() < 1e-3,
        "the shape vanished"
    );
    assert_eq!(
        surface.at(GLOW_RIGHT_EDGE as u32 + 1, 32)[3],
        0.0,
        "a glow with no falloff put ink outside its shape"
    );
}

// -- the rim's own geometry ----------------------------------------------------------------
//
// The mirror of the glow's four tests above, and there for the same reason: the floor check
// says the CPU tier draws nothing, and nothing else says the GPU side draws anything. A rim
// branch that returned zero would satisfy every floor assertion in the file.

/// The fixture rim uses the glow fixture's box, so a horizontal scan out of its right flank
/// at y = 32 meets no corner and the distance is just the gap to that edge.
const RIM_RIGHT_EDGE: f32 = 44.0;
const RIM_WIDTH: f32 = 6.0;

fn fixture_rim(color: Srgba) -> Instance {
    Instance::rim(20.0, 22.0, 24.0, 20.0, 6.0, RIM_WIDTH, color)
}

#[test]
fn rim_stays_at_its_floor_on_the_cpu_tier() {
    run_kind(PrimKind::Rim);
}

#[test]
fn a_rim_is_brightest_inside_its_own_edge_and_gone_by_its_width() {
    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    let surface = reference_pixels(fixture_rim(white));

    // Nothing outside the shape. This is what separates a rim from a glow, and it is the
    // first thing a sign error breaks: `distance / width` instead of `-distance / width`
    // lights the outside and leaves the inside dark, which still renders a picture.
    assert_eq!(
        surface.at(RIM_RIGHT_EDGE as u32, 32)[3],
        0.0,
        "the rim put ink outside its own shape"
    );

    // And nothing in the middle. A rim that filled its shape would be a fill with extra
    // steps, which is the other way the sign can be wrong.
    assert_eq!(
        surface.at(32, 32)[3],
        0.0,
        "the rim laid ink at the centre of its shape, so it is a fill and not an edge light"
    );

    // The profile, one pixel at a time, against the curve rather than against a recorded
    // array -- so a rim that changed shape but stayed monotonic fails here even though a
    // monotonicity check would pass it.
    let mut previous = f32::INFINITY;
    for step in 1..=8u32 {
        let px = RIM_RIGHT_EDGE as u32 - step;
        // Negative: inside the shape.
        let distance = (px as f32 + 0.5) - RIM_RIGHT_EDGE;
        let fade = 1.0 - (-distance / RIM_WIDTH).clamp(0.0, 1.0);
        let expected = (0.5 - distance).clamp(0.0, 1.0) * fade * fade;
        let measured = surface.at(px, 32)[3];
        assert!(
            (measured - expected).abs() < 2.0 / 255.0,
            "at {distance:.1}px inside the shape the rim measured {measured:.4}, not the \
             {expected:.4} a quadratic ramp over {RIM_WIDTH}px predicts"
        );
        assert!(
            measured <= previous + 1e-6,
            "the rim brightened at {distance:.1}px in -- the ramp has to fall away from the \
             edge or there is a visible band inside it"
        );
        previous = measured;
    }

    // The peak sits half a pixel inside the boundary rather than on it, because at the
    // boundary the shape itself is only half covered. A rim brighter *at* the edge than just
    // inside it would be a rim drawing outside its shape.
    let peak = surface.at(RIM_RIGHT_EDGE as u32 - 1, 32)[3];
    assert!(
        peak > 0.75,
        "the brightest pixel of the rim measured {peak:.4}: the light is not reaching the \
         edge it is supposed to be lighting"
    );

    // Genuinely gone by its own width, rather than merely faint. A ramp that stops at a
    // non-zero value has an inner edge, and an inner edge is a second line nobody asked for.
    assert_eq!(
        surface.at(RIM_RIGHT_EDGE as u32 - RIM_WIDTH as u32 - 1, 32)[3],
        0.0,
        "the rim still had ink past its own width"
    );
}

#[test]
fn a_rim_with_no_width_is_nothing_rather_than_the_whole_shape() {
    // `rim_t` divides by the width, so zero is the input that has to be answered rather than
    // computed -- and the two silent wrong answers are opposites. Returning 0 there would
    // make `fade` 1 and paint the entire shape solid; NaN would paint whichever fragments the
    // comparison happened to admit. The answer is nothing at all, because a light with no
    // depth is not a light.
    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    let surface = reference_pixels(Instance::rim(20.0, 22.0, 24.0, 20.0, 6.0, 0.0, white));

    let ink: f32 = surface.pixels.iter().map(|p| p[3]).sum();
    assert_eq!(
        ink, 0.0,
        "a rim with no width laid down ink somewhere: {ink} of it"
    );
}

#[test]
fn the_rim_fades_without_drifting_through_black() {
    // Why the rim carries its whole ramp in the coverage and leaves the tint alone.
    //
    // The obvious thing to do, with `ramp` sitting right there, is to walk from the rim's
    // colour to a transparent one in Oklab. `unpremultiply` returns black for a zero-alpha
    // stop, so that would fade every rim through black and leave a dark line just inside the
    // bright one -- on exactly the surfaces a rim is for. Premultiplied, the same colour at
    // decreasing alpha is bit-for-bit the same colour, which is what this measures.
    let instance = fixture_rim(Srgba::new(0.85, 0.92, 1.0, 0.6));
    let surface = reference_pixels(instance);
    // Against the *packed* colour, not the `Srgba` it came from: the instance buffer holds
    // premultiplied linear RGBA8, so measuring from the source would be measuring that
    // quantisation instead of whether the ramp drifts.
    let expected = shader::unpremultiply(shader::unpack4x8unorm(instance.color))[0];

    for step in 1..=5u32 {
        let px = RIM_RIGHT_EDGE as u32 - step;
        let p = surface.at(px, 32);
        assert!(p[3] > 0.01, "no ink to measure the hue of at -{step}px");
        let straight_r = p[0] / p[3];
        assert!(
            (straight_r - expected).abs() < 1e-5,
            "at -{step}px the rim's red channel unpremultiplied to {straight_r:.5} against \
             the {expected:.5} it started at -- the fade is dragging the colour somewhere"
        );
    }
}

#[test]
fn the_shader_lights_the_inside_of_an_edge_and_not_only_the_transcription() {
    // The suite's transcription is checked against `tiny-skia` everywhere else. A rim has no
    // CPU side to disagree with it, so the WGSL is read directly -- otherwise a correct
    // transcription of a shader that never grew the branch passes everything above.
    let source = include_str!("shaders/instance.wgsl");
    let fs_main = source
        .split_once("fn fs_main")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    assert!(
        fs_main.contains("KIND_RIM") && fs_main.contains("rim_t"),
        "fs_main has no rim branch; the transcription above is describing a shader that \
         does not exist"
    );
    assert!(
        source.contains("fn rim_t") && source.contains("-distance / width"),
        "the shader has no inward-reading ramp, so whatever fs_main is branching on is not \
         a rim"
    );
}

// -- the surface's own physics ------------------------------------------------------------
//
// The floor check says the CPU tier draws the albedo. Nothing in it says the GPU side shades
// anything, and a BRDF that returned its input would satisfy every assertion above. These
// tests are that half, and they are about the *physics* rather than about pixels: a wrong
// normal, a non-conserving diffuse lobe or a metal with a diffuse term all still render.

const PBR_BEVEL: f32 = 6.0;

fn fixture_pbr(roughness: f32, metallic: f32, albedo: Srgba) -> Instance {
    Instance::pbr(
        20.0, 22.0, 24.0, 20.0, 6.0, PBR_BEVEL, roughness, metallic, 1.0, 0.0, albedo,
    )
}

/// Where the emissive fixture sits: `x0, y0, x1, y1`.
///
/// Deliberately much larger than [`fixture_pbr`]'s 24x20. The interior more than a bevel from
/// every edge is what `emission_never_reaches_the_middle_of_a_surface` measures, and on the
/// smaller rectangle that interior is 60 pixels -- which the test refuses as too few to be a
/// measurement. Widening the fixture is the honest fix; loosening that guard would have been
/// the other one.
#[test]
fn pbr_stays_at_its_floor_on_the_cpu_tier() {
    run_kind(PrimKind::Pbr);
}

#[test]
fn a_lamp_lights_its_whole_face_and_never_past_the_peak_the_gate_checks() {
    // THE INVARIANT THAT REPLACED `emission_never_reaches_the_middle_of_a_surface`, and the
    // replacement is a deliberate trade rather than a relaxation.
    //
    // The old test asserted emission was **bit-identically zero** more than `bevel` inside
    // any edge — a geometric guarantee that the ground under a label never moves, which let
    // a material turn emission up without touching a single composite the contrast gate
    // could see. That bought safety by forbidding the thing the design now wants: a row that
    // reads as a lit panel rather than an outlined one.
    //
    // What replaces it is not "nothing". The lamp lights its whole face, so the ground under
    // the label DOES move — and the gate now follows it there: `Material::lit_composites`
    // adds the emission's closed-form peak and `material_results_policy` checks the lit ink
    // against it, in both themes. That check is only as honest as this bound, so this is the
    // test that keeps `Material::LAMP_PEAK` an over-estimate of what the shader can do.
    //
    // Two claims, and both matter:
    //   1. the profile never exceeds LAMP_PEAK anywhere, so the gate's number bounds reality;
    //   2. it is genuinely non-zero in the middle, or the lamp is not a lamp and the whole
    //      trade bought nothing.
    let bevel = PBR_BEVEL;
    let half = [120.0_f32, 21.0_f32];
    let mut peak = 0.0_f32;
    let mut middle_min = f32::INFINITY;
    for iy in -40..=40 {
        for ix in -220..=220 {
            let local = [ix as f32 * 0.5, iy as f32 * 0.5];
            if local[0].abs() > half[0] || local[1].abs() > half[1] {
                continue;
            }
            let distance = shader::fs_distance(local, half, 6.0);
            let v = shader::lamp_emission(distance, bevel, local, half);
            peak = peak.max(v);
            // "The middle" is the band a label sits in: the centre half of the height,
            // clear of the housing on every side.
            if local[1].abs() < half[1] * 0.5 && local[0].abs() < half[0] - bevel * 2.0 {
                middle_min = middle_min.min(v);
            }
        }
    }
    assert!(
        peak <= qs_ui_lamp_peak(),
        "the lamp profile peaks at {peak}, above the {} the contrast gate bounds it by —          every lit contrast result is now optimistic by that factor",
        qs_ui_lamp_peak()
    );
    assert!(
        middle_min > 0.2,
        "the lamp is dark in the middle at {middle_min}: it is an outlined row wearing a          lamp's name, and the contrast trade that allowed it bought nothing"
    );
}

/// `qs_ui::material::Material::LAMP_PEAK`, restated here because `qs-gpu` sits **below**
/// `qs-ui` and cannot import it. The duplication is the reason this test exists: it is the
/// one place the two numbers are compared, so they cannot drift apart silently.
fn qs_ui_lamp_peak() -> f32 {
    1.30
}

#[test]
fn emission_is_brightest_at_the_boundary_and_gone_by_the_bevel() {
    // The profile, sampled rather than assumed. A surface whose emission fell off linearly, or
    // reached past the bevel, would still satisfy the equality test above at a large enough
    // inset while lighting ground it must not.
    let weights: Vec<f32> = (0..=12)
        .map(|i| {
            // Distance from the boundary, inward: the SDF is negative inside.
            let inward = PBR_BEVEL * i as f32 / 12.0;
            shader::edge_emission(-inward, PBR_BEVEL)
        })
        .collect();

    assert!(
        (weights[0] - 1.0).abs() <= 1e-6,
        "emission is not full at the boundary: {}",
        weights[0]
    );
    assert!(
        weights[12].abs() <= 1e-6,
        "emission has not reached zero by the bevel: {}",
        weights[12]
    );
    for pair in weights.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1e-6,
            "emission rose on the way inward: {pair:?}"
        );
    }
    // Past the bevel it stays zero rather than going negative or wrapping.
    for multiple in [1.0_f32, 1.5, 4.0, 40.0] {
        let past = shader::edge_emission(-PBR_BEVEL * multiple, PBR_BEVEL);
        assert_eq!(past, 0.0, "emission at {multiple}x the bevel was {past}");
    }
    // And a surface with no bevel has no edge to emit from.
    assert_eq!(shader::edge_emission(-1.0, 0.0), 0.0);
}

#[test]
fn the_shader_and_the_renderer_agree_about_where_the_light_is() {
    // `LIGHT_DIR` used to be a WGSL constant and nothing else. It is now also a Rust constant,
    // because a material can cast a contact shadow and the direction a shadow falls has to be
    // the same vector the surface is shaded from -- a window whose shadows point one way and
    // whose highlights point the other is the defect this comparison exists to prevent.
    //
    // Compared as text, for the reason the dither constants are: `instance.wgsl` is parsed by
    // `cargo test` and never executed, so no fixture can catch the two drifting apart.
    let source = include_str!("shaders/instance.wgsl");
    let declared = source
        .lines()
        .find_map(|line| {
            let rest = line.trim().strip_prefix("const LIGHT_DIR: vec3<f32> =")?;
            let inner = rest.trim().strip_prefix("vec3<f32>(")?;
            let inner = inner.split(')').next()?;
            let mut parts = inner.split(',').map(|p| p.trim().parse::<f32>().ok());
            Some([parts.next()??, parts.next()??, parts.next()??])
        })
        .expect("LIGHT_DIR is not declared in instance.wgsl in the form this parses");

    assert_eq!(
        declared,
        crate::frame::LIGHT_DIR,
        "the shader lights from {declared:?} and qs_gpu::frame says {:?}",
        crate::frame::LIGHT_DIR
    );

    // And the shadow direction really is derived from it rather than being a second opinion:
    // down and to the right, unit length.
    let shadow = crate::frame::shadow_direction();
    assert!(
        shadow[0] > 0.0 && shadow[1] > 0.0,
        "the light is above and left, so shadows fall down and right; got {shadow:?}"
    );
    assert!(
        (shadow[0].hypot(shadow[1]) - 1.0).abs() <= 1e-6,
        "the shadow direction is not a unit vector: {shadow:?}"
    );
    let expected = [-declared[0], -declared[1]];
    let length = expected[0].hypot(expected[1]);
    assert!(
        (shadow[0] - expected[0] / length).abs() <= 1e-6
            && (shadow[1] - expected[1] / length).abs() <= 1e-6,
        "the shadow is not the key light's own direction negated: {shadow:?}"
    );
}

#[test]
fn the_bevel_normal_turns_from_facing_the_viewer_to_facing_out_along_the_edge() {
    // The geometric claim the whole BRDF rests on. A normal that stayed (0,0,1) would light
    // the surface flatly and still produce a picture; one whose sign was inverted would light
    // it from the wrong side and also still produce a picture.
    let half = [12.0f32, 10.0];

    // Deep inside: flat, facing the viewer.
    let deep = shader::bevel_normal(-9.0, [1.0, 0.0], PBR_BEVEL);
    assert!(
        (deep[2] - 1.0).abs() < 1e-5 && deep[0].abs() < 1e-5,
        "the middle of the surface is not flat: {deep:?}"
    );

    // At the boundary: vertical, facing out along the gradient.
    let edge = shader::bevel_normal(0.0, [1.0, 0.0], PBR_BEVEL);
    assert!(
        (edge[0] - 1.0).abs() < 1e-5 && edge[2].abs() < 1e-5,
        "the edge does not face outward: {edge:?}"
    );

    // Every sample is unit length, or the BRDF's cosines are not cosines.
    for step in 0..=12 {
        let d = -(step as f32) * 0.5;
        let g = shader::sd_rounded_box_grad([11.0, 0.0], half, 6.0);
        let n = shader::bevel_normal(d, g, PBR_BEVEL);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "normal at d={d} has length {len}");
    }

    // The gradient points away from the nearest edge, on the flanks and in a corner.
    assert_eq!(
        shader::sd_rounded_box_grad([11.0, 0.0], half, 6.0),
        [1.0, 0.0]
    );
    assert_eq!(
        shader::sd_rounded_box_grad([-11.0, 0.0], half, 6.0),
        [-1.0, 0.0]
    );
    assert_eq!(
        shader::sd_rounded_box_grad([0.0, 9.0], half, 6.0),
        [0.0, 1.0]
    );
    let corner = shader::sd_rounded_box_grad([11.0, 9.0], half, 6.0);
    assert!(
        corner[0] > 0.0 && corner[1] > 0.0,
        "the corner gradient does not point out of the corner: {corner:?}"
    );
}

#[test]
fn the_brdf_conserves_energy_and_a_metal_has_no_diffuse_lobe() {
    // Two claims a plausible-looking BRDF gets wrong silently.
    //
    // The first: at the flat normal -- the diffuse-dominated case, and the one every surface
    // in this application is mostly made of -- a surface returns no more light than fell on
    // it. A separable Smith term, a missing `kd`, or a `D` without its normalisation all break
    // this and all still look like shading.
    //
    // Deliberately not asserted at the specular peak, where a near-mirror exceeds one and is
    // *right* to: a mirror shows a light source brighter than any surface, and a test that
    // forbade it would be asserting an artefact rather than energy conservation. The honest
    // strong form is a hemispherical integral of the BRDF, which is a bigger unit than this.
    let dark = [0.0f32; 3];
    let flat = [0.0f32, 0.0, 1.0];
    for rough in [0.05f32, 0.2, 0.5, 0.8, 1.0] {
        for metal in [0.0f32, 1.0] {
            let out = shader::shade_pbr(flat, [1.0, 1.0, 1.0], rough, metal, 0.0, 0.0, dark, dark);
            for c in out {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "roughness {rough} metal {metal} returned {c}, which is more light than \
                     the one light in the scene emitted"
                );
            }
        }
    }

    // The second: a metal has no diffuse lobe. Measured off the specular direction, because
    // facing the highlight a metal is *brighter* than a dielectric -- its Fresnel is its
    // albedo rather than 0.04 -- and comparing there would assert the opposite of the physics.
    // This normal points down-right while the key light comes from up-left, so the specular
    // lobe is nearly gone and what is left is the diffuse term that only the dielectric has.
    let off_specular = [0.4511, 0.5513, 0.7017];
    let metal = shader::shade_pbr(
        off_specular,
        [0.8, 0.1, 0.1],
        0.6,
        1.0,
        0.0,
        0.0,
        dark,
        dark,
    );
    let dielectric = shader::shade_pbr(
        off_specular,
        [0.8, 0.1, 0.1],
        0.6,
        0.0,
        0.0,
        0.0,
        dark,
        dark,
    );
    assert!(
        metal[0] < dielectric[0] * 0.35,
        "metal {metal:?} kept a diffuse lobe against dielectric {dielectric:?}"
    );
    assert!(
        dielectric[0] > 0.01,
        "the dielectric returned nothing to compare against"
    );

    // And facing the highlight the order reverses, which is the other half of the same fact
    // and the one that stops the assertion above from being satisfied by a shader that simply
    // darkens everything it is told is metal.
    let lit_metal = shader::shade_pbr(flat, [0.8, 0.1, 0.1], 0.6, 1.0, 0.0, 0.0, dark, dark);
    let lit_dielectric = shader::shade_pbr(flat, [0.8, 0.1, 0.1], 0.6, 0.0, 0.0, 0.0, dark, dark);
    assert!(
        lit_metal[0] > lit_dielectric[0] * 0.6,
        "metal lost its tinted specular as well as its diffuse lobe: {lit_metal:?} against          {lit_dielectric:?}"
    );
}

#[test]
fn roughness_widens_the_highlight_rather_than_only_dimming_it() {
    // The property that distinguishes a microfacet distribution from a fudge factor: a
    // rougher surface spreads the same energy over a wider lobe. A shader that multiplied the
    // highlight by `1 - roughness` would dim it correctly and never widen it, and every still
    // image of a single surface would look plausible.
    let dark = [0.0f32; 3];
    let at = |angle: f32, rough: f32| -> f32 {
        let n = [angle.sin(), 0.0, angle.cos()];
        shader::shade_pbr(n, [0.5, 0.5, 0.5], rough, 1.0, 0.0, 0.0, dark, dark)[0]
    };
    // Sharpness measured as the ratio between the lobe's centre and its shoulder.
    let sharp = at(0.0, 0.1) / at(0.45, 0.1).max(1e-6);
    let broad = at(0.0, 0.6) / at(0.45, 0.6).max(1e-6);
    assert!(
        sharp > broad * 2.0,
        "the highlight did not narrow as roughness fell: sharp {sharp:.3}, broad {broad:.3}"
    );
}

#[test]
fn the_environment_is_a_reflection_and_not_a_constant_added_on() {
    // A sky the surface genuinely reflects moves when the sky moves, and moves *differently*
    // at different normals -- which is what separates an environment term from an ambient
    // constant, the thing it is most often quietly replaced by.
    let dim = [0.05f32, 0.05, 0.05];
    let bright = [0.9f32, 0.9, 0.9];
    let flat = [0.0f32, 0.0, 1.0];
    let tilted = [0.6f32, 0.0, 0.8];

    let flat_dim = shader::shade_pbr(flat, [0.5; 3], 0.15, 1.0, 1.0, 0.0, dim, dim)[0];
    let flat_bright = shader::shade_pbr(flat, [0.5; 3], 0.15, 1.0, 1.0, 0.0, dim, bright)[0];
    assert!(
        flat_bright > flat_dim + 0.05,
        "raising the zenith did not reach a surface facing it"
    );

    // A tilted surface reflects toward the horizon, so the same zenith change reaches it less.
    let tilt_dim = shader::shade_pbr(tilted, [0.5; 3], 0.15, 1.0, 1.0, 0.0, dim, dim)[0];
    let tilt_bright = shader::shade_pbr(tilted, [0.5; 3], 0.15, 1.0, 1.0, 0.0, dim, bright)[0];
    assert!(
        (flat_bright - flat_dim) > (tilt_bright - tilt_dim),
        "the sky reached both normals equally, so it is being added rather than reflected"
    );
}

#[test]
fn a_lit_surface_covers_exactly_the_pixels_its_floor_would() {
    // Why the floor can be `Plain(Rect)` at all. Shading replaces what is inside a shape and
    // must not touch which pixels the shape covers -- if it did, the fallback tier would draw
    // a differently shaped object and 10.7's "plainer, never inconsistent" would be broken.
    let albedo = Srgba::new(0.6, 0.7, 0.9, 1.0);
    let lit = reference_pixels(fixture_pbr(0.3, 0.0, albedo));
    let unlit = reference_pixels(Instance::rect(20.0, 22.0, 24.0, 20.0, 6.0, albedo));
    for y in 0..SURFACE {
        for x in 0..SURFACE {
            let a = lit.at(x, y)[3];
            let b = unlit.at(x, y)[3];
            assert!(
                (a - b).abs() < 1e-5,
                "coverage differs at ({x}, {y}): lit {a}, unlit {b}"
            );
        }
    }
}

#[test]
fn the_shader_shades_a_surface_and_not_only_the_transcription() {
    // The transcription above is checked against `tiny-skia` nowhere, because a floored PBR
    // instance never reaches it. So the WGSL is read directly, or a faithful transcription of
    // a shader that never grew the branch passes everything here.
    let source = include_str!("shaders/instance.wgsl");
    for needle in [
        "fn sd_rounded_box_grad",
        "fn bevel_normal",
        "fn distribution_ggx",
        "fn visibility_smith",
        "fn fresnel_schlick",
        "fn environment",
        "KIND_PBR",
    ] {
        assert!(
            source.contains(needle),
            "instance.wgsl has no `{needle}`; the transcription describes a shader that does \
             not exist"
        );
    }
    // The call form, not the word: the shader says "no `fwidth`" in three comments explaining
    // why there isn't one, and a check that matched those would be red on a correct file.
    assert!(
        !source.contains("fwidth(") && !source.contains("dpdx(") && !source.contains("dpdy("),
        "a derivative reached the shader, which the CPU tier cannot reproduce and the whole \
         analytic-normal argument exists to avoid"
    );
}

#[test]
fn no_prim_kind_can_ship_without_a_parity_test() {
    // The compile-time half of this guard is `cases_for`'s exhaustive match. This is the
    // run-time half: an arm that exists but returns nothing, or returns fixtures for the
    // wrong primitive, would satisfy the compiler and prove nothing.
    //
    // Fidelity does not enter into it. An `Enhanced` kind is checked by a different
    // question, not by a shorter list, and the reading it has to be refused is that
    // declaring a floor is a way to stop owing a fixture.
    for kind in PrimKind::ALL {
        let cases = cases_for(kind);
        assert!(
            !cases.is_empty(),
            "{kind:?} has an arm in cases_for but no fixtures: it would ship unchecked, \
             whatever fidelity class it declares"
        );
        for case in &cases {
            assert_eq!(
                case.instance.kind, kind as u32,
                "{}: filed under {kind:?} but draws a different primitive",
                case.name
            );
        }
    }
}

#[test]
fn the_shader_and_prim_kind_declare_the_same_primitives() {
    // Closes the one hole `cases_for`'s exhaustive match cannot see: a variant added to
    // the enum, given a shader branch, and left out of `PrimKind::ALL`. The match would
    // still compile (the loop above never visits the variant) and the suite would still
    // be green while a whole primitive went unchecked. Comparing against the shader's own
    // declarations catches it, because a primitive that renders must have a KIND_ there.
    let source = include_str!("shaders/instance.wgsl");
    let mut declared: Vec<(String, u32)> = source
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("const KIND_")?;
            let (name, value) = rest.split_once('=')?;
            let name = format!("KIND_{}", name.split(':').next()?.trim());
            let value = value.trim().trim_end_matches(';').trim_end_matches('u');
            Some((name, value.trim().parse().ok()?))
        })
        .collect();
    declared.sort();

    let mut known: Vec<(String, u32)> = PrimKind::ALL
        .iter()
        .map(|k| (k.shader_const().to_owned(), *k as u32))
        .collect();
    known.sort();

    assert_eq!(
        declared, known,
        "shaders/instance.wgsl and PrimKind::ALL declare different primitives; whichever \
         side gained one, the parity suite is not covering it"
    );
    assert!(!declared.is_empty(), "the KIND_ parser matched nothing");
}

#[test]
fn the_shader_and_the_palette_dither_by_the_same_numbers() {
    // The dither is the one piece of arithmetic that has to be *identical* on the two
    // tiers rather than merely close: it decides which of two adjacent output values a
    // pixel takes, so a constant that drifts by a per cent does not shift the picture by a
    // per cent -- it picks the other value on a scattering of pixels, everywhere, and the
    // gradient fixtures would go red with no clue as to why.
    //
    // Every other transcription in this file is checked by the parity fixtures running the
    // same maths. These constants cannot be, because `instance.wgsl` is never executed by
    // `cargo test` -- only parsed. So they are compared as text.
    let source = include_str!("shaders/instance.wgsl");
    let declared = |name: &str| -> f32 {
        source
            .lines()
            .find_map(|line| {
                let rest = line.trim().strip_prefix(&format!("const {name}: f32 ="))?;
                rest.trim().trim_end_matches(';').trim().parse().ok()
            })
            .unwrap_or_else(|| panic!("{name} is not declared in instance.wgsl"))
    };

    for (name, ours) in [
        ("DITHER_SLOPE", crate::color::DITHER_SLOPE),
        ("DITHER_EXPONENT", crate::color::DITHER_EXPONENT),
        ("DITHER_TOE", crate::color::DITHER_TOE),
        ("DITHER_TOE_SLOPE", crate::color::DITHER_TOE_SLOPE),
    ] {
        let theirs = declared(name);
        assert!(
            (theirs - ours).abs() <= ours.abs() * 1e-6,
            "{name}: the shader says {theirs}, qs_gpu::color says {ours}"
        );
    }

    // The mixer's own constants, which have no name on either side. Comparing them as
    // literals is crude and is the point: there is nothing else holding the two hashes
    // together, and a hash that differs by one constant is a hash that agrees nowhere.
    for word in ["0x27d4eb2du", "0x165667b1u", "0x2c1b3c6du", "0x297a2d39u"] {
        assert!(
            source.contains(word),
            "instance.wgsl no longer mixes with {word}; qs_gpu::color::dither_hash still does"
        );
    }
    for shift in ["h >> 15u", "h >> 12u"] {
        assert!(source.contains(shift), "the shader's mixer lost {shift}");
    }
}

#[test]
fn the_shader_compiles_and_validates() {
    // Until this existed, nothing in `cargo test` had ever compiled `instance.wgsl`. The
    // whole suite above transcribes the shader into Rust and checks the transcription
    // against `tiny-skia`, which is a strong check on the *maths* and no check at all on
    // whether the file is valid WGSL -- a missing semicolon would leave every test green
    // and every GPU tier rendering nothing, because the only thing that had ever parsed
    // it was `create_shader_module` at device startup, which no test reaches.
    //
    // `naga` is a dev-dependency at `wgpu`'s own major version, so this is the same front
    // end the device uses, minus the device.
    let source = include_str!("shaders/instance.wgsl");
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!(
            "instance.wgsl does not parse:\n{}",
            e.emit_to_string(source)
        )
    });

    // Parsing is not enough on its own: a type error, a bad swizzle or an entry point with
    // the wrong signature all parse and then fail validation, which is where a driver
    // would reject them.
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("instance.wgsl does not validate:\n{e:?}"));

    // Guard the guard: an `include_str!` pointed at the wrong file, or a shader that lost
    // its entry points, would validate perfectly and prove nothing.
    let entry_points: Vec<&str> = module
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(
        entry_points,
        ["vs_main", "fs_main"],
        "the pipeline binds these two by name"
    );
}

#[test]
fn a_glyph_quad_at_a_fractional_origin_does_not_agree_across_tiers() {
    // Not a parity case -- a pinned honest negative.
    //
    // The GPU samples the atlas bilinearly; the CPU tier rounds the quad's origin and
    // blits 1:1. On an integer-aligned quad those coincide exactly, which is why
    // `glyph/integer-origin` is allowed a bound of one. Move the quad half a pixel and
    // they do not: the GPU blurs across two texels and the CPU snaps to one.
    //
    // The pipeline never emits such a quad -- the fractional pen position is baked into
    // the subpixel variant, so the geometry is always integral. That guarantee currently
    // lives in a comment in `batcher.rs`. This test is the receipt for what it is buying,
    // so that if a future change starts emitting fractional glyph origins, the cost is
    // already written down and measured instead of being discovered as blurry text.
    let case = Case {
        name: "glyph/fractional-origin",
        instance: Instance::glyph(
            10.5,
            12.5,
            9.0,
            11.0,
            uv_for(3, 5, 9, 11),
            Srgba::new(1.0, 1.0, 1.0, 1.0),
        ),
        uploads: vec![PendingUpload {
            x: 3,
            y: 5,
            width: 9,
            height: 11,
            coverage: glyph_coverage(9, 11),
        }],
        max_channel: 255,
        mean_channel: 255.0,
        centroid_shift: 255.0,
        ink_area: 1.0,
    };
    let d = run_case(&case);
    eprintln!(
        "glyph/fractional-origin: max={} mean={:.4} centroid={:.4}",
        d.max_channel, d.mean_channel, d.centroid_shift
    );
    assert!(
        d.max_channel > 32,
        "a half-pixel glyph offset was expected to diverge between the tiers, but the \
         worst channel differed by only {}. Either the sampler stopped filtering or the \
         CPU tier stopped rounding -- in both cases `glyph/integer-origin`'s bound of one \
         is no longer buying what this test says it buys",
        d.max_channel
    );
}

#[test]
fn the_reference_reproduces_the_stroke_alignment_the_cpu_tier_was_fixed_to_match() {
    // Ties the new machinery back to the bug that motivated it. `stroke/sharp` asserts
    // the two tiers agree; this asserts they agree on the *right* answer, which a pair of
    // identically-wrong implementations would also satisfy.
    let mut list = DrawList::default();
    list.reset([40, 40], Srgba::TRANSPARENT, 1);
    list.instances.push(Instance::stroke(
        10.0,
        10.0,
        20.0,
        20.0,
        0.0,
        2.0,
        Srgba::new(1.0, 1.0, 1.0, 1.0),
    ));
    list.end_batch(None, false);

    let reference = render_reference(&list, &[], 0);
    let row: Vec<u32> = (0..40)
        .filter(|x| quantise(reference.at(*x, 20)[3]) > INK_FLOOR)
        .collect();
    assert_eq!(
        (row.first().copied(), row.last().copied()),
        (Some(10), Some(29)),
        "the shader's ring must sit inside the rect's bounds, touching both edges"
    );
}

// -- the focus lamp (US3, T063) ---------------------------------------------------------

/// A point the block's key-light shadow actually falls on.
///
/// Asserted rather than eyeballed by `the_shared_lamp_scene_really_does_cast_a_key_shadow`,
/// because every test below that reads it means something only if that shadow is real.
#[cfg(test)]
const KEY_SHADOWED: [f32; 2] = [252.0, 232.0];

/// The scene the lamp tests share: a canvas, one block raised high enough to be caught
/// throwing a key-light shadow, and the key rig.
///
/// Built once so the tests differ only in the lamp, which is what makes their comparisons
/// comparisons rather than several unrelated numbers.
#[cfg(test)]
fn lamp_scene() -> (Vec<crate::scene::Slab>, [f32; 3], f32) {
    use crate::scene::Slab;
    let slabs = vec![
        Slab {
            rect: [0.0, 0.0, 512.0, 512.0],
            elevation: 0.0,
            thickness: 1.0,
            attenuation_floor: 0.0,
            ..Slab::default()
        },
        Slab {
            rect: [200.0, 180.0, 80.0, 40.0],
            radius: 6.0,
            elevation: 20.0,
            thickness: 20.0,
            attenuation_floor: 0.0,
            ..Slab::default()
        },
    ];
    (slabs, [-0.32, -0.55, 0.77], crate::lighting::hardness(5.0))
}

/// A lamp of the shipped softness at `position`.
///
/// `ambient` defaults to zero so the shadow-mix tests below isolate `share` -- the two halves
/// of the lamp do different work and a test that moved both at once could not say which one
/// its number came from. `lamp_with_ambient` is the other half.
#[cfg(test)]
fn lamp_at(position: [f32; 3], share: f32) -> shader::FocusLamp {
    // A short strip centred on `position`, so these tests read as they did when the lamp was a
    // point and their numbers stay comparable to the ones recorded beside them.
    shader::FocusLamp {
        rect: [position[0] - 20.0, position[1] - 14.0, 40.0, 28.0],
        height: position[2],
        hardness: crate::lighting::hardness(22.0),
        share,
        ambient: 0.0,
    }
}

/// A lamp whose ambient half is on, for the tests that are about the room rather than the
/// shadow direction.
#[cfg(test)]
fn lamp_with_ambient(position: [f32; 3], ambient: f32) -> shader::FocusLamp {
    shader::FocusLamp {
        ambient,
        ..lamp_at(position, 0.0)
    }
}

#[test]
fn the_shared_lamp_scene_really_does_cast_a_key_shadow() {
    // The anti-vacuity guard for everything below. Three of the four lamp tests are about
    // what happens to a pixel the key light has shadowed; if `KEY_SHADOWED` sat in the open
    // they would all pass while asserting nothing, and would keep passing with the lamp
    // deleted. Same discipline as `a_floored_effect_drops_something_that_was_actually_there`:
    // establish the subject exists before measuring what is done to it.
    let (slabs, toward, k) = lamp_scene();
    let shadowed = shader::lit_attenuation(KEY_SHADOWED, &slabs, toward, k, None);
    assert!(
        shadowed < 0.9,
        "KEY_SHADOWED is not in the block's shadow ({shadowed}), so every lamp test that \
         reads it is measuring an unshadowed pixel"
    );
}

#[test]
fn a_lamp_with_no_share_is_the_frame_that_shipped_before_it() {
    // The claim the whole mix rests on: with nothing focused the pass is arithmetically what
    // it was before US3, so turning the lit mode on without touching the keyboard cannot
    // change a pixel that was already checked. `to_bits`, not a tolerance -- "close enough"
    // here would let the identity drift by an ulp a release until something noticed.
    let (slabs, toward, k) = lamp_scene();
    let dark = lamp_at([300.0, 260.0, 40.0], 0.0);

    for i in 0..(64u32 * 64) {
        let p = [(i % 64) as f32 * 8.0, (i / 64) as f32 * 8.0];
        let absent = shader::lit_attenuation(p, &slabs, toward, k, None);
        let unlit = shader::lit_attenuation(p, &slabs, toward, k, Some(dark));
        assert_eq!(
            absent.to_bits(),
            unlit.to_bits(),
            "a lamp with no share changed pixel {p:?}: {absent} became {unlit}"
        );
    }
}

#[test]
fn the_lamp_lights_the_shadow_it_stands_in() {
    // US3's acceptance scenario 1, at one pixel: put the lamp where the key light is blocked
    // and the pixel gets brighter, because the lamp can see it even though the key light
    // cannot. That is the whole reason a second light is worth a second shadow ray.
    //
    // Note what is NOT claimed: that the lamp brightens everywhere. It casts its own shadows,
    // so a pixel the KEY light reaches and the lamp does not gets darker -- which is the
    // story's "the shadows across the whole window lean away from it", a feature rather than a
    // leak. The invariant that actually holds is the next test's.
    let (slabs, toward, k) = lamp_scene();
    let lamp = lamp_at([300.0, 260.0, 40.0], 0.45);

    let dark = shader::lit_attenuation(KEY_SHADOWED, &slabs, toward, k, None);
    let lit = shader::lit_attenuation(KEY_SHADOWED, &slabs, toward, k, Some(lamp));
    assert!(
        lit > dark + 1e-3,
        "the lamp stood in the key light's shadow and lit nothing: {dark} became {lit}"
    );
}

#[test]
fn the_mix_stays_inside_the_unit_range_and_inside_the_allowance() {
    // The safety property, and it is NOT "the lamp only brightens" -- that was the first
    // guess and this scene refutes it in a few dozen pixels. What a convex mix of two terms
    // in 0..=1 guarantees is that the result is in 0..=1 too: the lamp can never take a
    // surface past its unlit colour, and can never produce a pixel darker than the darker of
    // the two lights alone.
    //
    // That is exactly what the contrast gate needs. The gate's worst case is the allowance
    // floor, and a term bounded by two terms that are each already bounded cannot reach past
    // it -- which is why no gate literal moves for this feature. The floor clamp is asserted
    // directly as well, since it is the line that makes the closed form a bound rather than
    // a hope.
    //
    // Both counters are checked, and the second is the interesting one: a lamp that deepened
    // nothing anywhere would be a brightness wash rather than a light.
    let (slabs, toward, k) = lamp_scene();
    let lamp = lamp_at([300.0, 260.0, 40.0], 0.45);
    let floored: Vec<crate::scene::Slab> = slabs
        .iter()
        .map(|s| crate::scene::Slab {
            attenuation_floor: 0.55,
            ..*s
        })
        .collect();

    let (mut lifted, mut deepened) = (0u32, 0u32);
    for i in 0..(96u32 * 96) {
        let p = [(i % 96) as f32 * 5.0, (i / 96) as f32 * 5.0];
        let key_only = shader::lit_attenuation(p, &slabs, toward, k, None);
        let mixed = shader::lit_attenuation(p, &slabs, toward, k, Some(lamp));

        assert!(
            (0.0..=1.0).contains(&mixed),
            "the mix left the unit range at {p:?}: {mixed}"
        );
        if mixed > key_only + 1e-4 {
            lifted += 1;
        }
        if mixed < key_only - 1e-4 {
            deepened += 1;
        }
        assert!(
            shader::lit_attenuation(p, &floored, toward, k, Some(lamp)) >= 0.55,
            "the lamp reached past the allowance floor at {p:?}"
        );
    }

    assert!(
        lifted > 0,
        "the lamp lifted nothing anywhere, so this test asserts a bound over a no-op"
    );
    assert!(
        deepened > 0,
        "the lamp deepened nothing anywhere. That is not a pass -- a second light that casts \
         no shadow of its own is a brightness wash, and this whole feature's premise is that \
         it is a light. If this fires, the lamp stopped being occluded."
    );
}

#[test]
fn the_lamps_influence_ends_and_the_key_light_owns_the_far_side() {
    // FOCUS_REACH's purpose. Without the falloff one focused row would relight the far corner
    // of the window, and the shadows there would stop agreeing with the key light -- two
    // sources disagreeing about where light comes from, which is what one fixed key direction
    // exists to prevent. Measured at the SAME pixel against two lamp positions, so the only
    // variable is distance; moving the pixel instead would compare two bits of geometry.
    let (slabs, toward, k) = lamp_scene();
    let key_only = shader::lit_attenuation(KEY_SHADOWED, &slabs, toward, k, None);
    let delta = |lamp: shader::FocusLamp| {
        (shader::lit_attenuation(KEY_SHADOWED, &slabs, toward, k, Some(lamp)) - key_only).abs()
    };

    let near = delta(lamp_at([300.0, 260.0, 40.0], 0.45));
    let far = delta(lamp_at(
        [300.0 + shader::FOCUS_REACH * 8.0, 260.0, 40.0],
        0.45,
    ));

    assert!(
        near > far,
        "the lamp reaches as far as it does near: near {near}, far {far}"
    );
    assert!(
        far < 0.01,
        "a lamp {} px away still moved the pixel by {far}",
        shader::FOCUS_REACH * 8.0
    );
}

#[test]
fn the_lamp_is_occluded_by_what_stands_between_it_and_the_surface() {
    // A light that reaches through geometry is not a light, it is a wash with a plausible
    // shape -- the finding d3bb539 recorded for bounce, asserted for the second caller of the
    // segment march. The control is the identical scene with the wall removed, because one
    // small number proves nothing about a sample that may simply have been out of reach.
    use crate::scene::Slab;
    let (mut slabs, toward, k) = lamp_scene();
    let lamp = lamp_at([300.0, 260.0, 40.0], 0.9);

    let open = shader::lit_attenuation(KEY_SHADOWED, &slabs, toward, k, Some(lamp));

    // Tall, thin, and directly between the lamp and KEY_SHADOWED.
    slabs.push(Slab {
        rect: [270.0, 240.0, 6.0, 30.0],
        elevation: 90.0,
        thickness: 90.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    });
    let blocked = shader::lit_attenuation(KEY_SHADOWED, &slabs, toward, k, Some(lamp));

    assert!(
        blocked < open,
        "the wall did not occlude the lamp: open {open}, blocked {blocked}"
    );
}

#[test]
fn the_room_is_brightest_under_the_lamp_and_dims_with_distance() {
    // The half of US3 a person actually sees, and the reason it exists at all.
    //
    // `focus_weight` alone was measured on the shipped list at 1200x700: peak 7/255, mean
    // 3/255 with focus moved eight rows. It is arithmetically correct and perceptually
    // absent, because it only changes a pixel where the two lights DISAGREE, and a list of
    // rows at one elevation gives them almost nothing to disagree about. This term is what
    // makes the story's "focus is where the light in the room is coming from" true.
    //
    // Asserted as a monotone ramp rather than at two points: a single near/far pair passes on
    // a step function, a ring, or a falloff with the sign flipped somewhere in the middle.
    //
    // On a FLAT scene, deliberately, and that is the claim rather than a convenience. The
    // shared `lamp_scene` has a raised block in it whose key-light shadow rises and falls
    // along any sample line, so a ramp measured there is measuring two things at once. A flat
    // list is also exactly the case this term exists for: it is where `focus_weight` has
    // nothing to do.
    use crate::scene::Slab;
    let slabs = vec![Slab {
        rect: [0.0, 0.0, 512.0, 640.0],
        elevation: 0.0,
        thickness: 1.0,
        attenuation_floor: 0.0,
        ..Slab::default()
    }];
    let (toward, k) = ([-0.32, -0.55, 0.77], crate::lighting::hardness(5.0));
    let lamp = lamp_with_ambient([256.0, 40.0, 40.0], 0.18);

    let at = |y: f32| shader::lit_attenuation([256.0, y], &slabs, toward, k, Some(lamp));
    let mut previous = at(40.0);
    for step in 1..=12 {
        let here = at(40.0 + step as f32 * 45.0);
        assert!(
            here <= previous + 1e-6,
            "the room got BRIGHTER further from the lamp at y {}: {previous} then {here}",
            40.0 + step as f32 * 45.0
        );
        previous = here;
    }
    assert!(
        at(40.0) - previous > 0.05,
        "the whole ramp from the lamp to the far edge is {}, which nobody will see",
        at(40.0) - previous
    );
}

#[test]
fn the_ambient_half_cannot_brighten_past_the_unlit_colour_or_darken_past_the_allowance() {
    // Both bounds, because the term is a multiply and a multiply gets both wrong at once if
    // its sign is off. Above: 1.0 is the unlit colour and no lamp may exceed it. Below: the
    // allowance floor is what the contrast gate computes its worst case at, so a term that
    // reached past it would make the gate's closed form stop being a bound — and the gate
    // would still be green, because it reads token values rather than frames.
    let (slabs, toward, k) = lamp_scene();
    let floored: Vec<crate::scene::Slab> = slabs
        .iter()
        .map(|s| crate::scene::Slab {
            attenuation_floor: 0.87,
            ..*s
        })
        .collect();
    // Deliberately far past anything authored: the bound has to hold for a hand-edited token
    // file too, and `FocusLightTokens::ambient` clamps for exactly this reason.
    let lamp = lamp_with_ambient([256.0, 40.0, 40.0], 1.0);

    for i in 0..(64u32 * 64) {
        let p = [(i % 64) as f32 * 8.0, (i / 64) as f32 * 8.0];
        let free = shader::lit_attenuation(p, &slabs, toward, k, Some(lamp));
        assert!(
            free <= 1.0,
            "the lamp took {p:?} past its unlit colour: {free}"
        );
        let held = shader::lit_attenuation(p, &floored, toward, k, Some(lamp));
        assert!(
            held >= 0.87,
            "the lamp reached past the light theme's text-ground floor at {p:?}: {held}"
        );
    }
}

#[test]
fn a_lamp_with_neither_strength_is_the_frame_that_shipped_before_it() {
    // The identity, restated over BOTH halves now that there are two. The earlier version of
    // this test set only `share` to zero and would have passed while a stray ambient dimmed
    // every frame in the product -- which is precisely the failure mode of adding a second
    // strength to a thing that used to have one.
    let (slabs, toward, k) = lamp_scene();
    let dark = shader::FocusLamp {
        share: 0.0,
        ambient: 0.0,
        ..lamp_at([300.0, 260.0, 40.0], 0.0)
    };

    for i in 0..(64u32 * 64) {
        let p = [(i % 64) as f32 * 8.0, (i / 64) as f32 * 8.0];
        let absent = shader::lit_attenuation(p, &slabs, toward, k, None);
        let unlit = shader::lit_attenuation(p, &slabs, toward, k, Some(dark));
        assert_eq!(
            absent.to_bits(),
            unlit.to_bits(),
            "a lamp with no strength at all changed pixel {p:?}: {absent} became {unlit}"
        );
    }
}
