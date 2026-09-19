//! Draw lists and the handoff between UI and render threads.
//!
//! The UI thread publishes an owned draw list through a triple buffer. The render
//! thread takes the newest available frame; intermediate frames may be dropped.
//! Each thread owns a separate slot, so rendering does not hold up frame building.
//!
//! `tests/loom_handoff.rs` checks the handoff's memory ordering.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

#[cfg(loom)]
use loom::sync::atomic::{AtomicU8, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(loom)]
use loom::cell::UnsafeCell;
#[cfg(not(loom))]
use std::cell::UnsafeCell;

use bytemuck::{Pod, Zeroable};

use crate::color::Srgba;
use crate::scene::SceneList;

// -- primitives --------------------------------------------------------------------

/// Which shader branch draws this instance.
///
/// One pipeline handles every primitive, discriminated here rather than by switching
/// pipelines. A pipeline switch costs a state change per batch; a branch on a value that
/// is uniform across a batch's worth of instances costs the GPU essentially nothing,
/// because every invocation in a warp takes the same side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PrimKind {
    /// Pass 0: filled rounded rectangle, signed-distance antialiased.
    Rect = 0,
    /// Pass 1: rounded-rectangle stroke. `param` is the stroke width in pixels.
    Stroke = 1,
    /// Pass 3: a glyph, sampling R8 coverage from the atlas.
    Glyph = 2,
    /// Pass 0 again, with two stops: the same rounded-rect fill, whose colour ramps across
    /// the shape.
    ///
    /// [`Instance::color`] is the near stop, [`Instance::uv`] the far one, and
    /// [`Instance::param`] the axis angle in radians. The ramp is walked in **Oklab**, not
    /// in linear sRGB -- see [`crate::color::linear_rgb_to_oklab`] for why that is a
    /// correctness question rather than a preference.
    Gradient = 3,
    /// A filled shape with a halo that fades outside its edge.
    ///
    /// [`Instance::color`] is the edge tint, [`Instance::uv`] is the outer tint
    /// in premultiplied linear RGBA, and [`Instance::param`] is the falloff
    /// distance in physical pixels. The shader uses the rounded-box distance
    /// field. The CPU fallback draws nothing.
    Glow = 4,
    /// Pass 0's shape again, read with the opposite sign: a soft light that is brightest at
    /// the boundary and vanishes [`Instance::param`] pixels *inward*.
    ///
    /// [`Instance::color`] is the tint and [`Instance::param`] the width in physical pixels.
    /// [`Instance::uv`] is unused -- a rim has one stop, because its second one is nothing.
    ///
    /// Not a stroke, which is why it is not spelled with one. A stroke is a band of uniform
    /// alpha at a fixed distance from the boundary; this is a ramp that reaches inward, and
    /// the difference is the whole reason a rounded rectangle stops reading as a colour and
    /// starts reading as a lit surface. It also puts **no ink outside the shape**, which is
    /// what separates it from [`PrimKind::Glow`]: the quad needs no extra padding, and the
    /// shape's own one-pixel coverage is what stops the light at the edge.
    Rim = 5,
    /// A beveled surface shaded with a Cook-Torrance microfacet BRDF.
    ///
    /// [`Instance::color`] is the albedo; [`Instance::uv`] stores bevel width
    /// in physical pixels, roughness, metallic fraction and environment weight.
    /// [`Instance::param`] carries emission.
    ///
    /// The shader derives the normal from the rounded-box distance field and
    /// samples an analytic environment. It needs no scene acceleration structure.
    /// The CPU fallback is a rectangle filled with the albedo.
    Pbr = 6,
    /// Pass 0's shape with the ramp taken **around** it rather than across it: a conic sweep,
    /// offset by a phase, so a highlight can travel around a border.
    ///
    /// The fields are [`PrimKind::Gradient`]'s, field for field -- [`Instance::color`] the near
    /// stop, [`Instance::uv`] the far one -- and [`Instance::param`] is the phase in radians
    /// where a gradient's is the axis angle. That is not a coincidence to be tidied away: this
    /// is the same two-stop ramp walked in the same Oklab, and only the parameter differs, so
    /// the shader shares `ramp_at` between them rather than growing a second copy.
    ///
    /// The angular parameter is **mirrored**: the ramp runs near to far and back over one full
    /// turn. An angle wraps, and a ramp laid straight onto one steps from the far stop back to
    /// the near stop where `atan2` crosses back to -π -- a hue seam fixed in place on the shape
    /// that no phase can move out of sight. Mirroring makes `t` continuous across the wrap by
    /// construction; what is left at the turning points is a change of slope, which is what a
    /// two-stop conic is supposed to look like.
    ///
    /// [`Fidelity::Exact`], and the only enhanced-looking primitive here that is not enhanced.
    /// See [`PrimKind::fidelity`] for why declaring a floor would have been a claim rather than
    /// a concession.
    Sweep = 7,
    /// A drifting color field composited over a base color.
    ///
    /// [`Instance::color`] is the base, [`Instance::uv`] is `[amplitude, 0, 0, 0]`,
    /// and [`Instance::param`] is the phase in radians. The centers come from the
    /// draw list's [`FieldWash`]. Centers contribute only within their reach.
    /// The CPU fallback draws a rectangle in the base color.
    Field = 8,
    /// An image sampled from the atlas's RGBA color page.
    ///
    /// [`Instance::uv`] holds normalized `[u0, v0, u1, v1]` coordinates.
    /// [`Instance::color`] multiplies the sample; opaque white preserves it.
    /// `radius` and `param` are unused. The image has square corners.
    /// Both GPU and CPU renderers support this primitive.
    Image = 9,
    /// A tinted, blurred copy of the content behind the shape.
    ///
    /// The renderer captures the backdrop before the first sampling instance
    /// and blurs it through [`crate::target::BlurChain`].
    ///
    /// [`Instance::color`] holds the opaque CPU fallback. [`Instance::uv`]
    /// holds the premultiplied glass tint, and [`Instance::param`] is unused.
    /// All blur instances in a frame share the blurred backdrop.
    Blur = 10,
    /// A PBR surface with refraction along its bevel.
    ///
    /// [`Instance::color`] is the albedo. [`Instance::uv`] stores bevel width,
    /// roughness, metallic fraction and environment weight. [`Instance::param`]
    /// is refraction strength in `0..=1`; zero gives the non-emissive PBR surface.
    ///
    /// The shader samples a sharp, full-resolution backdrop. Refraction falls
    /// to zero at the inner edge of the bevel, leaving the label background
    /// unchanged. Index of refraction, depth and dispersion are shader constants.
    ///
    /// Without a backdrop, the shader uses PBR shading. The CPU fallback draws
    /// a rectangle in the albedo color.
    Refract = 11,
    /// Pass 1's band drawn as dashes: the same inside-aligned stroke, its coverage cut by
    /// an on/off pattern walked along the shape's perimeter.
    ///
    /// [`Instance::param`] is the stroke width the way a stroke's is; [`Instance::uv`] is
    /// `[dash, gap, 0, 0]` in pixels of perimeter. The pattern is scaled so a whole number
    /// of periods closes the loop -- a pattern that merely tiled the perimeter would end on
    /// a fragment of a dash wherever the seam fell.
    ///
    /// A primitive rather than a row of short rects, because a dashed outline is a *shape*:
    /// the dashes follow the corners' arcs, a probe can count dashed outlines on a draw
    /// list the way it counts gradients, and a row of rects is indistinguishable from a
    /// row of rules. It exists for the design language's empty slots and drop targets --
    /// the one class its UI grammar allows a dashed outline.
    ///
    /// [`Fidelity::Exact`], and the CPU tier earns it by **transcription**: `tiny-skia` can
    /// dash a stroked path, but it dashes the *inset* path that tier strokes (a shorter
    /// perimeter) from wherever the path happens to start, so its dashes and the shader's
    /// drift apart around the shape. The CPU arm evaluates the same signed distance and the
    /// same perimeter walk per pixel instead.
    DashedStroke = 12,
}

impl PrimKind {
    /// Every primitive the pipeline can draw.
    ///
    /// Declared, not derived -- Rust has no stable way to enumerate an enum's variants.
    /// Two independent things keep the list from going stale, and neither is a comment
    /// asking people to remember:
    ///
    /// - [`PrimKind::index`] below is an *exhaustive* match, so adding a variant is a
    ///   compile error three lines from this array.
    /// - `the_shader_and_prim_kind_declare_the_same_primitives` in `tier_parity`
    ///   parses the `KIND_` constants out of `shaders/instance.wgsl` and asserts that set
    ///   equals this one. A primitive the shader can draw but this array omits fails the
    ///   test suite, which is the case the compile error above cannot see.
    ///
    /// It exists so the `tier_parity` suite can assert that *every* kind has a parity
    /// fixture, rather than asserting it about whichever kinds someone remembered.
    pub const ALL: [PrimKind; 13] = [
        Self::Rect,
        Self::Stroke,
        Self::Glyph,
        Self::Gradient,
        Self::Glow,
        Self::Rim,
        Self::Pbr,
        Self::Sweep,
        Self::Field,
        Self::Image,
        Self::Blur,
        Self::Refract,
        Self::DashedStroke,
    ];

    /// This variant's position in [`PrimKind::ALL`].
    ///
    /// The match is exhaustive on purpose. It is the compile-time half of the guard
    /// described on `ALL`, and its only job is to fail to build when a variant appears.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Rect => 0,
            Self::Stroke => 1,
            Self::Glyph => 2,
            Self::Gradient => 3,
            Self::Glow => 4,
            Self::Rim => 5,
            Self::Pbr => 6,
            Self::Sweep => 7,
            Self::Field => 8,
            Self::Image => 9,
            Self::Blur => 10,
            Self::Refract => 11,
            Self::DashedStroke => 12,
        }
    }

    /// The name this primitive answers to in `shaders/instance.wgsl`.
    ///
    /// Also an exhaustive match, and the bridge that lets a test compare the Rust enum
    /// against the WGSL constants by name rather than by discriminant alone -- a swapped
    /// pair of discriminants would otherwise compare equal as a set.
    #[must_use]
    pub const fn shader_const(self) -> &'static str {
        match self {
            Self::Rect => "KIND_RECT",
            Self::Stroke => "KIND_STROKE",
            Self::Glyph => "KIND_GLYPH",
            Self::Gradient => "KIND_GRADIENT",
            Self::Glow => "KIND_GLOW",
            Self::Rim => "KIND_RIM",
            Self::Pbr => "KIND_PBR",
            Self::Sweep => "KIND_SWEEP",
            Self::Field => "KIND_FIELD",
            Self::Image => "KIND_IMAGE",
            Self::Blur => "KIND_BLUR",
            Self::Refract => "KIND_REFRACT",
            Self::DashedStroke => "KIND_DASHED_STROKE",
        }
    }

    /// The CPU rendering requirement for this primitive.
    ///
    /// Exact primitives are compared with the shader output. Enhanced primitives
    /// declare a simpler fallback. Glow and rim effects draw nothing on the CPU;
    /// their associated state must also be conveyed by a fill or outline.
    /// Sweeps are rendered on both paths.
    #[must_use]
    pub const fn fidelity(self) -> Fidelity {
        match self {
            // [`PrimKind::Image`] is `Exact` for the reason the glyph is, one page over.
            // `tiny-skia` blits an RGBA pixmap, and both tiers read the same atlas
            // coordinates out of the same `GlyphAtlas`, so what the CPU tier draws is the
            // same picture in the same place. A floor here would be a concession nothing
            // forces -- and the concession worth refusing is specifically a greyscale one,
            // which is where this chunk started.
            Self::Rect
            | Self::Stroke
            // The dashed stroke is the stroke with its coverage cut, and the CPU arm is a
            // transcription of the same distance and perimeter arithmetic -- so the two
            // tiers are held to the same picture the way the plain stroke's are.
            | Self::DashedStroke
            | Self::Glyph
            | Self::Gradient
            | Self::Sweep
            | Self::Image => Fidelity::Exact,
            Self::Glow | Self::Rim => Fidelity::Enhanced {
                floor: Floor::Nothing,
            },
            // The third enhanced kind, and the first whose floor is a primitive rather than
            // absence. `tiny-skia` has no BRDF, but a lit surface and an unlit one cover the
            // same pixels -- the shading changes what is inside the shape, never the shape --
            // so dropping the light leaves the albedo, which is what the surface is made of.
            // That is the plainer version 10.7 asks for. `Nothing` would have been wrong here
            // in a way it was right for the halo: a halo is decoration around a shape, and
            // this *is* the shape.
            // The second kind whose floor is a primitive rather than absence, and the one
            // where the floor is not a degradation so much as a limit. Where no centre
            // reaches, the field IS its base colour; the floor is that everywhere. So the
            // fallback tier gets the flat ground the palette already says the window is,
            // which is exactly the plainer theme UXDD 10.7 asks for rather than a stand-in
            // for something absent.
            //
            // `Nothing` would have been wrong for a reason the halo's floor makes clear: a
            // halo is decoration around a shape, and this is the ground under everything.
            // Dropping it would leave the window unpainted.
            // The third kind with a primitive for a floor, and the only one whose floor was
            // written down years before the effect: UXDD 10.7's table says **opaque
            // `surface/raised`**, and names a translucent unblurred panel as the wrong answer.
            // `Plain(Rect)` is exactly that, because `Instance::color` on a blur carries the
            // opaque panel rather than the glass tint -- see [`PrimKind::Blur`] for why the
            // fields are that way round and what the tempting arrangement ships instead.
            //
            // `Nothing` is not available here for the reason it was not available to the PBR
            // surface: this IS the panel. Dropped, a popover would be its text over the list.
            // The fourth, and the one whose floor is a *reuse* rather than a decision. A
            // refracting surface is a PBR surface plus a transmitted term, so what it becomes
            // without the light and without the backdrop is what the PBR surface becomes:
            // its albedo. Naming `Plain(Pbr)` instead -- "the surface keeps its shading and
            // loses only the glass" -- is the reading the chunk's own acceptance suggests, and
            // it is refused two ways. `a_floor_is_always_a_primitive_the_cpu_tier_actually_draws`
            // rejects a floor that is itself enhanced, because `cpu_floor` resolves exactly one
            // level and a chain is a design nobody can picture. And `cpu_floor` drops `uv` on
            // the way down, so the Pbr instance that arrived would have a zero bevel, a zero
            // roughness clamped to a mirror and no sky -- not "its PBR form" but an accident
            // wearing its name.
            Self::Pbr | Self::Field | Self::Blur | Self::Refract => Fidelity::Enhanced {
                floor: Floor::Plain(PrimKind::Rect),
            },
        }
    }

    /// Whether the primitive needs to sample previously rendered content.
    ///
    /// Blur and refraction require an offscreen backdrop and declare CPU
    /// fallbacks. Bloom is handled separately in the resolve pass through
    /// [`Bloom`].
    #[must_use]
    pub const fn needs_backdrop(self) -> bool {
        match self {
            Self::Rect
            | Self::Stroke
            | Self::Glyph
            | Self::Gradient
            | Self::Glow
            | Self::Rim
            | Self::Pbr
            // A conic sweep is a function of the fragment's own angle about the shape's
            // centre, which is a function of its own position. Nothing is sampled.
            | Self::Sweep
            // A field is a function of that same position against a handful of uniforms.
            // Covering the window is a different question from sampling it.
            | Self::Field
            // A picture samples a texture, which is not the same as sampling the backdrop.
            // The atlas is content the frame put there itself; a backdrop is the frame's own
            // prior output, and only the second needs the two-pass path.
            | Self::Image => false,
            // A dash is a function of the fragment's own place on the perimeter.
            Self::DashedStroke => false,
            // And this is that second thing. A blurred pixel is a weighted sum over its
            // neighbours, which no amount of closed form reaches from one fragment.
            Self::Blur
            // A refracted pixel is a *displaced* one, which is the same obligation for the
            // opposite reason: a blur needs many neighbours and a refraction needs one
            // neighbour it cannot name in advance. Either way the fragment's own position is
            // not enough, and only the two-pass path can answer.
            | Self::Refract => true,
        }
    }

    /// The variant a raw [`Instance::kind`] names, or `None` if it names nothing.
    ///
    /// `Instance::kind` is a `u32` on the wire because that is what the vertex format carries,
    /// so anything reading it back has to answer for a value the enum does not cover. `None`
    /// rather than a default variant: a draw list holding an unknown kind is a bug somewhere
    /// upstream, and silently treating it as a rectangle would draw a rectangle nobody asked
    /// for instead of leaving a hole somebody investigates.
    #[must_use]
    pub const fn from_raw(kind: u32) -> Option<Self> {
        match kind {
            0 => Some(Self::Rect),
            1 => Some(Self::Stroke),
            2 => Some(Self::Glyph),
            3 => Some(Self::Gradient),
            4 => Some(Self::Glow),
            5 => Some(Self::Rim),
            6 => Some(Self::Pbr),
            7 => Some(Self::Sweep),
            8 => Some(Self::Field),
            // Absent until `prim-backdrop-blur` added the arm below it and noticed. It cost
            // nothing while it was missing -- every caller reads this to ask a question whose
            // answer for a picture is `false` either way -- but "the kind the enum does not
            // know" is precisely the diagnosis `from_raw` exists to make, and a kind the enum
            // knows perfectly well answering it is the failure mode `None` is meant to expose.
            9 => Some(Self::Image),
            10 => Some(Self::Blur),
            11 => Some(Self::Refract),
            12 => Some(Self::DashedStroke),
            _ => None,
        }
    }
}

/// How closely the CPU renderer must reproduce a primitive.
///
/// Exact primitives are compared with a Rust evaluation of the shader.
/// Enhanced primitives declare a fallback that the CPU renderer can draw;
/// tests compare against that fallback.
///
/// [`CpuRasterizer`]: crate::cpu_raster::CpuRasterizer
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fidelity {
    /// Both tiers draw it, and the parity fixtures hold them to the same pixels within
    /// bounds derived from a stated source.
    Exact,
    /// The CPU tier cannot draw it. It draws `floor` instead, and the parity fixtures hold
    /// it to *that*, exactly.
    Enhanced { floor: Floor },
}

/// What the CPU tier draws in place of an effect it cannot draw.
///
/// Naming a primitive rather than a picture is deliberate: the floor has to be something
/// the CPU tier already renders and the parity suite already covers, or the fallback is
/// itself unchecked and nothing has been gained.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Floor {
    /// Nothing at all. The right answer more often than it looks: a hard rectangle standing
    /// in for a halo reads as a bug, where an absent halo reads as a plainer theme.
    Nothing,
    /// This primitive, from the same `rect`, `radius` and `color`, with the kind-specific
    /// fields reset -- see [`Instance::cpu_floor`] for why they cannot come along.
    Plain(PrimKind),
}

/// One primitive. Exactly 48 bytes, asserted below.
///
/// The size is not arbitrary. At the fastest fling a frame carries roughly 45 rows of
/// perhaps 40 glyphs plus chrome -- call it 2,000 instances, or 96 KB. That fits in the
/// per-frame staging allocation with room to spare and streams to the GPU in one upload.
/// Growing the struct to 64 bytes would cost a third more bandwidth for fields that would
/// be zero on almost every instance.
///
/// Clipping is deliberately **not** a field. It lives on [`Batch`] as a scissor rect,
/// because clip regions are uniform across long runs of instances (a row's name column
/// clips identically for every glyph in it) and a per-instance copy would be 16 wasted
/// bytes repeated thousands of times.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, Default, Pod, Zeroable)]
pub struct Instance {
    /// `[x, y, width, height]` in physical pixels, rebased against the scroll
    /// offset before conversion to `f32`.
    pub rect: [f32; 4],
    /// Four primitive-specific values.
    ///
    /// Glyphs and images use normalized atlas coordinates. Other primitives
    /// use the values for colors or effect parameters; see [`PrimKind`].
    pub uv: [f32; 4],
    /// Premultiplied linear RGBA8 -- see [`crate::color`]. The near stop, for a gradient.
    pub color: u32,
    /// Corner radius in physical pixels.
    pub radius: f32,
    /// Kind-specific scalar: stroke width for [`PrimKind::Stroke`], axis angle in radians
    /// for [`PrimKind::Gradient`], unused otherwise.
    pub param: f32,
    /// A [`PrimKind`].
    pub kind: u32,
}

const _: () = assert!(
    size_of::<Instance>() == 48,
    "the instance stride is a bandwidth budget, not an accident -- if this changes, \
     re-derive the per-frame upload size before accepting it"
);

impl Instance {
    pub fn rect(x: f32, y: f32, w: f32, h: f32, radius: f32, color: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [0.0; 4],
            color: color.to_premul_linear_rgba8(),
            radius,
            param: 0.0,
            kind: PrimKind::Rect as u32,
        }
    }

    pub fn stroke(x: f32, y: f32, w: f32, h: f32, radius: f32, width: f32, color: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [0.0; 4],
            color: color.to_premul_linear_rgba8(),
            radius,
            param: width,
            kind: PrimKind::Stroke as u32,
        }
    }

    /// A rounded-rect stroke drawn as dashes: `width` thick, `dash` pixels of ink then
    /// `gap` pixels of nothing, walked clockwise along the shape's perimeter from the top
    /// edge's left end. See [`PrimKind::DashedStroke`] for the seam scaling and why the
    /// CPU tier transcribes rather than dashes a path.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn dashed_stroke(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        width: f32,
        dash: f32,
        gap: f32,
        color: Srgba,
    ) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [dash.max(0.0), gap.max(0.0), 0.0, 0.0],
            color: color.to_premul_linear_rgba8(),
            radius,
            param: width,
            kind: PrimKind::DashedStroke as u32,
        }
    }

    /// A rounded-rect fill whose colour ramps from `near` to `far` across the shape.
    ///
    /// `angle` is in radians, measured the way the surface's coordinates run: `0.0` ramps
    /// left-to-right and `FRAC_PI_2` ramps top-to-bottom, because `y` grows downward here.
    /// The ramp is normalized against how far the box reaches along that axis, so the
    /// stops land on the shape's extremes at any angle rather than only at 0 and 90
    /// degrees -- a diagonal gradient that ran out early would show a flat band in the far
    /// corner.
    // Eight rather than seven, and the eighth is the second colour, which is the entire
    // primitive. Collapsing the rect into a tuple to satisfy the count would make this the
    // one constructor here that does not take `x, y, w, h`.
    #[allow(clippy::too_many_arguments)]
    pub fn gradient(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        angle: f32,
        near: Srgba,
        far: Srgba,
    ) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: far.to_premul_linear_f32(),
            color: near.to_premul_linear_rgba8(),
            radius,
            param: angle,
            kind: PrimKind::Gradient as u32,
        }
    }

    /// The same two-stop ramp as [`Instance::gradient`], taken **around** the shape's centre
    /// instead of across it, and offset by `phase`.
    ///
    /// `phase` is in radians, and it is where the far stop sits: `0.0` puts it at the shape's
    /// right-hand side, and increasing it carries the highlight around through the bottom --
    /// the same sense in which a gradient's angle turns its axis, since `y` grows downward
    /// here. A caller animating it takes it from the material drive's phase (`qs-ui`) and not
    /// from a clock; nothing in this pipeline reads the time.
    ///
    /// The ramp is mirrored around the turn, so `near` appears at the phase's opposite side
    /// and there is no seam. See [`PrimKind::Sweep`].
    #[allow(clippy::too_many_arguments)]
    pub fn sweep(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        phase: f32,
        near: Srgba,
        far: Srgba,
    ) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: far.to_premul_linear_f32(),
            color: near.to_premul_linear_rgba8(),
            radius,
            param: phase,
            kind: PrimKind::Sweep as u32,
        }
    }

    /// Draw the ambient field over `base` across `rect`.
    ///
    /// Centers come from the draw list's [`FieldWash`]. `amplitude` is clamped
    /// to `0..=1`, and `phase` is in radians. The CPU fallback draws `base`.
    pub fn field(x: f32, y: f32, w: f32, h: f32, amplitude: f32, phase: f32, base: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [amplitude.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
            color: base.to_premul_linear_rgba8(),
            radius: 0.0,
            param: phase,
            kind: PrimKind::Field as u32,
        }
    }

    /// The shape, solid, surrounded by a halo of the same colour fading to nothing over
    /// `falloff` physical pixels.
    ///
    /// One colour rather than two, and both stops get it. The fade belongs to the coverage
    /// profile, so handing the outer stop a transparent colour as well would multiply two
    /// ramps together and pull the halo in tight against the shape -- an easy mistake to
    /// make and a hard one to see, since the result is still a glow, just a worse one.
    /// [`Instance::glow_two_tone`] is the form that spends the second stop deliberately.
    ///
    /// `falloff` is in **physical** pixels, like everything else in `rect`: a caller
    /// working in logical units multiplies by the scale first, or the halo is the wrong
    /// size on exactly the displays that make a halo worth having.
    pub fn glow(x: f32, y: f32, w: f32, h: f32, radius: f32, falloff: f32, color: Srgba) -> Self {
        Self::glow_two_tone(x, y, w, h, radius, falloff, color, color)
    }

    /// A glow whose tint travels from `inner` at the shape's edge to `outer` at the falloff
    /// limit.
    ///
    /// The ramp is walked in **premultiplied linear**, not in Oklab, and that is the
    /// opposite of the choice [`Instance::gradient`] makes. See the `halo` function in
    /// `shaders/instance.wgsl`: a stop at zero alpha has no hue to recover, so a perceptual
    /// walk drags the fade through black and leaves a dark rim.
    #[allow(clippy::too_many_arguments)]
    pub fn glow_two_tone(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        falloff: f32,
        inner: Srgba,
        outer: Srgba,
    ) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: outer.to_premul_linear_f32(),
            color: inner.to_premul_linear_rgba8(),
            radius,
            param: falloff.max(0.0),
            kind: PrimKind::Glow as u32,
        }
    }

    /// A soft light just inside the shape's edge: brightest at the boundary, gone `width`
    /// pixels inward, and nothing at all outside.
    ///
    /// `width` is in **physical** pixels, like everything else in `rect` and like
    /// [`Instance::stroke`]'s width. A caller working in logical units multiplies by the
    /// scale first, or the rim is a different thickness on exactly the displays where a
    /// one-pixel difference is visible.
    ///
    /// One colour and no second stop. The ramp runs from this colour to *nothing*, and it is
    /// walked in premultiplied linear for the reason [`Instance::glow_two_tone`] gives: a
    /// stop at zero alpha has no hue to recover, so the fade is carried by the coverage
    /// profile and the tint stays constant across it rather than being lerped toward a colour
    /// that is not there. That is why `uv` is left free here -- see the `rim_t` function in
    /// `shaders/instance.wgsl`.
    pub fn rim(x: f32, y: f32, w: f32, h: f32, radius: f32, width: f32, color: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [0.0; 4],
            color: color.to_premul_linear_rgba8(),
            radius,
            param: width.max(0.0),
            kind: PrimKind::Rim as u32,
        }
    }

    /// Create a beveled surface with microfacet shading.
    ///
    /// `bevel` is in physical pixels. `roughness`, `metallic` and `environment`
    /// range from zero to one. The shader clamps roughness away from zero to
    /// avoid a singular specular lobe.
    ///
    /// `emission` adds the albedo along the bevel. Its contribution reaches
    /// zero at the inner edge, leaving the background under the label unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn pbr(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        bevel: f32,
        roughness: f32,
        metallic: f32,
        environment: f32,
        emission: f32,
        albedo: Srgba,
    ) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [
                bevel.max(0.0),
                roughness.clamp(0.0, 1.0),
                metallic.clamp(0.0, 1.0),
                environment.max(0.0),
            ],
            color: albedo.to_premul_linear_rgba8(),
            radius,
            param: emission.max(0.0),
            kind: PrimKind::Pbr as u32,
        }
    }

    /// The same bevelled surface, with the bevel made of glass.
    ///
    /// Every argument up to `albedo` is [`Instance::pbr`]'s, in the same order and with the
    /// same meaning, and that is deliberate: a refracting panel and a lit one are the same
    /// surface, so a caller moving between them changes one word.
    ///
    /// `refraction` is the strength in `0..=1`, and it is spent where the PBR surface spends
    /// its emission. **Zero is not "off with rounding" -- it is the PBR surface exactly**,
    /// which `refraction_at_zero_strength_is_the_pbr_surface` holds byte for byte. That is what
    /// makes the degradation on a machine with no backdrop bound a property of the shader's
    /// arithmetic rather than a second code path.
    ///
    /// The index of refraction is **not** an argument. One blurred copy of the backdrop serves
    /// every blur in the frame for a structural reason; one IOR serves every glass surface for
    /// a design one -- see [`PrimKind::Refract`]. If two panels in one window are made of
    /// different glass, the window has a bigger problem than a missing parameter.
    #[allow(clippy::too_many_arguments)]
    pub fn refract(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        bevel: f32,
        roughness: f32,
        metallic: f32,
        environment: f32,
        refraction: f32,
        albedo: Srgba,
    ) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [
                bevel.max(0.0),
                roughness.clamp(0.0, 1.0),
                metallic.clamp(0.0, 1.0),
                environment.max(0.0),
            ],
            color: albedo.to_premul_linear_rgba8(),
            radius,
            // Clamped at both ends, unlike the PBR surface's emission which is only clamped
            // below. Emission above one is a brighter light and means something; refraction
            // above one is a fraction of a ray larger than the ray, and the shader would spend
            // it as a transmitted term brighter than what is behind the panel.
            param: refraction.clamp(0.0, 1.0),
            kind: PrimKind::Refract as u32,
        }
    }

    pub fn glyph(x: f32, y: f32, w: f32, h: f32, uv: [f32; 4], color: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv,
            color: color.to_premul_linear_rgba8(),
            radius: 0.0,
            param: 0.0,
            kind: PrimKind::Glyph as u32,
        }
    }

    /// A picture from the atlas's colour page, blitted into `w x h`.
    ///
    /// `uv` is the entry's normalized rectangle on the **colour** page -- which is a
    /// different texture from the glyph page and usually a different size, so a `uv`
    /// computed against `GlyphAtlas::size()` addresses the wrong texels. Take it from the
    /// [`crate::atlas::AtlasEntry`] and the arithmetic is already done.
    ///
    /// `tint` multiplies the sample. An opaque white tint leaves the picture as it
    /// was decoded; a lower alpha fades it, which is how a thumbnail arrives without a
    /// second primitive to cross-fade with.
    pub fn image(x: f32, y: f32, w: f32, h: f32, uv: [f32; 4], tint: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv,
            color: tint.to_premul_linear_rgba8(),
            radius: 0.0,
            param: 0.0,
            kind: PrimKind::Image as u32,
        }
    }

    /// Draw a tinted panel over a blurred backdrop.
    ///
    /// `floor` is the opaque CPU fallback stored in [`Instance::color`].
    /// `tint` is the glass color stored in [`Instance::uv`]. Every panel uses
    /// the frame's shared blur radius and backdrop from [`crate::target::BlurChain`].
    pub fn blur(x: f32, y: f32, w: f32, h: f32, radius: f32, tint: Srgba, floor: Srgba) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: tint.to_premul_linear_f32(),
            color: floor.to_premul_linear_rgba8(),
            radius,
            param: 0.0,
            kind: PrimKind::Blur as u32,
        }
    }

    /// What the CPU tier draws for this instance. `None` means nothing at all.
    ///
    /// The one route from an enhanced instance to CPU pixels, and the reason
    /// `tier_parity`'s floor check is a real assertion rather than a tautology: the failure
    /// it is watching for is a second route -- an arm added to `CpuRasterizer::draw_instance`
    /// that approximates an effect instead of degrading it, which is exactly the well-meant
    /// change that would put an unstated difference back into the tiers.
    ///
    /// `param` and `uv` are dropped on the way down to a [`Floor::Plain`]. They are where an
    /// enhanced primitive keeps its effect -- a falloff distance, an outer colour -- and the
    /// floor kind reads the same bytes as something else entirely, so carrying them over
    /// would render a glow's falloff as a stroke width. An effect whose floor genuinely needs
    /// one of them needs a `Floor` variant that says so, not a field that survives by
    /// accident.
    #[must_use]
    pub fn cpu_floor(&self) -> Option<Self> {
        let kind = match self.kind {
            k if k == PrimKind::Rect as u32 => PrimKind::Rect,
            k if k == PrimKind::Stroke as u32 => PrimKind::Stroke,
            k if k == PrimKind::Glyph as u32 => PrimKind::Glyph,
            k if k == PrimKind::Gradient as u32 => PrimKind::Gradient,
            k if k == PrimKind::Glow as u32 => PrimKind::Glow,
            k if k == PrimKind::Rim as u32 => PrimKind::Rim,
            k if k == PrimKind::Pbr as u32 => PrimKind::Pbr,
            k if k == PrimKind::Sweep as u32 => PrimKind::Sweep,
            k if k == PrimKind::Field as u32 => PrimKind::Field,
            k if k == PrimKind::Image as u32 => PrimKind::Image,
            // The one arm whose absence would have been a *silent* pass rather than a wrong
            // one: falling through to `Some(*self)` leaves a blur instance for the rasterizer,
            // whose unknown-kind fallback fills it with `color` -- which is the floor colour,
            // so the picture would have been right and `uv` and `param` would have travelled
            // anyway. A floor that is correct by accident is the thing `cpu_floor` centralises
            // fidelity to prevent.
            k if k == PrimKind::Blur as u32 => PrimKind::Blur,
            // Whose absence would fail the same silent way, and worse: the rasterizer's
            // unknown-kind fallback fills with `color`, which for this kind IS the albedo, so
            // a missing arm would render the correct picture while `uv` and `param` travelled
            // to a rasterizer that reads them as something else entirely.
            k if k == PrimKind::Refract as u32 => PrimKind::Refract,
            // A kind the enum does not know. The rasterizer's own fallback treats it as a
            // fill, which is the behaviour that predates this method; deciding it is enhanced
            // would silently drop an instance instead.
            _ => return Some(*self),
        };
        match kind.fidelity() {
            Fidelity::Exact => Some(*self),
            Fidelity::Enhanced {
                floor: Floor::Nothing,
            } => None,
            Fidelity::Enhanced {
                floor: Floor::Plain(plain),
            } => Some(Self {
                uv: [0.0; 4],
                param: 0.0,
                kind: plain as u32,
                ..*self
            }),
        }
    }
}

/// A run of instances sharing a scissor rect and a texture binding.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Batch {
    pub range: Range<u32>,
    /// `[x, y, w, h]` in physical pixels. `None` means the whole surface.
    pub scissor: Option<[u32; 4]>,
    /// Whether this batch samples the glyph atlas.
    pub textured: bool,
}

// -- draw list ---------------------------------------------------------------------

/// Per-frame counters, mirrored into `FrameSample` by the harness.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DrawStats {
    pub rows_laid_out: u32,
    pub glyphs_rasterized: u32,
    /// Requested glyphs the atlas could not supply this frame, usually because
    /// the upload budget deferred them. Expose this counter in diagnostics.
    pub glyphs_dropped: u32,
    /// Icons wanted this frame that the atlas could not supply.
    ///
    /// Separate from [`DrawStats::glyphs_dropped`] on purpose. The two share a meaning --
    /// something that should be on screen is not -- but not a cause: a dropped glyph sends
    /// you to the font stack, a dropped icon to [`crate::icon`] or to a size that fell
    /// outside the legible range. One number would send every investigation to the wrong
    /// place half the time.
    pub icons_dropped: u32,
    /// Distinct atlas entries this frame wanted and the upload bound could not reach.
    ///
    /// The companion to [`DrawStats::glyphs_dropped`] and not a duplicate of it: `dropped`
    /// counts *draws* that went without, which is what the reader sees missing, while this
    /// counts *entries* still owed, which is what decides how many more frames it takes to
    /// converge. A viewport short forty entries against a 64-upload bound converges on the
    /// next frame; one short two hundred does not, and only this number says which.
    pub glyph_shortfall: u32,
    pub shaped_runs_new: u32,
    pub instances: u32,
}

/// Everything the render thread needs for one frame, and nothing it has to look up.
///
/// This owns its data rather than borrowing from UI state. That is the point: once
/// published, the render thread can take arbitrarily long without the UI thread having to
/// care what it is still reading.
/// The room the window is in, as far as a lit surface is concerned.
///
/// Two stops of an infinite sky: `horizon` is what a ray sees looking along the surface plane
/// and `zenith` what it sees looking straight up. Intersecting a reflected ray with that has a
/// closed form, which is why a reflection costs no acceleration structure and no second pass
/// -- see [`PrimKind::Pbr`].
///
/// It lives on the [`DrawList`] and not on the [`Instance`] for two reasons, and the second is
/// the one that would have decided it alone: an environment is a property of the scene rather
/// than of each object, and the 48-byte instance stride is already spent.
///
/// The default is a neutral illuminant, deliberately dim. A surface reflecting it looks lit
/// rather than correct, which is the right failure for a caller that forgot
/// [`DrawList::set_environment`]: the palette is where the real answer comes from, and a
/// renderer that shipped its own would be a second place colour is decided.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Environment {
    pub horizon: Srgba,
    pub zenith: Srgba,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            horizon: Srgba::new(0.18, 0.19, 0.22, 1.0),
            zenith: Srgba::new(0.42, 0.45, 0.52, 1.0),
        }
    }
}

/// Where the key light is, as a unit vector pointing **towards** it: above and slightly to the
/// left, with `y` growing downward as everything here does.
///
/// A convention rather than a parameter, and it is stated on the Rust side as well as in
/// `shaders/instance.wgsl` because two different things now need it. The shader shades with it.
/// And a material that wants to cast a **contact shadow** needs to know which way the shadow
/// falls, which is this vector's `xy`, negated.
///
/// Exported rather than duplicated at the one call site that wants it, for the reason the
/// shader gives for fixing it at all: interfaces are lit from above because that is where light
/// comes from in the world the reader is sitting in, and a material that could author its own
/// light direction could author a surface that disagrees with every other surface in the
/// window. `the_shader_and_the_renderer_agree_about_where_the_light_is` compares this against
/// the shader's own text, the way the dither constants are already compared.
pub const LIGHT_DIR: [f32; 3] = [-0.32, -0.55, 0.77];

/// Which way a shadow falls, as a unit vector in the surface's own coordinates.
///
/// The key light's `xy`, negated and renormalized: down and to the right. A caller multiplies
/// it by a distance — see the `offset` a `qs-ui` material layer can state — rather than
/// authoring a vector, so every shadow in the window agrees about where the light is.
#[must_use]
pub fn shadow_direction() -> [f32; 2] {
    let (x, y) = (-LIGHT_DIR[0], -LIGHT_DIR[1]);
    let length = x.hypot(y);
    if length <= 0.0 {
        // A light directly overhead casts no directional shadow. Answered rather than
        // divided by, the same shape of total function `glow_t` gives a zero falloff.
        return [0.0, 0.0];
    }
    [x / length, y / length]
}

/// Number of centers stored in a [`FieldWash`].
///
/// The count is fixed by the uniform-buffer layout. Unused centers have
/// zero contribution.
pub const FIELD_CENTRES: usize = 4;

/// One coloured centre of a [`FieldWash`].
///
/// Positions are in **normalized viewport coordinates** — `[0, 0]` is the top-left of the
/// window and `[1, 1]` the bottom-right — because a field is authored against the window and
/// not against a pixel count. `reach` and `drift` are in units of the viewport's **width** on
/// both axes, so a centre is a circle rather than an ellipse that changes shape when the
/// window is resized.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct FieldCentre {
    /// Where this centre sits at phase zero, in normalized viewport coordinates.
    pub at: [f32; 2],
    /// How far it travels from `at` over one cycle, in units of the viewport's width.
    pub drift: [f32; 2],
    /// How far its light reaches, in units of the viewport's width. Beyond it the centre
    /// contributes exactly nothing — the falloff has **bounded support**, which is what lets
    /// four centres cost four `max`es rather than four unbounded tails.
    pub reach: f32,
    /// Where this centre starts in the cycle, in turns. Offsets exist so four centres drift
    /// as a field rather than as one rigid pattern sliding around.
    pub phase: f32,
    /// This centre's light. Its **alpha is its weight**: a centre at 0.5 contributes half as
    /// much as one at 1.0, and one at zero is off.
    pub tint: Srgba,
}

/// The ambient colour field behind the whole window.
///
/// A property of the **scene**, not of the instance that draws it, and it lives here for the
/// same two reasons [`Environment`] does — there is one field per window rather than one per
/// object, and the 48-byte instance stride is already spent. [`PrimKind::Field`] is the
/// primitive that reads it.
///
/// The default is **empty**, and empty is not a neutral field: with no centre reaching a
/// fragment the wash contributes nothing and the primitive draws its base colour, which is
/// exactly what its CPU floor draws. So a caller that forgot [`DrawList::set_field`] gets the
/// flat ground rather than a field somebody in this crate invented — the palette is where the
/// real answer comes from.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct FieldWash {
    pub centres: [FieldCentre; FIELD_CENTRES],
}

impl FieldWash {
    /// A field from up to [`FIELD_CENTRES`] centres. Extras are dropped, missing ones are off.
    #[must_use]
    pub fn new(centres: &[FieldCentre]) -> Self {
        let mut out = Self::default();
        for (slot, centre) in out.centres.iter_mut().zip(centres) {
            *slot = *centre;
        }
        out
    }
}

/// Bloom: what the frame's brightest parts bleed into everything around them.
///
/// # Why this is not a [`PrimKind`]
///
/// [`PrimKind::needs_backdrop`] predicted bloom would be its third `true` arm, beside
/// [`PrimKind::Blur`] and [`PrimKind::Refract`]. It is not, and the reason is a device limit
/// rather than a preference: the instance pipeline's layout already binds **four** groups —
/// globals, atlas, the blurred backdrop, the sharp backdrop — and four is what WebGPU
/// guarantees. There is no group 4 to hand a bloom texture, and sharing group 2 with the blur
/// would mean a frame carrying both a popover and a bloom could bind only one of the two
/// images, silently, in the middle of a draw span that sets it once.
///
/// So the composite happens in the **resolve** pass, whose layout is its own and had room.
/// That makes bloom a property of the frame, beside [`Environment`] and [`FieldWash`], rather
/// than of any instance — which it always was: nothing about "the bright parts of this window
/// bleed" is located anywhere.
///
/// # What it reads, and what it therefore cannot reach
///
/// The source is the offscreen colour target. On an ordinary frame that target holds the
/// whole window, so bloom covers everything. On a frame with a backdrop-sampling panel the
/// target stops at the backdrop cut, so bloom covers what is *behind* the panel and not the
/// panel itself — which is the right answer anyway, and worth stating because it is a
/// consequence of the source rather than a rule anybody wrote.
///
/// # Both numbers come from the palette
///
/// Neither is authored here. `qs_ui::Tokens::bloom` derives `threshold` from the token ramps
/// and `strength` from the measured lit allowance; this crate has no palette and must not
/// grow one. See that function for where each number comes from.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Bloom {
    /// Relative luminance, in `0..=1`, above which a pixel contributes. Below it a pixel
    /// contributes exactly nothing.
    pub threshold: f32,
    /// How much of the blurred bright-pass is added back, in **linear** light. Zero is off,
    /// and off is the default — a draw list nobody set a bloom on blooms not at all, which is
    /// [`FieldWash`]'s rule and for [`FieldWash`]'s reason.
    pub strength: f32,
}

impl Bloom {
    /// The bloom that does nothing.
    pub const NONE: Self = Self {
        threshold: 1.0,
        strength: 0.0,
    };

    /// Whether this frame has to run the bright pass at all.
    ///
    /// A zero `strength` is off and so is a `threshold` at or above 1.0 — nothing in an 8-bit
    /// target exceeds full white, so a threshold there selects nothing and the three passes
    /// would produce a black image to add nothing from. Checking both here is what keeps the
    /// light theme, whose brightest ordinary surface *is* white, from paying for a chain it
    /// can never see the output of.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.strength > 0.0 && self.threshold < 1.0
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct DrawList {
    pub instances: Vec<Instance>,
    pub batches: Vec<Batch>,
    /// Physical pixels.
    pub viewport: [u32; 2],
    /// Background clear colour.
    pub clear: Srgba,
    /// The sky a [`PrimKind::Pbr`] surface reflects.
    pub environment: Environment,
    /// The ambient colour field a [`PrimKind::Field`] instance draws.
    pub field: FieldWash,
    /// What the frame's brightest parts bleed. See [`Bloom`]; no instance draws it.
    pub bloom: Bloom,
    /// Monotonic; the render thread uses it to tell a re-presented frame from a new one.
    pub generation: u64,
    pub stats: DrawStats,
    /// Time from the triggering input event to draw-list publication.
    ///
    /// Excludes driver and compositor time. `None` for frames without a
    /// triggering input event, such as the initial paint.
    pub input_to_commit: Option<Duration>,
}

impl DrawList {
    /// Drop the contents but keep the allocations.
    ///
    /// Called on the slot the producer is about to write. Steady-state frame building must
    /// not allocate, and the only way to guarantee that is to reuse the same `Vec`s frame
    /// after frame rather than building fresh ones and trusting the allocator.
    pub fn reset(&mut self, viewport: [u32; 2], clear: Srgba, generation: u64) {
        self.instances.clear();
        self.batches.clear();
        self.viewport = viewport;
        self.clear = clear;
        self.environment = Environment::default();
        // Cleared with the rest of the frame, so a field survives exactly one frame and a
        // caller that stops setting it stops getting it. The alternative -- a field that
        // persists across `reset` -- is a scene property that outlives the scene.
        self.field = FieldWash::default();
        // Same rule as the field above, and the same reason: a bloom survives exactly one
        // frame, so a caller that stops setting it stops getting it. `Bloom::default()` is
        // strength zero, which `is_active` reads as off.
        self.bloom = Bloom::default();
        self.generation = generation;
        self.stats = DrawStats::default();
        self.input_to_commit = None;
    }

    /// Push a batch covering every instance added since the last batch ended.
    /// Set the sky lit surfaces reflect this frame.
    ///
    /// Separate from [`DrawList::reset`] rather than an argument to it, because `reset` has
    /// twenty-five call sites and all but one of them are fixtures with no palette to hand.
    /// The application calls this immediately after `reset`; everything else gets the neutral
    /// default and says so.
    pub fn set_environment(&mut self, environment: Environment) {
        self.environment = environment;
    }

    /// The ambient field behind the window, for this frame. See [`FieldWash`].
    pub fn set_field(&mut self, field: FieldWash) {
        self.field = field;
    }

    /// What this frame's bright parts bleed. See [`Bloom`].
    ///
    /// Separate from [`DrawList::reset`] for [`DrawList::set_environment`]'s reason: the
    /// numbers come from the palette and almost every `reset` call site is a fixture with no
    /// palette to hand.
    pub fn set_bloom(&mut self, bloom: Bloom) {
        self.bloom = bloom;
    }

    pub fn end_batch(&mut self, scissor: Option<[u32; 4]>, textured: bool) {
        let start = self.batches.last().map_or(0, |b| b.range.end);
        let end = self.instances.len() as u32;
        if end > start {
            self.batches.push(Batch {
                range: start..end,
                scissor,
                textured,
            });
        }
    }

    /// Instances past the end of the last batch.
    ///
    /// These are uploaded to the instance buffer and covered by no draw call, so they cost
    /// bandwidth and render nothing. Both consumers ([`crate::batcher`] and the CPU
    /// rasterizer) iterate `batches`, never `instances`, which makes the failure completely
    /// silent -- it looks exactly like code that did not run. Anything appending to a list
    /// after someone else has closed the final batch should assert this is zero.
    pub fn unbatched(&self) -> u32 {
        let covered = self.batches.last().map_or(0, |b| b.range.end);
        (self.instances.len() as u32).saturating_sub(covered)
    }

    pub fn finish(&mut self) {
        self.stats.instances = self.instances.len() as u32;
    }
}

// -- the triple buffer -------------------------------------------------------------

/// Bit 0..=1 of the state byte: which slot is ready for the consumer.
const IDX_MASK: u8 = 0b11;
/// Bit 2: the ready slot has not been consumed yet.
const FRESH: u8 = 0b100;

/// Lock-free single-producer / single-consumer handoff of three draw lists.
///
/// # Safety argument
///
/// At any instant the three slot indices are partitioned into exactly three roles --
/// producer-owned, consumer-owned, and "ready" -- and the partition is maintained by a
/// single atomic swap on each side. The producer only ever dereferences the index it holds
/// in `Producer::write`; the consumer only ever dereferences `Consumer::read`. A swap
/// hands an index across and takes a different one back in the same operation, so no index
/// is ever held by both sides. `Producer` and `Consumer` are separate, non-`Clone` types,
/// which is how single-producer/single-consumer is enforced rather than documented.
///
/// The `AcqRel` ordering on both swaps is what makes the *contents* of a slot visible: the
/// producer's writes happen-before its swap, which synchronizes-with the consumer's swap,
/// which happens-before its reads.
/// One slot's payload: the draw list, and beside it the scene the lit mode publishes.
///
/// The scene rides **inside the same slot** rather than through a second channel, and that
/// is the whole of scene-handoff rule 3's enforcement strategy: one atomic swap hands both
/// across, so a frame can never acquire this frame's geometry with last frame's shadows —
/// the tearing a second channel would have to be *kept* from doing is impossible here by
/// construction. `None` is the mode being off; `Some` with an empty slab list is a builder
/// that ran and found nothing, which is a bug worth finding (rule 2: absent is not empty).
struct FrameSlot {
    list: DrawList,
    scene: Option<SceneList>,
}

pub struct DrawListChannel {
    slots: [UnsafeCell<FrameSlot>; 3],
    state: AtomicU8,
}

// SAFETY: see the type-level safety argument. The `UnsafeCell`s are only ever accessed
// through `Producer`/`Consumer`, which hold disjoint indices by construction.
unsafe impl Send for DrawListChannel {}
unsafe impl Sync for DrawListChannel {}

impl std::fmt::Debug for DrawListChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawListChannel").finish_non_exhaustive()
    }
}

/// Build a connected producer/consumer pair.
pub fn draw_list_channel() -> (Producer, Consumer) {
    let empty = || {
        UnsafeCell::new(FrameSlot {
            list: DrawList::default(),
            scene: None,
        })
    };
    let channel = Arc::new(DrawListChannel {
        slots: [empty(), empty(), empty()],
        // Producer starts on slot 0, consumer on slot 1, slot 2 is ready-but-stale.
        state: AtomicU8::new(2),
    });
    (
        Producer {
            channel: Arc::clone(&channel),
            write: 0,
        },
        Consumer { channel, read: 1 },
    )
}

/// The UI-thread half.
#[derive(Debug)]
pub struct Producer {
    channel: Arc<DrawListChannel>,
    write: u8,
}

impl Producer {
    /// The slot to build this frame into. Retains last frame's allocations.
    pub fn slot(&mut self) -> &mut DrawList {
        let index = self.write as usize;
        // SAFETY: `self.write` is owned exclusively by this `Producer` until `publish`
        // swaps it away, and `Producer` is neither `Clone` nor `Sync`.
        unsafe { &mut self.channel.slot_mut(index).list }
    }

    /// The scene half of the same slot, published by the same `publish`.
    ///
    /// Set it every frame: `Some` when the lit mode is on, `None` when it is off. Slots are
    /// recycled, so a producer that only writes it when the mode is on would leave a stale
    /// scene in the slot from three frames ago — which is exactly the pairing
    /// [`Consumer::scene`]'s generation check exists to refuse, and better never published
    /// than published and refused.
    pub fn scene_slot(&mut self) -> &mut Option<SceneList> {
        let index = self.write as usize;
        // SAFETY: as for `slot`.
        unsafe { &mut self.channel.slot_mut(index).scene }
    }

    /// Publish the current slot and take ownership of another.
    ///
    /// Never blocks and never fails. If the consumer has not picked up the previous frame,
    /// this overwrites it -- deliberately. See the module docs on why dropping an
    /// intermediate frame beats queueing it.
    pub fn publish(&mut self) {
        let old = self
            .channel
            .state
            .swap(self.write | FRESH, Ordering::AcqRel);
        self.write = old & IDX_MASK;
    }
}

/// The render-thread half.
#[derive(Debug)]
pub struct Consumer {
    channel: Arc<DrawListChannel>,
    read: u8,
}

impl Consumer {
    /// Take the newest draw list that has not already been consumed.
    ///
    /// `None` means no new frame is available. The caller can keep the previous
    /// frame or leave the window idle.
    pub fn acquire(&mut self) -> Option<&DrawList> {
        if self.channel.state.load(Ordering::Acquire) & FRESH == 0 {
            return None;
        }
        let old = self.channel.state.swap(self.read, Ordering::AcqRel);
        self.read = old & IDX_MASK;
        let index = self.read as usize;
        // SAFETY: the swap transferred exclusive ownership of `old`'s index to this
        // consumer; the producer can no longer reach it.
        Some(unsafe { &self.channel.slot_ref(index).list })
    }

    /// The most recently acquired list, whether or not it is new.
    pub fn current(&self) -> &DrawList {
        // SAFETY: `self.read` is consumer-owned.
        unsafe { &self.channel.slot_ref(self.read as usize).list }
    }

    /// The scene belonging to [`Consumer::current`], or `None` if this frame is not lit.
    ///
    /// **Refuses a mismatched pair** (scene-handoff rule 3): a scene whose generation is not
    /// the draw list's is answered with `None`, exactly as if the mode were off, because the
    /// only way to produce one is a producer that reused a slot without rewriting the scene
    /// half — and lighting this frame's geometry with a stale frame's shadows reads as
    /// latency, which is nearly impossible to attribute. Refusal costs one unlit frame and
    /// is diagnosable; rendering the mismatch is neither.
    pub fn scene(&self) -> Option<&SceneList> {
        // SAFETY: `self.read` is consumer-owned.
        let slot = unsafe { self.channel.slot_ref(self.read as usize) };
        let scene = slot.scene.as_ref()?;
        (scene.generation == slot.list.generation).then_some(scene)
    }
}

impl DrawListChannel {
    /// # Safety
    /// The caller must own `index` per the type-level argument.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slot_mut(&self, index: usize) -> &mut FrameSlot {
        let cell = self.slot_cell(index);
        #[cfg(loom)]
        {
            cell.with_mut(|p| unsafe { &mut *p })
        }
        #[cfg(not(loom))]
        {
            unsafe { &mut *cell.get() }
        }
    }

    /// # Safety
    /// The caller must own `index` per the type-level argument.
    unsafe fn slot_ref(&self, index: usize) -> &FrameSlot {
        let cell = self.slot_cell(index);
        #[cfg(loom)]
        {
            cell.with(|p| unsafe { &*p })
        }
        #[cfg(not(loom))]
        {
            unsafe { &*cell.get() }
        }
    }

    fn slot_cell(&self, index: usize) -> &UnsafeCell<FrameSlot> {
        // The index always comes from a 2-bit field, so it is 0..=3; state is only ever
        // seeded with 0..=2. Clamping rather than indexing keeps a corrupted state byte
        // from becoming an out-of-bounds access.
        match index {
            0 => &self.slots[0],
            1 => &self.slots[1],
            _ => &self.slots[2],
        }
    }
}

// -- thread affinity ---------------------------------------------------------------

/// Debug assertions for UI and render thread affinity.
///
/// Use these guards to detect blocking operations on a frame thread.
/// They are compiled out in release builds.
pub mod affinity {
    use std::sync::OnceLock;
    use std::thread::ThreadId;

    static UI_THREAD: OnceLock<ThreadId> = OnceLock::new();
    static RENDER_THREAD: OnceLock<ThreadId> = OnceLock::new();

    /// Called once, from the thread that will build draw lists.
    pub fn register_ui_thread() {
        let _ = UI_THREAD.set(std::thread::current().id());
    }

    /// Called once, from the thread that will submit to the GPU.
    pub fn register_render_thread() {
        let _ = RENDER_THREAD.set(std::thread::current().id());
    }

    pub fn is_ui_thread() -> bool {
        UI_THREAD
            .get()
            .is_some_and(|&id| id == std::thread::current().id())
    }

    pub fn is_render_thread() -> bool {
        RENDER_THREAD
            .get()
            .is_some_and(|&id| id == std::thread::current().id())
    }

    /// True when the current thread must not block: it is the UI or the Render thread.
    pub fn is_frame_thread() -> bool {
        is_ui_thread() || is_render_thread()
    }

    /// Assert the caller is on the UI thread. No-op in release.
    #[inline]
    pub fn assert_ui_thread(what: &str) {
        debug_assert!(
            UI_THREAD.get().is_none() || is_ui_thread(),
            "{what} must run on the UI thread"
        );
    }

    /// Assert the caller is on the render thread. No-op in release.
    #[inline]
    pub fn assert_render_thread(what: &str) {
        debug_assert!(
            RENDER_THREAD.get().is_none() || is_render_thread(),
            "{what} must run on the render thread"
        );
    }

    /// Assert that a blocking operation is not running on a frame thread.
    ///
    /// Call before file I/O, a potentially contended lock, or a blocking wait.
    #[inline]
    pub fn assert_may_block(what: &str) {
        debug_assert!(
            !is_frame_thread(),
            "{what} blocks and must not run on the UI or Render thread (Constitution I)"
        );
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn instances_appended_after_the_last_batch_are_reported_as_uncovered() {
        // The silent-render bug in one assertion: appending to a list whose final batch is
        // already closed produces instances that no draw call ever reaches.
        let mut list = DrawList::default();
        list.instances
            .push(Instance::rect(0.0, 0.0, 4.0, 4.0, 0.0, Srgba::default()));
        list.end_batch(None, false);
        assert_eq!(list.unbatched(), 0);

        list.instances
            .push(Instance::rect(4.0, 0.0, 4.0, 4.0, 0.0, Srgba::default()));
        assert_eq!(list.unbatched(), 1, "the appended instance is in no batch");

        list.end_batch(None, true);
        assert_eq!(list.unbatched(), 0);
    }

    #[test]
    fn the_instance_stride_is_48_bytes() {
        assert_eq!(size_of::<Instance>(), 48);
        assert_eq!(align_of::<Instance>(), 4);
    }

    #[test]
    fn nothing_is_available_before_the_first_publish() {
        let (_p, mut c) = draw_list_channel();
        assert!(c.acquire().is_none());
    }

    #[test]
    fn a_published_list_is_acquired_exactly_once() {
        let (mut p, mut c) = draw_list_channel();
        p.slot().reset([100, 100], Srgba::TRANSPARENT, 7);
        p.publish();

        assert_eq!(c.acquire().map(|d| d.generation), Some(7));
        assert!(
            c.acquire().is_none(),
            "acquiring twice must not hand back the same frame as though it were new"
        );
    }

    #[test]
    fn the_consumer_sees_only_the_newest_of_several_frames() {
        // This is the behaviour that keeps a slow renderer from replaying a stale fling.
        let (mut p, mut c) = draw_list_channel();
        for generation in 1..=5 {
            p.slot().reset([100, 100], Srgba::TRANSPARENT, generation);
            p.publish();
        }
        assert_eq!(c.acquire().map(|d| d.generation), Some(5));
        assert!(c.acquire().is_none());
    }

    #[test]
    fn the_producer_never_writes_the_slot_the_consumer_holds() {
        let (mut p, mut c) = draw_list_channel();
        p.slot().reset([1, 1], Srgba::TRANSPARENT, 1);
        p.publish();
        let held = c.acquire().unwrap().generation;

        // Publish twice more while the consumer conceptually holds its slot.
        for generation in 2..=3 {
            p.slot().reset([1, 1], Srgba::TRANSPARENT, generation);
            p.publish();
        }
        assert_eq!(held, 1);
        assert_eq!(
            c.current().generation,
            1,
            "the held frame must not be mutated"
        );
    }

    #[test]
    fn a_scene_travels_in_the_same_slot_as_its_draw_list() {
        // Scene-handoff rules 2 and 6: the scene is published by the same swap as the list,
        // and absent is a state the consumer can see -- not an empty stand-in.
        let (mut p, mut c) = draw_list_channel();
        p.slot().reset([100, 100], Srgba::TRANSPARENT, 7);
        let mut scene = crate::scene::SceneList::default();
        scene.reset(7, crate::scene::Environment::default());
        scene.push(crate::scene::Slab {
            rect: [1.0, 2.0, 3.0, 4.0],
            ..crate::scene::Slab::default()
        });
        *p.scene_slot() = Some(scene);
        p.publish();

        assert_eq!(c.acquire().map(|d| d.generation), Some(7));
        assert!(c.scene().is_some(), "the scene published with generation 7");
        let held = c.scene().unwrap();
        assert_eq!(held.generation, 7);
        assert_eq!(held.slabs.len(), 1);
    }

    #[test]
    fn an_unlit_frame_carries_no_scene_even_after_a_lit_one_used_the_slot() {
        // Slots are recycled. A frame with the mode off must read as "no scene", not as
        // whichever scene a previous frame left in the slot -- which is why the producer
        // writes the scene half every frame and the docs on `scene_slot` say so.
        let (mut p, mut c) = draw_list_channel();
        for generation in 1..=4 {
            p.slot().reset([1, 1], Srgba::TRANSPARENT, generation);
            let mut scene = crate::scene::SceneList::default();
            scene.reset(generation, crate::scene::Environment::default());
            *p.scene_slot() = Some(scene);
            p.publish();
        }
        p.slot().reset([1, 1], Srgba::TRANSPARENT, 5);
        *p.scene_slot() = None;
        p.publish();

        assert_eq!(c.acquire().map(|d| d.generation), Some(5));
        assert!(
            c.scene().is_none(),
            "an unlit frame surfaced a stale scene from a recycled slot"
        );
    }

    #[test]
    fn a_scene_with_the_wrong_generation_is_refused_rather_than_rendered() {
        // Scene-handoff rule 3. The only way to build this pair is a producer defect, and
        // the failure it prevents is this frame's geometry lit with last frame's shadows --
        // a shadow lagging its object by one frame during a scroll, which reads as latency
        // and is nearly impossible to attribute.
        let (mut p, mut c) = draw_list_channel();
        p.slot().reset([100, 100], Srgba::TRANSPARENT, 9);
        let mut scene = crate::scene::SceneList::default();
        scene.reset(8, crate::scene::Environment::default());
        *p.scene_slot() = Some(scene);
        p.publish();

        assert_eq!(c.acquire().map(|d| d.generation), Some(9));
        assert!(
            c.scene().is_none(),
            "a scene from generation 8 was handed out with generation 9's draw list"
        );
    }

    #[test]
    fn reset_keeps_capacity_so_steady_state_does_not_allocate() {
        let mut list = DrawList::default();
        for _ in 0..1000 {
            list.instances
                .push(Instance::rect(0.0, 0.0, 1.0, 1.0, 0.0, Srgba::TRANSPARENT));
        }
        let capacity = list.instances.capacity();
        list.reset([0, 0], Srgba::TRANSPARENT, 0);
        assert!(list.instances.is_empty());
        assert_eq!(list.instances.capacity(), capacity);
    }

    #[test]
    fn batches_cover_every_instance_exactly_once() {
        let mut list = DrawList::default();
        let push = |n: usize, list: &mut DrawList| {
            for _ in 0..n {
                list.instances
                    .push(Instance::rect(0.0, 0.0, 1.0, 1.0, 0.0, Srgba::TRANSPARENT));
            }
        };
        push(3, &mut list);
        list.end_batch(None, false);
        push(4, &mut list);
        list.end_batch(Some([0, 0, 10, 10]), true);
        // An empty batch must not be recorded -- a zero-length draw call is pure overhead.
        list.end_batch(None, false);

        assert_eq!(list.batches.len(), 2);
        assert_eq!(list.batches[0].range, 0..3);
        assert_eq!(list.batches[1].range, 3..7);
        let covered: u32 = list
            .batches
            .iter()
            .map(|b| b.range.end - b.range.start)
            .sum();
        assert_eq!(covered as usize, list.instances.len());
    }

    #[test]
    fn prim_kind_all_is_self_consistent() {
        // `ALL` is a hand-written list, so something has to check it against the
        // exhaustive match that is the compile-time guard. Every entry must sit at its
        // own index, and the discriminant must match the position: a variant added to
        // `index` but forgotten in `ALL` shifts one of these and fails here.
        for (i, kind) in PrimKind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i, "{kind:?} is not at index {i} of ALL");
            assert_eq!(
                *kind as u32 as usize, i,
                "{kind:?}'s discriminant is not {i}"
            );
        }
        // Names are distinct, so `shader_const` cannot collapse two variants onto one
        // WGSL constant and make the cross-check in tier_parity vacuously pass.
        let mut names: Vec<&str> = PrimKind::ALL.iter().map(|k| k.shader_const()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PrimKind::ALL.len());
    }

    #[test]
    fn a_floor_is_always_a_primitive_the_cpu_tier_actually_draws() {
        // A floor naming another enhanced kind would be a fallback that itself falls back,
        // and `cpu_floor` resolves exactly one level -- so the instance would reach
        // `tiny-skia` as a primitive nobody checked. One level is the right depth (a chain
        // is a design nobody can picture); this is what makes it safe.
        for kind in PrimKind::ALL {
            if let Fidelity::Enhanced {
                floor: Floor::Plain(plain),
            } = kind.fidelity()
            {
                assert_eq!(
                    plain.fidelity(),
                    Fidelity::Exact,
                    "{kind:?} degrades to {plain:?}, which is itself enhanced"
                );
            }
        }
    }

    #[test]
    fn an_exact_primitive_reaches_the_cpu_tier_unchanged() {
        // The whole pipeline's instances go through `cpu_floor` now, so the identity case
        // is load-bearing: a rect that arrived at `tiny-skia` with its `param` cleared
        // would be a stroke width silently lost, and every existing parity fixture would
        // start measuring something else.
        let instances = [
            Instance::rect(1.0, 2.0, 3.0, 4.0, 5.0, Srgba::new(1.0, 1.0, 1.0, 1.0)),
            Instance::stroke(1.0, 2.0, 3.0, 4.0, 5.0, 1.5, Srgba::new(1.0, 1.0, 1.0, 1.0)),
            Instance::glyph(1.0, 2.0, 3.0, 4.0, [0.1, 0.2, 0.3, 0.4], Srgba::default()),
            Instance::gradient(
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                0.5,
                Srgba::new(1.0, 0.0, 0.0, 1.0),
                Srgba::new(0.0, 0.0, 1.0, 1.0),
            ),
        ];
        for instance in instances {
            assert_eq!(
                instance.cpu_floor(),
                Some(instance),
                "{instance:?} is Exact and must reach the CPU tier byte-for-byte"
            );
        }
    }

    #[test]
    fn a_glow_does_not_reach_the_cpu_tier_at_all() {
        // The other side of `an_exact_primitive_reaches_the_cpu_tier_unchanged`, and the
        // one assertion standing between UXDD 10.7's decision and a rasterizer that quietly
        // draws a hard rectangle where the design said nothing.
        let glow = Instance::glow(
            1.0,
            2.0,
            30.0,
            20.0,
            6.0,
            8.0,
            Srgba::new(0.2, 0.6, 1.0, 0.5),
        );
        assert_eq!(glow.kind, PrimKind::Glow as u32);
        assert_eq!(glow.param, 8.0, "the falloff distance lives in `param`");
        assert_eq!(glow.cpu_floor(), None);
    }

    #[test]
    fn a_plain_glow_puts_the_same_colour_in_both_stops() {
        // `Instance::glow` exists to stop a caller reaching for a transparent outer stop,
        // which fades the halo a second time and tightens it. If the two stops ever stopped
        // agreeing here, that mistake would be back and would still look like a glow.
        let color = Srgba::new(0.2, 0.6, 1.0, 0.5);
        let plain = Instance::glow(0.0, 0.0, 10.0, 10.0, 2.0, 6.0, color);
        assert_eq!(plain.uv, color.to_premul_linear_f32());
        assert_eq!(plain.color, color.to_premul_linear_rgba8());

        // And a negative falloff is clamped rather than dividing the shader by a negative
        // number: `glow_t` reads `param <= 0` as "no reach", not as "reach backwards".
        let degenerate = Instance::glow(0.0, 0.0, 10.0, 10.0, 2.0, -4.0, color);
        assert_eq!(degenerate.param, 0.0);
    }

    #[test]
    fn affinity_guards_pass_when_no_thread_is_registered() {
        // Tests and the bench harness run without a registered UI thread; the guards must
        // not fire there, or every test would need a setup ritual.
        affinity::assert_ui_thread("test");
        affinity::assert_may_block("test");
    }
}
