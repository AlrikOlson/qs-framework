// One pipeline for every primitive M0 draws.
//
// Passes 0 and 1 (rounded-rect fill and stroke) and pass 3 (text) share this shader and are
// discriminated by `kind`. The alternative -- a pipeline per primitive -- costs a pipeline
// switch per batch and forces the batcher to sort by primitive type, which in turn breaks
// the painter's-algorithm ordering a row needs (selection fill under text, status rail over
// both). Branching on a value that is uniform across every instance in a batch costs
// essentially nothing: every invocation in a warp takes the same side.
//
// Geometry is generated, not supplied. Four vertices per instance from `vertex_index`,
// drawn as a triangle strip. There is no vertex buffer for positions and no index buffer;
// the only per-draw traffic is the 48-byte instance.

const KIND_RECT:     u32 = 0u;
const KIND_STROKE:   u32 = 1u;
const KIND_GLYPH:    u32 = 2u;
const KIND_GRADIENT: u32 = 3u;
const KIND_GLOW:     u32 = 4u;
const KIND_RIM:      u32 = 5u;
const KIND_PBR:      u32 = 6u;
const KIND_SWEEP:    u32 = 7u;
const KIND_FIELD:    u32 = 8u;
const KIND_IMAGE:    u32 = 9u;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318531;

// How many centres the ambient field carries. Mirrors `qs_gpu::frame::FIELD_CENTRES`; a
// uniform's layout is fixed when the pipeline is compiled, so this cannot be a length.
const FIELD_CENTRES: u32 = 4u;

struct Globals {
    // Physical pixels.
    viewport: vec2<f32>,
    _pad: vec2<f32>,
    // The environment a PBR surface reflects, as premultiplied linear RGBA. Two stops of an
    // infinite sky: what a ray sees looking along the surface plane, and what it sees looking
    // straight up.
    //
    // A uniform rather than four more instance bytes, because an environment is a property of
    // the scene and not of each object -- every renderer that does this treats it that way,
    // and the 48-byte stride is spent regardless. It arrives from the palette through
    // `DrawList::environment`, so the room a surface reflects is a token and not a constant
    // compiled into this file.
    env_horizon: vec4<f32>,
    env_zenith: vec4<f32>,
    // The ambient field behind the window, here for exactly the reason the environment is:
    // there is one per scene rather than one per object, and four centres are roughly 190
    // bytes against a 48-byte stride. It arrives from the palette through `DrawList::field`.
    //
    // Three vec4s per centre, because a uniform block aligns each member to 16 bytes anyway
    // and packing them tighter would buy nothing but a decode:
    //   place = [x, y, drift_x, drift_y]   normalized viewport coords / widths
    //   tint  = premultiplied linear RGBA, alpha carrying the centre's weight
    //   form  = [reach, phase_offset_in_turns, 0, 0]
    field_place: array<vec4<f32>, 4>,
    field_tint: array<vec4<f32>, 4>,
    field_form: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var atlas_texture: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;
// The colour page. A second texture rather than a second bind group: it is the same cache
// with the same eviction and the same budget, and binding it beside the coverage page costs
// one descriptor rather than a second set_bind_group per batch.
//
// Rgba8UnormSrgb, so this sample is already linear. The stored bytes are STRAIGHT alpha --
// see qs_gpu::atlas::RgbaImage for why premultiplying them would be wrong here specifically.
@group(1) @binding(2) var colour_texture: texture_2d<f32>;

struct InstanceIn {
    // [x, y, width, height], physical pixels, top-left origin.
    @location(0) rect: vec4<f32>,
    // [u0, v0, u1, v1], normalized atlas coordinates.
    @location(1) uv: vec4<f32>,
    // Premultiplied linear RGBA8. Packed on the CPU -- see qs-gpu::color for why.
    @location(2) color: u32,
    @location(3) radius: f32,
    @location(4) param: f32,
    @location(5) kind: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Position relative to the rect centre, in pixels. The SDF is evaluated in this space
    // so that one pixel of distance is one pixel on screen, which is what makes the
    // analytic antialiasing below correct without a screen-space derivative.
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) half_size: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) radius: f32,
    @location(5) @interpolate(flat) param: f32,
    @location(6) @interpolate(flat) kind: u32,
    // The instance's four kind-specific floats, unmodified. `uv` above is the same field
    // *interpolated* across the quad, which is what a glyph wants and what a gradient's
    // far stop must not be -- a colour that varied per fragment would already be a ramp,
    // and the wrong one.
    @location(7) @interpolate(flat) aux: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, inst: InstanceIn) -> VsOut {
    // Triangle-strip corner order: (0,0) (1,0) (0,1) (1,1).
    let corner = vec2<f32>(
        f32(vertex_index & 1u),
        f32((vertex_index >> 1u) & 1u),
    );

    let half_size = inst.rect.zw * 0.5;
    let centre = inst.rect.xy + half_size;

    // Shapes are expanded by one pixel so the antialiased edge has somewhere to live --
    // an SDF evaluated only inside the geometry produces a hard, aliased outer edge.
    // Glyphs are not expanded: their coverage bitmap already carries its own antialiasing,
    // and padding the quad would shear the UV mapping.
    //
    // A glow needs its whole falloff, not one pixel. The fragment stage is only ever run
    // where the quad reaches, so a halo padded by one pixel is a halo cut off square at one
    // pixel -- which reads as clipping rather than as a padding bug, and reads as nothing at
    // all until somebody asks for a larger radius. The +1 on top is the same antialiasing
    // margin every other shape gets; the profile has already reached zero by then.
    var pad = 1.0;
    if (inst.kind == KIND_GLYPH || inst.kind == KIND_IMAGE) {
        // Same reason as the glyph: the quad IS the sampled rectangle, so padding it would
        // shear the UV mapping and read a neighbour's texels through the gutter.
        pad = 0.0;
    } else if (inst.kind == KIND_GLOW) {
        pad = max(inst.param, 0.0) + 1.0;
    }

    let local = (corner * 2.0 - 1.0) * (half_size + vec2<f32>(pad, pad));
    let position = centre + local;

    var out: VsOut;
    out.clip = vec4<f32>(
        position.x / globals.viewport.x * 2.0 - 1.0,
        1.0 - position.y / globals.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.local = local;
    out.half_size = half_size;
    out.uv = mix(inst.uv.xy, inst.uv.zw, corner);
    out.color = unpack4x8unorm(inst.color);
    out.radius = inst.radius;
    out.param = inst.param;
    out.kind = inst.kind;
    out.aux = inst.uv;
    return out;
}

// Signed distance to a rounded box, negative inside. Inigo Quilez's formulation.
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// -- Oklab -----------------------------------------------------------------------------
//
// A gradient interpolates here rather than in linear sRGB. The palette is authored in
// OKLCH so that hue and chroma stay perceptually stable across a ramp; a lerp between two
// of those stops in linear sRGB drifts through a duller hue in the middle while leaving
// both endpoints correct, which is the hard kind of wrong to see.
//
// These three functions mirror `qs_gpu::color::{signed_cbrt, linear_rgb_to_oklab,
// oklab_to_linear_rgb}` term for term. The CPU tier and the `tier_parity` transcription
// both read from that module, so this is the one copy that has to be kept in step by hand.

// WGSL has no cbrt, and `pow` of a negative is NaN.
fn signed_cbrt(x: f32) -> f32 {
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

fn linear_rgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = signed_cbrt(0.41222146 * c.r + 0.53633255 * c.g + 0.051445995 * c.b);
    let m = signed_cbrt(0.21190350 * c.r + 0.68069950 * c.g + 0.107396960 * c.b);
    let s = signed_cbrt(0.08830246 * c.r + 0.28171885 * c.g + 0.629978700 * c.b);
    return vec3<f32>(
        0.21045426 * l + 0.79361780 * m - 0.004072047 * s,
        1.97799850 * l - 2.42859220 * m + 0.450593700 * s,
        0.025904037 * l + 0.78277177 * m - 0.808675770 * s,
    );
}

fn oklab_to_linear_rgb(lab: vec3<f32>) -> vec3<f32> {
    let l = lab.x + 0.39633778 * lab.y + 0.21580376 * lab.z;
    let m = lab.x - 0.105561346 * lab.y - 0.06385417 * lab.z;
    let s = lab.x - 0.089484180 * lab.y - 1.29148550 * lab.z;
    let l3 = l * l * l;
    let m3 = m * m * m;
    let s3 = s * s * s;
    return vec3<f32>(
         4.0767417 * l3 - 3.3077116 * m3 + 0.23096994 * s3,
        -1.2684380 * l3 + 2.6097574 * m3 - 0.34131938 * s3,
        -0.004196086 * l3 - 0.7034186 * m3 + 1.7076147 * s3,
    );
}

// Premultiplied in, straight out. Both stops arrive premultiplied because that is the one
// form the instance buffer carries, and a ramp has to be walked in straight colour or a
// fading stop drags its hue toward black on the way out.
fn unpremultiply(c: vec4<f32>) -> vec3<f32> {
    if (c.a <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return c.rgb / c.a;
}

// -- the dither ---------------------------------------------------------------------------
//
// These four functions mirror `qs_gpu::color::{dither_hash, dither_noise, dither_amplitude,
// dithered}` term for term, and that module carries the reasoning at length. The short
// version: the framebuffer is `Bgra8UnormSrgb`, a ramp squeezed into a handful of output
// levels quantizes into visible contours, and offsetting each pixel by less than half a level
// before the write turns the contour into grain.
//
// The seed is the FRAMEBUFFER pixel rather than `local`, and the hash is integer rather than
// the usual fract-of-a-large-multiple. Both choices are about the CPU tier reaching the same
// number: `local` is only exact when the rect lands on an integer, and a float hash is one
// rounding away from landing on the other side of a discontinuity.

// `2.4 / 1.055`, the slope of the sRGB encode above its toe.
const DITHER_SLOPE: f32 = 2.2748815;
// `1.4 / 2.4`.
const DITHER_EXPONENT: f32 = 0.5833333;
// Where the encode stops being a power law, and the constant slope it has below there.
const DITHER_TOE: f32 = 0.0031308;
const DITHER_TOE_SLOPE: f32 = 0.07739938;

fn dither_hash(x: u32, y: u32) -> u32 {
    // WGSL defines unsigned overflow to wrap, which is what makes this identical to the
    // Rust side's `wrapping_mul` rather than merely close to it.
    var h: u32 = (x * 0x27d4eb2du) ^ (y * 0x165667b1u);
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    h = h * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return h;
}

// The offset for one pixel, in [-0.5, 0.5). The high bits, because the low bits of a
// multiply-shift mixer are its weakest and a dither correlated with pixel parity draws a
// checkerboard.
fn dither_noise(pixel: vec2<u32>) -> f32 {
    return f32(dither_hash(pixel.x, pixel.y) >> 8u) / 16777216.0 - 0.5;
}

// One least-significant bit of the output encoding, expressed in linear light. The branch is
// the sRGB encode's own: below the toe its slope is constant, and a power law extended down
// there reaches 78% of a level at l = 0.002 -- the darkest few per cent of the range, which is
// exactly where a dark theme's ramps live.
fn dither_amplitude(linear: f32) -> f32 {
    let l = max(linear, 0.0);
    let slope = select(DITHER_SLOPE * pow(l, DITHER_EXPONENT), DITHER_TOE_SLOPE, l <= DITHER_TOE);
    return slope / 255.0;
}

// Straight linear RGB, offset by less than half an output level. Alpha is left alone: it is
// coverage, and noise on it would fray the shape's antialiased edge to fix banding that lives
// in the interior.
fn dithered(rgb: vec3<f32>, pixel: vec2<u32>) -> vec3<f32> {
    let noise = dither_noise(pixel);
    let amplitude = vec3<f32>(
        dither_amplitude(rgb.r),
        dither_amplitude(rgb.g),
        dither_amplitude(rgb.b),
    );
    return clamp(rgb + noise * amplitude, vec3<f32>(0.0), vec3<f32>(1.0));
}

// How far along a linear ramp this fragment is: 0 at the near stop's end of the shape, 1 at
// the far stop's.
fn ramp_t(local: vec2<f32>, half_size: vec2<f32>, angle: f32) -> f32 {
    let axis = vec2<f32>(cos(angle), sin(angle));
    // The box's support along the axis: how far it reaches in that direction. Normalizing
    // by it is what puts the two stops on the shape's extremes at *any* angle -- dividing
    // by a fixed half-width instead would make a diagonal ramp finish early and leave a
    // flat band in the far corner.
    let extent = abs(half_size.x * axis.x) + abs(half_size.y * axis.y);
    if (extent <= 0.0) {
        return 0.5;
    }
    return clamp(dot(local, axis) / extent * 0.5 + 0.5, 0.0, 1.0);
}

// How far around the shape this fragment is, as a position on the SAME two-stop ramp a
// linear gradient walks: 0 at the near stop, 1 at the far one.
//
// The mirror is the whole design. An angle wraps, so a ramp laid straight onto one would
// step from the far stop back to the near stop wherever atan2 crosses back to -PI -- a hue
// seam, fixed in place on the shape, that no amount of phase can move out of sight. Walking
// near -> far -> near over a full turn removes it by construction rather than by tolerance:
// `t` is continuous across the wrap because both sides of it are zero. What is left at the
// two turning points is a change of slope, which is what a mirrored two-stop conic looks
// like and is the picture this primitive is for.
//
// The angle is taken in the shape's own normalized space, so a wide row spreads the sweep
// evenly around its perimeter instead of spending most of the turn on its two short sides.
//
// `phase` is in radians and turns the highlight the same way a gradient's `angle` turns its
// axis: at phase 0 the far stop sits at +x, and increasing it carries the highlight around
// through +y, which on this surface is downward.
fn sweep_t(local: vec2<f32>, half_size: vec2<f32>, phase: f32) -> f32 {
    let extent = max(half_size, vec2<f32>(1e-4, 1e-4));
    let n = local / extent;
    // atan2(0, 0) is an indeterminate value in WGSL and 0.0 in Rust, so the one fragment
    // that can reach the exact centre is answered here rather than left to the platform --
    // the CPU tier and this shader have to agree about it pixel for pixel.
    var a = 0.0;
    if (n.x != 0.0 || n.y != 0.0) {
        a = atan2(n.y, n.x);
    }
    let f = fract((a - phase) / TAU + 0.5);
    return 1.0 - abs(f * 2.0 - 1.0);
}

// The two-stop ramp's colour at one fragment, premultiplied, ready to be scaled by coverage.
//
// One helper for both ramps: a conic sweep differs from a linear gradient only in how `t` is
// found, so everything downstream of that -- the Oklab walk, the linear alpha, the gamut
// clamp, the dither -- is shared rather than copied. A second copy is how the two would come
// to disagree about the palette.
fn ramp_at(near: vec4<f32>, far: vec4<f32>, t: f32, pixel: vec2<u32>) -> vec4<f32> {
    let lab = mix(
        linear_rgb_to_oklab(unpremultiply(near)),
        linear_rgb_to_oklab(unpremultiply(far)),
        t,
    );
    // Alpha is a coverage fraction, not a colour, so it lerps linearly -- there is no
    // perceptual space for "how much of this is there".
    let a = mix(near.a, far.a, t);
    // The straight line between two in-gamut Oklab colours can leave sRGB in the middle
    // when the stops are far apart in hue. Clamping is the cheap answer; the correct one
    // is a per-fragment chroma bisection, which is not something a fill shader should run.
    let rgb = dithered(clamp(oklab_to_linear_rgb(lab), vec3<f32>(0.0), vec3<f32>(1.0)), pixel);
    return vec4<f32>(rgb * a, a);
}

// -- the field ----------------------------------------------------------------------------
//
// The one primitive whose subject is the window rather than a component, and the only one that
// reads a scene property rather than carrying its own data. Several coloured centres drift on
// the phase; where they reach, the ground takes their colour.
//
// Two properties do the load-bearing work, and both are about cost. The falloff has **bounded
// support**, so a centre past its reach contributes exactly zero rather than an ever-smaller
// tail -- which is what keeps a full-viewport pass to four `max`es per fragment. And the
// centres are blended with **each other** in Oklab and then composited over the base as one
// colour, rather than each being composited in turn: N-1 fewer blends, and it is also the only
// version whose worst case the contrast gate can check, since a single composite over the base
// is the same arithmetic `Srgba::over` runs on the CPU.

// How strongly one centre reaches a point, `0` at its limit and beyond.
//
// Quartic with bounded support: `(1 - (d/r)^2)^2` is one at the centre, zero at the reach, and
// has zero slope at both ends, so a centre fades out without a visible ring where its influence
// stops -- the same reason the glow's profile is quadratic rather than linear, one power up
// because this is spread across a window rather than a few pixels.
//
// A reach of zero is answered rather than divided by: a centre with no reach is off.
fn field_weight(delta: vec2<f32>, reach: f32) -> f32 {
    if (reach <= 0.0) {
        return 0.0;
    }
    let t = clamp(dot(delta, delta) / (reach * reach), 0.0, 1.0);
    let falloff = 1.0 - t;
    return falloff * falloff;
}

// The whole field at one fragment, composited over `base` and dithered, premultiplied.
//
// `uv01` is the fragment's position across the viewport in `0..1`. `aspect` converts the y
// axis into units of the viewport's *width*, so a centre is a circle rather than an ellipse
// that changes shape when the window is resized.
fn field(base: vec4<f32>, uv01: vec2<f32>, aspect: f32, phase: f32, amplitude: f32, pixel: vec2<u32>) -> vec4<f32> {
    var total = 0.0;
    var lab = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < FIELD_CENTRES; i = i + 1u) {
        let place = globals.field_place[i];
        let tint = globals.field_tint[i];
        let form = globals.field_form[i];

        // Where this centre has drifted to. Its own offset is what keeps four centres from
        // sliding around as one rigid pattern.
        let angle = phase + form.y * TAU;
        let at = place.xy + place.zw * vec2<f32>(cos(angle), sin(angle));
        let delta = vec2<f32>(uv01.x - at.x, (uv01.y - at.y) * aspect);

        // The tint's alpha IS the centre's weight, so a centre authored faint contributes
        // less rather than contributing the same colour more faintly -- which is what lets
        // the stops the gate checks be the tints themselves.
        let weight = field_weight(delta, form.x) * tint.a;
        total = total + weight;
        lab = lab + linear_rgb_to_oklab(unpremultiply(tint)) * weight;
    }

    if (total <= 0.0) {
        // No centre reaches here, so the field IS the base -- which is also exactly what the
        // CPU floor draws. The two agreeing at the limit is what makes the floor a limit
        // rather than a substitute.
        return base;
    }

    // The weighted mean in Oklab. Two overlapping centres in premultiplied linear would sag
    // through a duller hue in the middle, which across a window is a grey patch where two
    // colours meet.
    let mixed = oklab_to_linear_rgb(lab / total);
    let coverage = clamp(total, 0.0, 1.0) * clamp(amplitude, 0.0, 1.0);
    let wash = vec4<f32>(clamp(mixed, vec3<f32>(0.0), vec3<f32>(1.0)) * coverage, coverage);

    // Source-over in premultiplied linear, which is the same composite the blend state does
    // and the same one `Srgba::over` does on the CPU -- so a fragment where one centre
    // dominates lands on exactly the colour the contrast gate checked.
    let out = wash + base * (1.0 - coverage);

    // Dithered, because this is the largest surface in the window and the interval it moves
    // across is a handful of 8-bit levels: without the noise a field this subtle bands, and
    // banding across a whole window is the most visible artefact in the application.
    let straight = unpremultiply(out);
    return vec4<f32>(dithered(straight, pixel) * out.a, out.a);
}

// The gradient's colour at one fragment. Kept as a named function rather than inlined at the
// one call site so that `ramp` and `ramp_at` read as the pair they are.
fn ramp(near: vec4<f32>, far: vec4<f32>, local: vec2<f32>, half_size: vec2<f32>, angle: f32, pixel: vec2<u32>) -> vec4<f32> {
    return ramp_at(near, far, ramp_t(local, half_size, angle), pixel);
}

// -- the glow ----------------------------------------------------------------------------
//
// A blur needs neighbouring pixels: a second pass, a second render target, and a
// derivative-free formulation nobody has. A signed distance field already tells this
// fragment how far it is from the shape, so the falloff is a function of a number the
// shader is holding anyway. That is the whole reason a halo is reachable inside a
// single-pass pipeline that forbids derivatives.

// How far along the falloff this fragment is: 0 at the shape's edge and everywhere inside
// it, 1 at the limit and beyond.
//
// A falloff of zero degenerates to the plain fill rather than dividing by it -- the shape
// stays solid and nothing surrounds it, which is the sensible reading of "a glow with no
// reach" and keeps the function total.
fn glow_t(distance: f32, falloff: f32) -> f32 {
    if (falloff <= 0.0) {
        return select(1.0, 0.0, distance <= 0.0);
    }
    return clamp(distance / falloff, 0.0, 1.0);
}

// The halo's tint, mixed in **premultiplied linear** -- deliberately not in Oklab, which is
// the opposite of the choice `ramp` makes twenty lines up.
//
// A gradient's stops are two different opaque colours, and a linear lerp between them sags
// through a duller hue in the middle. A glow's stops are a colour and its own fade to
// nothing, and `unpremultiply` has no hue to recover from a zero-alpha stop -- it returns
// black, so walking that ramp perceptually would drag the halo through black on the way out
// and leave a dark rim around every glowing thing. Premultiplied, the same colour at
// decreasing alpha stays exactly the same colour, which is what "less of this light" means.
fn halo(near: vec4<f32>, far: vec4<f32>, t: f32) -> vec4<f32> {
    return mix(near, far, t);
}

// -- the rim -------------------------------------------------------------------------------
//
// The same distance field as the glow, read with the opposite sign. `glow_t` walks outward
// from the boundary; this walks inward from it, which is the only difference between a halo
// around a shape and a light along the inside of its edge.

// How far inside the shape this fragment is, as a fraction of the rim's width: 0 at the
// boundary and everywhere outside it, 1 at `width` inward and deeper.
//
// A width of zero returns 1 rather than dividing by it, so a rim with no width is no rim --
// the same shape of answer `glow_t` gives to a falloff of zero, and total for the same
// reason.
fn rim_t(distance: f32, width: f32) -> f32 {
    if (width <= 0.0) {
        return 1.0;
    }
    return clamp(-distance / width, 0.0, 1.0);
}

// -- the surface -------------------------------------------------------------------------
//
// Everything above tints a flat shape. This gives one a *surface*: a bevelled edge with a
// real normal, shaded by a real microfacet BRDF.
//
// The reason it fits a pipeline that forbids derivatives and second passes is that
// `sd_rounded_box` has an **analytic gradient**. The distance already tells a fragment how
// far it is from the edge; the gradient tells it which way the edge is. Those two numbers are
// a normal, exactly, with no `fwidth` and no neighbouring pixel -- which is also what lets the
// CPU-side transcription reproduce it and keeps `tier_parity` meaningful.
//
// A note on what this deliberately is not. Hardware ray tracing wants an acceleration
// structure over scene geometry, and a list of rectangles has none; it would also cost the
// Reduced tier, the CPU tier and SC-009. The ray tracing that *is* here is the part with a
// closed form: intersecting the reflected ray with an infinite environment, which needs no
// structure to traverse and no second pass to sample.

// The gradient of `sd_rounded_box`: a unit vector pointing away from the shape, i.e. the
// direction of the nearest edge.
//
// Derived rather than sampled. Outside the inner rectangle the nearest feature is a corner
// arc or a flank, and the gradient is the normalized outward part of `q`; inside it, the
// nearest edge is whichever axis is closer, so the gradient is that axis. `sign(p)` carries it
// back out of the `abs` the distance function folds the shape into.
fn sd_rounded_box_grad(p: vec2<f32>, b: vec2<f32>, r: f32) -> vec2<f32> {
    let q = abs(p) - b + vec2<f32>(r, r);
    let s = sign(p);
    if (max(q.x, q.y) > 0.0) {
        // At least one component is positive, so `max(q, 0)` is non-zero and normalizing it
        // is safe -- the case that would divide by zero cannot reach here.
        return normalize(max(q, vec2<f32>(0.0, 0.0))) * s;
    }
    if (q.x > q.y) {
        return vec2<f32>(s.x, 0.0);
    }
    return vec2<f32>(0.0, s.y);
}

// The surface normal of a quarter-round bevel `width` pixels wide.
//
// The profile is a quarter circle: at the boundary the surface is vertical, so the normal
// points straight out sideways and the Fresnel term goes to one -- which is what makes an
// edge catch light. `width` pixels inward the surface is flat and the normal points at the
// viewer. The angle sweeps linearly between the two, which is a circular arc in cross-section.
//
// A zero width is answered rather than divided by: a surface with no bevel is flat, and its
// normal is straight up everywhere.
fn bevel_normal(distance: f32, grad: vec2<f32>, width: f32) -> vec3<f32> {
    if (width <= 0.0) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let t = clamp(-distance / width, 0.0, 1.0);
    let theta = (1.0 - t) * PI * 0.5;
    // Already unit length: `grad` is unit in 2D and (sin, cos) is unit in the plane spanned by
    // it and the view axis. Normalizing again would only add rounding.
    return vec3<f32>(grad * sin(theta), cos(theta));
}

// What a ray sees. The environment is an infinite sky, so the intersection is a closed form:
// the ray's elevation alone decides the answer, and there is nothing to traverse.
fn environment(ray: vec3<f32>) -> vec3<f32> {
    let t = clamp(ray.z * 0.5 + 0.5, 0.0, 1.0);
    return mix(unpremultiply(globals.env_horizon), unpremultiply(globals.env_zenith), t);
}

// GGX / Trowbridge-Reitz normal distribution: what fraction of the microfacets are oriented to
// reflect the light straight at the viewer.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith height-correlated visibility, in the form that has already absorbed the BRDF's
// `1 / (4 (N.L) (N.V))` denominator -- so the specular term below is `D * Vis * F` and not
// `D * G * F / (4 ...)`. Height-correlated rather than separable because the separable form
// over-darkens grazing angles, which on a bevel is the entire visible area.
fn visibility_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let lv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let ll = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(lv + ll, 1e-5);
}

// Schlick's approximation to the Fresnel term.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let f = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    return f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * f;
}

// The same, weakened by roughness, for the environment term. A rough surface cannot show a
// sharp grazing reflection, and the unmodified Schlick term would give it one -- the standard
// artefact of applying a mirror's Fresnel to a matte surface.
fn fresnel_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let f = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * f;
}

// One directional key light, from above and slightly to the left.
//
// A convention rather than a parameter: interfaces are lit from above because that is where
// light comes from in the world the reader is sitting in, and a surface lit from below reads
// as inverted no matter how correct its BRDF is. Fixed here so a material cannot author a
// physically fine surface that disagrees with every other surface in the window.
const LIGHT_DIR: vec3<f32> = vec3<f32>(-0.32, -0.55, 0.77);

// The key light's radiance, and the one number here that is an exposure choice rather than a
// physical constant.
//
// A Lambertian surface returns `albedo * (N.L) / PI`, so a unit light leaves a flat surface at
// roughly a quarter of its own albedo. That is correct, and in a design system it is also
// unusable: the albedo of a material IS an authored token, and a lit surface that renders it
// four times darker would force every colour in `design/tokens.json` to be re-picked to mean
// what it already means. So the light is scaled to put a flat, front-facing dielectric back at
// its albedo -- `PI / (N.L)` at the flat normal. Nothing about the BRDF's *relative* behaviour
// changes; this is the film speed, not the physics.
const LIGHT_RADIANCE: f32 = 4.0757;

// Cook-Torrance, with an orthographic viewer.
//
// The view direction is (0, 0, 1) for every fragment -- a UI has no perspective -- which is
// why `n_dot_v` is just the normal's z and no per-fragment view vector is built.
// How much of a surface's emission reaches one fragment: all of it at the boundary, none of it
// `bevel` pixels inward.
//
// The rim's profile, deliberately reused rather than re-derived. Emission is only affordable at
// all because it is edge-localised -- contract rule 1a lets a meaning-bearing surface emit and
// forbids it to light the ground directly behind itself, and puts that boundary at the bevel --
// so the function that decides "how far inside the edge am I" must be the same one the rim uses,
// or there are two answers to one question.
//
// Squared for the same reason the rim's fade is: a linear ramp has a visible line where its
// slope stops, and light does not.
//
// A surface with no bevel does not emit. `rim_t` answers a zero width with 1, so this returns 0,
// which is the sensible reading of "a flat surface has no edge to catch light on".
fn edge_emission(distance: f32, bevel: f32) -> f32 {
    let edge = 1.0 - rim_t(distance, bevel);
    return edge * edge;
}

// The LAMP profile: a surface that emits across the whole of itself, brighter at the rim,
// like a frosted panel with a source behind it.
//
// `edge_emission` above confines emission to `bevel` of the boundary; this deliberately does
// not, and the difference is the difference between a surface with a glowing edge and a
// surface that IS a light. The interior sits at `LAMP_FLOOR` of full strength and the last
// `bevel` climbs to 1, so the panel reads as lit through rather than outlined.
//
// **This spends the geometric half of lit-contrast rule 1a and replaces it with a measured
// one.** The old guarantee was that emission was identically zero under any text, so a
// material could turn it up without moving a composite. A lamp lights its own label's
// ground, so what bounds legibility now is the authored strength and the measurement in
// `docs/lit-mode/` — not the shape of the falloff. `a_lamp_keeps_its_label_legible` in
// `tier_parity` is that bound as a test.
// How dim the panel gets away from its centre-line. Not zero: a lamp's housing still glows.
const LAMP_FLOOR: f32 = 0.34;
// Rib period, in physical pixels, and rib depth. The period is deliberately close to a row's
// own height so the ridges read as structure at the size a row actually is, rather than as
// moire at some other scale.
const LAMP_RIB_PERIOD: f32 = 17.0;
const LAMP_RIB_DEPTH: f32 = 0.16;

fn lamp_emission(distance: f32, bevel: f32, local: vec2<f32>, half_size: vec2<f32>) -> f32 {
    // The tube. A lamp is brightest along its axis and falls away toward the housing, and
    // for a surface that is far wider than it is tall the axis is horizontal — so this is a
    // function of the surface's own `y`, which keeps the hot line pinned to the panel rather
    // than to the screen when the row scrolls.
    let v = clamp(local.y / max(half_size.y, 1.0), -1.0, 1.0);
    let tube = pow(max(1.0 - abs(v), 0.0), 0.55);
    let body = LAMP_FLOOR + (1.0 - LAMP_FLOOR) * tube;

    // The housing's lip: the last `bevel` catches a little extra, which is what stops the
    // panel from ending in mid-air.
    let rim = 1.0 - rim_t(distance, max(bevel, 1.0));
    let lip = rim * rim * 0.30;

    // The texture, and it is what makes this read as a made object rather than a gradient.
    // Ribs across the panel like a frosted tube's ridges, plus a fine grain from the same
    // kind of hash the dither uses. Both are analytic — no derivatives, per this file's
    // header — and both are functions of the surface's own coordinates, so they travel with
    // the row instead of swimming under it.
    let ribs = 1.0 - LAMP_RIB_DEPTH * (0.5 + 0.5 * cos(local.x * (6.2831853 / LAMP_RIB_PERIOD)));
    let grain = 0.93 + 0.07 * fract(sin(dot(local, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    return (body + lip) * ribs * grain;
}

fn shade_pbr(
    normal: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    env_strength: f32,
    emissive: f32,
) -> vec3<f32> {
    let rough = clamp(roughness, 0.045, 1.0);
    let metal = clamp(metallic, 0.0, 1.0);

    let v = vec3<f32>(0.0, 0.0, 1.0);
    let l = normalize(LIGHT_DIR);
    let h = normalize(l + v);

    let n_dot_v = max(normal.z, 1e-4);
    let n_dot_l = max(dot(normal, l), 0.0);
    let n_dot_h = max(dot(normal, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // 0.04 is the reflectance of a common dielectric at normal incidence. A metal has no
    // diffuse lobe and tints its specular with its own albedo, which is the whole of what
    // "metallic" means in this parameterisation.
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metal);

    let d = distribution_ggx(n_dot_h, rough);
    let vis = visibility_smith(n_dot_v, n_dot_l, rough);
    let f = fresnel_schlick(v_dot_h, f0);
    let specular = d * vis * f;

    // Energy conservation: what the specular lobe reflected is not available to scatter
    // diffusely. Without this a rough metal is brighter than the light that lit it.
    let kd = (vec3<f32>(1.0, 1.0, 1.0) - f) * (1.0 - metal);
    let diffuse = kd * albedo / PI;
    // `env_strength` is a light *rig*, not a gain: it is the fraction of the illumination that
    // arrives from the sky rather than from the key light, so the two sum to one and a flat
    // dielectric returns its albedo whatever the mix. Adding a full sky on top of a full key
    // light is what the first attempt did, and a near-white albedo in the light theme clipped
    // to pure white -- physically what two bright sources do, and useless for a palette whose
    // colours are supposed to survive being lit.
    let sky_mix = clamp(env_strength, 0.0, 1.0);
    let direct = (diffuse + specular) * n_dot_l * LIGHT_RADIANCE * (1.0 - sky_mix);

    // The environment, along the reflected ray. `reflect(-v, n)` with v = (0,0,1) is
    // `2 n n.z - v`, and the sky it lands in has a closed form -- see `environment`.
    let reflected = 2.0 * normal * n_dot_v - v;
    let env_specular = environment(reflected) * fresnel_roughness(n_dot_v, f0, rough);
    // The diffuse half of the environment is the sky as seen along the normal itself, which
    // is the cheapest honest stand-in for an irradiance probe.
    let env_diffuse = environment(normal) * albedo * (1.0 - metal);
    let ambient = (env_diffuse + env_specular) * sky_mix;

    // Emission is added rather than mixed, because a light is not a different surface -- it is
    // this surface plus light. It returns the albedo, so a glowing thing glows its own colour
    // and there is no second token to keep in step with the first.
    //
    // `emissive` arrives already weighted by `edge_emission`, so it is zero everywhere deeper
    // than the bevel. That is the contrast argument and it is why this line is safe: whatever
    // this term does, it does not do it under a label.
    let emission = albedo * max(emissive, 0.0);

    return direct + ambient + emission;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.kind == KIND_GLYPH) {
        // R8 coverage. The colour is already premultiplied, so scaling the whole vector by
        // coverage keeps it premultiplied -- which is what the One / OneMinusSrcAlpha blend
        // state expects.
        let coverage = textureSample(atlas_texture, atlas_sampler, in.uv).r;
        return in.color * coverage;
    }

    if (in.kind == KIND_IMAGE) {
        // Rgba8UnormSrgb: rgb arrives linear, a arrives untouched, and both are STRAIGHT.
        // Premultiply here, after the hardware's sRGB decode -- doing it before would
        // compute srgb_to_linear(c * a) where premultiplied blending needs
        // srgb_to_linear(c) * a, and the two part company worst at a soft edge.
        let texel = textureSample(colour_texture, atlas_sampler, in.uv);
        let picture = vec4<f32>(texel.rgb * texel.a, texel.a);
        // in.color is a premultiplied tint. White at full alpha is the identity, and a
        // lower alpha scales colour and coverage together, which is what a fade is.
        return picture * in.color;
    }

    // Clamp the radius to what the rectangle can actually hold. An unclamped radius larger
    // than half the shorter side inverts the SDF and renders a bow-tie.
    let radius = clamp(in.radius, 0.0, min(in.half_size.x, in.half_size.y));
    let distance = sd_rounded_box(in.local, in.half_size, radius);

    var alpha: f32;
    if (in.kind == KIND_STROKE) {
        // Distance to the centre-line of a band of width `param`, then the same one-pixel
        // analytic coverage as the fill.
        let half_width = in.param * 0.5;
        alpha = clamp(0.5 - (abs(distance + half_width) - half_width), 0.0, 1.0);
    } else if (in.kind == KIND_GLOW) {
        // Quadratic, not linear: a linear ramp reads as a cone with a visible outer edge
        // where its slope stops, and a glow is mostly its bright near half. The curve is
        // continuous across the shape's boundary -- solid inside, falling from the edge
        // outward -- so there is no edge here to antialias and no `0.5 - d` to apply.
        let fade = 1.0 - glow_t(distance, in.param);
        alpha = fade * fade;
    } else if (in.kind == KIND_RIM) {
        // Quadratic inward, for the same reason the glow's is quadratic outward: a linear
        // ramp has a visible line where its slope stops, and a rim is mostly its bright near
        // half. Multiplied by the fill's own coverage rather than replacing it -- that factor
        // is what keeps the light inside the shape, gives the outer edge the same one-pixel
        // antialiasing every other fill has, and is why a rim needs no quad padding.
        //
        // The peak therefore sits half a pixel inside the boundary rather than exactly on it:
        // at the boundary the shape itself is only half covered, and a light brighter than
        // the surface carrying it would be drawing outside the shape.
        let fade = 1.0 - rim_t(distance, in.param);
        alpha = clamp(0.5 - distance, 0.0, 1.0) * fade * fade;
    } else {
        // `0.5 - d` is exact one-pixel-wide coverage for a distance field measured in
        // pixels: fully covered at d = -0.5, empty at d = +0.5. No `fwidth`, no smoothstep,
        // no derivative -- which also means it is correct on the CPU rasterizer, where
        // derivatives do not exist.
        alpha = clamp(0.5 - distance, 0.0, 1.0);
    }

    // Coverage and colour are separable: a gradient is a `KIND_RECT` whose tint varies
    // across the shape, so it takes the fill's coverage above unchanged and only replaces
    // what is being covered. Writing it as a third branch of the coverage `if` would have
    // duplicated the exact `0.5 - d` the CPU tier is matched against.
    var tint = in.color;
    if (in.kind == KIND_GRADIENT) {
        // `in.clip` is `@builtin(position)`, which in the fragment stage is the framebuffer
        // coordinate with its centre at (x + 0.5, y + 0.5). Flooring it is therefore the
        // integer pixel exactly, with no dependence on where the rect happens to sit -- which
        // is the property the CPU tier needs to reach the same dither value.
        let pixel = vec2<u32>(floor(in.clip.xy));
        tint = ramp(in.color, in.aux, in.local, in.half_size, in.param, pixel);
    } else if (in.kind == KIND_FIELD) {
        // The fragment's place across the viewport. Derived from the quad rather than from
        // `@builtin(position)` so the field belongs to the *instance*: a caller that draws it
        // over part of the window gets the whole field in that part, not a window-sized field
        // cropped to it.
        let pixel = vec2<u32>(floor(in.clip.xy));
        let uv01 = in.local / max(in.half_size, vec2<f32>(1e-4, 1e-4)) * 0.5 + vec2<f32>(0.5, 0.5);
        let aspect = in.half_size.y / max(in.half_size.x, 1e-4);
        tint = field(in.color, uv01, aspect, in.param, in.aux.x, pixel);
    } else if (in.kind == KIND_SWEEP) {
        // The same fill coverage and the same two stops as the gradient above; only the
        // parameter differs, which is why this shares `ramp_at` rather than restating the
        // Oklab walk. `in.param` is the sweep's phase, in radians.
        let pixel = vec2<u32>(floor(in.clip.xy));
        tint = ramp_at(in.color, in.aux, sweep_t(in.local, in.half_size, in.param), pixel);
    } else if (in.kind == KIND_GLOW) {
        // The tint carries hue only; the coverage above owns the fade. Letting the outer
        // stop's alpha fade it a second time would multiply two ramps together and pull the
        // halo in tight against the shape -- which is why `Instance::glow` writes the same
        // colour into both stops and `glow_two_tone` is the form that spends the difference
        // on purpose.
        tint = halo(in.color, in.aux, glow_t(distance, in.param));
    }
    else if (in.kind == KIND_PBR) {
        // The coverage is the fill's, untouched: a lit surface occupies exactly the shape a
        // flat one would, and only what is *inside* it differs. That separation is also what
        // makes the floor honest -- drop the shading and the same pixels are still covered.
        let grad = sd_rounded_box_grad(in.local, in.half_size, radius);
        let normal = bevel_normal(distance, grad, in.aux.x);
        let straight = unpremultiply(in.color);
        // `in.param` is the emissive strength. A lamp emits across its whole face; the
        // profile carries the rim falloff and the texture. See `lamp_emission`.
        let emissive = in.param * lamp_emission(distance, in.aux.x, in.local, in.half_size);
        let lit = shade_pbr(normal, straight, in.aux.y, in.aux.z, in.aux.w, emissive);
        // Back to premultiplied, which is the one form this pipeline's blend state accepts.
        tint = vec4<f32>(lit * in.color.a, in.color.a);
    }
    // KIND_RIM has no arm here on purpose, and the absence is the design rather than an
    // omission. Its ramp runs from one colour to nothing, so the whole of it lives in the
    // coverage above and the tint is constant across it. `return tint * alpha` is then a walk
    // in premultiplied linear -- the same colour at decreasing alpha, bit for bit -- which is
    // what `halo` spends a paragraph arriving at and what routing a rim through `ramp` would
    // undo, since `unpremultiply` has no hue to recover from the stop at the far end.

    return tint * alpha;
}
