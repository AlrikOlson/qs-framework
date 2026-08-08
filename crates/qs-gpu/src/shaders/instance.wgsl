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

const KIND_RECT:   u32 = 0u;
const KIND_STROKE: u32 = 1u;
const KIND_GLYPH:  u32 = 2u;

struct Globals {
    // Physical pixels.
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var atlas_texture: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

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
    var pad = 1.0;
    if (inst.kind == KIND_GLYPH) {
        pad = 0.0;
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
    return out;
}

// Signed distance to a rounded box, negative inside. Inigo Quilez's formulation.
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
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
    } else {
        // `0.5 - d` is exact one-pixel-wide coverage for a distance field measured in
        // pixels: fully covered at d = -0.5, empty at d = +0.5. No `fwidth`, no smoothstep,
        // no derivative -- which also means it is correct on the CPU rasterizer, where
        // derivatives do not exist.
        alpha = clamp(0.5 - distance, 0.0, 1.0);
    }

    return in.color * alpha;
}
