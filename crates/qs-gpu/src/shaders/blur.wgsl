// The backdrop blur chain: downsample, then a separable Gaussian, at a quarter of the
// viewport's resolution in each axis.
//
// Three fragment entry points over one fullscreen triangle, and no uniform buffer at all --
// every pass reads its own source size with `textureDimensions`, so there is nothing to keep
// in step with the texture and nothing to write per frame. The constants below are the whole
// specification of the kernel; `qs_gpu::target` states where each number comes from and pins
// it with a test.
//
// # Why a triangle, and no derivatives
//
// Both for `resolve.wgsl`'s reasons, which apply here unchanged: a fullscreen triangle has no
// diagonal seam, and `specs/002-ray-traced-mode/contracts/scene-effect.md` plus `tier_parity`
// both rest on there being no `fwidth`, `dpdx` or `dpdy` anywhere in this project's shaders.
// The sampled level is stated as 0.0 because these textures have exactly one mip level.
//
// # Linear, not sRGB, arithmetic
//
// Every texture in the chain carries the surface's own format. An sRGB texture is decoded to
// linear on sample and re-encoded on write, so all the weighted sums below happen in linear
// light -- which is the only space a blur is a blur. Filtering sRGB code values instead
// darkens every soft edge, most visibly where a bright accent meets a dark ground, and looks
// like the blur has a shadow in it.

// The Gaussian's standard deviation, in texels of the DOWNSAMPLED image. Must equal
// `qs_gpu::target::BLUR_SIGMA_TEXELS`.
const SIGMA: f32 = 4.0;
// Where the kernel is truncated, in the same texels: 2.5 * SIGMA, rounded. Must equal
// `qs_gpu::target::BLUR_RADIUS_TEXELS`.
const RADIUS: i32 = 10;
// How many adjacent pairs the taps beyond the centre fold into. RADIUS is even, so the
// 2 * RADIUS taps either side of centre pair exactly and nothing is left over. An odd RADIUS
// would leave one unpaired tap per side, which is why the constant is chosen even rather than
// handled with a branch nobody would notice was never taken.
const PAIRS: i32 = RADIUS / 2;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    // The standard oversized triangle: (-1,-1), (3,-1), (-1,3) in clip space, whose
    // intersection with the unit square is exactly the viewport.
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

fn source_texel() -> vec2<f32> {
    return 1.0 / vec2<f32>(textureDimensions(source_texture, 0));
}

fn gaussian(x: f32) -> f32 {
    return exp(-(x * x) / (2.0 * SIGMA * SIGMA));
}

// Downsample the full-resolution backdrop by four in each axis.
//
// Four bilinear taps, at +/- one SOURCE texel from the output texel's centre in each axis.
// Each tap sits exactly on a 2x2 texel corner, so the hardware's bilinear filter returns that
// block's unweighted average -- and four such blocks tile the 4x4 region exactly, with every
// texel counted once. So this is the exact 4x4 box average in four samples rather than
// sixteen, and it is exact because the factor is a power of two: at any other factor the taps
// land off-corner and the "average" acquires weights nobody chose.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let texel = source_texel();
    let sum = textureSampleLevel(source_texture, source_sampler, in.uv + vec2<f32>(-texel.x, -texel.y), 0.0)
        + textureSampleLevel(source_texture, source_sampler, in.uv + vec2<f32>(texel.x, -texel.y), 0.0)
        + textureSampleLevel(source_texture, source_sampler, in.uv + vec2<f32>(-texel.x, texel.y), 0.0)
        + textureSampleLevel(source_texture, source_sampler, in.uv + vec2<f32>(texel.x, texel.y), 0.0);
    return sum * 0.25;
}

// One axis of the separable Gaussian.
//
// `axis` is (1, 0) or (0, 1) in texels. The 2 * RADIUS + 1 discrete taps are folded to
// PAIRS + 1 samples: for adjacent taps at offsets `a` and `a + 1` with weights `wa` and `wb`,
// one linear sample taken at `(a * wa + (a + 1) * wb) / (wa + wb)` returns exactly
// `(wa * t[a] + wb * t[a+1]) / (wa + wb)`, so scaling it by `wa + wb` reproduces both taps.
// That identity is why this is a fold and not an approximation -- it depends only on the
// sampler being linear and on the offset lying between the two texel centres, both of which
// hold by construction.
//
// The running total is normalized by the accumulated weight rather than by a precomputed
// constant. Truncating a Gaussian and NOT renormalizing scales every output down by the
// weight in the tail -- 0.85% for this discrete kernel, measured in `qs_gpu::target` -- which
// reads as the panel being faintly darker than the tint asked for, and gets diagnosed as a
// palette bug rather than as a kernel bug.
fn blur_axis(uv: vec2<f32>, axis: vec2<f32>) -> vec4<f32> {
    let texel = source_texel() * axis;

    var total = textureSampleLevel(source_texture, source_sampler, uv, 0.0) * gaussian(0.0);
    var weight = gaussian(0.0);

    for (var i: i32 = 0; i < PAIRS; i = i + 1) {
        let a = f32(1 + i * 2);
        let b = a + 1.0;
        let wa = gaussian(a);
        let wb = gaussian(b);
        let pair = wa + wb;
        let at = (a * wa + b * wb) / pair;

        total = total + textureSampleLevel(source_texture, source_sampler, uv + texel * at, 0.0) * pair;
        total = total + textureSampleLevel(source_texture, source_sampler, uv - texel * at, 0.0) * pair;
        weight = weight + pair * 2.0;
    }

    return total / weight;
}

@fragment
fn fs_blur_h(in: VsOut) -> @location(0) vec4<f32> {
    return blur_axis(in.uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_blur_v(in: VsOut) -> @location(0) vec4<f32> {
    return blur_axis(in.uv, vec2<f32>(0.0, 1.0));
}
