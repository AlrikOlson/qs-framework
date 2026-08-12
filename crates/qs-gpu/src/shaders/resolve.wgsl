// The resolve pass: put the offscreen colour target back on the surface.
//
// One fullscreen triangle, one texture read, no filtering that could move a pixel. When no
// effect has sampled the backdrop this is exactly a copy, and `batcher.rs`'s
// `the_two_pass_path_is_pixel_identical_to_the_one_pass_path` holds it to that -- which is
// what makes "nothing regresses when the target is not needed" a checked claim rather than a
// hopeful one.
//
// # Why a triangle and not a quad
//
// A fullscreen triangle has no diagonal seam. Two triangles meeting across the middle of the
// screen sample the same texels twice along the shared edge and, with any filtering at all,
// can disagree there by a unit in the last place. It costs nothing to avoid.
//
// # No derivatives here either
//
// `specs/002-ray-traced-mode/contracts/scene-effect.md` and `tier_parity` both rest on there
// being no `fwidth`, `dpdx` or `dpdy` anywhere in this project's shaders. A resolve has no
// need of one, and it stays that way.

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var backdrop_texture: texture_2d<f32>;
@group(0) @binding(1) var backdrop_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    // The standard oversized triangle: (-1,-1), (3,-1), (-1,3) in clip space, whose
    // intersection with the unit square is exactly the viewport.
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;

    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    // Clip space has +y up and texture space has +y down, so the v axis is flipped here.
    // Getting this wrong renders a vertically mirrored frame, which is unmistakable -- and
    // the reason it is worth a comment is that it is unmistakable only once something is
    // actually drawn into the target, and until the first effect lands the target holds a
    // copy of an image that is not obviously asymmetric.
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // `textureSampleLevel` at level zero rather than `textureSample`: the latter computes an
    // implicit derivative to pick a mip, and this project does not allow one in a shader. The
    // target has exactly one mip level, so there is nothing to choose.
    return textureSampleLevel(backdrop_texture, backdrop_sampler, in.uv, 0.0);
}
