// The lighting pass. Reads the scene, writes attenuation and added light; the surface image is
// modulated by the result, and text is drawn afterwards and never lit.
//
// # No derivatives, ever
//
// `fwidth`, `dpdx` and `dpdy` are forbidden in this file, and `tier_parity` asserts their absence
// by reading the source. The reason is not style: every shading rule here has to be reproducible
// by the CPU-side transcription, and a screen-space derivative has no meaning on a rasterizer with
// no neighbouring fragment. The moment one appears, the parity suite stops being evidence about
// this shader and starts being a second opinion about geometry.
//
// Everything below is therefore analytic. The slab distance function has a closed-form gradient,
// so a normal costs no extra scene evaluations; the shadow term reads the distance field the march
// is already producing; the occlusion term samples a stated, bounded number of points. See
// `specs/002-ray-traced-mode/research.md` R4 and R5.
//
// # The camera is orthographic and that is the whole trick
//
// The ray for a pixel starts directly above it and travels straight down. No perspective divide,
// no field of view. Every slab therefore occupies exactly the screen rectangle its instance
// occupies, which is what lets the interface gain a third dimension without a person's hit targets
// moving a pixel.
//
// # How the result reaches the surface
//
// This pass draws one fullscreen triangle inside the SAME render pass as the surfaces, in the
// seam `batcher.rs` cut between the surface and content halves. The fragment output is
// `vec4(addition_rgb, attenuation)` and fixed-function blending applies it:
// `out = src.rgb + dst.rgb * src.a`. Until US2 lands bounce, `addition` is zero and this is a
// pure multiply — one draw, no sampling of the surface image, no second pass, and a frame with
// no scene encodes the identical command stream it always did.
//
// # Bounds, all stated
//
// - `LIT_SLABS = 192`: the shader-visible slab count, a UNIFORM array because the Reduced tier
//   is GLES 3.1-class and fragment-stage storage buffers are not guaranteed there
//   (`GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS` may be zero). 192 slabs x 2 vec4 = 6,144 bytes,
//   inside the 16 KB uniform minimum. The scene's own ceiling is 4,096; the uploader takes the
//   first 192 and says so — a real frame's walk admits well under a hundred.
// - `SHADOW_STEPS = 12`, starting at t = 0.35 px with steps clamped to [0.5, 24] px and reach
//   capped at 400 px: a fixed worst case, not a content-dependent one. The start matters more
//   than it looks: the shipped elevations are 1-8 physical pixels, so a march that began at
//   t = 2 skipped the entire shadow geometry of a row-height step and the mode changed nothing
//   — which is exactly what happened, found by diffing a --shot-gpu pair.
// - `AO_SAMPLES = 4`, 3 px apart, weights halving: research R3's bounded occlusion.
//
// # The allowance is enforced HERE, not hoped for
//
// Every slab carries `floor` — the least attenuation its material's allowance permits, per
// theme, from `Tokens::lit_bounds` — and the final attenuation is clamped to it. That clamp is
// what makes the contrast gate's closed-form worst case a bound on real frames rather than a
// prayer: a text ground in the light theme cannot be darkened past 0.87 no matter what stands
// over it, because the shader will not do it. Contract lit-contrast.md rules 1a and 3a.

const LIT_SLABS: u32 = 192u;
const SHADOW_STEPS: u32 = 12u;
const AO_SAMPLES: u32 = 4u;
const AO_STRENGTH: f32 = 0.35;

struct LitScene {
    viewport: vec2f,
    count: u32,
    _pad: u32,
    // xyz: toward the light, normalized. w: hardness k = 1 / tan(size / 2) — the inverse of
    // the light's angular size, which is the whole of contact hardening (research R4).
    light: vec4f,
    // x, y, w, h — physical pixels, exactly the instance's rect (scene-handoff rule 1).
    rect: array<vec4f, LIT_SLABS>,
    // x: corner radius. y: elevation (top face). z: thickness (downward). w: attenuation
    // floor from the material's allowance.
    shape: array<vec4f, LIT_SLABS>,
}

@group(0) @binding(0) var<uniform> scene: LitScene;

// One fullscreen triangle from the vertex index; no vertex buffer.
@vertex
fn vs_lit(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index >> 1u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

// The same 2D rounded-box distance `instance.wgsl` and the CPU transcription use.
fn sd_rounded_box(p: vec2f, b: vec2f, r: f32) -> f32 {
    let q = abs(p) - b + vec2f(r);
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Signed distance from `q` to slab `i`: the 2D rounded rect extruded from
// `elevation - thickness` up to `elevation`.
fn slab_distance(q: vec3f, i: u32) -> f32 {
    let r = scene.rect[i];
    let s = scene.shape[i];
    let centre = vec2f(r.x + r.z * 0.5, r.y + r.w * 0.5);
    let half = vec2f(r.z * 0.5, r.w * 0.5);
    let d2 = sd_rounded_box(q.xy - centre, half, s.x);
    let half_thick = max(s.z * 0.5, 0.5);
    let dz = abs(q.z - (s.y - half_thick)) - half_thick;
    let d = vec2f(d2, dz);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

// The scene: the least distance to any slab. O(count), and count is bounded above.
fn scene_distance(q: vec3f) -> f32 {
    var least = 1e9;
    for (var i = 0u; i < min(scene.count, LIT_SLABS); i++) {
        least = min(least, slab_distance(q, i));
    }
    return least;
}

// Contact-hardening shadow: track `min(k * h / t)` along one ray toward the light
// (research R4). The closer the ray passes to an occluder, and the earlier, the darker —
// penumbra falls out of the distance field for free.
fn soft_shadow(origin: vec3f, toward: vec3f, k: f32) -> f32 {
    var res = 1.0;
    var t = 0.35;
    for (var i = 0u; i < SHADOW_STEPS; i++) {
        let h = scene_distance(origin + toward * t);
        res = min(res, clamp(k * h / t, 0.0, 1.0));
        t += clamp(h, 0.5, 24.0);
        if res < 0.005 || t > 400.0 {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

// Bounded-sample occlusion straight up from the receiver (T038): how much of the sky the
// point can see. Weights halve so the nearest sample dominates, which is what darkens seams.
fn occlusion(origin: vec3f) -> f32 {
    var occ = 0.0;
    var weight = 0.5;
    for (var i = 1u; i <= AO_SAMPLES; i++) {
        let up = f32(i) * 3.0;
        let d = scene_distance(origin + vec3f(0.0, 0.0, up));
        occ += weight * clamp((up - d) / up, 0.0, 1.0);
        weight *= 0.5;
    }
    return clamp(occ, 0.0, 1.0);
}

@fragment
fn fs_lit(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let p = frag.xy;

    // The receiver: the topmost slab under this pixel. Rect containment, deliberately
    // ignoring the corner radius — a conservative receiver at a rounded corner shades a
    // sliver of canvas as its row, which is invisible; the alternative is a second SDF
    // evaluation per slab per pixel for nothing anyone can see.
    var top = -1e9;
    var floor_ = 1.0;
    var found = false;
    for (var i = 0u; i < min(scene.count, LIT_SLABS); i++) {
        let r = scene.rect[i];
        let s = scene.shape[i];
        if p.x >= r.x && p.x <= r.x + r.z && p.y >= r.y && p.y <= r.y + r.w && s.y >= top {
            top = s.y;
            floor_ = s.w;
            found = true;
        }
    }
    if !found {
        // Nothing under this pixel is in the scene: the pass leaves it exactly alone —
        // by not writing, for the same rounding reason the identity case below discards.
        discard;
    }

    // Half a pixel above the receiver's top face, so the receiver's own surface does not
    // register as an occluder at t = 0.
    let origin = vec3f(p, top + 0.5);
    var atten = soft_shadow(origin, scene.light.xyz, scene.light.w);
    atten *= 1.0 - AO_STRENGTH * occlusion(origin);

    // The allowance clamp. This line is the contrast gate's closed-form claim being true.
    atten = clamp(atten, floor_, 1.0);

    // An untouched pixel is not written at all. `dst * 1.0 + 0` is an identity only in
    // exact arithmetic — a real blend decodes and re-encodes the sRGB attachment, and the
    // round-trip is not promised bit-exact — so the identity fragment discards instead,
    // which is also what makes an unshadowed frame cost fill rate and nothing else.
    // `a_scene_with_nothing_elevated_changes_no_pixel` is this line as a test. When US2
    // adds bounce, the condition grows `&& addition == vec3f(0.0)`.
    if atten >= 1.0 {
        discard;
    }

    // addition_rgb is zero until US2's bounce; the blend state applies
    // `dst * atten + addition`.
    return vec4f(0.0, 0.0, 0.0, atten);
}
