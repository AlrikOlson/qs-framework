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
// How far bounced light reaches before it has fallen to a quarter, in physical pixels.
// Stated rather than tuned per material: one room, one falloff, so two emitters at the
// same distance contribute the same and a surface cannot buy itself extra reach.
const BOUNCE_REACH: f32 = 90.0;

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
    // rgb: what this slab emits into the scene, linear. w: strength, zero = not a light.
    emission: array<vec4f, LIT_SLABS>,
    // x: addition_max — how much light this slab may RECEIVE. Zero on a text ground.
    props: array<vec4f, LIT_SLABS>,
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

// Visibility along a segment of KNOWN, FINITE length — the receiver to a point on an emitter.
//
// `soft_shadow` above cannot answer this, and the reason is worth stating because the two look
// interchangeable. Its step is the scene distance, which is what makes a directional ray with no
// end cheap: empty space is crossed in one stride. A bounce ray skims half a pixel above its own
// receiver for its whole length, so the scene distance is ~0.5 the entire way, every step is
// clamped to the 0.5 minimum, and SHADOW_STEPS of budget is spent inside ~17 px. BOUNCE_REACH is
// 90. Used for bounce, it reports "unoccluded" for four fifths of the distance the light travels,
// which is not a soft shadow -- it is no shadow with a plausible shape.
//
// A finite segment is a different problem and takes the obvious answer: fixed steps across it, so
// coverage is uniform and the cost is exactly SHADOW_STEPS regardless of geometry. `t` stays
// strictly between the two ends -- the receiver's own face would occlude at 0 and the emitter's
// own face at `d`, and both would return black everywhere.
fn bounce_shadow(origin: vec3f, toward: vec3f, d: f32, k: f32) -> f32 {
    var res = 1.0;
    for (var i = 1u; i <= SHADOW_STEPS; i++) {
        let t = d * f32(i) / f32(SHADOW_STEPS + 1u);
        let h = scene_distance(origin + toward * t);
        res = min(res, clamp(k * h / t, 0.0, 1.0));
        if res < 0.005 {
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

// One-bounce light from every emitting slab (T052).
//
// The nearest point on the emitter is what the receiver sees, so a full-width row lights
// like a strip rather than like a bulb hanging over its centre — the difference is the
// whole reason a selected row can wash the rows beside it evenly. Falloff is inverse-square
// in a bounded form (`1 / (1 + (d/reach)^2)`), which is finite at zero distance instead of
// dividing by it, and the receiver's own upward normal gives the cosine term.
//
// Occluded, per contract: a shadow ray toward the emitter, so a surface hidden behind
// something taller receives nothing. That is what stops the glow leaking through the
// interface's own geometry.
fn bounce(origin: vec3f, receiver: u32) -> vec3f {
    var added = vec3f(0.0);
    for (var i = 0u; i < min(scene.count, LIT_SLABS); i++) {
        let e = scene.emission[i];
        if e.w <= 0.0 || i == receiver {
            continue;
        }
        let r = scene.rect[i];
        // Nearest point on the emitter's top face.
        let q = clamp(origin.xy, r.xy, r.xy + r.zw);
        let to_light = vec3f(q, scene.shape[i].y) - origin;
        let d = length(to_light);
        if d > BOUNCE_REACH * 2.0 {
            continue;
        }
        let n_dot_l = clamp(to_light.z / max(d, 0.001), 0.0, 1.0);
        let falloff = 1.0 / (1.0 + (d / BOUNCE_REACH) * (d / BOUNCE_REACH));
        let shade = bounce_shadow(origin, to_light / max(d, 0.001), d, scene.light.w);
        added += e.rgb * e.w * falloff * n_dot_l * shade;
    }
    return added;
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
    var take = 0.0;
    var receiver = 0u;
    var found = false;
    for (var i = 0u; i < min(scene.count, LIT_SLABS); i++) {
        let r = scene.rect[i];
        let s = scene.shape[i];
        if p.x >= r.x && p.x <= r.x + r.z && p.y >= r.y && p.y <= r.y + r.w && s.y >= top {
            top = s.y;
            floor_ = s.w;
            take = scene.props[i].x;
            receiver = i;
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

    // Bounced light, clamped to what THIS surface is allowed to receive. A text ground's
    // `addition_max` is zero, so a filename's background is untouched however hard the row
    // beside it glows — rule 1a as arithmetic rather than as a convention. The clamp is per
    // channel so a coloured bounce cannot exceed the allowance by arriving as three
    // components that individually fit and jointly do not.
    var added = vec3f(0.0);
    if take > 0.0 {
        added = clamp(bounce(origin, receiver), vec3f(0.0), vec3f(take));
    }

    // An untouched pixel is not written at all. `dst * 1.0 + 0` is an identity only in
    // exact arithmetic — a real blend decodes and re-encodes the sRGB attachment, and the
    // round-trip is not promised bit-exact — so the identity fragment discards instead,
    // which is also what makes an unshadowed frame cost fill rate and nothing else.
    // `a_scene_with_nothing_elevated_changes_no_pixel` is this line as a test. When US2
    // adds bounce, the condition grows `&& addition == vec3f(0.0)`.
    if atten >= 1.0 && added.r <= 0.0 && added.g <= 0.0 && added.b <= 0.0 {
        discard;
    }

    // The blend state applies `dst * atten + added`.
    return vec4f(added, atten);
}
