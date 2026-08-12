// A LOOKING SPIKE, not shipped code.
//
// It renders the proposal in section 01: the interface as rounded slabs at token-assigned
// elevations, under an ORTHOGRAPHIC camera, sphere-traced. Orthographic is the whole point --
// primary visibility is identical to the flat renderer, so the screen rectangle of every
// element is exactly where it is today. What the third dimension buys is the SECONDARY rays:
// shadows, occlusion, and light bouncing off one element onto another.
//
// Two honest differences from what production would do:
//
//   * Normals here are central differences of the scene distance function. The shipped
//     PrimKind::Pbr uses the analytic gradient of `sd_rounded_box`, which is exact and needs
//     no extra map() evaluations. The spike takes the lazy route because it is measuring
//     whether the LOOK is worth building, not how fast it can be.
//   * `map()` loops every slab. Production would trace hardware AABBs (wgpu ray queries with
//     procedural geometry) or at minimum bin the slabs by screen tile.

struct Slab {
    // xyz centre, w corner radius. All in physical pixels; z is elevation off the canvas.
    centre_radius: vec4<f32>,
    // xyz half-extent, w roughness.
    half_rough: vec4<f32>,
    // rgb albedo in linear light, w metallic.
    albedo_metal: vec4<f32>,
    // rgb emission in linear light, w strength. Strength zero means the slab is not a light.
    emissive: vec4<f32>,
};

struct Scene {
    viewport: vec2<f32>,
    count: u32,
    flags: u32,
    // xyz key light direction (toward the light), w penumbra hardness.
    light: vec4<f32>,
    env_horizon: vec4<f32>,
    env_zenith: vec4<f32>,
    // xyz world position of the focus lamp, w intensity.
    focus: vec4<f32>,
};

const FLAG_SHADOW:  u32 = 1u;
const FLAG_AO:      u32 = 2u;
const FLAG_BOUNCE:  u32 = 4u;
const FLAG_FOCUS:   u32 = 8u;

const PI: f32 = 3.14159265;

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> slabs: array<Slab>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle. Cheaper than a quad and has no diagonal seam.
    let x = f32(i32(i) / 2) * 4.0 - 1.0;
    let y = f32(i32(i) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, -y, 0.0, 1.0);
}

// Signed distance to one rounded slab.
fn sd_slab(p: vec3<f32>, s: Slab) -> f32 {
    let r = s.centre_radius.w;
    let q = abs(p - s.centre_radius.xyz) - s.half_rough.xyz + vec3<f32>(r, r, r);
    return length(max(q, vec3<f32>(0.0, 0.0, 0.0)))
        + min(max(q.x, max(q.y, q.z)), 0.0)
        - r;
}

fn map(p: vec3<f32>) -> f32 {
    var d = 1e9;
    for (var i = 0u; i < scene.count; i = i + 1u) {
        d = min(d, sd_slab(p, slabs[i]));
    }
    return d;
}

fn map_id(p: vec3<f32>) -> vec2<f32> {
    var d = 1e9;
    var id = -1.0;
    for (var i = 0u; i < scene.count; i = i + 1u) {
        let s = sd_slab(p, slabs[i]);
        if (s < d) {
            d = s;
            id = f32(i);
        }
    }
    return vec2<f32>(d, id);
}

fn normal_at(p: vec3<f32>) -> vec3<f32> {
    let e = 0.25;
    return normalize(vec3<f32>(
        map(p + vec3<f32>(e, 0.0, 0.0)) - map(p - vec3<f32>(e, 0.0, 0.0)),
        map(p + vec3<f32>(0.0, e, 0.0)) - map(p - vec3<f32>(0.0, e, 0.0)),
        map(p + vec3<f32>(0.0, 0.0, e)) - map(p - vec3<f32>(0.0, 0.0, e)),
    ));
}

// Inigo Quilez's soft shadow. `res = min(res, k*h/t)` is the whole idea: the closer the ray
// passed to an occluder, and the earlier along the ray it happened, the darker the penumbra.
//
// This is the function that gives the interface contact hardening for free -- a shadow that
// is tight where two surfaces nearly touch and broad where they are far apart. `k` is the
// inverse of the light's angular size.
fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, maxt: f32, k: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    for (var i = 0; i < 48; i = i + 1) {
        if (t >= maxt) { break; }
        let h = map(ro + rd * t);
        if (h < 0.02) { return 0.0; }
        res = min(res, k * h / t);
        t = t + clamp(h, 0.4, 24.0);
    }
    return clamp(res, 0.0, 1.0);
}

// Distance-field ambient occlusion: walk a short way along the normal and compare how far the
// field says the nearest surface is against how far we actually moved. Where a row meets the
// canvas the two disagree, and that disagreement is the seam.
fn occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        let h = 0.4 + 3.2 * f32(i) / 5.0;
        let d = map(p + n * h);
        occ = occ + (h - d) * sca;
        sca = sca * 0.72;
    }
    return clamp(1.0 - 0.85 * occ, 0.0, 1.0);
}

fn environment(ray: vec3<f32>) -> vec3<f32> {
    let t = clamp(ray.z * 0.5 + 0.5, 0.0, 1.0);
    return mix(scene.env_horizon.rgb, scene.env_zenith.rgb, t);
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn visibility_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let lv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let ll = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(lv + ll, 1e-5);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let f = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    return f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * f;
}

// One light's contribution through the same microfacet model the shipped renderer uses.
fn lobe(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, albedo: vec3<f32>, rough: f32, metal: f32) -> vec3<f32> {
    let h = normalize(l + v);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metal);
    let f = fresnel_schlick(v_dot_h, f0);
    let spec = distribution_ggx(n_dot_h, rough) * visibility_smith(n_dot_v, n_dot_l, rough) * f;
    let kd = (vec3<f32>(1.0, 1.0, 1.0) - f) * (1.0 - metal);
    return (kd * albedo / PI + spec) * n_dot_l;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // THE ORTHOGRAPHIC CAMERA. The ray for pixel (x, y) starts directly above that pixel and
    // travels straight down. No perspective divide, no field of view: the screen position of
    // every slab is exactly its x/y, whatever its elevation. That is what keeps the interface
    // two-dimensional to a user while the shading gets a third dimension to work with.
    let ro = vec3<f32>(frag.x, frag.y, 400.0);
    let rd = vec3<f32>(0.0, 0.0, -1.0);
    let v = -rd;

    var t = 0.0;
    var hit = -1.0;
    for (var i = 0; i < 96; i = i + 1) {
        let p = ro + rd * t;
        let m = map_id(p);
        if (m.x < 0.02) {
            hit = m.y;
            break;
        }
        t = t + max(m.x, 0.05);
        if (t > 520.0) { break; }
    }

    if (hit < 0.0) {
        return vec4<f32>(environment(vec3<f32>(0.0, 0.0, 1.0)), 1.0);
    }

    let p = ro + rd * t;
    let n = normal_at(p);
    let s = slabs[u32(hit)];
    let albedo = s.albedo_metal.rgb;
    let metal = s.albedo_metal.w;
    let rough = clamp(s.half_rough.w, 0.045, 1.0);

    let l = normalize(scene.light.xyz);
    var shadow = 1.0;
    if ((scene.flags & FLAG_SHADOW) != 0u) {
        // Offset along the normal so the surface does not shadow itself at the first step.
        shadow = soft_shadow(p + n * 0.35, l, 0.6, 320.0, scene.light.w);
    }

    var ao = 1.0;
    if ((scene.flags & FLAG_AO) != 0u) {
        ao = occlusion(p, n);
    }

    // The key light, calibrated the way the shipped shader is: a flat dielectric under it
    // returns its own albedo rather than a quarter of it.
    var lit = lobe(n, v, l, albedo, rough, metal) * 4.0757 * shadow;

    // Ambient from the sky, occluded in the seams.
    let refl = reflect(-v, n);
    let env_d = environment(n) * albedo * (1.0 - metal);
    let env_s = environment(refl) * mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metal);
    lit = lit * 0.62 + (env_d + env_s) * 0.38 * ao;

    // BOUNCE. Every emissive slab is a light, and its light is occluded by the geometry
    // between it and here -- which is what makes a lifted row spill accent onto its
    // neighbours and onto the canvas beneath, and what makes state visible peripherally.
    if ((scene.flags & FLAG_BOUNCE) != 0u) {
        for (var i = 0u; i < scene.count; i = i + 1u) {
            let e = slabs[i];
            if (e.emissive.w <= 0.0) { continue; }
            // Nearest point on the emitter, so a long row lights like the strip it is rather
            // than like a bulb at its centre.
            let lo = e.centre_radius.xyz - e.half_rough.xyz;
            let hi = e.centre_radius.xyz + e.half_rough.xyz;
            let closest = clamp(p, lo, hi);
            let to = closest - p;
            let dist = max(length(to), 0.001);
            if (dist < 0.5) { continue; }
            let dir = to / dist;
            let atten = e.emissive.w / (1.0 + dist * dist * 0.0009);
            let vis = soft_shadow(p + n * 0.35, dir, 0.6, dist, 8.0);
            lit = lit + e.emissive.rgb * atten * max(dot(n, dir), 0.0) * vis * albedo * 5.0;
        }
    }

    // The focus lamp: a light that sits where keyboard focus is, so moving focus swings every
    // shadow in the window instead of moving a rectangle around.
    if ((scene.flags & FLAG_FOCUS) != 0u) {
        let to = scene.focus.xyz - p;
        let dist = max(length(to), 0.001);
        let dir = to / dist;
        let vis = soft_shadow(p + n * 0.35, dir, 0.6, dist, 10.0);
        let atten = scene.focus.w / (1.0 + dist * dist * 0.0016);
        lit = lit + lobe(n, v, dir, albedo, rough, metal) * atten * vis;
    }

    // The slab's own emission, on top of everything, unshadowed.
    lit = lit + s.emissive.rgb * s.emissive.w;

    return vec4<f32>(lit, 1.0);
}
