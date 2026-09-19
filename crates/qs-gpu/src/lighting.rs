//! Lighting from scene geometry, applied after surfaces and before text.
//!
//! The pass multiplies surface colors by attenuation and adds light. Text and
//! icons are drawn afterward. Bounds on attenuation and addition allow callers
//! to check foreground contrast against the resulting surfaces.
//!
//! Each frame is computed from its scene and surface image. The pass has no
//! temporal accumulator or convergence step.

use bytemuck::{Pod, Zeroable};

use crate::path::RenderPath;
use crate::scene::SceneList;

/// How many slabs the shader can see. Mirrors `LIT_SLABS` in `shaders/lighting.wgsl`, and
/// `the_shader_and_the_uniform_agree_about_the_slab_bound` reads the source to hold them
/// equal — the same discipline the `KIND_` constants live under.
///
/// A **uniform** array rather than a storage buffer, because the Reduced tier is the lowest
/// tier that draws shadows and it is GLES 3.1-class, where fragment-stage storage buffers
/// are not guaranteed (`GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS` may be zero). 192 slabs at
/// two `vec4`s each is 6,144 bytes against the 16 KB uniform minimum. The scene's own
/// ceiling is 4,096; the pack takes the nearest 192 and reports the rest, and a real
/// frame's walk admits well under a hundred.
pub const LIT_SLABS: usize = 192;

/// The scene as the lighting shader reads it. Layout mirrors `LitScene` in
/// `shaders/lighting.wgsl` field for field; a divergence is a silently wrong render, not a
/// validation error, because the byte counts still line up.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LitSceneUniform {
    viewport: [f32; 2],
    count: u32,
    _pad: u32,
    /// xyz: toward the light, normalized. w: hardness `k` — see [`hardness`].
    light: [f32; 4],
    /// The focus lamp's **strip**: the focused row's rect, `[x, y, w, h]`. A separate field
    /// from `light` because a positional light has no direction and a directional one has no
    /// position — the distinction [`crate::scene::FocusLamp`] is a separate type for, carried
    /// into the uniform so a half-configured light cannot be assembled here either.
    focus: [f32; 4],
    /// The lamp's four scalars: share, ambient depth, height above the canvas, hardness.
    /// **Share zero means no lamp**, and zero is what a zeroed uniform already holds, so an
    /// absent lamp needs no branch on either side of the handoff.
    focus_mix: [f32; 4],
    rect: [[f32; 4]; LIT_SLABS],
    /// radius, elevation, thickness, attenuation floor.
    shape: [[f32; 4]; LIT_SLABS],
    /// Emission this slab gives the scene: linear rgb, and strength in `w`. Zero strength
    /// is not a light. `addition_max` — how much light this slab may RECEIVE, from its
    /// material's allowance — rides in the fourth component of [`LitSceneUniform::props`],
    /// because a surface's two lighting roles are independent: the selected row emits hard
    /// and receives nothing.
    emission: [[f32; 4]; LIT_SLABS],
    /// `addition_max`, then three unused. A whole `vec4` for one float because a uniform
    /// array's stride is 16 bytes whatever is in it; packing it into a spare component of
    /// one of the arrays above would save nothing and would make three meanings share a
    /// field. 192 slabs x 4 vec4 = 12,288 bytes, inside the 16 KB uniform minimum.
    props: [[f32; 4]; LIT_SLABS],
}

/// Contact-shadow constant for an angular light size: `1 / tan(size / 2)`.
///
/// A larger light gives a smaller constant and softer shadows. The denominator
/// is clamped away from zero.
#[must_use]
pub fn hardness(size_deg: f32) -> f32 {
    1.0 / (size_deg.to_radians() * 0.5).tan().max(1e-4)
}

impl std::fmt::Debug for LitSceneUniform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LitSceneUniform")
            .field("count", &self.count)
            .field("light", &self.light)
            .finish_non_exhaustive()
    }
}

impl LitSceneUniform {
    /// Pack `scene` for the shader. Returns the uniform and how many slabs did not fit.
    ///
    /// Over [`LIT_SLABS`] the excess is dropped from the END of the list — the builder
    /// admits in paint order, canvas first, so the slabs that go are the ones painted last
    /// — and the count is returned rather than swallowed, for the same reason
    /// [`SceneList::push`] counts: a missing shadow needs a number to look at.
    #[must_use]
    pub fn pack(scene: &SceneList, viewport: [f32; 2]) -> (Self, usize) {
        let mut uniform = Self::zeroed();
        uniform.viewport = viewport;

        let light = scene.key_light.as_ref();
        let (direction, size) = light.map_or(([-0.32, -0.55, 0.77], 5.0), |l| (l.vector, l.size));
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt()
            .max(1e-6);
        uniform.light = [
            direction[0] / len,
            direction[1] / len,
            direction[2] / len,
            hardness(size),
        ];

        // The focus lamp. Absent leaves the zeroed uniform in place, whose share is zero —
        // and a zero share is the shader's identity, not a special case it has to branch on.
        if let Some(lamp) = scene.focus_light.as_ref() {
            uniform.focus = lamp.rect;
            // Clamped here as well as at the token, because the two clamps guard different
            // things: the token's stops a hand-edited file, this one stops any future caller
            // that builds a `FocusLamp` without going through `qs_ui::scene::focus_light`. A
            // share above one makes the mix an extrapolation and a negative ambient
            // *brightens* — both push attenuation past 1.0, which is the one direction the
            // allowance clamp does not catch.
            uniform.focus_mix = [
                lamp.share.clamp(0.0, 1.0),
                lamp.ambient.clamp(0.0, 1.0),
                lamp.height,
                hardness(lamp.size),
            ];
        }

        let take = scene.slabs.len().min(LIT_SLABS);
        for (slab, (rect, shape)) in scene
            .slabs
            .iter()
            .zip(uniform.rect.iter_mut().zip(uniform.shape.iter_mut()))
        {
            *rect = slab.rect;
            *shape = [
                slab.radius,
                slab.elevation,
                slab.thickness,
                slab.attenuation_floor.clamp(0.0, 1.0),
            ];
        }
        for (slab, (emission, props)) in scene
            .slabs
            .iter()
            .zip(uniform.emission.iter_mut().zip(uniform.props.iter_mut()))
        {
            *emission = [
                slab.emission[0],
                slab.emission[1],
                slab.emission[2],
                slab.emission_strength.max(0.0),
            ];
            *props = [slab.addition_max.clamp(0.0, 4.0), 0.0, 0.0, 0.0];
        }
        uniform.count = take as u32;
        (uniform, scene.slabs.len() - take)
    }
}

/// One lighting behaviour, and what it becomes on every tier that cannot draw it.
///
/// The scene-level counterpart of [`crate::frame::Fidelity`], which already does this for
/// primitives. It is a separate mechanism because an effect is not a primitive — it is a property
/// of a pass that reads the whole scene — and reusing `PrimKind`'s machinery would mean inventing a
/// primitive nothing draws purely to hang a declaration on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SceneEffect {
    /// Contact-hardening shadow from the key light and, when present, the focus lamp.
    Shadow,
    /// Darkening where surfaces meet.
    Occlusion,
    /// Light travelling from an emitting surface onto its neighbours, occluded on the way.
    Bounce,
    /// A surface displacing what is behind it at its edges.
    Refraction,
}

/// What a tier draws in place of an effect it cannot draw.
///
/// Named after [`crate::frame::Floor`] and meaning the same thing, at a different scale.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SceneFloor {
    /// Omit the effect on this rendering tier.
    Nothing,
    /// The same effect with stated, bounded parameters.
    ///
    /// The bound is part of the declaration, not a tuning value: a step count that varies by
    /// machine is an unstated divergence.
    Bounded { steps: u32 },
}

impl SceneEffect {
    /// Every effect this feature can draw.
    ///
    /// Declared rather than derived, for the reason [`crate::frame::PrimKind::ALL`] is: Rust has no
    /// stable way to enumerate an enum's variants, and the exhaustive matches below make adding one
    /// a compile error rather than an omission.
    pub const ALL: [SceneEffect; 4] = [
        Self::Shadow,
        Self::Occlusion,
        Self::Bounce,
        Self::Refraction,
    ];

    /// The lowest tier that draws this effect in full.
    ///
    /// Exhaustive on purpose: a new effect cannot be added without answering this.
    #[must_use]
    pub const fn requires(self) -> RenderPath {
        match self {
            // One shadow ray and a bounded sample count are within reach of the GL tier.
            Self::Shadow | Self::Occlusion => RenderPath::Reduced,
            // Bounce needs a ray per emitter; refraction needs to sample the backdrop.
            Self::Bounce | Self::Refraction => RenderPath::Primary,
        }
    }

    /// What `tier` draws instead, or `None` when `tier` draws the effect in full.
    ///
    /// The second exhaustive match, and the one that makes a missing rung a compile error rather
    /// than a default. Every tier below [`SceneEffect::requires`] is answered here.
    #[must_use]
    pub const fn floor(self, tier: RenderPath) -> Option<SceneFloor> {
        match (self, tier) {
            // Drawn in full.
            (Self::Shadow | Self::Occlusion, RenderPath::Reduced | RenderPath::Primary) => None,
            (Self::Bounce | Self::Refraction, RenderPath::Primary) => None,

            // The CPU tier has no shader at all, so nothing here reaches it.
            (_, RenderPath::Cpu) => Some(SceneFloor::Nothing),

            // Bounce and refraction on the Reduced tier: absent rather than approximated.
            (Self::Bounce | Self::Refraction, RenderPath::Reduced) => Some(SceneFloor::Nothing),
        }
    }

    /// Whether `tier` draws this effect in full.
    ///
    /// Derived from [`SceneEffect::floor`] so the two can never disagree. Deliberately *not*
    /// written as a comparison against [`SceneEffect::requires`]: `RenderPath` derives `Ord` from
    /// its declaration order, which runs `Primary < Reduced < Cpu` — the opposite of capability.
    /// `tier <= requires()` therefore reads as "capable enough" and means the reverse, which is
    /// exactly the kind of inverted guard that produces an effect drawn on the tier that cannot
    /// afford it.
    #[must_use]
    pub const fn draws(self, tier: RenderPath) -> bool {
        self.floor(tier).is_none()
    }

    /// Benchmark scenario name for this effect.
    #[must_use]
    pub const fn scenario(self) -> &'static str {
        match self {
            Self::Shadow => "lit/shadow",
            Self::Occlusion => "lit/occlusion",
            Self::Bounce => "lit/bounce",
            Self::Refraction => "lit/refraction",
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn a_frame_is_a_complete_picture() {
        // T033 / FR-023 / SC-011: the pass is a pure function of the scene — no
        // accumulator, nothing to converge, nothing to terminate. Drawing the same view
        // twice produces bit-identical attenuation for every pixel, which is a stronger
        // guarantee than "the refinement stops" and is what keeps zero-work-at-rest true
        // with the mode on.
        use crate::scene::Slab;
        use crate::tier_parity::shader;

        let slabs = vec![
            Slab {
                rect: [0.0, 0.0, 256.0, 256.0],
                thickness: 1.0,
                attenuation_floor: 0.0,
                ..Slab::default()
            },
            Slab {
                rect: [40.0, 40.0, 80.0, 40.0],
                radius: 6.0,
                elevation: 12.0,
                thickness: 12.0,
                attenuation_floor: 0.0,
                ..Slab::default()
            },
        ];
        let toward = [-0.32, -0.55, 0.77];
        let k = hardness(5.0);

        let picture = || -> Vec<u32> {
            (0..64u32 * 64)
                .map(|i| {
                    let p = [(i % 64) as f32 * 4.0, (i / 64) as f32 * 4.0];
                    shader::lit_attenuation(p, &slabs, toward, k, None).to_bits()
                })
                .collect()
        };
        assert_eq!(
            picture(),
            picture(),
            "drawing the same view twice diverged: something accumulates"
        );
    }

    #[test]
    fn the_uniform_packs_the_scene_the_shader_expects() {
        // The pack takes the nearest-first slabs, normalizes the light, carries the floor,
        // and reports what did not fit — the counted-drop discipline at the second ceiling.
        use crate::scene::{Light, SceneList, Slab};

        let mut scene = SceneList::default();
        scene.reset(3, crate::scene::Environment::default());
        for i in 0..(LIT_SLABS + 5) {
            scene.push(Slab {
                rect: [i as f32, 0.0, 10.0, 10.0],
                attenuation_floor: 0.25,
                ..Slab::default()
            });
        }
        scene.key_light = Some(Light {
            vector: [0.0, 0.0, 2.0],
            colour: [1.0, 1.0, 1.0],
            intensity: 0.7,
            size: 10.0,
        });

        let (uniform, dropped) = LitSceneUniform::pack(&scene, [800.0, 600.0]);
        assert_eq!(dropped, 5, "the overflow was not counted");
        assert_eq!(uniform.count as usize, LIT_SLABS);
        assert!(
            (uniform.light[2] - 1.0).abs() < 1e-6,
            "the light was not normalized"
        );
        assert!((uniform.light[3] - hardness(10.0)).abs() < 1e-3);
        assert!(
            (uniform.shape[0][3] - 0.25).abs() < f32::EPSILON,
            "the floor was dropped"
        );
    }

    /// Capability rank, ascending. Stated here because `RenderPath`'s derived `Ord` is
    /// declaration order and runs the other way; a test that used `<` directly would pass while
    /// asserting the opposite of what it reads as.
    fn rank(tier: RenderPath) -> u8 {
        match tier {
            RenderPath::Cpu => 0,
            RenderPath::Reduced => 1,
            RenderPath::Primary => 2,
        }
    }

    #[test]
    fn every_effect_declares_a_floor_for_every_tier_below_the_one_it_needs() {
        // The rule the whole degradation discipline rests on: not "lower tiers do not draw it",
        // but each rung, named. A `None` where a floor was required would mean a tier drawing
        // whatever it happened to do, which is the unstated divergence `Fidelity` exists to stop.
        for effect in SceneEffect::ALL {
            for tier in [RenderPath::Cpu, RenderPath::Reduced, RenderPath::Primary] {
                let below = rank(tier) < rank(effect.requires());
                assert_eq!(
                    effect.floor(tier).is_some(),
                    below,
                    "{effect:?} on {tier:?}: a floor is declared exactly when the tier is below                      the one the effect requires, and this pair disagrees"
                );
                assert_eq!(
                    effect.draws(tier),
                    !below,
                    "{effect:?} on {tier:?}: `draws` and `floor` disagree"
                );
            }
        }
    }

    #[test]
    fn nothing_reaches_the_cpu_tier() {
        // There is no shader on that tier, so every effect floors to absence there. Asserted
        // rather than assumed, because an effect added later could plausibly declare `Bounded`
        // here and be wrong in a way that only shows on a machine nobody develops on.
        for effect in SceneEffect::ALL {
            assert_eq!(
                effect.floor(RenderPath::Cpu),
                Some(SceneFloor::Nothing),
                "{effect:?} claims the CPU tier can draw something"
            );
        }
    }

    #[test]
    fn every_effect_names_a_scenario_and_no_two_share_one() {
        // A shared scenario name would silently measure one effect twice and the other never.
        let mut seen = Vec::new();
        for effect in SceneEffect::ALL {
            let name = effect.scenario();
            assert!(!name.is_empty(), "{effect:?} names no scenario");
            assert!(
                !seen.contains(&name),
                "{effect:?} reuses the scenario name {name}"
            );
            seen.push(name);
        }
        assert_eq!(seen.len(), SceneEffect::ALL.len());
    }
}
