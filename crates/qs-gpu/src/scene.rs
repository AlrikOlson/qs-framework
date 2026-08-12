//! The interface as the lighting pass sees it: rounded slabs at elevations, under a fixed
//! square-on camera.
//!
//! # Why this is not on `Instance`
//!
//! The raster path never reads an elevation. Only the lighting pass does. Putting one on the
//! instance would spend the last field with room — the spare bits of [`crate::frame::Instance::kind`]
//! — on a consumer that does not exist, and would force every `kind ==` comparison to mask in
//! three places that must agree with each other: `shaders/instance.wgsl`,
//! `cpu_raster::draw_instance`, and the `tier_parity` transcription. It would also break the
//! set-equality test that parses `KIND_` constants out of the shader.
//!
//! Keeping the scene separate also makes the mode's cost structural rather than careful. With the
//! mode off no scene is published, so there is nothing to skip and no branch to get wrong. See
//! `specs/002-ray-traced-mode/research.md` R1.
//!
//! # The contract
//!
//! `specs/002-ray-traced-mode/contracts/scene-handoff.md` states six rules. Three of them are the
//! reason this module exists at all:
//!
//! - **The scene agrees with the draw list.** A slab's rect and radius equal the instance's
//!   *exactly*. A slab that disagrees casts a shadow from a shape nobody can see, and the symptom
//!   appears in the lit path while the cause is a disagreement between two descriptions of one
//!   thing. Both are built from the same `Material`/`Surface` pair, so agreement is achievable by
//!   construction and any difference is a mistake rather than a rounding.
//! - **Absent is not empty.** With the mode off a frame carries *no* scene. An empty scene means
//!   the interface has nothing in it, which is a bug worth finding; collapsing the two makes the
//!   first undiagnosable.
//! - **One generation, one interface.** A scene carries the generation of the draw list it belongs
//!   to and a mismatched pair is refused, or a dropped frame on one side would light this frame's
//!   geometry with the last frame's shadows — which reads as latency and is nearly impossible to
//!   attribute.

use crate::color::Srgba;

/// A surface, extruded. Physical pixels throughout, like everything in `Instance::rect`.
///
/// `elevation` is the height of the top face above the canvas; `thickness` is how far the slab
/// extends below it. A zero-thickness slab is a plane: it still casts, but it has no side for
/// light to catch, which is usually not what an author meant.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Slab {
    /// `[x, y, w, h]`, physical pixels. **Equals** the corresponding instance's `rect`.
    pub rect: [f32; 4],
    /// Corner radius, physical pixels. **Equals** the corresponding instance's `radius`.
    pub radius: f32,
    /// Top-face height above the canvas, physical pixels.
    pub elevation: f32,
    /// How far the slab extends below its top face, physical pixels.
    pub thickness: f32,
    /// Straight linear RGB. What the surface reflects.
    pub albedo: [f32; 3],
    /// `0..=1`.
    pub roughness: f32,
    /// `0..=1`.
    pub metalness: f32,
    /// Straight linear RGB.
    pub emission: [f32; 3],
    /// Zero means the slab is not a light. Kept separate from `emission` so a colour can be
    /// authored once and switched off without losing it.
    pub emission_strength: f32,
}

impl Slab {
    /// Whether this slab lights anything.
    #[must_use]
    pub fn emits(&self) -> bool {
        self.emission_strength > 0.0
    }
}

/// Which shape of light this is.
///
/// Two, and the distinction is not stylistic: a directional light has no position and therefore no
/// falloff, and a positional one has no single direction. Collapsing them into one type with unused
/// fields is how a light ends up half-configured.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LightKind {
    /// The key light. Infinitely far away, so only its direction matters.
    Directional,
    /// The focus lamp. Somewhere in the scene, so distance matters.
    Positional,
}

/// One source of illumination.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Light {
    pub kind: LightKind,
    /// Direction *toward* the light for [`LightKind::Directional`]; world position in physical
    /// pixels for [`LightKind::Positional`].
    pub vector: [f32; 3],
    /// Straight linear RGB, from a token.
    pub colour: [f32; 3],
    pub intensity: f32,
    /// **The field that matters.** How large the light is — angular size for a directional light,
    /// radius for a positional one. It governs how fast a shadow softens with distance, which is
    /// the whole of the feature's central claim. A light with no size casts a hard offset shadow
    /// and the claim collapses.
    pub size: f32,
}

/// The surroundings a lit surface reflects. Two stops of an infinite sky.
///
/// Both come from the palette, so the room changes with the theme. Authoring them in the renderer
/// would be a second place colour is decided, which Principle VII forbids.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Environment {
    pub horizon: Srgba,
    pub zenith: Srgba,
}

impl Default for Environment {
    fn default() -> Self {
        // A neutral illuminant, deliberately dim. A surface reflecting it looks lit rather than
        // correct, which is the right failure for a caller that forgot to set one from tokens.
        Self {
            horizon: Srgba::new(0.18, 0.19, 0.22, 1.0),
            zenith: Srgba::new(0.42, 0.45, 0.52, 1.0),
        }
    }
}

/// How many slabs one frame may carry.
///
/// Stated rather than discovered. A ceiling nobody can hit is still stated, because the
/// alternative is finding the real one on a customer's 8K display. At the largest supported window
/// and the densest supported layout the viewport holds a low hundreds of surfaces; this is an order
/// of magnitude above that.
pub const MAX_SLABS: usize = 4096;

/// One frame's worth of interface, as geometry.
///
/// The lit counterpart of [`crate::frame::DrawList`], published through the same handoff and
/// carrying the same generation.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct SceneList {
    pub slabs: Vec<Slab>,
    /// Exactly one. The interface is lit from a single key direction so that every surface agrees
    /// about where light comes from.
    pub key_light: Option<Light>,
    /// Present only when something has keyboard focus.
    pub focus_light: Option<Light>,
    pub environment: Environment,
    /// Matches the [`crate::frame::DrawList`] this scene belongs to.
    pub generation: u64,
    /// Slabs dropped by [`MAX_SLABS`]. Counted rather than silent: a support conversation about
    /// shadows going missing needs a number to look at.
    pub dropped: u32,
}

impl SceneList {
    /// Drop the contents but keep the allocations, exactly as `DrawList::reset` does.
    pub fn reset(&mut self, generation: u64, environment: Environment) {
        self.slabs.clear();
        self.key_light = None;
        self.focus_light = None;
        self.environment = environment;
        self.generation = generation;
        self.dropped = 0;
    }

    /// Add a slab, or count it as dropped if the ceiling is reached.
    ///
    /// Dropping is preferable to failing, and *counted* dropping is preferable to silent dropping.
    pub fn push(&mut self, slab: Slab) {
        if self.slabs.len() >= MAX_SLABS {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        self.slabs.push(slab);
    }

    /// Whether this scene can be lit at all.
    ///
    /// A scene with no slabs is refused rather than rendered black — see the module docs on
    /// "absent is not empty". Reaching here with an empty scene means the builder ran and found
    /// nothing, which is a defect, not a mode being off.
    #[must_use]
    pub fn is_renderable(&self) -> bool {
        !self.slabs.is_empty() && self.key_light.is_some()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn slab() -> Slab {
        Slab {
            rect: [10.0, 20.0, 100.0, 30.0],
            radius: 6.0,
            elevation: 8.0,
            thickness: 4.0,
            albedo: [0.2, 0.2, 0.25],
            roughness: 0.5,
            ..Slab::default()
        }
    }

    #[test]
    fn the_ceiling_drops_and_counts_rather_than_growing_without_bound() {
        // The failure this prevents is a scene that grows with the window until something else
        // fails first, at which point the cause is somewhere other than the symptom.
        let mut scene = SceneList::default();
        for _ in 0..(MAX_SLABS + 7) {
            scene.push(slab());
        }
        assert_eq!(scene.slabs.len(), MAX_SLABS);
        assert_eq!(scene.dropped, 7, "dropped slabs were not counted");
    }

    #[test]
    fn an_empty_scene_is_not_renderable() {
        // "Absent is not empty" from the other side: a scene that exists but holds nothing is a
        // builder that ran and found nothing, and rendering it would paint a black window over a
        // working interface.
        let mut scene = SceneList::default();
        assert!(!scene.is_renderable());

        scene.push(slab());
        assert!(
            !scene.is_renderable(),
            "a scene with geometry and no light is not renderable either"
        );

        scene.key_light = Some(Light {
            kind: LightKind::Directional,
            vector: [-0.42, -0.62, 0.66],
            colour: [1.0, 1.0, 1.0],
            intensity: 1.0,
            size: 5.0,
        });
        assert!(scene.is_renderable());
    }

    #[test]
    fn reset_keeps_allocations_and_clears_everything_else() {
        let mut scene = SceneList::default();
        scene.push(slab());
        scene.dropped = 3;
        let capacity = scene.slabs.capacity();

        scene.reset(9, Environment::default());

        assert!(scene.slabs.is_empty());
        assert_eq!(scene.dropped, 0);
        assert_eq!(scene.generation, 9);
        assert!(scene.key_light.is_none());
        assert!(
            scene.slabs.capacity() >= capacity,
            "reset dropped the allocation it exists to keep"
        );
    }

    #[test]
    fn a_slab_emits_only_when_its_strength_is_positive() {
        // Colour and strength are separate so a state's colour can be authored once and switched
        // off without losing it. That only helps if `emits` reads the strength.
        let mut s = slab();
        s.emission = [0.3, 0.5, 1.0];
        assert!(!s.emits(), "a colour with no strength is not a light");
        s.emission_strength = 0.4;
        assert!(s.emits());
    }
}
