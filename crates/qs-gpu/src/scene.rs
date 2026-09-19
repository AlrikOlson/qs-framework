//! Scene geometry for the lighting pass.
//!
//! Surfaces are rounded slabs with elevations, viewed by a fixed orthographic
//! camera. Their rectangles and radii must match the corresponding draw-list
//! instances.
//!
//! A frame carries no scene when lighting is disabled. When lighting is enabled,
//! the scene and draw list share a generation; mismatched pairs are rejected.

use crate::color::Srgba;

/// A surface, extruded. Physical pixels throughout, like everything in `Instance::rect`.
///
/// `elevation` is the height of the top face above the canvas; `thickness` is how far the slab
/// extends below it. A zero-thickness slab is a plane: it still casts, but it has no side for
/// light to catch, which is usually not what an author meant.
#[derive(Clone, Copy, PartialEq, Debug)]
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
    /// The least attenuation the lighting pass may apply to this surface — the material's
    /// allowance (`Tokens::lit_bounds`), per theme, carried onto the slab so the shader can
    /// enforce it per pixel. **This is how the contrast gate's closed-form worst case is a
    /// bound on real frames rather than a hope** (lit-contrast rules 1a and 3a).
    pub attenuation_floor: f32,
    /// The most light, in linear units, the pass may ADD to this surface — the other half
    /// of the same allowance. Zero for a text ground, which is rule 1a's whole content: a
    /// surface carrying a label may emit as hard as the design likes and may receive
    /// nothing, so bounced light lands on the canvas and the gaps rather than under
    /// anybody's filename.
    pub addition_max: f32,
}

impl Default for Slab {
    /// The floor defaults to **1.0** — no attenuation permitted at all — which is the safe
    /// direction: a slab built without going through `Tokens` cannot be darkened, where a
    /// zero default would hand full black to exactly the construction path that skipped the
    /// allowance. Everything else genuinely is zero.
    fn default() -> Self {
        Self {
            rect: [0.0; 4],
            radius: 0.0,
            elevation: 0.0,
            thickness: 0.0,
            albedo: [0.0; 3],
            roughness: 0.0,
            metalness: 0.0,
            emission: [0.0; 3],
            emission_strength: 0.0,
            attenuation_floor: 1.0,
            addition_max: 0.0,
        }
    }
}

impl Slab {
    /// Whether this slab lights anything.
    #[must_use]
    pub fn emits(&self) -> bool {
        self.emission_strength > 0.0
    }
}

/// A directional key light.
///
/// Its direction is independent of position. Positional lighting uses
/// [`FocusLamp`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Light {
    /// Direction *toward* the light, normalized at use.
    pub vector: [f32; 3],
    /// Straight linear RGB, from a token.
    pub colour: [f32; 3],
    pub intensity: f32,
    /// **The field that matters.** The light's angular size: it governs how fast a shadow
    /// softens with distance, which is the whole of the feature's central claim. A light with
    /// no size casts a hard offset shadow and the claim collapses.
    pub size: f32,
}

/// A positional light over the item with keyboard focus.
///
/// It changes attenuation without adding color to the focused row's background.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FocusLamp {
    /// Focused row bounds as `[x, y, width, height]` in physical pixels.
    pub rect: [f32; 4],
    /// How high the strip hangs above the canvas, physical pixels.
    pub height: f32,
    /// Angular size in degrees, as [`Light::size`].
    pub size: f32,
    /// Weight of the lamp's shadow ray relative to the key light beneath it.
    /// Zero leaves the key light's result unchanged.
    pub share: f32,
    /// How far the room dims at the edge of the lamp's reach.
    ///
    /// **The half a person actually sees.** `share` only changes a pixel where the two lights
    /// disagree, and on a list of rows at one elevation they agree almost everywhere —
    /// measured at a peak of 7/255 across a shipped window. This is what makes focus *lit*.
    pub ambient: f32,
}

/// Horizon and zenith colors of the reflected environment.
///
/// The caller supplies both colors from its theme.
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
    pub focus_light: Option<FocusLamp>,
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
