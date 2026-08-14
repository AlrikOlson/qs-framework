//! Building the scene the lighting pass reads, from the materials the draw list is already built
//! from.
//!
//! # One source, two descriptions
//!
//! A surface is described twice: once as instances for the raster path, once as a slab for the
//! lighting pass. Both come from the same [`crate::material::Material`] and [`crate::Surface`]
//! pair, which is what makes the two descriptions agree *by construction* rather than by care.
//!
//! `specs/002-ray-traced-mode/contracts/scene-handoff.md` states the rule: a slab's rect and radius
//! equal the corresponding instance's exactly, not within a tolerance. A slab that disagrees casts
//! a shadow from a shape nobody can see, and the symptom shows up in the lit path while the cause
//! is a disagreement between two descriptions of one thing — which is a genuinely hard bug to
//! attribute and a trivial one to prevent.
//!
//! # Bounded by the viewport, not by the folder
//!
//! Virtualization already solved the hard part: a million-row folder puts the same few dozen
//! surfaces on screen as a ten-row one. The scene is built from that same visible set, widened by
//! [`max_reach`] so a surface just off-screen still casts into it.

use crate::material::{Material, Surface};
use qs_gpu::scene::{Environment, Light, SceneList, Slab};

/// How far past the viewport the scene must reach, in **logical** pixels.
///
/// A surface outside the viewport still casts into it, so culling at the viewport edge makes a
/// shadow appear as its caster scrolls into view — the same defect [`Material::bleed`] already
/// exists to prevent for halos, in a new place.
///
/// Derived from the tokens the effects actually use rather than from a constant kept somewhere
/// else. `elevation_max` is the tallest thing in the scene and `grazing` is how far a shadow
/// travels per unit of height at the shallowest light angle the palette permits; their product is
/// the furthest any shadow can land from its caster.
#[must_use]
pub fn max_reach(elevation_max: f32, grazing: f32) -> f32 {
    // Clamped at zero rather than trusted: a negative elevation or a negative ratio is a token
    // authoring mistake, and a negative margin would cull surfaces that are on screen.
    (elevation_max.max(0.0) * grazing.max(0.0)).max(0.0)
}

/// Whether a material contributes anything to the scene.
///
/// A material with no lit surface is still a slab — it receives shadow and occlusion even if it
/// emits nothing and has no interesting material properties. This exists to answer the opposite
/// question: whether a *fully transparent* material should occupy space in the scene. It should
/// not; an invisible surface that casts a shadow is a shadow from nothing.
#[must_use]
pub fn occupies_scene(material: &Material) -> bool {
    material
        .layers
        .iter()
        .any(|layer| layer.near.a > 0.0 || layer.far.a > 0.0)
}

/// How far a shadow travels per unit of caster height, for the key light's direction.
///
/// The horizontal run over the vertical drop: a light straight overhead (`z` dominant)
/// throws almost nothing sideways, a grazing light throws far. This is [`max_reach`]'s
/// second input, derived from the rig rather than kept as a constant beside it.
#[must_use]
pub fn grazing(direction: [f32; 3]) -> f32 {
    let run = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
    // A light with no vertical component would throw an infinite shadow; clamping the drop
    // bounds the margin instead of letting one authored vector cull nothing forever.
    run / direction[2].abs().max(0.05)
}

/// Builds one frame's [`SceneList`] from the same material/surface pairs the draw list is
/// painted from.
///
/// The builder owns the two rules a call site would otherwise each keep its own copy of:
/// culling to the viewport widened by the margin (scene-handoff rule 4), and the ceiling
/// with its counted drop, which [`SceneList::push`] already enforces (rule 5). What it does
/// **not** own is deciding what gets painted -- the caller adds exactly the materials it
/// painted, which is what keeps the two descriptions of the frame in agreement (rule 1).
#[derive(Debug)]
pub struct SceneBuilder {
    scene: SceneList,
    /// `[w, h]`, physical pixels.
    viewport: [f32; 2],
    /// Physical pixels. See [`max_reach`].
    margin: f32,
}

impl SceneBuilder {
    /// Start a scene for the frame `generation`, culled to `viewport` widened by `margin`,
    /// both in physical pixels.
    #[must_use]
    pub fn new(generation: u64, viewport: [f32; 2], margin: f32, environment: Environment) -> Self {
        let mut scene = SceneList::default();
        scene.reset(generation, environment);
        Self {
            scene,
            viewport,
            margin: margin.max(0.0),
        }
    }

    /// Add the slab for one painted material, if it occupies the scene and reaches the
    /// widened viewport.
    ///
    /// Call this beside the `paint` that pushed the material's instances, with the same
    /// [`Surface`] -- the shared origin is what makes the slab equal the instance exactly
    /// rather than approximately. Prefer [`crate::tokens::Tokens::scene_slab`] +
    /// [`SceneBuilder::admit`] where a `Tokens` is at hand: this path leaves the
    /// attenuation floor at its safe 1.0, so the slab casts but cannot be darkened.
    pub fn add(&mut self, material: &Material, surface: Surface) {
        if let Some(slab) = material.slab(surface) {
            self.admit(slab);
        }
    }

    /// Admit one already-built slab, applying the widened-viewport cull (rule 4) and the
    /// counted ceiling (rule 5, inside [`SceneList::push`]).
    pub fn admit(&mut self, slab: Slab) {
        let [x, y, w, h] = slab.rect;
        let reach = self.margin;
        let inside = x < self.viewport[0] + reach
            && x + w > -reach
            && y < self.viewport[1] + reach
            && y + h > -reach;
        if inside {
            self.scene.push(slab);
        }
    }

    /// State the key light. Exactly one; the last call wins, which a frame builder never
    /// exercises because it sets the light once from the rig.
    pub fn set_key_light(&mut self, light: Light) {
        self.scene.key_light = Some(light);
    }

    /// State the focus lamp, present only while something has keyboard focus.
    pub fn set_focus_light(&mut self, light: Option<Light>) {
        self.scene.focus_light = light;
    }

    /// The finished scene, ready to publish.
    #[must_use]
    pub fn finish(self) -> SceneList {
        self.scene
    }

    /// The slab most recently accepted, for tests that assert on what was built.
    #[must_use]
    pub fn last(&self) -> Option<&Slab> {
        self.scene.slabs.last()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::material::name;
    use crate::tokens::{Theme, Tokens};

    #[test]
    fn the_margin_is_the_furthest_a_shadow_can_land() {
        // A tall surface under a shallow light throws further than a short one under a steep one,
        // and the margin has to cover the worst case rather than the typical one.
        assert_eq!(max_reach(20.0, 1.5), 30.0);
        assert!(max_reach(40.0, 1.5) > max_reach(20.0, 1.5));
    }

    #[test]
    fn a_negative_token_cannot_produce_a_margin_that_culls_the_visible() {
        // An authoring mistake should degrade to "no margin", never to "cull things that are on
        // screen" -- which is what a negative margin would do, and it would look like surfaces
        // vanishing near the viewport edge.
        assert_eq!(max_reach(-5.0, 1.5), 0.0);
        assert_eq!(max_reach(20.0, -1.0), 0.0);
    }

    #[test]
    fn a_fully_transparent_material_does_not_occupy_the_scene() {
        // A shadow from something nobody can see is the defect this guards.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let solid = tokens.material(name::ROW_SELECTED).unwrap();
        assert!(occupies_scene(solid));

        let mut invisible = solid.clone();
        for layer in &mut invisible.layers {
            layer.near.a = 0.0;
            layer.far.a = 0.0;
        }
        assert!(!occupies_scene(&invisible));
    }

    #[test]
    fn every_shipped_materials_slab_equals_one_of_its_instances_exactly() {
        // Scene-handoff rule 1, as the contract words it: paint every shipped material,
        // build both descriptions, and the slab's rect and radius equal the corresponding
        // instance's -- with `==`, not a tolerance. Both descriptions come from the same
        // `Material`/`Surface` pair through the same `layer_shape`, so a failure here is a
        // divergence in that shared origin, which is exactly what the rule exists to catch.
        use crate::material::{Drive, Surface};

        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::embedded(theme).unwrap();
            let names: Vec<String> = tokens.material_names().map(str::to_string).collect();
            assert!(!names.is_empty());
            for name in names {
                let material = tokens.material(&name).unwrap();
                let surface = Surface::new(37.0, 91.0, 240.0, 28.0, 12.0, 2.0);

                let Some(slab) = material.slab(surface) else {
                    panic!("shipped material `{name}` produced no slab");
                };
                let mut instances = Vec::new();
                material.compile(surface, 1.0, Drive::REST, true, &mut instances);
                assert!(
                    instances
                        .iter()
                        .any(|i| i.rect == slab.rect && i.radius == slab.radius),
                    "material `{name}` ({theme:?}): slab rect {:?} radius {} matches no \
                     compiled instance",
                    slab.rect,
                    slab.radius,
                );
            }
        }
    }

    #[test]
    fn a_slab_carries_its_materials_elevation_in_physical_pixels() {
        // The scale multiply happens exactly once, at the slab -- the same place every
        // other logical length meets physical pixels. A material that authored no step lies
        // on the canvas.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let material = tokens.material(name::ROW_SELECTED).unwrap();
        let slab = material
            .slab(crate::material::Surface::new(0.0, 0.0, 100.0, 30.0, 6.0, 2.0))
            .unwrap();
        assert_eq!(slab.elevation, material.elevation * 2.0);
    }

    #[test]
    fn the_builder_culls_at_the_widened_viewport_not_the_viewport() {
        // Scene-handoff rule 4. A surface just past the edge still casts into the viewport,
        // so it must be in the scene; one past the margin cannot reach, so it must not be.
        // Culling at the bare edge is the shadow-pops-in-as-its-caster-scrolls defect.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let material = tokens.material(name::ROW_BODY).unwrap();
        let margin = 40.0;
        let mut builder = SceneBuilder::new(1, [800.0, 600.0], margin, Environment::default());

        let at = |x: f32| crate::material::Surface::new(x, 100.0, 100.0, 28.0, 6.0, 1.0);
        builder.add(material, at(820.0)); // inside the margin: casts into view
        assert!(builder.last().is_some(), "a caster inside the margin was culled");

        let before = builder.finish().slabs.len();
        let mut builder = SceneBuilder::new(1, [800.0, 600.0], margin, Environment::default());
        builder.add(material, at(841.0)); // past the margin: cannot reach
        assert_eq!(
            builder.finish().slabs.len() + before,
            1,
            "a surface past the widened viewport was included"
        );
    }

    #[test]
    fn the_margin_derives_from_the_rig_not_from_a_constant() {
        // Rule 4's second sentence: max_reach comes from the active effects' tokens. The
        // two inputs are the tallest step on the elevation scale and the key light's
        // grazing ratio, both authored in design/tokens.json.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let direction = tokens.lighting().rig.key.direction;
        let reach = max_reach(tokens.elevation_max(), grazing(direction));
        assert!(
            reach > 0.0,
            "the shipped scale and rig produce no margin at all, so every shadow will pop \
             at the viewport edge"
        );

        // Overhead light: no run, so no reach regardless of elevation.
        assert_eq!(grazing([0.0, 0.0, 1.0]), 0.0);
    }
}
