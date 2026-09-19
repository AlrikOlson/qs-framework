//! Lighting scenes built alongside draw lists.
//!
//! Slabs and drawing instances come from the same material and surface, so their
//! rectangles and radii agree. The scene contains the visible rows plus the
//! margin required by [`crate::scene::max_reach`] for shadows from nearby surfaces.

use crate::material::{Material, Surface};
use crate::tokens::FocusLightTokens;
use qs_gpu::scene::{Environment, FocusLamp, Light, SceneList, Slab};

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
/// throws almost nothing sideways, a grazing light throws far. This is [`crate::scene::max_reach`]'s
/// second input, derived from the rig rather than kept as a constant beside it.
#[must_use]
pub fn grazing(direction: [f32; 3]) -> f32 {
    let run = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
    // A light with no vertical component would throw an infinite shadow; clamping the drop
    // bounds the margin instead of letting one authored vector cull nothing forever.
    run / direction[2].abs().max(0.05)
}

/// Build a focus lamp over the focused region.
///
/// `rect` and `elevation` come from the painted surface in physical pixels.
/// `gain` scales both strengths; zero has no lighting contribution.
/// The lamp covers the row's rectangle rather than a single point.
#[must_use]
pub fn focus_light(
    rect: [f32; 4],
    elevation: f32,
    tokens: FocusLightTokens,
    scale: f32,
    gain: f32,
) -> FocusLamp {
    let gain = gain.clamp(0.0, 1.0);
    FocusLamp {
        rect,
        // The authored height is a logical length, so it meets physical pixels here — the same
        // single multiply every other authored length gets, at the slab.
        height: elevation + tokens.height * scale.max(0.0),
        size: tokens.size_deg,
        share: tokens.share() * gain,
        ambient: tokens.ambient() * gain,
    }
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
    /// Physical pixels. See [`crate::scene::max_reach`].
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
    pub fn set_focus_light(&mut self, lamp: Option<FocusLamp>) {
        self.scene.focus_light = lamp;
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
            .slab(crate::material::Surface::new(
                0.0, 0.0, 100.0, 30.0, 6.0, 2.0,
            ))
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
        assert!(
            builder.last().is_some(),
            "a caster inside the margin was culled"
        );

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

    #[test]
    fn moving_focus_moves_the_lamp_in_the_published_scene() {
        // T059, and the assertion is about the *published* scene rather than about the
        // helper's return value on purpose. `set_focus_light` takes an `Option`, so the
        // failure this catches is not "the arithmetic is wrong" — it is a builder that
        // computes a lamp correctly and publishes the previous frame's, or none, which no
        // test of `focus_light` alone can see.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let focus = tokens.lighting().rig.focus;

        let published = |row_top: f32| -> FocusLamp {
            let mut builder = SceneBuilder::new(1, [800.0, 600.0], 40.0, Environment::default());
            builder.set_focus_light(Some(focus_light(
                [12.0, row_top, 300.0, 28.0],
                6.0,
                focus,
                2.0,
                1.0,
            )));
            builder
                .finish()
                .focus_light
                .expect("the lamp was not published")
        };

        let first = published(100.0);
        let second = published(240.0);

        // The lamp's strip IS the focused region's rect, not a rectangle derived from it. That
        // equality is what keeps the light over the ring: the two come from the same surface,
        // so they cannot drift, and a light that is not quite over the thing it is finding
        // sends the eye to the wrong place.
        assert_eq!(
            first.rect,
            [12.0, 100.0, 300.0, 28.0],
            "the lamp's strip is not the focused region's rect"
        );
        assert!(
            (second.rect[1] - first.rect[1] - 140.0).abs() < 1e-4,
            "focus moved 140 px and the lamp moved {}",
            second.rect[1] - first.rect[1]
        );
        assert_eq!(
            (second.rect[0], second.rect[2]),
            (first.rect[0], first.rect[2]),
            "the lamp moved sideways or changed width for a purely vertical move"
        );

        // The height is authored in logical pixels and multiplied by the scale exactly once.
        // A lamp that skipped the multiply hangs inside the surface it is meant to be over on
        // every high-DPI display, and the picture looks like the mode is simply off.
        assert!(
            (first.height - (6.0 + focus.height * 2.0)).abs() < 1e-4,
            "the lamp hangs at {} above the canvas, not at the authored height",
            first.height
        );
    }

    #[test]
    fn a_lamp_at_zero_gain_is_arithmetically_absent() {
        // The shader mixes the lamp's shading term in by its intensity, so intensity zero is
        // the identity — which is what makes "coming up" and "going out" continuous with
        // "no focus at all" rather than a step at each end. Asserted here because the
        // property lives in this multiply, not in the shader.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let focus = tokens.lighting().rig.focus;
        let at = |gain: f32| focus_light([0.0, 0.0, 100.0, 28.0], 4.0, focus, 1.0, gain);

        // BOTH halves scale with the gain. Scaling only the share would make a lamp at zero
        // gain still dim the room -- so "focus has not arrived yet" and "nothing has focus"
        // would draw differently, which is a step change in every shadow at the moment focus
        // lands, i.e. the snap the ramp exists to remove.
        assert_eq!(at(0.0).share, 0.0);
        assert_eq!(at(0.0).ambient, 0.0);
        assert!(at(0.5).share > 0.0 && at(0.5).share < at(1.0).share);
        assert!(at(0.5).ambient > 0.0 && at(0.5).ambient < at(1.0).ambient);
        assert!(
            (at(1.0).share - focus.share()).abs() < 1e-6
                && (at(1.0).ambient - focus.ambient()).abs() < 1e-6,
            "a fully-lit lamp does not burn at its authored strengths"
        );
        // A hand-edited token file cannot spend more than the budget, and a gain outside its
        // range cannot either — both clamp, so the mix stays a mix.
        assert!(at(4.0).share <= 1.0 && at(-1.0).share == 0.0 && at(-1.0).ambient == 0.0);
    }
}
