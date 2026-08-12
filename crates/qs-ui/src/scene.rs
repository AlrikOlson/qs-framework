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

use crate::material::Material;

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
}
