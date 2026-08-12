//! The lighting pass: what one surface does to another.
//!
//! # Why this is a second pass and not a branch in the first one
//!
//! Every primitive the renderer has drawn since M0 is a function of one fragment's own position.
//! That is exactly why one pass and one target sufficed, and it is not a coincidence — it is what
//! made the CPU tier reproducible and the parity suite meaningful.
//!
//! Shadows, occlusion and bounced light are functions of *other* surfaces. There is no formulation
//! of them that reads only the fragment being shaded, so a branch in the existing pass cannot
//! express them: the fragment would have to know about geometry it cannot see. This is the one
//! place where a second pass is not architectural preference but arithmetic. See
//! `specs/002-ray-traced-mode/research.md` R2 and the plan's Complexity Tracking.
//!
//! # Modulate, do not replace
//!
//! The surface passes render as they always have. This pass multiplies attenuation into the result
//! and adds light. Text and icons are drawn **afterwards** and are never lit.
//!
//! That ordering is what makes the contrast obligation solvable at all. A lit glyph's contrast
//! against its background varies per pixel with the geometry, and no build-time gate can bound
//! that. Drawn after, a glyph always sits on `surface x attenuation + addition`, both terms
//! bounded, so the worst case is a closed-form expression rather than a sample. See
//! `specs/002-ray-traced-mode/contracts/lit-contrast.md`.
//!
//! # Nothing accumulates
//!
//! This pass is a pure function of the scene and the surface image. There is no accumulator, so
//! there is nothing to converge and nothing to terminate, and a frame is a complete picture the
//! moment it is drawn. That is a stronger guarantee than "the refinement stops on its own", and it
//! is what keeps the zero-work-at-rest obligation untouched by this feature.

use crate::path::RenderPath;

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
    /// Nothing at all. The interface is plainer, not partial.
    ///
    /// The right answer more often than it looks. An approximation of bounced light is a coloured
    /// wash somebody has to invent, and an invented wash is decoration — which Principle IX
    /// refuses. Absent reads as a plainer theme.
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

    /// The name the `qs-bench` scenario for this effect answers to.
    ///
    /// A third exhaustive match, and the one that makes Principle IX.6 a field rather than a line
    /// in a checklist: an effect that names no scenario cannot be measured, and an effect that
    /// cannot be measured is not complete.
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
