//! File-size bevels for row surfaces.
//!
//! Size changes bevel width near the edge while leaving the label's background
//! unchanged. Material roughness, metalness and lighting remain fixed.
//! Modification time and read-only status are available in row descriptions.
//! This module does not read the clock.

use serde::Deserialize;

use crate::row_source::{LoadState, RowView};

/// How a fact maps onto a material property, in **token** values.
///
/// Two ends and the span between them. `at_min` is what the surface is at the bottom of the
/// range and `at_max` at the top; neither is required to be the larger, because an encoding
/// that runs downward (a fresh file is *smoother*) is as legitimate as one that runs upward
/// and inverting it in code would hide the direction from the file that authors it.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct Response {
    pub at_min: f32,
    pub at_max: f32,
    /// Where the top of the range sits, in whatever unit the field it belongs to is measured
    /// in. Documented per field rather than here, because one shared doc comment would be
    /// wrong for every field but one.
    #[serde(default = "one")]
    pub span: f32,
}

fn one() -> f32 {
    1.0
}

impl Response {
    /// The identity: a response that says the same thing everywhere.
    pub const FLAT: Self = Self {
        at_min: 0.0,
        at_max: 0.0,
        span: 1.0,
    };

    /// Interpolate between the endpoints with `t` in `0..=1`.
    ///
    /// The result stays between the bounds regardless of their order.
    #[must_use]
    pub fn at(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        self.at_min + (self.at_max - self.at_min) * t
    }
}

/// The response curves, from `design/tokens.json`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct SubstanceTokens {
    /// Bevel width in **logical pixels** against size. `span` is an exponent: a size of
    /// `2^span` bytes or more sits at `at_max`. Logarithmic because file sizes are, and a
    /// linear ramp would put every ordinary file at the bottom of the scale and one video at
    /// the top.
    pub size: Response,
}

impl Default for SubstanceTokens {
    /// Every encoding flat.
    ///
    /// A token file that says nothing about substance gets no substance, for the reason
    /// [`crate::tokens::LitAllowance::default`] gives: the safe default for an expressive
    /// feature is off, and inventing a curve would put facts on screen nobody authored.
    fn default() -> Self {
        Self {
            size: Response::FLAT,
        }
    }
}

/// What a surface is made of, as opposed to what it is doing.
///
/// The counterpart to [`crate::material::Drive`], and separate from it because the two come
/// from different places and retire at different times: a drive falls to zero when its motion
/// plan settles, and a substance does not change until the file does. Folding them into one
/// type would make a row's age something an animation could alter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Substance {
    /// Bevel width in **logical** pixels, the same unit [`crate::material::Layer::bevel`] is
    /// authored in. The physical multiply happens where every other length's does.
    pub bevel: f32,
}

impl Substance {
    /// What a row gets when nothing is known about it.
    ///
    /// Not a midpoint and not an average: a stub row must look like a row nobody has anything
    /// to say about, and the midpoint of an encoding is a *claim* — "this file is of middling
    /// age" — made about a file whose attributes are undefined rather than merely unread.
    pub const UNKNOWN: Self = Self { bevel: 0.0 };

    /// Derive a surface from one row.
    ///
    /// A [`LoadState::Stub`] row returns [`Substance::UNKNOWN`] and reads none of its fields.
    /// That is not defensiveness: `size`, `mtime` and `kind` on a stub are **undefined**, not
    /// stale, so deriving a material from them would put a confident-looking surface on a row
    /// whose attributes are zeroes standing in for nothing.
    #[must_use]
    pub fn of(row: &RowView, tokens: SubstanceTokens) -> Self {
        if row.state == LoadState::Stub {
            return Self::UNKNOWN;
        }

        // Size, logarithmically. `size + 1` so an empty file is defined rather than -inf.
        let bits = ((row.size + 1) as f32).log2();
        let size_t = if tokens.size.span > 0.0 {
            bits / tokens.size.span
        } else {
            0.0
        };

        Self {
            bevel: tokens.size.at(size_t).max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]

    use super::*;
    use crate::row_source::{KindId, RowFlags, RowId};

    const NOW: i64 = 1_786_060_800;

    fn tokens() -> SubstanceTokens {
        SubstanceTokens {
            size: Response {
                at_min: 0.5,
                at_max: 3.0,
                span: 30.0,
            },
        }
    }

    fn row(state: LoadState) -> RowView {
        RowView {
            id: RowId(0),
            name: 0..0,
            size: 1024,
            mtime: NOW,
            kind: KindId(0),
            flags: RowFlags::EMPTY,
            state,
            depth: 0,
        }
    }

    #[test]
    fn a_stub_gets_the_unknown_surface_and_not_a_derived_one() {
        // The whole point. `size`, `mtime` and `kind` are undefined on a stub, so a stub
        // holding the *same* zeroes as a filled row must still not be read: this row claims
        // a size and an mtime, and the state says do not believe them.
        let mut stub = row(LoadState::Stub);
        stub.size = u64::MAX;
        stub.mtime = 0;
        assert_eq!(Substance::of(&stub, tokens()), Substance::UNKNOWN);
    }

    #[test]
    fn the_bevel_stays_an_edge_treatment_at_every_shipped_row_height() {
        // The invariant the whole encoding rests on, and the reason bevel survived while
        // roughness did not. `bevel_normal` perturbs the surface normal only within `bevel`
        // of the edge, so the row's centre -- where the label is -- shades identically
        // whatever the file's size, and the contrast gate's composite stays honest.
        //
        // That holds only while two bevels do not meet in the middle. The smallest row the
        // product ships is Compact at the smallest text scale, and this is what stops a
        // future "make it more visible" edit from quietly turning an edge treatment into a
        // full-surface one that no gate can see.
        let file: crate::tokens::TokenFile =
            serde_json::from_str(include_str!("../../../design/tokens.json")).expect("tokens");
        let widest = file.substance.size.at_min.max(file.substance.size.at_max);

        let smallest_row =
            crate::density::Density::Compact.base_height() * crate::density::MIN_TEXT_SCALE;
        assert!(
            widest * 2.0 < smallest_row,
            "a bevel of {widest} logical px meets itself on a {smallest_row} px row, so the              centre is no longer flat and the encoding has become a brightness change the              contrast gate cannot see"
        );
    }

    #[test]
    fn size_is_logarithmic_so_ordinary_files_are_not_all_at_the_bottom() {
        let sizes = [0u64, 4_096, 4_194_304, 4_294_967_296];
        let bevels: Vec<f32> = sizes
            .iter()
            .map(|&size| {
                let mut r = row(LoadState::Basic);
                r.size = size;
                Substance::of(&r, tokens()).bevel
            })
            .collect();

        for pair in bevels.windows(2) {
            assert!(pair[0] < pair[1], "{bevels:?} must increase with size");
        }
        // The load-bearing property: a 4 KB file and a 4 MB file differ by *more* than a
        // linear ramp would give them, which is the entire reason the scale is logarithmic.
        // On a linear scale over 4 GB, both would sit within 0.1% of the bottom.
        let small_step = bevels[2] - bevels[1];
        let full = bevels[3] - bevels[0];
        assert!(
            small_step > full * 0.2,
            "a thousandfold step in the middle of the range must be visible: {bevels:?}"
        );
    }

    #[test]
    fn permission_changes_nothing_about_the_surface() {
        // The drop, asserted rather than left as a deletion. Reflectivity-as-permission was
        // tried, rendered, and came out inverted -- `env` is the key-light/sky mix, not
        // reflectivity, so `env = 0` is a surface lit ENTIRELY by the key light and therefore
        // brighter. Re-adding it by mapping permission onto any material field puts a row's
        // lighting rig in the hands of its ACL, and this is what says no.
        //
        // Permission still reaches the user: the flag is populated and the a11y description
        // announces it. See `a11y::describe_tests`.
        let mut locked = row(LoadState::Basic);
        locked.flags = RowFlags::IS_READONLY;
        assert_eq!(
            Substance::of(&locked, tokens()),
            Substance::of(&row(LoadState::Basic), tokens())
        );
    }

    #[test]
    fn the_default_tokens_encode_nothing_at_all() {
        // A token file that never opted in must not acquire an opinion about anybody's
        // files. Every row is the same surface, whatever it is made of.
        let flat = SubstanceTokens::default();
        let mut a = row(LoadState::Basic);
        a.size = 17;

        let mut b = row(LoadState::Basic);
        b.size = 9_000_000_000;
        b.mtime = NOW;

        assert_eq!(Substance::of(&a, flat), Substance::of(&b, flat));
    }

    #[test]
    fn the_row_body_composites_to_exactly_the_colour_it_replaced() {
        // Acceptance 4, as an assertion rather than a hope: "a failure means an encoding
        // leaked into brightness". The row body used to be the canvas on even rows and a
        // bare rect in `surface/row-alt` on odd ones. `Material::composites` walks Pbr
        // layers, so if either albedo is off by anything at all, every contrast ratio for
        // every label on every row moves -- and it moves by an amount small enough to pass
        // the gate while being exactly the leak this encoding promised not to have.
        use crate::material::name;
        use crate::tokens::{Theme, Tokens};

        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::embedded(theme).unwrap();
            let base = tokens.color("surface/base");

            for (material, expected_name) in [
                (name::ROW_BODY, "surface/base"),
                (name::ROW_BODY_ALT, "surface/row-alt"),
            ] {
                let expected = tokens.color(expected_name);
                let composites = tokens.material(material).unwrap().composites(base);
                // One colour, and this is the half that actually guards the encoding: a
                // material that presented two would be one whose ground varies, and a
                // ground that varies per file is the brightness leak this whole test is
                // about.
                assert_eq!(
                    composites.len(),
                    1,
                    "{material} in the {theme:?} theme presents more than one colour: \
                     {composites:?}"
                );
                let got = composites[0];

                // The row bodies used to be opaque, and this used to assert the composite
                // was the albedo byte for byte. They are translucent now, so the ambient
                // field reads through them, and `row/body-alt` over `surface/base` is
                // deliberately a BLEND of the two rather than `surface/row-alt` itself --
                // the zebra banding is a third of what it was, which the token file already
                // wanted near-invisible.
                //
                // So the claim moves to the one that was always load-bearing: the composite
                // introduces NO THIRD COLOUR. It lies on the straight line between the
                // ground and the declared albedo, at one mixing fraction shared by every
                // channel. An albedo carrying a fact -- warmer for a recent file, darker for
                // a big one -- leaves that line immediately, which is exactly what this test
                // exists to catch and exactly what it still catches.
                //
                // Deriving the fraction from the red channel and checking the other two
                // against it is what makes "one fraction" a measurement rather than three
                // independent tolerances that a per-channel tint would slip through.
                let span = expected.r - base.r;
                let mix = if span.abs() > 1e-6 {
                    (got.r - base.r) / span
                } else {
                    // Identical albedo and ground -- `row/body` over `surface/base`. Any
                    // fraction reproduces it, so the check below degenerates correctly.
                    1.0
                };
                assert!(
                    (0.0..=1.0).contains(&mix),
                    "{material} in the {theme:?} theme composites to {got:?}, which is not \
                     between {base:?} and {expected_name} {expected:?} at all"
                );
                for (channel, (under, over, is)) in [
                    ("g", (base.g, expected.g, got.g)),
                    ("b", (base.b, expected.b, got.b)),
                ] {
                    let want = under + (over - under) * mix;
                    assert!(
                        (is - want).abs() <= 1e-5,
                        "{material} in the {theme:?} theme mixes {mix} of {expected_name} on \
                         red but {channel} lands at {is} rather than {want}: the albedo is \
                         tinted away from its token, which is a fact leaking into colour"
                    );
                }
                assert!(
                    (got.a - 1.0).abs() <= 1e-5,
                    "{material} in the {theme:?} theme left the ground translucent: {got:?}"
                );
            }
        }
    }

    #[test]
    fn a_substance_cannot_change_what_the_contrast_gate_sees() {
        // The same guarantee from the other direction, and the stronger half: the composite
        // must be the same for every row whatever its file is like. Roughness, bevel and
        // environment are shading inputs; if any of them ever reached the albedo, an old
        // read-only 4 GB file would sit under text at a different ratio from a new one and
        // no gate would know, because the gate checks the material and not the row.
        use crate::material::name;
        use crate::tokens::{Theme, Tokens};

        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let base = tokens.color("surface/base");
        let material = tokens.material(name::ROW_BODY).unwrap();
        let reference = material.composites(base);

        for substance in [
            Substance::UNKNOWN,
            Substance { bevel: 12.0 },
            Substance { bevel: 0.0 },
        ] {
            let mut layers = material.clone();
            for layer in &mut layers.layers {
                *layer = layer.substanced(Some(substance));
            }
            assert_eq!(
                layers.composites(base),
                reference,
                "substance {substance:?} moved a colour the contrast gate reads"
            );
        }
    }

    #[test]
    fn a_zero_span_does_not_divide_by_zero() {
        // An authoring mistake should degrade to "encodes nothing", never to NaN, which
        // would reach a vertex buffer and take the whole frame with it.
        let mut broken = tokens();
        broken.size.span = 0.0;
        let s = Substance::of(&row(LoadState::Basic), broken);
        assert!(s.bevel.is_finite());
    }
}
