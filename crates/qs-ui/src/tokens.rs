//! The token set: every visual value, resolved for a theme.
//!
//! Constitution VII makes `design/tokens.json` authoritative and forbids hard-coded visual
//! values in widget code. This module is the only place that reads that file, and
//! [`Tokens::color`] is the only way to obtain a colour. A widget that wants a shade not in
//! the file adds a token; it does not add a literal.
//!
//! # Forced colours
//!
//! High-contrast and forced-colours modes are not "a third theme". The operating system
//! supplies a small, fixed palette and the correct response is to use *only* those colours
//! and to switch effects off -- a shadow or a translucent overlay in forced-colours mode
//! defeats the entire point of the mode, which is that the user has told the system exactly
//! which colours they can see. [`Tokens::forced`] maps every token onto that palette and
//! sets [`Tokens::effects_enabled`] to false.

use std::collections::BTreeMap;

use qs_gpu::color::{Oklch, Srgba};
use qs_gpu::frame::Instance;
use serde::Deserialize;

use crate::material::{Drive, Material, MaterialDef, Surface};
use crate::substance::{Substance, SubstanceTokens};

/// One colour family: a fixed hue, and optionally the chroma curve its neutral stops take.
///
/// A ramp owns the hue and nothing else owns it. That is the whole mechanism behind "the
/// greys hold a constant hue": there is no per-token hue to drift, because a token names a
/// ramp and a lightness rather than a colour. Before this existed the single grey family
/// spanned 264.5° to 285.8° — the light theme near 268°, the dark *surfaces* at 285.6° and
/// the dark *text* at 269–279°, so one theme contained two different greys and nothing in
/// the file recorded which was intended.
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Ramp {
    /// Hue angle in degrees. Every stop on this ramp uses it, in both themes.
    pub hue: f32,
    /// Peak chroma of the ramp's curve, for stops that state no chroma of their own.
    ///
    /// The curve is `chroma · (4·l·(1−l))^1.5`: zero at both ends, peaking at `l = 0.5`. It
    /// has to vanish at the ends because the sRGB gamut does — at `l = 1.0` the only colour
    /// is white — and a UI grey wants its faint cast in the mid-tones, where there is room
    /// for it, rather than in the near-blacks where it reads as a colour cast.
    #[serde(default)]
    pub chroma: Option<f32>,
}

impl Ramp {
    pub fn curve_chroma(self, lightness: f32) -> f32 {
        let peak = self.chroma.unwrap_or(0.0);
        peak * (4.0 * lightness * (1.0 - lightness)).max(0.0).powf(1.5)
    }
}

/// How one theme's value of a token is authored.
///
/// Ramp-relative is the form the shipped file uses; `check_every_token_is_ramp_relative`
/// refuses a bare hex in it. `Hex` stays supported because test fixtures need a literal and
/// because a token file is a public artifact that should not fail to parse over a form it
/// used to accept — but a hex literal in the shipped palette is exactly the hand-picking
/// this ramp model exists to remove, so it is a build failure rather than a style note.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ColorValue {
    /// `"#rrggbb"`.
    Hex(String),
    /// `{ "ramp": "neutral", "l": 974 }`, plus at most one of `c` or `s`.
    Ramped {
        /// Which ramp, and therefore which hue.
        ramp: String,
        /// Lightness in **per-mille** OKLCH `l`, so `974` is `0.974`. An integer because a
        /// design file full of `0.974` is harder to scan and invites false precision.
        l: u16,
        /// Absolute chroma. For **tints** — a colour whose intent is "this much colour",
        /// independent of what the gamut happens to allow at this lightness.
        #[serde(default)]
        c: Option<f32>,
        /// Chroma as a fraction of the maximum in-gamut chroma at this lightness and hue.
        /// For **accents** — a colour whose intent is "as colourful as this hue gets here".
        ///
        /// Both spellings are needed and neither covers the other, because sRGB is not a
        /// cylinder. Matching `s` across two themes is the only way to say "equally
        /// saturated" when the two themes put a token at lightnesses whose chroma ceilings
        /// differ by 2x or more; matching `c` is the only way to keep a *subtle* tint
        /// subtle, since a fixed fraction of a ceiling that varies 5x does not stay subtle.
        #[serde(default)]
        s: Option<f32>,
    },
}

/// Which theme a token resolves against.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Theme {
    #[default]
    Light,
    Dark,
}

/// What contrast rule a pair is held to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PairKind {
    /// Body text: 4.5:1 (WCAG 1.4.3 AA).
    Text,
    /// Text at 18.66px bold or 24px regular: 3:1.
    LargeText,
    /// A meaningful non-text boundary: 3:1 (WCAG 1.4.11).
    Boundary,
}

impl PairKind {
    pub fn minimum_ratio(self) -> f32 {
        match self {
            Self::Text => 4.5,
            Self::LargeText | Self::Boundary => 3.0,
        }
    }
}

/// What kind of value a token holds. Drives which contrast rule could apply.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TokenRole {
    Foreground,
    #[default]
    Background,
    Border,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TokenDef {
    pub role: TokenRole,
    pub light: ColorValue,
    pub dark: ColorValue,
    #[serde(default)]
    pub description: String,
}

/// A colour that is **light rather than paint**, and therefore not a token.
///
/// # Why this is its own block and not a `TokenRole`
///
/// [`Tokens::bloom_ceiling`] is a fold over every entry in `tokens`: a pixel blooms when it
/// is brighter than every colour the palette can name. So an emissive placed *in* the palette
/// raises the ceiling to exactly its own luminance and is then not above it — the bloom stays
/// off, and the criterion "an emissive above the ceiling" becomes unsatisfiable by
/// construction. That is not a bug in the fold; it is the fold saying an emissive is not a
/// colour a designer wrote down.
///
/// Keeping the block outside `tokens` makes three separate safeties structural rather than
/// careful. The ceiling cannot rise to meet an emissive because it never sees one.
/// [`Tokens::color`] cannot reach one, so no call site can paint a fill or set a glyph in it
/// by typo. And "not meaning-bearing" stops being prose in a design document: it is refused
/// at load time by [`crate::material::resolve_all`], which admits an emissive only on a layer
/// that declares an `edge` — the band geometry [`crate::material::Material::composites`]
/// already excludes — and never in a material's `text`, `text_lit` or `over`.
///
/// # There is no `role`
///
/// [`TokenRole`] exists so the contrast gate knows which side of a pair a colour can be on.
/// An emissive is on neither side: it is not a background, because nothing is read on it, and
/// it is not a foreground, because it carries no meaning. Giving it a role would be inviting
/// the gate to check a pair that cannot occur.
/// What every emissive's name must start with. See [`EmissiveDef`].
pub const EMISSIVE_PREFIX: &str = "emissive/";

#[derive(Clone, Debug, Deserialize)]
pub struct EmissiveDef {
    pub light: ColorValue,
    pub dark: ColorValue,
    #[serde(default)]
    pub description: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ContrastPair {
    pub foreground: String,
    pub background: String,
    pub kind: PairKind,
}

/// One of the five type roles from UXDD §10.1.
///
/// Size is in **logical** pixels at the default density; it is multiplied by the device
/// scale and the OS text-size setting at use, never baked. Weight is an OpenType weight
/// *class*, not a font file — which face satisfies it is a platform question, and on
/// Windows the answer is a different font family per weight (see `qs_text::fontdb`).
#[derive(Clone, Copy, PartialEq, Debug, Deserialize)]
pub struct TypeRole {
    pub size: f32,
    pub weight: u16,
}

impl TypeRole {
    /// Physical pixel size for this role.
    pub fn size_px(self, scale: f32, text_scale: f32) -> f32 {
        self.size * crate::density::clamp_text_scale(text_scale) * sane_scale(scale)
    }
}

fn sane_scale(scale: f32) -> f32 {
    if scale.is_nan() || scale <= 0.0 {
        1.0
    } else {
        scale.clamp(0.5, 8.0)
    }
}

/// Focus-indicator widths, in logical pixels (UXDD 10.5).
#[derive(Clone, Copy, PartialEq, Debug, Deserialize)]
pub struct FocusTokens {
    pub ring_width: f32,
    pub outline_width: f32,
}

impl Default for FocusTokens {
    fn default() -> Self {
        // A focus ring is an accessibility requirement, not a decoration, so a token file
        // that forgets it still gets one rather than silently rendering none.
        Self {
            ring_width: 2.0,
            outline_width: 1.0,
        }
    }
}

/// Space-scale step names. The scale is a closed set on purpose -- see the token file.
pub mod space {
    pub const XS: &str = "xs";
    pub const SM: &str = "sm";
    pub const MD: &str = "md";
    pub const LG: &str = "lg";
    pub const XL: &str = "xl";
    pub const XXL: &str = "2xl";
    pub const XXXL: &str = "3xl";
}

/// Radius names the surface class, not the number.
pub mod radius {
    pub const CHIP: &str = "chip";
    pub const ROW: &str = "row";
    pub const PANEL: &str = "panel";
    pub const POPOVER: &str = "popover";
}

/// The role names the row renderer asks for. Constants rather than bare strings so a typo
/// is a compile error instead of a silently missing role.
pub mod role {
    /// Badges, column headers.
    pub const XS: &str = "ui/xs";
    /// Secondary metadata: size, modified, kind.
    pub const SM: &str = "ui/sm";
    /// Row names, body.
    pub const MD: &str = "ui/md";
    /// Section headers.
    pub const LG: &str = "ui/lg";
    /// Empty-state headlines.
    pub const XL: &str = "ui/xl";
}

/// Raw deserialized form of `design/tokens.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct TokenFile {
    pub version: u32,
    /// The colour families. Defaulted so a token file predating them still parses; a token
    /// naming a ramp that is absent is an error rather than a fallback colour.
    #[serde(default)]
    pub ramps: BTreeMap<String, Ramp>,
    pub tokens: BTreeMap<String, TokenDef>,
    /// Light the renderer makes, deliberately *not* in `tokens`. Defaulted so a token file
    /// predating the bloom's source still loads — and loads with the bloom off, which is what
    /// it had. See [`EmissiveDef`] for why the placement is the whole safety.
    #[serde(default)]
    pub emissive: BTreeMap<String, EmissiveDef>,
    /// The five type roles. Defaulted so a token file predating them still loads; a caller
    /// asking for a role that is absent gets `None` and can say so.
    #[serde(default, rename = "type")]
    pub type_roles: BTreeMap<String, TypeRole>,
    /// The space scale, in logical pixels.
    #[serde(default)]
    pub space: BTreeMap<String, f32>,
    /// Corner radii by surface class, in logical pixels.
    #[serde(default)]
    pub radius: BTreeMap<String, f32>,
    /// The elevation scale, in logical pixels: how far a surface stands off the canvas, as a
    /// closed set of named steps. Defaulted so a token file predating the lit mode still
    /// loads; a material naming a step then fails resolution, which is the honest outcome —
    /// the file authored a height against a scale it does not carry.
    #[serde(default)]
    pub elevation: BTreeMap<String, f32>,
    #[serde(default)]
    pub focus: FocusTokens,
    /// Named looks, each a stack of effect layers over the colour and space scales.
    ///
    /// Defaulted so a token file predating materials still loads. See [`crate::material`]
    /// for what a material is and why the contrast gate reads them.
    #[serde(default)]
    pub materials: BTreeMap<String, MaterialDef>,
    /// Entries that are pure documentation (an object with only `$comment`) are dropped
    /// during deserialization by [`ContrastPair`]'s required fields, so the list here is
    /// already only real pairs.
    #[serde(default, deserialize_with = "pairs_ignoring_comments")]
    pub contrast_pairs: Vec<ContrastPair>,
    /// What the lighting pass may do to each class of surface, per theme.
    ///
    /// Defaulted to no lighting at all so a token file predating the lit mode still loads and is
    /// gated exactly as it is today. See [`LightingTokens::default`] for why the safe default is
    /// "none" here and "something" for the focus ring.
    #[serde(default)]
    pub lighting: LightingTokens,
    /// How a file's facts map onto what its surface is made of. See [`crate::substance`].
    #[serde(default)]
    pub substance: SubstanceTokens,
}

fn pairs_ignoring_comments<'de, D>(deserializer: D) -> Result<Vec<ContrastPair>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // The file interleaves comment objects with real pairs so the rationale lives next to
    // the data it explains. Dropping anything that does not parse as a pair keeps that
    // possible without a second file or a stripped-comment build step.
    let raw = Vec::<serde_json::Value>::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .filter_map(|value| serde_json::from_value::<ContrastPair>(value).ok())
        .collect())
}

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("could not read the token file: {0}")]
    Io(#[from] std::io::Error),
    #[error("the token file is not valid JSON: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("token `{token}` has an unparseable {theme} colour `{value}`")]
    BadColor {
        token: String,
        theme: &'static str,
        value: String,
    },
    #[error("token `{token}` names ramp `{ramp}` in its {theme} value, which is not declared")]
    UnknownRamp {
        token: String,
        theme: &'static str,
        ramp: String,
    },
    #[error(
        "token `{token}` gives both `c` and `s` for its {theme} value; they are two ways to \
         say the same thing and only one can win"
    )]
    ChromaStatedTwice { token: String, theme: &'static str },
    #[error("contrast pair references unknown token `{0}`")]
    UnknownToken(String),
    #[error("material `{material}` names token `{token}`, which is not declared")]
    MaterialUnknownToken { material: String, token: String },
    #[error("material `{material}` names space step `{step}`, which is not on the scale")]
    MaterialUnknownStep { material: String, step: String },
    #[error("material `{material}` names radius class `{class}`, which is not declared")]
    MaterialUnknownRadius { material: String, class: String },
    #[error("material `{material}` names elevation step `{step}`, which is not on the scale")]
    MaterialUnknownElevation { material: String, step: String },
    #[error(
        "material `{material}` declares {centres} field centres, but the field uniform holds \
         {limit}; the extras would be authored, ungated and never drawn"
    )]
    MaterialTooManyCentres {
        material: String,
        centres: usize,
        limit: usize,
    },
    #[error(
        "material `{material}` declares text but no `over`, so the gate would have nothing \
         to composite its layers against"
    )]
    MaterialMissingBase { material: String },
    #[error(
        "material `{material}` paints emissive `{token}` on a layer with no `edge`; an \
         emissive is light rather than paint and is admitted only on an edge band, which is \
         the one geometry `Material::composites` excludes and therefore the one colour no \
         text is ever read against"
    )]
    MaterialEmissiveOffABand { material: String, token: String },
    #[error(
        "material `{material}` names emissive `{token}` as text or as a base; an emissive has \
         no `role` because it belongs on neither side of a contrast pair, and this one would \
         put it on one"
    )]
    MaterialEmissiveIsNotAColour { material: String, token: String },
    #[error(
        "emissive `{name}` does not start with `{prefix}`; a layer names one string and two \
         maps can answer it, so the name has to say which",
        prefix = EMISSIVE_PREFIX
    )]
    EmissiveUnprefixed { name: String },
}

impl ColorValue {
    /// Resolve to sRGB. `ramps` supplies the hue; out-of-gamut requests come back with
    /// their chroma reduced rather than their hue clipped.
    pub fn resolve(
        &self,
        ramps: &BTreeMap<String, Ramp>,
        token: &str,
        theme: &'static str,
    ) -> Result<Srgba, TokenError> {
        match self {
            Self::Hex(text) => Srgba::parse_hex(text).ok_or_else(|| TokenError::BadColor {
                token: token.to_string(),
                theme,
                value: text.clone(),
            }),
            Self::Ramped { ramp, l, c, s } => {
                if c.is_some() && s.is_some() {
                    return Err(TokenError::ChromaStatedTwice {
                        token: token.to_string(),
                        theme,
                    });
                }
                let family = ramps.get(ramp).ok_or_else(|| TokenError::UnknownRamp {
                    token: token.to_string(),
                    theme,
                    ramp: ramp.clone(),
                })?;
                let lightness = f32::from(*l) / 1000.0;
                let chroma = match (c, s) {
                    (Some(absolute), _) => *absolute,
                    (_, Some(fraction)) => fraction * Oklch::max_chroma(lightness, family.hue),
                    _ => family.curve_chroma(lightness),
                };
                Ok(Oklch::new(lightness, chroma, family.hue).to_srgb())
            }
        }
    }

    /// The ramp this value is authored against, if any. `None` means a hex literal.
    pub fn ramp_name(&self) -> Option<&str> {
        match self {
            Self::Hex(_) => None,
            Self::Ramped { ramp, .. } => Some(ramp),
        }
    }
}

/// A theme's worth of resolved colours.
#[derive(Clone, Debug)]
pub struct Tokens {
    theme: Theme,
    colors: BTreeMap<String, Srgba>,
    /// Resolved emissives, kept in a **separate map from `colors`** on purpose. See
    /// [`EmissiveDef`]: merging the two would let [`Tokens::bloom_ceiling`] rise to meet the
    /// very thing it exists to be cleared by, and would put light within reach of
    /// [`Tokens::color`].
    emissive: BTreeMap<String, Srgba>,
    roles: BTreeMap<String, TokenRole>,
    type_roles: BTreeMap<String, TypeRole>,
    space: BTreeMap<String, f32>,
    radius: BTreeMap<String, f32>,
    elevation: BTreeMap<String, f32>,
    focus: FocusTokens,
    materials: BTreeMap<String, Material>,
    lighting: LightingTokens,
    substance: SubstanceTokens,
    effects_enabled: bool,
}

impl Tokens {
    /// The token file compiled into the binary.
    ///
    /// Embedded rather than read from disk at runtime: a token file that can be edited
    /// beside the executable is a token file that can disagree with the contrast gate that
    /// passed at build time, which turns a build-time guarantee into a hope.
    pub fn embedded(theme: Theme) -> Result<Self, TokenError> {
        Self::from_str(include_str!("../../../design/tokens.json"), theme)
    }

    pub fn from_str(text: &str, theme: Theme) -> Result<Self, TokenError> {
        let file: TokenFile = serde_json::from_str(text)?;
        Self::from_file(&file, theme)
    }

    pub fn from_file(file: &TokenFile, theme: Theme) -> Result<Self, TokenError> {
        let mut colors = BTreeMap::new();
        let mut roles = BTreeMap::new();
        for (name, def) in &file.tokens {
            let (value, label) = match theme {
                Theme::Light => (&def.light, "light"),
                Theme::Dark => (&def.dark, "dark"),
            };
            let color = value.resolve(&file.ramps, name, label)?;
            colors.insert(name.clone(), color);
            roles.insert(name.clone(), def.role);
        }
        let mut emissive = BTreeMap::new();
        for (name, def) in &file.emissive {
            // The prefix is required, not conventional. A material's layer names one string
            // and two different maps can answer it; if the name did not say which, a reader of
            // `chrome/bar` could not tell that its far stop is light rather than a border
            // colour, and the rules that apply to it are entirely different.
            if !name.starts_with(EMISSIVE_PREFIX) {
                return Err(TokenError::EmissiveUnprefixed { name: name.clone() });
            }
            let (value, label) = match theme {
                Theme::Light => (&def.light, "light"),
                Theme::Dark => (&def.dark, "dark"),
            };
            emissive.insert(name.clone(), value.resolve(&file.ramps, name, label)?);
        }
        let materials = crate::material::resolve_all(
            &file.materials,
            &colors,
            &emissive,
            &file.space,
            &file.radius,
            &file.elevation,
        )?;
        Ok(Self {
            theme,
            colors,
            emissive,
            roles,
            type_roles: file.type_roles.clone(),
            space: file.space.clone(),
            radius: file.radius.clone(),
            elevation: file.elevation.clone(),
            focus: file.focus,
            materials,
            lighting: file.lighting.clone(),
            substance: file.substance,
            effects_enabled: true,
        })
    }

    /// What the lighting pass may do to a surface painted with `material`.
    ///
    /// **Derived, never declared** — the same reason [`Material::fidelity`] is derived: a material
    /// that could *declare* its own allowance could declare a wide one while carrying a label, and
    /// the gate would then hold it to a range it chose for itself. Whether a material carries text
    /// is already recorded, in the `text` list the contrast gate reads, so the class is a fact
    /// about the material rather than a second thing to keep in sync.
    ///
    /// Contract `lit-contrast.md` rule 1a: a meaning-bearing element may emit but must never
    /// receive, and the exclusion extends to the ground directly behind it.
    #[must_use]
    pub fn lit_bounds(&self, material: &Material) -> LitBounds {
        let class = if material.text.is_empty() {
            self.lighting.allowance.receiver
        } else {
            self.lighting.allowance.text_ground
        };
        class.for_theme(self.theme).bounds()
    }

    /// The authored allowances, for the lighting pass and for the instruments.
    #[must_use]
    pub fn lighting(&self) -> LightingTokens {
        self.lighting.clone()
    }

    /// The slab for one painted material, with its **allowance carried onto it**.
    ///
    /// The one route from a material to a scene slab that fills `attenuation_floor` from
    /// [`Tokens::lit_bounds`] — per material, per theme. [`crate::material::Material::slab`]
    /// alone leaves the floor at its safe default of 1.0 (no darkening permitted), so a
    /// call site that bypasses this helper gets a surface shadows cannot touch rather than
    /// one they can over-touch. Call it beside the `paint` that pushed the material's
    /// instances, with the same [`Surface`].
    #[must_use]
    pub fn scene_slab(
        &self,
        name: &str,
        surface: crate::material::Surface,
    ) -> Option<qs_gpu::scene::Slab> {
        let material = self.material(name)?;
        let mut slab = material.slab(surface)?;
        let bounds = self.lit_bounds(material);
        slab.attenuation_floor = bounds.attenuation.0;
        // The other half of the same allowance. A material that declares `text` gets zero
        // here by derivation — it may emit, and it may not receive.
        slab.addition_max = bounds.addition.1;
        Some(slab)
    }

    /// The rig's environment, resolved for this theme.
    ///
    /// The two stops are token *names* in the file, so the room a lit surface reflects
    /// changes with the theme rather than being one room both themes share. Resolution goes
    /// through [`Tokens::color`], which is the same lookup every other consumer uses.
    #[must_use]
    pub fn environment(&self) -> qs_gpu::scene::Environment {
        qs_gpu::scene::Environment {
            horizon: self.color(&self.lighting.rig.environment.horizon),
            zenith: self.color(&self.lighting.rig.environment.zenith),
        }
    }

    /// The tallest step on the elevation scale, in logical pixels. Zero when the file has no
    /// scale, which is also the honest margin: nothing can stand up, so nothing can cast
    /// past the viewport edge.
    #[must_use]
    pub fn elevation_max(&self) -> f32 {
        self.elevation.values().copied().fold(0.0_f32, f32::max)
    }

    /// Replace every colour with the OS-supplied forced-colours palette.
    ///
    /// The four inputs are what every platform's high-contrast API actually provides:
    /// window background, window text, highlight background, highlight text. Everything
    /// else is derived from those, because inventing a fifth colour would be exactly the
    /// substitution the mode exists to prevent.
    pub fn forced(
        window: Srgba,
        window_text: Srgba,
        highlight: Srgba,
        highlight_text: Srgba,
    ) -> Self {
        let mut colors = BTreeMap::new();
        let mut roles = BTreeMap::new();
        // Geometry survives forced colours; only the palette is replaced.
        let embedded = embedded_file().ok();

        let mut set = |name: &str, color: Srgba, role: TokenRole| {
            colors.insert(name.to_string(), color);
            roles.insert(name.to_string(), role);
        };

        set("surface/base", window, TokenRole::Background);
        // No banding, no hover tint: there is no fifth colour to make them out of, and
        // faking one by blending defeats the user's stated constraint.
        set("surface/row-alt", window, TokenRole::Background);
        set("surface/row-hover", window, TokenRole::Background);
        set("surface/row-selected", highlight, TokenRole::Background);
        set("surface/overlay", window, TokenRole::Background);
        // The popover's panel, and unlike `surface/overlay-lift` below this one IS DRAWN with
        // effects off: a backdrop layer collapses to its flat stop, and its flat stop is this.
        // So the value is not arbitrary -- it is the panel a forced-colours popover shows,
        // and `window` is the right answer twice over. It is a background, so the OS's
        // background colour is the colour the user chose for one; and the panel is then
        // separated from the list by its border rather than by a lightness the OS did not
        // supply, which is exactly the separation UXDD 10.5 leaves when it disables blur and
        // tint.
        //
        // Omitting it is not a missing panel. `resolve_all` here is followed by
        // `.ok().unwrap_or_default()`, so one unknown token removes EVERY material and
        // forced-colours mode renders a window with no rows in it -- which is what happened
        // when this material was added and this line was not, caught by the same row test the
        // field centres' comment above names. `forced-colours-resolve-is-all-or-nothing` is
        // the roadmap chunk for making that loud instead of silent.
        set("surface/raised", window, TokenRole::Background);

        // The stop that exists only as one end of a ramp. It is mapped onto the same OS
        // colour as the stop it ramps from, which is the honest answer here: with effects
        // switched off a material collapses to its flat stop and this is never drawn, but a
        // material that cannot *resolve* is an error, and forced-colours mode is not a mode
        // to discover a missing token in.
        set("surface/overlay-lift", window, TokenRole::Background);

        // The ambient field's four centres, onto the window colour, for exactly the reason
        // above and with a sharper edge to it. With effects off the field collapses to its
        // flat stop and no centre is ever drawn, so the value is arbitrary -- but the material
        // still has to *resolve*, and `resolve_all` here is followed by `.ok().unwrap_or_default()`,
        // so one unknown token does not make one material fail: it silently removes EVERY
        // material and forced-colours mode renders a window with no rows in it. That is what
        // adding these four tokens without adding them here actually did, and the only thing
        // that noticed was a row test asserting the selection had not vanished.
        // A shadow is an absence of light and forced colours has no colour to spend on one, so
        // it maps to the window: the layer that uses it is a glow, which is enhanced, so it is
        // dropped entirely with effects off and this value is never drawn. It is here so the
        // material RESOLVES -- see the four below, and the comment on `surface/overlay-lift`.
        set("shadow/contact", window, TokenRole::Background);

        set("field/one", window, TokenRole::Background);
        set("field/two", window, TokenRole::Background);
        set("field/three", window, TokenRole::Background);
        set("field/four", window, TokenRole::Background);

        set("content/primary", window_text, TokenRole::Foreground);
        set("content/secondary", window_text, TokenRole::Foreground);
        set("content/tertiary", window_text, TokenRole::Foreground);
        set("content/on-overlay", window_text, TokenRole::Foreground);

        set("border/subtle", window_text, TokenRole::Border);
        set("border/focus", window_text, TokenRole::Border);
        // The outline's whole job is to contrast with the ring. In forced-colours mode the
        // OS gives no fifth colour to make one from, so the window background is used --
        // which is the correct contrast against window text and is a colour the user chose.
        set("border/focus-outline", window, TokenRole::Border);

        // The badge inverts against the window, which is the one polarity forced colours can
        // express with the palette it gives: chip in the text colour, extension in the window
        // colour. That is the same trick `border/focus-outline` uses, and it keeps the badge a
        // badge rather than a rectangle the user cannot see.
        set("icon/badge", window_text, TokenRole::Background);
        set("icon/badge-text", window, TokenRole::Foreground);

        // A folder is a *silhouette* in forced-colours mode, not an amber one: the OS palette
        // has no fifth colour to spend on decoration, and inventing one would be exactly the
        // override forced colours exists to forbid. The icon still distinguishes folders,
        // because the shape always did the work and the tint only ever reinforced it.
        set("icon/folder", window_text, TokenRole::Border);

        // Status is carried by shape and by the accessibility tree in forced-colours mode.
        // Colour-coding it here would produce three indistinguishable rails.
        set("rail/modified", window_text, TokenRole::Border);
        set("rail/added", window_text, TokenRole::Border);
        set("rail/conflict", window_text, TokenRole::Border);

        // The selected row needs its own text colour, which is the one extra the OS gives.
        colors.insert("content/on-selected".to_string(), highlight_text);
        roles.insert("content/on-selected".to_string(), TokenRole::Foreground);

        // Forced-colours mode overrides colour, not geometry: the OS supplies a palette, not
        // a spacing scale, a type scale or a set of looks.
        let space = embedded
            .as_ref()
            .map(|f| f.space.clone())
            .unwrap_or_default();
        let radius = embedded
            .as_ref()
            .map(|f| f.radius.clone())
            .unwrap_or_default();
        // The materials survive with their layering and geometry intact and every colour
        // replaced. `effects_enabled` is false below, so what actually reaches the screen is
        // each stack's Exact layers at their flat stops -- which is the same answer the
        // hand-written `if effects_enabled` in chrome.rs used to give, now given once.
        // The elevation scale is geometry, like space and radius, and survives for the same
        // reason: forced colours replace the palette, not the shape of the world.
        let elevation = embedded
            .as_ref()
            .map(|f| f.elevation.clone())
            .unwrap_or_default();
        // Forced colours shows no light: `effects_enabled` is false below, so `bloom` returns
        // NONE whatever is in here. But the names still have to RESOLVE, and mapping them to
        // the OS foreground is the same substitution every other colour above gets rather
        // than a fifth colour invented for them -- an emissive is only ever authored on an
        // edge band, and a band the user's own text colour is exactly the separation forced
        // colours asks for.
        //
        // Leaving this empty would not be "no light". `resolve_all` here is followed by
        // `.ok().unwrap_or_default()`, so one unresolvable name removes EVERY material and
        // the window renders with no rows in it -- the failure the `surface/raised` comment
        // above records happening once already, and `forced-colours-resolve-is-all-or-nothing`
        // exists to make loud.
        let emissive: BTreeMap<String, Srgba> = embedded
            .as_ref()
            .map(|f| {
                f.emissive
                    .keys()
                    .map(|name| (name.clone(), window_text))
                    .collect()
            })
            .unwrap_or_default();
        let materials = embedded
            .as_ref()
            .and_then(|f| {
                crate::material::resolve_all(
                    &f.materials,
                    &colors,
                    &emissive,
                    &space,
                    &radius,
                    &elevation,
                )
                .ok()
            })
            .unwrap_or_default();

        Self {
            theme: Theme::Light,
            colors,
            emissive,
            roles,
            type_roles: embedded
                .as_ref()
                .map(|f| f.type_roles.clone())
                .unwrap_or_default(),
            space,
            radius,
            elevation,
            focus: embedded.map(|f| f.focus).unwrap_or_default(),
            materials,
            // Deliberately not carried over from the embedded file. Forced colours supply a
            // fixed palette; darkening or brightening it is exactly the substitution the mode
            // exists to refuse, and it would do so on the surfaces of a user who asked for
            // high contrast specifically.
            lighting: LightingTokens::default(),
            // Same refusal as the lighting above: the OS gave a fixed palette to a user
            // who asked for high contrast, and varying surfaces by age is a variation
            // they did not ask for and cannot switch off.
            substance: SubstanceTokens::default(),
            effects_enabled: false,
        }
    }

    pub fn theme(&self) -> Theme {
        self.theme
    }

    /// Whether shadows, translucency and animation tinting may be drawn.
    pub fn effects_enabled(&self) -> bool {
        self.effects_enabled
    }

    /// Look up a token.
    ///
    /// Returns transparent for an unknown name rather than panicking. A missing token is a
    /// bug, but it is a bug that should show as a missing element in a screenshot, not as a
    /// crashed frame thread -- and `Tokens::validate` catches it at startup anyway.
    pub fn color(&self, name: &str) -> Srgba {
        self.colors.get(name).copied().unwrap_or(Srgba::TRANSPARENT)
    }

    pub fn try_color(&self, name: &str) -> Option<Srgba> {
        self.colors.get(name).copied()
    }

    pub fn role(&self, name: &str) -> Option<TokenRole> {
        self.roles.get(name).copied()
    }

    /// A type role by name. Use the [`role`] constants rather than a literal.
    ///
    /// Falls back to `ui/md` for an unknown name, then to a hard 13/400 if even that is
    /// missing. A missing role must render *something* readable rather than nothing --
    /// text that fails to draw is a worse failure than text at the wrong size.
    pub fn type_role(&self, name: &str) -> TypeRole {
        self.type_roles
            .get(name)
            .or_else(|| self.type_roles.get(role::MD))
            .copied()
            .unwrap_or(TypeRole {
                size: 13.0,
                weight: 400,
            })
    }

    /// A step from the space scale, in **logical** pixels.
    ///
    /// An unknown step falls back to `md` and then to 6. A layout that silently collapses
    /// to zero spacing is harder to spot than one that is slightly wrong.
    pub fn space(&self, step: &str) -> f32 {
        self.space
            .get(step)
            .or_else(|| self.space.get(space::MD))
            .copied()
            .unwrap_or(6.0)
    }

    /// A corner radius by surface class, in **logical** pixels.
    pub fn radius(&self, class: &str) -> f32 {
        self.radius
            .get(class)
            .or_else(|| self.radius.get(radius::ROW))
            .copied()
            .unwrap_or(6.0)
    }

    pub fn focus(&self) -> FocusTokens {
        self.focus
    }

    /// A named look. Use the [`crate::material::name`] constants rather than a literal.
    pub fn material(&self, name: &str) -> Option<&Material> {
        self.materials.get(name)
    }

    pub fn material_names(&self) -> impl Iterator<Item = &str> {
        self.materials.keys().map(String::as_str)
    }

    /// The ambient field a named material declares, for [`qs_gpu::frame::DrawList::set_field`].
    ///
    /// `None` when the material has no field layer, and `None` is the honest answer rather
    /// than an empty wash: a draw list whose field was never set draws no field, and a call
    /// site that asked the wrong material should find that out here instead of getting a
    /// window painted with nothing in it.
    ///
    /// # Why this reads back off the material
    ///
    /// The centres cannot ride the instance — four of them are roughly 190 bytes against a
    /// 48-byte stride — so they have to reach the shader through the draw list, exactly as the
    /// environment does. The question was then where they are *authored*, and the answer that
    /// keeps one copy is: on the layer that draws them. The contrast gate reads that same
    /// layer through [`Material::composites`], so the numbers the gate checks and the numbers
    /// the shader draws cannot drift apart, which a separate scene-level block in the token
    /// file would have allowed on the first edit that touched one and not the other.
    ///
    /// The cost of that choice is that a field is a scene property authored per material, so
    /// two materials could each declare one and only the one a call site names would be drawn.
    /// That is a real question — a window with two fields — and it is left as one rather than
    /// answered by a rule nobody needs yet. The call site names its material, so which field
    /// is the scene's is explicit at the point it is decided.
    #[must_use]
    pub fn field(&self, material: &str) -> Option<qs_gpu::frame::FieldWash> {
        self.materials
            .get(material)?
            .layers
            .iter()
            .find(|layer| layer.kind == qs_gpu::frame::PrimKind::Field)
            .map(|layer| layer.field)
    }

    /// The frame's bloom, both numbers derived from the palette, for
    /// [`qs_gpu::frame::DrawList::set_bloom`].
    ///
    /// # The threshold clears every colour the palette can name
    ///
    /// Not a constant, and the criterion this chunk was accepted against says so: "a threshold
    /// stated in luminance against the token ramps rather than as a magic constant". The
    /// statement is that a pixel blooms when it is brighter than **every token in this theme**,
    /// whatever its role — because a pixel brighter than any colour a designer could have
    /// written down is not a fill. It is light the *renderer* made: a glow's falloff, a sweep's
    /// travelling highlight, a rim, the lit mode's addition to a receiver. Those are what "let
    /// bright accents bleed light" means, and they are exactly what a token-derived ceiling
    /// separates from flat colour.
    ///
    /// **The first version of this took the brightest `Background` token, and it was wrong in a
    /// way worth recording.** In the dark theme the brightest background is `icon/badge` at
    /// 0.618 relative luminance, while `content/primary` — body text — sits at 0.817. So that
    /// threshold bloomed *every glyph on screen*, which is not a look; it lifts the ground
    /// around each mark and reduces exactly the contrast Principle VI protects, invisibly to
    /// `cargo xtask contrast`, which reads tokens and not frames. Clearing every token instead
    /// makes that structurally impossible: no authored colour can bloom, so the gate's world is
    /// untouched and only rendered light is affected.
    ///
    /// Relative luminance is [`Srgba::relative_luminance`], the same WCAG function the contrast
    /// gate uses and the same three coefficients `blur.wgsl` applies per fragment. Measuring
    /// brightness two different ways on the two sides of this number would select a different
    /// set of pixels from the one it was chosen against.
    ///
    /// # The light theme blooms nothing, and that is the answer rather than a gap
    ///
    /// Several light-theme tokens resolve to white, so the threshold is 1.0, nothing in an
    /// 8-bit target exceeds it, and [`qs_gpu::frame::Bloom::is_active`] reads that as off. The
    /// light theme pays for no passes and shows no bloom.
    ///
    /// That is the same finding research R13 recorded for the lit palette, arriving again from
    /// a different direction: *the light theme has no emitter*. There the page is the source
    /// and dark marks are absorbers, so added light has nothing to come from. A bloom forced on
    /// anyway would have to bleed the page into the text, which is the one direction the
    /// contrast budget cannot afford.
    ///
    /// # The strength is the measured lit allowance
    ///
    /// A bloom *adds light to a receiver*, which is exactly the quantity
    /// `lighting.allowance.receiver.addition_max` bounds — measured by the `lit_probe` example
    /// and recorded in research R13, not chosen here. Reusing it rather than authoring a second
    /// number keeps one answer to "how much light may this palette add", so a future probe run
    /// that moves it moves the bloom too.
    /// How much light a bloom would add, in linear light: the receiver's measured allowance.
    ///
    /// Deliberately the **receiver's** and never the text ground's, which is 0.0 in both themes
    /// precisely because a mark's ground may not take light. Beside [`Tokens::bloom_ceiling`]
    /// rather than folded into [`Tokens::bloom`] for the same reason that one is: the number is
    /// derived, checkable and ready, and the decision about whether to *spend* it is separate
    /// from where it comes from.
    #[must_use]
    pub fn bloom_strength(&self) -> f32 {
        self.lighting
            .allowance
            .receiver
            .for_theme(self.theme)
            .addition_max
    }

    #[must_use]
    pub fn bloom_ceiling(&self) -> f32 {
        // Every token, every role. See below for why narrowing this to backgrounds was the
        // wrong answer and what it did.
        //
        // `self.colors`, and NOT `self.emissive`. That is the one line this whole mechanism
        // rests on: an emissive folded in here would raise the ceiling to exactly its own
        // luminance, and a value is not greater than itself, so the bloom would be off
        // forever with a source authored, present and drawn. See `EmissiveDef`.
        self.colors
            .values()
            .map(|color| color.relative_luminance())
            .fold(0.0_f32, f32::max)
    }

    /// A named emissive, if this palette has one. See [`EmissiveDef`].
    ///
    /// Separate from [`Tokens::color`] rather than folded into it, so that "a call site cannot
    /// paint a fill in light" is a fact about the API rather than a rule about call sites.
    #[must_use]
    pub fn emissive(&self, name: &str) -> Option<Srgba> {
        self.emissive.get(name).copied()
    }

    /// Every emissive this palette carries, brightest first. For gates and instruments.
    pub fn emissives(&self) -> impl Iterator<Item = (&str, Srgba)> {
        self.emissive.iter().map(|(name, c)| (name.as_str(), *c))
    }

    /// Whether this palette gives the bright pass anything to select: an emissive strictly
    /// above [`Tokens::bloom_ceiling`].
    ///
    /// The light theme answers `false` by **derivation and not by a special case**: several of
    /// its tokens resolve to white, so its ceiling is 1.0 and nothing in an 8-bit target can
    /// exceed it. The same emissive that is a source in the dark theme is not one here, and no
    /// code says so — the fold does.
    #[must_use]
    pub fn has_bloom_source(&self) -> bool {
        let ceiling = self.bloom_ceiling();
        self.emissive
            .values()
            .any(|light| light.relative_luminance() > ceiling)
    }

    /// The frame's bloom, for [`qs_gpu::frame::DrawList::set_bloom`].
    ///
    /// **Active on the shipped dark theme and off on the light one, and both answers are
    /// derived rather than chosen.** The mechanism is costed at +0.021 ms on Vulkan and
    /// +0.171 ms on GL (`cargo run --release -p qs-gpu --example bloom_cost`). For three
    /// chunks it had no source: nothing the product drew exceeded the palette's own ink
    /// ceiling. `emissive/sweep-peak` is that source — see [`EmissiveDef`] for why it is not
    /// a token, and `chrome/bar` in `design/tokens.json` for why the command bar's travelling
    /// hairline is the one band in the window that can carry it.
    ///
    /// # The two numbers
    ///
    /// The threshold is [`Tokens::bloom_ceiling`]: brighter than every colour this theme can
    /// name. Above that line a pixel is light the *renderer* made — a glow's falloff, a sweep's
    /// highlight, the lit mode's addition — and below it, it is a fill somebody authored.
    ///
    /// Nothing *composited from the palette* can reach that line, and that is structural rather
    /// than a gap waiting on content. Every primitive composites authored colours, so none can
    /// exceed the brightest authored colour; and the lit mode's addition is bounded to
    /// **receivers** by `lighting.allowance`, which are grounds sitting two orders of magnitude
    /// below the ink. Before the emissive existed, the shipped dark theme through `--shot-gpu`
    /// peaked at 0.8969 relative luminance against a ceiling of 0.8986 — short by one 8-bit
    /// step, because the ceiling is the ideal value of a token whose quantized form is what
    /// actually lands. That measurement is why the source had to come from *outside* the
    /// palette rather than from a brighter entry inside it.
    ///
    /// # Why the threshold is not simply lowered
    ///
    /// Because that blooms the ink, and blooming ink was measured: in `bloom_cost`'s
    /// `contrast_cost_of_blooming_a_mark`, a mark at 0.817 on a ground at 0.011 goes from
    /// **14.34:1 to 8.21:1 against its own ground — a 42.8% loss** — and `cargo xtask contrast`
    /// sees none of it, because it reads tokens and not frames. A pair with less headroom than
    /// body text would cross 4.5:1. This is `specs/002-ray-traced-mode/contracts/lit-contrast.md`
    /// rule 1a arriving from a third direction: a mark may emit, but the ground behind it may
    /// never receive, and bloom from a glyph lands on exactly that ground.
    ///
    /// **The first version of this took the brightest `Background` token, and it was wrong in a
    /// way worth recording.** In the dark theme the brightest background is `icon/badge` at
    /// 0.618 while body text sits at 0.817, so that threshold bloomed every glyph on screen —
    /// the 42.8% above, everywhere, under a green build.
    ///
    /// # What the threshold cannot see, and what does
    ///
    /// The bright pass selects by luminance; it has no idea which pixels are grounds behind
    /// text. So the threshold buys one guarantee only — no *authored* colour is a source — and
    /// says nothing about where the light it does select **lands**. A text ground's allowance
    /// is `lighting.allowance.text_ground.addition_max`, which is 0.0 in both themes, and this
    /// function cannot check it because a token file has no frame in it.
    ///
    /// `cargo run --release -p qs -- --bloom-reach` is what checks it: one frame rendered
    /// twice in one process, bloom present and bloom absent, reporting the worst lift on a
    /// ground behind text. That is why the spend site is a one-logical-pixel edge band with no
    /// text within reach of it rather than an argument about where a glow looks nice.
    ///
    /// # The other route, still open
    ///
    /// Sourcing the bright pass from the **surface half** of the frame only, using the
    /// `surface_content_split` seam the lighting pass already uses, so ink is not in the
    /// source at all and the threshold could drop below it. That is a second render target
    /// rather than a threshold change — the offscreen target holds the whole frame and a pass
    /// cannot sample the attachment it is writing — and it is what would let a *mark* bloom.
    #[must_use]
    pub fn bloom(&self) -> qs_gpu::frame::Bloom {
        // Forced colours switch effects off entirely, and a bloom is the loudest possible
        // violation of "use only the colours the user said they can see".
        if !self.effects_enabled {
            return qs_gpu::frame::Bloom::NONE;
        }
        // NONE, not `{ threshold: self.bloom_ceiling(), strength: ... }`. The distinction is a
        // frame's worth of work: an inactive bloom allocates no chain and encodes no passes,
        // where a bloom whose threshold nothing reaches would allocate 1.6 MB and run three
        // full-screen passes every frame to produce a black image and add it to nothing. That
        // is the light theme's answer, and it arrives from `has_bloom_source` rather than from
        // a branch on the theme.
        if !self.has_bloom_source() {
            return qs_gpu::frame::Bloom::NONE;
        }
        qs_gpu::frame::Bloom {
            threshold: self.bloom_ceiling(),
            strength: self.bloom_strength(),
        }
    }

    /// Paint a named material onto `surface`, appending to `out`.
    ///
    /// **The** call a component makes. It replaces a hand-built stack of `Instance::…`
    /// pushes, and the point is not that it is shorter: it is that after this there is no
    /// second place that decides what a selected row looks like.
    ///
    /// An unknown name draws nothing, for the reason [`Tokens::color`] returns transparent —
    /// a missing look should show as a missing element in a screenshot, not as a crashed
    /// frame thread. `every_named_material_exists_in_the_shipped_file` is what keeps that
    /// from being how a typo ships.
    pub fn paint(&self, name: &str, surface: Surface, alpha: f32, out: &mut Vec<Instance>) {
        self.paint_driven(name, surface, alpha, Drive::REST, out);
    }

    /// Paint a named material at a **drive** in `0..=1`, the way an animation asks for it.
    ///
    /// `alpha` is how *present* the material is — a state fading in — and the [`Drive`] is
    /// how *loud* it is and *where in a cycle* it is, which are two more questions again:
    /// the selection is fully present the whole time it is travelling, only its halo swells,
    /// and neither of those says where a drifting light has got to. The drive comes from
    /// [`crate::motion::InteractionMotion::drive`] and from nowhere else, which is what makes
    /// an animated material retire when its plan does and what keeps the phase off the frame
    /// loop's critical path. See [`crate::material::SwellDef`] and
    /// [`crate::material::PhaseDef`].
    pub fn paint_driven(
        &self,
        name: &str,
        surface: Surface,
        alpha: f32,
        drive: Drive,
        out: &mut Vec<Instance>,
    ) {
        self.paint_substance(name, surface, alpha, drive, None, out);
    }

    /// Paint a named material as a surface **made of something**.
    ///
    /// The third question, after how present a material is and how loud. A [`Substance`]
    /// says what the surface is: how rough, how thick, how much sky it picks up. It comes
    /// from [`Substance::of`] and from nowhere else, so a row's appearance is a function of
    /// its attributes and the token file rather than of anything a call site decided.
    ///
    /// `None` means this material carries no facts and its `Pbr` layers keep what they were
    /// authored as. That is a different thing from [`Substance::UNKNOWN`], which is a row
    /// whose attributes have not arrived.
    pub fn paint_substance(
        &self,
        name: &str,
        surface: Surface,
        alpha: f32,
        drive: Drive,
        substance: Option<Substance>,
        out: &mut Vec<Instance>,
    ) {
        if let Some(material) = self.materials.get(name) {
            material.compile_with(surface, alpha, drive, substance, self.effects_enabled, out);
        }
    }

    /// The response curves a [`Substance`] is derived through.
    #[must_use]
    pub fn substance(&self) -> SubstanceTokens {
        self.substance
    }

    pub fn try_type_role(&self, name: &str) -> Option<TypeRole> {
        self.type_roles.get(name).copied()
    }

    pub fn type_role_names(&self) -> impl Iterator<Item = &str> {
        self.type_roles.keys().map(String::as_str)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.colors.keys().map(String::as_str)
    }
}

/// One pair's result. Reported by the build-time gate.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastResult {
    pub foreground: String,
    pub background: String,
    pub kind: PairKind,
    pub theme: Theme,
    pub ratio: f32,
    pub required: f32,
}

impl ContrastResult {
    pub fn passes(&self) -> bool {
        self.ratio >= self.required
    }
}

/// Every check one theme's materials imply, as composites rather than as token pairs.
///
/// A declared pair asks "is this foreground legible on that token". A material asks the
/// question text actually faces: **is this foreground legible on whatever the layer stack
/// composited to**. Those are different questions whenever a look is more than one flat
/// fill, and the second is the one that catches a halo moved above the fill it belongs
/// under — a change that alters no token and would leave every declared pair green.
///
/// Each material contributes `text × over × composites`, and the composites are the
/// cartesian product of its layers' in-shape stops. That is where "checked against its
/// worst stop" comes from: every stop is checked, and the worst one is the one that fails.
///
/// The `PairKind` is derived from the foreground's own role, so a boundary listed on a
/// material is held to 3:1 and text to 4.5:1 without the material restating it.
/// How far lighting can move a surface, as a closed-form range.
///
/// The heart of `specs/002-ray-traced-mode/contracts/lit-contrast.md`. Once a surface is lit, the
/// colour text sits on is no longer the token it was authored from — it is
/// `surface x attenuation + addition`, varying across the surface with the geometry. A gate that
/// checks the authored colour reports green while a label sits at 3:1 in a shadow, which is exactly
/// the failure Principle VI's build-time validation exists to make impossible.
///
/// Both bounds are derived arithmetically from token values. Obtaining them by rendering and
/// measuring is forbidden: a sampled bound is a claim about the frames someone happened to render,
/// and the frame that violates it is the one nobody rendered.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LitBounds {
    /// Least and greatest multiplication the lighting pass can apply. Shadow and occlusion only
    /// ever remove light, so both are in `0..=1` and `max` is 1 for a fully lit surface.
    pub attenuation: (f32, f32),
    /// Least and greatest light the pass can add, in linear light. Bounce and direct light only
    /// ever add, so `min` is 0.
    pub addition: (f32, f32),
}

impl LitBounds {
    /// A surface no lighting touches. The identity: the range collapses to the authored colour, so
    /// an unlit surface is checked exactly as it is today.
    pub const UNLIT: Self = Self {
        attenuation: (1.0, 1.0),
        addition: (0.0, 0.0),
    };

    /// Provisional bounds for the first evaluation of the palette, before the lighting tokens
    /// exist.
    ///
    /// Deliberately pessimistic in both directions. The purpose of the first run is to find out
    /// what the palette costs, and optimistic provisional bounds would report a cost lower than
    /// the real one — which is the one number this exercise must not get wrong.
    pub const PROVISIONAL: Self = Self {
        // A surface in full shadow keeps only what the sky gives it.
        attenuation: (0.35, 1.0),
        // A surface beside a bright emitter, in linear light.
        addition: (0.0, 0.12),
    };

    /// Whether these bounds are closed.
    ///
    /// An unbounded surface cannot be gated, so it may not carry text — contract rule 4, and the
    /// clause with actual teeth: the remedy is to bound the emitters, not to widen the tolerance.
    #[must_use]
    pub fn are_closed(&self) -> bool {
        let finite = |v: f32| v.is_finite();
        finite(self.attenuation.0)
            && finite(self.attenuation.1)
            && finite(self.addition.0)
            && finite(self.addition.1)
            && self.attenuation.0 <= self.attenuation.1
            && self.addition.0 <= self.addition.1
    }

    /// The darkest and brightest this surface can become.
    ///
    /// Applied in **linear** light, because that is where attenuation and addition physically act,
    /// and then returned to sRGB so the existing contrast machinery reads it unchanged. Applying
    /// them to sRGB values directly would understate a shadow's depth by roughly the transfer
    /// curve — which is the same linear-versus-sRGB confusion that has already produced two
    /// authoring mistakes in this codebase.
    #[must_use]
    pub fn extremes(&self, surface: Srgba) -> (Srgba, Srgba) {
        let apply = |factor: f32, add: f32| -> Srgba {
            let ch = |c: f32| -> f32 {
                let linear = qs_gpu::color::srgb_to_linear(c) * factor + add;
                qs_gpu::color::linear_to_srgb(linear.clamp(0.0, 1.0))
            };
            Srgba {
                r: ch(surface.r),
                g: ch(surface.g),
                b: ch(surface.b),
                a: surface.a,
            }
        };
        (
            apply(self.attenuation.0, self.addition.0),
            apply(self.attenuation.1, self.addition.1),
        )
    }
}

/// One class of surface's lighting allowance, as the token file authors it.
///
/// Two numbers rather than four, because the other two are not free parameters: attenuation's
/// maximum is always 1 (shadow and occlusion only ever *remove* light) and addition's minimum is
/// always 0 (bounce only ever *adds*). Authoring them would let a token file state a range the
/// physics cannot produce, and the gate would then check an impossible frame.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct LitRange {
    /// The least multiplication the pass may apply. `1.0` is no shadow at all; `0.0` permits full
    /// black.
    pub attenuation_min: f32,
    /// The most light the pass may add, in **linear** light.
    pub addition_max: f32,
}

impl LitRange {
    /// The allowance that changes nothing.
    pub const UNLIT: Self = Self {
        attenuation_min: 1.0,
        addition_max: 0.0,
    };

    #[must_use]
    pub fn bounds(self) -> LitBounds {
        LitBounds {
            attenuation: (self.attenuation_min, 1.0),
            addition: (0.0, self.addition_max),
        }
    }
}

/// A [`LitRange`] per theme, which is what contract rule 3a requires.
///
/// A shadow darkens a ground: that *helps* light-on-dark text and *hurts* dark-on-light text, and
/// added light does the reverse. So the affordable direction flips with the theme, and one range
/// for both reports the intersection of two constraints that never bind at the same time. Research
/// R13 measured the cost of collapsing them at roughly an order of magnitude.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct ThemedLitRange {
    pub light: LitRange,
    pub dark: LitRange,
}

impl ThemedLitRange {
    pub const UNLIT: Self = Self {
        light: LitRange::UNLIT,
        dark: LitRange::UNLIT,
    };

    #[must_use]
    pub fn for_theme(self, theme: Theme) -> LitRange {
        match theme {
            Theme::Light => self.light,
            Theme::Dark => self.dark,
        }
    }
}

/// What the lighting pass is permitted to **do to a surface**, by class and by theme.
///
/// The two classes come from `specs/002-ray-traced-mode/contracts/lit-contrast.md` rule 1a: a
/// meaning-bearing element may emit but must never receive, and the exclusion extends to the
/// ground directly behind it. So a **text ground** takes almost nothing and a **receiver** takes
/// everything — which is also where the drama physically lands, since the gaps and the canvas are
/// what a grazing light actually reaches.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct LitAllowance {
    /// What a surface that carries a label may take.
    pub text_ground: ThemedLitRange,
    /// What a surface that carries none may take. Not gated for contrast — there is no text on it
    /// to fail — but bounded anyway, because contract rule 4 admits no unbounded surface.
    pub receiver: ThemedLitRange,
}

impl Default for LitAllowance {
    /// No lighting at all.
    ///
    /// The conservative default, for the reason [`FocusTokens::default`] is the opposite one: a
    /// token file that forgets its focus ring must still get a ring, because a missing ring is an
    /// accessibility failure. A token file that forgets its lighting allowance must get **none**,
    /// because an invented allowance is a licence to darken text nobody authorised.
    fn default() -> Self {
        Self {
            text_ground: ThemedLitRange::UNLIT,
            receiver: ThemedLitRange::UNLIT,
        }
    }
}

/// The token file's `lighting` block.
///
/// Two halves that answer different questions and must not be confused. The **allowance** is what
/// the pass may do to a surface: a contrast question, and gated. The **rig** — key light,
/// environment — is where the light is: an art-direction question, and not gated. They share one
/// key because they are one subject, and sit in separate fields because a value from one is never
/// a valid answer for the other.
///
/// The rig lands here under `tasks.md` T010. Every exposure control it gains must be a **mix
/// toward a bound** and never a multiplier — research R8, a mistake this codebase has made four
/// times in four places.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct LightingTokens {
    pub allowance: LitAllowance,
    /// Where the light is. See [`RigTokens`]. Not `Copy` any more, and deliberately: the
    /// environment's two stops are token *names*, resolved per theme at use, so the room
    /// changes with the theme instead of being one room both themes share.
    pub rig: RigTokens,
}

/// The rig: where the light is, as authored (tasks.md T010). An art-direction question, not
/// a contrast one — the allowance beside it is what is gated, and a value from one is never
/// a valid answer for the other.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct RigTokens {
    pub key: KeyLightTokens,
    /// The focus lamp. Its *position* is deliberately absent: it is wherever focus is, which
    /// is a layout fact and not something a designer authors. What is authored here is how
    /// the lamp behaves once it is there.
    pub focus: FocusLightTokens,
    pub environment: EnvironmentTokens,
}

impl Default for RigTokens {
    /// The authored rig, not "no rig": a token file predating the rig still lights the way
    /// the shipped file does, and the mode is off by default anyway (FR-002), so the default
    /// is never seen until someone turns the light on.
    fn default() -> Self {
        Self {
            key: KeyLightTokens::default(),
            focus: FocusLightTokens::default(),
            environment: EnvironmentTokens::default(),
        }
    }
}

/// The key light. One, directional, fixed — a window whose shadows point in several
/// directions reads as broken in a way nobody can name, which is why this is a token and not
/// a per-material choice.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct KeyLightTokens {
    /// Toward the light. Normalized at use, and doubling as the reduced-motion resting
    /// position (FR-029): under Reduce Motion the light sits exactly here.
    pub direction: [f32; 3],
    /// The key light's fraction of an exposure budget that sums to one; the environment gets
    /// the remainder. A **mix toward a bound, never a multiplier** (research R8) — there is
    /// no spelling of this field that pushes total exposure past the bound.
    pub share: f32,
    /// Angular size in degrees. What makes a shadow soften with distance; a light with no
    /// size casts the hard offset shadow this feature exists to replace.
    pub size_deg: f32,
}

impl Default for KeyLightTokens {
    fn default() -> Self {
        Self {
            // Exactly `qs_gpu::frame::LIGHT_DIR`, and `the_rig_and_the_shader_agree_about
            // _the_light` holds the two equal: the PBR bevels, the contact-shadow offsets
            // and the raymarched shadows must agree about where the light is, or the
            // window's shadows point two ways at once in a way nobody can name.
            direction: [-0.32, -0.55, 0.77],
            share: 0.72,
            size_deg: 5.0,
        }
    }
}

impl KeyLightTokens {
    /// The share, held to its budget. Clamped at read rather than trusted at parse, so a
    /// hand-edited file cannot spend more than the whole budget.
    #[must_use]
    pub fn share(self) -> f32 {
        self.share.clamp(0.0, 1.0)
    }
}

/// The focus lamp: a positional light that sits over whatever has keyboard focus.
///
/// # Why there is no direction, and no colour
///
/// A positional light has no single direction — that is why [`qs_gpu::scene::FocusLamp`] is a
/// separate type from `Light`, and a field for one here would be a value nothing reads.
/// Colour is absent for the reason the key light's is `[1, 1, 1]`: this lamp contributes
/// **attenuation only**, and attenuation is a colourless multiply. A tint authored here
/// would be dead metadata that looks like it does something.
///
/// # Why it cannot brighten anything past its unlit colour
///
/// The lamp's shadow term and the key light's are both in `0..=1`, and the pass mixes them
/// convexly, so the result is in `0..=1` too. A lamp can therefore *lift* a shadow the key
/// light cast — which is what makes focus visible — and can never push a surface past the
/// colour it has with no lighting at all. That is why this is art direction with no gate
/// obligation: the contrast gate's worst case is the allowance floor, and nothing here
/// moves it. See `contracts/lit-contrast.md` and research R8.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct FocusLightTokens {
    /// The lamp's fraction of the shading budget **directly beneath it**, falling off with
    /// distance. A **mix toward a bound, never a multiplier** (research R8): at `0.0` the
    /// pass is arithmetically what it is today, and at `1.0` the lamp owns the shading of
    /// the pixel under it outright. There is no spelling of this field that darkens a frame.
    pub share: f32,
    /// How far the room dims at the edge of the lamp's reach, `0..=1`. The half of the lamp
    /// a person actually sees.
    ///
    /// [`FocusLightTokens::share`] redistributes the shading between two lights, and on a
    /// list of rows at one elevation the two lights agree everywhere, so it changes almost
    /// nothing: measured at a peak of 7/255 and a mean of 3/255 across a shipped 1200x700
    /// window with focus moved eight rows. That is a feature that is arithmetically present
    /// and perceptually absent. This is what makes focus *lit*: brightest under the lamp,
    /// dimming with distance, so focus is found by where the light is.
    ///
    /// Bounded by the same allowance clamp the key light's shadow is, so the light theme's
    /// 0.87 text-ground floor holds it to a shallower gradient than the dark theme's 0.55 —
    /// the theme asymmetry research R13 measured, handled by the clamp that already exists
    /// rather than by a second rule.
    pub ambient: f32,
    /// Angular size in degrees, exactly as [`KeyLightTokens::size_deg`]. Large on purpose:
    /// a focus lamp hanging a few pixels above a row is physically a broad source, and a
    /// small one would cast a hard second shadow that reads as a rendering fault rather
    /// than as light.
    pub size_deg: f32,
    /// How far above the focused surface's top face the lamp hangs, in **logical** pixels
    /// (scaled where it is built, like every other authored length).
    ///
    /// This is the closest thing the lamp has to a resting position, and FR-029 is satisfied
    /// by a different mechanism: under Reduce Motion the lamp does not travel between rows,
    /// it is simply *at* the focused row — an authored place, not wherever an interrupted
    /// animation stopped. See [`crate::motion::MotionPattern::FocusLight`].
    pub height: f32,
}

impl Default for FocusLightTokens {
    fn default() -> Self {
        Self {
            // Enough that the room around focus visibly changes, well short of the lamp
            // taking over the shading from the key light and flattening the scene.
            share: 0.45,
            // Deep enough to read as light rather than as a rendering artefact, shallow
            // enough that the rows past the lamp's reach are still comfortably readable —
            // they sit at 0.82 of their colour in the dark theme, well above the 0.55 the
            // allowance would permit. A vignette that followed the keyboard at full
            // allowance would be the mode making the product worse, which SC-012 exists to
            // catch and which no amount of "it looks impressive" should be allowed to buy.
            ambient: 0.18,
            // Roughly four times the key light's, which is what keeps the lamp's shadow a
            // soft lift rather than a second hard edge competing with the key's.
            size_deg: 22.0,
            // Just above a compact row's own height, so the lamp clears the surface it sits
            // over instead of being embedded in it.
            height: 14.0,
        }
    }
}

impl FocusLightTokens {
    /// The share, held to its budget. Clamped at read rather than trusted at parse, for the
    /// same reason [`KeyLightTokens::share`] is: a hand-edited file cannot spend more than
    /// the whole budget, and a negative one cannot invert the mix.
    #[must_use]
    pub fn share(self) -> f32 {
        self.share.clamp(0.0, 1.0)
    }

    /// The ambient depth, held to its range. Clamped for [`FocusLightTokens::share`]'s
    /// reason, and with one more of its own: a negative value would *brighten* a surface past
    /// its unlit colour, which is the one direction the allowance clamp does not catch.
    #[must_use]
    pub fn ambient(self) -> f32 {
        self.ambient.clamp(0.0, 1.0)
    }
}

/// The environment's two stops, as token names. Resolved per theme at use.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct EnvironmentTokens {
    pub horizon: String,
    pub zenith: String,
}

impl Default for EnvironmentTokens {
    fn default() -> Self {
        Self {
            horizon: "surface/base".to_string(),
            zenith: "surface/overlay-lift".to_string(),
        }
    }
}

/// Which bounds the gate checks each material against.
///
/// Two policies rather than a single `LitBounds` parameter, because the two callers want opposite
/// things. The gate wants each material held to *its own* allowance, which is the shipped rule. The
/// probe wants every material held to *one* range so it can bisect for the range — and a bisection
/// over per-material bounds would be bisecting a number it had already fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LitPolicy {
    /// Every material checked at one range. What `lit_probe` bisects with.
    Uniform(LitBounds),
    /// Each material at the allowance its class and theme give it. What ships.
    PerMaterial,
}

pub fn material_results(file: &TokenFile, theme: Theme) -> Result<Vec<ContrastResult>, TokenError> {
    material_results_policy(file, theme, LitPolicy::Uniform(LitBounds::UNLIT))
}

/// [`material_results`], with every material held to one stated range.
///
/// Retained as its own entry point because bisecting a range is a different question from gating
/// against one, and the instrument that asks the first must not accidentally ask the second.
pub fn material_results_lit(
    file: &TokenFile,
    theme: Theme,
    lit: LitBounds,
) -> Result<Vec<ContrastResult>, TokenError> {
    material_results_policy(file, theme, LitPolicy::Uniform(lit))
}

/// [`material_results`], plus the two extremes lighting can move each surface to.
///
/// Separate entry point rather than a changed signature so every existing caller keeps the
/// behaviour it has, and so the lit check is something a caller opts into with a stated policy
/// rather than something that appears by default with numbers nobody chose.
pub fn material_results_policy(
    file: &TokenFile,
    theme: Theme,
    policy: LitPolicy,
) -> Result<Vec<ContrastResult>, TokenError> {
    let tokens = Tokens::from_file(file, theme)?;
    let mut results = Vec::new();

    for (name, material) in &tokens.materials {
        if material.text.is_empty() {
            continue;
        }
        // Only reached for a material that carries text, so the class is settled: it is a text
        // ground. `Tokens::lit_bounds` is still what answers, rather than an inline branch, so the
        // gate and the renderer cannot disagree about which allowance a material gets.
        let lit = match policy {
            LitPolicy::Uniform(bounds) => bounds,
            LitPolicy::PerMaterial => tokens.lit_bounds(material),
        };
        if material.over.is_empty() {
            return Err(TokenError::MaterialMissingBase {
                material: name.clone(),
            });
        }
        for base_name in &material.over {
            let base = tokens
                .try_color(base_name)
                .ok_or_else(|| TokenError::UnknownToken(base_name.clone()))?;

            // The LIT extreme, when this material emits: the same composites with the
            // emission's closed-form peak added, checked against the ink the material says
            // it carries when lit. This is the half the gate was structurally blind to —
            // `Material::composites` walks albedo, and a lamp moves the ground under its
            // own label without moving a single albedo. Checking it here is what lets a
            // surface be bright and legible instead of one or the other.
            let emitting = material.emission_peak().iter().any(|v| *v > 0.0);
            if emitting {
                let lit_inks: &[String] = if material.text_lit.is_empty() {
                    &material.text
                } else {
                    &material.text_lit
                };
                for composite in material.lit_composites(base) {
                    for foreground_name in lit_inks {
                        let foreground = tokens
                            .try_color(foreground_name)
                            .ok_or_else(|| TokenError::UnknownToken(foreground_name.clone()))?;
                        let kind = match tokens.role(foreground_name) {
                            Some(TokenRole::Border) => PairKind::Boundary,
                            _ => PairKind::Text,
                        };
                        results.push(ContrastResult {
                            foreground: foreground_name.clone(),
                            background: format!("{name} over {base_name} (lit)"),
                            kind,
                            theme,
                            ratio: foreground.over(composite).contrast_ratio(composite),
                            required: kind.minimum_ratio(),
                        });
                    }
                }
            }

            for composite in material.composites(base) {
                for foreground_name in &material.text {
                    let foreground = tokens
                        .try_color(foreground_name)
                        .ok_or_else(|| TokenError::UnknownToken(foreground_name.clone()))?;
                    let kind = match tokens.role(foreground_name) {
                        Some(TokenRole::Border) => PairKind::Boundary,
                        _ => PairKind::Text,
                    };
                    results.push(ContrastResult {
                        foreground: foreground_name.clone(),
                        // Named so a failure is attributable to a stack and a ground rather
                        // than to an anonymous colour nobody can find in the file.
                        background: format!("{name} over {base_name}"),
                        kind,
                        theme,
                        ratio: foreground.over(composite).contrast_ratio(composite),
                        required: kind.minimum_ratio(),
                    });

                    // And again at both ends of what lighting can do to this surface.
                    //
                    // The unlit check above is not replaced -- contract rule 6: a label legible
                    // only when lit is illegible on the tier that cannot light it. Both apply.
                    //
                    // The two extremes fail in opposite directions and need opposite remedies,
                    // so they are reported as separate results rather than as a single worst
                    // case. A report that said only "failed" would send an author to guess
                    // between raising a floor and reducing a shadow, and the wrong guess makes
                    // it worse.
                    //
                    // The label carries the bound it was computed from, because under
                    // contract rule 3a `0.863` is a *light-theme* number and applying it to
                    // the dark theme is one of the two wrong guesses the report exists to
                    // prevent. A bare "(shadowed)" no longer identifies the check.
                    if lit != LitBounds::UNLIT {
                        let (dark, bright) = lit.extremes(composite);
                        let shadowed = format!("shadowed to {:.3}", lit.attenuation.0);
                        let brightened = format!("lit +{:.4}", lit.addition.1);
                        for (end, surface) in [(&shadowed, dark), (&brightened, bright)] {
                            results.push(ContrastResult {
                                foreground: foreground_name.clone(),
                                background: format!("{name} over {base_name} ({end})"),
                                kind,
                                theme,
                                ratio: foreground.over(surface).contrast_ratio(surface),
                                required: kind.minimum_ratio(),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(results)
}

/// Check every declared pair and every material composite, in both themes.
///
/// This is the function `cargo xtask contrast` calls, and it is in the library rather than
/// in `xtask` so that a unit test can assert the shipped token file passes without shelling
/// out to a build tool.
pub fn check_contrast(file: &TokenFile) -> Result<Vec<ContrastResult>, TokenError> {
    let mut results = Vec::new();

    for theme in [Theme::Light, Theme::Dark] {
        let tokens = Tokens::from_file(file, theme)?;
        for pair in &file.contrast_pairs {
            let foreground = tokens
                .try_color(&pair.foreground)
                .ok_or_else(|| TokenError::UnknownToken(pair.foreground.clone()))?;
            let background = tokens
                .try_color(&pair.background)
                .ok_or_else(|| TokenError::UnknownToken(pair.background.clone()))?;

            // Composite first. A translucent foreground has no single contrast ratio -- it
            // depends what is behind it -- so the ratio is computed against the colour the
            // user actually sees.
            let composited = foreground.over(background);

            results.push(ContrastResult {
                foreground: pair.foreground.clone(),
                background: pair.background.clone(),
                kind: pair.kind,
                theme,
                ratio: composited.contrast_ratio(background),
                required: pair.kind.minimum_ratio(),
            });
        }
        // Each material at its own allowance, not at one global range. Under
        // `LightingTokens::default` — no lighting authored — this is bit-for-bit the unlit check
        // the gate has always made, so a token file that has not opted in sees no change.
        results.extend(material_results_policy(
            file,
            theme,
            LitPolicy::PerMaterial,
        )?);
    }
    Ok(results)
}

/// Parse the token file that ships with the binary.
pub fn embedded_file() -> Result<TokenFile, TokenError> {
    Ok(serde_json::from_str(include_str!(
        "../../../design/tokens.json"
    ))?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn the_shipped_token_file_parses_in_both_themes() {
        Tokens::embedded(Theme::Light).expect("light theme must resolve");
        Tokens::embedded(Theme::Dark).expect("dark theme must resolve");
    }

    #[test]
    fn no_authored_colour_can_bloom_in_either_theme() {
        // The acceptance criterion in the words it was written in -- "a threshold stated in
        // luminance against the token ramps rather than as a magic constant" -- and the
        // property that makes it safe rather than merely derived. Every token in the file,
        // whatever its role, must sit at or below the threshold. Nothing a designer can write
        // down blooms; only light the renderer made does.
        //
        // This is what keeps the effect outside `cargo xtask contrast`'s blind spot. The gate
        // reads tokens, not frames, so it cannot see a bloom lifting a ground -- and it does
        // not have to, because no token is above the line.
        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::embedded(theme).unwrap();
            let ceiling = tokens.bloom_ceiling();
            assert!(
                (0.0..=1.0).contains(&ceiling),
                "{theme:?}: a relative luminance is in 0..=1"
            );
            for name in tokens.colors.keys() {
                let l = tokens.color(name).relative_luminance();
                assert!(
                    l <= ceiling + 1e-6,
                    "{theme:?}: {name} at {l} is above the bloom ceiling {ceiling}; an authored \
                     colour that blooms lifts the ground around every mark drawn in it, which \
                     the contrast gate cannot see"
                );
            }
            // And the ceiling is *reached* rather than parked above everything: it is the
            // maximum, so at least one token is exactly on it. A 1.0 chosen to be safe would
            // pass the loop above and select nothing, ever, for a different reason.
            assert!(
                tokens
                    .colors
                    .values()
                    .any(|c| (c.relative_luminance() - ceiling).abs() < 1e-6),
                "{theme:?}: the ceiling must be a token's luminance, not a round number"
            );
        }
    }

    #[test]
    fn the_dark_theme_blooms_from_one_emissive_and_the_light_theme_still_pays_nothing() {
        // THIS TEST USED TO ASSERT THE OPPOSITE, and the history is the reason it was updated
        // rather than replaced. For three chunks it read
        // `the_shipped_palette_blooms_nothing_and_pays_nothing_for_it`: the bloom was built,
        // exercised and costed (+0.021 ms Vulkan, +0.171 ms GL, 84/255 change 20,808 px from a
        // bright element) and switched OFF, because nothing the product drew reached the
        // ceiling that spares text. `--shot-gpu` on the dark theme peaked at 0.8969 against
        // 0.8986 -- short by one 8-bit step. That was a measurement, not an oversight, and it
        // is why the source could not be a brighter token: `bloom_ceiling` folds over the token
        // map, so a bright token raises the line it was meant to clear.
        //
        // What changed is `design/tokens.json`'s `emissive` block, which is outside `tokens`
        // for exactly that reason. Both halves below are DERIVED -- one value, 0.9414, above
        // the dark ceiling and below the light one -- so neither theme is special-cased.
        let dark = Tokens::embedded(Theme::Dark).unwrap();
        assert!(
            dark.has_bloom_source(),
            "the dark theme must have something above its ink ceiling to bleed, or the bloom \
             is three passes producing a black image"
        );
        let bloom = dark.bloom();
        assert!(bloom.is_active(), "dark: {bloom:?}");
        assert!(
            (bloom.threshold - dark.bloom_ceiling()).abs() < 1e-6,
            "the threshold is the ink ceiling and nothing else: below it, a bloom lifts the \
             ground behind a mark by the 42.8% `bloom_cost` measured"
        );
        assert!(
            (bloom.strength - dark.bloom_strength()).abs() < 1e-6,
            "the strength is the measured receiver allowance, not a number chosen here"
        );

        // The light theme pays for no chain at all, and NOT because anything says so: several
        // of its tokens resolve to white, its ceiling is 1.0, and nothing in an 8-bit target
        // exceeds that. Deleting the emissive block would make this half pass for the wrong
        // reason, which is what the dark assertions above are guarding.
        let light = Tokens::embedded(Theme::Light).unwrap();
        assert!(
            !light.has_bloom_source(),
            "the light theme has no emitter: its page is already the brightest thing in it"
        );
        assert!(
            !light.bloom().is_active(),
            "an inactive bloom allocates no chain and encodes no passes"
        );
    }

    #[test]
    fn an_emissive_is_light_and_not_a_colour_the_palette_can_name() {
        // The three structural properties, asserted rather than trusted to the comments that
        // explain them. Any one of them failing puts the emissive back inside the palette in
        // effect, whatever the file looks like.
        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::embedded(theme).unwrap();
            assert!(
                tokens.emissives().next().is_some(),
                "{theme:?}: no emissive at all, so the two assertions below check nothing"
            );
            for (name, light) in tokens.emissives() {
                // (1) It is not reachable as a colour. A call site that names it gets
                // transparent, which is a missing element in a screenshot rather than a glyph
                // drawn in light.
                assert!(
                    tokens.try_color(name).is_none(),
                    "{theme:?}: {name} is reachable through Tokens::color, so a fill or a glyph \
                     can be painted in light"
                );
                // (2) It does not raise the ceiling it has to clear. This is the one that
                // makes the whole mechanism possible; folding the two maps together in
                // `bloom_ceiling` would leave the bloom off forever with a source drawn.
                let _ = light;
                assert!(
                    tokens
                        .colors
                        .values()
                        .any(|c| (c.relative_luminance() - tokens.bloom_ceiling()).abs() < 1e-6),
                    "{theme:?}: the ceiling is no longer any token's luminance, so an emissive \
                     has leaked into the fold"
                );
            }
        }
        // (3) And the dark theme's is genuinely above the line rather than equal to it, with
        // enough headroom that an interpolated sweep has an arc above the threshold rather
        // than one pixel. The margin is authored in tokens.json; this is what makes it a
        // number somebody has to keep rather than a comment.
        let dark = Tokens::embedded(Theme::Dark).unwrap();
        let peak = dark
            .emissive("emissive/sweep-peak")
            .expect("the shipped emissive")
            .relative_luminance();
        let margin = peak - dark.bloom_ceiling();
        assert!(
            margin > 0.02,
            "emissive/sweep-peak clears the ceiling by only {margin}; a sweep interpolates to \
             this stop, so a thin margin means a source one pixel wide that quantizes away"
        );
    }

    #[test]
    fn body_text_would_have_bloomed_under_the_first_threshold_this_chunk_tried() {
        // The refutation, kept. Taking the brightest *Background* token reads well and is
        // wrong: in the dark theme the brightest background is `icon/badge`, a mark rather
        // than a ground, and body text is brighter than it. That threshold bloomed every glyph
        // on screen.
        //
        // Asserting the inequality rather than describing it means the trap stays visible if
        // somebody narrows the fold again -- and it goes red the moment the palette changes in
        // a way that would have made the rejected version look fine, which is exactly when a
        // reader would be tempted to try it.
        let dark = Tokens::embedded(Theme::Dark).unwrap();
        let brightest_ground = dark
            .colors
            .iter()
            .filter(|(name, _)| dark.role(name) == Some(TokenRole::Background))
            .map(|(_, c)| c.relative_luminance())
            .fold(0.0_f32, f32::max);
        let text = dark.color("content/primary").relative_luminance();
        assert!(
            text > brightest_ground,
            "dark theme body text at {text} must be brighter than the brightest background \
             token at {brightest_ground}; if this stops being true, re-read why the threshold \
             is taken over every role"
        );
        assert!(
            text <= dark.bloom().threshold,
            "and the shipped threshold must still spare it"
        );
    }

    #[test]
    fn the_light_theme_has_no_headroom_for_a_bloom_at_all() {
        // Not a gap. `surface/base` in the light theme is white, so no pixel in an 8-bit target
        // is brighter than every ground and `is_active` reads the threshold as off. This is
        // research R13's finding arriving from a second direction -- the light theme has no
        // emitter, because there the page is the source and marks are absorbers.
        //
        // Worth a test rather than a comment because "the effect does nothing in one theme"
        // is indistinguishable from "the effect is broken" without one, and the next person to
        // look at a light-theme screenshot will have exactly that question.
        let light = Tokens::embedded(Theme::Light).unwrap();
        assert!(
            (light.color("surface/base").relative_luminance() - 1.0).abs() < 1e-6,
            "the light theme's list background is white; if that changes, so does this"
        );
        assert!(
            (light.bloom_ceiling() - 1.0).abs() < 1e-6,
            "so the ceiling is 1.0 and there is no room above it for anything to be"
        );

        // The dark theme has headroom -- which is why the shipped answer there turns on
        // whether anything occupies it, and not on whether the palette permits it at all. The
        // two themes are off for different reasons, and collapsing them would lose one.
        let dark = Tokens::embedded(Theme::Dark).unwrap();
        assert!(
            dark.bloom_ceiling() < 0.95,
            "the dark theme's brightest token is short of white; got {}",
            dark.bloom_ceiling()
        );
    }

    #[test]
    fn the_bloom_strength_is_the_measured_receiver_allowance_and_not_a_second_number() {
        // Bloom adds light to a receiver, which is the quantity `receiver.addition_max`
        // bounds -- measured by the `lit_probe` example, recorded in research R13. Asserting
        // the identity rather than the value is what keeps a future probe run from moving one
        // and not the other.
        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::embedded(theme).unwrap();
            let allowance = tokens
                .lighting()
                .allowance
                .receiver
                .for_theme(theme)
                .addition_max;
            assert_eq!(tokens.bloom_strength(), allowance, "{theme:?}");
            // Never the text ground's allowance, which is 0.0 in both themes precisely
            // because a mark's ground may not take light. Reading that one here would
            // silently disable the effect and look like a tuning choice.
            assert_eq!(
                tokens
                    .lighting()
                    .allowance
                    .text_ground
                    .for_theme(theme)
                    .addition_max,
                0.0,
                "{theme:?}: if this stops being zero, re-read lit-contrast rule 1a before \
                 assuming the bloom may use it"
            );
        }
    }

    #[test]
    fn forced_colours_bloom_nothing() {
        // A bloom is the loudest possible violation of "use only the colours the user said
        // they can see" -- it adds light the OS palette never offered, over the whole window.
        let forced = Tokens::forced(
            Srgba::new(0.0, 0.0, 0.0, 1.0),
            Srgba::new(1.0, 1.0, 1.0, 1.0),
            Srgba::new(1.0, 1.0, 0.0, 1.0),
            Srgba::new(0.0, 0.0, 0.0, 1.0),
        );
        assert!(!forced.effects_enabled());
        assert!(!forced.bloom().is_active());
    }

    #[test]
    fn the_five_type_roles_match_the_uxdd_table() {
        // UXDD 10.1. Hard-coded here on purpose: this test is the check that the token file
        // still says what the design document says, so reading the expectation from the
        // token file would make it vacuous.
        let tokens = Tokens::embedded(Theme::Light).unwrap();
        for (name, size, weight) in [
            (role::XS, 11.0, 500),
            (role::SM, 12.0, 400),
            (role::MD, 13.0, 400),
            (role::LG, 15.0, 600),
            (role::XL, 20.0, 600),
        ] {
            let found = tokens
                .try_type_role(name)
                .unwrap_or_else(|| panic!("missing type role {name}"));
            assert_eq!(found.size, size, "{name} size");
            assert_eq!(found.weight, weight, "{name} weight");
        }
    }

    #[test]
    fn the_row_roles_are_distinct_sizes_on_a_shared_weight() {
        // The row's hierarchy is size plus colour, not weight -- both ui/sm and ui/md are
        // 400. If someone "fixes" that by bolding filenames, this fails and they have to
        // argue with the UXDD rather than with a reviewer.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let name = tokens.type_role(role::MD);
        let meta = tokens.type_role(role::SM);
        assert!(
            name.size > meta.size,
            "metadata must be smaller than the name"
        );
        assert_eq!(
            name.weight, meta.weight,
            "the row does not use weight for hierarchy"
        );
    }

    #[test]
    fn a_type_role_scales_with_device_and_text_scale() {
        let role = TypeRole {
            size: 13.0,
            weight: 400,
        };
        assert!((role.size_px(1.0, 1.0) - 13.0).abs() < 1e-6);
        assert!((role.size_px(2.0, 1.0) - 26.0).abs() < 1e-6);
        assert!((role.size_px(1.0, 2.0) - 26.0).abs() < 1e-6);
        // A broken scale must not produce a zero or NaN font size.
        assert!(role.size_px(0.0, 1.0) > 0.0);
        assert!(role.size_px(f32::NAN, f32::NAN) > 0.0);
    }

    #[test]
    fn an_unknown_role_falls_back_to_body_rather_than_vanishing() {
        // Text that fails to draw is a worse failure than text at the wrong size.
        let tokens = Tokens::embedded(Theme::Light).unwrap();
        assert_eq!(
            tokens.type_role("ui/nonexistent"),
            tokens.type_role(role::MD)
        );
        assert_eq!(tokens.try_type_role("ui/nonexistent"), None);
    }

    #[test]
    fn forced_colours_keeps_the_type_scale() {
        // The OS supplies a palette, not a type scale.
        let c = Srgba::new(0.0, 0.0, 0.0, 1.0);
        let w = Srgba::new(1.0, 1.0, 1.0, 1.0);
        let tokens = Tokens::forced(c, w, c, w);
        assert_eq!(tokens.type_role(role::MD).size, 13.0);
        assert_eq!(tokens.type_role(role::LG).weight, 600);
    }

    #[test]
    fn the_shipped_token_file_passes_its_own_contrast_gate() {
        // The same assertion `cargo xtask contrast` makes, run as a unit test so a token
        // edit fails `cargo test` too rather than only the build tool nobody runs locally.
        let file = embedded_file().unwrap();
        let results = check_contrast(&file).unwrap();
        assert!(!results.is_empty(), "the pair list must not be empty");

        let failures: Vec<_> = results.iter().filter(|r| !r.passes()).collect();
        assert!(
            failures.is_empty(),
            "contrast failures:\n{}",
            failures
                .iter()
                .map(|r| format!(
                    "  {:?} {} on {} = {:.2}:1 (needs {:.1}:1)",
                    r.theme, r.foreground, r.background, r.ratio, r.required
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn a_material_that_carries_text_gets_the_text_grounds_allowance() {
        // The derivation, which is the whole of contract rule 1a in code. A material does not
        // declare its class; the class follows from whether it declares `text`, so a material
        // cannot award itself the receivers' allowance while carrying a label.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let labelled = tokens
            .material(crate::material::name::ROW_SELECTED)
            .unwrap();
        assert!(!labelled.text.is_empty(), "this material carries text");
        assert_eq!(
            tokens.lit_bounds(labelled),
            tokens.lighting().allowance.text_ground.dark.bounds()
        );

        let mut bare = labelled.clone();
        bare.text.clear();
        assert_eq!(
            tokens.lit_bounds(&bare),
            tokens.lighting().allowance.receiver.dark.bounds(),
            "the same layers with no label on them may take the whole drama"
        );
    }

    #[test]
    fn the_two_themes_do_not_get_the_same_allowance() {
        // R13. Asserting the *shape* of the asymmetry rather than its numbers, which
        // `xtask::contrast` pins: whatever the values become, the dark theme must be able to
        // shadow a text ground harder than the light theme can. Collapsing them back into one
        // range is the specific regression this catches, and it would look like a tidy-up.
        let light = Tokens::embedded(Theme::Light).unwrap();
        let dark = Tokens::embedded(Theme::Dark).unwrap();

        let floor = |t: &Tokens| {
            let m = t.material(crate::material::name::ROW_SELECTED).unwrap();
            t.lit_bounds(m).attenuation.0
        };
        assert!(
            floor(&dark) < floor(&light),
            "a shadow helps light-on-dark text and hurts dark-on-light text, so the dark theme \
             affords a deeper one: dark {} vs light {}",
            floor(&dark),
            floor(&light)
        );
    }

    #[test]
    fn a_text_ground_receives_no_added_light_in_either_theme() {
        // Rule 1a's teeth. Emission is allowed; reception is not, and the exclusion covers the
        // ground directly behind the mark -- which is the term that actually bounds contrast.
        // A non-zero addition here would be the rule abandoned while the contract still claimed
        // it, which is the failure mode a contract has that code does not.
        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::embedded(theme).unwrap();
            for (name, material) in &tokens.materials {
                if material.text.is_empty() {
                    continue;
                }
                assert_eq!(
                    tokens.lit_bounds(material).addition,
                    (0.0, 0.0),
                    "`{name}` carries text in the {theme:?} theme and must not receive"
                );
            }
        }
    }

    #[test]
    fn a_token_file_with_no_lighting_section_is_gated_exactly_as_it_is_today() {
        // The default has to be *no lighting*, not plausible lighting. A file that never opted
        // in must see bit-for-bit the check it saw before the lit mode existed, or every token
        // file in the wild silently acquires a licence to darken its own text.
        let mut file = embedded_file().unwrap();
        file.lighting = LightingTokens::default();

        for theme in [Theme::Light, Theme::Dark] {
            assert_eq!(
                material_results_policy(&file, theme, LitPolicy::PerMaterial).unwrap(),
                material_results(&file, theme).unwrap(),
                "an unlit file must produce the unlit checks and no others"
            );
        }
    }

    #[test]
    fn forced_colours_never_light_anything() {
        // The OS supplied this palette to a user who asked for high contrast. Attenuating or
        // brightening it is exactly the substitution the mode exists to refuse, and it would
        // land on the surfaces of the person least able to absorb it.
        let black = Srgba::new(0.0, 0.0, 0.0, 1.0);
        let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
        let forced = Tokens::forced(white, black, black, white);
        assert_eq!(forced.lighting(), LightingTokens::default());
        for material in forced.materials.values() {
            assert_eq!(forced.lit_bounds(material), LitBounds::UNLIT);
        }
    }

    #[test]
    fn every_token_in_the_shipped_file_is_ramp_relative() {
        // Criterion 1, in the only form that can fail. "Generated from OKLCH anchors" is a
        // claim about the file, not about the code, and the way it decays is one token
        // added back as a hex literal because somebody had a value from a design tool.
        let file = embedded_file().unwrap();
        let literals: Vec<_> = file
            .tokens
            .iter()
            .flat_map(|(name, def)| {
                [("light", &def.light), ("dark", &def.dark)]
                    .into_iter()
                    .filter(|(_, value)| value.ramp_name().is_none())
                    .map(move |(theme, _)| format!("{name} ({theme})"))
            })
            .collect();
        assert!(
            literals.is_empty(),
            "these are hand-picked hex, not ramp stops: {literals:?}"
        );
    }

    #[test]
    fn the_neutral_family_holds_one_hue_in_both_themes() {
        // Criterion 2, measured off the emitted sRGB rather than off the declaration --
        // asserting the declared hue would be circular, since one ramp trivially has one
        // hue. This checks the colour that actually reaches the screen.
        //
        // The chroma floor is not slack. Hue is an angle around the neutral axis, so at low
        // chroma a single 8-bit step swings it wildly: measured, one step is an 18 degree
        // error at c = 0.003 and a 1 degree error at c = 0.014. Below the floor the emitted
        // colour has no measurable hue to be wrong about. That is a property of 8-bit sRGB
        // output, not of the ramp.
        const FLOOR: f32 = 0.012;

        let file = embedded_file().unwrap();
        let hue = file.ramps.get("neutral").unwrap().hue;
        let mut checked = 0;

        for (theme, label) in [(Theme::Light, "light"), (Theme::Dark, "dark")] {
            let tokens = Tokens::from_file(&file, theme).unwrap();
            for (name, def) in &file.tokens {
                let value = match theme {
                    Theme::Light => &def.light,
                    Theme::Dark => &def.dark,
                };
                if value.ramp_name() != Some("neutral") {
                    continue;
                }
                let measured = tokens.color(name).to_oklch();
                if measured.c < FLOOR {
                    continue;
                }
                checked += 1;
                let error = ((measured.h - hue + 180.0).rem_euclid(360.0) - 180.0).abs();
                assert!(
                    error < 6.0,
                    "{name} ({label}) measures {:.1} degrees, not {hue}: the grey family \
                     has drifted, which is the defect this ramp exists to prevent",
                    measured.h
                );
            }
        }

        // Guard the guard: a floor high enough to exempt everything would make this pass
        // vacuously.
        assert!(checked >= 8, "only {checked} neutral stops carried a hue");
    }

    #[test]
    fn there_is_exactly_one_grey_family() {
        // The hole the test above cannot cover, and I only found it by mutating the file:
        // that test asks whether tokens on the `neutral` ramp emit the neutral hue, so a
        // *second* grey ramp under another name is skipped rather than caught -- which is
        // the exact shape of the defect being fixed here, one theme carrying two greys.
        //
        // The chroma curve is what makes a ramp a grey family: a token that states neither
        // `c` nor `s` is asking for "whatever faint cast this family has", and that is a
        // sentence only a neutral ramp can answer. So there may be exactly one such ramp,
        // and every curve-taking token must be on it.
        let file = embedded_file().unwrap();

        let with_curve: Vec<&str> = file
            .ramps
            .iter()
            .filter(|(_, ramp)| ramp.chroma.is_some())
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(
            with_curve,
            ["neutral"],
            "a second ramp declaring a chroma curve is a second grey family"
        );

        for (name, def) in &file.tokens {
            for (label, value) in [("light", &def.light), ("dark", &def.dark)] {
                if let ColorValue::Ramped { ramp, c, s, .. } = value {
                    assert!(
                        c.is_some() || s.is_some() || ramp == "neutral",
                        "{name} ({label}) takes ramp `{ramp}`'s curve, but only the neutral \
                         ramp may have one"
                    );
                }
            }
        }
    }

    #[test]
    fn selection_stays_distinguishable_from_hover_by_chroma() {
        // The near-miss this token file records. Authoring surface/row-selected as a
        // fraction of its chroma ceiling instead of an absolute chroma drops the light
        // theme from c = 0.024 to c = 0.008, and in the light theme selection is separated
        // from hover by 5 per-mille of lightness and essentially nothing else. Every
        // contrast check stays green through it, because a contrast ratio is a function of
        // luminance and is blind to chroma by construction.
        let file = embedded_file().unwrap();
        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::from_file(&file, theme).unwrap();
            let selected = tokens.color("surface/row-selected").to_oklch();
            let hover = tokens.color("surface/row-hover").to_oklch();
            assert!(
                selected.c - hover.c > 0.015,
                "{theme:?}: selected c={:.4} against hover c={:.4} is not a visible tint",
                selected.c,
                hover.c
            );
        }
    }

    #[test]
    fn the_four_row_surfaces_step_uniformly_in_lightness() {
        // OKLCH lightness is perceptually uniform, so an equal step in `l` is an equal step
        // to the eye -- which is the reason to author in it at all. The dark theme did not
        // step uniformly before: 224, 246, 288, 312, so hover was nearly twice the jump
        // that banding was, and nothing in a hex file made that visible.
        let file = embedded_file().unwrap();
        for theme in [Theme::Light, Theme::Dark] {
            let tokens = Tokens::from_file(&file, theme).unwrap();
            let steps: Vec<f32> = ["surface/base", "surface/row-alt", "surface/row-hover"]
                .iter()
                .zip(["surface/row-alt", "surface/row-hover", "border/subtle"])
                .map(|(from, to)| {
                    (tokens.color(to).to_oklch().l - tokens.color(from).to_oklch().l).abs()
                })
                .collect();
            for step in &steps {
                assert!(
                    (step - 0.026).abs() < 0.004,
                    "{theme:?}: surface steps {steps:?} are not uniform"
                );
            }
        }
    }

    #[test]
    fn a_token_naming_an_undeclared_ramp_is_an_error() {
        let bad = r##"{
            "version": 2,
            "ramps": { "neutral": { "hue": 268, "chroma": 0.024 } },
            "tokens": { "x/y": {
                "role": "foreground",
                "light": { "ramp": "chartreuse", "l": 500 },
                "dark": { "ramp": "neutral", "l": 500 }
            } },
            "contrast_pairs": []
        }"##;
        assert!(matches!(
            Tokens::from_str(bad, Theme::Light),
            Err(TokenError::UnknownRamp { .. })
        ));
    }

    #[test]
    fn stating_chroma_two_ways_at_once_is_refused_rather_than_ranked() {
        // `c` and `s` are two sentences about the same axis. Picking one silently would
        // make the file's meaning depend on a precedence rule nobody reads.
        let bad = r##"{
            "version": 2,
            "ramps": { "accent": { "hue": 260 } },
            "tokens": { "x/y": {
                "role": "border",
                "light": { "ramp": "accent", "l": 500, "c": 0.1, "s": 0.9 },
                "dark": { "ramp": "accent", "l": 500 }
            } },
            "contrast_pairs": []
        }"##;
        assert!(matches!(
            Tokens::from_str(bad, Theme::Light),
            Err(TokenError::ChromaStatedTwice { .. })
        ));
    }

    #[test]
    fn the_neutral_curve_vanishes_at_both_ends_and_peaks_in_the_middle() {
        let ramp = Ramp {
            hue: 268.0,
            chroma: Some(0.024),
        };
        assert_eq!(ramp.curve_chroma(0.0), 0.0);
        assert_eq!(ramp.curve_chroma(1.0), 0.0);
        assert!((ramp.curve_chroma(0.5) - 0.024).abs() < 1e-6);
        assert!(ramp.curve_chroma(0.2) < ramp.curve_chroma(0.45));

        // A ramp with no curve is achromatic, not a panic and not a default cast.
        let flat = Ramp {
            hue: 260.0,
            chroma: None,
        };
        assert_eq!(flat.curve_chroma(0.5), 0.0);
    }

    #[test]
    fn the_light_end_of_the_neutral_ramp_is_pure_white() {
        // l = 1000 has to emit #ffffff exactly: it is the page, and a page with a cast is
        // the most visible possible way to get this wrong.
        let tokens = Tokens::embedded(Theme::Light).unwrap();
        let base = tokens.color("surface/base");
        assert!(
            base.r > 0.999 && base.g > 0.999 && base.b > 0.999,
            "{base:?}"
        );
    }

    #[test]
    fn dark_is_not_an_inversion_of_light() {
        // Asserting the authoring rule from the token file's own header. An inverted palette
        // would have every dark value equal to 1 - light, and that is what this rejects.
        let file = embedded_file().unwrap();
        let light = Tokens::from_file(&file, Theme::Light).unwrap();
        let dark = Tokens::from_file(&file, Theme::Dark).unwrap();

        let inverted = light.names().all(|name| {
            let l = light.color(name);
            let d = dark.color(name);
            (l.r + d.r - 1.0).abs() < 0.02
                && (l.g + d.g - 1.0).abs() < 0.02
                && (l.b + d.b - 1.0).abs() < 0.02
        });
        assert!(
            !inverted,
            "the dark theme appears to be a mechanical inversion"
        );
    }

    #[test]
    fn a_malformed_colour_is_an_error_not_a_default() {
        let bad = r##"{
            "version": 1,
            "tokens": { "x/y": { "role": "foreground", "light": "#zzz", "dark": "#000" } },
            "contrast_pairs": []
        }"##;
        assert!(matches!(
            Tokens::from_str(bad, Theme::Light),
            Err(TokenError::BadColor { .. })
        ));
    }

    #[test]
    fn a_pair_naming_an_unknown_token_fails_the_gate() {
        let bad = r##"{
            "version": 1,
            "tokens": { "a": { "role": "foreground", "light": "#000", "dark": "#fff" } },
            "contrast_pairs": [{ "foreground": "a", "background": "nope", "kind": "text" }]
        }"##;
        let file: TokenFile = serde_json::from_str(bad).unwrap();
        assert!(matches!(
            check_contrast(&file),
            Err(TokenError::UnknownToken(_))
        ));
    }

    #[test]
    fn comment_objects_in_the_pair_list_are_ignored() {
        let text = r##"{
            "version": 1,
            "tokens": {
                "f": { "role": "foreground", "light": "#000000", "dark": "#ffffff" },
                "b": { "role": "background", "light": "#ffffff", "dark": "#000000" }
            },
            "contrast_pairs": [
                { "$comment": "why these pairs" },
                { "foreground": "f", "background": "b", "kind": "text" }
            ]
        }"##;
        let file: TokenFile = serde_json::from_str(text).unwrap();
        assert_eq!(file.contrast_pairs.len(), 1);
        let results = check_contrast(&file).unwrap();
        assert_eq!(results.len(), 2, "one pair, two themes");
        assert!(results.iter().all(|r| r.passes()));
    }

    #[test]
    fn forced_colours_uses_only_the_supplied_palette_and_disables_effects() {
        let window = Srgba::new(0.0, 0.0, 0.0, 1.0);
        let text = Srgba::new(1.0, 1.0, 1.0, 1.0);
        let highlight = Srgba::new(0.0, 0.0, 1.0, 1.0);
        let highlight_text = Srgba::new(1.0, 1.0, 0.0, 1.0);

        let tokens = Tokens::forced(window, text, highlight, highlight_text);
        assert!(!tokens.effects_enabled());

        let allowed = [window, text, highlight, highlight_text];
        for name in tokens.names() {
            let color = tokens.color(name);
            assert!(
                allowed.contains(&color),
                "token {name} used a colour the OS did not supply"
            );
        }
    }

    #[test]
    fn an_unknown_token_reads_as_transparent_rather_than_panicking() {
        let tokens = Tokens::embedded(Theme::Light).unwrap();
        assert_eq!(tokens.color("no/such/token"), Srgba::TRANSPARENT);
        assert_eq!(tokens.try_color("no/such/token"), None);
    }

    #[test]
    fn the_rig_and_the_shader_agree_about_the_light() {
        // The PBR bevels shade from `qs_gpu::frame::LIGHT_DIR`, the contact-shadow offsets
        // displace along its negation, and the raymarched shadows march toward the rig's
        // direction. One window, one light: the authored rig and the shader constant may
        // only move together, and this is the test that makes drifting them a red build
        // instead of a window whose shadows point two ways in a way nobody can name.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        assert_eq!(
            tokens.lighting().rig.key.direction,
            qs_gpu::frame::LIGHT_DIR,
            "design/tokens.json's rig.key.direction diverged from qs_gpu::frame::LIGHT_DIR"
        );
        assert_eq!(
            KeyLightTokens::default().direction,
            qs_gpu::frame::LIGHT_DIR
        );
    }

    #[test]
    fn a_scene_slab_carries_the_materials_allowance_and_a_bare_slab_carries_none() {
        // The floor is how the contrast gate's closed-form worst case binds real frames:
        // `Tokens::scene_slab` fills it from `lit_bounds` (per material, per theme), and
        // the raw `Material::slab` leaves the safe 1.0 — a slab that skipped the allowance
        // cannot be darkened at all, which fails visibly rather than fails legibility.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let surface = crate::material::Surface::new(0.0, 0.0, 100.0, 30.0, 6.0, 1.0);

        let material = tokens.material(crate::material::name::ROW_BODY).unwrap();
        let bare = material.slab(surface).unwrap();
        assert!((bare.attenuation_floor - 1.0).abs() < f32::EPSILON);

        let filled = tokens
            .scene_slab(crate::material::name::ROW_BODY, surface)
            .unwrap();
        let expected = tokens.lit_bounds(material).attenuation.0;
        assert!(
            (filled.attenuation_floor - expected).abs() < f32::EPSILON,
            "scene_slab carried {} where the allowance says {expected}",
            filled.attenuation_floor
        );
        // The dark theme's text grounds afforded FULL BLACK when R13 measured them, and
        // no longer do: the lamp made an emitting row's ground run from shadowed to lit,
        // and no single ink survives a range that reaches black at one end and
        // accent-bright at the other — measured at 1.04:1 before the floor moved. 0.55 is
        // the value that lets one ink cover the whole range. See `lighting.$allowance_note`
        // in design/tokens.json.
        assert!(
            (expected - 0.55).abs() < 1e-6,
            "dark text-ground floor is {expected}"
        );

        // And the light theme's is 0.87 — the per-theme half of rule 3a, on the slab.
        let light = Tokens::embedded(Theme::Light).unwrap();
        let light_floor = light
            .scene_slab(crate::material::name::ROW_BODY, surface)
            .unwrap()
            .attenuation_floor;
        assert!((light_floor - 0.87).abs() < 1e-6);
    }
}
