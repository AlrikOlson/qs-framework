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
    roles: BTreeMap<String, TokenRole>,
    type_roles: BTreeMap<String, TypeRole>,
    space: BTreeMap<String, f32>,
    radius: BTreeMap<String, f32>,
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
        let materials =
            crate::material::resolve_all(&file.materials, &colors, &file.space, &file.radius)?;
        Ok(Self {
            theme,
            colors,
            roles,
            type_roles: file.type_roles.clone(),
            space: file.space.clone(),
            radius: file.radius.clone(),
            focus: file.focus,
            materials,
            lighting: file.lighting,
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
        self.lighting
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
        let materials = embedded
            .as_ref()
            .and_then(|f| crate::material::resolve_all(&f.materials, &colors, &space, &radius).ok())
            .unwrap_or_default();

        Self {
            theme: Theme::Light,
            colors,
            roles,
            type_roles: embedded
                .as_ref()
                .map(|f| f.type_roles.clone())
                .unwrap_or_default(),
            space,
            radius,
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
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct LightingTokens {
    pub allowance: LitAllowance,
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
}
