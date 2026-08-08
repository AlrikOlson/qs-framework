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

use qs_gpu::color::Srgba;
use serde::Deserialize;

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
    pub light: String,
    pub dark: String,
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
    /// Entries that are pure documentation (an object with only `$comment`) are dropped
    /// during deserialization by [`ContrastPair`]'s required fields, so the list here is
    /// already only real pairs.
    #[serde(default, deserialize_with = "pairs_ignoring_comments")]
    pub contrast_pairs: Vec<ContrastPair>,
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
    #[error("contrast pair references unknown token `{0}`")]
    UnknownToken(String),
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
            let (raw, label) = match theme {
                Theme::Light => (&def.light, "light"),
                Theme::Dark => (&def.dark, "dark"),
            };
            let color = Srgba::parse_hex(raw).ok_or_else(|| TokenError::BadColor {
                token: name.clone(),
                theme: label,
                value: raw.clone(),
            })?;
            colors.insert(name.clone(), color);
            roles.insert(name.clone(), def.role);
        }
        Ok(Self {
            theme,
            colors,
            roles,
            type_roles: file.type_roles.clone(),
            space: file.space.clone(),
            radius: file.radius.clone(),
            focus: file.focus,
            effects_enabled: true,
        })
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

        // Status is carried by shape and by the accessibility tree in forced-colours mode.
        // Colour-coding it here would produce three indistinguishable rails.
        set("rail/modified", window_text, TokenRole::Border);
        set("rail/added", window_text, TokenRole::Border);
        set("rail/conflict", window_text, TokenRole::Border);

        // The selected row needs its own text colour, which is the one extra the OS gives.
        colors.insert("content/on-selected".to_string(), highlight_text);
        roles.insert("content/on-selected".to_string(), TokenRole::Foreground);

        Self {
            theme: Theme::Light,
            colors,
            roles,
            // Forced-colours mode overrides colour, not type. The OS supplies a palette,
            // not a type scale, so the roles carry through unchanged.
            // Forced-colours mode overrides colour, not geometry: the OS supplies a
            // palette, not a spacing scale.
            type_roles: embedded
                .as_ref()
                .map(|f| f.type_roles.clone())
                .unwrap_or_default(),
            space: embedded
                .as_ref()
                .map(|f| f.space.clone())
                .unwrap_or_default(),
            radius: embedded
                .as_ref()
                .map(|f| f.radius.clone())
                .unwrap_or_default(),
            focus: embedded.map(|f| f.focus).unwrap_or_default(),
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

/// Check every declared pair, in both themes.
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
