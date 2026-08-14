//! Named looks: the component half of the shader/component pairing.
//!
//! A [`crate::tokens::Tokens`] answers "what colour is this" and "how much space is that".
//! A **material** answers "what does a selected row look like", and it answers it from
//! `design/tokens.json` rather than from a function. Before this existed, `row.rs` decided
//! its own radii, its own washes and its own layering; `chrome.rs` decided the bar's; the
//! grid path decided a third copy of the first. Each new surface copied whichever it was
//! sitting next to, and there was no single place a look could be changed.
//!
//! # A material is a stack of layers, and a layer is one [`PrimKind`]
//!
//! ```json
//! "row/selected": {
//!   "over": ["surface/base", "surface/row-alt"],
//!   "text": ["content/primary", "content/secondary", "content/tertiary"],
//!   "layers": [
//!     { "effect": "glow", "color": "border/focus", "alpha": 0.45, "reach": "lg" },
//!     { "effect": "fill", "color": "surface/row-selected" }
//!   ]
//! }
//! ```
//!
//! Every value in it is already a token: a colour name, a step from the space scale, a
//! radius class. There is no number in a material that widget code could have hard-coded
//! instead, which is Constitution VII applied to a *look* rather than to a colour.
//!
//! # `over` and `text` are what make the contrast gate see a composite
//!
//! Text does not sit on a token; it sits on whatever the layer stack composited to at that
//! fragment. `cargo xtask contrast` could not see that before, which is why the ordering
//! rule "the halo goes **under** the fill" was a comment in `row.rs` rather than a gate: a
//! glow drawn on top would deepen the selected row's background by whatever the accent
//! contributes and quietly cost the secondary and tertiary text the 4.5:1 their token is
//! authored to preserve, and every check would stay green.
//!
//! A material states the surfaces it is drawn `over` and the foregrounds drawn on it, and
//! [`Material::composites`] walks every in-shape stop combination of the stack. Moving the
//! glow above the fill now changes a colour the gate is looking at.
//!
//! # Fidelity does not leak into call sites
//!
//! A layer names an effect kind, and [`PrimKind::fidelity`] already says what the CPU tier
//! owes that kind. So [`Material::fidelity`] is derived, not declared, and the tier decision
//! stays in `qs-gpu`: [`Material::compile`] emits the enhanced instance and
//! [`qs_gpu::frame::Instance::cpu_floor`] resolves it to the floor. A call site asking for
//! `row/selected` on the CPU tier gets the fill without the halo and never learns which tier
//! it is on.
//!
//! Forced-colours mode is the one thing resolved here rather than there, because it is a
//! *token-layer* fact and not a tier one: [`crate::tokens::Tokens::effects_enabled`] is
//! false, so enhanced layers are dropped and a gradient collapses to the stop it names as
//! its `flat`.

use std::collections::BTreeMap;

use qs_gpu::color::Srgba;
use qs_gpu::frame::{FIELD_CENTRES, Fidelity, FieldCentre, FieldWash, Instance, PrimKind};
use serde::Deserialize;

use crate::tokens::TokenError;

/// The material names the row renderer and the chrome ask for.
///
/// Constants rather than bare strings, for the reason [`crate::tokens::role`] gives: a typo
/// is then a compile error instead of a surface that silently draws nothing.
/// `every_named_material_exists_in_the_shipped_file` holds the two halves together.
use crate::substance::Substance;

pub mod name {
    /// The window's ground, painted before anything else.
    ///
    /// The one material whose surface is the whole viewport. It replaces nothing a call site
    /// used to build by hand, because there was nothing: the ground was the draw list's clear
    /// colour, which is a single value and cannot be lit.
    pub const SURFACE_CANVAS: &str = "surface/canvas";
    /// A row's own body on an unbanded row, and the surface
    /// [`crate::substance::Substance`] varies. Albedo `surface/base`.
    pub const ROW_BODY: &str = "row/body";
    /// [`ROW_BODY`] on a banded row. Albedo `surface/row-alt`; nothing else differs.
    pub const ROW_BODY_ALT: &str = "row/body-alt";
    /// Pointer hover and press on a list row. Also the grid cell's wash.
    pub const ROW_HOVER: &str = "row/hover";
    /// A selected row: the halo and the fill, in that order.
    pub const ROW_SELECTED: &str = "row/selected";
    /// The command bar's backing: the ramp, its specular top edge and the hairline.
    pub const CHROME_BAR: &str = "chrome/bar";
    /// A hovered chip inside the command bar.
    pub const CHROME_CHIP_HOVER: &str = "chrome/chip-hover";
    /// The status shelf's backing and its hairline.
    pub const CHROME_SHELF: &str = "chrome/shelf";

    /// Every material a call site in this workspace names.
    pub const ALL: [&str; 8] = [
        SURFACE_CANVAS,
        ROW_BODY,
        ROW_BODY_ALT,
        ROW_HOVER,
        ROW_SELECTED,
        CHROME_BAR,
        CHROME_CHIP_HOVER,
        CHROME_SHELF,
    ];
}

/// The shape a material is painted onto, in **physical** pixels.
///
/// The call site owns the geometry and the material owns the look. That split is what keeps
/// `row/selected` and `row/hover` exactly coincident: both are painted onto the one
/// `StateRegion`, so a row that is hovered *and* selected cannot show a rim of the weaker
/// state around the edge of the stronger one.
///
/// `scale` rides along because every length in a material is authored in **logical** pixels,
/// like the space scale and the radius classes it is stated in.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Surface {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Corner radius in physical pixels.
    pub radius: f32,
    /// Device pixel ratio, already clamped by the caller if it wants to be careful.
    pub scale: f32,
}

impl Surface {
    #[must_use]
    pub fn new(x: f32, y: f32, w: f32, h: f32, radius: f32, scale: f32) -> Self {
        Self {
            x,
            y,
            w,
            h,
            radius,
            scale,
        }
    }
}

/// Which edge band of the surface a layer occupies.
///
/// A hairline and a specular top light are the same idea with a different sign, and both
/// are geometry a *material* should be able to state — the command bar's hairline used to be
/// a second `Instance::rect` pushed by `chrome.rs` immediately after the gradient, which is
/// exactly the hand-built stack this chunk exists to remove.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Edge {
    Top,
    Bottom,
    Left,
    Right,
}

/// One layer as authored in `design/tokens.json`.
///
/// `effect` is the tag, and it names a [`PrimKind`] rather than a picture: a material that
/// could ask for something the pipeline does not draw would be a material the parity suite
/// cannot check.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "effect", rename_all = "lowercase")]
pub enum LayerDef {
    /// A flat rounded-rect fill.
    Fill {
        color: String,
        #[serde(default)]
        alpha: Option<f32>,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
    /// A two-stop ramp across the shape. `angle` is in **degrees**, measured the way the
    /// surface's coordinates run: 0 ramps left to right, 90 top to bottom.
    Gradient {
        color: String,
        to: String,
        #[serde(default)]
        angle: f32,
        #[serde(default)]
        alpha: Option<f32>,
        /// What this becomes when effects are switched off. Defaults to the near stop.
        ///
        /// Explicit rather than conventional: the command bar's resting colour is its *far*
        /// stop and the shelf's is its *near* one — the two ramp in opposite directions so
        /// they bookend the window — so any rule that picked a stop for the author would be
        /// wrong for one of them, and wrong in forced-colours mode, which is the mode with
        /// the least slack.
        #[serde(default)]
        flat: Option<String>,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        /// What the material's phase does to this layer. See [`PhaseDef`].
        ///
        /// Only the two ramps have one, and the omission from the other four variants is the
        /// design rather than an oversight: `PhaseDef` rotates an axis, and a fill, a glow,
        /// a rim and a stroke have no axis to rotate. Spelling it on them would be a knob
        /// an author could turn to no effect, which is the silent-layer failure this file
        /// refuses everywhere else.
        #[serde(default)]
        phase: PhaseDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
    /// The same two-stop ramp as [`LayerDef::Gradient`], taken **around** the shape instead of
    /// across it. `angle` is in **degrees** and is where the far stop sits at phase zero: 0
    /// puts the highlight on the right-hand side, 90 at the bottom.
    ///
    /// The ramp is mirrored around the turn, so the near stop appears opposite the highlight
    /// and there is no seam — see [`PrimKind::Sweep`]. That is also why the two stops are not
    /// interchangeable with a gradient's in feel: a sweep shows *both* of them on every
    /// surface it is painted on, at opposite sides, however the phase is turned.
    ///
    /// [`Fidelity::Exact`], unlike everything else that looks like this: the CPU tier draws the
    /// same arithmetic. A material may therefore say something with a sweep alone, which a glow
    /// or a rim may not.
    Sweep {
        color: String,
        to: String,
        /// Degrees. Where the far stop sits at phase zero.
        #[serde(default)]
        angle: f32,
        #[serde(default)]
        alpha: Option<f32>,
        /// What this becomes when effects are switched off. Defaults to the near stop, and an
        /// author who means the highlight has to say so — the same explicit choice
        /// [`LayerDef::Gradient::flat`] makes and for the same reason.
        #[serde(default)]
        flat: Option<String>,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        /// What the material's phase does to this layer — here, how many turns the highlight
        /// travels per cycle. See [`PhaseDef`].
        #[serde(default)]
        phase: PhaseDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
    /// The ambient colour field: several coloured centres drifting behind everything.
    ///
    /// The only layer whose subject is the **window** rather than the shape it is painted on,
    /// and the only one that puts data somewhere other than the instance — four centres are
    /// roughly 190 bytes against a 48-byte stride, so they travel on the [`qs_gpu::frame::DrawList`]
    /// as a [`FieldWash`] the way the environment does. [`crate::tokens::Tokens::field`] is how
    /// a call site gets them there, and it reads them from **this layer**, so the numbers the
    /// contrast gate checks and the numbers the shader draws are the same numbers.
    ///
    /// [`Fidelity::Enhanced`] with a floor of the plain fill: unlit by any centre, the field is
    /// `color`, which is what the fallback tier draws everywhere. That is a limit rather than a
    /// substitute — it is the colour the field already converges to between its centres.
    Field {
        /// The ground the centres are composited over, and what the CPU tier draws instead of
        /// the whole thing. `surface/base`, for the canvas.
        color: String,
        #[serde(default)]
        alpha: Option<f32>,
        /// How much of the centres' colour reaches the ground, `0..=1`.
        ///
        /// The one number that decides whether this is ambience or a competitor for the
        /// content in front of it. There is no separate ceiling on it, because the ceiling
        /// that can be enforced already exists: [`Material::composites`] hands the gate every
        /// centre **at this amplitude**, so an amplitude that costs text its ratio fails the
        /// build rather than being reviewed.
        #[serde(default = "one")]
        amplitude: f32,
        /// The centres. At most [`qs_gpu::frame::FIELD_CENTRES`]; extras are refused rather
        /// than dropped, because a centre the author wrote and the renderer ignores is a
        /// silent layer.
        centres: Vec<FieldCentreDef>,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        /// What the material's phase does — here, how many turns the centres travel per cycle.
        #[serde(default)]
        phase: PhaseDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
    /// The shape, solid, with a halo fading to nothing over `reach`.
    ///
    /// The one [`Fidelity::Enhanced`] effect: `tiny-skia` cannot approximate it at any
    /// tolerance, so a material using it degrades rather than diverging.
    Glow {
        color: String,
        /// The tint at the falloff limit. Defaults to `color` — see
        /// [`Instance::glow`] for why handing it a transparent colour is the easy mistake.
        #[serde(default)]
        to: Option<String>,
        #[serde(default)]
        alpha: Option<f32>,
        /// How far the halo reaches, as a step from the **space scale**.
        reach: String,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
    /// A soft light just inside the shape's edge, brightest at the boundary and gone `width`
    /// pixels inward.
    ///
    /// [`Fidelity::Enhanced`], like [`LayerDef::Glow`] and floored at nothing for a reason
    /// [`PrimKind::fidelity`] states: a material asking for one degrades rather than
    /// diverging, so it must not be the only thing saying what a surface is.
    ///
    /// Deliberately not spelled as a [`LayerDef::Stroke`] with a soft edge. A stroke is a
    /// band of uniform alpha at a fixed distance from the boundary; this reaches inward, and
    /// that difference is what makes a rounded rectangle read as a lit surface instead of as
    /// a colour.
    Rim {
        color: String,
        #[serde(default)]
        alpha: Option<f32>,
        /// How far the light reaches inward, in logical pixels. A number rather than a scale
        /// step for the same reason [`LayerDef::Stroke`]'s width is one: a rim is measured
        /// against the shape's edge, not against the space scale.
        width: f32,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
    /// A bevelled surface shaded by a real microfacet BRDF.
    ///
    /// The one layer that describes a *material* in the physical sense rather than a colour:
    /// how rough the surface is, whether it is metal, how wide the bevel that catches the
    /// light is. What it looks like is then computed rather than authored, which is the point
    /// -- two surfaces given the same roughness agree about how light behaves on them, where
    /// two hand-tuned gradients only agree until one of them is edited.
    ///
    /// [`Fidelity::Enhanced`] with a floor of the plain fill: unlit, a surface is its albedo.
    Pbr {
        /// The base colour. For a metal this tints the reflection; for a dielectric it is
        /// what scatters.
        color: String,
        #[serde(default)]
        alpha: Option<f32>,
        /// How wide the lit edge is, in logical pixels. Zero is a flat surface with no edge
        /// to catch anything, which is a legitimate thing to ask for and looks like a fill.
        #[serde(default)]
        bevel: f32,
        /// `0..=1`. Near zero is a mirror, near one is chalk.
        #[serde(default = "one")]
        roughness: f32,
        /// `0..=1`. One removes the diffuse lobe entirely, which is what makes metal metal.
        #[serde(default)]
        metallic: f32,
        /// How much of the surrounding sky the surface picks up. `0..=1` by convention,
        /// though it is a multiplier and not a fraction.
        #[serde(default = "one")]
        env: f32,
        /// How brightly the surface's **edge** emits its own albedo, on top of what it
        /// reflects. Zero is a surface that is only lit.
        ///
        /// This is what lets a selected row be the light rather than sit next to one. It is
        /// confined to `bevel` of the boundary and is identically zero deeper than that, which
        /// is not a nicety — contract rule 1a lets a meaning-bearing element emit and forbids
        /// it to light the ground directly behind itself, and puts the boundary at the bevel.
        /// So a material may turn this up without moving a single composite the contrast gate
        /// checks, and `emission_never_reaches_the_middle_of_a_surface` in `qs-gpu` measures
        /// that rather than trusting it.
        ///
        /// A surface with no bevel does not emit, because it has no edge to emit from.
        #[serde(default)]
        emission: f32,
        /// What colour this surface emits **into the scene**, when the lit mode is on.
        ///
        /// Two things emit and they are deliberately different. The `emission` above is what
        /// the surface's own EDGE draws — a bevel that glows, bounded by rule 1a. This is
        /// what the surface contributes to everything AROUND it: the light the bounce pass
        /// carries onto neighbouring rows and the canvas.
        ///
        /// Separating them is what lets a selected row read as fully emissive without
        /// touching the ground under its own filename. The row is a **source** at whatever
        /// strength the design wants; what it may not do is receive, and receivers are
        /// bounded by their own `addition_max` allowance. Contract lit-contrast rule 1a,
        /// which is the rule that makes "the whole row glows" and "the label stays legible"
        /// compatible rather than opposed.
        ///
        /// Absent means the surface lights nothing, which is every material but one.
        #[serde(default)]
        emits: Option<String>,
        /// How brightly, into the scene. Zero, or an absent [`LayerDef::Pbr::emits`], is a
        /// surface that is not a light.
        #[serde(default)]
        emits_strength: f32,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        /// What the material's phase does to this layer — here, the lamp's shimmer. See
        /// [`PhaseDef::flicker`].
        #[serde(default)]
        phase: PhaseDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
    /// An inside-aligned rounded-rect stroke.
    Stroke {
        color: String,
        #[serde(default)]
        alpha: Option<f32>,
        /// Stroke width in logical pixels. A number rather than a scale step because there
        /// is no width scale — `design/tokens.json` states focus widths the same way.
        width: f32,
        /// How this layer responds to the material's drive. See [`SwellDef`].
        #[serde(default)]
        swell: SwellDef,
        #[serde(flatten)]
        geometry: GeometryDef,
    },
}

/// One centre of a [`LayerDef::Field`], as authored.
///
/// Positions are in normalized viewport coordinates, `[0, 0]` top-left and `[1, 1]`
/// bottom-right, because a field is authored against the window rather than against a pixel
/// count. `reach` and `drift` are in units of the window's **width** on both axes, so a centre
/// stays a circle when the window is resized rather than stretching with it.
#[derive(Clone, Debug, Deserialize)]
pub struct FieldCentreDef {
    /// This centre's light.
    pub color: String,
    /// Its weight, `0..=1`. A centre at 0.5 contributes half as much as one at 1.0 — which is
    /// what makes the tint itself the colour the contrast gate checks.
    #[serde(default = "one")]
    pub alpha: f32,
    /// Where it sits at phase zero.
    pub at: [f32; 2],
    /// How far it travels over one cycle. Zero is a centre that stays put.
    #[serde(default)]
    pub drift: [f32; 2],
    /// How far its light reaches, past which it contributes exactly nothing.
    pub reach: f32,
    /// Where it starts in the cycle, in turns. Offsets are what stop four centres drifting as
    /// one rigid pattern.
    #[serde(default)]
    pub phase: f32,
}

/// What a material is being told about the moment it is being painted in.
///
/// Two numbers that answer two different questions, which is why they are one type rather
/// than one float:
///
/// - `intensity` is **how loud**. It comes from a [`crate::motion::MotionPlan`] that is
///   running, falls to zero as that plan settles, and is what [`SwellDef`] factors against.
/// - `phase` is **where in a cycle**. A swell can make a halo twice as bright; it cannot say
///   that a highlight is three-quarters of the way around a border, and no amount of
///   factoring toward a swell target expresses a position.
///
/// # The phase is not a clock, and that is structural
///
/// It is advanced inside [`crate::motion::InteractionMotion::advance`], which the frame loop
/// already calls once per frame while some *other* animation holds a ticket open — and it is
/// deliberately not part of what that function returns. So the phase rides wakefulness the
/// application already had and can never manufacture any, which is the property
/// `frame-pacing-bound` exists to protect. Reading a wall clock here instead would advance
/// the phase while the loop slept and jump the picture on the first frame after, which is a
/// sweep that teleports.
///
/// See `think:52` for why a shader uniform lost, and `think:59` for why this is the only
/// source that neither wakes the loop nor discontinues across a sleep.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Drive {
    /// How hard the material is being driven, `0.0..=1.0`.
    pub intensity: f32,
    /// Where the material is in its cycle, in **turns**: `0.0..1.0`, wrapping.
    ///
    /// Turns rather than radians because every consumer so far multiplies it by something —
    /// a number of rotations, a drift distance — and a turn is the unit that makes "one
    /// cycle" the literal number one.
    pub phase: f32,
}

impl Drive {
    /// A material that is not being driven and is at the start of its cycle.
    ///
    /// The value [`crate::tokens::Tokens::paint`] uses, and the one a reduced-motion frame
    /// is pinned to — see [`PhaseDef`].
    pub const REST: Self = Self {
        intensity: 0.0,
        phase: 0.0,
    };

    #[must_use]
    pub fn new(intensity: f32, phase: f32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, 1.0),
            // `rem_euclid` rather than `fract`: a negative phase is a caller counting
            // backwards, not a caller with a bug, and `fract` would hand it back negative
            // and put the ramp a whole turn away from where the next frame puts it.
            phase: if phase.is_finite() {
                phase.rem_euclid(1.0)
            } else {
                0.0
            },
        }
    }
}

/// What the material's **phase** does to one layer.
///
/// Zero everywhere means the layer is phase-inert, which is the default and what every
/// shipped material is today: [`Layer::driven`] then returns the layer untouched, so a
/// material with no phase-consuming layer compiles to bit-identical instances whatever the
/// phase is. `a_material_with_no_phase_consuming_layer_is_untouched_by_one` is what says so.
///
/// One field, and it is spelled on a primitive that already existed rather than on one that
/// was waiting: a ramp whose axis rotates is a drifting light, drawn by the gradient the
/// pipeline has drawn since M0. [`PrimKind::Sweep`] arrived later and needed nothing added
/// here — it states its travel through the same `rotate`, applied to the same `angle` field —
/// which is what settling the contract before the shader was for.
#[derive(Clone, Copy, PartialEq, Debug, Default, Deserialize)]
pub struct PhaseDef {
    /// Whole turns this layer's ramp axis rotates over one cycle.
    ///
    /// `1.0` is one full rotation per cycle; `-1.0` is one the other way. The authored
    /// [`LayerDef::Gradient::angle`] — or [`LayerDef::Sweep::angle`] — stays the layer's angle
    /// at phase zero, which is what makes a pinned reduced-motion frame the picture the author
    /// actually drew.
    ///
    /// It means the same thing on both ramps, and on a sweep it is the literal one: the
    /// mirror is in the *parameter*, not in the period, so the highlight has exactly one peak
    /// and `rotate: 1.0` carries it around the shape exactly once per cycle.
    #[serde(default)]
    pub rotate: f32,
    /// How hard this layer's **emission** shimmers over one cycle, as a fraction.
    ///
    /// A real lamp is never perfectly steady: mains ripple, filament thermals and the
    /// convection over a tube all put a small aperiodic wobble on the light. `0.06` is a
    /// six per cent peak-to-peak shimmer, which is about what a person reads as "alive"
    /// without reading as "faulty".
    ///
    /// It rides the same **phase** the canvas field and the chrome sweeps ride, and that is
    /// the whole reason it is affordable: `InteractionMotion::advance` moves the phase only
    /// while some other animation is already holding the frame loop open, so the lamp
    /// shimmers through a scroll, a selection move or a navigation and is a still picture
    /// between them. A lamp that flickered forever would render forever, and SC-003 — zero
    /// rendering work at rest — is a gate this project measures rather than hopes for.
    ///
    /// The shimmer's own peak is folded into [`Material::LAMP_PEAK`]'s companion
    /// [`Material::FLICKER_PEAK`], so the contrast gate bounds the brightest instant rather
    /// than the average one.
    #[serde(default)]
    pub flicker: f32,
}

/// How much louder a layer gets while its material is being driven.
///
/// A material is painted with a **drive** in `0..=1` that a call site reads off a
/// [`crate::motion::MotionPlan`] — the selection's halo flares while the region is
/// travelling and settles when it lands. Each factor is what the parameter is multiplied by
/// at full drive, interpolated from `1.0` at rest, so a layer that states no swell is
/// unaffected by the drive and a material with no swells is a still picture.
///
/// Factors rather than absolute values, for the reason [`crate::tokens::Tokens`] scales an
/// animated state's alpha rather than setting it: the authored value stays the thing the
/// design system says, and the animation is a proportion of it.
///
/// This is the whole of "animated materials". There is no clock: the drive arrives per
/// frame from the same plan machinery that already moves hover, press, selection and
/// density, so an animated material retires exactly when that plan does and Reduce Motion
/// switches it off at the source — a reduced plan is instant, so its progress is immediately
/// 1, so the drive is 0, so nothing swells. See `think:52` for why a uniform clock lost.
#[derive(Clone, Copy, PartialEq, Debug, Deserialize)]
pub struct SwellDef {
    /// Multiplies the layer's opacity. Clamped at full opacity, not beyond it.
    #[serde(default = "one")]
    pub alpha: f32,
    /// Multiplies a glow's falloff distance.
    #[serde(default = "one")]
    pub reach: f32,
    /// Multiplies a stroke's width, or how far a rim reaches inward.
    #[serde(default = "one")]
    pub width: f32,
}

fn one() -> f32 {
    1.0
}

impl Default for SwellDef {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            reach: 1.0,
            width: 1.0,
        }
    }
}

/// Geometry a layer may state relative to the surface it is painted onto.
///
/// Flattened into every layer variant so an author writes `"edge": "bottom"` beside the
/// colour rather than inside a nested object.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct GeometryDef {
    /// Shrink the shape on all sides by this step from the space scale.
    #[serde(default)]
    pub inset: Option<String>,
    /// Occupy one edge band of the shape instead of the whole of it.
    #[serde(default)]
    pub edge: Option<Edge>,
    /// The band's thickness in logical pixels. Only read when `edge` is set.
    #[serde(default)]
    pub thickness: Option<f32>,
    /// Override the surface's corner radius with a radius class. An edge band is square
    /// regardless: a one-pixel hairline with a six-pixel radius is a dashed line.
    #[serde(default)]
    pub radius: Option<String>,
    /// Displace this layer from its surface by this many logical pixels **along the direction
    /// a shadow falls**. Negative moves it toward the light.
    ///
    /// A distance and not a vector, and that is the design rather than a simplification. The
    /// direction comes from [`qs_gpu::frame::shadow_direction`], which is the key light's own
    /// vector negated — and the key light is fixed in the shader precisely so that a material
    /// cannot author a physically fine surface that disagrees with every other surface in the
    /// window. An `offset: [x, y]` would hand that back one layer at a time, and a window
    /// whose shadows point in several directions reads as broken in a way nobody can name.
    ///
    /// It exists so a **contact shadow** is a layer rather than a call site pushing an
    /// instance: a dark glow, displaced a little, is the strongest cue that a surface sits
    /// above the ground rather than being printed on it. That shadow lands on the canvas,
    /// which contract rule 1a calls a receiver and lets take all the light, so it costs the
    /// contrast budget nothing.
    #[serde(default)]
    pub offset: f32,
}

/// One material as authored.
#[derive(Clone, Debug, Deserialize)]
pub struct MaterialDef {
    pub layers: Vec<LayerDef>,
    /// Surfaces this material is painted over, for the contrast gate. Empty means the
    /// material's own bottom layer is what everything above it sits on.
    #[serde(default)]
    pub over: Vec<String>,
    /// Foregrounds drawn on top of this material, for the contrast gate.
    #[serde(default)]
    pub text: Vec<String>,
    /// The foregrounds this material carries **when it is drawn lit**.
    ///
    /// A surface that emits changes the ground under its own label, so the ink that works
    /// on it unlit is not the ink that works on it lit — an emissive panel wants dark
    /// lettering the way a lightbox sign does. Declaring the second set is what lets the
    /// gate check BOTH states instead of one: `text` against the albedo composite, these
    /// against the same composite plus the emission's closed-form peak.
    ///
    /// Empty means the material's ink does not change, which is every material that does
    /// not emit. A material that declares emission and no lit ink is checked at its lit
    /// extreme with its ordinary ink, and fails there if the lamp washes it out — which is
    /// the honest outcome rather than a special case.
    #[serde(default)]
    pub text_lit: Vec<String>,
    #[serde(default)]
    pub description: String,
    /// A step from the elevation scale, or absent for a surface that lies on the canvas.
    ///
    /// A property of the **material**, never of a layer: every layer of one material sits at
    /// the same height, or the material would describe an object with two of them
    /// (specs/002-ray-traced-mode/data-model.md). A step name rather than a number, exactly
    /// as `radius` is a class and `reach` is a space step — Principle VII.
    #[serde(default)]
    pub elevation: Option<String>,
}

/// A layer with its colours resolved for one theme and its lengths in **logical** pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Layer {
    pub kind: PrimKind,
    /// The near stop; the only stop for a fill or a stroke.
    pub near: Srgba,
    /// The far stop. Equal to `near` for a fill or a stroke.
    pub far: Srgba,
    /// What a gradient becomes with effects switched off. Equal to `near` otherwise.
    pub flat: Srgba,
    /// Radians, for a gradient.
    pub angle: f32,
    /// Glow falloff, logical pixels.
    pub reach: f32,
    /// Stroke width, or a rim's inward reach, in logical pixels.
    pub width: f32,
    /// All-sides inset, logical pixels.
    pub inset: f32,
    pub edge: Option<Edge>,
    /// Edge-band thickness, logical pixels.
    pub thickness: f32,
    /// Radius override, logical pixels. `None` keeps the surface's own radius.
    pub radius: Option<f32>,
    /// What the drive multiplies this layer by at full swell. See [`SwellDef`].
    pub swell: SwellDef,
    /// What the drive's phase moves on this layer. See [`PhaseDef`].
    pub phase: PhaseDef,
    /// Lit-edge width for [`PrimKind::Pbr`], logical pixels.
    pub bevel: f32,
    /// Microfacet roughness, `0..=1`.
    pub roughness: f32,
    /// Metalness, `0..=1`.
    pub metallic: f32,
    /// Environment strength.
    pub env: f32,
    /// Edge emission, for [`PrimKind::Pbr`]. See [`LayerDef::Pbr::emission`].
    pub emission: f32,
    /// Emission shimmer amplitude. See [`PhaseDef::flicker`].
    pub flicker: f32,
    /// What this surface contributes to the scene around it, and how brightly. See
    /// [`LayerDef::Pbr::emits`]. Transparent at zero strength means "not a light".
    pub emits: Srgba,
    pub emits_strength: f32,
    /// Displacement along the shadow direction, logical pixels. See [`GeometryDef::offset`].
    pub offset: f32,
    /// The ambient field's centres, for [`PrimKind::Field`]. See [`LayerDef::Field`].
    ///
    /// Carried on the layer even though the *instance* does not carry it, and that is the
    /// whole point: this is the one copy, read twice. [`crate::tokens::Tokens::field`] takes it
    /// to the draw list for the shader, and [`Layer::in_shape_stops`] takes it to the contrast
    /// gate. A field authored anywhere else would be a second place colour is decided.
    pub field: FieldWash,
    /// How much of the field's colour reaches its ground, `0..=1`.
    pub amplitude: f32,
}

impl Layer {
    /// The colours this layer can present **inside** the shape.
    ///
    /// One for a fill or a stroke, two for a ramp. A glow is solid inside its own shape, so
    /// its in-shape colour is its near stop and its far stop never sits under text — which
    /// is exactly why the composite has to be walked rather than the stops listed: the far
    /// stop of a halo is not a background anything is read on.
    /// This layer as it is at `drive`.
    ///
    /// The two halves of a [`Drive`] are applied independently, because a layer may state
    /// either, both or neither. The **intensity** interpolates from the authored value at
    /// rest to `swell x authored` at full drive, with opacity clamped at fully opaque rather
    /// than allowed past it: a swell says "more of this", and there is no more than all of
    /// it. The **phase** moves geometry instead of loudness, and today that is the ramp's
    /// axis.
    ///
    /// A layer that states neither comes back untouched — not merely equal, but the same
    /// value, which is what lets a material with no phase-consuming layer compile to
    /// bit-identical instances at every phase.
    ///
    /// This layer as the surface it is painted on is **made of**.
    ///
    /// Only [`PrimKind::Pbr`]'s **bevel** responds, and the narrowness is the design.
    /// Roughness, metalness and environment are deliberately untouched: each of the three
    /// changes how much light the *whole* surface returns, including the part under the
    /// label, and the contrast gate checks the albedo and so cannot see any of it. A bevel
    /// only perturbs the normal within its own width of the edge, which is the same reason
    /// [`Material::composites`] already skips `Stroke` and `Rim`. All three were tried as
    /// encodings; [`crate::substance`] records what each one did.
    ///
    /// The substance **replaces** the authored values rather than scaling them, because the
    /// authored value is the material's answer for a surface carrying no facts and the
    /// substance is the answer for one that does. Scaling would make a row's age depend on
    /// which material it happened to be painted with, which is the drift the token file
    /// exists to prevent.
    #[must_use]
    pub fn substanced(self, substance: Option<Substance>) -> Self {
        match substance {
            Some(s) if self.kind == PrimKind::Pbr => Self {
                bevel: s.bevel,
                ..self
            },
            _ => self,
        }
    }

    #[must_use]
    pub fn driven(self, drive: Drive) -> Self {
        let phased = if self.phased() {
            Self {
                angle: self.angle + drive.phase * self.phase.rotate * std::f32::consts::TAU,
                // The lamp's shimmer. Three incommensurate components rather than one sine,
                // because a single sine reads as a pulsing prop and what a real lamp does is
                // never quite repeat: the periods here (1, 2.7 and 6.3 per cycle) share no
                // common multiple inside a cycle, so the wobble does not visibly loop. The
                // amplitudes fall off with frequency, which is what makes it read as ripple
                // over a steady source rather than as noise.
                emission: self.emission * flicker(drive.phase, self.flicker),
                ..self
            }
        } else {
            self
        };
        let intensity = drive.intensity.clamp(0.0, 1.0);
        if intensity <= 0.0 || !self.swells() {
            return phased;
        }
        let factor = |swell: f32| 1.0 + (swell - 1.0) * intensity;
        let louder = |color: Srgba| Srgba {
            a: (color.a * factor(self.swell.alpha)).clamp(0.0, 1.0),
            ..color
        };
        Self {
            near: louder(self.near),
            far: louder(self.far),
            flat: louder(self.flat),
            reach: self.reach * factor(self.swell.reach),
            width: self.width * factor(self.swell.width),
            ..phased
        }
    }

    /// The shimmer factor at `phase`, for an amplitude of `amount`.
    ///
    /// Bounded by construction: the three components sum to at most `amount`, so the factor
    /// lies in `1 ± amount` and [`Material::FLICKER_PEAK`] can state the brightest instant
    /// in closed form — which is what the contrast gate needs to bound a lamp that moves.
    #[must_use]
    pub fn shimmer(phase: f32, amount: f32) -> f32 {
        flicker(phase, amount)
    }

    /// Whether the drive's **intensity** changes this layer at all.
    #[must_use]
    pub fn swells(self) -> bool {
        self.swell.alpha != 1.0 || self.swell.reach != 1.0 || self.swell.width != 1.0
    }

    /// Whether the drive's **phase** changes this layer at all.
    ///
    /// Separate from [`Layer::swells`] rather than folded into one `is_animated`, because
    /// the two answer different questions and one of them is load-bearing for the contrast
    /// gate: [`Material::composites`] has to walk both ends of a swell, and must not walk
    /// the phase, since rotating a ramp's axis moves where its stops land and never which
    /// colours they are.
    #[must_use]
    pub fn phased(self) -> bool {
        // A shimmering lamp is phase-consuming even with no rotation: its emission moves
        // with the cycle. Missing this arm is how a layer that declares a flicker gets
        // returned untouched by `driven` and never flickers — the bit-for-bit claim in
        // `driven`'s docs is about layers that state NEITHER, and this is the second one.
        self.phase.rotate != 0.0 || self.flicker > 0.0
    }

    /// Which pass this layer belongs to. A halo is the only thing that leaves its own rect.
    fn pass(self) -> Pass {
        match self.kind {
            PrimKind::Glow => Pass::Bleed,
            _ => Pass::Body,
        }
    }

    /// A [`Vec`] rather than a pair, because the field has as many in-shape colours as it has
    /// centres and the gate has to see every one of them. Not hot: this runs in
    /// [`Material::composites`], which runs in the build-time gate.
    fn in_shape_stops(self) -> Vec<Srgba> {
        match self.kind {
            // A sweep shows both stops on every surface it is painted on, at opposite sides,
            // whatever the phase — so unlike a rotating gradient it cannot be reduced to one
            // background even in principle, and both belong here.
            PrimKind::Gradient | PrimKind::Sweep => vec![self.near, self.far],
            // The ground where no centre reaches, plus what each centre makes of it at full
            // amplitude. That second set is what the acceptance means by "the field's WORST
            // stop, not its average": a field checked at its mean colour would be a gate
            // measuring a surface that does not exist anywhere in the window.
            //
            // Each entry is the same arithmetic the shader runs at a fragment one centre
            // dominates — the tint at `amplitude` composited over the ground — rather than an
            // approximation of it. Where two centres overlap the shader blends them in Oklab
            // and the gate checks no such blend; `the_worst_colour_a_field_reaches_is_one_of_
            // its_own_centres` in `qs-gpu` is what says those blends stay bracketed by these.
            PrimKind::Field => {
                let mut stops = vec![self.near];
                for centre in &self.field.centres {
                    if centre.tint.a <= 0.0 {
                        continue;
                    }
                    stops.push(scaled(centre.tint, self.amplitude).over(self.near));
                }
                stops
            }
            _ => vec![self.near, self.near],
        }
    }
}

/// Which half of a material a call site is painting.
///
/// A halo reaches past the shape it belongs to, so several instances of the same material
/// drawn in a loop would paint one shape's halo over the next shape's fill — a bright accent
/// band across the bottom of every selected row but the last, which reads as a rendering bug
/// rather than as a glow. A surface drawing many copies puts every [`Pass::Bleed`] layer down
/// first and every [`Pass::Body`] layer after.
///
/// It is a property of the layer, not of the caller: [`PrimKind::Glow`] is the only kind that
/// draws outside its own rect. A surface drawing one copy calls [`Material::compile`] and
/// never has to know this exists.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pass {
    /// Layers that reach outside the shape.
    Bleed,
    /// Layers contained by the shape.
    Body,
}

/// A named look: a stack of layers, plus what the contrast gate needs to check it.
#[derive(Clone, Debug)]
pub struct Material {
    pub layers: Vec<Layer>,
    pub over: Vec<String>,
    pub text: Vec<String>,
    /// See [`MaterialDef::text_lit`].
    pub text_lit: Vec<String>,
    pub description: String,
    /// Height of this material's top face above the canvas, **logical** pixels. Zero for a
    /// material that authored no step. Resolved from the elevation scale, so an unknown step
    /// was already an error by the time this exists.
    pub elevation: f32,
}

impl Material {
    /// The strongest fidelity any layer needs.
    ///
    /// Derived from [`PrimKind::fidelity`] rather than declared in the token file, so a
    /// material cannot claim a tier promise the pipeline does not keep. A material with no
    /// enhanced layer is [`Fidelity::Exact`], which is the honest answer for a stack of
    /// fills.
    #[must_use]
    pub fn fidelity(&self) -> Fidelity {
        self.layers
            .iter()
            .map(|layer| layer.kind.fidelity())
            .find(|f| !matches!(f, Fidelity::Exact))
            .unwrap_or(Fidelity::Exact)
    }

    /// Push this material's instances for `surface`, scaling every layer's alpha by `alpha`.
    ///
    /// `effects` is [`crate::tokens::Tokens::effects_enabled`]. With it false, enhanced
    /// layers are dropped and a ramp collapses to its `flat` stop — forced-colours mode
    /// supplies a fixed palette, and a halo or a ramp needs a colour the OS did not give.
    ///
    /// Nothing here consults a tier. An enhanced instance is emitted and
    /// [`Instance::cpu_floor`] resolves it, which is what keeps the tier decision in one
    /// place instead of at every call site.
    pub fn compile(
        &self,
        surface: Surface,
        alpha: f32,
        drive: Drive,
        effects: bool,
        out: &mut Vec<Instance>,
    ) {
        self.compile_with(surface, alpha, drive, None, effects, out);
    }

    /// [`Material::compile`], with the surface's [`Substance`] overriding what its `Pbr`
    /// layers were authored as.
    ///
    /// A separate entry point rather than a changed signature, for the reason
    /// `material_results_policy` is one: every existing caller keeps the behaviour it has,
    /// and a substance is something a call site opts into with a stated value rather than
    /// something that appears by default with numbers nobody chose.
    ///
    /// `None` is not "a neutral substance" — it means *this material was not authored to
    /// carry facts*, and its `Pbr` layers keep exactly the roughness, bevel and environment
    /// the token file gave them. [`Substance::UNKNOWN`] is the different thing: a row we
    /// know nothing about, which is what a stub gets.
    pub fn compile_with(
        &self,
        surface: Surface,
        alpha: f32,
        drive: Drive,
        substance: Option<Substance>,
        effects: bool,
        out: &mut Vec<Instance>,
    ) {
        self.compile_pass_with(Pass::Bleed, surface, alpha, drive, substance, effects, out);
        self.compile_pass_with(Pass::Body, surface, alpha, drive, substance, effects, out);
    }

    /// One half of [`Material::compile`], for a surface drawing many copies. See [`Pass`].
    #[allow(clippy::too_many_arguments)]
    pub fn compile_pass(
        &self,
        pass: Pass,
        surface: Surface,
        alpha: f32,
        drive: Drive,
        effects: bool,
        out: &mut Vec<Instance>,
    ) {
        self.compile_pass_with(pass, surface, alpha, drive, None, effects, out);
    }

    /// [`Material::compile_pass`], with a [`Substance`]. See [`Material::compile_with`].
    #[allow(clippy::too_many_arguments)]
    pub fn compile_pass_with(
        &self,
        pass: Pass,
        surface: Surface,
        alpha: f32,
        drive: Drive,
        substance: Option<Substance>,
        effects: bool,
        out: &mut Vec<Instance>,
    ) {
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return;
        }
        for layer in &self.layers {
            if layer.pass() != pass {
                continue;
            }
            // The substance is applied *before* the drive, and the order is load-bearing: a
            // drive swells what a layer is, so a swell over a substance-set bevel scales the
            // bevel this file earned. The other order would swell the authored bevel and then
            // throw it away, and a big file's press animation would look like a small one's.
            let layer = &layer.substanced(substance).driven(drive);
            // Enhanced layers are dropped with effects off -- except the ones that name a
            // flat form to collapse to. A halo has none: an opaque stand-in for a glow is the
            // hard rectangle 10.7 calls a bug. A lit surface has one, and it is its albedo.
            if !effects
                && !matches!(layer.kind.fidelity(), Fidelity::Exact)
                && layer.kind != PrimKind::Pbr
                && layer.kind != PrimKind::Field
            {
                continue;
            }
            let Some(shape) = layer_shape(*layer, surface) else {
                continue;
            };
            let (x, y, w, h, radius) = shape;
            let near = scaled(layer.near, alpha);
            let far = scaled(layer.far, alpha);
            let instance = match layer.kind {
                PrimKind::Rect => Instance::rect(x, y, w, h, radius, near),
                // Both ramps collapse to the stop they name as `flat`. A sweep is `Exact`, so
                // unlike the halo and the rim it is not dropped by the fidelity test above —
                // this arm exists because forced colours is a *palette* question and not a
                // tier one: the OS supplied four colours and a ramp between two of them is
                // not one of the four.
                // All three ramps collapse to the stop they name as `flat`. A field's is its
                // ground, which is also its CPU floor — so forced colours and the fallback
                // tier agree about the window rather than being two guesses.
                PrimKind::Gradient | PrimKind::Sweep | PrimKind::Field if !effects => {
                    Instance::rect(x, y, w, h, radius, scaled(layer.flat, alpha))
                }
                // A lit surface flattens rather than disappearing, exactly as a ramp does and
                // for a stronger reason: it is the *body* of the thing it belongs to. Dropped
                // instead, `row/selected` in forced-colours mode became a halo and a rim that
                // are themselves enhanced -- three enhanced layers, nothing left, and a
                // selected row that could not be seen. Its flat stop is its albedo, which is
                // also what `Instance::cpu_floor` leaves on the fallback tier, so the two
                // degraded forms agree instead of being two guesses.
                PrimKind::Pbr if !effects => {
                    Instance::rect(x, y, w, h, radius, scaled(layer.flat, alpha))
                }
                PrimKind::Gradient => {
                    Instance::gradient(x, y, w, h, radius, layer.angle, near, far)
                }
                // The same field, and deliberately: `Layer::driven` has already turned
                // `angle` by the drive's phase, so a travelling highlight needs nothing here
                // that a rotating gradient did not already need.
                PrimKind::Sweep => Instance::sweep(x, y, w, h, radius, layer.angle, near, far),
                // And a third consumer of the same rotated angle. The centres are not here:
                // they went to the draw list through `Tokens::field`, which is what lets one
                // instance carry a field that does not fit in one instance.
                PrimKind::Field => Instance::field(x, y, w, h, layer.amplitude, layer.angle, near),
                PrimKind::Glow => Instance::glow_two_tone(
                    x,
                    y,
                    w,
                    h,
                    radius,
                    layer.reach * surface.scale,
                    near,
                    far,
                ),
                PrimKind::Stroke => {
                    Instance::stroke(x, y, w, h, radius, layer.width * surface.scale, near)
                }
                // Same multiply as the stroke above, and stated here rather than inside
                // `Instance::rim`: every distance in an instance's `rect` is physical, and
                // every length a material authors is logical, so this line is the only place
                // the two meet for a rim.
                PrimKind::Rim => {
                    Instance::rim(x, y, w, h, radius, layer.width * surface.scale, near)
                }
                // The bevel takes the same logical-to-physical multiply the stroke and the
                // rim take. Roughness, metalness and the environment do not: they are
                // dimensionless, so a surface does not become rougher on a denser display.
                PrimKind::Pbr => Instance::pbr(
                    x,
                    y,
                    w,
                    h,
                    radius,
                    layer.bevel * surface.scale,
                    layer.roughness,
                    layer.metallic,
                    layer.env,
                    // Dimensionless, like roughness and metalness: a surface does not emit
                    // harder on a denser display. The *extent* of the emission scales, because
                    // it is the bevel's, and the bevel is a length.
                    layer.emission,
                    near,
                ),
                // A glyph is not a material layer: it samples the atlas, and what it samples
                // is text, which is content rather than a look. `LayerDef` cannot spell one.
                PrimKind::Glyph => continue,
            };
            out.push(instance);
        }
    }

    /// How far past `surface` this material reaches, in physical pixels.
    ///
    /// A halo is drawn outside the shape it belongs to, so a caller culling by the shape's
    /// own band would make the glow pop in and out rather than scroll. This is the number
    /// to widen the cull by, and it is derived from the material rather than from a constant
    /// the call site keeps its own copy of.
    #[must_use]
    pub fn bleed(&self, scale: f32, drive: Drive) -> f32 {
        self.layers
            .iter()
            .filter(|layer| layer.kind == PrimKind::Glow)
            .map(|layer| (layer.driven(drive).reach - layer.inset) * scale)
            .fold(0.0_f32, f32::max)
            .max(0.0)
    }

    /// This material as the lighting pass sees it: one slab, the shape of its body layer.
    ///
    /// The geometry comes from [`layer_shape`] on the same [`Layer`] that
    /// [`Material::compile`] turns into the body instance — scene-handoff rule 1's exact
    /// agreement is this shared origin, not a comparison somebody remembers to run. The body
    /// is the first full-coverage surface layer: not a halo (drawn outside the shape), not an
    /// edge band, not a displaced contact shadow, because a shadow cast by any of those would
    /// come from a shape that is not the surface.
    ///
    /// `None` for a material that is fully transparent — an invisible surface that casts a
    /// shadow is a shadow from nothing (see [`crate::scene::occupies_scene`]) — or one with
    /// no full-coverage layer, which today does not exist and would be a stack of hairlines.
    ///
    /// The albedo is the body's `flat` stop: the colour the CPU floor and forced-colours mode
    /// already collapse this surface to, so the three degraded descriptions of one thing
    /// agree instead of being three guesses. Emission stays zero here — authoring it is US2's
    /// task (T050), and a slab that emitted before the tokens could say so would be a light
    /// nobody can turn off.
    #[must_use]
    pub fn slab(&self, surface: Surface) -> Option<qs_gpu::scene::Slab> {
        if !crate::scene::occupies_scene(self) {
            return None;
        }
        let body = self.layers.iter().find(|layer| {
            matches!(
                layer.kind,
                PrimKind::Rect
                    | PrimKind::Gradient
                    | PrimKind::Sweep
                    | PrimKind::Field
                    | PrimKind::Pbr
            ) && layer.edge.is_none()
                && layer.inset == 0.0
                && layer.offset == 0.0
        })?;
        let (x, y, w, h, radius) = layer_shape(*body, surface)?;
        // `Instance::field` hardcodes its radius to zero -- the field is the window's ground
        // and a rounded ground has nothing behind it to show. The slab mirrors the *drawn*
        // truth, not the requested one, or rule 1's exactness fails at the first rounded
        // surface someone paints a field on.
        let radius = if body.kind == PrimKind::Field {
            0.0
        } else {
            radius
        };
        let linear = |c: Srgba| {
            [
                qs_gpu::color::srgb_to_linear(c.r),
                qs_gpu::color::srgb_to_linear(c.g),
                qs_gpu::color::srgb_to_linear(c.b),
            ]
        };
        let (roughness, metalness) = self
            .layers
            .iter()
            .find(|layer| layer.kind == PrimKind::Pbr)
            .map_or((1.0, 0.0), |layer| (layer.roughness, layer.metallic));
        let emitter = self
            .layers
            .iter()
            .find(|layer| layer.emits_strength > 0.0 && layer.emits.a > 0.0);
        let elevation = self.elevation * surface.scale;
        Some(qs_gpu::scene::Slab {
            rect: [x, y, w, h],
            radius,
            elevation,
            // Standing on the canvas: the slab extends from its top face down to the ground.
            // A floating surface is a choice nothing has made yet, and it would be authored,
            // not defaulted.
            thickness: elevation,
            albedo: linear(body.flat),
            roughness,
            metalness,
            // What this surface gives the room. Read off whichever layer declares it —
            // one per material in practice — so a material becomes a light by authoring a
            // token rather than by a call site deciding.
            emission: linear(emitter.map_or(Srgba::TRANSPARENT, |layer| layer.emits)),
            emission_strength: emitter.map_or(0.0, |layer| layer.emits_strength),
            // The safe default: no darkening permitted. `Tokens::scene_slab` is the route
            // that fills the real allowance, because the allowance is a `Tokens` fact (per
            // material AND per theme) and this function has neither.
            attenuation_floor: 1.0,
            addition_max: 0.0,
        })
    }

    /// The most this material's emission can add to its own surface, as a multiplier on the
    /// emitting layer's albedo.
    ///
    /// **Closed form, and it has to be**: the contrast gate's whole method is bounding what
    /// a frame can do without rendering one. `lamp_emission` in `shaders/instance.wgsl` is
    /// `(body + lip) * ribs * grain`, whose factors peak at `1.0`, `0.30`, `1.0` and `1.0`
    /// respectively — the centre-line and the housing lip cannot both peak at the same
    /// pixel, so `1.30` is an over-estimate rather than a sample, which is the direction a
    /// bound has to err in. `the_lamp_peak_matches_the_shader` holds this against the
    /// shader's own constants.
    pub const LAMP_PEAK: f32 = 1.30;

    /// The brightest instant of the lamp's shimmer, as a multiplier on its authored
    /// emission — `1 + amplitude`, since [`flicker`]'s three components sum to at most one.
    ///
    /// Folded into [`Material::emission_peak`] so the contrast gate bounds the lamp at its
    /// PEAK rather than at its resting value. A gate that checked the average would pass a
    /// row whose label is unreadable for a tenth of every cycle, which is worse than one
    /// that is unreadable all the time: it would never reproduce.
    #[must_use]
    pub fn flicker_peak(&self) -> f32 {
        1.0 + self
            .layers
            .iter()
            .filter(|layer| layer.kind == PrimKind::Pbr)
            .map(|layer| layer.flicker)
            .fold(0.0_f32, f32::max)
    }

    /// What this material adds to its own ground when lit, per channel, in linear light.
    ///
    /// Zero for everything that does not emit, which is why the lit gate costs the other
    /// seven materials nothing.
    #[must_use]
    pub fn emission_peak(&self) -> [f32; 3] {
        self.layers
            .iter()
            .find(|layer| layer.kind == PrimKind::Pbr && layer.emission > 0.0)
            .map_or([0.0; 3], |layer| {
                let a = layer.near;
                let e = layer.emission * Self::LAMP_PEAK * self.flicker_peak();
                [
                    qs_gpu::color::srgb_to_linear(a.r) * e,
                    qs_gpu::color::srgb_to_linear(a.g) * e,
                    qs_gpu::color::srgb_to_linear(a.b) * e,
                ]
            })
    }

    /// [`Material::composites`], at the material's **lit** extreme: every composite with the
    /// emission's closed-form peak added. What the lit ink is checked against.
    #[must_use]
    pub fn lit_composites(&self, base: Srgba) -> Vec<Srgba> {
        let add = self.emission_peak();
        self.composites(base)
            .into_iter()
            .map(|c| {
                let lift = |v: f32, a: f32| {
                    qs_gpu::color::linear_to_srgb(
                        (qs_gpu::color::srgb_to_linear(v) + a).clamp(0.0, 1.0),
                    )
                };
                Srgba {
                    r: lift(c.r, add[0]),
                    g: lift(c.g, add[1]),
                    b: lift(c.b, add[2]),
                    a: c.a,
                }
            })
            .collect()
    }

    /// Every colour text can end up sitting on, when this material is painted over `base`.
    ///
    /// The cartesian product of each layer's in-shape stops, composited in paint order. Two
    /// layers with two stops each is four composites; the stacks are two and three layers
    /// deep, so this stays a handful rather than an explosion.
    ///
    /// This is the function that makes the ordering rule checkable. An opaque fill above a
    /// halo produces composites identical to the fill alone, so the gate stays green and
    /// says nothing; move the halo on top and every composite gains the accent, which is a
    /// colour the gate is looking at.
    #[must_use]
    pub fn composites(&self, base: Srgba) -> Vec<Srgba> {
        // Both ends of the drive's *intensity*, when anything swells. A swell raises a
        // layer's opacity, so the colour text sits on at rest is not the colour it sits on
        // mid-animation, and gating only the resting stack would leave the loudest moment of
        // every animated material unchecked -- the same hole `over`/`text` closed for layer
        // ordering, reopened along a second axis.
        //
        // The *phase* is deliberately not a third axis here. What it moves is geometry --
        // where a ramp's stops land -- and `in_shape_stops` returns the same two colours
        // whatever the axis is, so walking it would multiply the work by the sample count
        // and produce the set it already has. `a_phase_cannot_change_a_colour_the_gate_checks`
        // is what keeps that a measured claim rather than an assumption about ramps.
        let drives: &[Drive] = if self.layers.iter().any(|layer| layer.swells()) {
            &[Drive::REST, Drive::new(1.0, 0.0)]
        } else {
            &[Drive::REST]
        };
        let mut results = Vec::new();
        for &drive in drives {
            let mut stack = vec![base];
            for layer in &self.layers {
                // A stroke and a rim both sit on the boundary rather than under the text in
                // the middle of the shape, so neither is a background and counting either
                // would gate text against a colour it is never read on.
                //
                // Counting the rim was tried first, on the argument that a ramp reaching
                // inward is not the same as a band at a fixed distance. The measurement
                // refused it: `chrome/chip-hover` has no headroom at all -- its dim ink is
                // already within a tenth of 4.5:1 over `surface/overlay-lift` -- so a rim
                // modelled as a full-shape background fails the gate at 0.10 opacity and is
                // invisible at the 0.076 that would pass. A model that makes a primitive
                // unauthorable at every opacity it can be seen at is describing the wrong
                // thing, and what it was describing wrongly is *where the light is*.
                //
                // The claim this rests on is the same one the stroke rests on, and it is a
                // contract on the author rather than something the type system holds: a rim
                // is an edge treatment. A light that is meant to reach under a label is a
                // fill or a gradient, both `Exact`, both counted. What keeps that honest is
                // that the rim's floor is `Nothing` (see `PrimKind::fidelity`), so UXDD
                // 10.7's "contrast is checked against the floor as well as the effect" is
                // satisfied by the composite without it -- which is the one every tier draws.
                if matches!(layer.kind, PrimKind::Stroke | PrimKind::Rim) {
                    continue;
                }
                // An edge band is likewise a hairline along one side, not a background.
                if layer.edge.is_some() {
                    continue;
                }
                let stops = layer.driven(drive).in_shape_stops();
                let mut next = Vec::with_capacity(stack.len() * stops.len());
                for under in &stack {
                    for stop in &stops {
                        let stop = *stop;
                        let over = stop.over(*under);
                        if !next.contains(&over) {
                            next.push(over);
                        }
                    }
                }
                stack = next;
            }
            for color in stack {
                if !results.contains(&color) {
                    results.push(color);
                }
            }
        }
        results
    }
}

/// Where one layer lands on the surface, or `None` if it collapses to nothing.
fn layer_shape(layer: Layer, surface: Surface) -> Option<(f32, f32, f32, f32, f32)> {
    let scale = surface.scale;
    let inset = layer.inset * scale;
    // Along the direction a shadow falls, which comes from the key light rather than from the
    // author -- see `GeometryDef::offset`. Zero for every layer that does not state one, and
    // `shadow_direction` is a constant fold, so a material with no displaced layer is
    // unaffected rather than merely unchanged.
    let displaced = qs_gpu::frame::shadow_direction();
    let shift = layer.offset * scale;
    let (mut x, mut y) = (
        surface.x + inset + displaced[0] * shift,
        surface.y + inset + displaced[1] * shift,
    );
    let (mut w, mut h) = (surface.w - inset * 2.0, surface.h - inset * 2.0);
    let mut radius = match layer.radius {
        Some(logical) => logical * scale,
        None => (surface.radius - inset).max(0.0),
    };

    if let Some(edge) = layer.edge {
        let thickness = (layer.thickness * scale).max(1.0);
        match edge {
            Edge::Top => h = thickness,
            Edge::Bottom => {
                y += (h - thickness).max(0.0);
                h = thickness;
            }
            Edge::Left => w = thickness,
            Edge::Right => {
                x += (w - thickness).max(0.0);
                w = thickness;
            }
        }
        // A one-pixel band with a six-pixel radius is a dashed line, not a hairline.
        radius = 0.0;
    }

    (w > 0.0 && h > 0.0).then_some((x, y, w, h, radius))
}

/// The lamp shimmer: three incommensurate ripples over one cycle, summing to `amount`.
///
/// Free-standing so the contrast gate, the renderer and the test that bounds it all read
/// one definition. See [`PhaseDef::flicker`] for why it is shaped this way and why it costs
/// no wakefulness.
#[must_use]
fn flicker(phase: f32, amount: f32) -> f32 {
    if amount <= 0.0 {
        return 1.0;
    }
    let tau = std::f32::consts::TAU;
    let a = (phase * tau).sin() * 0.55;
    let b = (phase * tau * 2.7 + 1.7).sin() * 0.30;
    let c = (phase * tau * 6.3 + 3.1).sin() * 0.15;
    1.0 + amount * (a + b + c)
}

/// Scale a colour's opacity, the way an animated state scales a token's.
///
/// Multiplicative: a layer authored translucent stays in proportion, so a material can never
/// be painted *more* opaque than the token file says it is.
fn scaled(color: Srgba, alpha: f32) -> Srgba {
    Srgba {
        a: color.a * alpha,
        ..color
    }
}

/// Resolve every material against an already-resolved palette.
///
/// Called from [`crate::tokens::Tokens::from_file`] and from
/// [`crate::tokens::Tokens::forced`], which is the reason it takes a colour *map* rather
/// than the token file: forced-colours mode has no ramps to resolve, only the four colours
/// the OS supplied, and a material has to come out the other side of it either way.
///
/// A material naming a token or a scale step that does not exist is an error rather than a
/// silently transparent layer, because a material that draws nothing looks exactly like a
/// call site that did not run.
pub(crate) fn resolve_all(
    defs: &BTreeMap<String, MaterialDef>,
    colors: &BTreeMap<String, Srgba>,
    space: &BTreeMap<String, f32>,
    radius: &BTreeMap<String, f32>,
    elevation: &BTreeMap<String, f32>,
) -> Result<BTreeMap<String, Material>, TokenError> {
    let color = |name: &str, material: &str| -> Result<Srgba, TokenError> {
        colors
            .get(name)
            .copied()
            .ok_or_else(|| TokenError::MaterialUnknownToken {
                material: material.to_string(),
                token: name.to_string(),
            })
    };
    let step = |name: &str, material: &str| -> Result<f32, TokenError> {
        space
            .get(name)
            .copied()
            .ok_or_else(|| TokenError::MaterialUnknownStep {
                material: material.to_string(),
                step: name.to_string(),
            })
    };
    let radius_class = |name: &str, material: &str| -> Result<f32, TokenError> {
        radius
            .get(name)
            .copied()
            .ok_or_else(|| TokenError::MaterialUnknownRadius {
                material: material.to_string(),
                class: name.to_string(),
            })
    };

    let mut out = BTreeMap::new();
    for (material, def) in defs {
        let mut layers = Vec::with_capacity(def.layers.len());
        for layer in &def.layers {
            let geometry = match layer {
                LayerDef::Fill { geometry, .. }
                | LayerDef::Gradient { geometry, .. }
                | LayerDef::Sweep { geometry, .. }
                | LayerDef::Field { geometry, .. }
                | LayerDef::Glow { geometry, .. }
                | LayerDef::Rim { geometry, .. }
                | LayerDef::Pbr { geometry, .. }
                | LayerDef::Stroke { geometry, .. } => geometry,
            };
            let swell = match layer {
                LayerDef::Fill { swell, .. }
                | LayerDef::Gradient { swell, .. }
                | LayerDef::Sweep { swell, .. }
                | LayerDef::Field { swell, .. }
                | LayerDef::Glow { swell, .. }
                | LayerDef::Rim { swell, .. }
                | LayerDef::Pbr { swell, .. }
                | LayerDef::Stroke { swell, .. } => *swell,
            };
            let inset = match &geometry.inset {
                Some(name) => step(name, material)?,
                None => 0.0,
            };
            let radius_override = match &geometry.radius {
                Some(name) => Some(radius_class(name, material)?),
                None => None,
            };
            let common = Layer {
                kind: PrimKind::Rect,
                near: Srgba::TRANSPARENT,
                far: Srgba::TRANSPARENT,
                flat: Srgba::TRANSPARENT,
                angle: 0.0,
                reach: 0.0,
                width: 0.0,
                inset,
                edge: geometry.edge,
                thickness: geometry.thickness.unwrap_or(1.0),
                radius: radius_override,
                swell,
                // Phase-inert unless the variant below says otherwise, which today only the
                // ramp can. A layer that never rotates is returned untouched by
                // `Layer::driven`, which is what the bit-for-bit claim rests on.
                phase: PhaseDef::default(),
                bevel: 0.0,
                flicker: 0.0,
                emits: Srgba::TRANSPARENT,
                emits_strength: 0.0,
                roughness: 1.0,
                metallic: 0.0,
                env: 0.0,
                // Fieldless unless the variant below says otherwise. An empty wash is not a
                // neutral one: no centre reaches anywhere, so the field is its ground, which
                // is exactly what its floor draws.
                field: FieldWash::default(),
                amplitude: 0.0,
                emission: 0.0,
                offset: geometry.offset,
            };
            let resolved = match layer {
                LayerDef::Fill {
                    color: name, alpha, ..
                } => {
                    let tint = at(color(name, material)?, *alpha);
                    Layer {
                        kind: PrimKind::Rect,
                        near: tint,
                        far: tint,
                        flat: tint,
                        ..common
                    }
                }
                LayerDef::Gradient {
                    color: name,
                    to,
                    angle,
                    alpha,
                    flat,
                    phase,
                    ..
                } => {
                    let near = at(color(name, material)?, *alpha);
                    let far = at(color(to, material)?, *alpha);
                    let flat = match flat {
                        Some(token) => at(color(token, material)?, *alpha),
                        None => near,
                    };
                    Layer {
                        kind: PrimKind::Gradient,
                        near,
                        far,
                        flat,
                        // The authored angle is the layer's angle at phase zero, which is
                        // what makes a pinned reduced-motion frame the picture the author
                        // drew rather than wherever a rotation happened to stop.
                        angle: angle.to_radians(),
                        phase: *phase,
                        ..common
                    }
                }
                LayerDef::Sweep {
                    color: name,
                    to,
                    angle,
                    alpha,
                    flat,
                    phase,
                    ..
                } => {
                    let near = at(color(name, material)?, *alpha);
                    let far = at(color(to, material)?, *alpha);
                    let flat = match flat {
                        Some(token) => at(color(token, material)?, *alpha),
                        None => near,
                    };
                    Layer {
                        kind: PrimKind::Sweep,
                        near,
                        far,
                        flat,
                        // Where the highlight sits at phase zero, which is the picture a
                        // pinned reduced-motion frame shows. Degrees in, radians on the
                        // instance, exactly as the gradient's axis is.
                        angle: angle.to_radians(),
                        phase: *phase,
                        ..common
                    }
                }
                LayerDef::Field {
                    color: name,
                    alpha,
                    amplitude,
                    centres,
                    phase,
                    ..
                } => {
                    let ground = at(color(name, material)?, *alpha);
                    // Refused rather than truncated. A centre the author wrote, the gate
                    // never checked and the renderer never drew is the silent-layer failure
                    // this file refuses everywhere else -- and it would be silent in the
                    // worst way, since the field would still look like a field.
                    if centres.len() > FIELD_CENTRES {
                        return Err(TokenError::MaterialTooManyCentres {
                            material: material.to_string(),
                            centres: centres.len(),
                            limit: FIELD_CENTRES,
                        });
                    }
                    let mut resolved = [FieldCentre::default(); FIELD_CENTRES];
                    for (slot, def) in resolved.iter_mut().zip(centres) {
                        *slot = FieldCentre {
                            at: def.at,
                            drift: def.drift,
                            reach: def.reach,
                            phase: def.phase,
                            tint: at(color(&def.color, material)?, Some(def.alpha)),
                        };
                    }
                    Layer {
                        kind: PrimKind::Field,
                        near: ground,
                        far: ground,
                        // Unlit by any centre, a field is its ground -- the same answer
                        // `Instance::cpu_floor` gives, so forced colours and the fallback
                        // tier agree instead of being two guesses.
                        flat: ground,
                        field: FieldWash { centres: resolved },
                        amplitude: amplitude.clamp(0.0, 1.0),
                        phase: *phase,
                        ..common
                    }
                }
                LayerDef::Glow {
                    color: name,
                    to,
                    alpha,
                    reach,
                    ..
                } => {
                    let near = at(color(name, material)?, *alpha);
                    // Both stops take the same colour by default. The fade belongs to the
                    // coverage profile, so a transparent outer stop multiplies two ramps
                    // together and pulls the halo in tight against the shape.
                    let far = match to {
                        Some(token) => at(color(token, material)?, *alpha),
                        None => near,
                    };
                    Layer {
                        kind: PrimKind::Glow,
                        near,
                        far,
                        flat: near,
                        reach: step(reach, material)?,
                        ..common
                    }
                }
                LayerDef::Rim {
                    color: name,
                    alpha,
                    width,
                    ..
                } => {
                    // One stop. The rim's ramp runs to nothing rather than to a second
                    // colour, and the fade lives in the coverage profile -- so `far` takes
                    // `near` here the way a fill's does, and not a transparent copy of it,
                    // which would be a second ramp multiplied into the first.
                    let tint = at(color(name, material)?, *alpha);
                    Layer {
                        kind: PrimKind::Rim,
                        near: tint,
                        far: tint,
                        flat: tint,
                        width: *width,
                        ..common
                    }
                }
                LayerDef::Pbr {
                    color: name,
                    alpha,
                    bevel,
                    roughness,
                    metallic,
                    env,
                    emission,
                    emits,
                    emits_strength,
                    phase,
                    ..
                } => {
                    let tint = at(color(name, material)?, *alpha);
                    let emits = match emits {
                        Some(token) => color(token, material)?,
                        None => Srgba::TRANSPARENT,
                    };
                    Layer {
                        kind: PrimKind::Pbr,
                        near: tint,
                        far: tint,
                        // Unlit, a surface is its albedo -- the same answer its CPU floor
                        // gives, so forced colours and the fallback tier agree about it.
                        flat: tint,
                        bevel: *bevel,
                        roughness: *roughness,
                        metallic: *metallic,
                        env: *env,
                        emission: emission.max(0.0),
                        flicker: phase.flicker.max(0.0),
                        emits,
                        emits_strength: emits_strength.max(0.0),
                        ..common
                    }
                }
                LayerDef::Stroke {
                    color: name,
                    alpha,
                    width,
                    ..
                } => {
                    let tint = at(color(name, material)?, *alpha);
                    Layer {
                        kind: PrimKind::Stroke,
                        near: tint,
                        far: tint,
                        flat: tint,
                        width: *width,
                        ..common
                    }
                }
            };
            layers.push(resolved);
        }
        // An unknown elevation step is an error exactly as an unknown colour token is: a
        // material that silently lay on the canvas would cast no shadow, and a missing
        // shadow looks like a lighting bug rather than a typo in this file.
        let elevation = match &def.elevation {
            Some(step) => {
                elevation
                    .get(step)
                    .copied()
                    .ok_or_else(|| TokenError::MaterialUnknownElevation {
                        material: material.clone(),
                        step: step.clone(),
                    })?
            }
            None => 0.0,
        };
        out.insert(
            material.clone(),
            Material {
                layers,
                over: def.over.clone(),
                text: def.text.clone(),
                text_lit: def.text_lit.clone(),
                description: def.description.clone(),
                elevation,
            },
        );
    }
    Ok(out)
}

/// Apply a layer's authored opacity to a token colour.
fn at(color: Srgba, alpha: Option<f32>) -> Srgba {
    match alpha {
        Some(alpha) => Srgba {
            a: color.a * alpha.clamp(0.0, 1.0),
            ..color
        },
        None => color,
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
    use crate::tokens::{Theme, Tokens};
    use qs_gpu::frame::Floor;

    fn surface() -> Surface {
        Surface::new(10.0, 20.0, 400.0, 28.0, 6.0, 1.0)
    }

    fn compile(name: &str, theme: Theme, effects: bool) -> Vec<Instance> {
        let tokens = Tokens::embedded(theme).unwrap();
        let material = tokens
            .material(name)
            .unwrap_or_else(|| panic!("missing material {name}"));
        let mut out = Vec::new();
        material.compile(surface(), 1.0, Drive::REST, effects, &mut out);
        out
    }

    #[test]
    fn every_named_material_exists_in_the_shipped_file() {
        // `name::ALL` is what call sites reach for. A constant that resolves to nothing is a
        // surface that draws nothing, and nothing else in the build would say so.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        for material in name::ALL {
            assert!(
                tokens.material(material).is_some(),
                "`{material}` is named in code and absent from design/tokens.json"
            );
        }
    }

    #[test]
    fn a_material_compiles_to_instances_of_the_effect_kinds_it_names() {
        // The selection is the stack the chunk is about: a halo, then the fill over it.
        let instances = compile(name::ROW_SELECTED, Theme::Dark, true);
        assert_eq!(
            instances[0].kind,
            PrimKind::Glow as u32,
            "the halo must be first, or it paints over the fill it belongs under"
        );
        assert!(
            instances.iter().any(|i| i.kind == PrimKind::Pbr as u32),
            "the selected fill is missing"
        );
        assert_eq!(
            instances.last().map(|i| i.kind),
            Some(PrimKind::Rim as u32),
            "the rim must be last, or the fill paints over the light it is supposed to have"
        );

        // And a stack that is more than one kind, drawn in the order it is authored: the
        // bar's ramp, then the specular band on its top edge, then the hairline.
        let bar = compile(name::CHROME_BAR, Theme::Dark, true);
        assert_eq!(bar[0].kind, PrimKind::Gradient as u32);
        assert_eq!(bar.len(), 3, "{bar:?}");
        assert_eq!(
            instances.len(),
            4,
            "the selection is a contact shadow, a halo, a lit surface and a rim"
        );
    }

    #[test]
    fn a_material_declares_the_fidelity_it_needs_without_naming_a_tier() {
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let selected = tokens.material(name::ROW_SELECTED).unwrap();
        let bar = tokens.material(name::CHROME_BAR).unwrap();
        let canvas = tokens.material(name::SURFACE_CANVAS).unwrap();
        let hover = tokens.material(name::ROW_HOVER).unwrap();

        // Derived from PrimKind, so the material cannot promise what the pipeline does not.
        assert_eq!(selected.fidelity(), PrimKind::Glow.fidelity());
        // A stack of exact layers is exact. The command bar is a ramp and two edge fills, all
        // three drawn on every tier.
        assert_eq!(bar.fidelity(), Fidelity::Exact);
        // The canvas used to be the example of an effect-free material, and is now the
        // opposite: its ground is a field, which is enhanced. Moving the assertion rather than
        // deleting it, because this is the direction the test exists to notice -- fidelity is
        // derived from the layers, so it moves on its own when a material is re-authored, and
        // a material that quietly became enhanced is a material whose CPU tier quietly changed.
        assert_eq!(canvas.fidelity(), PrimKind::Field.fidelity());
        // And hover stopped being exact the moment it gained a rim, which is the direction
        // this assertion exists to notice: fidelity is derived, so it moves on its own.
        assert_eq!(hover.fidelity(), PrimKind::Rim.fidelity());
    }

    #[test]
    fn an_enhanced_layer_resolves_to_the_floor_rather_than_failing() {
        // The tier decision stays in qs-gpu: `compile` emits the halo and `cpu_floor` drops
        // it. Nothing here asks which tier is running, which is the property being asserted
        // -- a material that consulted the tier would be the leak this chunk forbids.
        let instances = compile(name::ROW_SELECTED, Theme::Dark, true);
        let floored: Vec<Instance> = instances.iter().filter_map(Instance::cpu_floor).collect();

        assert!(
            instances.iter().any(|i| i.kind == PrimKind::Glow as u32),
            "the GPU tier draws the halo"
        );
        assert!(
            !floored.iter().any(|i| i.kind == PrimKind::Glow as u32),
            "the CPU tier drew a halo it cannot draw"
        );
        assert!(
            !floored.iter().any(|i| i.kind == PrimKind::Rim as u32),
            "the CPU tier drew a rim it cannot draw"
        );
        // Three enhanced layers now -- the contact shadow, the halo and the rim -- so the
        // fallback row is its lit surface flattened to its albedo and nothing else. Counted
        // rather than named, because the number is the claim: a fourth effect added without a
        // floor would change it.
        //
        // Worth being clear about what the CPU tier therefore loses here, since it is most of
        // the look: no shadow, so no sense of the row sitting above the list, and no spill.
        // What survives is the selected fill, which is what the product always had and what
        // UXDD 10.7 asks a fallback to be -- a plainer version of the same thing.
        assert_eq!(
            floored.len(),
            instances.len() - 3,
            "the floor dropped something other than the three enhanced layers"
        );
    }

    /// Every shipped material that is a **light source**, by the same test the scene builder
    /// uses to decide what emits: a layer that declares both a colour and a strength.
    ///
    /// Derived rather than listed. A hand-written list of emitting states is a second answer
    /// to "what emits", and the failure it produces is the quiet one — a state authored as a
    /// lamp after this test was written, checked by nothing.
    fn emitting_materials(tokens: &Tokens) -> Vec<(&'static str, &Material)> {
        name::ALL
            .iter()
            .filter_map(|&n| tokens.material(n).map(|m| (n, m)))
            .filter(|(_, m)| {
                m.layers
                    .iter()
                    .any(|l| l.emits_strength > 0.0 && l.emits.a > 0.0)
            })
            .collect()
    }

    /// The inverse of [`Srgba::to_premul_linear_rgba8`], written independently of it.
    ///
    /// An instance carries its colour packed, and `cpu_floor` passes that field through
    /// untouched — so recovering it is the only way to ask what the CPU tier actually draws
    /// rather than what the token file hoped it would. Written as the inverse rather than by
    /// comparing against a re-encoded candidate, for the reason `drop.rs`'s `unquote` is: a
    /// comparison against a value the same file just built agrees with itself.
    fn unpack_premul_linear(packed: u32) -> Srgba {
        let byte = |shift: u32| f32::from(((packed >> shift) & 0xFF) as u8) / 255.0;
        let a = byte(24);
        let un = |c: f32| {
            if a <= 0.0 {
                0.0
            } else {
                qs_gpu::color::linear_to_srgb((c / a).clamp(0.0, 1.0))
            }
        };
        Srgba {
            r: un(byte(0)),
            g: un(byte(8)),
            b: un(byte(16)),
            a,
        }
    }

    /// WCAG 2.1 SC 1.4.11 (non-text contrast): the floor for a user-interface component or a
    /// state indicator against what is next to it.
    ///
    /// Cited rather than chosen. A threshold picked here would be a number that moves when a
    /// material fails it, which is the failure mode `cargo xtask contrast` exists to prevent
    /// one level up.
    const STATE_CUE_MIN: f32 = 3.0;

    /// The strongest contrast this material's CPU-tier drawing reaches against `ground`.
    ///
    /// Goes through `compile` and `Instance::cpu_floor` — the authority on what a tier draws
    /// — rather than reading the token file's layers, because the two can disagree and the
    /// disagreement is exactly what a fallback bug is.
    fn cpu_cue_against(material: &Material, ground: Srgba) -> f32 {
        let mut out = Vec::new();
        material.compile(surface(), 1.0, Drive::REST, true, &mut out);
        out.iter()
            .filter_map(Instance::cpu_floor)
            .map(|i| unpack_premul_linear(i.color).over(ground).contrast_ratio(ground))
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn every_lit_state_is_also_drawn_unlit() {
        // T048 — FR-026, SC-006. The claim is not that the fallback looks as good; it is that
        // the fallback still SAYS THE SAME THING. A state carried only by emitted light is
        // invisible on the two tiers that declare `SceneFloor::Nothing` for bounce, and on
        // every machine in forced-colours mode.
        //
        // This is not hypothetical for the material it currently finds. `row/selected` became
        // a lamp and lost its status rail in the same commit, so the question "what is left
        // when nothing can light" had a new answer and nothing was asking it.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let emitting = emitting_materials(&tokens);
        assert!(
            !emitting.is_empty(),
            "no material emits, so this test is checking nothing — if emission was removed \
             on purpose, remove this test with it rather than leaving it green"
        );

        for (theme, name) in [Theme::Dark, Theme::Light]
            .into_iter()
            .flat_map(|t| emitting.iter().map(move |(n, _)| (t, *n)))
        {
            let tokens = Tokens::embedded(theme).unwrap();
            let material = tokens.material(name).unwrap();

            // 1. Something survives the floor at all.
            let mut out = Vec::new();
            material.compile(surface(), 1.0, Drive::REST, true, &mut out);
            let floored: Vec<Instance> = out.iter().filter_map(Instance::cpu_floor).collect();
            assert!(
                !floored.is_empty(),
                "{name} on {theme:?} draws nothing at all on the CPU tier, so the state it \
                 carries is conveyed by light and by nothing else"
            );

            // 2. And what survives is legible as a cue against every ground it is painted
            //    over — not merely present, which a transparent instance also satisfies.
            assert!(
                !material.over.is_empty(),
                "{name} declares no `over`, so there is nothing to be a cue against"
            );
            for base in &material.over {
                let ground = tokens.color(base);
                let ratio = cpu_cue_against(material, ground);
                assert!(
                    ratio >= STATE_CUE_MIN,
                    "{name} over {base} on {theme:?}: the CPU tier's strongest cue is \
                     {ratio:.2}:1, under the {STATE_CUE_MIN}:1 non-text floor. The state is \
                     readable only when the machine can light it."
                );
            }

            // 3. And the unlit form is not merely the lit form with the light removed: with
            //    effects off entirely — forced colours — it still draws.
            let mut forced = Vec::new();
            material.compile(surface(), 1.0, Drive::REST, false, &mut forced);
            assert!(
                !forced.is_empty(),
                "{name} on {theme:?} draws nothing in forced-colours mode"
            );
        }
    }

    #[test]
    fn an_emitting_state_is_identifiable_in_both_themes() {
        // T049 — research R6. The dark theme reflects less, so the same authored emission
        // buys less visible bounce there; a state tuned until it read on the light theme can
        // land under the noise on the dark one. The two themes are therefore checked
        // separately and against each other, not averaged.
        let dark = Tokens::embedded(Theme::Dark).unwrap();
        let light = Tokens::embedded(Theme::Light).unwrap();

        for (name, _) in emitting_materials(&dark) {
            let mut ratios = Vec::new();
            for (theme, tokens) in [("dark", &dark), ("light", &light)] {
                let material = tokens.material(name).unwrap();
                // The worst ground the material declares, since a cue only has to fail
                // against one of them to be a state somebody cannot find.
                let worst = material
                    .over
                    .iter()
                    .map(|base| cpu_cue_against(material, tokens.color(base)))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    worst >= STATE_CUE_MIN,
                    "{name} is not identifiable in the {theme} theme: worst ground gives \
                     {worst:.2}:1"
                );
                ratios.push((theme, worst));

                // And the emission itself is not what is carrying it: the peak the gate
                // bounds is a property of the theme's own palette, and a state whose lit and
                // unlit forms differ in KIND rather than in degree is two designs.
                let peak = material.emission_peak();
                assert!(
                    peak.iter().any(|c| *c > 0.0),
                    "{name} is in the emitting set and its emission peak is zero in the \
                     {theme} theme — `emitting_materials` and `emission_peak` disagree about \
                     what emits"
                );
            }

            // Neither theme may be carrying the state on its own. Stated as a ratio between
            // the two rather than as two independent floors, because a state that is 12:1 in
            // one theme and 3.1:1 in the other passes both floors and is still a state that
            // only really exists in one of them.
            let [(_, a), (_, b)] = ratios[..] else {
                unreachable!("two themes")
            };
            let spread = a.max(b) / a.min(b);
            assert!(
                spread <= 4.0,
                "{name} reads {a:.2}:1 dark and {b:.2}:1 light — a {spread:.1}x spread means \
                 the state was authored for one theme and inherited by the other"
            );
        }
    }

    #[test]
    fn effects_off_drops_the_halo_and_flattens_the_ramp() {
        // Forced-colours mode. Both are token-layer facts rather than tier ones: the OS
        // supplies a fixed palette, and a ramp or a halo needs a colour it did not give.
        let selected = compile(name::ROW_SELECTED, Theme::Dark, false);
        assert!(
            selected.iter().all(|i| i.kind == PrimKind::Rect as u32),
            "an effect survived effects_enabled = false: {selected:?}"
        );
        assert_eq!(selected.len(), 1, "the fill, and nothing else");

        // The ramp flattens rather than disappearing: the bar still has a backing, a
        // specular band and a hairline, and only the *ramp* is gone.
        let bar = compile(name::CHROME_BAR, Theme::Dark, false);
        assert!(
            bar.iter().all(|i| i.kind == PrimKind::Rect as u32),
            "{bar:?}"
        );
        assert_eq!(bar.len(), 3);
    }

    #[test]
    fn a_ramp_collapses_to_the_stop_the_material_names_and_not_to_a_convention() {
        // The bar's resting colour is its FAR stop and the selection's is its NEAR one, so
        // any rule that picked for the author would be wrong for one of them.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let bar = tokens.material(name::CHROME_BAR).unwrap();
        let ramp = bar
            .layers
            .iter()
            .find(|l| l.kind == PrimKind::Gradient)
            .expect("the bar is a ramp");
        assert_eq!(
            ramp.flat,
            tokens.color("surface/overlay"),
            "the bar flattened to its lit stop, which is the brighter one"
        );
        assert_ne!(ramp.flat, ramp.near);
    }

    #[test]
    fn an_opaque_fill_hides_what_is_under_it_and_a_reordered_halo_does_not() {
        // The ordering rule that was a comment in row.rs. With the fill on top the composite
        // is the fill; move the halo above it and every composite gains the accent.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let base = tokens.color("surface/base");
        let mut material = tokens.material(name::ROW_SELECTED).unwrap().clone();

        let correct = material.composites(base);
        material.layers.reverse();
        let wrong = material.composites(base);

        assert!(
            correct.iter().all(|c| (c.a - 1.0).abs() < 1e-6),
            "the selected fill is not opaque, so text sits on a composite that varies"
        );
        assert_ne!(
            correct, wrong,
            "moving the halo above the fill changed no colour the gate can see, which is \
             the hole `composites` exists to close"
        );
    }

    #[test]
    fn a_drive_makes_the_selection_halo_louder_and_nothing_else() {
        // The whole of "animated materials": one number, arriving per frame from a plan,
        // multiplying the parameters the layer says it may multiply. No clock reaches the
        // shader -- these two instances differ in their *contents*, and the shader they are
        // fed to is the same pure function it was.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let material = tokens.material(name::ROW_SELECTED).unwrap();

        let mut rest = Vec::new();
        material.compile(surface(), 1.0, Drive::REST, true, &mut rest);
        let mut driven = Vec::new();
        material.compile(surface(), 1.0, Drive::new(1.0, 0.0), true, &mut driven);

        assert_eq!(
            rest.len(),
            driven.len(),
            "the drive changed the layer count"
        );
        // The row draws two glows, and only one of them is the halo. The other is the contact
        // shadow, which is displaced along the shadow direction and deliberately does **not**
        // swell -- a shadow that grew while the selection travelled would say the row was
        // lifting off the page, which is not what moving means. So this picks the glow that
        // sits exactly on its surface, rather than the first one it finds: `find` used to be
        // unambiguous and now silently returns the shadow, which swells by nothing and would
        // fail this test for the right reason with a completely misleading message.
        let halo = |list: &[Instance]| -> Instance {
            *list
                .iter()
                .find(|i| i.kind == PrimKind::Glow as u32 && i.rect[0] == surface().x)
                .expect("the selection has a halo coincident with its surface")
        };
        assert!(
            halo(&driven).param > halo(&rest).param,
            "the halo did not reach further at full drive"
        );
        assert!(
            halo(&driven).color >> 24 > halo(&rest).color >> 24,
            "the halo did not brighten at full drive"
        );

        // The fill states no swell, so the drive must not touch it. A drive that leaked into
        // every layer would be a material where one animation moves everything.
        // The ramp states no swell, so the drive must not touch it. A drive that leaked into
        // every layer would be a material where one animation moves everything.
        let fill = |list: &[Instance]| -> Instance {
            *list
                .iter()
                .find(|i| i.kind == PrimKind::Pbr as u32)
                .expect("the selection has a fill")
        };
        assert_eq!(fill(&rest), fill(&driven));
    }

    #[test]
    fn a_drive_past_the_ends_is_clamped_rather_than_extrapolated() {
        // A drive is a progress, and a caller handing over 1.4 has a bug -- but the material
        // must not turn that bug into a halo four times the size, because that failure looks
        // like a rendering defect rather than like a caller mistake.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let material = tokens.material(name::ROW_SELECTED).unwrap();
        let at = |drive: f32| {
            let mut out = Vec::new();
            material.compile(surface(), 1.0, Drive::new(drive, 0.0), true, &mut out);
            out
        };
        assert_eq!(at(1.4), at(1.0));
        assert_eq!(at(-0.5), at(0.0));

        // And full opacity is the ceiling: a 45% halo doubled is 90%, but a swell can never
        // make a layer more than fully present.
        let loud = at(1.0);
        let halo = loud
            .iter()
            .find(|i| i.kind == PrimKind::Glow as u32)
            .expect("halo");
        assert!(halo.color >> 24 <= 255);
    }

    #[test]
    fn the_contrast_gate_sees_a_material_at_full_swell_as_well_as_at_rest() {
        // A swell raises a layer's opacity, so the colour text sits on at rest is not the
        // colour it sits on mid-animation. Checking only the resting stack would leave the
        // loudest moment of every animated material ungated -- the same hole `over`/`text`
        // closed for layer ordering, reopened along the drive axis.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let base = tokens.color("surface/base");
        let mut swelling = tokens.material(name::ROW_SELECTED).unwrap().clone();
        assert!(swelling.layers.iter().any(|layer| layer.swells()));

        // Under an opaque fill the two drives composite identically, which is the correct
        // answer and not a hole: what the gate must not do is fail to *look*. Make the fill
        // translucent and the two drives have to diverge.
        for layer in &mut swelling.layers {
            if layer.kind == PrimKind::Pbr {
                layer.near.a = 0.5;
                layer.far.a = 0.5;
            }
        }
        let composites = swelling.composites(base);
        assert!(
            composites.len() > 1,
            "a swelling material over a translucent fill composited to one colour, so the \
             drive axis is not being walked"
        );
    }

    #[test]
    fn a_glow_material_reports_the_bleed_a_caller_must_cull_by() {
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let selected = tokens.material(name::ROW_SELECTED).unwrap();
        assert!(
            selected.bleed(2.0, Drive::REST) > selected.bleed(1.0, Drive::REST),
            "the halo's reach is authored in logical pixels and must scale"
        );
        // `ROW_BODY`, not `ROW_HOVER`. Hover gained its own spill when the selection
        // became a lamp — the hovered row is a dimmer light, not a flat wash — so the
        // no-halo case had to move to a material that genuinely has no glow layer. The
        // claim being made is unchanged: a stack with no halo reaches past nothing, at any
        // drive.
        assert_eq!(
            tokens
                .material(name::ROW_BODY)
                .unwrap()
                .bleed(1.0, Drive::new(1.0, 0.0)),
            0.0,
            "a stack with no halo reaches past nothing"
        );
    }

    /// The shipped bar with its ramp made phase-consuming: one full turn per cycle.
    ///
    /// Built from a real material rather than from a hand-rolled one, so the layers around
    /// the ramp are the ones an author actually stacks -- an edge band and a hairline, both
    /// phase-inert, both of which have to come through untouched.
    fn rotating_bar(tokens: &Tokens) -> Material {
        let mut material = tokens.material(name::CHROME_BAR).unwrap().clone();
        let ramp = material
            .layers
            .iter_mut()
            .find(|l| l.kind == PrimKind::Gradient)
            .expect("the bar is a ramp");
        ramp.phase.rotate = 1.0;
        material
    }

    #[test]
    fn a_material_with_no_phase_consuming_layer_is_untouched_by_one() {
        // The claim the whole phase channel rests on. Every shipped material is phase-inert
        // today, so if a phase could perturb one of them, this chunk would have changed how
        // the application looks while claiming to be plumbing -- and it would have done it
        // by a float that is zero, which is the hardest kind of change to see in a diff.
        //
        // Bit-for-bit rather than approximately: `Layer::driven` returns the layer itself
        // when it is neither swelling nor phased, so there is no arithmetic to be nearly
        // right about, and a tolerance here would hide the day someone adds some.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let (mut inert, mut phased) = (0, 0);
        for material_name in name::ALL {
            let material = tokens.material(material_name).unwrap();
            if material.layers.iter().any(|l| l.phased()) {
                phased += 1;
                continue;
            }
            inert += 1;
            let at = |phase: f32| {
                let mut out = Vec::new();
                material.compile(surface(), 1.0, Drive::new(0.0, phase), true, &mut out);
                out
            };
            let rest = at(0.0);
            for phase in [0.125, 0.25, 0.5, 0.75, 0.999] {
                assert_eq!(
                    at(phase),
                    rest,
                    "{material_name} moved at phase {phase} without declaring a PhaseDef"
                );
            }
        }
        // Both counts, so the test cannot quietly become vacuous from either end: all-inert
        // means nothing consumes the channel, and all-phased means nothing is being held to
        // the bit-for-bit claim.
        assert!(inert > 0 && phased > 0, "inert {inert}, phased {phased}");
    }

    #[test]
    fn a_phase_rotates_the_layer_that_asked_for_it_and_leaves_its_neighbours_alone() {
        // The other half: a channel nothing can consume is indistinguishable from no channel,
        // so the inertness above only means something beside a layer that does move.
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let material = rotating_bar(&tokens);
        let at = |phase: f32| {
            let mut out = Vec::new();
            material.compile(surface(), 1.0, Drive::new(0.0, phase), true, &mut out);
            out
        };

        let rest = at(0.0);
        let quarter = at(0.25);
        assert_eq!(
            rest.len(),
            quarter.len(),
            "the phase changed the layer count"
        );

        // The ramp's angle is `Instance::param`, and a quarter turn is a quarter turn.
        let angle = |list: &[Instance]| {
            list.iter()
                .find(|i| i.kind == PrimKind::Gradient as u32)
                .expect("the bar draws a ramp")
                .param
        };
        assert!(
            (angle(&quarter) - angle(&rest) - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "a quarter turn moved the axis by {} radians",
            angle(&quarter) - angle(&rest)
        );

        // A whole turn is the authored angle again -- exactly, not nearly, because
        // `Drive::new` wraps the phase rather than letting the angle accumulate. That is what
        // makes the cycle a cycle instead of a ramp that runs away, and it is why the
        // rotation can be spelled as a plain multiply with no unwinding anywhere.
        assert_eq!(
            at(1.0),
            rest,
            "one turn did not come back round to the authored angle"
        );

        // And the layers that did not ask are byte-identical, so a phase is not a broadcast:
        // a material where one animation moves everything is the failure `SwellDef` already
        // had to refuse once.
        //
        // Which layers *did* ask is read off the material rather than named here. It used to
        // be spelled `kind != Gradient`, on the true-at-the-time assumption that the bar's
        // ramp was its only phase-consuming layer; when the hairline became a travelling
        // sweep that filter started holding a rotating layer to a bit-for-bit claim and this
        // test failed for a reason that had nothing to do with broadcasting. Derived, it
        // cannot go stale again the next time a layer learns to turn.
        let phased: Vec<u32> = material
            .layers
            .iter()
            .filter(|layer| layer.phased())
            .map(|layer| layer.kind as u32)
            .collect();
        assert!(
            !phased.is_empty(),
            "no layer on this material consumes a phase, so the claim below is vacuous"
        );
        let others = |list: &[Instance]| -> Vec<Instance> {
            list.iter()
                .filter(|i| !phased.contains(&i.kind))
                .copied()
                .collect()
        };
        assert!(
            !others(&rest).is_empty(),
            "every layer is phased, so nothing is being held to the bit-for-bit claim"
        );
        assert_eq!(others(&rest), others(&quarter));
    }

    #[test]
    fn a_phase_cannot_change_a_colour_the_gate_checks() {
        // Why `composites` walks the drive's intensity and deliberately not its phase.
        //
        // A swell raises opacity, so it changes what text sits on and has to be gated at both
        // ends. A phase rotates an axis: it moves where a ramp's stops land and never which
        // colours they are. If that were ever untrue, walking only the intensity would leave
        // an ungated colour under the label -- so it is measured here rather than assumed,
        // and this is the test that goes red the day a PhaseDef learns to move a colour.
        let tokens = Tokens::embedded(Theme::Light).unwrap();
        let base = tokens.color("surface/base");
        let still = tokens.material(name::CHROME_BAR).unwrap();
        let rotating = rotating_bar(&tokens);
        assert_eq!(still.composites(base), rotating.composites(base));
    }

    #[test]
    fn a_rim_lights_the_edge_and_is_not_counted_as_something_text_sits_on() {
        // Two halves of one decision, asserted together because either alone is a hole.
        //
        // The rim is skipped by `composites` for the reason a stroke is: it is on the
        // boundary, not behind the label. What makes that safe rather than optimistic is the
        // second half -- its floor is `Nothing`, so the composite the gate *does* check is
        // the one every tier draws. If someone later gives the rim a floor that draws, this
        // test is where the two halves stop agreeing.
        let tokens = Tokens::embedded(Theme::Light).unwrap();
        let material = tokens.material(name::CHROME_CHIP_HOVER).unwrap();
        assert!(
            material.layers.iter().any(|l| l.kind == PrimKind::Rim),
            "the chip lost its rim, so this test is checking nothing"
        );

        let base = tokens.color("surface/overlay-lift");
        let mut without = material.clone();
        without.layers.retain(|l| l.kind != PrimKind::Rim);
        assert_eq!(
            material.composites(base),
            without.composites(base),
            "the rim changed the colour the gate checks the label against, so it is being \
             treated as a background it is not"
        );

        assert_eq!(
            PrimKind::Rim.fidelity(),
            Fidelity::Enhanced {
                floor: Floor::Nothing
            },
            "a rim with a floor that draws would put an unchecked colour under the label on \
             the very tier that has no rim to justify skipping it"
        );
    }

    #[test]
    fn an_edge_band_is_square_and_sits_on_the_edge_it_names() {
        // The command bar's hairline, which used to be a second rect pushed by chrome.rs.
        let instances = compile(name::CHROME_BAR, Theme::Light, true);
        let s = surface();
        let hairline = instances
            .last()
            .copied()
            .expect("the bar draws a hairline last");
        assert_eq!(hairline.radius, 0.0);
        assert_eq!(hairline.rect[0], s.x);
        assert_eq!(hairline.rect[2], s.w);
        assert!(
            (hairline.rect[1] + hairline.rect[3] - (s.y + s.h)).abs() < 1e-3,
            "the bottom band does not end at the bottom edge: {hairline:?}"
        );
    }

    #[test]
    fn alpha_scales_every_layer_and_zero_draws_nothing() {
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let material = tokens.material(name::ROW_SELECTED).unwrap();

        let mut half = Vec::new();
        material.compile(surface(), 0.5, Drive::REST, true, &mut half);
        let mut full = Vec::new();
        material.compile(surface(), 1.0, Drive::REST, true, &mut full);
        assert_eq!(half.len(), full.len());
        assert_ne!(half[0].color, full[0].color, "alpha did not reach the halo");

        let mut none = Vec::new();
        material.compile(surface(), 0.0, Drive::REST, true, &mut none);
        assert!(none.is_empty(), "a fully faded material still drew");
    }

    #[test]
    fn a_degenerate_surface_emits_nothing_rather_than_an_inverted_rect() {
        let tokens = Tokens::embedded(Theme::Light).unwrap();
        let material = tokens.material(name::ROW_HOVER).unwrap();
        let mut out = Vec::new();
        material.compile(
            Surface::new(0.0, 0.0, 0.0, 0.0, 6.0, 1.0),
            1.0,
            Drive::REST,
            true,
            &mut out,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn a_material_naming_a_token_that_does_not_exist_is_an_error() {
        // A silently transparent layer looks exactly like a call site that did not run.
        let bad = r##"{
            "version": 2,
            "ramps": { "neutral": { "hue": 268, "chroma": 0.024 } },
            "tokens": { "a/b": {
                "role": "background",
                "light": { "ramp": "neutral", "l": 500 },
                "dark": { "ramp": "neutral", "l": 500 }
            } },
            "materials": { "x/y": { "layers": [
                { "effect": "fill", "color": "a/nope" }
            ] } },
            "contrast_pairs": []
        }"##;
        assert!(matches!(
            Tokens::from_str(bad, Theme::Light),
            Err(TokenError::MaterialUnknownToken { .. })
        ));
    }

    #[test]
    fn a_material_naming_a_space_step_that_does_not_exist_is_an_error() {
        let bad = r##"{
            "version": 2,
            "ramps": { "neutral": { "hue": 268, "chroma": 0.024 } },
            "space": { "lg": 8 },
            "tokens": { "a/b": {
                "role": "background",
                "light": { "ramp": "neutral", "l": 500 },
                "dark": { "ramp": "neutral", "l": 500 }
            } },
            "materials": { "x/y": { "layers": [
                { "effect": "glow", "color": "a/b", "reach": "enormous" }
            ] } },
            "contrast_pairs": []
        }"##;
        assert!(matches!(
            Tokens::from_str(bad, Theme::Light),
            Err(TokenError::MaterialUnknownStep { .. })
        ));
    }
}
