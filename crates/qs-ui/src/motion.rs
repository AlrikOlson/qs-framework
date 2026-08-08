//! Motion, and how it stops.
//!
//! Constitution VI requires reduced motion to be honoured. The requirement is narrower and
//! stricter than "make animations shorter":
//!
//! * **No spring.** A spring overshoots and settles, and overshoot is precisely the motion
//!   that triggers vestibular symptoms. Reduced motion replaces it with an instant change
//!   or an 80 ms cross-fade -- an opacity change, which moves nothing.
//! * **Position never animates.** A fling is motion the *user* initiated and is still
//!   directing; a density change animating row heights is motion the application initiated.
//!   Under reduced motion the second one becomes instant. The first is not an animation in
//!   the relevant sense and is left alone -- disabling scrolling momentum because someone
//!   asked for reduced motion would be a misreading that makes the application harder to
//!   use for the person who asked.
//!
//! # The curve belongs to the pattern, not to the kind
//!
//! An earlier version of this module decided the curve from the [`MotionKind`] alone: every
//! layout change got a spring. UXDD 10.3 does not work that way. It is a table of *patterns*,
//! and it hands out a spring to exactly one of its nine rows -- sort/filter reorder, at
//! 220 ms with 0.8 damping. Selection change, which is unambiguously a geometry change, is
//! 120 ms `ease-out`. So the curve is an input, and [`MotionKind`] retains the one job it is
//! actually good at: deciding what reduced motion *does* to a pattern. A fade may survive,
//! capped; a layout change becomes instant.
//!
//! # Every animation retires, and that is the whole of SC-003
//!
//! The frame loop stays awake exactly as long as one [`Animation`] is live. Idle is not a
//! slow timer, it is the absence of any reason to draw, so an animation that never formally
//! completes is the single most likely cause of an SC-003 failure. Three properties defend
//! it, and all three are tested:
//!
//! 1. [`Animation::advance`] always terminates -- there is no asymptotic path.
//! 2. [`Phase::set`] is a **no-op when the target is unchanged**. Pointer motion arrives as
//!    a stream, and a caller that re-asserted the same hover row every frame would otherwise
//!    extend the animation forever and the loop would never return to `Wait`. That is a
//!    correctness property, not an optimization.
//! 3. An [`MotionPlan::Instant`] plan never constructs an `Animation` at all, so it opens no
//!    ticket for even one tick. Under reduced motion every layout change is instant, and a
//!    wake-per-keypress on the configuration whose entire point is to do less would be a
//!    particularly bad way to fail.

use crate::density::{Density, DensityTransition};
use crate::row::Interaction;

/// The user's motion preference, as reported by the OS.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MotionPreference {
    #[default]
    Full,
    Reduced,
}

/// Cross-fade duration under reduced motion, in seconds.
pub const REDUCED_CROSSFADE: f32 = 0.080;

/// What kind of change is being animated.
///
/// This decides what *reduced motion* does to a pattern. It deliberately does not decide the
/// curve -- see the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MotionKind {
    /// Row heights, panel sizes -- geometry the application changed on its own.
    Layout,
    /// Colour and opacity only.
    Fade,
    /// Scroll momentum. User-directed; see the module docs.
    UserDirectedScroll,
}

/// The shape of a transition over time.
///
/// [`Curve::Spring`] is reachable only through [`plan`] directly: none of the patterns M0
/// has a surface for uses one. It is here because "reduced motion removes the spring" is a
/// guarantee with no witness unless a spring exists to remove, and because it is the curve
/// UXDD 10.3 assigns to sort/filter reorder, which is the first pattern M1 will need.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Curve {
    /// Fast start, settled finish -- the shape that reads as "responsive".
    EaseOut,
    /// Overshoots slightly, then converges. Never produced under reduced motion.
    Spring,
}

impl Curve {
    /// Map linear time to eased progress.
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::EaseOut => crate::density::ease_out_cubic(t),
            Self::Spring => spring_out(t),
        }
    }
}

/// How a change should be presented.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MotionPlan {
    /// Jump straight to the end state.
    Instant,
    /// Animate over this many seconds.
    Animate { duration: f32, curve: Curve },
}

impl MotionPlan {
    pub fn duration(self) -> f32 {
        match self {
            Self::Instant => 0.0,
            Self::Animate { duration, .. } => duration,
        }
    }

    pub fn is_instant(self) -> bool {
        matches!(self, Self::Instant)
    }

    pub fn uses_spring(self) -> bool {
        matches!(
            self,
            Self::Animate {
                curve: Curve::Spring,
                ..
            }
        )
    }
}

/// A row of the UXDD 10.3 motion table that this build actually drives.
///
/// UXDD 10.3 has nine rows; four of them have a surface in M0. The other five -- row
/// insert/remove, sort/filter reorder, panel open/close, navigation, toast -- are
/// deliberately **absent** rather than transcribed, because a variant nothing constructs is
/// the same dead metadata this chunk exists to remove. [`MotionPattern::ALL`] and the
/// exhaustive matches below mean a fifth cannot be added without naming its duration, its
/// curve and its kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MotionPattern {
    /// Pointer enters or leaves a row. Also the *release* half of a press -- see
    /// [`InteractionMotion::set_pressed`] for why release is the slower of the two.
    HoverFeedback,
    /// Pointer goes down on a row.
    PressFeedback,
    /// The selection region moves to a different row.
    SelectionChange,
    /// Row heights change because the density changed.
    DensityChange,
}

impl MotionPattern {
    pub const ALL: [Self; 4] = [
        Self::HoverFeedback,
        Self::PressFeedback,
        Self::SelectionChange,
        Self::DensityChange,
    ];

    /// Full-motion duration in seconds, from UXDD 10.3.
    pub fn duration(self) -> f32 {
        match self {
            Self::HoverFeedback => 0.080,
            Self::PressFeedback => 0.040,
            Self::SelectionChange => 0.120,
            // Read from the transition type rather than repeated as a literal, so the
            // pattern table and the thing it drives cannot drift apart. The test
            // `the_density_pattern_and_the_density_transition_agree` is what makes that
            // more than a hope.
            Self::DensityChange => DensityTransition::DURATION,
        }
    }

    /// The curve, from UXDD 10.3. All four of M0's patterns are `ease-out`; the table's one
    /// spring belongs to a pattern M0 has no surface for.
    pub fn curve(self) -> Curve {
        match self {
            Self::HoverFeedback
            | Self::PressFeedback
            | Self::SelectionChange
            | Self::DensityChange => Curve::EaseOut,
        }
    }

    /// What reduced motion does to it.
    pub fn kind(self) -> MotionKind {
        match self {
            // Hover and press are a colour wash. Nothing moves, so reduced motion may keep
            // them -- capped at the cross-fade budget, which they are already under.
            Self::HoverFeedback | Self::PressFeedback => MotionKind::Fade,
            // Both move geometry, and both become instant.
            Self::SelectionChange | Self::DensityChange => MotionKind::Layout,
        }
    }

    /// Resolve this pattern against a preference.
    pub fn plan(self, preference: MotionPreference) -> MotionPlan {
        plan(preference, self.kind(), self.duration(), self.curve())
    }
}

/// Decide how to present a change.
pub fn plan(
    preference: MotionPreference,
    kind: MotionKind,
    full_duration: f32,
    curve: Curve,
) -> MotionPlan {
    // No spring survives reduced motion, anywhere, including on the user-directed path
    // below. Keeping momentum is not the same as keeping overshoot: the exemption exists so
    // that a fling still feels like a fling, and a fling does not need to bounce.
    let curve = match preference {
        MotionPreference::Full => curve,
        MotionPreference::Reduced => Curve::EaseOut,
    };

    match (preference, kind) {
        // User-directed scrolling is never reduced -- it is the user's own gesture still in
        // flight, not an application animation.
        (_, MotionKind::UserDirectedScroll) => MotionPlan::Animate {
            duration: full_duration,
            curve,
        },

        (MotionPreference::Full, MotionKind::Layout | MotionKind::Fade) => MotionPlan::Animate {
            duration: full_duration,
            curve,
        },

        // Geometry changes become instant. There is no "short spring".
        (MotionPreference::Reduced, MotionKind::Layout) => MotionPlan::Instant,
        // A fade moves nothing, so it may remain -- capped at 80 ms.
        (MotionPreference::Reduced, MotionKind::Fade) => MotionPlan::Animate {
            duration: full_duration.min(REDUCED_CROSSFADE),
            curve,
        },
    }
}

/// A running animation ticket.
///
/// The frame loop stays awake exactly as long as at least one of these is live. See the
/// module docs for why [`Animation::advance`] is written to always terminate.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Animation {
    elapsed: f32,
    plan: MotionPlan,
}

impl Animation {
    pub fn new(plan: MotionPlan) -> Self {
        Self { elapsed: 0.0, plan }
    }

    /// Advance by `dt` seconds. Returns `true` while still running.
    pub fn advance(&mut self, dt: f32) -> bool {
        if self.plan.is_instant() {
            return false;
        }
        self.elapsed += dt.max(0.0);
        self.elapsed < self.plan.duration()
    }

    /// Eased progress in `0.0..=1.0`.
    pub fn progress(self) -> f32 {
        let duration = self.plan.duration();
        if duration <= 0.0 {
            return 1.0;
        }
        let t = (self.elapsed / duration).clamp(0.0, 1.0);
        match self.plan {
            MotionPlan::Instant => 1.0,
            MotionPlan::Animate { curve, .. } => curve.apply(t),
        }
    }

    pub fn is_complete(self) -> bool {
        self.progress() >= 1.0
    }
}

/// A settling spring: overshoots slightly, then converges. Never used under reduced motion.
fn spring_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t >= 1.0 {
        return 1.0;
    }
    // Damped cosine. These constants give roughly 8% overshoot and one visible settle --
    // enough to read as physical, far short of the pronounced bounce that makes motion
    // uncomfortable for the people reduced-motion exists for.
    let decay = (-8.0 * t).exp();
    1.0 - decay * (10.0 * t).cos()
}

/// A value on its way from one state to another.
///
/// # `set` is idempotent on an unchanged target, and that is load-bearing
///
/// Re-targeting to the value already being animated toward does **nothing** -- it does not
/// restart the animation and it does not extend it. Hover is driven by a pointer event
/// stream; any caller that asserted the same row every frame would otherwise hold an
/// animation ticket open forever and the frame loop would never reach `Wait`. Callers should
/// not have to know that, so the invariant lives here.
///
/// # Re-targeting mid-flight snaps to the outgoing target
///
/// Changing target at progress 0.4 sets `from` to the *old target*, not to the interpolated
/// value under it, so the displayed value jumps once. At 40-120 ms that discontinuity is
/// below anything a person resolves, and the alternative -- carrying velocity across the
/// retarget so the value tracks continuously -- is a spring, which is the one thing reduced
/// motion must be able to remove entirely.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Phase<T> {
    from: T,
    to: T,
    animation: Option<Animation>,
}

impl<T> Phase<T>
where
    T: Copy + PartialEq,
{
    pub fn settled(value: T) -> Self {
        Self {
            from: value,
            to: value,
            animation: None,
        }
    }

    /// Aim at a new target. A no-op if that is already the target.
    pub fn set(&mut self, to: T, plan: MotionPlan) {
        if self.to == to {
            return;
        }
        if plan.is_instant() {
            // No ticket, not even for one tick.
            self.from = to;
            self.to = to;
            self.animation = None;
            return;
        }
        self.from = self.to;
        self.to = to;
        self.animation = Some(Animation::new(plan));
    }

    /// Advance by `dt` seconds. Returns `true` while still animating.
    pub fn advance(&mut self, dt: f32) -> bool {
        let Some(animation) = &mut self.animation else {
            return false;
        };
        if animation.advance(dt) {
            return true;
        }
        // Retire: the ticket is dropped and `from` collapses onto `to`, so a settled phase
        // is indistinguishable from one that never moved.
        self.animation = None;
        self.from = self.to;
        false
    }

    pub fn is_animating(&self) -> bool {
        self.animation.is_some()
    }

    pub fn from(&self) -> T {
        self.from
    }

    pub fn to(&self) -> T {
        self.to
    }

    /// Eased progress from `from` to `to`; `1.0` once settled.
    pub fn progress(&self) -> f32 {
        self.animation.map_or(1.0, Animation::progress)
    }

    /// How strongly `value` is present: `1.0` when it is the settled state, ramping down as
    /// it is animated away from and up as it is animated toward.
    pub fn intensity(&self, value: T) -> f32 {
        let progress = self.progress();
        let mut intensity = 0.0f32;
        if self.to == value {
            intensity = progress;
        }
        if self.from == value {
            intensity = intensity.max(1.0 - progress);
        }
        intensity.clamp(0.0, 1.0)
    }
}

/// How the selection region should be drawn this frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SelectionDraw {
    /// The row whose slot the region is anchored to.
    pub row: u64,
    /// Vertical displacement from that row's top, in **rows**. Zero once settled; positive
    /// means the region is still above `row`, on its way down.
    pub offset_rows: f32,
    /// Opacity, `0.0..=1.0`.
    pub alpha: f32,
}

/// The animated presentation of [`Interaction`], plus the density transition.
///
/// This is deliberately the **only** thing that answers "is anything animating". The frame
/// loop's decision to keep polling or go back to `Wait` is SC-003, and splitting that answer
/// across two owners is how one of them gets forgotten.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InteractionMotion {
    preference: MotionPreference,
    hover: Phase<Option<u64>>,
    pressed: Phase<Option<u64>>,
    selected: Phase<Option<u64>>,
    density: Option<DensityTransition>,
}

impl InteractionMotion {
    pub fn new(preference: MotionPreference) -> Self {
        Self {
            preference,
            hover: Phase::settled(None),
            pressed: Phase::settled(None),
            selected: Phase::settled(None),
            density: None,
        }
    }

    pub fn preference(&self) -> MotionPreference {
        self.preference
    }

    /// Bring the animated state in line with the interaction state. Idempotent: calling this
    /// every frame with unchanged input starts nothing.
    pub fn sync(&mut self, interaction: Interaction) {
        self.set_hovered(interaction.hovered);
        self.set_pressed(interaction.pressed);
        self.set_selected(interaction.selected);
    }

    pub fn set_hovered(&mut self, row: Option<u64>) {
        let plan = MotionPattern::HoverFeedback.plan(self.preference);
        self.hover.set(row, plan);
    }

    /// Press down is faster than release, which is UXDD 10.3's "press is faster than
    /// release" read literally: going *to* a pressed row uses the 40 ms press timing, and
    /// letting go uses the 80 ms hover timing, because releasing is the row returning to the
    /// hover state it came from.
    pub fn set_pressed(&mut self, row: Option<u64>) {
        let pattern = match row {
            Some(_) => MotionPattern::PressFeedback,
            None => MotionPattern::HoverFeedback,
        };
        let plan = pattern.plan(self.preference);
        self.pressed.set(row, plan);
    }

    pub fn set_selected(&mut self, row: Option<u64>) {
        let plan = MotionPattern::SelectionChange.plan(self.preference);
        self.selected.set(row, plan);
    }

    /// Start a density morph, returning the transition to drive the layout with.
    ///
    /// `None` means present the change instantly -- either because reduced motion asked for
    /// that, or because the two densities are the same.
    pub fn begin_density(&mut self, from: Density, to: Density) -> Option<DensityTransition> {
        self.density = None;
        if from == to {
            return None;
        }
        if MotionPattern::DensityChange
            .plan(self.preference)
            .is_instant()
        {
            return None;
        }
        self.density = Some(DensityTransition::new(from, to));
        self.density
    }

    pub fn density(&self) -> Option<DensityTransition> {
        self.density
    }

    /// Advance every live animation by `dt` seconds. Returns `true` while any is still
    /// running.
    ///
    /// Every phase is advanced unconditionally -- no short-circuit -- because a `||` chain
    /// would leave the later ones frozen for as long as an earlier one runs, and a frozen
    /// animation is a ticket that never retires.
    pub fn advance(&mut self, dt: f32) -> bool {
        let hover = self.hover.advance(dt);
        let pressed = self.pressed.advance(dt);
        let selected = self.selected.advance(dt);
        let density = match &mut self.density {
            Some(transition) => {
                if transition.advance(dt) {
                    true
                } else {
                    self.density = None;
                    false
                }
            }
            None => false,
        };
        hover || pressed || selected || density
    }

    pub fn is_animating(&self) -> bool {
        self.hover.is_animating()
            || self.pressed.is_animating()
            || self.selected.is_animating()
            || self.density.is_some()
    }

    /// Hover intensity for one row, `0.0..=1.0`.
    pub fn hover_alpha(&self, row: u64) -> f32 {
        self.hover.intensity(Some(row))
    }

    /// Press intensity for one row, `0.0..=1.0`.
    pub fn press_alpha(&self, row: u64) -> f32 {
        self.pressed.intensity(Some(row))
    }

    /// Where to draw the selection region, given how many rows the viewport holds.
    ///
    /// A move of more than a viewport's worth of rows **cross-fades in place instead of
    /// sliding**. A region that has to travel further than the screen is tall never reads as
    /// travelling: it reads as a flash, and at a million rows the naive version would try to
    /// slide the region across half the corpus in 120 ms. The viewport is the bound because
    /// it is exactly the distance beyond which the start and the end cannot both be seen.
    pub fn selection_draw(&self, viewport_rows: u32) -> Option<SelectionDraw> {
        if !self.selected.is_animating() {
            return self.selected.to().map(|row| SelectionDraw {
                row,
                offset_rows: 0.0,
                alpha: 1.0,
            });
        }

        let progress = self.selected.progress();
        match (self.selected.from(), self.selected.to()) {
            (Some(from), Some(to)) => {
                let travel = (from as i64) - (to as i64);
                if travel.unsigned_abs() <= u64::from(viewport_rows.max(1)) {
                    Some(SelectionDraw {
                        row: to,
                        offset_rows: (1.0 - progress) * travel as f32,
                        alpha: 1.0,
                    })
                } else {
                    Some(SelectionDraw {
                        row: to,
                        offset_rows: 0.0,
                        alpha: progress,
                    })
                }
            }
            (None, Some(to)) => Some(SelectionDraw {
                row: to,
                offset_rows: 0.0,
                alpha: progress,
            }),
            (Some(from), None) => Some(SelectionDraw {
                row: from,
                offset_rows: 0.0,
                alpha: 1.0 - progress,
            }),
            // Unreachable while animating: `Phase::set` refuses a no-op target, so `from`
            // and `to` differ for as long as a ticket is live.
            (None, None) => None,
        }
    }
}

impl Default for InteractionMotion {
    fn default() -> Self {
        Self::new(MotionPreference::default())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    const FRAME: f32 = 1.0 / 120.0;

    /// Advance until settled, asserting termination. Returns the number of frames it took.
    fn run_to_rest(motion: &mut InteractionMotion) -> u32 {
        let mut frames = 0;
        while motion.advance(FRAME) {
            frames += 1;
            assert!(frames < 10_000, "an animation never retired");
        }
        assert!(!motion.is_animating());
        frames
    }

    // -- the plan model -----------------------------------------------------------------

    #[test]
    fn reduced_motion_removes_the_spring_entirely() {
        let full = plan(
            MotionPreference::Full,
            MotionKind::Layout,
            0.3,
            Curve::Spring,
        );
        assert!(full.uses_spring());

        let reduced = plan(
            MotionPreference::Reduced,
            MotionKind::Layout,
            0.3,
            Curve::Spring,
        );
        assert!(reduced.is_instant());
        assert!(!reduced.uses_spring(), "no spring survives reduced motion");
    }

    #[test]
    fn no_plan_produced_under_reduced_motion_uses_a_spring() {
        // The exceptionless version of the test above, over the whole input space --
        // including the user-directed path, which is exempt from being *shortened* but not
        // from having its overshoot removed.
        for kind in [
            MotionKind::Layout,
            MotionKind::Fade,
            MotionKind::UserDirectedScroll,
        ] {
            for curve in [Curve::EaseOut, Curve::Spring] {
                for duration in [0.0f32, 0.04, 0.12, 2.0] {
                    let reduced = plan(MotionPreference::Reduced, kind, duration, curve);
                    assert!(
                        !reduced.uses_spring(),
                        "({kind:?}, {curve:?}, {duration}s) kept its spring under reduced motion"
                    );
                }
            }
        }
    }

    #[test]
    fn reduced_motion_caps_a_fade_at_80ms() {
        let reduced = plan(
            MotionPreference::Reduced,
            MotionKind::Fade,
            0.5,
            Curve::EaseOut,
        );
        assert!((reduced.duration() - REDUCED_CROSSFADE).abs() < 1e-6);

        // A fade already shorter than the cap is not lengthened.
        let short = plan(
            MotionPreference::Reduced,
            MotionKind::Fade,
            0.02,
            Curve::EaseOut,
        );
        assert!((short.duration() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn user_directed_scrolling_is_not_shortened() {
        // Disabling scroll momentum for someone who asked for reduced motion would make the
        // application harder to use for exactly the person who asked.
        let reduced = plan(
            MotionPreference::Reduced,
            MotionKind::UserDirectedScroll,
            0.4,
            Curve::EaseOut,
        );
        assert!(!reduced.is_instant());
        assert!((reduced.duration() - 0.4).abs() < 1e-6);
    }

    // -- the UXDD table -----------------------------------------------------------------

    #[test]
    fn the_pattern_table_matches_uxdd_10_3() {
        // These are the numbers from the design document, asserted against it rather than
        // read back out of the code that is supposed to implement it.
        let expected = [
            (MotionPattern::HoverFeedback, 0.080, MotionKind::Fade),
            (MotionPattern::PressFeedback, 0.040, MotionKind::Fade),
            (MotionPattern::SelectionChange, 0.120, MotionKind::Layout),
            (MotionPattern::DensityChange, 0.120, MotionKind::Layout),
        ];
        assert_eq!(expected.len(), MotionPattern::ALL.len());

        for (pattern, duration, kind) in expected {
            assert!(
                (pattern.duration() - duration).abs() < 1e-6,
                "{pattern:?} is {}s, UXDD 10.3 says {duration}s",
                pattern.duration()
            );
            assert_eq!(pattern.kind(), kind, "{pattern:?}");
            assert_eq!(
                pattern.curve(),
                Curve::EaseOut,
                "{pattern:?}: every pattern M0 drives is ease-out; the table's one spring \
                 belongs to sort/filter reorder, which M0 has no surface for"
            );
        }
    }

    #[test]
    fn press_is_faster_than_release() {
        // UXDD 10.3, verbatim. Pressing uses the 40 ms timing; letting go is the row
        // returning to hover and uses the 80 ms one.
        assert!(MotionPattern::PressFeedback.duration() < MotionPattern::HoverFeedback.duration());

        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.set_pressed(Some(7));
        let down = run_to_rest(&mut motion);

        motion.set_pressed(None);
        let up = run_to_rest(&mut motion);

        assert!(down < up, "press took {down} frames, release {up}");
    }

    #[test]
    fn the_density_pattern_and_the_density_transition_agree() {
        // The pattern table drives DensityTransition, so a duration that drifted from it
        // would make the motion table a decoration. Read from one, asserted against both.
        assert!(
            (MotionPattern::DensityChange.duration() - DensityTransition::DURATION).abs() < 1e-6
        );
        assert_eq!(MotionPattern::DensityChange.curve(), Curve::EaseOut);
    }

    #[test]
    fn every_pattern_is_instant_or_capped_under_reduced_motion() {
        // The acceptance criterion, over the whole table: instant, or a cross-fade no longer
        // than 80 ms, and never a spring.
        for pattern in MotionPattern::ALL {
            let reduced = pattern.plan(MotionPreference::Reduced);
            assert!(!reduced.uses_spring(), "{pattern:?}");
            assert!(
                reduced.is_instant() || reduced.duration() <= REDUCED_CROSSFADE + 1e-6,
                "{pattern:?} is neither instant nor within the {REDUCED_CROSSFADE}s cross-fade \
                 budget: {reduced:?}"
            );
        }
    }

    // -- Animation ----------------------------------------------------------------------

    #[test]
    fn an_instant_plan_never_opens_an_animation_ticket() {
        // SC-003: an animation ticket that never retires is the usual cause of an idle
        // frame loop that will not go quiet.
        let mut animation = Animation::new(MotionPlan::Instant);
        assert!(!animation.advance(FRAME));
        assert!(animation.is_complete());
        assert_eq!(animation.progress(), 1.0);
    }

    #[test]
    fn every_animation_terminates() {
        for plan in [
            MotionPlan::Instant,
            MotionPlan::Animate {
                duration: 0.08,
                curve: Curve::EaseOut,
            },
            MotionPlan::Animate {
                duration: 0.3,
                curve: Curve::Spring,
            },
            MotionPlan::Animate {
                duration: 0.0,
                curve: Curve::Spring,
            },
        ] {
            let mut animation = Animation::new(plan);
            let mut frames = 0;
            while animation.advance(FRAME) {
                frames += 1;
                assert!(frames < 10_000, "animation {plan:?} never terminated");
            }
            assert!(animation.is_complete());
        }
    }

    #[test]
    fn progress_is_bounded_even_when_overshooting() {
        let mut animation = Animation::new(MotionPlan::Animate {
            duration: 0.3,
            curve: Curve::Spring,
        });
        for _ in 0..200 {
            animation.advance(FRAME);
            let p = animation.progress();
            // The spring overshoots by design, but the reported progress must stay usable
            // as an interpolation factor without callers clamping it themselves.
            assert!((0.0..=1.2).contains(&p), "progress {p} escaped its range");
        }
        assert_eq!(animation.progress(), 1.0);
    }

    #[test]
    fn the_spring_actually_overshoots_and_the_ease_does_not() {
        let overshoots = (1..100).any(|i| Curve::Spring.apply(i as f32 / 100.0) > 1.0);
        assert!(
            overshoots,
            "the spring never exceeds 1.0, so it is not a spring"
        );

        let monotonic = (1..100).all(|i| Curve::EaseOut.apply(i as f32 / 100.0) <= 1.0);
        assert!(monotonic, "the reduced-motion easing must not overshoot");
    }

    // -- Phase --------------------------------------------------------------------------

    #[test]
    fn retargeting_to_the_same_value_does_not_restart_the_animation() {
        // THE SC-003 property. Pointer motion arrives as a stream; a caller re-asserting the
        // same hover row every frame must not hold the ticket open.
        let animate = MotionPlan::Animate {
            duration: 0.08,
            curve: Curve::EaseOut,
        };
        let mut phase = Phase::settled(None);
        phase.set(Some(3), animate);

        let mut frames = 0;
        while phase.advance(FRAME) {
            // Exactly what a stream of CursorMoved events over one row would do.
            phase.set(Some(3), animate);
            frames += 1;
            assert!(
                frames < 200,
                "re-asserting the same target kept the ticket open"
            );
        }
        assert!(!phase.is_animating());
        assert_eq!(phase.progress(), 1.0);
    }

    #[test]
    fn an_instant_plan_leaves_a_phase_with_no_ticket_at_all() {
        let mut phase = Phase::settled(None);
        phase.set(Some(3), MotionPlan::Instant);
        assert!(
            !phase.is_animating(),
            "an instant plan opened a ticket for at least one tick"
        );
        assert_eq!(phase.to(), Some(3));
        assert_eq!(
            phase.from(),
            Some(3),
            "an instant change has no origin left"
        );
        assert_eq!(phase.intensity(Some(3)), 1.0);
    }

    #[test]
    fn a_retired_phase_is_indistinguishable_from_one_that_never_moved() {
        let animate = MotionPlan::Animate {
            duration: 0.08,
            curve: Curve::EaseOut,
        };
        let mut moved = Phase::settled(None);
        moved.set(Some(1), animate);
        while moved.advance(FRAME) {}

        let never = Phase::settled(Some(1));
        assert_eq!(moved, never);
    }

    #[test]
    fn intensity_hands_off_from_one_row_to_the_next() {
        let animate = MotionPlan::Animate {
            duration: 0.08,
            curve: Curve::EaseOut,
        };
        let mut phase = Phase::settled(Some(1u64));
        assert_eq!(phase.intensity(Some(1)), 1.0);
        assert_eq!(phase.intensity(Some(2)), 0.0);

        phase.set(Some(2), animate);
        phase.advance(FRAME * 2.0);

        let leaving = phase.intensity(Some(1));
        let arriving = phase.intensity(Some(2));
        assert!(leaving > 0.0 && leaving < 1.0, "leaving {leaving}");
        assert!(arriving > 0.0 && arriving < 1.0, "arriving {arriving}");
        // The pair is a cross-fade, so together they cover the row completely at every
        // instant. Without this the list visibly dims as hover moves.
        assert!(
            (leaving + arriving - 1.0).abs() < 1e-5,
            "{leaving} + {arriving} is not a cross-fade"
        );

        while phase.advance(FRAME) {}
        assert_eq!(phase.intensity(Some(1)), 0.0);
        assert_eq!(phase.intensity(Some(2)), 1.0);
    }

    // -- InteractionMotion --------------------------------------------------------------

    #[test]
    fn syncing_unchanged_interaction_state_starts_nothing() {
        // `sync` runs once per event. If it started an animation on unchanged input, every
        // scroll notch would open three tickets.
        let interaction = Interaction {
            hovered: Some(4),
            focused: Some(4),
            selected: Some(4),
            pressed: None,
        };
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.sync(interaction);
        run_to_rest(&mut motion);

        for _ in 0..100 {
            motion.sync(interaction);
            assert!(
                !motion.is_animating(),
                "an unchanged sync opened an animation ticket"
            );
        }
    }

    #[test]
    fn the_whole_interaction_sequence_returns_to_rest() {
        // Hover, press, release, select, hover elsewhere, deselect, density change -- the
        // sequence a user produces in two seconds -- and then the loop must go quiet.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        for step in 0..40u64 {
            motion.set_hovered(Some(step));
            motion.set_pressed(Some(step));
            motion.advance(FRAME);
            motion.set_pressed(None);
            motion.set_selected(Some(step));
            motion.advance(FRAME);
        }
        motion.set_hovered(None);
        motion.set_selected(None);
        motion.begin_density(Density::Default, Density::Compact);

        let frames = run_to_rest(&mut motion);
        assert!(frames > 0, "nothing animated, so this proves nothing");
        assert!(motion.density().is_none(), "the density ticket outlived it");
    }

    #[test]
    fn under_reduced_motion_a_layout_change_opens_no_ticket_at_all() {
        // The configuration whose whole point is to do less must not wake the loop once per
        // keypress.
        let mut motion = InteractionMotion::new(MotionPreference::Reduced);
        motion.set_selected(Some(9));
        assert!(!motion.is_animating(), "selection opened a ticket");
        assert!(
            motion
                .begin_density(Density::Default, Density::Compact)
                .is_none(),
            "the density change was animated under reduced motion"
        );
        assert!(!motion.is_animating(), "density opened a ticket");

        // A fade may still run -- it moves nothing -- but it is capped and it retires.
        motion.set_hovered(Some(9));
        assert!(motion.is_animating());
        run_to_rest(&mut motion);
    }

    #[test]
    fn concurrent_animations_run_concurrently_rather_than_in_sequence() {
        // `advance` must not be a `||` chain: `a || b` skips `b` for as long as `a` is
        // running, which freezes the later phases and turns overlapping animations into
        // consecutive ones. That is not just slow, it is an extra 80 ms of open ticket.
        //
        // Hover is 80 ms and selection 120 ms. Run concurrently the pair takes as long as
        // the longer one; run in sequence it takes their sum, which is 60% longer and is far
        // outside any rounding.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.set_selected(Some(1));
        motion.set_hovered(Some(1));

        let mut frames = 0;
        while motion.advance(FRAME) {
            frames += 1;
            assert!(frames < 1000, "never retired");
        }

        // Bounded by the LONGER animation, not by the sum. The two differ by 80 ms, which is
        // ten frames here -- but the bound has to be the tight one, because a version that
        // merely allowed "less than the sum" passes when the phases run in sequence and the
        // shorter one happens to finish early.
        let concurrent = (MotionPattern::SelectionChange.duration() / FRAME).ceil() as u32;
        let sequential =
            concurrent + (MotionPattern::HoverFeedback.duration() / FRAME).ceil() as u32;
        assert!(
            frames <= concurrent,
            "took {frames} frames; the longer animation alone is {concurrent} and running \
             them one after the other would be about {sequential} -- these did not overlap"
        );
        assert_eq!(motion.hover_alpha(1), 1.0);
    }

    #[test]
    fn begin_density_replaces_a_transition_rather_than_stacking_them() {
        // Mashing the density key must not leave a queue of transitions, each of which is
        // its own ticket.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        for _ in 0..20 {
            motion.begin_density(Density::Default, Density::Compact);
            motion.begin_density(Density::Compact, Density::Default);
        }
        assert!(motion.density().is_some());
        run_to_rest(&mut motion);
    }

    #[test]
    fn a_density_change_to_the_same_density_is_not_an_animation() {
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        assert!(
            motion
                .begin_density(Density::Default, Density::Default)
                .is_none()
        );
        assert!(!motion.is_animating());
    }

    #[test]
    fn a_density_morph_keeps_the_anchor_row_at_the_top_of_the_viewport_every_frame() {
        // This reproduces the app's loop exactly: capture the anchor once, then rebase the
        // scroll every frame against the height that frame will draw with. The one-shot
        // version -- rebase at the start and let the morph run -- passes at both endpoints
        // and drifts through the whole middle, which is the failure this checks for.
        const SCALE: f32 = 1.0;
        let anchor_row = 500_000u64;
        let (from, to) = (Density::Default, Density::Compact);
        let from_height = from.row_height_px(SCALE, 1.0);
        // Partway into the anchor row, so the sub-row remainder has to scale too.
        let within = 7.0f64;

        let mut motion = InteractionMotion::new(MotionPreference::Full);
        assert!(
            motion.begin_density(from, to).is_some(),
            "full motion must animate a density change, or this tests nothing"
        );

        let mut frames = 0;
        let mut last_height;
        loop {
            let height = match motion.density() {
                Some(transition) => transition.row_height_px(SCALE, 1.0),
                None => to.row_height_px(SCALE, 1.0),
            };
            let scroll =
                DensityTransition::rebase_scroll(anchor_row, within, from_height, height, 0.0);

            // The observable promise: the row at the top of the viewport is still the row
            // the user was looking at.
            let row_at_top = (scroll / f64::from(height)).floor() as u64;
            assert_eq!(
                row_at_top, anchor_row,
                "frame {frames}: row height {height} put row {row_at_top} at the top"
            );

            // And the stronger half, which the row index alone cannot see: the *same point*
            // of that row stays at the top. The remainder has to scale with the row, or a
            // pointer resting two thirds down a row ends up a quarter of the way down it.
            let fraction = (scroll - anchor_row as f64 * f64::from(height)) / f64::from(height);
            assert!(
                (fraction - within / f64::from(from_height)).abs() < 1e-6,
                "frame {frames}: the scroll sat {fraction} of the way into the anchor row, \
                 not the {} it started at",
                within / f64::from(from_height)
            );

            last_height = height;
            frames += 1;
            assert!(frames < 1000, "the morph never retired");
            if !motion.advance(FRAME) {
                break;
            }
        }

        assert!(
            frames > 1,
            "the morph finished in one frame, so nothing was animated"
        );
        assert_eq!(
            last_height,
            to.row_height_px(SCALE, 1.0),
            "the morph must land exactly on the target height"
        );
        assert!(
            motion.density().is_none(),
            "the density ticket outlived the morph"
        );

        // The control: keeping the raw pixel offset -- the thing `rebase_scroll` exists to
        // not do -- moves a million-row list by a sixth of the corpus.
        let naive = anchor_row as f64 * f64::from(from_height) + within;
        let naive_row = (naive / f64::from(to.row_height_px(SCALE, 1.0))) as u64;
        assert!(
            naive_row.abs_diff(anchor_row) > 10_000,
            "the naive version lands on row {naive_row}, which is too close to {anchor_row} \
             for this test to be checking anything"
        );
    }

    // -- selection geometry -------------------------------------------------------------

    #[test]
    fn a_settled_selection_draws_at_its_own_row_with_no_offset() {
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        assert!(motion.selection_draw(40).is_none(), "nothing is selected");

        motion.set_selected(Some(12));
        run_to_rest(&mut motion);

        let draw = motion.selection_draw(40).unwrap();
        assert_eq!(draw.row, 12);
        assert_eq!(draw.offset_rows, 0.0);
        assert_eq!(draw.alpha, 1.0);
    }

    #[test]
    fn a_nearby_selection_change_slides_and_lands_exactly_on_the_target() {
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.set_selected(Some(10));
        run_to_rest(&mut motion);
        motion.set_selected(Some(14));

        let start = motion.selection_draw(40).unwrap();
        assert_eq!(start.row, 14);
        assert_eq!(start.alpha, 1.0, "a geometry morph does not fade");
        assert!(
            (start.offset_rows - -4.0).abs() < 1e-5,
            "the region should start four rows above its target, got {}",
            start.offset_rows
        );

        let mut previous = start.offset_rows;
        while motion.advance(FRAME) {
            let draw = motion.selection_draw(40).unwrap();
            assert_eq!(draw.row, 14);
            assert!(
                draw.offset_rows >= previous - 1e-5,
                "the slide reversed: {previous} then {}",
                draw.offset_rows
            );
            previous = draw.offset_rows;
        }
        let end = motion.selection_draw(40).unwrap();
        assert_eq!(end.offset_rows, 0.0, "the morph must land exactly");
    }

    #[test]
    fn a_selection_change_further_than_the_viewport_fades_instead_of_sliding() {
        // At a million rows the naive version tries to slide the region across half the
        // corpus in 120 ms, which reads as a flash rather than as movement.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.set_selected(Some(3));
        run_to_rest(&mut motion);
        motion.set_selected(Some(900_000));

        let draw = motion.selection_draw(40).unwrap();
        assert_eq!(draw.row, 900_000);
        assert_eq!(draw.offset_rows, 0.0, "it must not try to travel");
        assert!(draw.alpha < 1.0, "it must fade in instead");

        // And the boundary is the viewport, not a number someone liked: the same move is a
        // slide when the viewport is tall enough to hold both ends.
        let mut wide = InteractionMotion::new(MotionPreference::Full);
        wide.set_selected(Some(3));
        run_to_rest(&mut wide);
        wide.set_selected(Some(43));
        assert_ne!(wide.selection_draw(40).unwrap().offset_rows, 0.0);
        assert_eq!(wide.selection_draw(39).unwrap().offset_rows, 0.0);
    }

    #[test]
    fn selecting_from_nothing_fades_in_and_deselecting_fades_out() {
        // There is nothing to morph *from*, so these two are the only cases where the
        // selection region changes opacity rather than position. Both are checked over the
        // whole run: the first frame proves nothing on its own, because a fade begins at the
        // opacity it is leaving.
        let mut motion = InteractionMotion::new(MotionPreference::Full);

        motion.set_selected(Some(5));
        assert_eq!(
            motion.selection_draw(40).unwrap().alpha,
            0.0,
            "fades in from nothing"
        );
        let mut previous = 0.0;
        while motion.advance(FRAME) {
            let draw = motion.selection_draw(40).unwrap();
            assert_eq!(draw.row, 5);
            assert_eq!(draw.offset_rows, 0.0, "there was nowhere to travel from");
            assert!(draw.alpha >= previous, "the fade-in reversed");
            previous = draw.alpha;
        }
        assert_eq!(motion.selection_draw(40).unwrap().alpha, 1.0);

        motion.set_selected(None);
        let leaving = motion.selection_draw(40).unwrap();
        assert_eq!(
            leaving.row, 5,
            "the outgoing row is what is still on screen"
        );
        assert_eq!(leaving.alpha, 1.0, "fades out from fully present");
        let mut previous = 1.0;
        while motion.advance(FRAME) {
            let draw = motion.selection_draw(40).unwrap();
            assert_eq!(
                draw.row, 5,
                "the region must not jump to a row nothing selected"
            );
            assert!(draw.alpha <= previous, "the fade-out reversed");
            previous = draw.alpha;
        }
        assert!(previous < 1.0, "the fade-out never actually faded");
        assert!(
            motion.selection_draw(40).is_none(),
            "a deselected list must draw no selection region at all"
        );
    }
}
