//! Animation plans, timing and reduced-motion behavior.
//!
//! Callers choose the curve and motion kind. Reduced motion makes layout changes
//! instant and caps fades at 80 milliseconds. User-controlled scrolling keeps
//! its momentum.
//!
//! [`duration_for`] scales movement duration with distance. [`Animation::advance`]
//! completes finite animations, and [`Phase::set`] leaves an unchanged target
//! alone. An instant plan creates no animation, allowing the frame loop to sleep
//! when nothing changes.

use crate::density::{Density, DensityTransition};
use crate::material::Drive;
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

/// Duration of one material cycle, in seconds of active animation time.
pub const CYCLE_SECONDS: f32 = 6.0;

/// Where the cycle sits under Reduce Motion. See [`InteractionMotion::phase`].
///
/// Zero rather than a mid-cycle value because a layer's authored angle *is* its angle at
/// phase zero, so this makes the reduced picture the one in `design/tokens.json`.
pub const PINNED_PHASE: f32 = 0.0;

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

/// Transition curve used by an animation plan.
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

/// Named interaction patterns with durations, curves and motion kinds.
///
/// [`MotionPattern::ALL`] lists the supported patterns.
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
    /// Move the focus lamp to another row.
    ///
    /// See [`InteractionMotion::focus_light_draw`].
    FocusLight,
    /// Show the accent when a session starts waiting for approval.
    ///
    /// The accent appears once and holds. Reduced motion makes the change instant.
    SummonsArrival,
}

impl MotionPattern {
    pub const ALL: [Self; 6] = [
        Self::HoverFeedback,
        Self::PressFeedback,
        Self::SelectionChange,
        Self::DensityChange,
        Self::FocusLight,
        Self::SummonsArrival,
    ];

    /// The floor and the span for a distance-scaled duration, or `None` for a pattern with
    /// no distance to scale by.
    ///
    /// The floor is what the shortest possible move takes; the span is how far a move has to
    /// be, in the pattern's own units, before it earns the whole of [`MotionPattern::duration`].
    ///
    /// Two of the four patterns are `None`, and neither is an oversight:
    ///
    /// - Hover and press are **opacity**. A fade is the same fade wherever it happens, and a
    ///   distance for it would have to be made up.
    /// - A density change reflows every row, so what moves is not one row height but each
    ///   row's accumulated offset -- the bottom of a full viewport travels forty times what
    ///   the top does. There is no single distance, and the one that matters is already long,
    ///   which is exactly the case a fixed duration suits.
    pub fn distance_scaling(self) -> Option<(f32, f32)> {
        match self {
            // Sixteen rows is roughly the distance past which a viewer stops reading the
            // move as "the region stepped" and starts reading it as "the region travelled".
            // Below it the region should arrive; at or above it, it should be seen going.
            Self::SelectionChange => Some((0.045, 16.0)),
            // The lamp scales over the same sixteen rows as the selection region, and takes
            // longer at every distance. It is meant to be seen arriving *after* the ring
            // that is already there — a light swinging over to where focus went, rather
            // than a second thing moving in lockstep with the first, which reads as one
            // thicker object.
            Self::FocusLight => Some((0.070, 16.0)),
            // An arrival happens where the session already is; there is no distance to scale
            // by, only a moment to mark.
            Self::HoverFeedback
            | Self::PressFeedback
            | Self::DensityChange
            | Self::SummonsArrival => None,
        }
    }

    /// Duration in seconds with full motion enabled.
    ///
    /// For distance-scaled patterns, this is the maximum duration.
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
            // Half again the selection region's ceiling. The gap is the effect: the ring
            // lands, then the room catches up.
            Self::FocusLight => 0.180,
            // Longer than everything above, deliberately: this is the one pattern that plays
            // when the user did nothing, so it has to be legible from the corner of an eye,
            // and it plays exactly once so the length costs no ongoing attention.
            Self::SummonsArrival => 0.260,
        }
    }

    /// Transition curve for this pattern.
    pub fn curve(self) -> Curve {
        match self {
            Self::HoverFeedback
            | Self::PressFeedback
            | Self::SelectionChange
            | Self::DensityChange
            | Self::FocusLight
            | Self::SummonsArrival => Curve::EaseOut,
        }
    }

    /// What reduced motion does to it.
    pub fn kind(self) -> MotionKind {
        match self {
            // Hover and press are a colour wash. Nothing moves, so reduced motion may keep
            // them -- capped at the cross-fade budget, which they are already under.
            Self::HoverFeedback | Self::PressFeedback => MotionKind::Fade,
            // All three move something across the window, and all three become instant. The
            // lamp is `Layout` rather than `Fade` even though nothing in the *layout* moves:
            // what travels is a light, and a light sweeping across the whole window is more
            // of the large-area movement FR-029 exists for than a row height changing is.
            // The summons is `Layout` for the same argument one step further: nothing moves
            // at all, but an arrival's entire purpose is to catch an eye that was elsewhere,
            // and reduced motion is the request not to be caught that way. Instant, not a
            // shorter fade — the accent still appears, it just appears settled.
            Self::SelectionChange
            | Self::DensityChange
            | Self::FocusLight
            | Self::SummonsArrival => MotionKind::Layout,
        }
    }

    /// Resolve this pattern against a preference, at its full duration.
    ///
    /// For a pattern that scales with distance this is the longest move's plan. Prefer
    /// [`MotionPattern::plan_over`] wherever the distance is known, which for the selection
    /// morph is everywhere it is actually started.
    pub fn plan(self, preference: MotionPreference) -> MotionPlan {
        plan(preference, self.kind(), self.duration(), self.curve())
    }

    /// Resolve this pattern for a move of `distance`, in the pattern's own units.
    ///
    /// A pattern with no [`MotionPattern::distance_scaling`] ignores the argument rather than
    /// pretending to use it.
    pub fn plan_over(self, preference: MotionPreference, distance: f32) -> MotionPlan {
        let duration = match self.distance_scaling() {
            Some((shortest, span)) => duration_for(distance, span, shortest, self.duration()),
            None => self.duration(),
        };
        plan(preference, self.kind(), duration, self.curve())
    }
}

/// Scale duration from `shortest` to `longest` with the square root of distance.
///
/// Moves of at least `span` use `longest`. A non-finite or non-positive
/// `distance` or `span` returns `shortest`.
pub fn duration_for(distance: f32, span: f32, shortest: f32, longest: f32) -> f32 {
    // Spelled out rather than negated, because `!(x > 0.0)` and `x <= 0.0` differ on NaN and
    // the difference is the one that matters here.
    if distance.is_nan() || distance <= 0.0 || span.is_nan() || span <= 0.0 {
        return shortest;
    }
    let t = (distance / span).clamp(0.0, 1.0).sqrt();
    shortest + (longest - shortest) * t
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

/// Where the lit mode's focus lamp should hang this frame.
///
/// Deliberately **not** a [`SelectionDraw`] with a different name. The selection region has an
/// alpha because it is drawn and can fade; a lamp cannot fade, it can only be somewhere or
/// nowhere, and the somewhere is the whole content. Giving it an unread alpha would be the
/// half-configured light [`qs_gpu::scene::FocusLamp`] is a separate type from `Light` to prevent.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FocusLightDraw {
    /// The row whose slot the lamp is anchored to.
    pub row: u64,
    /// Vertical displacement from that row's top, in **rows**. Zero once settled, and zero at
    /// every instant under Reduce Motion; positive means the lamp is still above `row`, on
    /// its way down.
    pub offset_rows: f32,
}

/// Animated interaction and density state.
///
/// The frame loop checks this state to decide whether another frame is needed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InteractionMotion {
    preference: MotionPreference,
    hover: Phase<Option<u64>>,
    pressed: Phase<Option<u64>>,
    selected: Phase<Option<u64>>,
    /// Keyboard focus, tracked separately from `selected` because they are separate: the
    /// keyboard moves focus through rows without selecting them, and a lamp riding the
    /// selection would sit still while the thing it is supposed to be finding moves.
    focused: Phase<Option<u64>>,
    density: Option<DensityTransition>,
    /// Where the material cycle is, in turns. See [`InteractionMotion::phase`].
    cycle: f32,
}

impl InteractionMotion {
    pub fn new(preference: MotionPreference) -> Self {
        Self {
            preference,
            hover: Phase::settled(None),
            pressed: Phase::settled(None),
            selected: Phase::settled(None),
            focused: Phase::settled(None),
            density: None,
            cycle: 0.0,
        }
    }

    pub fn preference(&self) -> MotionPreference {
        self.preference
    }

    /// Bring the animated state in line with the interaction state. Idempotent: calling this
    /// every frame with unchanged input starts nothing.
    /// The morph follows [`Selection::morph_target`](crate::selection::Selection::morph_target)
    /// rather than the selection itself: a multiple selection has no single region to move,
    /// and the renderer draws its rows directly instead.
    pub fn sync(&mut self, interaction: Interaction<'_>) {
        self.set_hovered(interaction.hovered);
        self.set_pressed(interaction.pressed);
        self.set_selected(interaction.selection.morph_target());
        self.set_focused(interaction.focused);
    }

    pub fn set_hovered(&mut self, row: Option<u64>) {
        let plan = MotionPattern::HoverFeedback.plan(self.preference);
        self.hover.set(row, plan);
    }

    /// Use 40 milliseconds for a press and 80 milliseconds for release.
    pub fn set_pressed(&mut self, row: Option<u64>) {
        let pattern = match row {
            Some(_) => MotionPattern::PressFeedback,
            None => MotionPattern::HoverFeedback,
        };
        let plan = pattern.plan(self.preference);
        self.pressed.set(row, plan);
    }

    /// The one place the selection morph's distance is known, which is why the duration is
    /// decided here rather than read off the pattern.
    ///
    /// The distance is measured from where the region is currently *heading*, not from where
    /// it started: hold an arrow key and each press restarts the morph from the row the last
    /// one was aimed at, so a run of single-row steps stays a run of short, snappy moves
    /// instead of the second press inheriting the first one's length.
    ///
    /// Appearing or disappearing has no distance -- it is a fade in place -- so it takes the
    /// floor, which is also the shortest thing this pattern can do.
    pub fn set_selected(&mut self, row: Option<u64>) {
        let distance = match (self.selected.to(), row) {
            (Some(from), Some(to)) => from.abs_diff(to) as f32,
            _ => 0.0,
        };
        let plan = MotionPattern::SelectionChange.plan_over(self.preference, distance);
        self.selected.set(row, plan);
    }

    /// Aim the focus lamp at `row`, or extinguish it when nothing has focus.
    ///
    /// The distance is measured from where the lamp is *heading*, exactly as
    /// [`InteractionMotion::set_selected`] does and for the same reason: holding an arrow key
    /// must not make each successive move inherit the last one's length.
    pub fn set_focused(&mut self, row: Option<u64>) {
        let distance = match (self.focused.to(), row) {
            (Some(from), Some(to)) => from.abs_diff(to) as f32,
            // Arriving or leaving is not a journey — the lamp comes up or goes out where it
            // is — so it takes the floor, which is the shortest thing the pattern can do.
            _ => 0.0,
        };
        let plan = MotionPattern::FocusLight.plan_over(self.preference, distance);
        self.focused.set(row, plan);
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
        // The material cycle, and the one line in this function whose result is deliberately
        // *not* in the expression returned below.
        //
        // That exclusion is the whole safety argument. A phase is a position, so something
        // has to move it, and anything that both moves it and reports "still running" would
        // make every frame an animating frame -- which is precisely the defect
        // `frame-pacing-bound` was opened to remove. Here the cycle rides wakefulness that
        // some other animation already bought and can never buy any of its own: for the
        // phase to hold the loop open, someone would have to add `cycle` to a boolean it is
        // not part of, which is a visible edit rather than an oversight.
        //
        // It also means `dt` only arrives while the loop is awake, so a travelling highlight
        // resumes where it stopped instead of jumping by the length of the sleep. Reading a
        // wall clock in `phase()` would have been the same number of lines and would have
        // shipped that jump.
        self.cycle = (self.cycle + dt.max(0.0) / CYCLE_SECONDS).rem_euclid(1.0);

        let hover = self.hover.advance(dt);
        let pressed = self.pressed.advance(dt);
        let selected = self.selected.advance(dt);
        let focused = self.focused.advance(dt);
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
        hover || pressed || selected || focused || density
    }

    pub fn is_animating(&self) -> bool {
        self.hover.is_animating()
            || self.pressed.is_animating()
            || self.selected.is_animating()
            || self.focused.is_animating()
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

    /// Selection movement intensity in `0.0..=1.0`.
    ///
    /// Starts at one and falls with animation progress. Returns zero when
    /// selection is still, including when reduced motion makes the change instant.
    #[must_use]
    pub fn selection_swell(&self) -> f32 {
        if !self.selected.is_animating() {
            return 0.0;
        }
        (1.0 - self.selected.progress()).clamp(0.0, 1.0)
    }

    /// Material-cycle position in turns, `0.0..1.0`.
    ///
    /// Reduced motion returns zero, placing the layer at its configured angle.
    #[must_use]
    pub fn phase(&self) -> f32 {
        match self.preference {
            MotionPreference::Reduced => PINNED_PHASE,
            MotionPreference::Full => self.cycle,
        }
    }

    /// What to hand [`crate::tokens::Tokens::paint_driven`] this frame.
    ///
    /// One call rather than two so a surface cannot pick up the swell and forget the phase,
    /// which would show as a material that animates its opacity and not its geometry — the
    /// kind of half-wired effect that reads as a rendering bug.
    #[must_use]
    pub fn drive(&self) -> Drive {
        Drive::new(self.selection_swell(), self.phase())
    }

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

    /// Focus-lamp position for this frame, or `None` when nothing has focus.
    ///
    /// Reduced motion places the lamp at the focused row immediately, with
    /// zero travel offset.
    #[must_use]
    pub fn focus_light_draw(&self) -> Option<FocusLightDraw> {
        if !self.focused.is_animating() {
            return self.focused.to().map(|row| FocusLightDraw {
                row,
                offset_rows: 0.0,
            });
        }

        let progress = self.focused.progress();
        match (self.focused.from(), self.focused.to()) {
            (Some(from), Some(to)) => Some(FocusLightDraw {
                row: to,
                offset_rows: (1.0 - progress) * ((from as i64) - (to as i64)) as f32,
            }),
            // Coming up: the lamp is already at its destination, and what ramps is its
            // strength, which the scene builder scales by `progress`. Going out: it stays
            // where it was while the strength ramps down. Both are handled by
            // `focus_light_gain` rather than by moving something.
            (None, Some(to)) => Some(FocusLightDraw {
                row: to,
                offset_rows: 0.0,
            }),
            (Some(from), None) => Some(FocusLightDraw {
                row: from,
                offset_rows: 0.0,
            }),
            (None, None) => None,
        }
    }

    /// How strongly the focus lamp burns this frame, `0.0..=1.0`.
    ///
    /// One on a settled lamp. Ramping while it comes up or goes out, so focus arriving is a
    /// light coming on rather than a light appearing at full strength — which is the same
    /// snap the mode exists to replace, one level in. A lamp merely *travelling* burns at
    /// full strength throughout: it is the same light, in a new place.
    #[must_use]
    pub fn focus_light_gain(&self) -> f32 {
        if !self.focused.is_animating() {
            return f32::from(u8::from(self.focused.to().is_some()));
        }
        match (self.focused.from(), self.focused.to()) {
            (Some(_), Some(_)) => 1.0,
            (None, Some(_)) => self.focused.progress(),
            (Some(_), None) => 1.0 - self.focused.progress(),
            (None, None) => 0.0,
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

    // -- distance-scaled duration ---------------------------------------------------------

    #[test]
    fn a_duration_grows_with_distance_and_stops_growing_at_the_span() {
        let (shortest, longest, span) = (0.045, 0.120, 16.0);
        let at = |d: f32| duration_for(d, span, shortest, longest);

        assert!(
            (at(0.0) - shortest).abs() < 1e-6,
            "a standing start is the floor"
        );
        assert!(
            (at(span) - longest).abs() < 1e-6,
            "a span-long move is the ceiling"
        );
        assert!(
            (at(span * 4.0) - longest).abs() < 1e-6,
            "a move four spans long still takes one span's time -- past the span the region \
             cross-fades rather than travelling, so there is nothing longer to pay for"
        );

        // Monotonic, with no step in it: a duration that jumped would make two nearly equal
        // moves feel unequal, which is more noticeable than either being slightly off.
        let mut previous = at(0.0);
        for i in 1..=64 {
            let next = at(span * i as f32 / 64.0);
            assert!(next >= previous - 1e-6, "duration fell between samples");
            previous = next;
        }

        // The square root, stated as the property that distinguishes it from the linear
        // alternative: four times the distance is *half* the extra time, not four times it.
        let quarter = at(span * 0.25) - shortest;
        let full = longest - shortest;
        assert!(
            (quarter - full * 0.5).abs() < 1e-6,
            "a quarter-span move spent {quarter:.4}s of the range rather than half of it -- \
             the growth is not sqrt, so either short moves crawl or long ones do"
        );
    }

    #[test]
    fn a_degenerate_distance_is_the_floor_and_never_a_nan() {
        // A NaN duration makes `Animation::advance` compare false forever, the ticket never
        // retires, and the frame loop never sleeps -- an SC-003 failure arriving through the
        // motion table. Cheaper to make impossible than to detect.
        for bad in [0.0, -3.0, f32::NAN, f32::INFINITY] {
            for span in [0.0, -1.0, f32::NAN, 16.0] {
                let d = duration_for(bad, span, 0.045, 0.120);
                assert!(d.is_finite(), "distance {bad} over span {span} gave {d}");
                assert!((0.045 - 1e-6..=0.120 + 1e-6).contains(&d));
            }
        }
    }

    /// Frames at 120 Hz for a morph to settle.
    fn frames_to_settle(motion: &mut InteractionMotion) -> u32 {
        let mut frames = 0;
        while motion.advance(1.0 / 120.0) {
            frames += 1;
            assert!(frames < 10_000, "the morph never retired");
        }
        frames
    }

    #[test]
    fn a_one_row_move_finishes_much_sooner_than_a_screen_long_one() {
        // The complaint this exists to answer: at one fixed duration a single-row step spends
        // the same time as a move across the viewport, and the short one reads as lag. What
        // is asserted is the *ratio*, not either number -- the two constants may be tuned,
        // and the thing that must not come back is them being equal.
        let mut short = InteractionMotion::new(MotionPreference::Full);
        short.set_selected(Some(0));
        short.set_selected(Some(1));
        let short_frames = frames_to_settle(&mut short);

        let mut long = InteractionMotion::new(MotionPreference::Full);
        long.set_selected(Some(0));
        long.set_selected(Some(40));
        let long_frames = frames_to_settle(&mut long);

        assert!(
            short_frames * 3 < long_frames * 2,
            "a one-row step took {short_frames} frames against {long_frames} for a forty-row \
             one: the duration is barely reading the distance"
        );
        assert!(
            short_frames > 0,
            "a one-row step became instant, which is a different bug -- the region would \
             teleport and the morph would be pointless"
        );
    }

    #[test]
    fn each_press_of_a_held_arrow_key_starts_a_fresh_short_move() {
        // Distance is measured from where the region is *heading*, not from where it started.
        // Measured from the origin instead, the second press of a held arrow key would be a
        // two-row move, the third a three-row one, and holding the key would make each step
        // slower than the last while the key repeat stayed constant -- which reads as the
        // list bogging down.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.set_selected(Some(0));
        motion.set_selected(Some(1));
        let first = frames_to_settle(&mut motion);

        motion.set_selected(Some(2));
        // Interrupted a third of the way through, exactly as a key repeat would.
        motion.advance(1.0 / 120.0);
        motion.set_selected(Some(3));
        let interrupted = frames_to_settle(&mut motion);

        assert!(
            interrupted <= first + 1,
            "an interrupted single-row step took {interrupted} frames against {first} for an \
             uninterrupted one: the distance is being measured from the wrong end"
        );
    }

    #[test]
    fn a_fade_ignores_the_distance_rather_than_inventing_one() {
        // Hover and press are opacity. There is no distance, and scaling their duration by a
        // row count would be a number with no meaning behind it.
        for pattern in [MotionPattern::HoverFeedback, MotionPattern::PressFeedback] {
            assert_eq!(pattern.distance_scaling(), None, "{pattern:?}");
            for distance in [0.0, 1.0, 500.0] {
                let scaled = pattern
                    .plan_over(MotionPreference::Full, distance)
                    .duration();
                assert!(
                    (scaled - pattern.duration()).abs() < 1e-6,
                    "{pattern:?} changed length with a distance it does not have"
                );
            }
        }
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
        // Patterns that are NOT rows of UXDD 10.3, each with the document that owns it.
        // Listed rather than subtracted, so the count assertion below keeps doing its job:
        // every pattern is accounted for by exactly one design source, and adding a fifth
        // without saying which one it came from fails here instead of quietly joining a
        // table it is not in.
        let elsewhere = [
            // spec 002 US3, the lit mode's focus lamp.
            (MotionPattern::FocusLight, 0.180, MotionKind::Layout),
            // roadmap summons-arrival, the Sessions workstream's needs-you arrival.
            (MotionPattern::SummonsArrival, 0.260, MotionKind::Layout),
        ];
        assert_eq!(expected.len() + elsewhere.len(), MotionPattern::ALL.len());

        for (pattern, duration, kind) in expected.into_iter().chain(elsewhere) {
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
        let mut selection = crate::selection::Selection::new();
        selection.select_only(4);
        let interaction = Interaction {
            hovered: Some(4),
            focused: Some(4),
            selection: &selection,
            pressed: None,
            ..Interaction::default()
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

    #[test]
    fn the_selection_swell_peaks_on_departure_and_is_zero_under_reduce_motion() {
        // The drive an animated material reads. Two properties, and the second is an
        // accessibility requirement rather than a preference: UXDD 10.3 says Reduce Motion
        // makes every pattern instant, and a halo that still flares would be the setting
        // ignored in the one channel nobody thought to check.
        let mut full = InteractionMotion::new(MotionPreference::Full);
        // Run the first selection's own fade-in to rest. Appearing from nothing is a
        // selection change too, and it swells like one -- which is correct, and is why the
        // resting assertion below has to come after it rather than before.
        full.set_selected(Some(0));
        while full.advance(FRAME) {}
        assert_eq!(full.selection_swell(), 0.0, "a settled list must not swell");

        full.set_selected(Some(9));
        let departure = full.selection_swell();
        assert!(
            departure > 0.9,
            "the flare is on the departure: {departure}"
        );

        // Monotonically down, and exactly zero once the ticket retires -- a drive that
        // settled at 0.02 would hold a material a hair brighter than its token says forever.
        let mut previous = departure;
        while full.advance(FRAME) {
            let now = full.selection_swell();
            assert!(
                now <= previous + 1e-6,
                "the swell rose again: {previous} → {now}"
            );
            previous = now;
        }
        assert_eq!(full.selection_swell(), 0.0);

        // Reduce Motion needs no branch anywhere: the plan is instant, so no ticket opens,
        // so the drive is zero from the first frame and the material is simply a still one.
        let mut reduced = InteractionMotion::new(MotionPreference::Reduced);
        reduced.set_selected(Some(0));
        reduced.set_selected(Some(9));
        assert!(!reduced.is_animating());
        assert_eq!(reduced.selection_swell(), 0.0);
    }

    #[test]
    fn an_animation_takes_the_same_wall_clock_time_at_any_frame_rate() {
        // The property `frame-pacing-bound` rests on, and the reason pacing is allowed to
        // drop frames at all: the clock is advance-to-advance, so a loop running at a
        // quarter of the rate hands over four times the `dt` and the plan finishes at the
        // same instant. Without this, pacing would not be a saving -- it would be a
        // slowdown, and one that only shows up as "the interface feels sluggish now".
        //
        // The rates are the ones actually measured on the machine this was written on: an
        // unpaced loop reached 3,937 Hz and the paced loop reaches 119, so the ratio under
        // test is the real one rather than a round number.
        let elapsed_to_settle = |hz: f32| -> f32 {
            let mut motion = InteractionMotion::new(MotionPreference::Full);
            motion.set_selected(Some(0));
            let _ = motion.advance(0.0);
            motion.set_selected(Some(8));
            assert!(motion.is_animating(), "nothing to measure");

            let dt = 1.0 / hz;
            let mut elapsed = 0.0;
            let mut frames = 0u32;
            while motion.advance(dt) {
                elapsed += dt;
                frames += 1;
                assert!(frames < 100_000, "an animation never retired at {hz} Hz");
            }
            elapsed + dt
        };

        let fast = elapsed_to_settle(3937.0);
        let paced = elapsed_to_settle(119.0);

        // Within one frame of the slower rate, which is the granularity the slower rate can
        // express at all. A pacing bug would not be off by a frame -- it would be off by the
        // ratio, 33x here.
        let tolerance = 1.0 / 119.0;
        assert!(
            (fast - paced).abs() <= tolerance,
            "the same animation took {fast:.4}s at 3937 Hz and {paced:.4}s at 119 Hz: \
             pacing the loop changed how long the animation lasts, which means `dt` is not \
             advance-to-advance somewhere"
        );
    }
    #[test]
    fn the_phase_advances_without_ever_holding_the_frame_loop_open() {
        // The hazard the chunk named: an always-advancing phase makes every frame an
        // animating frame, which is exactly what `frame-pacing-bound` was opened to remove.
        //
        // Nothing here is animating -- no hover, no press, no selection, no density -- so
        // `advance` must report false however long it is given, while the phase still moves.
        // Those two facts together are the whole safety argument, and asserting only one of
        // them would pass with the feature deleted.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        assert!(!motion.is_animating());

        let before = motion.phase();
        let still_running = motion.advance(CYCLE_SECONDS / 4.0);

        assert!(
            !still_running,
            "the cycle reported the loop as still animating, so the application would never              sleep again"
        );
        assert!(!motion.is_animating(), "the cycle opened a ticket");
        assert!(
            (motion.phase() - before - 0.25).abs() < 1e-5,
            "a quarter of a cycle moved the phase to {}",
            motion.phase()
        );
    }

    #[test]
    fn the_phase_wraps_rather_than_running_away() {
        // A cycle that accumulated would drift out of `0..1` and, in an f32, eventually stop
        // resolving small steps at all -- the same class of failure as an absolute scroll
        // offset at a million rows, and just as invisible until it is enormous.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        for _ in 0..1000 {
            motion.advance(CYCLE_SECONDS * 0.7);
            assert!(
                (0.0..1.0).contains(&motion.phase()),
                "the phase left its range: {}",
                motion.phase()
            );
        }
    }

    #[test]
    fn reduce_motion_pins_the_phase_rather_than_freezing_it() {
        // The distinction UXDD 10.3 needs for a cyclic effect, and it is not pedantry.
        //
        // Freezing means "stop advancing and keep what you had", so the still frame depends
        // on when the preference was read -- a different picture on every machine, none of
        // them drawn by anyone. Pinning means a stated value, and a layer's authored angle is
        // its angle at phase zero, so the reduced picture is the one in design/tokens.json.
        //
        // The cycle is deliberately still advanced underneath: reduced motion is a
        // presentation choice, not a broken clock, and a preference that can be turned back
        // off must not resume from a stale value.
        let mut reduced = InteractionMotion::new(MotionPreference::Reduced);
        let mut full = InteractionMotion::new(MotionPreference::Full);
        for _ in 0..5 {
            reduced.advance(CYCLE_SECONDS / 8.0);
            full.advance(CYCLE_SECONDS / 8.0);
        }

        assert_eq!(
            reduced.phase(),
            PINNED_PHASE,
            "a reduced-motion frame is showing whatever the cycle happened to reach"
        );
        assert!(
            full.phase() != PINNED_PHASE,
            "the comparison arm never moved, so this test cannot fail"
        );
        assert_eq!(
            reduced.drive().phase,
            PINNED_PHASE,
            "the pin is applied by `phase` and skipped by `drive`, so a call site can reach              around it"
        );
    }

    #[test]
    fn the_focus_lamp_arrives_without_travelling_under_reduced_motion() {
        // T061 / FR-029, and the two halves are separate claims.
        //
        // FIRST: under Reduce Motion the lamp never has a non-zero offset at ANY instant. It
        // is not enough that it ends up in the right place -- a light sweeping across the
        // window and then stopping there is exactly the large-area movement the preference
        // exists to remove, and a test that only checked the final frame would pass on it.
        let mut reduced = InteractionMotion::new(MotionPreference::Reduced);
        reduced.set_focused(Some(3));
        reduced.set_focused(Some(40));
        assert_eq!(
            reduced.focus_light_draw(),
            Some(FocusLightDraw {
                row: 40,
                offset_rows: 0.0,
            }),
            "the lamp did not arrive at the focused row instantly"
        );
        assert!(
            !reduced.is_animating(),
            "a reduced-motion lamp opened an animation ticket, which SC-003 pays for"
        );
        for _ in 0..12 {
            reduced.advance(0.016);
            let draw = reduced.focus_light_draw().unwrap();
            assert_eq!(
                draw.offset_rows, 0.0,
                "the lamp travelled under Reduce Motion, at row {} offset {}",
                draw.row, draw.offset_rows
            );
            assert_eq!(draw.row, 40);
        }

        // SECOND: the comparison arm. A test whose reduced case is right because NOTHING
        // moves in either case is not evidence about reduced motion -- it is evidence the
        // feature is missing. This is the half that would have caught a lamp wired to the
        // wrong phase.
        let mut full = InteractionMotion::new(MotionPreference::Full);
        full.set_focused(Some(3));
        full.set_focused(Some(40));
        let travelling = full.focus_light_draw().unwrap();
        assert_eq!(travelling.row, 40);
        assert!(
            travelling.offset_rows.abs() > 0.0,
            "with full motion the lamp arrived instantly too, so the reduced arm above \
             asserts nothing"
        );
        assert!(full.is_animating());
    }

    #[test]
    fn the_lamp_rests_where_the_layout_put_it_and_not_where_motion_stopped() {
        // FR-029's second clause, which is the one that is easy to satisfy wrongly: the
        // resting position must be one somebody chose. Turn the preference on mid-flight and
        // the lamp must snap to the focused row -- not freeze at the fraction of the journey
        // it happened to have covered, which is a position nobody authored and which differs
        // by machine depending on when the setting was read.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.set_focused(Some(0));
        motion.set_focused(Some(20));
        motion.advance(0.030);
        let mid = motion.focus_light_draw().unwrap();
        assert!(
            mid.offset_rows.abs() > 0.0,
            "the lamp was already settled, so there is no interrupted journey to test"
        );

        let mut reduced = InteractionMotion::new(MotionPreference::Reduced);
        reduced.set_focused(Some(20));
        assert_eq!(
            reduced.focus_light_draw().unwrap().offset_rows,
            0.0,
            "the reduced lamp rests somewhere other than the focused row"
        );
    }

    #[test]
    fn the_lamp_goes_out_and_comes_up_rather_than_appearing() {
        // The gain is a separate question from the position, and collapsing them is how a
        // lamp ends up teleporting: focus arriving from nothing has no distance to travel,
        // so if strength did not ramp there would be nothing to see but a step change in
        // every shadow in the window.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        assert_eq!(motion.focus_light_gain(), 0.0, "a lamp burns with no focus");

        motion.set_focused(Some(5));
        assert!(
            motion.focus_light_gain() < 1.0,
            "the lamp came up at full strength instead of ramping"
        );
        for _ in 0..24 {
            motion.advance(0.016);
        }
        assert_eq!(motion.focus_light_gain(), 1.0);
        assert!(!motion.is_animating(), "the lamp never settled");

        // Travelling is the same light in a new place, so it does not dim on the way.
        motion.set_focused(Some(9));
        motion.advance(0.016);
        assert_eq!(
            motion.focus_light_gain(),
            1.0,
            "the lamp dimmed while merely moving"
        );

        motion.set_focused(None);
        motion.advance(0.016);
        let going = motion.focus_light_gain();
        assert!(
            going > 0.0 && going < 1.0,
            "the lamp did not ramp down on the way out, it was switched off: {going}"
        );
    }

    #[test]
    fn the_lamp_and_the_selection_are_separate_lights_to_aim() {
        // The keyboard moves focus through rows without selecting them. A lamp riding
        // `selected` would sit still while the thing it exists to find walks away, and the
        // symptom is a mode that looks broken only for keyboard users -- who are the people
        // US3 is for.
        let mut motion = InteractionMotion::new(MotionPreference::Full);
        motion.set_selected(Some(2));
        motion.set_focused(Some(2));
        for _ in 0..24 {
            motion.advance(0.016);
        }

        motion.set_focused(Some(11));
        assert_eq!(
            motion.focus_light_draw().map(|d| d.row),
            Some(11),
            "the lamp did not follow focus"
        );
        assert_eq!(
            motion.selection_draw(40).map(|d| d.row),
            Some(2),
            "moving focus moved the selection region, which is a different thing"
        );
    }
}
