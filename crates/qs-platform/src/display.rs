//! Display enumeration: per-monitor DPI scale, refresh rate, and change events.
//!
//! # Why refresh rate is an `Option`
//!
//! Not every platform reports it. Wayland does, X11 sometimes does, and a virtual display
//! in a VM frequently reports something meaningless. The frame budget is derived from the
//! refresh rate (8.33 ms *is* one frame at 120 Hz), so a wrong value is worse than no
//! value: it would produce a budget the harness gates on and nobody can explain.

use crate::PhysicalSize;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DisplayId(pub u32);

#[derive(Clone, PartialEq, Debug)]
pub struct DisplayInfo {
    pub id: DisplayId,
    pub name: String,
    pub size: PhysicalSize,
    /// Device pixel ratio. Always positive and finite; a platform reporting otherwise is
    /// normalized to 1.0 by the backend rather than propagating a divide-by-zero upward.
    pub scale_factor: f64,
    /// `None` when the platform does not report one. See the module docs.
    pub refresh_hz: Option<f64>,
}

impl DisplayInfo {
    /// The frame budget this display implies, in milliseconds.
    ///
    /// `None` when the refresh rate is unknown, which is the honest answer: the alternative
    /// is assuming 60 Hz and gating a 120 Hz machine against a budget twice as generous as
    /// it should be.
    pub fn frame_budget_ms(&self) -> Option<f64> {
        self.refresh_hz.filter(|hz| *hz > 0.0).map(|hz| 1000.0 / hz)
    }
}

/// Display enumeration and change notification.
pub trait PlatformDisplay {
    fn displays(&self) -> Vec<DisplayInfo>;
    fn primary(&self) -> Option<DisplayInfo>;
}

/// What changed about the display configuration.
///
/// A DPI change must re-lay-out at the *exact* new physical resolution (FR-014). Treating
/// it as a resize would work by accident on most platforms and produce half-pixel text on
/// the ones where the scale changes without the physical size doing so.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DisplayChange {
    /// The window moved to a monitor with a different scale, or the user changed the
    /// current monitor's scale.
    ScaleChanged { from: f64, to: f64 },
    /// The window's physical size changed.
    Resized { to: PhysicalSize },
    /// A monitor was attached or removed.
    ConfigurationChanged,
}

/// Normalize a platform-reported scale factor.
///
/// Every entry point that accepts a scale from the OS runs it through here. A zero or NaN
/// scale has been observed from remote-desktop sessions and from X11 servers mid-
/// reconfiguration, and it turns into a division by zero three crates away from its source.
pub fn sane_scale_factor(reported: f64) -> f64 {
    if reported.is_finite() && reported > 0.0 {
        reported.clamp(0.5, 8.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display(refresh: Option<f64>) -> DisplayInfo {
        DisplayInfo {
            id: DisplayId(0),
            name: "test".into(),
            size: PhysicalSize {
                width: 1920,
                height: 1080,
            },
            scale_factor: 1.0,
            refresh_hz: refresh,
        }
    }

    #[test]
    fn the_frame_budget_follows_the_refresh_rate() {
        // 8.33 ms is one frame at 120 Hz -- the number SC-001 gates on.
        let budget = display(Some(120.0)).frame_budget_ms().unwrap_or_default();
        assert!((budget - 8.3333).abs() < 0.001);
        let budget = display(Some(60.0)).frame_budget_ms().unwrap_or_default();
        assert!((budget - 16.667).abs() < 0.001);
    }

    #[test]
    fn an_unknown_refresh_rate_yields_no_budget_rather_than_assuming_sixty() {
        assert_eq!(display(None).frame_budget_ms(), None);
        assert_eq!(display(Some(0.0)).frame_budget_ms(), None);
    }

    #[test]
    fn a_broken_scale_factor_is_normalized_to_one() {
        assert_eq!(sane_scale_factor(0.0), 1.0);
        assert_eq!(sane_scale_factor(-2.0), 1.0);
        assert_eq!(sane_scale_factor(f64::NAN), 1.0);
        assert_eq!(sane_scale_factor(f64::INFINITY), 1.0);
        assert_eq!(sane_scale_factor(1.25), 1.25);
        assert_eq!(sane_scale_factor(100.0), 8.0);
    }
}
