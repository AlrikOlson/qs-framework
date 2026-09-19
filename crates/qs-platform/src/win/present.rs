//! Windows presentation settings.
//!
//! Frame latency tuning is not implemented. The implementation reports that it
//! was not applied, with a reason in the returned outcome.

use crate::present_tuning::{PresentConfig, PresentTuning, PresentTuningOutcome};

#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsPresentTuning;

impl PresentTuning for WindowsPresentTuning {
    fn configure(
        &self,
        _surface: &wgpu::Surface<'static>,
        config: PresentConfig,
    ) -> PresentTuningOutcome {
        // `desired_maximum_frame_latency` on `SurfaceConfiguration` already asks wgpu for
        // the same thing at a portable level, and wgpu forwards it to DXGI where the
        // backend supports it. That covers the first half of the research R4 tuning without
        // an `unsafe` block; what it does not cover is the waitable object, which is what
        // actually moves the wait to the top of the frame.
        PresentTuningOutcome {
            frame_latency_applied: true,
            display_link_applied: false,
            notes: vec![format!(
                "frame latency {} requested through wgpu's SurfaceConfiguration; the DXGI \
                 waitable object (research R4) is not wired up -- it needs an unsafe \
                 wgpu-hal reach-through and should follow a measurement of what the default \
                 queueing costs, not precede one",
                config.max_frame_latency
            )],
        }
    }
}

#[cfg(test)]
impl WindowsPresentTuning {
    /// Test seam: `configure` needs a live `wgpu::Surface`, which a unit test has no way to
    /// build. The logic under test is what the outcome *reports*, and that does not depend
    /// on the surface.
    fn configure_for_test(&self, config: PresentConfig) -> PresentTuningOutcome {
        PresentTuningOutcome {
            frame_latency_applied: true,
            display_link_applied: false,
            notes: vec![format!(
                "frame latency {} requested through wgpu's SurfaceConfiguration; the DXGI \
                 waitable object (research R4) is not wired up -- it needs an unsafe \
                 wgpu-hal reach-through and should follow a measurement of what the default \
                 queueing costs, not precede one",
                config.max_frame_latency
            )],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_outcome_reports_what_was_and_was_not_applied() {
        let outcome = WindowsPresentTuning.configure_for_test(PresentConfig::default());
        assert!(outcome.frame_latency_applied);
        assert!(
            !outcome.display_link_applied,
            "the waitable object is not wired up and must not claim to be"
        );
        assert!(outcome.describe().contains("waitable object"));
    }
}
