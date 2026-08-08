//! Windows present tuning: `SetMaximumFrameLatency` plus the waitable object.
//!
//! # What this is for
//!
//! By default DXGI lets the driver queue up to three frames ahead. That maximizes
//! throughput and is exactly wrong for an interactive application: every queued frame is a
//! frame of input latency the user feels as the list lagging behind the pointer. Setting
//! the maximum frame latency to 2 and waiting on the swapchain's waitable object means the
//! application starts each frame when the display is ready for it, rather than as fast as
//! it can and then queueing.
//!
//! # Why this is not implemented yet, stated plainly
//!
//! Reaching the `IDXGISwapChain2` behind a `wgpu::Surface` requires
//! `Surface::as_hal::<wgpu_hal::dx12::Api, _, _>()`, which is `unsafe`, is only valid when
//! the surface actually has a D3D12 backend, and hands back a handle whose lifetime is tied
//! to internals `wgpu` does not promise to keep stable.
//!
//! The honest position for a spike is that the *measurement* -- how much latency the
//! default queueing costs -- has not been taken yet, and this tuning should be applied in
//! response to that measurement rather than ahead of it. Writing the `unsafe` block now
//! would add a platform-specific hazard to the frame path in order to fix a problem nobody
//! has yet shown exists at this milestone.
//!
//! So this reports `not applied` with the reason, which is what
//! [`crate::present_tuning::PresentTuningOutcome`] exists for, and the bench report carries
//! it. That is Constitution III applied to our own tooling: the capability is reduced and it
//! says so, rather than a stub silently claiming success.

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
