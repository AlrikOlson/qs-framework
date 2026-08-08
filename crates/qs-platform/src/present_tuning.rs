//! Present tuning, behind a trait with a no-op default.
//!
//! Research R4 specifies exactly two platform tunings and nothing else:
//!
//! * **Windows**: `IDXGISwapChain2::SetMaximumFrameLatency` plus the waitable object.
//! * **macOS**: `CAMetalLayer.displaySyncEnabled` and display-link phase alignment.
//!
//! Both reach past `wgpu` into `wgpu-hal`, which is the only place in this codebase that
//! does. The trait exists so that reaching is contained: `qs` calls
//! [`PresentTuning::configure`] and does not know whether anything happened.
//!
//! # The default is a no-op, deliberately
//!
//! Linux has no equivalent tuning to apply -- Wayland's presentation model already does
//! what the Windows waitable object is for -- and a platform where the tuning is
//! unavailable must behave identically to one where it is unimplemented. Making the default
//! a no-op rather than an error means a new platform works on day one and gets tuned later,
//! which is the opposite of the usual arrangement where an unimplemented platform panics.
//!
//! # A correction worth recording (research R4)
//!
//! SDD §7.1 names `CADisplayLink` for macOS phase alignment. `CADisplayLink` is available
//! for AppKit views only from **macOS 14**, while SDD §1.5 sets the floor at **macOS 13
//! Ventura**. M0 therefore uses `CVDisplayLink` on 13 and `CADisplayLink` on 14+. See
//! `docs/adr/009-macos-display-link.md`.

/// What the application asks the platform to do about presentation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PresentConfig {
    /// How many frames the driver may queue ahead. Lower is more responsive and more
    /// likely to stall; 2 is the value that keeps a 120 Hz display fed without adding a
    /// frame of input latency.
    pub max_frame_latency: u32,
    /// Whether to align frame production to the display's refresh phase.
    pub align_to_display_link: bool,
}

impl Default for PresentConfig {
    fn default() -> Self {
        Self {
            max_frame_latency: 2,
            align_to_display_link: true,
        }
    }
}

/// What the platform actually did.
///
/// Returned rather than discarded because Constitution III applies to our own telemetry:
/// a run where the tuning silently did not apply is not comparable to one where it did,
/// and the bench report should be able to say so.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct PresentTuningOutcome {
    pub frame_latency_applied: bool,
    pub display_link_applied: bool,
    /// Why a tuning did not apply, when it did not.
    pub notes: Vec<String>,
}

impl PresentTuningOutcome {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            frame_latency_applied: false,
            display_link_applied: false,
            notes: vec![reason.into()],
        }
    }

    /// A one-line summary, including any notes.
    ///
    /// The notes are appended rather than shown only when nothing applied. A *partial*
    /// application is the case that most needs explaining -- "frame latency applied,
    /// display link not applied" invites the question "why not", and the answer should not
    /// require reading the source.
    pub fn describe(&self) -> String {
        let status = if self.frame_latency_applied || self.display_link_applied {
            format!(
                "frame latency: {}, display link: {}",
                yes_no(self.frame_latency_applied),
                yes_no(self.display_link_applied)
            )
        } else {
            "no tuning applied".to_string()
        };

        if self.notes.is_empty() {
            status
        } else {
            format!("{status} ({})", self.notes.join("; "))
        }
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "applied" } else { "not applied" }
}

pub trait PresentTuning {
    fn configure(
        &self,
        surface: &wgpu::Surface<'static>,
        config: PresentConfig,
    ) -> PresentTuningOutcome;
}

/// The default on every platform without a specific implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopPresentTuning;

impl PresentTuning for NoopPresentTuning {
    fn configure(
        &self,
        _surface: &wgpu::Surface<'static>,
        _config: PresentConfig,
    ) -> PresentTuningOutcome {
        PresentTuningOutcome::unavailable(
            "no present tuning is defined for this platform; the surface's present mode \
             governs pacing on its own",
        )
    }
}

/// The tuning implementation for the current platform.
pub fn for_platform() -> Box<dyn PresentTuning> {
    #[cfg(target_os = "windows")]
    {
        Box::new(crate::win::present::WindowsPresentTuning)
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(crate::mac::present::MacPresentTuning)
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        Box::new(NoopPresentTuning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_config_matches_the_research_note() {
        let config = PresentConfig::default();
        assert_eq!(config.max_frame_latency, 2);
        assert!(config.align_to_display_link);
    }

    #[test]
    fn an_unavailable_tuning_explains_itself_rather_than_reporting_nothing() {
        // Constitution III: a run where the tuning did not apply is not comparable to one
        // where it did, so "nothing happened" has to be distinguishable from "we did not
        // look".
        let outcome = PresentTuningOutcome::unavailable("no waitable object on this driver");
        assert!(!outcome.frame_latency_applied);
        assert!(outcome.describe().contains("waitable object"));
    }

    #[test]
    fn an_outcome_with_no_notes_still_describes_itself() {
        let text = PresentTuningOutcome::default().describe();
        assert!(!text.is_empty());
        assert!(text.contains("no tuning applied"), "{text}");
    }

    #[test]
    fn a_partial_application_is_reported_as_partial() {
        let outcome = PresentTuningOutcome {
            frame_latency_applied: true,
            display_link_applied: false,
            notes: Vec::new(),
        };
        let text = outcome.describe();
        assert!(
            text.contains("applied") && text.contains("not applied"),
            "{text}"
        );
    }
}
