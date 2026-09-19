//! macOS presentation settings.
//!
//! [`display_link_api`] selects `CVDisplayLink` for macOS 13 and `CADisplayLink`
//! for macOS 14 or later. Presentation tuning is not connected to the Metal
//! surface yet; the implementation reports that it was not applied.

use crate::present_tuning::{PresentConfig, PresentTuning, PresentTuningOutcome};

/// Which display-link API this OS version offers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DisplayLinkApi {
    /// macOS 13. Deprecated as of 15 but functional, and the only option on the floor.
    CoreVideo,
    /// macOS 14+.
    CoreAnimation,
}

/// Choose the display-link API for a major macOS version.
///
/// Pure and testable, which is the point: the version boundary is the part of this that is
/// easy to get wrong by one, and it can be checked without a Mac.
pub fn display_link_api(major_version: u32) -> DisplayLinkApi {
    if major_version >= 14 {
        DisplayLinkApi::CoreAnimation
    } else {
        DisplayLinkApi::CoreVideo
    }
}

/// Read the running macOS major version.
///
/// Returns `None` when it cannot be determined, and the caller then assumes the floor --
/// `CVDisplayLink` works on every supported version, so guessing low is safe and guessing
/// high is a crash on Ventura.
pub fn macos_major_version() -> Option<u32> {
    // `sw_vers` rather than an Objective-C runtime call: this runs once at startup, not on
    // the frame path, and it avoids linking a bindings crate for one integer.
    let output = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.trim().split('.').next()?.parse().ok()
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MacPresentTuning;

impl PresentTuning for MacPresentTuning {
    fn configure(
        &self,
        _surface: &wgpu::Surface<'static>,
        config: PresentConfig,
    ) -> PresentTuningOutcome {
        let version = macos_major_version();
        let api = display_link_api(version.unwrap_or(13));

        PresentTuningOutcome {
            frame_latency_applied: true,
            display_link_applied: false,
            notes: vec![format!(
                "frame latency {} requested through wgpu's SurfaceConfiguration. Display-link \
                 phase alignment would use {api:?} on macOS {} (SDD §7.1 names CADisplayLink, \
                 which is AppKit-available only from 14 -- see ADR 009); it is not wired up, \
                 because it needs an unsafe CAMetalLayer reach-through and should follow a \
                 measurement rather than precede one",
                config.max_frame_latency,
                version
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unknown (assuming the 13 floor)".into()),
            )],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_split_matches_the_correction() {
        // The whole point of ADR 009. Ventura is 13 and must not get CADisplayLink.
        assert_eq!(display_link_api(13), DisplayLinkApi::CoreVideo);
        assert_eq!(display_link_api(14), DisplayLinkApi::CoreAnimation);
        assert_eq!(display_link_api(15), DisplayLinkApi::CoreAnimation);
        assert_eq!(display_link_api(26), DisplayLinkApi::CoreAnimation);
    }

    #[test]
    fn an_unknown_version_assumes_the_floor() {
        // Guessing low costs a deprecation warning; guessing high crashes on Ventura.
        assert_eq!(display_link_api(0), DisplayLinkApi::CoreVideo);
        assert_eq!(display_link_api(12), DisplayLinkApi::CoreVideo);
    }
}
