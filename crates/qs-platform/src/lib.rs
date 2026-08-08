//! Windowing, input, display enumeration and present tuning — behind traits that outlive
//! the backend.
//!
//! # Why `winit`, and why behind a trait
//!
//! Research R1: SDD §3.2 specifies a per-OS `qs-platform` implementation, but hand-writing
//! Win32 + AppKit + Wayland/X11 windowing is roughly a milestone of work by itself and
//! retires **none** of M0's risk. The risk M0 exists to measure is rendering throughput, not
//! window creation.
//!
//! Wrapping `winit` behind our own trait means the eventual hand-written layer is a swap
//! rather than a rewrite. The trait surface here is deliberately the *minimum* `qs-ui`
//! actually consumes -- every method that leaks a `winit` type into it is a method that
//! makes the swap harder, and there are none.
//!
//! M0 implements 2 of the ~12 traits SDD §9.1 eventually needs: [`PlatformWindow`] and
//! [`PlatformDisplay`].

pub mod display;
pub mod present_tuning;
pub mod winit_backend;

#[cfg(target_os = "windows")]
pub mod win {
    pub mod present;
}

#[cfg(target_os = "macos")]
pub mod mac {
    pub mod present;
}

pub use display::{DisplayId, DisplayInfo, PlatformDisplay};
pub use present_tuning::{NoopPresentTuning, PresentTuning};
pub use winit_backend::WinitWindow;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Logical (pre-scale) size.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct LogicalSize {
    pub width: f64,
    pub height: f64,
}

/// Physical (post-scale) size. Everything below `qs-ui` works in these.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PhysicalSize {
    pub width: u32,
    pub height: u32,
}

/// A window, as everything above the platform layer sees it.
///
/// Note what is *not* here: no event loop, no message pump, no `winit` types. Events are
/// delivered by the backend to the application rather than pulled through this trait,
/// because an event model is the single hardest thing to keep portable and the easiest to
/// let leak.
pub trait PlatformWindow: HasWindowHandle + HasDisplayHandle {
    /// Physical pixels.
    fn size(&self) -> PhysicalSize;

    /// Device pixel ratio for the monitor this window is currently on.
    fn scale_factor(&self) -> f64;

    /// Ask for a repaint. Idempotent within a frame.
    fn request_redraw(&self);

    /// The display this window is on, if the platform can say.
    fn current_display(&self) -> Option<DisplayInfo>;

    fn set_title(&self, title: &str);

    /// Whether the window is currently visible enough to be worth rendering. A minimized
    /// or fully occluded window should not burn power drawing pixels nobody sees.
    fn is_visible(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_default_to_zero_rather_than_one() {
        // A zero-sized window is a real state -- it is what a minimized window reports --
        // and the renderer clamps rather than the platform layer lying about it.
        assert_eq!(PhysicalSize::default().width, 0);
        assert_eq!(LogicalSize::default().width, 0.0);
    }
}
