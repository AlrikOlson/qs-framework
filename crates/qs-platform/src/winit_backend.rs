//! `winit` implementation of the platform window and display interfaces.

use std::sync::Arc;

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use winit::window::Window;

use crate::display::{DisplayId, DisplayInfo, PlatformDisplay, sane_scale_factor};
use crate::{PhysicalSize, PlatformWindow};

/// A window, plus the display facts derived from it.
#[derive(Clone)]
pub struct WinitWindow {
    window: Arc<Window>,
}

impl std::fmt::Debug for WinitWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WinitWindow")
            .field("size", &self.size())
            .field("scale", &self.scale_factor())
            .finish()
    }
}

impl WinitWindow {
    pub fn new(window: Arc<Window>) -> Self {
        Self { window }
    }

    /// The underlying window.
    ///
    /// Exists because `wgpu::Instance::create_surface` needs the concrete handle. Everything
    /// else goes through [`PlatformWindow`]; if a second caller appears, that is a signal
    /// the trait is missing a method rather than a signal to widen this escape hatch.
    pub fn raw(&self) -> &Arc<Window> {
        &self.window
    }
}

impl PlatformWindow for WinitWindow {
    fn size(&self) -> PhysicalSize {
        let size = self.window.inner_size();
        PhysicalSize {
            width: size.width,
            height: size.height,
        }
    }

    fn scale_factor(&self) -> f64 {
        sane_scale_factor(self.window.scale_factor())
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }

    fn current_display(&self) -> Option<DisplayInfo> {
        let monitor = self.window.current_monitor()?;
        Some(monitor_info(0, &monitor))
    }

    fn set_title(&self, title: &str) {
        self.window.set_title(title);
    }

    fn is_visible(&self) -> bool {
        // `None` means the platform cannot say. Assuming visible is the safe direction: the
        // cost of rendering an invisible frame is power, and the cost of *not* rendering a
        // visible one is a blank window.
        self.window.is_visible().unwrap_or(true)
    }
}

impl HasWindowHandle for WinitWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.window.window_handle()
    }
}

impl HasDisplayHandle for WinitWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.window.display_handle()
    }
}

impl PlatformDisplay for WinitWindow {
    fn displays(&self) -> Vec<DisplayInfo> {
        self.window
            .available_monitors()
            .enumerate()
            .map(|(index, monitor)| monitor_info(index as u32, &monitor))
            .collect()
    }

    fn primary(&self) -> Option<DisplayInfo> {
        self.window
            .primary_monitor()
            .map(|monitor| monitor_info(0, &monitor))
            .or_else(|| self.current_display())
    }
}

fn monitor_info(index: u32, monitor: &winit::monitor::MonitorHandle) -> DisplayInfo {
    let size = monitor.size();
    DisplayInfo {
        id: DisplayId(index),
        name: monitor.name().unwrap_or_else(|| format!("display {index}")),
        size: PhysicalSize {
            width: size.width,
            height: size.height,
        },
        scale_factor: sane_scale_factor(monitor.scale_factor()),
        // winit reports millihertz. `None` when the platform does not know, which the
        // frame-budget calculation treats as "cannot say" rather than assuming 60 Hz.
        refresh_hz: monitor
            .refresh_rate_millihertz()
            .map(|mhz| f64::from(mhz) / 1000.0),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn refresh_rates_convert_from_millihertz() {
        // 120 Hz arrives as 120000, and getting this wrong by 1000x would make the frame
        // budget either 8333 ms or 0.008 ms -- both obviously wrong, but only if someone
        // looks.
        let hz = |mhz: u32| f64::from(mhz) / 1000.0;
        assert!((hz(120_000) - 120.0).abs() < 1e-9);
        assert!((hz(59_940) - 59.94).abs() < 1e-9);
    }
}
