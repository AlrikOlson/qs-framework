//! Adapter selection, device creation, surface management and capability checks.
//!
//! Failures are returned to the caller so it can choose another rendering tier.

use std::sync::Arc;

use crate::path::RenderPath;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no graphics adapter is usable on this machine")]
    NoAdapter,
    #[error("the adapter refused to create a device: {0}")]
    DeviceCreation(String),
    #[error("could not create a surface for this window: {0}")]
    Surface(String),
    #[error("the device was lost: {0}")]
    DeviceLost(String),
}

/// Adapter capabilities, queried at startup.
///
/// Timestamp support determines whether GPU execution time can be measured.
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub backend: wgpu::Backend,
    pub adapter_name: String,
    pub driver: String,
    pub device_type: wgpu::DeviceType,
    /// Timestamp queries inside render passes are available.
    pub timestamps: bool,
    pub max_texture_dimension: u32,
    /// Present modes offered by the surface. Record these when comparing frame timings.
    pub present_modes: Vec<wgpu::PresentMode>,
    pub surface_format: wgpu::TextureFormat,
}

impl Capabilities {
    /// Which tier this adapter can support.
    pub fn tier(&self) -> RenderPath {
        match self.backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => {
                RenderPath::Primary
            }
            wgpu::Backend::Gl => RenderPath::Reduced,
            // `Noop` and anything future: this crate cannot claim it renders.
            _ => RenderPath::Cpu,
        }
    }

    /// Whether the adapter uses a software implementation such as WARP,
    /// lavapipe or SwiftShader. Its timings include software rendering costs.
    pub fn is_software(&self) -> bool {
        self.device_type == wgpu::DeviceType::Cpu
    }
}

/// A live device plus everything derived from it.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub capabilities: Capabilities,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("backend", &self.capabilities.backend)
            .field("adapter", &self.capabilities.adapter_name)
            .finish()
    }
}

/// Backends to enable for a tier.
pub fn backends_for(tier: RenderPath) -> wgpu::Backends {
    match tier {
        RenderPath::Primary => {
            wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12
        }
        RenderPath::Reduced => wgpu::Backends::GL,
        // The CPU tier does not use wgpu at all; see `crate::cpu_raster`.
        RenderPath::Cpu => wgpu::Backends::empty(),
    }
}

pub fn new_instance(tier: RenderPath) -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends_for(tier);
    descriptor.flags = wgpu::InstanceFlags::from_build_config();
    wgpu::Instance::new(descriptor)
}

/// Find the best usable rendering tier without creating a window.
///
/// Returns [`RenderPath::Cpu`] when neither GPU tier has a usable adapter.
pub fn probe_tier() -> RenderPath {
    tier_from_adapter_counts(|tier| {
        let instance = new_instance(tier);
        pollster::block_on(instance.enumerate_adapters(backends_for(tier))).len()
    })
}

/// Select a rendering tier using the supplied adapter counts.
///
/// The closure returns the number of adapters for each tier, including
/// software adapters. Zero adapters for both GPU tiers selects the CPU renderer.
pub fn tier_from_adapter_counts(mut adapters_for: impl FnMut(RenderPath) -> usize) -> RenderPath {
    for tier in [RenderPath::Primary, RenderPath::Reduced] {
        if adapters_for(tier) > 0 {
            return tier;
        }
    }
    RenderPath::Cpu
}

impl GpuContext {
    /// Create a device for `tier`, optionally compatible with `surface`.
    ///
    /// The caller must already have persisted the crash-attempt counter -- see
    /// [`crate::path::CrashCounter::begin_attempt`]. This is the risky work that counter
    /// exists to observe.
    pub fn new(
        tier: RenderPath,
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'static>>,
    ) -> Result<Self, GpuError> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            ..Default::default()
        }))
        .or_else(|_| {
            // Second chance on the software adapter. On a machine with a broken hardware
            // driver this is the difference between running and not.
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: true,
                compatible_surface: surface,
                ..Default::default()
            }))
        })
        .map_err(|_| GpuError::NoAdapter)?;

        let info = adapter.get_info();
        let adapter_features = adapter.features();

        // Timestamp queries are requested only when available. Requesting an unsupported
        // feature fails device creation outright, which would turn "we cannot measure GPU
        // time on this machine" into "this machine cannot run the application".
        let mut required_features = wgpu::Features::empty();
        let timestamps = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("qs-device"),
            required_features,
            // Downlevel defaults, raised to the adapter's real limits. Requesting the
            // adapter's maximum would fail on the exact low-end hardware the Reduced tier
            // is for; requesting the downlevel minimum would cap the atlas at 2048 on
            // machines that can do far better.
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            ..Default::default()
        }))
        .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        // An uncaptured validation error must be loud during development and survivable in
        // release. The default handler panics, which on the render thread means the window
        // vanishes with no message the user can act on.
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            tracing::error!(target: "qs::gpu", %error, "uncaptured wgpu error");
        }));

        let (present_modes, surface_format) = match surface {
            Some(surface) => {
                let caps = surface.get_capabilities(&adapter);
                let format = preferred_format(&caps);
                (caps.present_modes, format)
            }
            None => (
                vec![wgpu::PresentMode::Fifo],
                wgpu::TextureFormat::Bgra8UnormSrgb,
            ),
        };

        let capabilities = Capabilities {
            backend: info.backend,
            adapter_name: info.name.clone(),
            driver: if info.driver_info.is_empty() {
                info.driver.clone()
            } else {
                format!("{} {}", info.driver, info.driver_info)
            },
            device_type: info.device_type,
            timestamps,
            max_texture_dimension: device.limits().max_texture_dimension_2d,
            present_modes,
            surface_format,
        };

        tracing::info!(
            target: "qs::gpu",
            backend = ?capabilities.backend,
            adapter = %capabilities.adapter_name,
            driver = %capabilities.driver,
            timestamps = capabilities.timestamps,
            software = capabilities.is_software(),
            %tier,
            "device created"
        );

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
        })
    }

    /// Configure a surface, preferring `Mailbox` and falling back to `Fifo`.
    ///
    /// `FifoRelaxed` is not selected automatically because it can tear when a
    /// frame misses vblank.
    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> wgpu::PresentMode {
        let present_mode = self.preferred_present_mode();
        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.capabilities.surface_format,
                // A zero-sized surface is not configurable and is a normal state -- it is
                // what a minimized window reports.
                width: width.max(1),
                height: height.max(1),
                present_mode,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );
        present_mode
    }

    pub fn preferred_present_mode(&self) -> wgpu::PresentMode {
        if self
            .capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else {
            // Always available per the WebGPU specification.
            wgpu::PresentMode::Fifo
        }
    }
}

/// Prefer an sRGB surface format.
///
/// The instance buffer carries premultiplied **linear** colour, so the hardware must apply
/// the sRGB transfer function on write. Picking a non-sRGB format here would render the
/// entire UI visibly too dark, and it is the kind of error that looks like a design choice
/// until someone compares against another application.
fn preferred_format(caps: &wgpu::SurfaceCapabilities) -> wgpu::TextureFormat {
    caps.formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .or_else(|| caps.formats.first().copied())
        .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb)
}

/// What to do when the next surface texture cannot be used as-is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceRecovery {
    /// Skip this frame; the next one will be fine.
    SkipFrame,
    /// Reconfigure the surface and retry.
    Reconfigure,
    /// The device was lost. Rebuild its resources.
    RebuildDevice,
}

/// The outcome of asking the surface for a frame.
pub enum Acquired {
    /// Render into this and present it.
    Frame(wgpu::SurfaceTexture),
    /// Usable, but the surface no longer matches the window. Render and present it, then
    /// reconfigure -- dropping the frame instead would make a window resize flicker.
    Suboptimal(wgpu::SurfaceTexture),
    /// Nothing to render into. Take the named action.
    Recover(SurfaceRecovery),
}

impl std::fmt::Debug for Acquired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Frame(_) => f.write_str("Frame"),
            Self::Suboptimal(_) => f.write_str("Suboptimal"),
            Self::Recover(r) => write!(f, "Recover({r:?})"),
        }
    }
}

/// Acquire the next surface frame and classify the result.
///
/// An outdated surface needs reconfiguration; device loss needs resources
/// rebuilt. An occluded window can skip rendering until it becomes visible.
pub fn acquire(surface: &wgpu::Surface<'static>) -> Acquired {
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture) => Acquired::Frame(texture),
        wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Acquired::Suboptimal(texture),
        // The surface changed size, or the compositor reconfigured underneath us.
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            Acquired::Recover(SurfaceRecovery::Reconfigure)
        }
        // Minimized or fully covered. Not an error; there is genuinely nothing to draw,
        // and continuing to render would burn power for pixels nobody sees.
        wgpu::CurrentSurfaceTexture::Occluded => Acquired::Recover(SurfaceRecovery::SkipFrame),
        // Took too long. Dropping the frame is correct; the alternative is queueing one the
        // user will never see.
        wgpu::CurrentSurfaceTexture::Timeout => Acquired::Recover(SurfaceRecovery::SkipFrame),
        // The driver rejected the acquisition. This is the device-loss path.
        wgpu::CurrentSurfaceTexture::Validation => {
            Acquired::Recover(SurfaceRecovery::RebuildDevice)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn probing_always_yields_a_usable_tier() {
        // RP-1: even with no adapter at all, the answer is a tier, never a panic.
        //
        // Worth being precise about what this does and does not establish. `RenderPath` has
        // exactly three variants, so the `matches!` below accepts every value the function
        // can return -- it is a tautology, and the only real assertion here is that the
        // call returns at all rather than panicking. That is worth having, since a panic
        // inside adapter enumeration is SDD R1's named top risk. It is not evidence for
        // SC-009's actual claim; `no_usable_adapter_anywhere_falls_back_to_the_cpu_tier`
        // is.
        let tier = probe_tier();
        assert!(matches!(
            tier,
            RenderPath::Primary | RenderPath::Reduced | RenderPath::Cpu
        ));
    }

    #[test]
    fn no_usable_adapter_anywhere_falls_back_to_the_cpu_tier() {
        // The falsifiability proof for SC-009, and the branch no development machine and no
        // CI runner in this project can reach for real: both have adapters.
        assert_eq!(tier_from_adapter_counts(|_| 0), RenderPath::Cpu);
    }

    #[test]
    fn a_machine_with_only_gl_lands_on_reduced_rather_than_the_cpu_floor() {
        // The middle tier is the one a fallback chain gets wrong: it is easy to write a
        // probe that tries the good path and then gives up entirely.
        let tier = tier_from_adapter_counts(|tier| usize::from(tier == RenderPath::Reduced));
        assert_eq!(tier, RenderPath::Reduced);
    }

    #[test]
    fn a_machine_with_a_primary_adapter_never_settles_for_less() {
        assert_eq!(tier_from_adapter_counts(|_| 1), RenderPath::Primary);

        // And the tiers are tried in order, not sampled: asking about Reduced before
        // Primary has been ruled out would be a silent demotion on healthy hardware.
        let mut asked = Vec::new();
        let tier = tier_from_adapter_counts(|tier| {
            asked.push(tier);
            1
        });
        assert_eq!(tier, RenderPath::Primary);
        assert_eq!(asked, vec![RenderPath::Primary]);
    }

    #[test]
    fn the_cpu_tier_requests_no_wgpu_backends() {
        assert!(backends_for(RenderPath::Cpu).is_empty());
        assert!(backends_for(RenderPath::Primary).contains(wgpu::Backends::DX12));
        assert_eq!(backends_for(RenderPath::Reduced), wgpu::Backends::GL);
    }

    #[test]
    fn an_srgb_surface_format_is_preferred_over_a_linear_one() {
        // Picking a non-sRGB format renders the whole UI too dark, and it looks like a
        // design choice until someone compares against another application.
        let caps = wgpu::SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Bgra8UnormSrgb,
            ],
            ..Default::default()
        };
        assert_eq!(preferred_format(&caps), wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn a_surface_offering_no_srgb_format_still_yields_one() {
        let caps = wgpu::SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Rgba16Float],
            ..Default::default()
        };
        assert_eq!(preferred_format(&caps), wgpu::TextureFormat::Rgba16Float);

        let empty = wgpu::SurfaceCapabilities::default();
        assert_eq!(
            preferred_format(&empty),
            wgpu::TextureFormat::Bgra8UnormSrgb
        );
    }
}
