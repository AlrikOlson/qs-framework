//! The offscreen colour target, and the pass that puts it back on screen.
//!
//! # Why this is infrastructure and not a primitive
//!
//! Every primitive shipped so far is a function of **one fragment's own position**: a rounded
//! box's distance field, a ramp's projection onto an axis, a microfacet BRDF evaluated at one
//! normal. That is exactly why all of them fit one pass with no second target, and it is not a
//! coincidence — it is the constraint that kept the pipeline one shader and one draw call per
//! batch.
//!
//! Blur, bloom and refraction are functions of **neighbouring pixels**, and no amount of
//! cleverness makes them fit. What they need is not a fifth `KIND_` constant but a target to
//! render into, a way to sample what was already drawn, and a second pipeline to put it back.
//! This module is that, separated from the first effect that uses it so its memory cost, its
//! resize behaviour and its tier answer are decided in the open rather than under a visual
//! feature — and so that bloom does not inherit them unreviewed.
//!
//! # Lifetime and resize
//!
//! The target is owned by the [`crate::batcher::Renderer`] and lives across frames. It is
//! created **lazily**, on the first frame that asks for it, so an installation that never
//! enables a neighbourhood effect never allocates it. It is recreated when the viewport
//! changes size, and only then.
//!
//! Exact size, never rounded up and never kept larger than needed. A target smaller than the
//! viewport samples outside itself at the edges; a larger one wastes the whole difference,
//! and at 4K the difference is measured in tens of megabytes rather than in kilobytes. The
//! cost of recreating it is paid on resize, which is already the frame nobody is measuring.
//!
//! # Format, and why the resolve is exactly a copy
//!
//! The target carries the **surface's own format**. That makes the resolve pass a straight
//! round trip: an sRGB texture is decoded to linear when sampled and re-encoded on write to an
//! sRGB surface, so a resolve of an untouched target is bit-identical to having drawn directly
//! to the surface. `the_two_pass_path_is_pixel_identical_to_the_one_pass_path` is that claim as
//! a test, and it is the whole reason acceptance's "nothing regresses" is checkable rather than
//! asserted. Choosing a different format here — a wider one for headroom, say — would make the
//! resolve a colour conversion, and every effect built on top would be authored against a
//! subtly different image from the one the single-pass path produces.
//!
//! # The CPU tier's answer, decided here
//!
//! `tiny-skia` has no target chain and is not getting one. So the answer for the CPU tier is
//! not "a slower version of this", it is [`Floor`](crate::frame::Floor) — the mechanism
//! [`crate::frame::Fidelity`] already provides, and the same one `PrimKind::Glow` uses. Every
//! effect built on this target **must** declare `Fidelity::Enhanced { floor }`, and the parity
//! suite then holds the CPU tier to that floor exactly, rather than to whatever a missing pass
//! happens to leave behind.
//!
//! Stating it here rather than in the blur chunk is the point. Deferred, the question gets
//! answered three times, differently, by whoever is implementing each effect.

use crate::path::RenderPath;

/// Bytes per texel of every format this target can take. All the surface formats the device
/// module accepts are 8-bit RGBA or BGRA.
const BYTES_PER_TEXEL: u64 = 4;

/// An offscreen colour target the instance pass renders into and a later pass samples.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: [u32; 2],
    format: wgpu::TextureFormat,
}

impl std::fmt::Debug for OffscreenTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OffscreenTarget")
            .field("size", &self.size)
            .field("format", &self.format)
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl OffscreenTarget {
    /// Allocate a target of exactly `size`.
    ///
    /// `size` is clamped to at least one texel in each axis: a zero-sized texture is a
    /// validation error, and a minimised window legitimately reports zero.
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        format: wgpu::TextureFormat,
        size: [u32; 2],
    ) -> Self {
        let size = [size[0].max(1), size[1].max(1)];
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qs-offscreen-colour"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Both, and both are load-bearing: RENDER_ATTACHMENT is what the instance pass
            // draws into, TEXTURE_BINDING is what the resolve pass reads back out. A target
            // with only the first is a target nothing can sample, which is the whole point.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qs-offscreen-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            texture,
            view,
            bind_group,
            size,
            format,
        }
    }

    /// The view the instance pass renders into.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The bind group the resolve pass reads.
    #[must_use]
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    #[must_use]
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Whether this target can serve a frame at `size` in `format`.
    ///
    /// Exact match on both, for the reason in the module docs: a mismatched size samples
    /// outside itself and a mismatched format turns the resolve into a colour conversion.
    #[must_use]
    pub fn fits(&self, size: [u32; 2], format: wgpu::TextureFormat) -> bool {
        self.size == [size[0].max(1), size[1].max(1)] && self.format == format
    }

    /// GPU memory this target holds, in bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.size[0]) * u64::from(self.size[1]) * BYTES_PER_TEXEL
    }

    /// What a target at `size` would cost, without allocating one.
    ///
    /// Exposed so the cost can be reported and budgeted before the allocation is made, which
    /// is the difference between a stated memory cost and a discovered one. At 1920x1080 this
    /// is 8.3 MB and at 3840x2160 it is 33.2 MB — both single figures worth knowing before an
    /// effect is authored against them, and both counted against the same GPU allocation
    /// ceiling everything else is.
    #[must_use]
    pub fn bytes_at(size: [u32; 2]) -> u64 {
        u64::from(size[0].max(1)) * u64::from(size[1].max(1)) * BYTES_PER_TEXEL
    }
}

/// Whether a tier can hold an offscreen target at all.
///
/// Not a capability probe: a second colour attachment is available on every device that can
/// run either GPU tier, and there is no version of this question that a GL 4.3 or GLES 3.1
/// device answers differently from a Vulkan one. The line is between **GPU and not**.
///
/// The CPU tier's answer is [`crate::frame::Floor`], not a slower target chain — see the
/// module docs. This function exists so that the answer is asked of the tier in one place
/// rather than inferred from whether a `GpuContext` happens to exist.
#[must_use]
pub fn tier_can_hold_target(path: RenderPath) -> bool {
    match path {
        RenderPath::Primary | RenderPath::Reduced => true,
        RenderPath::Cpu => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_memory_cost_is_stated_at_the_resolutions_the_application_runs_at() {
        // Acceptance: "memory cost stated at the resolutions the application actually runs
        // at, not asymptotically". These are the two numbers, pinned so that a format change
        // that doubles them cannot land quietly -- a 16-bit target for headroom is a
        // reasonable idea and a 66 MB one at 4K is a reasonable thing to have to argue for.
        assert_eq!(OffscreenTarget::bytes_at([1920, 1080]), 8_294_400);
        assert_eq!(OffscreenTarget::bytes_at([3840, 2160]), 33_177_600);
    }

    #[test]
    fn a_minimised_window_does_not_ask_for_a_zero_sized_texture() {
        // A zero extent is a validation error, and a minimised window reports zero on
        // Windows. Clamping in `bytes_at` and in `new` keeps the two agreeing about what a
        // degenerate viewport costs.
        assert_eq!(OffscreenTarget::bytes_at([0, 0]), 4);
        assert_eq!(OffscreenTarget::bytes_at([1920, 0]), 1920 * 4);
    }

    #[test]
    fn the_cpu_tier_gets_a_floor_rather_than_a_target() {
        // The decision this module exists to make in the open. `tiny-skia` has no target
        // chain, so every effect built on this declares `Fidelity::Enhanced { floor }` and
        // the parity suite holds the CPU tier to that floor exactly.
        assert!(tier_can_hold_target(RenderPath::Primary));
        assert!(tier_can_hold_target(RenderPath::Reduced));
        assert!(!tier_can_hold_target(RenderPath::Cpu));
    }

    #[test]
    fn a_tier_that_can_hold_a_target_is_not_the_one_that_cannot() {
        // `RenderPath` derives `Ord` from DECLARATION order -- Primary < Reduced < Cpu --
        // which is the OPPOSITE of capability, and `tier <= something` reads as "capable
        // enough" while meaning the reverse. That has already cost real time once. This
        // asserts the predicate rather than the ordering, so a future reader reaching for
        // `path <= RenderPath::Reduced` has something correct to reach for instead.
        assert!(RenderPath::Primary < RenderPath::Cpu);
        assert!(tier_can_hold_target(RenderPath::Primary));
        assert!(!tier_can_hold_target(RenderPath::Cpu));
    }
}
