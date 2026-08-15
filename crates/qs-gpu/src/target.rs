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

/// The lighting pass's own target: what one surface does to another, per pixel (T018).
///
/// # Three channels, and why
///
/// Bounced light is chromatic — an amber selection spills amber, not grey — so a
/// single-channel target cannot carry it (research R9's note, revising R11's single-channel
/// estimate). The format is `Rgba8Unorm`: attenuation and addition are both bounded inside
/// `0..=1` by the allowance tokens (`addition_max` ships at 0.35 linear), so 8 bits per
/// channel suffice, and the fourth channel is padding — stated here so a reader does not go
/// looking for what alpha means. It means nothing.
///
/// # Cost, stated before it is spent
///
/// `width x height x 4`: **8,294,400 bytes at 1920x1080 and 33,177,600 at 3840x2160** — the
/// same figures as the colour target, because the format has the same stride. Counted
/// against the same GPU allocation ceiling as everything else, and pinned by
/// `the_lighting_targets_cost_is_the_one_that_was_stated` so a wider format for headroom is
/// argued for rather than merged. Allocated **lazily**, on the first frame that carries a
/// renderable scene, so an installation that never turns the mode on carries none of it.
///
/// # What does not exist yet
///
/// No pipeline reads or writes this target — the lighting shaders are US1's tasks. It is
/// allocated here, ahead of them, because its memory cost and resize behaviour are exactly
/// the decisions T018 wants made in the open rather than under a visual feature, and because
/// `usage` is part of that decision: RENDER_ATTACHMENT for the pass that will fill it,
/// TEXTURE_BINDING for the modulation that will read it.
pub struct LightingTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: [u32; 2],
}

impl std::fmt::Debug for LightingTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LightingTarget")
            .field("size", &self.size)
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl LightingTarget {
    /// The one format this target takes. A constant rather than a parameter, because unlike
    /// the colour target it does not have to match a surface — it has to match the lighting
    /// maths, which is the same on every machine.
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// Allocate a target of exactly `size`, clamped to one texel per axis like its sibling.
    pub fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let size = [size[0].max(1), size[1].max(1)];
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qs-lighting"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            size,
        }
    }

    /// The view the lighting pass will render into.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    #[must_use]
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// Whether this target can serve a frame at `size`. Exact, like the colour target's.
    #[must_use]
    pub fn fits(&self, size: [u32; 2]) -> bool {
        self.size == [size[0].max(1), size[1].max(1)]
    }

    /// GPU memory this target holds, in bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        Self::bytes_at(self.size)
    }

    /// What a target at `size` would cost, without allocating one.
    #[must_use]
    pub fn bytes_at(size: [u32; 2]) -> u64 {
        u64::from(size[0].max(1)) * u64::from(size[1].max(1)) * BYTES_PER_TEXEL
    }
}

/// How much smaller than the viewport the blur chain works, per axis.
///
/// **Four, and the number is derived rather than tuned.** After a Gaussian of standard
/// deviation `sigma`, the signal carries essentially no energy above `1/(2*sigma)` cycles per
/// pixel; sampling it on a grid whose own Nyquist limit is above that loses nothing visible.
/// At [`BLUR_SIGMA_TEXELS`] = 4 quarter-resolution texels the blurred signal's content sits
/// below 1/32 cycles per full-resolution pixel and a quarter-resolution grid resolves to 1/8,
/// which is four times the margin needed.
///
/// Being a **power of two** is the second half. A factor of four makes the downsample an exact
/// 4x4 box average, reachable as four bilinear taps at half-texel offsets with every weight
/// equal; a factor of three would need nine taps with unequal weights and would put a
/// resampling kernel of somebody's choosing under an effect whose whole claim is that it does
/// not invent detail.
///
/// The saving is the reason it is worth stating at all: blur fragment work falls by sixteen,
/// which is what moves the effect from "measure before shipping" to "measure and ship".
pub const BLUR_DOWNSAMPLE: u32 = 4;

/// The Gaussian's standard deviation, in **downsampled** texels.
///
/// Four texels, which at [`BLUR_DOWNSAMPLE`] is **16 physical pixels** at 1x scale. That is the
/// figure to argue with: it is a little over one row height at the comfortable density, so a
/// popover shows the list behind it as bands of colour with no legible glyph left in them --
/// which is UXDD 10.2's "legible as context" in the direction that matters, since context that
/// can still be *read* competes with the panel's own text.
pub const BLUR_SIGMA_TEXELS: f32 = 4.0;

/// Where the kernel is truncated, in downsampled texels: `2.5 * sigma`, rounded.
///
/// A Gaussian has infinite support and a shader does not. The continuous distribution puts
/// 1.24% of its mass beyond 2.5 sigma; the **discrete** kernel the shader sums -- integer
/// offsets, sigma 4 -- drops **0.85%**, which is the figure that matters and is measured by
/// `the_kernel_is_stated_as_a_truncated_gaussian_and_not_as_a_tap_count` rather than quoted.
/// The two are not the same number and the test is what made the difference visible.
///
/// Whatever it is, it is renormalized into what remains rather than dropped: truncating
/// without renormalizing scales the whole blur down by that fraction, which reads as the panel
/// being faintly darker than its tint asked for and gets diagnosed as a palette bug.
///
/// Ten texels each side is 21 discrete taps, folded to **11 texture samples** by taking each
/// adjacent pair at its weighted midpoint through a linear sampler. So the chain costs
/// 1 + 11 + 11 = 23 samples per output texel across its three passes, at one sixteenth of the
/// viewport's fragment count.
pub const BLUR_RADIUS_TEXELS: u32 = 10;

/// The blur's reach in physical pixels at scale 1, for anything that needs to know how far
/// past a panel's edge the effect reads from.
#[must_use]
pub const fn blur_reach_pixels() -> u32 {
    BLUR_RADIUS_TEXELS * BLUR_DOWNSAMPLE
}

/// The quarter-resolution working set the separable blur runs on, shared by the blur and the
/// bloom.
///
/// # Why two textures and not one, and why a third
///
/// A separable Gaussian is two passes, and neither can read and write the same texture. `ping`
/// and `pong` are that pair. They are allocated together and sized together because they are
/// one resource with two halves; splitting them into lazily-allocated fields would allow the
/// state where one exists at the current size and the other does not.
///
/// `bloom` is a third half, and it exists because **both effects have to survive to the resolve
/// at once**. The blur leaves its result in `pong`, which the panel samples while it is drawn
/// over the resolved surface; the bloom leaves its result in `bloom`, which the resolve adds
/// back. Neither can be recomputed at that point and neither can be where the other is. What
/// they *do* share is everything else: one struct, one downsample step, one Gaussian shader,
/// one pair of blur pipelines, and `ping` as scratch — the bloom runs first and is finished
/// with `ping` before the blur clobbers it. That ordering is why one extra texture buys both
/// effects in one frame instead of two chains.
///
/// # Cost, stated before it is spent
///
/// Each half is `ceil(w/4) x ceil(h/4) x 4` bytes. At 1920x1080 that is 480x270, **518,400
/// bytes each and 1,555,200 for the three**; at 3840x2160, 3,110,400. Against the 8.3 MB and
/// 33.2 MB the colour target already costs at those sizes, the chain adds 18.75% -- which is
/// the whole argument for downsampling stated as a number, and the third half costs 6.25% of
/// a target rather than 100% of a second one.
///
/// All three carry the **surface's own format**, for [`OffscreenTarget`]'s reason: the blur
/// reads an sRGB texture (hardware-decoded to linear), filters in linear, and writes back
/// through the same encode, so no pass in the chain is a colour conversion.
pub struct BlurChain {
    ping: wgpu::TextureView,
    pong: wgpu::TextureView,
    /// Where the bloom's bright pass lands and where its vertical blur writes back: the
    /// finished bloom the resolve adds. Held apart from the pair for the reason in the type
    /// docs -- the blur's result and the bloom's are both alive at resolve time.
    bloom: wgpu::TextureView,
    bloom_bind_group: wgpu::BindGroup,
    /// Reads the [`OffscreenTarget`] this chain was built against: the source of the
    /// downsample pass.
    ///
    /// Held here rather than beside the target because a bind group outlives neither, and
    /// keeping it as a separate `Option` on the renderer creates the one state that must never
    /// exist -- a chain pointing at a target that has been replaced. [`BlurChain::fits`] takes
    /// the target's allocation generation for exactly that reason: a resize recreates the
    /// target, and this chain has to go with it even when the viewport rounds to the same
    /// downsampled size.
    source_bind_group: wgpu::BindGroup,
    generation: u32,
    ping_bind_group: wgpu::BindGroup,
    pong_bind_group: wgpu::BindGroup,
    size: [u32; 2],
    format: wgpu::TextureFormat,
}

impl std::fmt::Debug for BlurChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlurChain")
            .field("size", &self.size)
            .field("format", &self.format)
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl BlurChain {
    /// Allocate a pair sized for a viewport of `viewport`, reading `source`.
    ///
    /// `generation` is the [`OffscreenTarget`] allocation count `source` belongs to; see
    /// [`BlurChain::fits`].
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        format: wgpu::TextureFormat,
        viewport: [u32; 2],
        source: &wgpu::TextureView,
        generation: u32,
    ) -> Self {
        let size = Self::size_for(viewport);
        let make = |label: &str| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let ping = make("qs-blur-ping");
        let pong = make("qs-blur-pong");
        let bloom = make("qs-bloom");
        let bind = |label: &str, view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        let ping_bind_group = bind("qs-blur-ping-bind-group", &ping);
        let pong_bind_group = bind("qs-blur-pong-bind-group", &pong);
        let bloom_bind_group = bind("qs-bloom-bind-group", &bloom);
        let source_bind_group = bind("qs-blur-source-bind-group", source);
        Self {
            ping,
            pong,
            bloom,
            bloom_bind_group,
            source_bind_group,
            generation,
            ping_bind_group,
            pong_bind_group,
            size,
            format,
        }
    }

    /// The downsampled size a viewport of `viewport` needs.
    ///
    /// Rounded **up**, and clamped to one texel per axis. Rounding down would leave the last
    /// partial block of the viewport with no texel to blur into, so a panel touching the right
    /// or bottom edge would sample the clamped edge of a texture that stops short -- a smear
    /// along exactly the edge a maximised popover sits on.
    #[must_use]
    pub fn size_for(viewport: [u32; 2]) -> [u32; 2] {
        [
            viewport[0].div_ceil(BLUR_DOWNSAMPLE).max(1),
            viewport[1].div_ceil(BLUR_DOWNSAMPLE).max(1),
        ]
    }

    /// Where the **horizontal** pass writes, and the source of the vertical one.
    #[must_use]
    pub fn ping(&self) -> &wgpu::TextureView {
        &self.ping
    }

    /// Where the **downsample** writes and where the **vertical** pass writes: the finished
    /// blur the instance pipeline samples.
    ///
    /// The same texture twice, one pass apart, and that is safe rather than lucky: the
    /// horizontal pass in between reads it and writes `ping`, so nothing ever reads and writes
    /// this view in one pass. Three passes over two textures is the minimum a separable blur
    /// with a downsample can use, and the alternative -- a third texture so each pass has a
    /// fresh target -- costs another 518 KB at 1080p to avoid a comment.
    #[must_use]
    pub fn pong(&self) -> &wgpu::TextureView {
        &self.pong
    }

    /// Reads `ping`: the source of the vertical pass.
    #[must_use]
    pub fn ping_bind_group(&self) -> &wgpu::BindGroup {
        &self.ping_bind_group
    }

    /// Reads `pong`: the source of the horizontal pass, and — after the vertical pass has
    /// written back into it — the finished blur.
    #[must_use]
    pub fn pong_bind_group(&self) -> &wgpu::BindGroup {
        &self.pong_bind_group
    }

    /// Where the bloom's bright pass writes and where its vertical blur writes back: the
    /// finished bloom.
    ///
    /// The same texture twice, one pass apart, exactly as [`BlurChain::pong`] is for the blur
    /// -- the horizontal pass in between reads it and writes `ping`, so nothing reads and
    /// writes this view within one pass.
    #[must_use]
    pub fn bloom(&self) -> &wgpu::TextureView {
        &self.bloom
    }

    /// Reads the finished bloom: what the resolve adds back.
    #[must_use]
    pub fn bloom_bind_group(&self) -> &wgpu::BindGroup {
        &self.bloom_bind_group
    }

    #[must_use]
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// Reads the offscreen target: the source of the downsample pass.
    #[must_use]
    pub fn source_bind_group(&self) -> &wgpu::BindGroup {
        &self.source_bind_group
    }

    /// Whether this chain can serve a frame at `viewport` in `format`, reading the target that
    /// has been allocated `generation` times.
    ///
    /// Exact on size and format for [`OffscreenTarget::fits`]'s reasons, and on the generation
    /// for one this chain adds: the downsample pass reads the target through a bind group
    /// captured at construction, so a target replaced by a resize leaves this chain pointing at
    /// a texture nothing renders into any more. Sizes alone would not catch it -- 1920 and 1921
    /// both round to 480 -- and the symptom would be a panel showing the frame before last.
    #[must_use]
    pub fn fits(&self, viewport: [u32; 2], format: wgpu::TextureFormat, generation: u32) -> bool {
        self.size == Self::size_for(viewport)
            && self.format == format
            && self.generation == generation
    }

    /// How many quarter-resolution textures a chain holds: `ping`, `pong` and `bloom`.
    ///
    /// Named rather than written as a `3` in two places, because the memory figures below and
    /// the ones pinned in the tests have to move together when a fourth arrives.
    pub const HALVES: u64 = 3;

    /// GPU memory this chain holds, in bytes: all three halves.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.size[0]) * u64::from(self.size[1]) * BYTES_PER_TEXEL * Self::HALVES
    }

    /// What a chain for `viewport` would cost, without allocating one.
    #[must_use]
    pub fn bytes_at(viewport: [u32; 2]) -> u64 {
        let size = Self::size_for(viewport);
        u64::from(size[0]) * u64::from(size[1]) * BYTES_PER_TEXEL * Self::HALVES
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
    fn the_lighting_targets_cost_is_the_one_that_was_stated() {
        // T018: the cost is stated and pinned, like the colour target's, so a format change
        // that doubles it has to be argued for. Three meaningful channels in four bytes --
        // Rgba8Unorm -- because attenuation and addition are both bounded in 0..=1 by the
        // allowance tokens and bounced light is chromatic.
        assert_eq!(LightingTarget::bytes_at([1920, 1080]), 8_294_400);
        assert_eq!(LightingTarget::bytes_at([3840, 2160]), 33_177_600);
        assert_eq!(LightingTarget::FORMAT, wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn the_blur_chains_cost_is_stated_at_the_resolutions_the_application_runs_at() {
        // Same discipline as the colour target above, and the same reason: the downsample
        // factor is the entire argument for the chain being affordable, so the number it buys
        // is pinned. A change to BLUR_DOWNSAMPLE lands here first.
        assert_eq!(BlurChain::size_for([1920, 1080]), [480, 270]);
        assert_eq!(BlurChain::bytes_at([1920, 1080]), 1_555_200);
        assert_eq!(BlurChain::size_for([3840, 2160]), [960, 540]);
        assert_eq!(BlurChain::bytes_at([3840, 2160]), 6_220_800);

        // And the ratio that is the point: the chain is a small fraction of the target it
        // reads, not a second copy of it. All three halves together, against one colour
        // target. `prim-bloom` moved this from an eighth to three sixteenths by adding the
        // third half -- 518 KB at 1080p -- which is what buys a blur and a bloom in the same
        // frame instead of a second chain at 8.3 MB.
        let chain = BlurChain::bytes_at([1920, 1080]) as f64;
        let target = OffscreenTarget::bytes_at([1920, 1080]) as f64;
        let expected = BlurChain::HALVES as f64 / 16.0;
        assert!(
            (chain / target - expected).abs() < 1e-9,
            "three halves at 1/16 the area each is exactly three sixteenths; got {}",
            chain / target
        );
    }

    #[test]
    fn the_downsampled_size_rounds_up_so_the_last_partial_block_has_somewhere_to_go() {
        // Rounding down leaves the right and bottom edges of the viewport with no texel of
        // their own, so a panel against that edge samples a clamped texture that stops short
        // -- a smear along exactly the edge a maximised popover sits on. 1921 needs 481.
        assert_eq!(BlurChain::size_for([1921, 1081]), [481, 271]);
        assert_eq!(BlurChain::size_for([4, 4]), [1, 1]);
        assert_eq!(BlurChain::size_for([1, 1]), [1, 1]);
        // A minimised window, for the same reason the colour target clamps.
        assert_eq!(BlurChain::size_for([0, 0]), [1, 1]);
    }

    #[test]
    fn the_kernel_is_stated_as_a_truncated_gaussian_and_not_as_a_tap_count() {
        // The three constants have to keep agreeing, because each is derived from the one
        // above it: the radius is 2.5 sigma rounded, and the reach in physical pixels is the
        // radius through the downsample. A tap count edited on its own is how a kernel stops
        // being a Gaussian and becomes whatever fits.
        let expected_radius = (BLUR_SIGMA_TEXELS * 2.5).round() as u32;
        assert_eq!(BLUR_RADIUS_TEXELS, expected_radius);
        assert_eq!(blur_reach_pixels(), 40);

        // What truncation costs, computed rather than asserted from memory -- and it caught
        // the number that was asserted from memory. 1.24% is the CONTINUOUS Gaussian's mass
        // beyond 2.5 sigma; what this kernel actually drops, summing integer offsets at
        // sigma 4, is 0.85%. Either way the shader renormalizes, which is why the exact value
        // is a fact to record rather than a threshold to meet.
        let total: f64 = (-60..=60)
            .map(|i| gaussian(f64::from(i), f64::from(BLUR_SIGMA_TEXELS)))
            .sum();
        let kept: f64 = (-(BLUR_RADIUS_TEXELS as i32)..=(BLUR_RADIUS_TEXELS as i32))
            .map(|i| gaussian(f64::from(i), f64::from(BLUR_SIGMA_TEXELS)))
            .sum();
        let dropped = 1.0 - kept / total;
        assert!(
            (0.008..0.009).contains(&dropped),
            "the truncated tail should be about 0.85% of the weight; got {dropped}"
        );
    }

    fn gaussian(x: f64, sigma: f64) -> f64 {
        (-(x * x) / (2.0 * sigma * sigma)).exp()
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
