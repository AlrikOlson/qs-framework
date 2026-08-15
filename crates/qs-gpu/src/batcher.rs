//! The instanced pipeline: one shader, one vertex buffer, one draw call per batch.
//!
//! Passes 0-1 (rounded-rect fill and stroke) and pass 3 (text) all run through
//! `shaders/instance.wgsl`. See that file for why the primitive kind is a branch inside one
//! pipeline rather than three pipelines.
//!
//! Geometry is generated in the vertex shader from `vertex_index`, so the only per-frame
//! upload is the instance buffer. At the fastest fling that is roughly 96 KB -- one
//! `write_buffer` and one `draw` per batch, which is what keeps CPU submit time off the
//! frame budget.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::atlas::{PendingImage, PendingUpload};
use crate::device::GpuContext;
use crate::frame::{Batch, DrawList, FIELD_CENTRES, Instance, PrimKind};
use crate::scene::SceneList;
use crate::target::{BlurChain, LightingTarget, OffscreenTarget};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct Globals {
    viewport: [f32; 2],
    _pad: [f32; 2],
    /// Premultiplied linear RGBA. See `qs_gpu::frame::Environment`.
    ///
    /// `vec4` alignment in a uniform block is 16 bytes, which the two `[f32; 2]` above
    /// happen to satisfy exactly. A field inserted between them and these would silently
    /// shift the shader's view of the struct, so anything added goes after.
    env_horizon: [f32; 4],
    env_zenith: [f32; 4],
    /// The ambient field, laid out as three `vec4` arrays rather than an array of structs.
    ///
    /// A uniform block pads every struct member to 16 bytes, so an array of a 3-`vec4` struct
    /// and three arrays of `vec4` occupy the same 192 bytes — and three flat arrays have one
    /// layout instead of one layout per driver's opinion about the struct's stride. See
    /// `qs_gpu::frame::FieldWash` for what the three carry.
    field_place: [[f32; 4]; FIELD_CENTRES],
    field_tint: [[f32; 4]; FIELD_CENTRES],
    field_form: [[f32; 4]; FIELD_CENTRES],
}

/// One row of the field uniform, taken from the draw list's [`crate::frame::FieldWash`].
///
/// Three of these rather than one loop building three arrays, because the alternative is
/// three indices kept in step by hand across a `for` body — and a field whose tint came from
/// one centre and whose reach came from another is a bug that renders as *almost* right.
fn field_of(
    list: &DrawList,
    row: impl Fn(&crate::frame::FieldCentre) -> [f32; 4],
) -> [[f32; 4]; FIELD_CENTRES] {
    let mut out = [[0.0; 4]; FIELD_CENTRES];
    for (slot, centre) in out.iter_mut().zip(&list.field.centres) {
        *slot = row(centre);
    }
    out
}

/// Initial instance-buffer capacity, in instances. Grown geometrically on demand; the
/// starting size is chosen so a 4K viewport's first frame does not reallocate.
const INITIAL_INSTANCE_CAPACITY: u64 = 8192;

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    atlas_bind_group: wgpu::BindGroup,
    atlas_texture: wgpu::Texture,
    /// The colour page. Bound in the same group as the coverage page, so a batch mixing
    /// text and pictures is still one draw.
    colour_texture: wgpu::Texture,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
    format: wgpu::TextureFormat,
    /// The resolve half of the two-pass path. See [`crate::target`].
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    resolve_sampler: wgpu::Sampler,
    /// Allocated on the first frame that needs it and never before, so an installation with
    /// no neighbourhood effect enabled does not carry 33 MB at 4K for nothing.
    offscreen: Option<OffscreenTarget>,
    /// Take the two-pass path even when no effect asks for it.
    ///
    /// For tests and harnesses. It exists because the path would otherwise be unreachable
    /// until the first effect built on it lands, and unreachable code is untested code — the
    /// blur would then be debugging the target and the blur at once. See
    /// [`Renderer::force_offscreen`].
    force_offscreen: bool,
    /// How many times a target has been allocated over this renderer's life.
    ///
    /// Reported rather than assumed, in the same spirit as `glyphs_dropped`: "the target is
    /// recreated only on resize" is a claim about a thing that is otherwise invisible, and a
    /// per-frame reallocation of 33 MB looks exactly like a correct render.
    offscreen_allocations: u32,
    /// The lighting pass's target (T018). Lazy for the same reason as `offscreen`, one
    /// mode further out: allocated on the first frame that carries a renderable scene, so
    /// the mode being off costs no memory as well as no work. See
    /// [`crate::target::LightingTarget`] for the format, the three channels and the cost.
    lighting: Option<LightingTarget>,
    /// Counted for the same reason as `offscreen_allocations`.
    lighting_allocations: u32,
    /// The lighting pass itself: one fullscreen triangle drawn in the seam between the
    /// surface and content halves, blending `dst * attenuation + addition` with
    /// fixed-function state. Compiled unconditionally — a pipeline is a few kilobytes and a
    /// mode that compiled its shader on first use would pay a hitch at the exact keystroke
    /// that turns it on.
    lit_pipeline: wgpu::RenderPipeline,
    lit_buffer: wgpu::Buffer,
    lit_bind_group: wgpu::BindGroup,
    /// Slabs the last packed scene could not fit into [`crate::lighting::LIT_SLABS`].
    /// Reported rather than swallowed, like every other drop in this module.
    lit_dropped: usize,
    /// The layout every backdrop-shaped binding shares: one filtered texture and one sampler.
    ///
    /// One layout for four different bind groups — the blur chain's two halves, its view of
    /// the offscreen target, and the placeholder — because they are the same thing seen from
    /// different passes. It is also group 2 of the instance pipeline, which is what makes
    /// "bind the finished blur" and "bind a 1x1 stand-in" the same call.
    backdrop_layout: wgpu::BindGroupLayout,
    /// **Linear**, unlike [`Renderer::resolve_sampler`], and the difference is the whole
    /// technique. The downsample's four taps land on 2x2 texel corners and the blur's pairs
    /// land between texel centres; both depend on the hardware returning a weighted average
    /// rather than a nearest texel. With a nearest sampler the chain still runs, still looks
    /// blurred, and quietly computes a different kernel from the one `blur.wgsl` documents.
    blur_sampler: wgpu::Sampler,
    blur_downsample_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    /// The quarter-resolution ping-pong pair. Lazy for the same reason the colour target is,
    /// and allocated with it — see [`crate::target::BlurChain`] for the cost.
    blur: Option<BlurChain>,
    /// What group 2 binds on a frame with no blur in it: a 1x1 texture nothing samples.
    ///
    /// A placeholder rather than an optional binding, because a pipeline layout is fixed at
    /// creation. The alternative is two instance pipelines differing only in whether group 2
    /// exists, which doubles a shader compile to avoid four bytes.
    backdrop_placeholder: wgpu::BindGroup,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("format", &self.format)
            .field("instance_capacity", &self.instance_capacity)
            .field("offscreen", &self.offscreen)
            .finish()
    }
}

impl Renderer {
    pub fn new(ctx: &GpuContext, atlas_size: u32) -> Self {
        let device = &ctx.device;
        let format = ctx.capabilities.surface_format;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qs-instance-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/instance.wgsl").into()),
        });

        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qs-globals-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Both stages. The vertex stage has always needed the viewport; the fragment
                // stage now needs the environment, because a lit surface reflects a sky and a
                // sky is a per-frame constant rather than a per-instance one.
                //
                // Worth knowing how this fails: a fragment shader reading a binding declared
                // vertex-only is a *validation* error, and `on_uncaptured_error` logs it to
                // `tracing`, so a harness with no subscriber renders a blank frame and says
                // nothing at all. It cost a confused minute; it would cost longer in a window.
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let atlas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qs-atlas-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The colour page, beside the coverage page in the same group. One
                // set_bind_group serves both, which is what keeps a batch that mixes text
                // and thumbnails a single draw rather than two.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qs-globals"),
            contents: bytemuck::bytes_of(&Globals::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qs-globals-bind-group"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qs-glyph-atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // R8: one coverage byte per texel. See qs-text::raster for why coverage rather
            // than subpixel RGB.
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let colour_size = crate::atlas::DEFAULT_COLOUR_PAGE;
        let colour_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qs-colour-atlas"),
            size: wgpu::Extent3d {
                width: colour_size,
                height: colour_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Srgb, not plain Unorm: a decoder hands over sRGB bytes and the whole pipeline
            // downstream is linear, so the conversion has to happen somewhere. In the
            // texture format it is free and exact; in the shader it is three `pow`s per
            // fragment; at admission it would bake a lossy 8-bit linear encoding into the
            // cache. See qs_gpu::atlas::RgbaImage.
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let colour_view = colour_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("qs-atlas-sampler"),
            // Clamp, not repeat: a glyph sampled slightly outside its rect must read the
            // gutter, never wrap to the opposite edge of the atlas.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            // Linear filtering: glyph quads land on integer pixels (the fractional part is
            // baked into the subpixel variant), so this is effectively a 1:1 blit, but
            // linear keeps a fractional DPI scale from producing hard edges.
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qs-atlas-bind-group"),
            layout: &atlas_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&colour_view),
                },
            ],
        });

        // Group 2: whatever a blur samples. The same layout serves the chain's two halves, its
        // view of the offscreen target and the placeholder, so every one of them is
        // interchangeable at the `set_bind_group` call site.
        let backdrop_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qs-backdrop-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qs-pipeline-layout"),
            bind_group_layouts: &[
                Some(&globals_layout),
                Some(&atlas_layout),
                Some(&backdrop_layout),
                // Group 3: the SHARP backdrop, for `KIND_REFRACT`. The same layout as group 2
                // and deliberately not the same binding — see the shader, where the two are
                // declared side by side with the reason they cannot be one.
                Some(&backdrop_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("qs-instance-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &INSTANCE_ATTRIBUTES,
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Premultiplied source-over. The instance colour is already
                    // premultiplied (qs-gpu::color), so the source factor is One rather
                    // than SrcAlpha -- using SrcAlpha here would multiply by alpha twice
                    // and every translucent edge would be too dark.
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // No culling: quads are generated in a fixed winding, and a culled
                // primitive on a backend with the opposite convention is an empty window
                // that takes a day to diagnose.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qs-instances"),
            size: INITIAL_INSTANCE_CAPACITY * size_of::<Instance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The resolve half. Built at construction rather than lazily beside the target,
        // because a pipeline is cheap to hold and expensive to compile: creating it on the
        // first frame that needs a backdrop would put a shader compile inside that frame,
        // which is the one frame a neighbourhood effect is already making expensive. The
        // *target* is still lazy — that is where the megabytes are.
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qs-resolve-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resolve.wgsl").into()),
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qs-resolve-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Nearest, and clamped. The resolve is a 1:1 blit — the target is exactly the
        // viewport's size — so linear filtering would sample the same texel and cost
        // nothing, right up until a fractional viewport or a half-texel offset made it
        // sample two and blur the whole frame by a hair that nobody could attribute.
        let resolve_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("qs-resolve-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let resolve_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("qs-resolve-pipeline-layout"),
                bind_group_layouts: &[Some(&resolve_layout)],
                immediate_size: 0,
            });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("qs-resolve-pipeline"),
            layout: Some(&resolve_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // `None`, not premultiplied source-over. The offscreen target already
                    // holds the composited frame; blending it *again* over the surface would
                    // composite every translucent pixel twice. The resolve replaces.
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // The blur chain. Three pipelines over one shader module and one bind-group layout,
        // built at construction for the resolve pipeline's reason: a shader compile inside the
        // frame that first opens a popover is a hitch on the exact keystroke the effect exists
        // to make feel immediate. The *textures* stay lazy.
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qs-blur-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });
        let blur_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("qs-blur-sampler"),
            // Clamped, so a tap that reaches past the edge reads the edge rather than wrapping
            // to the far side of the window -- which is what a repeat address mode does to a
            // panel in the corner, and it looks like the blur has torn.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qs-blur-pipeline-layout"),
            bind_group_layouts: &[Some(&backdrop_layout)],
            immediate_size: 0,
        });
        let blur_pipeline = |label: &str, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&blur_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &blur_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blur_shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        // `None`, like the resolve and for the same reason: each pass in the
                        // chain writes the whole of its target from its source. Blending would
                        // composite the previous frame's blur under this one, which converges
                        // to a smear that only shows up while something moves.
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let blur_downsample_pipeline = blur_pipeline("qs-blur-downsample", "fs_downsample");
        let blur_h_pipeline = blur_pipeline("qs-blur-horizontal", "fs_blur_h");
        let blur_v_pipeline = blur_pipeline("qs-blur-vertical", "fs_blur_v");

        // The placeholder group 2 binds when no blur is in the frame. One texel, never
        // sampled: a KIND_BLUR instance is exactly what makes the chain run, so a frame that
        // binds this contains nothing that reads it.
        let placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qs-backdrop-placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let placeholder_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());
        let backdrop_placeholder = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qs-backdrop-placeholder-bind-group"),
            layout: &backdrop_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blur_sampler),
                },
            ],
        });

        // The lighting pass (specs/002 US1). One fullscreen triangle, fixed-function
        // blending: the fragment writes `vec4(addition, attenuation)` and the blend applies
        // `out = src.rgb + dst.rgb * src.a` — a multiply-and-add that never samples the
        // surface image, which is what lets it draw INSIDE the same render pass, in the
        // seam between the surface and content halves.
        let lit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qs-lighting-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lighting.wgsl").into()),
        });
        let lit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qs-lighting-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let lit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qs-lighting-scene"),
            size: std::mem::size_of::<crate::lighting::LitSceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qs-lighting-bind-group"),
            layout: &lit_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lit_buffer.as_entire_binding(),
            }],
        });
        let lit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qs-lighting-pipeline-layout"),
            bind_group_layouts: &[Some(&lit_layout)],
            immediate_size: 0,
        });
        let lit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("qs-lighting-pipeline"),
            layout: Some(&lit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &lit_shader,
                entry_point: Some("vs_lit"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &lit_shader,
                entry_point: Some("fs_lit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        // `out.rgb = src.rgb + dst.rgb * src.a`: the addition (zero until
                        // US2's bounce) plus the surface attenuated. The alpha component
                        // keeps the destination's, untouched — the pass has no opinion
                        // about coverage.
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::SrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            globals_buffer,
            globals_bind_group,
            atlas_bind_group,
            atlas_texture,
            colour_texture,
            instance_buffer,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            format,
            resolve_pipeline,
            resolve_layout,
            resolve_sampler,
            offscreen: None,
            force_offscreen: false,
            offscreen_allocations: 0,
            backdrop_layout,
            blur_sampler,
            blur_downsample_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            blur: None,
            backdrop_placeholder,
            lighting: None,
            lighting_allocations: 0,
            lit_pipeline,
            lit_buffer,
            lit_bind_group,
            lit_dropped: 0,
        }
    }

    /// Copy newly rasterized glyphs into the atlas texture.
    pub fn upload_glyphs(&self, ctx: &GpuContext, uploads: &[PendingUpload]) {
        for upload in uploads {
            if upload.width == 0 || upload.height == 0 {
                continue;
            }
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.atlas_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.x,
                        y: upload.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.coverage,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // R8 is one byte per texel and the rasterizer emits unpadded rows, so
                    // the source stride is exactly the width.
                    bytes_per_row: Some(upload.width),
                    rows_per_image: Some(upload.height),
                },
                wgpu::Extent3d {
                    width: upload.width,
                    height: upload.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Copy newly admitted pictures into the colour texture.
    ///
    /// A separate call from [`Renderer::upload_glyphs`] rather than a branch inside it, for
    /// the reason [`PendingImage`] is a separate type from [`PendingUpload`]: a frame with
    /// no images passes an empty slice, and this loop does not run at all.
    pub fn upload_images(&self, ctx: &GpuContext, uploads: &[PendingImage]) {
        for upload in uploads {
            if upload.width == 0 || upload.height == 0 {
                continue;
            }
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.colour_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.x,
                        y: upload.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // Four bytes per texel, unpadded rows -- the atlas emits exactly
                    // `width * height * 4` and `RgbaImage::is_malformed` refuses anything
                    // that does not, which is what makes this stride safe to assert.
                    bytes_per_row: Some(upload.width * 4),
                    rows_per_image: Some(upload.height),
                },
                wgpu::Extent3d {
                    width: upload.width,
                    height: upload.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn ensure_capacity(&mut self, ctx: &GpuContext, needed: u64) {
        if needed <= self.instance_capacity {
            return;
        }
        // Geometric growth. Growing to exactly `needed` would reallocate on almost every
        // frame during a scroll that is steadily revealing more content.
        let capacity = needed.next_power_of_two();
        self.instance_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qs-instances"),
            size: capacity * size_of::<Instance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_capacity = capacity;
    }

    /// Take the two-pass path on every frame, whatever the draw list asks for.
    ///
    /// Off by default and never set by the application. Its only callers are tests and
    /// harnesses, and it exists so the offscreen path is exercised **before** the first
    /// effect is built on it — otherwise the blur chunk debugs the target and the blur
    /// simultaneously, with no way to tell which one is wrong.
    pub fn force_offscreen(&mut self, force: bool) {
        self.force_offscreen = force;
    }

    /// The offscreen target, if one has been allocated.
    ///
    /// `None` until a frame needs it. Exposed so the memory it holds can be reported rather
    /// than estimated: `renderer.offscreen().map_or(0, OffscreenTarget::bytes)`.
    #[must_use]
    pub fn offscreen(&self) -> Option<&OffscreenTarget> {
        self.offscreen.as_ref()
    }

    /// How many offscreen targets this renderer has allocated, ever.
    ///
    /// One after the first frame that needs one, and one more per resize. Anything else is
    /// the per-frame reallocation this counter exists to make visible.
    #[must_use]
    pub fn offscreen_allocations(&self) -> u32 {
        self.offscreen_allocations
    }

    /// Whether this frame needs the offscreen target.
    ///
    /// Derived from the draw list rather than declared on it, and that is acceptance's "the
    /// draw-list handoff is unchanged" holding: the UI thread does not learn a second thing.
    /// A primitive that samples its neighbourhood says so on [`PrimKind::needs_backdrop`],
    /// and the renderer notices.
    ///
    /// Scanning the instance list is O(n) over a few thousand instances, which is far cheaper
    /// than the upload that follows it on the same data. Making it O(1) would mean a flag
    /// somebody has to remember to set, and a flag that disagrees with the instances is a
    /// frame that samples an unallocated target.
    fn needs_offscreen(&self, list: &DrawList) -> bool {
        self.force_offscreen || list.instances.iter().any(instance_needs_backdrop)
    }

    /// The blur chain, if one has been allocated. Its memory is reported the way the colour
    /// target's is — `renderer.blur().map_or(0, BlurChain::bytes)`.
    #[must_use]
    pub fn blur(&self) -> Option<&BlurChain> {
        self.blur.as_ref()
    }

    /// Allocate or resize the blur chain if this frame draws a backdrop-sampling primitive.
    ///
    /// Called after [`Renderer::ensure_offscreen`] and reading its allocation count, because
    /// the chain's downsample pass holds a bind group onto the target's view: a resize that
    /// replaces the target has to replace the chain even when the viewport rounds to the same
    /// downsampled size. Like the target, an existing chain is **not** freed on a frame that
    /// happens to draw no panel — that would reallocate once per popover open.
    fn ensure_blur(&mut self, ctx: &GpuContext, list: &DrawList, offscreen: bool) {
        // Keyed on a **blur** instance, not on any backdrop-sampling one. `KIND_REFRACT` also
        // needs the two-pass path, and reads the sharp target rather than the chain -- so a
        // frame whose only glass is refracting would otherwise allocate 3.1 MB and run three
        // full-screen passes that nothing samples. The cut below is still general; only the
        // chain is specific, because only the chain is the blur's.
        if !offscreen || !list.instances.iter().any(instance_is_blur) {
            return;
        }
        let Some(target) = self.offscreen.as_ref() else {
            return;
        };
        let viewport = [list.viewport[0].max(1), list.viewport[1].max(1)];
        let generation = self.offscreen_allocations;
        let fits = self
            .blur
            .as_ref()
            .is_some_and(|c| c.fits(viewport, self.format, generation));
        if !fits {
            self.blur = Some(BlurChain::new(
                &ctx.device,
                &self.backdrop_layout,
                &self.blur_sampler,
                self.format,
                viewport,
                target.view(),
                generation,
            ));
        }
    }

    /// Encode one frame.
    ///
    /// # Two paths, and the first one is still the normal one
    ///
    /// With no primitive asking to sample its neighbourhood, this renders straight to
    /// `target` exactly as it always has: one pass, one command buffer, no extra allocation.
    /// With one, the instance pass renders into an offscreen colour target and a resolve pass
    /// puts it back. `the_two_pass_path_is_pixel_identical_to_the_one_pass_path` asserts the
    /// two produce the same image when nothing has actually sampled the backdrop, which is
    /// what keeps the addition from being a silent regression.
    ///
    /// # The surface/content split (T019)
    ///
    /// The batch sequence is drawn in two halves around [`surface_content_split`]: the
    /// leading run of untextured batches — the **surfaces** — and everything from the first
    /// atlas-sampled batch on — the **content**. The lighting pass, when it lands (US1),
    /// slots exactly between them, which is what makes "text is drawn after lighting and
    /// never lit" (lit-contrast rule 1) a property of this function's shape rather than of
    /// anyone's care. The split is a *cut*, never a re-sort: batches keep their order on both
    /// sides, so composition is untouched and an overlay ground drawn above earlier text
    /// stays above it — unlit, which rule 1 permits; reordered, which it does not, is the
    /// version [`crate::batcher::tests::the_split_is_a_cut_at_the_first_textured_batch`]
    /// goes red on.
    ///
    /// `scene` is the frame's lit-mode geometry, from [`crate::frame::Consumer::scene`].
    /// Today it drives exactly one thing: a renderable scene allocates the lighting target
    /// (T018), so the mode's memory cost appears when the mode does. No pass reads the
    /// target yet.
    pub fn render(
        &mut self,
        ctx: &GpuContext,
        target: &wgpu::TextureView,
        list: &DrawList,
        scene: Option<&SceneList>,
        timing: Option<&mut crate::timing::GpuTimer>,
    ) -> wgpu::CommandBuffer {
        self.ensure_capacity(ctx, list.instances.len() as u64);
        let offscreen = self.ensure_offscreen(ctx, list);
        self.ensure_blur(ctx, list, offscreen);
        self.ensure_lighting(ctx, list, scene);

        // The lit frame's scene, packed and uploaded before the encoder opens, beside the
        // other per-frame writes. An unlit frame writes nothing — the buffer keeps stale
        // bytes nobody reads, because the draw below is guarded by the same condition.
        let lit = scene.filter(|s| s.is_renderable());
        if let Some(scene) = lit {
            let (uniform, dropped) = crate::lighting::LitSceneUniform::pack(
                scene,
                [list.viewport[0] as f32, list.viewport[1] as f32],
            );
            if dropped > 0 && self.lit_dropped == 0 {
                tracing::warn!(
                    target: "qs::lighting",
                    dropped,
                    "the scene exceeds the shader's slab bound; the last {dropped} slabs \
                     cast no shadow this frame"
                );
            }
            self.lit_dropped = dropped;
            ctx.queue
                .write_buffer(&self.lit_buffer, 0, bytemuck::bytes_of(&uniform));
        }

        ctx.queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::bytes_of(&Globals {
                viewport: [list.viewport[0] as f32, list.viewport[1] as f32],
                _pad: [0.0; 2],
                env_horizon: list.environment.horizon.to_premul_linear_f32(),
                env_zenith: list.environment.zenith.to_premul_linear_f32(),
                field_place: field_of(list, |c| [c.at[0], c.at[1], c.drift[0], c.drift[1]]),
                field_tint: field_of(list, |c| c.tint.to_premul_linear_f32()),
                field_form: field_of(list, |c| [c.reach, c.phase, 0.0, 0.0]),
            }),
        );
        if !list.instances.is_empty() {
            ctx.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&list.instances),
            );
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qs-frame"),
            });

        let timestamp_writes = timing.and_then(|t| t.begin(&mut encoder));

        // Where the instance pass draws. The offscreen target when something will sample it,
        // the surface otherwise -- and the branch is here, once, rather than duplicated as
        // two copies of the batch loop that could drift apart.
        let instance_target = match (offscreen, self.offscreen.as_ref()) {
            (true, Some(t)) => t.view(),
            _ => target,
        };

        // The two cuts in the batch sequence, and they are independent.
        //
        // `lit_at` is US1's surface/content seam. `backdrop_at` is where the frame stops being
        // *the backdrop*: the first batch carrying a primitive that samples what is behind it.
        // Everything before it goes into the offscreen target, the chain blurs THAT, and the
        // rest is drawn on the surface afterwards. Without the cut, the target a blur sampled
        // would contain the panel itself, and the effect would be a feedback loop rather than
        // depth.
        //
        // With no such primitive -- including the `force_offscreen` case, which is what keeps
        // the path exercised -- `backdrop_at` is the end of the list, the whole frame goes into
        // the target and the second pass is exactly the resolve it has always been. That is
        // what lets `the_two_pass_path_is_pixel_identical_to_the_one_pass_path` keep meaning
        // what it meant.
        let lit_at = surface_content_split(&list.batches);
        let backdrop_at = if offscreen {
            backdrop_split(list)
        } else {
            u32::MAX
        };
        // Two questions, because they became different ones when a second backdrop-sampling
        // primitive arrived. `sampling_backdrop` is whether anything at all is drawn after the
        // cut -- the surface span exists for a refracting panel exactly as it does for a
        // blurring one. `blurring` is whether the chain has to run, which only a blur asks for.
        let sampling_backdrop = backdrop_at != u32::MAX;
        let blurring = sampling_backdrop && list.instances.iter().any(instance_is_blur);

        {
            let clear = list.clear;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("qs-main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: instance_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // The clear colour goes through the same linear conversion as every
                        // other colour; wgpu treats `Color` as linear and the sRGB surface
                        // format applies the transfer function on write.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(crate::color::srgb_to_linear(clear.r)),
                            g: f64::from(crate::color::srgb_to_linear(clear.g)),
                            b: f64::from(crate::color::srgb_to_linear(clear.b)),
                            a: f64::from(clear.a),
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Everything up to the backdrop cut. On the ordinary frame that is every batch,
            // and this call is the whole of `render` as it was.
            self.draw_span(
                &mut pass,
                list,
                0..backdrop_at,
                lit.is_some().then_some(lit_at),
                self.backdrop_placeholder(),
                // The placeholder at group 3 as well, and here it is not merely unread: this
                // pass is rendering INTO the offscreen target, and binding a texture one is
                // writing is the one thing the sharp backdrop cannot do. Nothing in this span
                // samples it -- the cut is defined as the first instance that would.
                self.backdrop_placeholder(),
            );
            // The panel and everything above it are NOT here; they are drawn in the resolve
            // pass below, over the surface, with the blur bound. `backdrop_at` is u32::MAX
            // when nothing samples a backdrop, and this span is then the whole frame.
        }

        // The chain: downsample the backdrop, then one Gaussian per axis, all at a quarter of
        // the viewport in each direction. Three passes rather than two because the downsample
        // is exact only as its own step -- folding it into the horizontal blur would make the
        // first axis sample a full-resolution image with a quarter-resolution kernel, which is
        // aliasing dressed as an optimisation.
        if blurring {
            if let Some(blur) = self.blur.as_ref() {
                let mut chain_pass =
                    |label: &str,
                     pipeline: &wgpu::RenderPipeline,
                     source: &wgpu::BindGroup,
                     into: &wgpu::TextureView| {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some(label),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: into,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    // `Load` for the resolve's reason: the triangle covers every
                                    // texel of its target, so a clear would be a full write
                                    // immediately overwritten.
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, source, &[]);
                        pass.draw(0..3, 0..1);
                    };
                // Full-resolution backdrop -> ping, at a quarter of the size.
                chain_pass(
                    "qs-blur-downsample-pass",
                    &self.blur_downsample_pipeline,
                    blur.source_bind_group(),
                    blur.pong(),
                );
                // ping <- horizontal(pong), pong <- ... -- the names follow the ping-pong and
                // the LAST write lands in `ping`, which is what the instance pipeline samples.
                chain_pass(
                    "qs-blur-horizontal-pass",
                    &self.blur_h_pipeline,
                    blur.pong_bind_group(),
                    blur.ping(),
                );
                chain_pass(
                    "qs-blur-vertical-pass",
                    &self.blur_v_pipeline,
                    blur.ping_bind_group(),
                    blur.pong(),
                );
            }
        }

        // The resolve, and everything from the backdrop cut on. Nothing at all on the
        // single-pass path, which is the shape acceptance asks for: the existing path stays the
        // path when no effect wants the target.
        if let (true, Some(offscreen)) = (offscreen, self.offscreen.as_ref()) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("qs-resolve-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // `Load`, not `Clear`. The triangle covers every pixel, so clearing
                        // would be a full-surface write that is immediately overwritten --
                        // and on a tiler it would also discard the very contents some later
                        // effect might want. Nothing here depends on the prior contents; the
                        // saving is the point.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                // The timestamp pair belongs to the instance pass. A second pair would need a
                // second query slot and would report the resolve separately, which is worth
                // doing when the resolve stops being a copy and not before.
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, offscreen.bind_group(), &[]);
            pass.set_scissor_rect(0, 0, list.viewport[0].max(1), list.viewport[1].max(1));
            pass.draw(0..3, 0..1);

            // The panel and everything above it, drawn onto the surface the resolve just
            // restored, with the blurred backdrop bound. Empty unless something asked for a
            // backdrop -- so a forced two-pass frame emits the resolve and stops, exactly as
            // it did before this chunk.
            if sampling_backdrop {
                self.draw_span(
                    &mut pass,
                    list,
                    backdrop_at..u32::MAX,
                    // The lighting seam is NOT re-emitted here. It belongs to the surfaces it
                    // modulates, which are behind the panel and were drawn in the first pass;
                    // emitting it again would light the panel and everything above it, which
                    // is lit-contrast rule 1 broken in the loudest possible way -- the glyphs
                    // on a popover are content, and content is drawn after lighting and never
                    // lit. A frame whose seam falls after the backdrop cut therefore lights
                    // nothing beyond it, which is the conservative direction: rule 1 permits
                    // an unlit surface and forbids a lit glyph.
                    None,
                    self.blurred_bind_group(),
                    // The offscreen target itself, at full resolution: what is behind the
                    // panel, cut at the same instance as the blurred copy beside it, with
                    // nothing done to it. The resolve above has already copied it to the
                    // surface, so it is finished being written and is safe to sample.
                    offscreen.bind_group(),
                );
            }
        }

        encoder.finish()
    }

    /// What group 2 binds: the finished blur when the chain ran this frame, the placeholder
    /// otherwise.
    fn blurred_bind_group(&self) -> &wgpu::BindGroup {
        self.blur
            .as_ref()
            .map_or(&self.backdrop_placeholder, BlurChain::pong_bind_group)
    }

    fn backdrop_placeholder(&self) -> &wgpu::BindGroup {
        &self.backdrop_placeholder
    }

    /// Draw `span` of `list`'s batches through the instance pipeline, emitting the lighting
    /// triangle when `lit_at` falls inside the span.
    ///
    /// A span rather than the whole list, because there are now two cuts in the sequence and
    /// the second one ends a render pass. Passing the lighting index in rather than recomputing
    /// it is what keeps the seam in one place: a helper that found its own cut would find it
    /// relative to the span and put the light in the middle of the second half.
    /// Draw the instances of `list` inside `span`, emitting the lighting triangle at the
    /// batch boundary `lit_at` when it falls in this pass.
    ///
    /// Two indices in two different units, which is not an oversight. The lighting seam is a
    /// **batch** boundary because lit-contrast rule 1 is about atlas-sampled batches — text is
    /// drawn after lighting — and a batch is exactly what carries "samples the atlas". The
    /// backdrop cut is an **instance** index because nothing groups instances by kind, and a
    /// panel is routinely in the same batch as the rows behind it; see [`backdrop_split`] for
    /// what cutting it at the batch actually produced.
    fn draw_span(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        list: &DrawList,
        span: std::ops::Range<u32>,
        lit_at: Option<usize>,
        backdrop: &wgpu::BindGroup,
        sharp: &wgpu::BindGroup,
    ) {
        let bind_instances = |pass: &mut wgpu::RenderPass<'_>| {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.globals_bind_group, &[]);
            pass.set_bind_group(1, &self.atlas_bind_group, &[]);
            pass.set_bind_group(2, backdrop, &[]);
            pass.set_bind_group(3, sharp, &[]);
            pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
        };
        bind_instances(pass);

        // `split_at`, clamped, rather than two slice expressions: the caller hands in an index
        // it computed from a *different* batch list in principle, and a seam past the end
        // should light everything rather than panic mid-frame.
        let cut = lit_at.unwrap_or(list.batches.len()).min(list.batches.len());
        let (before_seam, after_seam) = list.batches.split_at(cut);
        draw_batches_within(pass, list, before_seam, span.clone());

        // The lighting pass, in the seam (US1): it modulates the surfaces just drawn
        // and is finished before any glyph exists to be lit — lit-contrast rule 1 as
        // draw order. One fullscreen triangle inside the SAME wgpu pass, so an unlit
        // frame's command stream is exactly what it always was; with a scene, the only
        // additions are one pipeline switch each way and one draw.
        if lit_at.is_some() {
            pass.set_pipeline(&self.lit_pipeline);
            pass.set_bind_group(0, &self.lit_bind_group, &[]);
            pass.set_scissor_rect(0, 0, list.viewport[0].max(1), list.viewport[1].max(1));
            pass.draw(0..3, 0..1);

            // The instance pipeline back, for the content half: pipeline, all three bind
            // groups and the vertex buffer, because a render pass forgets nothing but
            // guarantees nothing across a pipeline switch.
            bind_instances(pass);
        }

        draw_batches_within(pass, list, after_seam, span);
    }

    /// The lighting target, if one has been allocated.
    #[must_use]
    pub fn lighting_target(&self) -> Option<&LightingTarget> {
        self.lighting.as_ref()
    }

    /// How many lighting targets this renderer has allocated, ever. See
    /// [`Renderer::offscreen_allocations`] for why this is counted.
    #[must_use]
    pub fn lighting_allocations(&self) -> u32 {
        self.lighting_allocations
    }

    /// Allocate or resize the lighting target if this frame carries a renderable scene.
    ///
    /// The same lazy discipline as [`Renderer::ensure_offscreen`], keyed on the scene
    /// rather than the draw list: the lit mode's memory cost appears when the mode does and
    /// never before. An existing target is kept across unlit frames for the same
    /// reallocation-per-scroll reason the colour target is.
    fn ensure_lighting(&mut self, ctx: &GpuContext, list: &DrawList, scene: Option<&SceneList>) {
        if !scene.is_some_and(SceneList::is_renderable) {
            return;
        }
        let size = [list.viewport[0].max(1), list.viewport[1].max(1)];
        let fits = self.lighting.as_ref().is_some_and(|t| t.fits(size));
        if !fits {
            self.lighting_allocations = self.lighting_allocations.saturating_add(1);
            self.lighting = Some(LightingTarget::new(&ctx.device, size));
        }
    }

    /// Allocate or resize the offscreen target if this frame needs one.
    ///
    /// Returns whether the two-pass path should run. Returning a bool rather than a
    /// reference is what keeps this callable from `render` without borrowing `self` for the
    /// rest of the frame — the target is read back out of `self.offscreen` at each use.
    fn ensure_offscreen(&mut self, ctx: &GpuContext, list: &DrawList) -> bool {
        if !self.needs_offscreen(list) {
            // Deliberately does NOT free an existing target. A frame that happens to contain
            // no blurred surface is followed by one that does, and freeing on the first would
            // reallocate on the second -- once per scroll past a popover. The target is freed
            // when the renderer is, or when a resize replaces it.
            return false;
        }
        let size = [list.viewport[0].max(1), list.viewport[1].max(1)];
        let fits = self
            .offscreen
            .as_ref()
            .is_some_and(|t| t.fits(size, self.format));
        if !fits {
            self.offscreen_allocations = self.offscreen_allocations.saturating_add(1);
            self.offscreen = Some(OffscreenTarget::new(
                &ctx.device,
                &self.resolve_layout,
                &self.resolve_sampler,
                self.format,
                size,
            ));
        }
        true
    }
}

/// Where the batch sequence divides into surfaces and content (T019).
///
/// The index of the first atlas-sampled batch: everything before it is a surface the
/// lighting pass may modulate, everything from it on is content — drawn after lighting,
/// never lit (lit-contrast rule 1). A **cut, not a partition by flag**: an untextured batch
/// *after* the first textured one stays in the content half, because moving it would
/// reorder composition — an overlay's ground drawn above a lower layer's text has to stay
/// above it. The price is that such a ground goes unlit, which rule 1 permits; the
/// alternative prices are a reordered frame or lit glyphs, and both are defects.
fn surface_content_split(batches: &[Batch]) -> usize {
    batches
        .iter()
        .position(|batch| batch.textured)
        .unwrap_or(batches.len())
}

/// Whether this instance samples what is behind it.
///
/// Named rather than inlined because two places ask it — whether the frame needs a target, and
/// whether it needs a chain — and the two answering differently is a frame that blurs a texture
/// nothing filled.
fn instance_needs_backdrop(instance: &Instance) -> bool {
    PrimKind::from_raw(instance.kind).is_some_and(PrimKind::needs_backdrop)
}

/// Whether this instance is the one kind that reads the blur chain.
///
/// Spelled against the kind rather than against `needs_backdrop`, because those stopped being
/// the same question when `KIND_REFRACT` landed: it needs the target and does not need the
/// chain. Asking `needs_backdrop` here is the mistake that runs three blur passes for a panel
/// that never samples their output.
fn instance_is_blur(instance: &Instance) -> bool {
    instance.kind == PrimKind::Blur as u32
}

/// Where the batch sequence stops being **the backdrop**.
///
/// The index of the first batch carrying a primitive that samples what is behind it, or the
/// end of the list when none does. Everything before this index is drawn into the offscreen
/// target and blurred; everything from it on is drawn afterwards, over the resolved surface,
/// with that blur bound.
///
/// A **cut, not a partition**, for exactly [`surface_content_split`]'s reason and with a
/// sharper consequence: a later instance that samples nothing stays on the far side of the
/// cut, because it was authored to sit *above* the panel and moving it under would reorder
/// composition. The price is that such an instance is not part of any panel's backdrop — which
/// is correct, since it is drawn after the panel and a backdrop is what is behind.
///
/// # An INSTANCE index, and the first version was a batch index
///
/// A batch is a run sharing a scissor rect and a texture binding, and **nothing groups
/// instances by kind** — so a panel is very often in the same batch as the rows behind it. Cut
/// at the batch, that panel's backdrop is everything before its batch, which for a list drawn
/// in one batch is *nothing at all*: the target holds the clear colour, the chain blurs a flat
/// field, and the panel renders as its own tint over a uniform ground.
///
/// That is not a hypothetical. The first build cut at the batch, and `blur_panel` — one
/// `end_batch` at the end, like any small draw list — produced two panels three levels apart
/// out of 255 and a chain that had allocated, run three passes and been given a blank image to
/// blur. Every test was green. The failure mode is exactly research R15's: an effect that is
/// correct, bounded, allocated, executed, and invisible.
fn backdrop_split(list: &DrawList) -> u32 {
    list.instances
        .iter()
        .position(instance_needs_backdrop)
        .map_or(u32::MAX, |i| i as u32)
}

/// One half of the batch loop. Factored so the two halves around the lighting seam cannot
/// drift apart — the scissor clamping below is exactly the kind of detail a second copy
/// forgets.
/// Draw `batches`, restricted to the instances inside `span`.
///
/// A batch is a run sharing a scissor and a texture binding, and nothing groups it by kind, so
/// the backdrop cut lands **inside** a batch far more often than between two. Clamping each
/// batch's instance range is what lets the cut be an instance rather than a batch, and every
/// batch keeps its own scissor either way — the run is narrowed, never merged or reordered.
fn draw_batches_within(
    pass: &mut wgpu::RenderPass<'_>,
    list: &DrawList,
    batches: &[Batch],
    span: std::ops::Range<u32>,
) {
    for batch in batches {
        let lo = batch.range.start.max(span.start);
        let hi = batch.range.end.min(span.end);
        if lo >= hi {
            continue;
        }
        if let Some([x, y, w, h]) = batch.scissor {
            // A scissor rect outside the surface is a validation error, and a
            // resize race can produce one. Clamping is cheaper than the frame it
            // would otherwise cost.
            let (vw, vh) = (list.viewport[0], list.viewport[1]);
            let x = x.min(vw);
            let y = y.min(vh);
            let w = w.min(vw.saturating_sub(x));
            let h = h.min(vh.saturating_sub(y));
            if w == 0 || h == 0 {
                continue;
            }
            pass.set_scissor_rect(x, y, w, h);
        } else {
            pass.set_scissor_rect(0, 0, list.viewport[0].max(1), list.viewport[1].max(1));
        }
        pass.draw(0..4, lo..hi);
    }
}

/// Mirrors the `InstanceIn` struct in `shaders/instance.wgsl`. The two must agree; a
/// mismatch is a silently wrong render, not a validation error, because the byte counts
/// still line up.
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    0 => Float32x4,  // rect
    1 => Float32x4,  // uv
    2 => Uint32,     // color (premultiplied linear rgba8)
    3 => Float32,    // radius
    4 => Float32,    // param
    5 => Uint32,     // kind
];

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn batch(textured: bool) -> Batch {
        Batch {
            range: 0..1,
            scissor: None,
            textured,
        }
    }

    fn white() -> crate::color::Srgba {
        crate::color::Srgba::new(1.0, 1.0, 1.0, 1.0)
    }

    fn glass() -> crate::color::Srgba {
        crate::color::Srgba::new(0.2, 0.2, 0.2, 0.9)
    }

    /// A list with `behind` plain instances, then a panel, then `above` more -- all in ONE
    /// batch, which is what a small draw list actually produces.
    fn panelled(behind: u32, above: u32) -> DrawList {
        let mut list = DrawList::default();
        list.reset([64, 64], crate::color::Srgba::TRANSPARENT, 1);
        for _ in 0..behind {
            list.instances
                .push(Instance::rect(0.0, 0.0, 8.0, 8.0, 0.0, white()));
        }
        list.instances
            .push(Instance::blur(2.0, 2.0, 32.0, 32.0, 4.0, glass(), white()));
        for _ in 0..above {
            list.instances
                .push(Instance::rect(0.0, 0.0, 4.0, 4.0, 0.0, white()));
        }
        list.end_batch(None, false);
        list
    }

    #[test]
    fn a_panel_sharing_a_batch_with_its_backdrop_still_has_one() {
        // THE DEFECT THIS TEST IS NAMED FOR. `backdrop_split` was a BATCH index first, and
        // every gate in the workspace stayed green while the effect was blank: a list with one
        // batch cut at index 0, so nothing at all reached the offscreen target, the chain
        // blurred the clear colour, and `blur_panel` rendered two panels three levels apart out
        // of 255. Nothing groups instances by kind, so "the panel is in the same batch as the
        // rows behind it" is the normal case and not the corner one.
        for (behind, above) in [(5_u32, 0_u32), (5, 3), (1, 1), (40, 40)] {
            let list = panelled(behind, above);
            assert_eq!(
                backdrop_split(&list),
                behind,
                "the cut has to land on the panel's own instance, not on its batch"
            );
        }
    }

    #[test]
    fn a_frame_with_nothing_behind_the_panel_cuts_at_zero_rather_than_reporting_none() {
        // The panel first in the list: legitimate (a popover over the bare canvas) and the one
        // case where "no backdrop" and "an empty backdrop" have to stay different. The cut is
        // 0, so the target is cleared and blurred -- a flat field, which is the truthful answer
        // -- rather than u32::MAX, which would skip the chain and leave the panel sampling the
        // placeholder.
        let list = panelled(0, 4);
        assert_eq!(backdrop_split(&list), 0);
    }

    #[test]
    fn a_frame_with_no_panel_asks_for_no_cut_and_no_target() {
        // The ordinary frame, and the claim that the whole path is inert without a backdrop
        // primitive: no cut, and nothing that would allocate 8.3 MB.
        let mut list = DrawList::default();
        list.reset([64, 64], crate::color::Srgba::TRANSPARENT, 1);
        for _ in 0..8 {
            list.instances
                .push(Instance::rect(0.0, 0.0, 8.0, 8.0, 0.0, white()));
        }
        list.end_batch(None, false);
        assert_eq!(backdrop_split(&list), u32::MAX);
        assert!(!list.instances.iter().any(instance_needs_backdrop));
    }

    #[test]
    fn no_atlas_sampled_batch_is_drawn_before_the_lighting_seam() {
        // T020, and the property lit-contrast rule 1 depends on: every textured batch sits
        // at or after the split, so the lighting pass that slots into the seam is finished
        // before the first glyph is drawn. Trivially true of `position(first textured)` --
        // which is the point: the test exists for the future reordering that replaces it.
        let sequences: &[&[Batch]] = &[
            &[],
            &[batch(false)],
            &[batch(true)],
            &[batch(false), batch(true)],
            &[batch(false), batch(true), batch(false), batch(true)],
            &[batch(true), batch(false)],
        ];
        for batches in sequences {
            let split = surface_content_split(batches);
            assert!(
                batches.iter().take(split).all(|b| !b.textured),
                "an atlas-sampled batch sits in the surface half, so it would be lit"
            );
        }
    }

    #[test]
    fn the_split_is_a_cut_at_the_first_textured_batch() {
        // The version that goes red on the tempting rewrite: splitting at the LAST
        // untextured batch. That version draws the sandwiched textured batch before the
        // lighting seam -- a lit glyph -- and re-orders nothing else, so only this exact
        // assertion catches it.
        let batches = [batch(false), batch(true), batch(false), batch(true)];
        assert_eq!(surface_content_split(&batches), 1);

        // An overlay ground after text stays in the content half: unlit, but in order.
        let overlay = [batch(false), batch(true), batch(false)];
        assert_eq!(surface_content_split(&overlay), 1);

        // No content at all: everything is surface, the seam is at the end.
        let plain = [batch(false), batch(false)];
        assert_eq!(surface_content_split(&plain), 2);
    }
}
