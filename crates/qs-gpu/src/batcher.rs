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

use crate::atlas::PendingUpload;
use crate::device::GpuContext;
use crate::frame::{Batch, DrawList, FIELD_CENTRES, Instance, PrimKind};
use crate::scene::SceneList;
use crate::target::{LightingTarget, OffscreenTarget};

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
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qs-pipeline-layout"),
            bind_group_layouts: &[Some(&globals_layout), Some(&atlas_layout)],
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

        Self {
            pipeline,
            globals_buffer,
            globals_bind_group,
            atlas_bind_group,
            atlas_texture,
            instance_buffer,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            format,
            resolve_pipeline,
            resolve_layout,
            resolve_sampler,
            offscreen: None,
            force_offscreen: false,
            offscreen_allocations: 0,
            lighting: None,
            lighting_allocations: 0,
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
        self.force_offscreen
            || list
                .instances
                .iter()
                .any(|i| PrimKind::from_raw(i.kind).is_some_and(PrimKind::needs_backdrop))
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
        self.ensure_lighting(ctx, list, scene);

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

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.globals_bind_group, &[]);
            pass.set_bind_group(1, &self.atlas_bind_group, &[]);
            pass.set_vertex_buffer(0, self.instance_buffer.slice(..));

            let split = surface_content_split(&list.batches);
            let (surface_half, content_half) = list.batches.split_at(split);

            // The surface half: every batch before the first atlas-sampled one.
            draw_batches(&mut pass, list, surface_half);

            // The lighting pass slots HERE (US1): it modulates the surfaces just drawn and
            // is finished before any glyph exists to be lit. Today there is nothing to run
            // -- the scene above has allocated the target and no more -- and keeping the
            // seam inside one wgpu pass is what keeps the mode-off frame the exact frame it
            // always was: no second pass begins until something renders into it.

            // The content half: text, icons, and whatever is composed above them, in the
            // order the list stated.
            draw_batches(&mut pass, list, content_half);
        }

        // The resolve. Nothing at all on the single-pass path, which is the shape acceptance
        // asks for: the existing path stays the path when no effect wants the target.
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
        }

        encoder.finish()
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

/// One half of the batch loop. Factored so the two halves around the lighting seam cannot
/// drift apart — the scissor clamping below is exactly the kind of detail a second copy
/// forgets.
fn draw_batches(pass: &mut wgpu::RenderPass<'_>, list: &DrawList, batches: &[Batch]) {
    for batch in batches {
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
        pass.draw(0..4, batch.range.clone());
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
