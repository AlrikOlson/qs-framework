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
use crate::frame::{DrawList, Instance};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct Globals {
    viewport: [f32; 2],
    _pad: [f32; 2],
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
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("format", &self.format)
            .field("instance_capacity", &self.instance_capacity)
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
                visibility: wgpu::ShaderStages::VERTEX,
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

        Self {
            pipeline,
            globals_buffer,
            globals_bind_group,
            atlas_bind_group,
            atlas_texture,
            instance_buffer,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            format,
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

    /// Encode one frame.
    pub fn render(
        &mut self,
        ctx: &GpuContext,
        target: &wgpu::TextureView,
        list: &DrawList,
        timing: Option<&mut crate::timing::GpuTimer>,
    ) -> wgpu::CommandBuffer {
        self.ensure_capacity(ctx, list.instances.len() as u64);

        ctx.queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::bytes_of(&Globals {
                viewport: [list.viewport[0] as f32, list.viewport[1] as f32],
                _pad: [0.0; 2],
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

        {
            let clear = list.clear;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("qs-main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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

            for batch in &list.batches {
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

        encoder.finish()
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
