//! A picture put in the atlas comes out of the real pipeline as that picture.
//!
//! # Why this test and not the parity suite
//!
//! `tier_parity::image_agrees_across_tiers` compares the CPU rasterizer against a Rust
//! transcription of `instance.wgsl`. Both halves are Rust, neither is a GPU, and the WGSL
//! itself is never executed. Everything that separates a shader that *reads correctly* from
//! a shader that *runs correctly* is therefore outside it: the second bind-group entry, the
//! texture's `Rgba8UnormSrgb` format, the `write_texture` row stride, and whether the
//! sampler reaches the colour page at all.
//!
//! Each of those fails silently in a way the parity suite would keep passing through. A
//! missing binding-2 entry makes the pipeline fail to build, which is loud; but a colour
//! page created as plain `Rgba8Unorm` renders every thumbnail too dark by a gamma curve
//! while every Rust-vs-Rust comparison stays green, and a `bytes_per_row` computed as
//! `width` rather than `width * 4` shears the picture and nothing in the crate would say so.
//!
//! So this runs the shipped shader on a real adapter and looks at the bytes that come back.
//! It is the same reason `crates/qs-gpu/tests/offscreen_target.rs` exists: an image half
//! nothing renders from is an image half nothing is checking.

// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the vocabulary of a
// test, not a hazard in one.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use qs_gpu::atlas::{AtlasBudget, DEFAULT_COLOUR_PAGE, GlyphAtlas, ImageKey, RgbaImage};
use qs_gpu::batcher::Renderer;
use qs_gpu::color::Srgba;
use qs_gpu::device::{GpuContext, new_instance};
use qs_gpu::frame::{DrawList, Instance};
use qs_gpu::path::RenderPath;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
/// The picture's edge. Even, so it halves into four exact quadrants.
const PIC: u32 = 16;

/// A context, or `None` on a machine with no usable adapter.
///
/// Skipped rather than failed, for `offscreen_target.rs`'s reason: a headless runner with no
/// GPU is a real configuration, and a test that fails there teaches people to ignore the
/// suite. The skip prints, so a run that covered nothing is distinguishable from a pass.
fn context() -> Option<GpuContext> {
    let instance = new_instance(RenderPath::Primary);
    match GpuContext::new(RenderPath::Primary, instance, None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("the_colour_page_reaches_the_screen: SKIPPED, no usable adapter ({e})");
            None
        }
    }
}

/// Four opaque quadrants: red, green, blue, white.
///
/// Four *different* colours in four *known* places, which is what makes the two mistakes
/// this test is really about visible. A channel swizzle turns the red quadrant blue and
/// leaves the white one alone. A row-stride error moves the boundary between quadrants
/// without changing which colours appear, so a test that only counted colours would pass.
fn quadrants() -> RgbaImage {
    let half = PIC / 2;
    let mut rgba = Vec::with_capacity((PIC * PIC * 4) as usize);
    for y in 0..PIC {
        for x in 0..PIC {
            let texel = match (x < half, y < half) {
                (true, true) => [255, 0, 0, 255],
                (false, true) => [0, 255, 0, 255],
                (true, false) => [0, 0, 255, 255],
                (false, false) => [255, 255, 255, 255],
            };
            rgba.extend_from_slice(&texel);
        }
    }
    RgbaImage {
        width: PIC,
        height: PIC,
        rgba,
    }
}

/// Render one draw list and read the surface back as RGBA8.
fn draw(ctx: &GpuContext, renderer: &mut Renderer, list: &DrawList) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("colour page test surface"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ctx.capabilities.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let commands = renderer.render(ctx, &view, list, None, None);

    let unpadded = WIDTH * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("colour page readback"),
        size: u64::from(padded) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([commands, encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for row in 0..HEIGHT {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    buffer.unmap();
    pixels
}

/// One pixel as `[r, g, b, a]`, with the surface's channel order normalised to RGBA.
fn at(pixels: &[u8], ctx: &GpuContext, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * WIDTH + x) * 4) as usize;
    let p = [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]];
    // The surface may be Bgra. Reading the raw bytes and calling the first one red is how a
    // test convinces itself of a swizzle that is not there -- or misses one that is.
    if matches!(
        ctx.capabilities.surface_format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        [p[2], p[1], p[0], p[3]]
    } else {
        p
    }
}

/// Which of the three primary channels dominates, as a label. `None` if none does.
fn dominant(px: [u8; 4]) -> Option<&'static str> {
    let (r, g, b) = (u32::from(px[0]), u32::from(px[1]), u32::from(px[2]));
    if r > 128 && g > 128 && b > 128 {
        return Some("white");
    }
    if r > 128 && g < 96 && b < 96 {
        return Some("red");
    }
    if g > 128 && r < 96 && b < 96 {
        return Some("green");
    }
    if b > 128 && r < 96 && g < 96 {
        return Some("blue");
    }
    None
}

#[test]
fn a_picture_admitted_to_the_atlas_comes_out_of_the_pipeline_as_that_picture() {
    let Some(ctx) = context() else { return };

    // The real atlas, so the entry's uv is the one the shipped code computes rather than
    // one this test worked out for itself. A uv computed here would agree with a broken
    // atlas.
    let mut atlas = GlyphAtlas::new(512, 64);
    atlas.begin_frame();
    let entry = atlas
        .get_or_decode_image(ImageKey(1), |_| Some(quadrants()))
        .expect("the atlas refused a 16x16 picture");
    assert_eq!(atlas.colour_size(), DEFAULT_COLOUR_PAGE);

    let mut renderer = Renderer::new(&ctx, 512);
    renderer.upload_images(&ctx, &atlas.take_image_uploads());

    // Drawn at exactly its own size at an integer origin, which is the geometry the
    // pipeline emits: the bilinear taps collapse to nearest and one destination pixel is
    // one source texel.
    let origin = 8.0;
    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.0, 0.0, 0.0, 1.0), 1);
    list.instances.push(Instance::image(
        origin,
        origin,
        PIC as f32,
        PIC as f32,
        entry.uv,
        Srgba::new(1.0, 1.0, 1.0, 1.0),
    ));
    list.end_batch(None, true);

    let pixels = draw(&ctx, &mut renderer, &list);
    let o = origin as u32;
    let q = PIC / 4; // a quarter in, so no sample sits on a quadrant boundary

    for (dx, dy, expected) in [
        (q, q, "red"),
        (PIC - q, q, "green"),
        (q, PIC - q, "blue"),
        (PIC - q, PIC - q, "white"),
    ] {
        let px = at(&pixels, &ctx, o + dx, o + dy);
        assert_eq!(
            dominant(px),
            Some(expected),
            "the {expected} quadrant came back as {px:?}. A wrong colour in the right place \
             is a channel swizzle or a format mistake; the right colours in the wrong places \
             is a row-stride mistake"
        );
    }

    // Outside the picture is the clear colour. A picture that bled past its rectangle would
    // mean the quad is padded -- which is the bug `quad_pad` returning 1.0 for an image
    // would cause, and it is invisible from inside the rectangle.
    let outside = at(&pixels, &ctx, o - 2, o - 2);
    assert!(
        outside[0] < 16 && outside[1] < 16 && outside[2] < 16,
        "the picture put ink outside its own rectangle: {outside:?}"
    );
}

#[test]
fn a_picture_is_lit_by_its_own_bytes_and_not_by_the_glyph_page() {
    // The mistake this catches is one binding: if `colour_texture` resolved to the coverage
    // page -- a plausible copy-paste in `batcher.rs`'s bind group -- every channel would
    // carry the same value, because R8 broadcasts one number. Four *different* colours is
    // what makes that impossible to mistake for success, and it is why the fixture is not a
    // grey ramp.
    let Some(ctx) = context() else { return };

    let mut atlas = GlyphAtlas::new(512, 64);
    atlas.begin_frame();
    let entry = atlas
        .get_or_decode_image(ImageKey(2), |_| Some(quadrants()))
        .expect("the atlas refused a 16x16 picture");

    let mut renderer = Renderer::new(&ctx, 512);
    renderer.upload_images(&ctx, &atlas.take_image_uploads());

    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.0, 0.0, 0.0, 1.0), 1);
    list.instances.push(Instance::image(
        8.0,
        8.0,
        PIC as f32,
        PIC as f32,
        entry.uv,
        Srgba::new(1.0, 1.0, 1.0, 1.0),
    ));
    list.end_batch(None, true);

    let pixels = draw(&ctx, &mut renderer, &list);
    let red = at(&pixels, &ctx, 8 + PIC / 4, 8 + PIC / 4);
    assert!(
        red[0] > red[1] + 64 && red[0] > red[2] + 64,
        "the red quadrant came back as {red:?}, whose channels are too close together to \
         have come from a four-channel texture -- the sampler is reading a single-channel \
         page and broadcasting it"
    );
}

/// A `wide` x `PIC` stripe: red everywhere but its last `PIC` columns, which are green.
///
/// Wider than [`DEFAULT_COLOUR_PAGE`] on purpose -- the shelf allocator places it from the
/// page's left edge, so its green tail lands past the edge of the default page and only a
/// texture sized to the wider page can hold it.
fn stripe(wide: u32) -> RgbaImage {
    let mut rgba = Vec::with_capacity((wide * PIC * 4) as usize);
    for _y in 0..PIC {
        for x in 0..wide {
            let texel = if x >= wide - PIC {
                [0, 255, 0, 255]
            } else {
                [255, 0, 0, 255]
            };
            rgba.extend_from_slice(&texel);
        }
    }
    RgbaImage {
        width: wide,
        height: PIC,
        rgba,
    }
}

#[test]
fn a_picture_past_the_default_page_reaches_the_screen_on_a_renderer_sized_to_the_atlas() {
    // A consumer that widened the colour page (`GlyphAtlas::with_budget`) admits pictures
    // the default page could not hold, and the atlas places them past `DEFAULT_COLOUR_PAGE`
    // texels. A renderer still holding the default-sized texture would `write_texture`
    // past its own edge -- a validation error, not a clip -- so the renderer takes the
    // page's edge from the atlas, exactly as `CpuRasterizer::with_colour_page` does.
    let Some(ctx) = context() else { return };

    let colour_size = 2 * DEFAULT_COLOUR_PAGE;
    let mut atlas = GlyphAtlas::try_with_budget(
        AtlasBudget {
            coverage_size: 512,
            colour_size,
        },
        64,
    )
    .expect("two pages of 512 and 2048 fit the cap");
    atlas.begin_frame();
    let wide = DEFAULT_COLOUR_PAGE + PIC * 4;
    let entry = atlas
        .get_or_decode_image(ImageKey(3), |_| Some(stripe(wide)))
        .expect("the wider page admits a stripe wider than the default page");
    assert_eq!(atlas.colour_size(), colour_size);
    assert!(
        entry.width == wide
            && (entry.uv[2] * colour_size as f32).round() as u32 > DEFAULT_COLOUR_PAGE,
        "the stripe's right edge should sit past the default page's edge: {:?}",
        entry.uv
    );

    let mut renderer = Renderer::with_colour_page(&ctx, 512, atlas.colour_size());
    renderer.upload_images(&ctx, &atlas.take_image_uploads());

    // Two slices of one entry, cut along `u`: the head (red) and the tail (green). The
    // tail is the part that lives past the default page, and a sliced `uv` is how a
    // consumer shows the rows of a picture that are on screen without re-admitting it.
    let [u0, v0, u1, v1] = entry.uv;
    let span = u1 - u0;
    let head = [u0, v0, u0 + span * (PIC as f32 / wide as f32), v1];
    let tail = [u1 - span * (PIC as f32 / wide as f32), v0, u1, v1];

    let mut list = DrawList::default();
    list.reset([WIDTH, HEIGHT], Srgba::new(0.0, 0.0, 0.0, 1.0), 1);
    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    list.instances.push(Instance::image(
        4.0, 4.0, PIC as f32, PIC as f32, head, white,
    ));
    list.instances.push(Instance::image(
        36.0, 36.0, PIC as f32, PIC as f32, tail, white,
    ));
    list.end_batch(None, true);

    let pixels = draw(&ctx, &mut renderer, &list);
    let mid = PIC / 2;
    assert_eq!(
        dominant(at(&pixels, &ctx, 4 + mid, 4 + mid)),
        Some("red"),
        "the head of the stripe should read red"
    );
    assert_eq!(
        dominant(at(&pixels, &ctx, 36 + mid, 36 + mid)),
        Some("green"),
        "the tail of the stripe -- the part past the default page's edge -- should read \
         green; black means the upload never reached the texture"
    );
}
