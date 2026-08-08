//! The no-acceleration tier: `tiny-skia` consuming **the same draw lists**.
//!
//! # RP-2 is the guarantee worth defending
//!
//! > All three tiers consume the same draw lists.
//!
//! The moment this file grows its own layout, its own colour handling, or its own idea of
//! what a row looks like, parity stops being checkable and the CPU fallback quietly becomes
//! a second renderer with its own bugs -- which is exactly the outcome the reference-image
//! suite exists to prevent. So this module takes a [`DrawList`] and a set of
//! [`PendingUpload`]s and nothing else. It cannot know what a row is, because it is never
//! told.
//!
//! # Colour space
//!
//! The pixmap holds **linear** premultiplied RGBA8 while compositing, and is converted to
//! sRGB once at the end. That mirrors what the GPU does: blend in linear, apply the
//! transfer function on write to an sRGB surface. Compositing in sRGB directly -- the
//! obvious thing, since tiny-skia does not care -- would make every antialiased edge and
//! every translucent fill differ from the GPU tiers by more than any sane perceptual
//! tolerance, and the parity suite would fail for a reason that has nothing to do with
//! rasterization.
//!
//! The cost is 8 bits of linear precision during compositing, which bands in very dark
//! greys. It is recorded here rather than discovered: if the parity suite ever fails only
//! in dark-theme shadows, this is why.

use tiny_skia::{
    BlendMode, FillRule, Paint, PathBuilder, Pixmap, PremultipliedColorU8, Rect, Stroke, Transform,
};

use crate::atlas::PendingUpload;
use crate::color::linear_to_srgb;
use crate::frame::{DrawList, Instance, PrimKind};

/// Circle-to-bezier constant. Four cubics with control points at this fraction of the
/// radius approximate a quarter circle to within about 0.02% -- far below a pixel.
const KAPPA: f32 = 0.552_284_8;

pub struct CpuRasterizer {
    pixmap: Pixmap,
    /// CPU-side mirror of the glyph atlas, R8 coverage. The same
    /// [`crate::atlas::GlyphAtlas`] drives both this and the GPU texture, so a glyph is at
    /// the same coordinates in both -- which is what makes the parity comparison meaningful
    /// rather than coincidental.
    atlas: Vec<u8>,
    atlas_size: u32,
}

impl std::fmt::Debug for CpuRasterizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuRasterizer")
            .field("width", &self.pixmap.width())
            .field("height", &self.pixmap.height())
            .field("atlas_size", &self.atlas_size)
            .finish()
    }
}

impl CpuRasterizer {
    pub fn new(width: u32, height: u32, atlas_size: u32) -> Option<Self> {
        let pixmap = Pixmap::new(width.max(1), height.max(1))?;
        Some(Self {
            pixmap,
            atlas: vec![0; (atlas_size as usize).saturating_mul(atlas_size as usize)],
            atlas_size,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> bool {
        match Pixmap::new(width.max(1), height.max(1)) {
            Some(pixmap) => {
                self.pixmap = pixmap;
                true
            }
            None => false,
        }
    }

    pub fn upload_glyphs(&mut self, uploads: &[PendingUpload]) {
        for upload in uploads {
            for row in 0..upload.height {
                let src_start = (row as usize) * (upload.width as usize);
                let Some(src) = upload
                    .coverage
                    .get(src_start..src_start + upload.width as usize)
                else {
                    continue;
                };
                let y = upload.y + row;
                if y >= self.atlas_size {
                    continue;
                }
                let dst_start = (y as usize) * (self.atlas_size as usize) + upload.x as usize;
                let Some(dst) = self
                    .atlas
                    .get_mut(dst_start..dst_start + upload.width as usize)
                else {
                    continue;
                };
                dst.copy_from_slice(src);
            }
        }
    }

    /// Rasterize one frame. The result is linear premultiplied; call
    /// [`CpuRasterizer::to_srgb_rgba8`] to get displayable pixels.
    pub fn render(&mut self, list: &DrawList) -> &Pixmap {
        let clear = list.clear;
        self.pixmap.fill(linear_premul(
            crate::color::srgb_to_linear(clear.r),
            crate::color::srgb_to_linear(clear.g),
            crate::color::srgb_to_linear(clear.b),
            clear.a,
        ));

        for batch in &list.batches {
            let clip = batch
                .scissor
                .and_then(|[x, y, w, h]| Rect::from_xywh(x as f32, y as f32, w as f32, h as f32));

            let start = batch.range.start as usize;
            let end = batch.range.end as usize;
            let Some(instances) = list.instances.get(start..end) else {
                continue;
            };
            for instance in instances {
                self.draw_instance(instance, clip);
            }
        }
        &self.pixmap
    }

    fn draw_instance(&mut self, instance: &Instance, clip: Option<Rect>) {
        let [x, y, w, h] = instance.rect;
        if !(w > 0.0 && h > 0.0) {
            return;
        }
        let (r, g, b, a) = unpack_premul_linear(instance.color);
        if a <= 0.0 {
            return;
        }

        match instance.kind {
            k if k == PrimKind::Glyph as u32 => self.draw_glyph(instance, r, g, b, a, clip),
            k if k == PrimKind::Stroke as u32 => {
                // The shader strokes **inside** the shape: its band spans the signed
                // distance range [-width, 0], so a ring on a rect sits entirely within the
                // rect's bounds. `tiny-skia` strokes **centred** on the path, which would
                // put half the ring outside and break RP-2 -- all three tiers must consume
                // the same draw list and produce the same pixels.
                //
                // Insetting the path by half the stroke width converts a centred stroke
                // into an inside-aligned one. Verified by
                // `the_cpu_stroke_is_inside_aligned_like_the_gpu`, which measured a
                // one-pixel disagreement before this inset existed.
                let half = instance.param.max(0.1) * 0.5;
                let (ix, iy) = (x + half, y + half);
                let (iw, ih) = ((w - instance.param).max(0.1), (h - instance.param).max(0.1));
                let Some(path) = rounded_rect(ix, iy, iw, ih, (instance.radius - half).max(0.0))
                else {
                    return;
                };
                let mut paint = Paint::default();
                // tiny-skia takes straight (non-premultiplied) colour and premultiplies
                // internally, so the instance colour has to be undone first.
                paint.set_color(straight_color(r, g, b, a));
                paint.anti_alias = true;
                paint.blend_mode = BlendMode::SourceOver;
                let stroke = Stroke {
                    width: instance.param.max(0.1),
                    ..Default::default()
                };
                self.pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    Transform::identity(),
                    clip_mask(clip).as_ref(),
                );
            }
            _ => {
                let Some(path) = rounded_rect(x, y, w, h, instance.radius) else {
                    return;
                };
                let mut paint = Paint::default();
                paint.set_color(straight_color(r, g, b, a));
                paint.anti_alias = true;
                paint.blend_mode = BlendMode::SourceOver;
                self.pixmap.fill_path(
                    &path,
                    &paint,
                    FillRule::Winding,
                    Transform::identity(),
                    clip_mask(clip).as_ref(),
                );
            }
        }
    }

    /// Blit a glyph from the atlas mirror.
    ///
    /// Done by hand rather than through tiny-skia's shader pipeline because the operation
    /// is a 1:1 axis-aligned blit of a coverage mask -- the glyph quad's size in pixels
    /// equals its size in atlas texels by construction, since the fractional pen position
    /// is baked into the subpixel variant rather than applied as a transform. Routing it
    /// through a general path fill would resample an already-antialiased mask and soften
    /// every glyph.
    #[allow(clippy::too_many_arguments)]
    fn draw_glyph(
        &mut self,
        instance: &Instance,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        clip: Option<Rect>,
    ) {
        let [x, y, w, h] = instance.rect;
        let size = self.atlas_size as f32;
        let u0 = (instance.uv[0] * size).round() as i64;
        let v0 = (instance.uv[1] * size).round() as i64;

        let dst_x0 = x.round() as i64;
        let dst_y0 = y.round() as i64;
        let width = w.round() as i64;
        let height = h.round() as i64;

        let pw = self.pixmap.width() as i64;
        let ph = self.pixmap.height() as i64;

        // Clip bounds in destination space, so the inner loop has no per-pixel branch.
        let (cx0, cy0, cx1, cy1) = match clip {
            Some(rect) => (
                rect.left() as i64,
                rect.top() as i64,
                rect.right() as i64,
                rect.bottom() as i64,
            ),
            None => (0, 0, pw, ph),
        };

        for row in 0..height {
            let dy = dst_y0 + row;
            if dy < cy0.max(0) || dy >= cy1.min(ph) {
                continue;
            }
            let sy = v0 + row;
            if sy < 0 || sy >= self.atlas_size as i64 {
                continue;
            }
            for col in 0..width {
                let dx = dst_x0 + col;
                if dx < cx0.max(0) || dx >= cx1.min(pw) {
                    continue;
                }
                let sx = u0 + col;
                if sx < 0 || sx >= self.atlas_size as i64 {
                    continue;
                }

                let coverage = self
                    .atlas
                    .get((sy as usize) * (self.atlas_size as usize) + sx as usize)
                    .copied()
                    .unwrap_or(0);
                if coverage == 0 {
                    continue;
                }
                let coverage = f32::from(coverage) / 255.0;

                let index = (dy as usize) * (self.pixmap.width() as usize) + dx as usize;
                let Some(dst) = self.pixmap.pixels_mut().get_mut(index) else {
                    continue;
                };

                // Premultiplied source-over, exactly as the shader does it:
                //   out = src * coverage + dst * (1 - src.a * coverage)
                let sa = a * coverage;
                let inv = 1.0 - sa;
                let dr = f32::from(dst.red()) / 255.0;
                let dg = f32::from(dst.green()) / 255.0;
                let db = f32::from(dst.blue()) / 255.0;
                let da = f32::from(dst.alpha()) / 255.0;

                let out = |s: f32, d: f32| -> u8 {
                    ((s * coverage + d * inv).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
                };
                if let Some(px) = PremultipliedColorU8::from_rgba(
                    out(r, dr),
                    out(g, dg),
                    out(b, db),
                    ((sa + da * inv).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                ) {
                    *dst = px;
                }
            }
        }
    }

    /// Convert the linear working buffer to displayable sRGB RGBA8, un-premultiplied.
    pub fn to_srgb_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixmap.data().len());
        for px in self.pixmap.pixels() {
            let a = f32::from(px.alpha()) / 255.0;
            let unpremul = |c: u8| -> u8 {
                let linear = f32::from(c) / 255.0;
                let straight = if a > 0.0 { linear / a } else { 0.0 };
                (linear_to_srgb(straight.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8
            };
            out.push(unpremul(px.red()));
            out.push(unpremul(px.green()));
            out.push(unpremul(px.blue()));
            out.push(px.alpha());
        }
        out
    }

    pub fn width(&self) -> u32 {
        self.pixmap.width()
    }

    pub fn height(&self) -> u32 {
        self.pixmap.height()
    }
}

fn clip_mask(_clip: Option<Rect>) -> Option<tiny_skia::Mask> {
    // tiny-skia clips through a `Mask`, which would mean allocating and filling a
    // full-surface alpha mask per batch. For M0 the scissor rects are handled by clamping
    // geometry instead: every batch's instances are already inside their column, because
    // the row builder does middle-ellipsis truncation rather than relying on clipping.
    // Recorded rather than silently omitted -- if a future primitive genuinely overflows
    // its batch's scissor, the GPU tiers will clip it and this one will not, and the parity
    // suite will catch it as a diff.
    None
}

fn unpack_premul_linear(packed: u32) -> (f32, f32, f32, f32) {
    let byte = |shift: u32| f32::from(((packed >> shift) & 0xFF) as u8) / 255.0;
    (byte(0), byte(8), byte(16), byte(24))
}

fn straight_color(r: f32, g: f32, b: f32, a: f32) -> tiny_skia::Color {
    let unpremul = |c: f32| {
        if a > 0.0 {
            (c / a).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    tiny_skia::Color::from_rgba(unpremul(r), unpremul(g), unpremul(b), a)
        .unwrap_or(tiny_skia::Color::TRANSPARENT)
}

fn linear_premul(r: f32, g: f32, b: f32, a: f32) -> tiny_skia::Color {
    tiny_skia::Color::from_rgba(
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        a.clamp(0.0, 1.0),
    )
    .unwrap_or(tiny_skia::Color::TRANSPARENT)
}

/// Build a rounded-rectangle path. `radius` is clamped to what the rectangle can hold, for
/// the same reason the shader clamps it: an over-large radius inverts the corners.
fn rounded_rect(x: f32, y: f32, w: f32, h: f32, radius: f32) -> Option<tiny_skia::Path> {
    let r = radius.clamp(0.0, (w.min(h)) * 0.5);
    let mut pb = PathBuilder::new();

    if r <= 0.01 {
        pb.push_rect(Rect::from_xywh(x, y, w, h)?);
        return pb.finish();
    }

    let c = r * KAPPA;
    let (l, t, right, bottom) = (x, y, x + w, y + h);

    pb.move_to(l + r, t);
    pb.line_to(right - r, t);
    pb.cubic_to(right - r + c, t, right, t + r - c, right, t + r);
    pb.line_to(right, bottom - r);
    pb.cubic_to(
        right,
        bottom - r + c,
        right - r + c,
        bottom,
        right - r,
        bottom,
    );
    pb.line_to(l + r, bottom);
    pb.cubic_to(l + r - c, bottom, l, bottom - r + c, l, bottom - r);
    pb.line_to(l, t + r);
    pb.cubic_to(l, t + r - c, l + r - c, t, l + r, t);
    pb.close();
    pb.finish()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use super::*;
    use crate::color::Srgba;

    #[test]
    fn a_filled_rect_lands_where_the_draw_list_says() {
        let mut r = CpuRasterizer::new(64, 64, 64).unwrap();
        let mut list = DrawList::default();
        list.reset([64, 64], Srgba::new(0.0, 0.0, 0.0, 1.0), 1);
        list.instances.push(Instance::rect(
            16.0,
            16.0,
            32.0,
            32.0,
            0.0,
            Srgba::new(1.0, 1.0, 1.0, 1.0),
        ));
        list.end_batch(None, false);

        let pixmap = r.render(&list);
        let at = |x: u32, y: u32| pixmap.pixels()[(y * 64 + x) as usize];

        assert!(at(32, 32).red() > 200, "the centre should be white");
        assert_eq!(at(4, 4).red(), 0, "outside the rect should be background");
    }

    #[test]
    fn an_over_large_radius_does_not_invert_the_shape() {
        // A radius larger than half the shorter side must clamp, matching the shader.
        let path = rounded_rect(0.0, 0.0, 10.0, 10.0, 1000.0);
        assert!(path.is_some());

        let mut r = CpuRasterizer::new(16, 16, 16).unwrap();
        let mut list = DrawList::default();
        list.reset([16, 16], Srgba::TRANSPARENT, 1);
        list.instances.push(Instance::rect(
            0.0,
            0.0,
            16.0,
            16.0,
            1000.0,
            Srgba::new(1.0, 1.0, 1.0, 1.0),
        ));
        list.end_batch(None, false);
        let pixmap = r.render(&list);
        // A clamped radius on a square yields a circle: the centre is covered, the corner
        // is not. An inverted SDF would give the opposite.
        assert!(pixmap.pixels()[(8 * 16 + 8) as usize].alpha() > 200);
        assert!(pixmap.pixels()[0].alpha() < 60);
    }

    #[test]
    fn a_zero_sized_instance_is_skipped_rather_than_panicking() {
        let mut r = CpuRasterizer::new(8, 8, 8).unwrap();
        let mut list = DrawList::default();
        list.reset([8, 8], Srgba::TRANSPARENT, 1);
        list.instances.push(Instance::rect(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            Srgba::new(1.0, 0.0, 0.0, 1.0),
        ));
        list.instances.push(Instance::rect(
            0.0,
            0.0,
            -5.0,
            4.0,
            0.0,
            Srgba::new(1.0, 0.0, 0.0, 1.0),
        ));
        list.end_batch(None, false);
        r.render(&list);
    }

    #[test]
    fn a_fully_transparent_instance_draws_nothing() {
        let mut r = CpuRasterizer::new(8, 8, 8).unwrap();
        let mut list = DrawList::default();
        list.reset([8, 8], Srgba::TRANSPARENT, 1);
        list.instances
            .push(Instance::rect(0.0, 0.0, 8.0, 8.0, 0.0, Srgba::TRANSPARENT));
        list.end_batch(None, false);
        let pixmap = r.render(&list);
        assert!(pixmap.pixels().iter().all(|p| p.alpha() == 0));
    }

    #[test]
    fn glyph_uploads_reach_the_atlas_mirror() {
        let mut r = CpuRasterizer::new(16, 16, 16).unwrap();
        r.upload_glyphs(&[PendingUpload {
            x: 2,
            y: 3,
            width: 2,
            height: 2,
            coverage: vec![255, 128, 64, 0],
        }]);
        assert_eq!(r.atlas[3 * 16 + 2], 255);
        assert_eq!(r.atlas[3 * 16 + 3], 128);
        assert_eq!(r.atlas[4 * 16 + 2], 64);
    }

    #[test]
    fn an_upload_that_would_run_off_the_atlas_is_clipped_not_wrapped() {
        let mut r = CpuRasterizer::new(8, 8, 8).unwrap();
        r.upload_glyphs(&[PendingUpload {
            x: 0,
            y: 7,
            width: 4,
            height: 4,
            coverage: vec![255; 16],
        }]);
        // Row 7 is written; rows 8..10 do not exist and must not have wrapped to row 0.
        assert_eq!(r.atlas[7 * 8], 255);
        assert_eq!(r.atlas[0], 0);
    }

    /// The GPU's stroke coverage, evaluated in Rust so the two tiers can be compared
    /// without a device. Mirrors the `KIND_STROKE` branch of `shaders/instance.wgsl`
    /// exactly; if that shader changes, this must change with it.
    fn gpu_stroke_alpha(distance: f32, width: f32) -> f32 {
        let half_width = width * 0.5;
        (0.5 - ((distance + half_width).abs() - half_width)).clamp(0.0, 1.0)
    }

    #[test]
    fn the_gpu_stroke_is_inside_aligned() {
        // Pin the alignment the shader actually implements. The band spans the signed
        // distance range [-width, 0]: fully inside the shape, touching the edge. An
        // outside- or centre-aligned ring would bleed past the rect it is meant to mark.
        let w = 2.0;
        assert!(
            gpu_stroke_alpha(0.0, w) > 0.4 && gpu_stroke_alpha(0.0, w) < 0.6,
            "half covered at the edge"
        );
        assert_eq!(
            gpu_stroke_alpha(-1.0, w),
            1.0,
            "fully covered at the band centre"
        );
        assert!(
            gpu_stroke_alpha(-2.0, w) < 0.6,
            "band ends one width inside"
        );
        assert_eq!(gpu_stroke_alpha(1.0, w), 0.0, "nothing outside the shape");
        assert_eq!(
            gpu_stroke_alpha(-3.0, w),
            0.0,
            "nothing deeper than the band"
        );
    }

    /// Horizontal extent of any ink on the given row of the pixmap.
    fn ink_span(pixmap: &Pixmap, y: u32) -> Option<(u32, u32)> {
        let w = pixmap.width();
        let mut first = None;
        let mut last = 0;
        for x in 0..w {
            let px = pixmap.pixels()[(y * w + x) as usize];
            if px.alpha() > 8 {
                first.get_or_insert(x);
                last = x;
            }
        }
        first.map(|f| (f, last))
    }

    #[test]
    fn the_cpu_stroke_is_inside_aligned_like_the_gpu() {
        // RP-2: all three tiers consume the same draw lists, so a stroke must land in the
        // same pixels on each. tiny-skia strokes CENTRED on the path by default while the
        // shader strokes INSIDE, so the CPU path has to inset the geometry to match. This
        // test is what caught that: before the inset, the ink started 1px outside the rect.
        let mut r = CpuRasterizer::new(40, 40, 8).unwrap();
        let mut list = DrawList::default();
        list.reset([40, 40], Srgba::TRANSPARENT, 1);
        // A 20x20 rect at (10,10), 2px ring, no corner radius so the edges are exact.
        list.instances.push(Instance::stroke(
            10.0,
            10.0,
            20.0,
            20.0,
            0.0,
            2.0,
            Srgba::new(1.0, 1.0, 1.0, 1.0),
        ));
        list.end_batch(None, false);
        let pixmap = r.render(&list);

        // Scan a row through the middle of the ring's left and right verticals.
        let (first, last) = ink_span(pixmap, 20).expect("the ring must draw something");
        assert_eq!(
            first, 10,
            "the ring must start at the rect's left edge, not outside it"
        );
        assert_eq!(
            last, 29,
            "the ring must end at the rect's right edge, not outside it"
        );
    }

    #[test]
    fn the_srgb_conversion_produces_four_bytes_per_pixel() {
        let r = CpuRasterizer::new(4, 4, 8).unwrap();
        assert_eq!(r.to_srgb_rgba8().len(), 4 * 4 * 4);
    }
}
