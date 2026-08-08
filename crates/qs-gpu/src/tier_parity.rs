//! Per-primitive agreement between the CPU tier and the shader, with no GPU device.
//!
//! # Why this exists
//!
//! RP-2 says all three tiers consume the same draw lists. Consuming the same list is not
//! the same as producing the same pixels, and the gap between those two statements is
//! where [`PrimKind::Stroke`] lived for a while: the shader strokes *inside* the shape,
//! `tiny-skia` strokes *centred* on the path, and the two disagreed by a pixel on every
//! edge. Nothing caught it, because nothing had ever emitted a stroke.
//!
//! The thing that *would* have caught it is the cross-tier reference-image suite, which
//! needs a GPU device and a perceptual threshold that is still uncalibrated (research
//! R10). So this module takes the cheap half of that job and does it now: it transcribes
//! `shaders/instance.wgsl` into Rust, evaluates it per pixel, and compares the result
//! against [`CpuRasterizer`]'s output. No device, milliseconds, and a failure names one
//! primitive instead of one frame.
//!
//! It does **not** replace the reference-image suite. It cannot see anything that happens
//! above the draw list -- wrong colour, wrong position, a row that forgot to emit -- and
//! it cannot see driver-specific behaviour, because there is no driver. What it sees is
//! exactly one class of bug: the two tiers implementing the same primitive differently.
//!
//! # Why a transcription is possible at all
//!
//! `fs_main` has no `fwidth`, no `smoothstep` and no derivative of any kind -- the comment
//! at line 124 of the shader explains that `0.5 - d` is exact one-pixel coverage for a
//! distance field measured in pixels, and says so *because* it is what keeps the CPU tier
//! correct. That same property is what makes the shader reproducible in ordinary Rust: a
//! fragment's output depends only on its own interpolated inputs.
//!
//! # What the numbers can and cannot mean
//!
//! Two things make the tiers differ for reasons that are not bugs, and both were found by
//! building this suite rather than assumed beforehand.
//!
//! The first is definitional. `tiny-skia` computes analytic *area* coverage; the shader
//! computes `0.5` minus a *distance*. On a straight edge those are the same number. On a
//! curve they are two different exact answers to slightly different questions, and they
//! diverge in proportion to curvature -- which is why the rounded fixtures carry a larger
//! bound than the sharp ones.
//!
//! The second is [`CPU_EDGE_QUANTUM`]: `tiny-skia` resolves an edge to the nearest quarter
//! pixel, and the shader's distance field does not resolve it to anything, because it is
//! continuous. Geometry landing on a quarter therefore agrees to the last bit, and
//! geometry landing on an eighth -- a 1.875px stroke at 125% display scale, say -- does
//! not. That is a property of the fallback tier, measured here and bounded here.
//!
//! Every bound below is derived from one of those two numbers and stated in the fixture
//! that uses it. None was raised until a test went green.
//!
//! # What the bounds were checked against
//!
//! Nine mutations, each applied for real and re-run: the CPU tier shifted a whole pixel;
//! shifted a *quarter* pixel; the stroke inset deleted (the original bug); the glyph blit
//! reading one texel across; the transcription's stroke band turned into a fill; its
//! radius clamp deleted; a `PrimKind` dropped from `ALL`; a `cases_for` arm stubbed out to
//! nothing; and `CPU_EDGE_QUANTUM` sharpened. All nine fail the suite. The radius clamp
//! was the one that initially did *not* -- no fixture asked for a radius large enough to
//! clamp, so deleting the clamp changed nothing anywhere. `rect/over-large-radius` and
//! `stroke/over-large-radius` exist because of that survivor.
//!
//! A tolerance on its own is a weak assertion, so the comparison does not rest on one.
//! Alongside the channel difference it pins the ink's bounding box *exactly*, its
//! coverage-weighted centroid to a fraction of a pixel, and the total ink laid down to a
//! fraction of a percent. That last one carries the weight where the others cannot: the
//! quarter-pixel step takes from one edge exactly what it gives the opposite edge, so it
//! moves the centroid and cannot move the total.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use tiny_skia::Pixmap;

use crate::atlas::PendingUpload;
use crate::color::Srgba;
use crate::cpu_raster::CpuRasterizer;
use crate::frame::{DrawList, Instance, PrimKind};

/// `shaders/instance.wgsl`, transcribed function for function.
///
/// The transcription is the whole point, so it is kept boring and literal: same names,
/// same order, same operations, one Rust function per WGSL function, with the shader's
/// line numbers in the doc comments. If it were rewritten into idiomatic Rust it would
/// stop being evidence about the shader and start being a second opinion about geometry.
///
/// **If `instance.wgsl` changes, this must change with it.** Nothing enforces that
/// automatically; what the suite does enforce is that the *set of primitives* on both
/// sides matches, which is the drift that has actually happened.
mod shader {
    /// `const KIND_STROKE: u32 = 1u;` -- shader line 15.
    pub const KIND_STROKE: u32 = 1;
    /// `const KIND_GLYPH: u32 = 2u;` -- shader line 16.
    pub const KIND_GLYPH: u32 = 2;
    // `KIND_RECT` (line 14) has no constant here on purpose: the shader reaches it as the
    // `else` of both branches, and a constant nothing compares against is dead weight
    // that would still have to be kept in sync.

    /// WGSL `unpack4x8unorm`: byte 0 of the word becomes component `x`.
    ///
    /// Worth transcribing rather than assuming -- the packing order is the one place a
    /// silent red/blue swap between the tiers could hide, since both tiers would still
    /// produce a plausible picture.
    pub fn unpack4x8unorm(v: u32) -> [f32; 4] {
        [
            f32::from(((v) & 0xFF) as u8) / 255.0,
            f32::from(((v >> 8) & 0xFF) as u8) / 255.0,
            f32::from(((v >> 16) & 0xFF) as u8) / 255.0,
            f32::from(((v >> 24) & 0xFF) as u8) / 255.0,
        ]
    }

    /// `fn sd_rounded_box(p, b, r)` -- shader line 96. Inigo Quilez's formulation,
    /// negative inside.
    pub fn sd_rounded_box(p: [f32; 2], b: [f32; 2], r: f32) -> f32 {
        let q = [p[0].abs() - b[0] + r, p[1].abs() - b[1] + r];
        let m = [q[0].max(0.0), q[1].max(0.0)];
        (m[0] * m[0] + m[1] * m[1]).sqrt() + q[0].max(q[1]).min(0.0) - r
    }

    /// The quad expansion from `vs_main` -- shader line 70. Shapes grow by one pixel so
    /// the antialiased edge has somewhere to live; glyphs do not, because padding would
    /// shear their UV mapping.
    pub fn quad_pad(kind: u32) -> f32 {
        if kind == KIND_GLYPH { 0.0 } else { 1.0 }
    }

    /// The signed-distance half of `fs_main` -- shader lines 113 to 128.
    ///
    /// `local` is `VsOut::local`: the fragment's position relative to the rect centre, in
    /// pixels. Evaluating in that space is what makes one unit of distance one pixel on
    /// screen, which is what makes the coverage below correct without a derivative.
    pub fn fs_alpha(
        kind: u32,
        local: [f32; 2],
        half_size: [f32; 2],
        radius: f32,
        param: f32,
    ) -> f32 {
        // Line 113: an unclamped radius larger than half the shorter side inverts the SDF
        // and renders a bow-tie.
        let radius = radius.clamp(0.0, half_size[0].min(half_size[1]));
        let distance = sd_rounded_box(local, half_size, radius);

        if kind == KIND_STROKE {
            // Lines 120-121: distance to the centre-line of a band of width `param`.
            let half_width = param * 0.5;
            (0.5 - ((distance + half_width).abs() - half_width)).clamp(0.0, 1.0)
        } else {
            // Line 127.
            (0.5 - distance).clamp(0.0, 1.0)
        }
    }

    /// `textureSample(atlas_texture, atlas_sampler, uv).r` -- shader line 107.
    ///
    /// The sampler is `FilterMode::Linear` with `AddressMode::ClampToEdge` (see
    /// `batcher.rs`), so this is bilinear, not nearest. That distinction is invisible in
    /// the pipeline as it stands, because glyph quads land on integer pixels and the
    /// bilinear weights collapse to `1, 0, 0, 0` -- but it is exactly why a quad at a
    /// *fractional* origin does not agree between the tiers, and it has to be modelled
    /// honestly for that measurement to mean anything.
    pub fn texture_sample_r(atlas: &[u8], size: u32, uv: [f32; 2]) -> f32 {
        let last = i64::from(size) - 1;
        // Texel centres sit at +0.5, so the filter grid is offset by half a texel.
        let x = uv[0] * size as f32 - 0.5;
        let y = uv[1] * size as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);

        let texel = |ix: f32, iy: f32| -> f32 {
            let cx = (ix as i64).clamp(0, last) as usize;
            let cy = (iy as i64).clamp(0, last) as usize;
            f32::from(atlas[cy * size as usize + cx]) / 255.0
        };

        let top = texel(x0, y0) * (1.0 - fx) + texel(x0 + 1.0, y0) * fx;
        let bottom = texel(x0, y0 + 1.0) * (1.0 - fx) + texel(x0 + 1.0, y0 + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

// -- the reference tier --------------------------------------------------------------

/// A linear premultiplied RGBA surface, which is what the GPU's framebuffer holds before
/// the sRGB transfer function is applied on write. [`CpuRasterizer`]'s pixmap is in the
/// same space, so the comparison happens there rather than after `to_srgb_rgba8` -- a
/// difference of one in a dark linear value becomes a difference of twelve in sRGB, and
/// comparing after the transfer would make every threshold a statement about gamma.
struct Surface {
    pixels: Vec<[f32; 4]>,
    width: u32,
    height: u32,
}

impl Surface {
    fn at(&self, x: u32, y: u32) -> [f32; 4] {
        self.pixels[(y * self.width + x) as usize]
    }
}

/// Run the shader over every pixel of every instance, blending as the pipeline does.
///
/// This is a software GPU for exactly one shader. It reproduces the parts of the fixed
/// function pipeline the shader's output actually depends on -- the quad the vertex stage
/// generates, which fragments that quad covers, the interpolation of `local` and `uv`
/// across it, and the `One / OneMinusSrcAlpha` blend -- and nothing else.
fn render_reference(list: &DrawList, atlas: &[u8], atlas_size: u32) -> Surface {
    let (width, height) = (list.viewport[0], list.viewport[1]);
    let clear = {
        let (r, g, b, a) = (
            crate::color::srgb_to_linear(list.clear.r),
            crate::color::srgb_to_linear(list.clear.g),
            crate::color::srgb_to_linear(list.clear.b),
            list.clear.a,
        );
        // `Pixmap::fill` takes straight colour and premultiplies, which is what
        // `cpu_raster::linear_premul` relies on.
        [r * a, g * a, b * a, a]
    };
    let mut surface = Surface {
        pixels: vec![clear; (width as usize) * (height as usize)],
        width,
        height,
    };

    for batch in &list.batches {
        let range = batch.range.start as usize..batch.range.end as usize;
        for instance in &list.instances[range] {
            draw_reference_instance(&mut surface, instance, atlas, atlas_size);
        }
    }
    surface
}

fn draw_reference_instance(
    surface: &mut Surface,
    instance: &Instance,
    atlas: &[u8],
    atlas_size: u32,
) {
    let [x, y, w, h] = instance.rect;
    // The CPU tier's own early-outs, mirrored so the two tiers agree about *nothing*
    // being drawn as well as about something being drawn.
    if !(w > 0.0 && h > 0.0) {
        return;
    }
    let color = shader::unpack4x8unorm(instance.color);
    if color[3] <= 0.0 {
        return;
    }

    let half_size = [w * 0.5, h * 0.5];
    let centre = [x + half_size[0], y + half_size[1]];
    let pad = shader::quad_pad(instance.kind);

    // Which fragments the vertex stage's quad actually generates. Outside it the shader
    // never runs, so neither does this.
    let x0 = (centre[0] - half_size[0] - pad).floor().max(0.0) as u32;
    let y0 = (centre[1] - half_size[1] - pad).floor().max(0.0) as u32;
    let x1 = (centre[0] + half_size[0] + pad)
        .ceil()
        .min(surface.width as f32) as u32;
    let y1 = (centre[1] + half_size[1] + pad)
        .ceil()
        .min(surface.height as f32) as u32;

    for py in y0..y1 {
        for px in x0..x1 {
            // Fragment shading is at the pixel centre.
            let p = [px as f32 + 0.5, py as f32 + 0.5];
            let local = [p[0] - centre[0], p[1] - centre[1]];
            if local[0].abs() > half_size[0] + pad || local[1].abs() > half_size[1] + pad {
                continue;
            }

            let alpha = if instance.kind == shader::KIND_GLYPH {
                // `out.uv = mix(inst.uv.xy, inst.uv.zw, corner)`, interpolated: the
                // fragment's fraction across the quad picks the same fraction across the
                // atlas sub-rectangle.
                let fx = (p[0] - x) / w;
                let fy = (p[1] - y) / h;
                let uv = [
                    instance.uv[0] + (instance.uv[2] - instance.uv[0]) * fx,
                    instance.uv[1] + (instance.uv[3] - instance.uv[1]) * fy,
                ];
                shader::texture_sample_r(atlas, atlas_size, uv)
            } else {
                shader::fs_alpha(
                    instance.kind,
                    local,
                    half_size,
                    instance.radius,
                    instance.param,
                )
            };

            // `return in.color * alpha` under `One / OneMinusSrcAlpha`. The colour is
            // already premultiplied, so scaling the whole vector keeps it that way.
            let src = [
                color[0] * alpha,
                color[1] * alpha,
                color[2] * alpha,
                color[3] * alpha,
            ];
            let inv = 1.0 - src[3];
            let dst = surface.pixels[(py * surface.width + px) as usize];
            surface.pixels[(py * surface.width + px) as usize] = [
                src[0] + dst[0] * inv,
                src[1] + dst[1] * inv,
                src[2] + dst[2] * inv,
                src[3] + dst[3] * inv,
            ];
        }
    }
}

// -- comparison ----------------------------------------------------------------------

/// Alpha below which a pixel is treated as empty when measuring extent. Deliberately
/// low: the point of the bounding box is to catch an edge that moved, so it must not
/// quietly forgive a faint extra row.
const INK_FLOOR: u8 = 8;

#[derive(Debug, PartialEq, Eq)]
struct Bounds {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

/// What the two tiers disagree about, in the five ways worth knowing.
///
/// Five rather than one because [`CPU_EDGE_QUANTUM`] blunts some of them and not others.
/// Snapping an edge to the nearest quarter pixel moves coverage around, so `max_channel`
/// and `centroid_shift` both have to carry slack for it; but it moves an equal amount off
/// one edge and onto the opposite one, so `total_ink` is untouched by it and stays sharp.
/// A metric that survives the known noise is worth more than four that do not.
#[derive(Debug)]
struct Divergence {
    /// Largest absolute difference in any channel of any pixel, in 0..=255 units.
    max_channel: u8,
    /// Mean absolute channel difference over the whole surface.
    mean_channel: f32,
    cpu_ink: Option<Bounds>,
    gpu_ink: Option<Bounds>,
    /// Distance between the two coverage-weighted centroids, in pixels.
    centroid_shift: f32,
    /// Total alpha, in whole-pixel units: how much ink each tier laid down.
    cpu_ink_area: f32,
    gpu_ink_area: f32,
}

impl Divergence {
    /// Difference in laid-down ink, relative to the larger of the two.
    fn area_error(&self) -> f32 {
        let scale = self.cpu_ink_area.max(self.gpu_ink_area);
        if scale <= 0.0 {
            return 0.0;
        }
        (self.cpu_ink_area - self.gpu_ink_area).abs() / scale
    }
}

fn ink_bounds(sample: impl Fn(u32, u32) -> [u8; 4], width: u32, height: u32) -> Option<Bounds> {
    let mut bounds: Option<Bounds> = None;
    for y in 0..height {
        for x in 0..width {
            if sample(x, y)[3] <= INK_FLOOR {
                continue;
            }
            match &mut bounds {
                None => {
                    bounds = Some(Bounds {
                        x0: x,
                        y0: y,
                        x1: x,
                        y1: y,
                    });
                }
                Some(b) => {
                    b.x0 = b.x0.min(x);
                    b.y0 = b.y0.min(y);
                    b.x1 = b.x1.max(x);
                    b.y1 = b.y1.max(y);
                }
            }
        }
    }
    bounds
}

/// Coverage-weighted centroid, and the total coverage it was weighted by (in whole-pixel
/// units). `None` when there is no ink to weigh.
///
/// Two measurements from one pass because they are complementary. The centroid moves
/// continuously with the shape, so it has sub-pixel resolution a whole-pixel bounding box
/// cannot reach -- but a thin ring levers a displacement of its *hole* by the ratio of
/// hole area to ring area, so on strokes it is sensitive to noise as well as to bugs. The
/// total is the opposite: blind to where the ink is, but immune to an edge sliding one
/// way while the opposite edge slides with it.
fn coverage(
    sample: impl Fn(u32, u32) -> [u8; 4],
    width: u32,
    height: u32,
) -> Option<((f32, f32), f32)> {
    let (mut sx, mut sy, mut total) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..height {
        for x in 0..width {
            let a = f64::from(sample(x, y)[3]);
            sx += f64::from(x) * a;
            sy += f64::from(y) * a;
            total += a;
        }
    }
    if total <= 0.0 {
        return None;
    }
    Some((
        ((sx / total) as f32, (sy / total) as f32),
        (total / 255.0) as f32,
    ))
}

fn quantise(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn compare(cpu: &Pixmap, gpu: &Surface) -> Divergence {
    let (width, height) = (cpu.width(), cpu.height());
    assert_eq!((width, height), (gpu.width, gpu.height));

    let cpu_at = |x: u32, y: u32| -> [u8; 4] {
        let p = cpu.pixels()[(y * width + x) as usize];
        [p.red(), p.green(), p.blue(), p.alpha()]
    };
    let gpu_at = |x: u32, y: u32| -> [u8; 4] {
        let p = gpu.at(x, y);
        [
            quantise(p[0]),
            quantise(p[1]),
            quantise(p[2]),
            quantise(p[3]),
        ]
    };

    let mut max_channel = 0u8;
    let mut total = 0u64;
    for y in 0..height {
        for x in 0..width {
            let (a, b) = (cpu_at(x, y), gpu_at(x, y));
            for c in 0..4 {
                let d = a[c].abs_diff(b[c]);
                max_channel = max_channel.max(d);
                total += u64::from(d);
            }
        }
    }

    let cpu_c = coverage(cpu_at, width, height);
    let gpu_c = coverage(gpu_at, width, height);
    let centroid_shift = match (cpu_c, gpu_c) {
        (Some((a, _)), Some((b, _))) => ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt(),
        // One tier drew nothing and the other did. Report it as an enormous shift so the
        // assertion fails rather than silently skipping the sharpest check it has.
        (None, Some(_)) | (Some(_), None) => f32::INFINITY,
        (None, None) => 0.0,
    };

    Divergence {
        max_channel,
        mean_channel: total as f32 / (width * height * 4) as f32,
        cpu_ink: ink_bounds(cpu_at, width, height),
        gpu_ink: ink_bounds(gpu_at, width, height),
        centroid_shift,
        cpu_ink_area: cpu_c.map_or(0.0, |c| c.1),
        gpu_ink_area: gpu_c.map_or(0.0, |c| c.1),
    }
}

// -- the fixtures --------------------------------------------------------------------

const SURFACE: u32 = 64;
const ATLAS: u32 = 32;

/// How finely the CPU tier can place an edge, in pixels.
///
/// `tiny-skia`'s antialiased fill resolves an edge to the nearest quarter pixel: a shape
/// whose left edge is at 12.375 rasterizes as though it were at 12.5. The shader's
/// distance field has no such step. Measured, not assumed -- see
/// `the_cpu_tier_resolves_an_edge_to_a_quarter_of_a_pixel`, which fails if `tiny-skia`
/// ever gets finer, because at that point every bound derived from this number is looser
/// than it needs to be and the suite has quietly stopped being sharp.
const CPU_EDGE_QUANTUM: f32 = 0.25;

/// The most an edge can be misplaced by that quantisation: half a step.
const MAX_EDGE_ERROR: f32 = CPU_EDGE_QUANTUM * 0.5;

/// The most a *single pixel* can therefore differ by, in 0..=255 units.
///
/// A pixel straddling a misplaced edge loses or gains exactly the coverage the edge
/// swept over it, so the worst case is [`MAX_EDGE_ERROR`] of coverage: 32 levels. Any
/// case tolerating more than this is tolerating something other than the quantisation,
/// and has to say what.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const QUANTISATION_CHANNEL_BUDGET: u8 = (MAX_EDGE_ERROR * 255.0) as u8;

/// One primitive drawn one way, plus the bounds its two tiers are allowed to differ by.
///
/// Every bound is measured and then justified in the fixture's comment. A bound with no
/// stated source is a bound someone raised until the test went green.
struct Case {
    name: &'static str,
    instance: Instance,
    /// Atlas content the case needs. Empty for everything but glyphs.
    uploads: Vec<PendingUpload>,
    /// Largest tolerated single-channel difference, in 0..=255 units.
    max_channel: u8,
    /// Largest tolerated mean channel difference over the surface.
    mean_channel: f32,
    /// Largest tolerated centroid drift, in pixels.
    centroid_shift: f32,
    /// Largest tolerated difference in total laid-down ink, as a fraction. This is the
    /// bound that stays tight even where the edge quantisation forces the others open.
    ink_area: f32,
}

/// A blob of coverage standing in for a glyph.
///
/// Deliberately not uniform: a solid block would agree between a bilinear sample and a
/// nearest blit no matter how badly the addressing was wrong, because every texel it
/// could possibly read holds the same value. The gradient plus the notch means a
/// half-texel addressing error changes the answer.
fn glyph_coverage(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let notch = x * 3 < w && y * 3 < h;
            let v = if notch {
                0
            } else {
                ((x * 37 + y * 61) % 256) as u8
            };
            out.push(v);
        }
    }
    out
}

/// Normalised atlas coordinates for a sub-rectangle of the atlas.
fn uv_for(x: u32, y: u32, w: u32, h: u32) -> [f32; 4] {
    let s = ATLAS as f32;
    [
        x as f32 / s,
        y as f32 / s,
        (x + w) as f32 / s,
        (y + h) as f32 / s,
    ]
}

/// The parity fixtures for one primitive.
///
/// **This match is the guard.** It is exhaustive over [`PrimKind`], so a new primitive
/// cannot compile until someone writes down how to check it, and
/// `no_prim_kind_can_ship_without_a_parity_test` refuses an empty list -- a stub arm
/// returning nothing fails the suite rather than passing it.
fn cases_for(kind: PrimKind) -> Vec<Case> {
    let white = Srgba::new(1.0, 1.0, 1.0, 1.0);
    // Measured on the quarter-aligned rounded fixtures, where the edge quantisation
    // contributes nothing and the only thing left is the difference between an analytic
    // area and `0.5 - d`. Corner pixels are where the two part company.
    const CURVATURE: u8 = 24;
    // The two budgets are independent -- one is a rasterizer's placement step, the other
    // is a definitional difference about coverage -- so a fixture that suffers both is
    // allowed their sum, and nothing more.
    const ROUNDED_AND_OFF_GRID: u8 = QUANTISATION_CHANNEL_BUDGET + CURVATURE;

    match kind {
        PrimKind::Rect => vec![
            Case {
                // The baseline. Axis-aligned, integer bounds, no curvature: the analytic
                // area and the distance field are answering the same question, and the
                // edges are already on the quarter grid. Nothing may differ here at all,
                // and nothing does -- this fixture measures 0.
                name: "rect/sharp",
                instance: Instance::rect(16.0, 20.0, 32.0, 24.0, 0.0, white),
                uploads: Vec::new(),
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.001,
            },
            Case {
                // Curvature is where the two definitions of coverage part company: 22 of
                // the 24 allowed, all of it in the corner pixels.
                name: "rect/rounded",
                instance: Instance::rect(12.0, 10.0, 40.0, 28.0, 6.0, white),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.10,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
            Case {
                // 125% display scale. Every coordinate lands off the integer grid, though
                // still on quarters, so this one is bounded by curvature alone -- which is
                // itself the finding: a fill is only safe off-grid because 125% happens to
                // land on the CPU tier's quarter-pixel step.
                name: "rect/fractional-scale",
                instance: Instance::rect(15.0, 12.5, 37.5, 27.5, 7.5, white),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.12,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
            Case {
                // 150% display scale, where a fill does *not* land on quarters: the right
                // edge falls on an eighth and gets snapped. Bounded by the sum.
                // Displacing one edge of a `w` by `wide` fill by `d` moves the centroid by
                // at most `d / 2`, so an eighth of a pixel at the edge is a sixteenth at
                // the centroid. Measured 0.022 against that 0.0625 ceiling.
                name: "rect/eighth-grid",
                instance: Instance::rect(9.375, 11.0, 30.375, 24.0, 9.0, white),
                uploads: Vec::new(),
                max_channel: ROUNDED_AND_OFF_GRID,
                mean_channel: 0.20,
                centroid_shift: MAX_EDGE_ERROR * 0.5,
                ink_area: 0.006,
            },
            Case {
                // A radius larger than the shape can hold. Both tiers clamp -- the shader
                // to `min(half_size)`, `rounded_rect` to half the shorter side, which are
                // the same number -- and both produce a stadium rather than the bow-tie an
                // unclamped SDF would invert into.
                //
                // This fixture exists because mutation testing found the suite blind
                // without it: deleting the clamp from the transcription changed no
                // measured value anywhere, since nothing asked for a radius big enough to
                // clamp. A guard nothing exercises is a guard nobody is checking.
                // The bounds here are looser than `CURVATURE` and stated rather than
                // shared, because a clamped radius makes the shape a stadium -- arc all
                // the way round, where the radius-6 fixtures are arc at four short
                // corners. Measured 26 and 0.005. They are also not what this fixture
                // rests on: if either tier failed to clamp, the SDF would invert into a
                // bow-tie and the *bounding box* would change, which no tolerance forgives.
                name: "rect/over-large-radius",
                instance: Instance::rect(16.0, 20.0, 32.0, 24.0, 1000.0, white),
                uploads: Vec::new(),
                max_channel: 32,
                mean_channel: 0.25,
                centroid_shift: 0.01,
                ink_area: 0.008,
            },
            Case {
                // Translucent and coloured, to exercise the premultiplied blend rather
                // than just the coverage. The CPU tier un-premultiplies for tiny-skia and
                // tiny-skia premultiplies again in u8, so a couple of levels of round-trip
                // loss are real and expected on top of the curvature.
                name: "rect/translucent",
                instance: Instance::rect(8.0, 8.0, 30.0, 30.0, 4.0, Srgba::new(0.2, 0.6, 0.9, 0.5)),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.05,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
        ],
        PrimKind::Stroke => vec![
            Case {
                // The case that started the chunk: before the CPU tier inset its path,
                // this ring began one pixel outside the rect. Still exact.
                name: "stroke/sharp",
                instance: Instance::stroke(10.0, 10.0, 20.0, 20.0, 0.0, 2.0, white),
                uploads: Vec::new(),
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.002,
            },
            Case {
                // `ink_area` is looser on a ring than on a fill and it is not the
                // quantisation doing it: the corner arcs' curvature error is an absolute
                // number of square pixels, and a 1.5px ring has an eighth the area of the
                // fill it surrounds, so the same error is eight times the fraction.
                // Measured 0.0085 of 144 square pixels, which is 1.2 of them.
                name: "stroke/rounded",
                instance: Instance::stroke(8.0, 8.0, 32.0, 24.0, 6.0, 1.5, white),
                uploads: Vec::new(),
                max_channel: CURVATURE,
                mean_channel: 0.25,
                centroid_shift: 0.01,
                ink_area: 0.010,
            },
            Case {
                // 125% scale: a 1.5px stroke becomes 1.875px, which is the width the icon
                // set actually draws with and the one nobody has looked at. The ring's
                // inner contour lands on an eighth and snaps, and this is the fixture
                // where the centroid has to give ground: displacing a *hole* moves the
                // ring's centroid by the ratio of hole area to ring area, which here is
                // about 610 to 220 -- so an eighth of a pixel at the edge becomes a third
                // of a pixel at the centroid. The bound is that lever arm, not a fudge,
                // and `ink_area` stays tight underneath it because the quantisation takes
                // from one edge exactly what it gives the other.
                name: "stroke/fractional-scale",
                instance: Instance::stroke(10.5, 9.5, 33.75, 25.0, 7.5, 1.875, white),
                uploads: Vec::new(),
                max_channel: ROUNDED_AND_OFF_GRID,
                mean_channel: 1.00,
                centroid_shift: 0.40,
                ink_area: 0.010,
            },
            Case {
                // The clamp again, on the harder side: the two tiers reach it by different
                // routes. The shader clamps the *outer* radius and then offsets the
                // distance field inward; the CPU tier insets the path first and clamps the
                // *reduced* radius against the *reduced* rectangle. Those are only the
                // same number because insetting a rounded rect by `t` reduces its radius by
                // exactly `t`, which is an identity worth having a fixture for rather than
                // a comment.
                // Loosest bounds in the file, and the reason is geometric rather than
                // convenient: this ring has *two* fully-curved boundaries, and the inner
                // one is tighter than the outer, so it carries the stadium's curvature
                // error twice over on an eighth of the ink. Measured 49 and 0.021. As
                // above, the clamp itself is caught by the bounding box, not by these.
                name: "stroke/over-large-radius",
                instance: Instance::stroke(12.0, 14.0, 32.0, 24.0, 1000.0, 2.0, white),
                uploads: Vec::new(),
                max_channel: 56,
                mean_channel: 0.65,
                centroid_shift: 0.01,
                ink_area: 0.025,
            },
        ],
        PrimKind::Glyph => vec![
            Case {
                // An integer-aligned quad, which is the only kind the pipeline emits: the
                // fractional pen position is baked into the subpixel variant instead. The
                // bilinear weights collapse to nearest and there is no edge to quantise,
                // so this is exact -- it measures 0, not 1.
                name: "glyph/integer-origin",
                instance: Instance::glyph(10.0, 12.0, 9.0, 11.0, uv_for(3, 5, 9, 11), white),
                uploads: vec![PendingUpload {
                    x: 3,
                    y: 5,
                    width: 9,
                    height: 11,
                    coverage: glyph_coverage(9, 11),
                }],
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.001,
            },
            Case {
                // The same glyph at 125%: the atlas entry is re-rasterized larger, so the
                // quad grows to 11x14 and still lands on integers. This is what "glyph at
                // a non-integer scale" means in the shipped pipeline -- the scale is
                // fractional, the geometry that results is not. The case where it is not
                // is pinned separately, by
                // `a_glyph_quad_at_a_fractional_origin_does_not_agree_across_tiers`.
                name: "glyph/fractional-scale",
                instance: Instance::glyph(14.0, 7.0, 11.0, 14.0, uv_for(6, 9, 11, 14), white),
                uploads: vec![PendingUpload {
                    x: 6,
                    y: 9,
                    width: 11,
                    height: 14,
                    coverage: glyph_coverage(11, 14),
                }],
                max_channel: 1,
                mean_channel: 0.01,
                centroid_shift: 0.01,
                ink_area: 0.001,
            },
        ],
    }
}

fn run_case(case: &Case) -> Divergence {
    let mut cpu = CpuRasterizer::new(SURFACE, SURFACE, ATLAS).unwrap();
    cpu.upload_glyphs(&case.uploads);

    // The reference's atlas mirror, filled by the same uploads. Both tiers read the same
    // bytes at the same coordinates, which is what makes the comparison about
    // rasterization rather than about atlas packing.
    let mut atlas = vec![0u8; (ATLAS as usize) * (ATLAS as usize)];
    for upload in &case.uploads {
        for row in 0..upload.height {
            let src = (row as usize) * (upload.width as usize);
            let dst = ((upload.y + row) as usize) * (ATLAS as usize) + upload.x as usize;
            atlas[dst..dst + upload.width as usize]
                .copy_from_slice(&upload.coverage[src..src + upload.width as usize]);
        }
    }

    let mut list = DrawList::default();
    // Transparent clear: with nothing behind it, every non-zero pixel is the primitive's
    // own contribution and a difference cannot be diluted by a shared background.
    list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
    list.instances.push(case.instance);
    // No scissor. `cpu_raster::clip_mask` documents that the CPU tier does not implement
    // one, so a scissored fixture would be measuring that known gap instead of parity.
    list.end_batch(None, !case.uploads.is_empty());

    let reference = render_reference(&list, &atlas, ATLAS);
    let pixmap = cpu.render(&list);
    compare(pixmap, &reference)
}

#[track_caller]
fn assert_parity(case: &Case) {
    let d = run_case(case);
    // Printed unconditionally: when a threshold fails, the first question is always "by
    // how much, and in which of the four ways", and cargo hides this on success anyway.
    eprintln!(
        "{}: max={} mean={:.4} centroid={:.4} area={:.5} ({:.3} vs {:.3}) ink={:?}",
        case.name,
        d.max_channel,
        d.mean_channel,
        d.centroid_shift,
        d.area_error(),
        d.cpu_ink_area,
        d.gpu_ink_area,
        d.cpu_ink
    );

    assert!(
        d.area_error() <= case.ink_area,
        "{}: the tiers laid down different amounts of ink -- {:.4} against {:.4}, a \
         relative difference of {:.5} over the bound of {:.5}. This is the one measure \
         the CPU tier's quarter-pixel edge step cannot move, so a failure here is a real \
         difference in how much of the primitive got drawn",
        case.name,
        d.cpu_ink_area,
        d.gpu_ink_area,
        d.area_error(),
        case.ink_area
    );
    assert_eq!(
        d.cpu_ink, d.gpu_ink,
        "{}: the two tiers disagree about where the primitive is, not merely about how it \
         is shaded -- an edge moved",
        case.name
    );
    assert!(
        d.centroid_shift <= case.centroid_shift,
        "{}: coverage centroid drifted {:.4}px, over the {:.4}px bound",
        case.name,
        d.centroid_shift,
        case.centroid_shift
    );
    assert!(
        d.max_channel <= case.max_channel,
        "{}: worst channel differs by {}, over the bound of {}",
        case.name,
        d.max_channel,
        case.max_channel
    );
    assert!(
        d.mean_channel <= case.mean_channel,
        "{}: mean channel difference {:.4}, over the bound of {:.4}",
        case.name,
        d.mean_channel,
        case.mean_channel
    );
}

fn run_kind(kind: PrimKind) {
    for case in &cases_for(kind) {
        assert_parity(case);
    }
}

// -- the tests -----------------------------------------------------------------------

#[test]
fn rect_agrees_across_tiers() {
    run_kind(PrimKind::Rect);
}

#[test]
fn stroke_agrees_across_tiers() {
    run_kind(PrimKind::Stroke);
}

#[test]
fn glyph_agrees_across_tiers() {
    run_kind(PrimKind::Glyph);
}

#[test]
fn the_cpu_tier_resolves_an_edge_to_a_quarter_of_a_pixel() {
    // The source of every loosened bound in this file, measured rather than asserted from
    // documentation -- `tiny-skia` does not document it.
    //
    // Slide a hard vertical edge across a pixel in sixteenths and read the coverage of the
    // pixel it crosses. The shader's answer is the offset itself. `tiny-skia`'s answer is
    // that offset snapped to the nearest quarter.
    //
    // This test is also the tripwire on the rest of the file. If `tiny-skia` ever
    // rasterizes more finely, `worst` collapses and this fails -- which is the signal that
    // `CPU_EDGE_QUANTUM` and every bound derived from it are now looser than the truth,
    // and the suite has gone quietly blunt. A tolerance nobody re-derives is a tolerance
    // that only ever grows.
    let mut worst = 0.0f32;
    for sixteenth in 0..16 {
        let offset = sixteenth as f32 / 16.0;
        let left = 20.0 + offset;

        let mut cpu = CpuRasterizer::new(SURFACE, SURFACE, ATLAS).unwrap();
        let mut list = DrawList::default();
        list.reset([SURFACE, SURFACE], Srgba::TRANSPARENT, 1);
        list.instances.push(Instance::rect(
            left,
            10.0,
            20.0,
            20.0,
            0.0,
            Srgba::new(1.0, 1.0, 1.0, 1.0),
        ));
        list.end_batch(None, false);
        let got =
            f32::from(cpu.render(&list).pixels()[(20 * SURFACE + 20) as usize].alpha()) / 255.0;

        // The pixel spans [20, 21] and the shape starts at `left`, so exact coverage is
        // `1 - offset`. That is also what the reference produces, which the parity
        // fixtures already prove; here only the CPU tier is under the microscope.
        //
        // What snaps is the *edge*, not the coverage -- the distinction matters, because
        // rounding a coverage of 0.875 to a quarter gives 1.0 while rounding the edge
        // gives 0.75, and only the second one is what happens.
        let exact = 1.0 - offset;
        let snapped = 1.0 - ((left / CPU_EDGE_QUANTUM).round() * CPU_EDGE_QUANTUM - 20.0);
        assert!(
            (got - snapped).abs() <= 2.0 / 255.0,
            "an edge at +{offset} produced coverage {got:.4}; snapping the edge to the \
             quarter-pixel grid predicts {snapped:.4}"
        );
        worst = worst.max((got - exact).abs());
    }

    assert!(
        worst >= MAX_EDGE_ERROR - 1.0 / 255.0,
        "the CPU tier placed every edge to within {worst:.4} of exact, better than the \
         {MAX_EDGE_ERROR} this file's bounds are derived from. That is good news and a \
         broken assumption: re-measure CPU_EDGE_QUANTUM and tighten the fixtures, because \
         they are now tolerating error that no longer exists"
    );
}

#[test]
fn no_prim_kind_can_ship_without_a_parity_test() {
    // The compile-time half of this guard is `cases_for`'s exhaustive match. This is the
    // run-time half: an arm that exists but returns nothing, or returns fixtures for the
    // wrong primitive, would satisfy the compiler and prove nothing.
    for kind in PrimKind::ALL {
        let cases = cases_for(kind);
        assert!(
            !cases.is_empty(),
            "{kind:?} has an arm in cases_for but no fixtures: it would ship unchecked"
        );
        for case in &cases {
            assert_eq!(
                case.instance.kind, kind as u32,
                "{}: filed under {kind:?} but draws a different primitive",
                case.name
            );
        }
    }
}

#[test]
fn the_shader_and_prim_kind_declare_the_same_primitives() {
    // Closes the one hole `cases_for`'s exhaustive match cannot see: a variant added to
    // the enum, given a shader branch, and left out of `PrimKind::ALL`. The match would
    // still compile (the loop above never visits the variant) and the suite would still
    // be green while a whole primitive went unchecked. Comparing against the shader's own
    // declarations catches it, because a primitive that renders must have a KIND_ there.
    let source = include_str!("shaders/instance.wgsl");
    let mut declared: Vec<(String, u32)> = source
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("const KIND_")?;
            let (name, value) = rest.split_once('=')?;
            let name = format!("KIND_{}", name.split(':').next()?.trim());
            let value = value.trim().trim_end_matches(';').trim_end_matches('u');
            Some((name, value.trim().parse().ok()?))
        })
        .collect();
    declared.sort();

    let mut known: Vec<(String, u32)> = PrimKind::ALL
        .iter()
        .map(|k| (k.shader_const().to_owned(), *k as u32))
        .collect();
    known.sort();

    assert_eq!(
        declared, known,
        "shaders/instance.wgsl and PrimKind::ALL declare different primitives; whichever \
         side gained one, the parity suite is not covering it"
    );
    assert!(!declared.is_empty(), "the KIND_ parser matched nothing");
}

#[test]
fn a_glyph_quad_at_a_fractional_origin_does_not_agree_across_tiers() {
    // Not a parity case -- a pinned honest negative.
    //
    // The GPU samples the atlas bilinearly; the CPU tier rounds the quad's origin and
    // blits 1:1. On an integer-aligned quad those coincide exactly, which is why
    // `glyph/integer-origin` is allowed a bound of one. Move the quad half a pixel and
    // they do not: the GPU blurs across two texels and the CPU snaps to one.
    //
    // The pipeline never emits such a quad -- the fractional pen position is baked into
    // the subpixel variant, so the geometry is always integral. That guarantee currently
    // lives in a comment in `batcher.rs`. This test is the receipt for what it is buying,
    // so that if a future change starts emitting fractional glyph origins, the cost is
    // already written down and measured instead of being discovered as blurry text.
    let case = Case {
        name: "glyph/fractional-origin",
        instance: Instance::glyph(
            10.5,
            12.5,
            9.0,
            11.0,
            uv_for(3, 5, 9, 11),
            Srgba::new(1.0, 1.0, 1.0, 1.0),
        ),
        uploads: vec![PendingUpload {
            x: 3,
            y: 5,
            width: 9,
            height: 11,
            coverage: glyph_coverage(9, 11),
        }],
        max_channel: 255,
        mean_channel: 255.0,
        centroid_shift: 255.0,
        ink_area: 1.0,
    };
    let d = run_case(&case);
    eprintln!(
        "glyph/fractional-origin: max={} mean={:.4} centroid={:.4}",
        d.max_channel, d.mean_channel, d.centroid_shift
    );
    assert!(
        d.max_channel > 32,
        "a half-pixel glyph offset was expected to diverge between the tiers, but the \
         worst channel differed by only {}. Either the sampler stopped filtering or the \
         CPU tier stopped rounding -- in both cases `glyph/integer-origin`'s bound of one \
         is no longer buying what this test says it buys",
        d.max_channel
    );
}

#[test]
fn the_reference_reproduces_the_stroke_alignment_the_cpu_tier_was_fixed_to_match() {
    // Ties the new machinery back to the bug that motivated it. `stroke/sharp` asserts
    // the two tiers agree; this asserts they agree on the *right* answer, which a pair of
    // identically-wrong implementations would also satisfy.
    let mut list = DrawList::default();
    list.reset([40, 40], Srgba::TRANSPARENT, 1);
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

    let reference = render_reference(&list, &[], 0);
    let row: Vec<u32> = (0..40)
        .filter(|x| quantise(reference.at(*x, 20)[3]) > INK_FLOOR)
        .collect();
    assert_eq!(
        (row.first().copied(), row.last().copied()),
        (Some(10), Some(29)),
        "the shader's ring must sit inside the rect's bounds, touching both edges"
    );
}
