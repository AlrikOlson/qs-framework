//! Render the state-icon set at the sizes chrome asks for, in both themes, and write a PNG.
//!
//! `cargo run -p qs-gpu --example state_strip -- out.png`
//!
//! chunk:harness-adapters puts six states on screen — four from the PTY floor and two only an
//! adapter can see — and the criterion that matters for them is not a draw count. It is whether
//! a person scanning a tab strip can find the one session that is blocked on them without
//! reading a word. That is settled by looking, so this is the looking, in the tradition of
//! `emblem_strip`.
//!
//! # Three sizes, not one
//!
//! The kind icons live in a 20 px column and are judged there. A state icon does not: it sits
//! in a tab, on a directory row beside a kind icon, and in the overview's session lines, so it
//! is asked for at around 12 px and the set has to survive that. Rendering only at 20 would
//! photograph the size at which every one of these looks fine.
//!
//! # What this is not
//!
//! It rasterizes masks and composites them by hand, exactly as `emblem_strip` does, and it
//! asserts nothing. The tests in `qs_gpu::icon` hold the set to measurable floors; this is for
//! the judgement no measurement makes.

// Same allowance, and the same reason: this renders a picture and exits.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use qs_gpu::icon::{self, IconKey, IconKind, IconShape, StateIcon};
use tiny_skia::{Pixmap, PremultipliedColorU8};

/// The sizes the set is actually asked for. 12 is a tab, 16 a row indicator, 20 the overview.
const SIZES: [u16; 3] = [12, 16, 20];
const CELL: u32 = 28;
const PAD: u32 = 8;
const ZOOM: u32 = 6;

struct Theme {
    bg: [u8; 3],
    /// The neutral rail's ink: a live shell says nothing about how it is doing.
    neutral: [u8; 3],
    /// `rail/added` — the one outcome worth reporting in green.
    good: [u8; 3],
    /// `rail/conflict` — every way a session stopped working.
    bad: [u8; 3],
    /// The accent, which is what a summons gets and nothing else does.
    accent: [u8; 3],
    /// A file icon, drawn beside the states so the two sets can be compared where they meet.
    file: [u8; 3],
    /// `content/secondary` — what a record of a session is drawn in, being the one state here
    /// that is not happening.
    dim: [u8; 3],
}

fn tint(state: StateIcon, theme: &Theme) -> [u8; 3] {
    match state {
        // The floor's live state and the adapter's busy state share the neutral ink for the
        // reason `qs::terminal::status` gives: `border/focus` is the accent *and* the focused
        // region's edge, and a green rail beside a green edge reads as one thicker line.
        StateIcon::Running | StateIcon::Working => theme.neutral,
        StateIcon::Finished => theme.good,
        StateIcon::Failed => theme.bad,
        // The one state that gets the accent, because it is the one asking for something.
        StateIcon::AwaitingApproval => theme.accent,
        // Deliberately quiet: a degraded adapter is a claim withdrawn, not a fault to report.
        StateIcon::Degraded => theme.neutral,
        // Quieter still, and it is the only state here drawn in the secondary ink: a record of
        // a session is the one thing in this set that is not happening. `qs::terminal` spends
        // the same token on it for the same reason.
        StateIcon::Remembered => theme.dim,
    }
}

fn main() {
    let themes = [
        Theme {
            bg: [0x1b, 0x1b, 0x1f],
            neutral: [0x9a, 0xa0, 0xb0],
            good: [0x5c, 0xb8, 0x7a],
            bad: [0xe0, 0x6c, 0x6c],
            accent: [0x6e, 0xa8, 0xff],
            file: [0x6a, 0x70, 0x80],
            dim: [0x7a, 0x80, 0x90],
        },
        Theme {
            bg: [0xff, 0xff, 0xff],
            neutral: [0x5c, 0x62, 0x72],
            good: [0x1f, 0x7a, 0x43],
            bad: [0xa8, 0x2a, 0x2a],
            accent: [0x1e, 0x54, 0xb7],
            file: [0x8a, 0x90, 0x9e],
            dim: [0x6c, 0x72, 0x82],
        },
    ];

    // One column per state, plus two at the end holding kind icons: the comparison that matters
    // on a directory row, where a state icon and a kind icon are drawn a few pixels apart.
    let cols = StateIcon::ALL.len() as u32 + 2;
    let band = SIZES.len() as u32 * CELL;
    let w = cols * CELL + PAD * 2;
    let h = themes.len() as u32 * band + PAD * 2;

    let mut pm = Pixmap::new(w, h).expect("pixmap");

    for (t, theme) in themes.iter().enumerate() {
        let y0 = PAD + t as u32 * band;
        fill(&mut pm, 0, y0.saturating_sub(PAD), w, band + PAD, theme.bg);
        for (r, &px) in SIZES.iter().enumerate() {
            let y = y0 + r as u32 * CELL + (CELL - u32::from(px)) / 2;
            for (c, &state) in StateIcon::ALL.iter().enumerate() {
                let x = PAD + c as u32 * CELL + (CELL - u32::from(px)) / 2;
                blit(
                    &mut pm,
                    shape(IconShape::State(state), px),
                    x,
                    y,
                    tint(state, theme),
                );
            }
            for (c, &kind) in [IconKind::Code, IconKind::Folder].iter().enumerate() {
                let x = PAD + (StateIcon::ALL.len() + c) as u32 * CELL + (CELL - u32::from(px)) / 2;
                blit(&mut pm, shape(IconShape::Kind(kind), px), x, y, theme.file);
            }
        }
    }

    let out = std::env::args()
        .nth(1)
        .expect("usage: state_strip <out.png>");
    zoom(&pm, ZOOM).save_png(&out).expect("save");
    pm.save_png(out.replace(".png", "-1x.png"))
        .expect("save 1x");
    println!("wrote {out}");
    println!(
        "columns: {} then Code, Folder; rows: {SIZES:?} px",
        StateIcon::ALL
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn shape(shape: IconShape, px: u16) -> qs_text::RasterizedGlyph {
    icon::rasterize(IconKey { shape, px }).unwrap_or_else(|| panic!("{shape:?} at {px}px"))
}

fn fill(pm: &mut Pixmap, x0: u32, y0: u32, w: u32, h: u32, c: [u8; 3]) {
    let px = PremultipliedColorU8::from_rgba(c[0], c[1], c[2], 255).expect("opaque");
    let stride = pm.width();
    for y in y0..(y0 + h).min(pm.height()) {
        for x in x0..(x0 + w).min(stride) {
            pm.pixels_mut()[(y * stride + x) as usize] = px;
        }
    }
}

/// Source-over of a coverage mask tinted `c`, which is what `Instance::glyph` does on the GPU.
fn blit(pm: &mut Pixmap, g: qs_text::RasterizedGlyph, x0: u32, y0: u32, c: [u8; 3]) {
    let stride = pm.width();
    for gy in 0..g.height {
        for gx in 0..g.width {
            let a = u32::from(g.coverage[(gy * g.width + gx) as usize]);
            if a == 0 {
                continue;
            }
            let (x, y) = (x0 + gx, y0 + gy);
            if x >= stride || y >= pm.height() {
                continue;
            }
            let dst = pm.pixels_mut()[(y * stride + x) as usize];
            let mix = |s: u8, d: u8| ((u32::from(s) * a + u32::from(d) * (255 - a)) / 255) as u8;
            pm.pixels_mut()[(y * stride + x) as usize] = PremultipliedColorU8::from_rgba(
                mix(c[0], dst.red()),
                mix(c[1], dst.green()),
                mix(c[2], dst.blue()),
                255,
            )
            .expect("opaque");
        }
    }
}

fn zoom(src: &Pixmap, n: u32) -> Pixmap {
    let mut out = Pixmap::new(src.width() * n, src.height() * n).expect("pixmap");
    let (ow, oh) = (out.width(), out.height());
    for y in 0..oh {
        for x in 0..ow {
            let s = src.pixels()[((y / n) * src.width() + x / n) as usize];
            out.pixels_mut()[(y * ow + x) as usize] = s;
        }
    }
    out
}
