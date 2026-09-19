//! Save grid cells at three sizes using the CPU renderer.
//!
//! `cargo run -p qs-ui --example grid_badge -- out.png`
//!
//! Use the image to inspect extension labels and their spacing beside symlink
//! emblems.

// A developer tool that renders a picture and exits; the crate's production lints are about
// code that runs inside a frame.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use qs_gpu::{DrawList, cpu_raster::CpuRasterizer};
use qs_ui::row::{GridMetrics, ListRenderer};
use qs_ui::row_source::{KindId, LoadState, RowBuf, RowFlags, RowView};
use qs_ui::tokens::{Theme, Tokens};

/// Name, KindId, and whether it is a symlink.
///
/// The kinds are varied on purpose: the chip lands at the bottom of the icon box, where
/// `Code` and `Config` have no ink and `Text`, `Archive` and `Data` do. Rendering one kind
/// would show the knockout doing nothing and prove nothing.
const NAMES: [(&[u8], u16, bool); 9] = [
    (b"main.rs", 2, false),
    (b"main.rs", 2, true),
    (b"photo.png", 5, false),
    (b"notes.txt", 6, true),
    (b"data.json", 7, false),
    (b"archive.tar.gz", 8, false),
    (b"readme.md", 4, false),
    (b"Makefile", 0, false),
    (b"a.javascript", 3, false),
];

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: grid_badge <out.png>");
    let scale = 1.0f32;

    for (i, cell) in [60.0f32, 90.0, 140.0].into_iter().enumerate() {
        let db: Arc<dyn qs_text::FontDb> = Arc::new(qs_text::SystemFontDb::scan());
        let tokens = Tokens::embedded(Theme::Dark).unwrap();
        let bg = tokens.color("surface/base");
        let mut renderer =
            ListRenderer::new(tokens, db, qs_gpu::config_for(qs_gpu::RenderPath::Cpu));

        let width = (cell * NAMES.len() as f32 * 1.05).ceil() as u32;
        let metrics = GridMetrics::fit(width, cell, scale, 16.0 * scale, 8.0);
        let height = metrics.cell_height + 8;

        let source = qs_ui::row_source::StubbornSource {
            count: NAMES.len() as u64,
        };
        let layout = qs_ui::recycler::Recycler::new().layout_grid(
            &source,
            metrics.columns,
            metrics.cell_height,
            width,
            height,
            scale,
            1.0,
            qs_ui::density::Density::default(),
            0.0,
        );

        let mut buf = RowBuf::new();
        for slot in 0..layout.visible.count as usize {
            let (name, kind, link) = NAMES[slot % NAMES.len()];
            buf.push(
                RowView {
                    kind: KindId(kind),
                    state: LoadState::Basic,
                    flags: if link {
                        RowFlags::IS_SYMLINK
                    } else {
                        RowFlags::EMPTY
                    },
                    ..RowView::default()
                },
                name,
            );
        }

        let mut list = DrawList::default();
        list.reset([width, height], bg, 1);
        renderer.render_grid(
            &mut list,
            &buf,
            &layout,
            metrics,
            qs_ui::row::Interaction::default(),
            &qs_ui::motion::InteractionMotion::default(),
        );

        let mut raster = CpuRasterizer::new(width, height, 1024).expect("rasterizer");
        raster.upload_glyphs(&renderer.take_uploads());
        let pixmap = raster.render(&list);
        let path = out.replace(".png", &format!("-cell{}.png", cell as u32));
        pixmap.save_png(&path).expect("save");
        println!(
            "wrote {path} (cell {cell}, icon {}px, {i}) glyphs_dropped={} icons_dropped={} rasterized={}",
            metrics.icon_px,
            list.stats.glyphs_dropped,
            list.stats.icons_dropped,
            list.stats.glyphs_rasterized
        );
    }
}
