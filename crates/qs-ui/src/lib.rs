//! Layout, row recycling, scroll state, densities, tokens and the accessibility tree.
//!
//! The boundary this crate defends is that **nothing below it knows what a file is, and
//! nothing above it knows what a glyph is**. Rows arrive through [`row_source::RowSource`]
//! and leave as a `qs_gpu::DrawList`. Whether the rows came from a synthetic corpus, a
//! filesystem, or a search index is not visible here -- which is the test of whether the
//! `RowSource` boundary is drawn in the right place.

pub mod a11y;
pub mod density;
pub mod fenwick;
pub mod material;
pub mod motion;
pub mod recycler;
pub mod row;
pub mod row_source;
/// Building the lit scene from the materials the draw list is built from.
pub mod scene;
pub mod scroll;
pub mod selection;
/// Facts about a file, expressed as what its surface is made of.
pub mod substance;
pub mod tokens;

pub use a11y::{SemanticNode, SemanticTree, audit};
pub use density::{Density, DensityTransition};
pub use fenwick::{Fenwick, Heights};
pub use material::{Material, Surface};
pub use motion::{Animation, MotionKind, MotionPlan, MotionPreference};
pub use recycler::{Recycler, ViewportLayout};
pub use row::{
    Columns, GridMetrics, ListRenderer, ResolvedRole, format_mtime, format_size, kind_of,
};
pub use row_source::{
    EmptySource, KindId, LoadState, RowBuf, RowFlags, RowId, RowSource, RowView, StubbornSource,
};
pub use scroll::{ScrollState, VisibleRange, rebase_to_viewport, visible_range};
pub use selection::Selection;
pub use substance::{Substance, SubstanceTokens};
pub use tokens::{Theme, Tokens};
