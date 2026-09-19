//! List layout, scrolling, selection, materials and accessibility.
//!
//! Rows come from [`row_source::RowSource`] and are drawn into a
//! `qs_gpu::DrawList`. The crate also builds an accessibility tree from the
//! same row and interaction state.

pub mod a11y;
pub mod density;
pub mod fenwick;
/// What the application knows about a row that the row source does not.
pub mod mark;
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
pub use mark::{NO_MARKS, SessionMark, SessionMarks};
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
