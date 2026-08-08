//! Device and surface management, draw lists, the instanced pipeline, the glyph atlas,
//! and the rendering-tier machinery.
//!
//! The boundary this crate defends is that **nothing above it knows what a pipeline is**.
//! `qs-ui` produces a [`frame::DrawList`] of [`frame::Instance`]s and hands it over; which
//! tier consumes it, and whether that tier is D3D12 or a software rasterizer, is not
//! visible from above. That is what makes RP-2 -- all three tiers consume the same draw
//! lists -- something the type system helps with rather than a rule people remember.

pub mod atlas;
pub mod batcher;
pub mod color;
pub mod cpu_raster;
pub mod device;
pub mod frame;
pub mod gl_tier;
pub mod icon;
pub mod path;
/// Per-primitive CPU-versus-shader agreement. Tests only -- it exists to be run, not to
/// be called, and compiling it into the shipped binary would drag a second rasterizer
/// along for no reason.
#[cfg(test)]
mod tier_parity;
pub mod timing;

pub use atlas::{AtlasEntry, AtlasKey, AtlasStats, GlyphAtlas, PendingUpload};
pub use color::Srgba;
pub use device::{Capabilities, GpuContext, GpuError, SurfaceRecovery};
pub use frame::{
    Batch, Consumer, DrawList, DrawStats, Instance, PrimKind, Producer, affinity, draw_list_channel,
};
pub use gl_tier::{TierConfig, config_for};
pub use icon::{IconKey, IconKind};
pub use path::{CrashCounter, PathReason, RenderPath, RenderPathSelector, Resolution};
