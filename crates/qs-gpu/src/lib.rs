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
/// The lighting pass and its per-tier degradation. See `specs/002-ray-traced-mode/`.
pub mod lighting;
pub mod path;
/// The interface as geometry, for the lighting pass to read.
pub mod scene;
/// The offscreen colour target and its resolve pass: the architecture every
/// neighbourhood effect needs and none of the shipped ones did.
pub mod target;
/// Per-primitive CPU-versus-shader agreement. Tests only -- it exists to be run, not to
/// be called, and compiling it into the shipped binary would drag a second rasterizer
/// along for no reason.
#[cfg(test)]
mod tier_parity;
pub mod timing;

pub use atlas::{
    AtlasEntry, AtlasKey, AtlasStats, GlyphAtlas, PendingUpload, STRUCTURAL_UPLOADS_PER_FRAME,
    UploadClass,
};
pub use color::Srgba;
pub use device::{Capabilities, GpuContext, GpuError, SurfaceRecovery};
pub use frame::{
    Batch, Consumer, DrawList, DrawStats, Fidelity, Floor, Instance, PrimKind, Producer, affinity,
    draw_list_channel,
};
pub use gl_tier::{TierConfig, config_for};
pub use icon::{Emblem, IconKey, IconKind, IconShape};
pub use lighting::{SceneEffect, SceneFloor};
pub use path::{CrashCounter, PathReason, RenderPath, RenderPathSelector, Resolution};
pub use scene::{Environment, FocusLamp, Light, MAX_SLABS, SceneList, Slab};
pub use target::{OffscreenTarget, tier_can_hold_target};
