//! Resource limits for the reduced rendering tier.
//!
//! This tier uses the shared pipeline with a smaller glyph atlas and a lower
//! per-frame upload budget.

use crate::path::RenderPath;

/// Tier-specific tuning.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TierConfig {
    /// Square glyph-atlas edge, in texels.
    pub atlas_size: u32,
    /// Glyphs rasterized and uploaded per frame before the rest are deferred.
    pub max_glyph_uploads_per_frame: u32,
    /// Shaped runs held in the cache.
    pub shaped_run_cache_entries: usize,
}

pub fn config_for(tier: RenderPath) -> TierConfig {
    match tier {
        RenderPath::Primary => TierConfig {
            atlas_size: 2048,
            max_glyph_uploads_per_frame: 256,
            shaped_run_cache_entries: 8192,
        },
        RenderPath::Reduced => TierConfig {
            atlas_size: 1024,
            max_glyph_uploads_per_frame: 96,
            shaped_run_cache_entries: 4096,
        },
        RenderPath::Cpu => TierConfig {
            // The CPU tier's atlas is a plain `Vec<u8>` with no texture-size limit, but a
            // smaller one keeps the working set in cache, which matters far more here than
            // on a GPU.
            atlas_size: 1024,
            // Frame times are recorded but not bounded on this tier (decision A-3), so the
            // bound exists to keep the UI responsive, not to hit a budget.
            max_glyph_uploads_per_frame: 64,
            shaped_run_cache_entries: 4096,
        },
    }
}

/// Whether a tier can be expected to report GPU timestamps.
///
/// GL exposes `GL_ARB_timer_query` widely but wgpu's GL backend does not surface it as
/// `TIMESTAMP_QUERY`, so the Reduced tier reports `gpu_ms: null`. Callers must not treat
/// that as a measurement failure -- see [`crate::timing`].
pub fn expects_gpu_timestamps(tier: RenderPath) -> bool {
    matches!(tier, RenderPath::Primary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tier_has_a_workable_configuration() {
        for tier in [RenderPath::Primary, RenderPath::Reduced, RenderPath::Cpu] {
            let config = config_for(tier);
            assert!(config.atlas_size >= 256);
            assert!(config.max_glyph_uploads_per_frame >= 1);
            assert!(config.shaped_run_cache_entries >= 256);
        }
    }

    #[test]
    fn lower_tiers_ask_for_less() {
        let primary = config_for(RenderPath::Primary);
        let reduced = config_for(RenderPath::Reduced);
        assert!(reduced.atlas_size <= primary.atlas_size);
        assert!(reduced.max_glyph_uploads_per_frame <= primary.max_glyph_uploads_per_frame);
    }

    #[test]
    fn only_the_primary_tier_promises_gpu_timestamps() {
        assert!(expects_gpu_timestamps(RenderPath::Primary));
        assert!(!expects_gpu_timestamps(RenderPath::Reduced));
        assert!(!expects_gpu_timestamps(RenderPath::Cpu));
    }
}
