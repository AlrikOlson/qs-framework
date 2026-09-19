//! Row density and text scaling.
//!
//! Compact and default densities use 24- and 28-pixel rows at the base scale.
//! Rows grow with text size to avoid clipping. Platform text-scale values are
//! clamped to the supported range.

/// How tall rows are.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Density {
    /// 24 logical pixels.
    Compact,
    /// 28 logical pixels.
    #[default]
    Default,
}

/// Text scales outside this range are clamped. 0.75 is the smallest any platform offers;
/// 3.0 is past every OS maximum and is where a row stops being a row.
pub const MIN_TEXT_SCALE: f32 = 0.75;
pub const MAX_TEXT_SCALE: f32 = 3.0;

impl Density {
    /// Row height in *logical* pixels at 100% text scale.
    pub fn base_height(self) -> f32 {
        match self {
            Self::Compact => 24.0,
            Self::Default => 28.0,
        }
    }

    /// Font size in logical pixels at 100% text scale.
    pub fn base_font_size(self) -> f32 {
        match self {
            // Deliberately the same at both densities. Compact removes *padding*, not
            // legibility; shrinking the text as well is how "compact" turns into
            // "unreadable" and why so many applications' compact modes go unused.
            Self::Compact | Self::Default => 13.0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compact => "compact-24",
            Self::Default => "default-28",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "compact" | "compact-24" => Some(Self::Compact),
            "default" | "default-28" => Some(Self::Default),
            _ => None,
        }
    }

    pub fn other(self) -> Self {
        match self {
            Self::Compact => Self::Default,
            Self::Default => Self::Compact,
        }
    }

    /// Row height in **physical** pixels.
    ///
    /// Rounded to a whole physical pixel, and never below 1. Fractional row heights would
    /// put every row on a different subpixel phase, which makes the whole list shimmer
    /// during a slow scroll and makes the Fenwick tree's integer heights a lie.
    pub fn row_height_px(self, scale: f32, text_scale: f32) -> u32 {
        let logical = self.base_height() * clamp_text_scale(text_scale);
        ((logical * sane_scale(scale)).round() as u32).max(1)
    }

    /// Font size in physical pixels.
    pub fn font_size_px(self, scale: f32, text_scale: f32) -> f32 {
        self.base_font_size() * clamp_text_scale(text_scale) * sane_scale(scale)
    }
}

/// Clamp an OS-reported text scale into the range the layout can actually honour.
pub fn clamp_text_scale(scale: f32) -> f32 {
    // NaN carries no direction, so it falls back to 100%. An infinity does carry one --
    // "as large as possible" -- and `clamp` maps it to the maximum, which is closer to what
    // the user asked for than silently resetting them to 100%.
    if scale.is_nan() {
        1.0
    } else {
        scale.clamp(MIN_TEXT_SCALE, MAX_TEXT_SCALE)
    }
}

/// Clamp a device pixel ratio. A non-finite or zero scale is a platform bug, and the right
/// response is to render at 1x rather than to divide by zero three call levels down.
fn sane_scale(scale: f32) -> f32 {
    if scale.is_nan() || scale <= 0.0 {
        1.0
    } else {
        scale.clamp(0.5, 8.0)
    }
}

/// An in-progress density change.
///
/// Row height animates, and the scroll position must be preserved across it. "Preserved"
/// means the row the user is looking at stays under the pointer -- **not** that the pixel
/// offset stays the same, which would move them hundreds of rows away in a million-row
/// list. That distinction is the entire content of this type.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DensityTransition {
    pub from: Density,
    pub to: Density,
    /// 0.0 at the start, 1.0 when complete.
    pub progress: f32,
    /// Total duration in seconds.
    pub duration: f32,
}

impl DensityTransition {
    /// Duration of the height cross-fade. Short enough to feel immediate, long enough that
    /// rows do not appear to teleport.
    pub const DURATION: f32 = 0.12;

    pub fn new(from: Density, to: Density) -> Self {
        Self {
            from,
            to,
            progress: 0.0,
            duration: Self::DURATION,
        }
    }

    /// Advance by `dt` seconds. Returns `true` while still animating.
    pub fn advance(&mut self, dt: f32) -> bool {
        if self.duration <= 0.0 {
            self.progress = 1.0;
            return false;
        }
        self.progress = (self.progress + dt / self.duration).clamp(0.0, 1.0);
        self.progress < 1.0
    }

    pub fn is_complete(self) -> bool {
        self.progress >= 1.0
    }

    /// Interpolated row height in physical pixels.
    pub fn row_height_px(self, scale: f32, text_scale: f32) -> u32 {
        let a = self.from.row_height_px(scale, text_scale) as f32;
        let b = self.to.row_height_px(scale, text_scale) as f32;
        let t = ease_out_cubic(self.progress);
        ((a + (b - a) * t).round() as u32).max(1)
    }

    /// Where the scroll offset must move so that `anchor_row` stays at `anchor_offset`
    /// pixels from the top of the viewport.
    ///
    /// This is the calculation FR's "preserving scroll position" actually requires. Getting
    /// it wrong is not subtle at a million rows: keeping the raw pixel offset across a
    /// 28→24 px change moves the view by one sixth of the corpus.
    pub fn rebase_scroll(
        anchor_row: u64,
        offset_within_row: f64,
        old_height: u32,
        new_height: u32,
        anchor_viewport_y: f64,
    ) -> f64 {
        let ratio = if old_height > 0 {
            f64::from(new_height) / f64::from(old_height)
        } else {
            1.0
        };
        let row_top = anchor_row as f64 * f64::from(new_height);
        (row_top + offset_within_row * ratio - anchor_viewport_y).max(0.0)
    }
}

/// Cubic ease-out. Fast start, settled finish -- the shape that reads as "responsive"
/// rather than "animated".
pub fn ease_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    1.0 - inv * inv * inv
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn densities_match_the_data_model() {
        assert_eq!(Density::Compact.base_height(), 24.0);
        assert_eq!(Density::Default.base_height(), 28.0);
        assert_eq!(Density::Compact.row_height_px(1.0, 1.0), 24);
        assert_eq!(Density::Default.row_height_px(1.0, 1.0), 28);
    }

    #[test]
    fn density_names_round_trip_for_the_bench_report() {
        for d in [Density::Compact, Density::Default] {
            assert_eq!(Density::parse(d.as_str()), Some(d));
        }
        assert_eq!(Density::parse("comfortable"), None, "deferred to M1");
    }

    #[test]
    fn row_height_grows_with_the_text_scale() {
        // SC-007: at 200%, text must not be clipped by a row that did not grow with it.
        let base = Density::Default.row_height_px(1.0, 1.0);
        let large = Density::Default.row_height_px(1.0, 2.0);
        assert_eq!(large, base * 2);

        let font_base = Density::Default.font_size_px(1.0, 1.0);
        let font_large = Density::Default.font_size_px(1.0, 2.0);
        assert!((font_large - font_base * 2.0).abs() < 0.01);
    }

    #[test]
    fn row_heights_are_whole_physical_pixels() {
        // A fractional row height puts every row on a different subpixel phase and makes
        // the whole list shimmer during a slow scroll.
        for scale in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] {
            for text in [1.0, 1.15, 1.5, 2.0] {
                let h = Density::Default.row_height_px(scale, text);
                assert!(h >= 1);
            }
        }
    }

    #[test]
    fn an_absurd_text_scale_is_clamped_rather_than_honoured() {
        assert_eq!(clamp_text_scale(100.0), MAX_TEXT_SCALE);
        assert_eq!(clamp_text_scale(0.01), MIN_TEXT_SCALE);
        assert_eq!(clamp_text_scale(f32::NAN), 1.0);
        assert_eq!(clamp_text_scale(f32::INFINITY), MAX_TEXT_SCALE);
    }

    #[test]
    fn a_broken_device_scale_falls_back_to_1x_instead_of_dividing_by_zero() {
        assert_eq!(Density::Default.row_height_px(0.0, 1.0), 28);
        assert_eq!(Density::Default.row_height_px(f32::NAN, 1.0), 28);
        assert_eq!(Density::Default.row_height_px(-2.0, 1.0), 28);
    }

    #[test]
    fn a_transition_completes_and_lands_exactly_on_the_target_height() {
        let mut t = DensityTransition::new(Density::Default, Density::Compact);
        assert_eq!(t.row_height_px(1.0, 1.0), 28);

        let mut steps = 0;
        while t.advance(1.0 / 120.0) {
            steps += 1;
            assert!(steps < 1000, "the transition never finished");
        }
        assert!(t.is_complete());
        assert_eq!(
            t.row_height_px(1.0, 1.0),
            24,
            "must land exactly on the target"
        );
    }

    #[test]
    fn density_change_keeps_the_anchor_row_under_the_pointer() {
        // The calculation that matters. At row 500,000 a naive "keep the pixel offset"
        // would move the view by a sixth of the corpus.
        let anchor_row = 500_000u64;
        let anchor_viewport_y = 300.0;

        let new_scroll =
            DensityTransition::rebase_scroll(anchor_row, 0.0, 28, 24, anchor_viewport_y);

        // The anchor row's new top edge, minus where it sat on screen.
        let expected = anchor_row as f64 * 24.0 - anchor_viewport_y;
        assert!((new_scroll - expected).abs() < 0.001);

        // And confirm the naive version really is badly wrong, so this test is testing
        // something.
        let naive = anchor_row as f64 * 28.0 - anchor_viewport_y;
        assert!(naive - new_scroll > 1_000_000.0);
    }

    #[test]
    fn rebasing_never_produces_a_negative_scroll() {
        let scroll = DensityTransition::rebase_scroll(0, 0.0, 28, 24, 500.0);
        assert!(scroll >= 0.0);
    }

    #[test]
    fn the_easing_curve_is_bounded_and_monotonic() {
        assert_eq!(ease_out_cubic(0.0), 0.0);
        assert_eq!(ease_out_cubic(1.0), 1.0);
        assert_eq!(ease_out_cubic(-5.0), 0.0);
        assert_eq!(ease_out_cubic(5.0), 1.0);

        let mut last = 0.0;
        for i in 0..=100 {
            let v = ease_out_cubic(i as f32 / 100.0);
            assert!(v >= last);
            last = v;
        }
    }
}
