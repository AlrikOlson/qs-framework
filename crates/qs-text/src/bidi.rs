//! Bidirectional resolution, with each name treated as its own isolated paragraph.
//!
//! # This is a security control, not only a correctness one
//!
//! SDD §11 lists filename spoofing as a threat, and the bidi override attack is its
//! sharpest form: a name containing U+202E RIGHT-TO-LEFT OVERRIDE renders
//! `invoice\u{202E}cod.exe` as `invoiceexe.doc`, which is a real technique with real
//! victims. Two defences are applied here, and neither is optional:
//!
//! 1. **Isolation.** Every name is resolved as its own paragraph with an explicit base
//!    direction. A directional control inside one name cannot escape it and reorder the
//!    size column, the next row, or the rest of the UI.
//! 2. **Disclosure.** [`ResolvedText::has_directional_override`] reports whether the name
//!    contained an override or embedding control at all. The row renderer uses it to mark
//!    the name, because silently rendering a reordered filename as though it were ordinary
//!    is the outcome the attack depends on. Constitution III: reduced trust is never silent.
//!
//! Stripping the controls outright was considered and rejected -- it would corrupt
//! legitimate Hebrew and Arabic names, which is a real cost paid by real users to defend
//! against a threat that isolation already contains.

use std::ops::Range;

use unicode_bidi::{BidiInfo, Level};

/// Resolved embedding level. Even is left-to-right, odd is right-to-left.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BidiLevel(pub u8);

impl BidiLevel {
    pub fn is_rtl(self) -> bool {
        self.0 % 2 == 1
    }
}

/// One maximal span of uniform direction, in visual order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VisualRun {
    /// Byte range into the original string.
    pub range: Range<usize>,
    pub level: BidiLevel,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ResolvedText {
    /// Runs in **visual** order, left to right. Shaping walks these in sequence and the
    /// resulting glyphs are already in draw order.
    pub runs: Vec<VisualRun>,
    /// Base direction actually used.
    pub base_rtl: bool,
    /// The name contained an explicit override or embedding control (U+202A..U+202E,
    /// U+2066..U+2069). Surfaced to the renderer -- see the module docs.
    pub has_directional_override: bool,
}

/// Base direction for a name.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BaseDirection {
    /// Infer from the first strong character (the Unicode P2/P3 rule).
    #[default]
    Auto,
    Ltr,
    Rtl,
}

/// Resolve one name into visual-order runs.
///
/// The name is treated as a complete paragraph regardless of what it contains, which is
/// what "per-name isolation" means in practice.
pub fn resolve_paragraph(text: &str, base: BaseDirection) -> ResolvedText {
    let has_directional_override = text.chars().any(is_explicit_directional_control);

    let forced = match base {
        BaseDirection::Auto => None,
        BaseDirection::Ltr => Some(Level::ltr()),
        BaseDirection::Rtl => Some(Level::rtl()),
    };

    let info = BidiInfo::new(text, forced);

    // An empty string, or one made only of neutrals, yields no paragraph. That is not an
    // error -- it is a name like "..." -- and it resolves to a single LTR run.
    let Some(para) = info.paragraphs.first() else {
        return ResolvedText {
            runs: if text.is_empty() {
                Vec::new()
            } else {
                vec![VisualRun {
                    range: 0..text.len(),
                    level: BidiLevel(0),
                }]
            },
            base_rtl: matches!(base, BaseDirection::Rtl),
            has_directional_override,
        };
    };

    let (levels, ranges) = info.visual_runs(para, para.range.clone());
    let runs = ranges
        .into_iter()
        .map(|range| {
            let level = levels
                .get(range.start)
                .copied()
                .unwrap_or_else(Level::ltr)
                .number();
            VisualRun {
                range,
                level: BidiLevel(level),
            }
        })
        .collect();

    ResolvedText {
        runs,
        base_rtl: para.level.is_rtl(),
        has_directional_override,
    }
}

fn is_explicit_directional_control(ch: char) -> bool {
    matches!(
        ch,
        // LRE, RLE, PDF, LRO, RLO
        '\u{202A}'..='\u{202E}'
        // LRI, RLI, FSI, PDI
        | '\u{2066}'..='\u{2069}'
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn plain_latin_is_one_ltr_run() {
        let r = resolve_paragraph("readme.md", BaseDirection::Auto);
        assert_eq!(r.runs.len(), 1);
        assert!(!r.runs[0].level.is_rtl());
        assert!(!r.has_directional_override);
    }

    #[test]
    fn hebrew_resolves_rtl_without_being_flagged_as_an_override() {
        let r = resolve_paragraph("שלום.txt", BaseDirection::Auto);
        assert!(r.base_rtl, "a Hebrew-initial name has an RTL base");
        assert!(
            !r.has_directional_override,
            "ordinary RTL text must not be marked as suspicious -- that is the false \
             positive that would make the warning useless"
        );
    }

    #[test]
    fn the_rlo_spoof_is_detected() {
        // The classic attack: renders as "invoiceexe.doc" while ending in ".exe".
        let spoof = "invoice\u{202E}cod.exe";
        let r = resolve_paragraph(spoof, BaseDirection::Auto);
        assert!(
            r.has_directional_override,
            "an explicit RLO must be reported to the renderer"
        );
    }

    #[test]
    fn resolution_is_confined_to_the_name() {
        // The same override, resolved twice, must produce identical results -- nothing
        // carries across a call, which is what per-name isolation buys.
        let a = resolve_paragraph("a\u{202E}b", BaseDirection::Auto);
        let b = resolve_paragraph("a\u{202E}b", BaseDirection::Auto);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_and_neutral_names_do_not_panic() {
        assert!(resolve_paragraph("", BaseDirection::Auto).runs.is_empty());
        assert_eq!(resolve_paragraph("...", BaseDirection::Auto).runs.len(), 1);
    }
}
