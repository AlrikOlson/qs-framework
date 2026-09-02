//! Shaping: text plus a font plus a size becomes positioned glyphs.
//!
//! # Two paths, one implementation
//!
//! Task ordering split this work: US1 needs "just enough shaping to render real file
//! names", US2 needs full script coverage. The temptation is two functions. That would be
//! a mistake -- the US1 frame-time measurement would then be a measurement of the *fast*
//! path, and US2 would silently make it wrong.
//!
//! So there is one [`Shaper::shape`], with an internal short-circuit: text that is pure
//! ASCII and fully covered by the requested face skips bidi resolution and font
//! segmentation, because for that input both are provably identity operations. The
//! measured path and the correct path are the same path; the short-circuit only skips work
//! that would have no effect.

use std::collections::HashMap;
use std::sync::Arc;

use rustybuzz::{Direction, Feature, UnicodeBuffer};

use crate::bidi::{BaseDirection, resolve_paragraph};
use crate::fontdb::{FontDb, FontId};
use crate::{Features, PxSize};

/// One positioned glyph, ready for the atlas and the batcher.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ShapedGlyph {
    /// The face this glyph belongs to. Not necessarily the requested face -- fallback may
    /// have selected another, and the atlas key must carry the real one or two different
    /// glyphs collide on one cache slot.
    pub font: FontId,
    pub glyph_id: u16,
    /// Pen position within the run, in pixels, already including the glyph's own offset.
    pub x: f32,
    pub y: f32,
    pub advance: f32,
    /// Byte offset of the grapheme cluster this glyph came from, in the original string.
    /// Truncation (middle ellipsis) cuts on cluster boundaries, never between glyphs of
    /// one cluster -- that is how you get half a Devanagari syllable on screen.
    pub cluster: u32,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct ShapedRun {
    pub glyphs: Vec<ShapedGlyph>,
    /// Total advance width in pixels.
    pub width: f32,
    /// Clusters for which no installed face had a glyph. Non-zero is the SC-006 failure
    /// and is reported per frame rather than discovered in a screenshot.
    pub missing_glyphs: u32,
    /// The source text contained an explicit bidi override -- see [`crate::bidi`].
    pub has_directional_override: bool,
}

impl ShapedRun {
    /// Advance width up to (not including) the glyph at `index`.
    pub fn width_before(&self, index: usize) -> f32 {
        self.glyphs.get(index).map_or(self.width, |g| g.x)
    }

    /// Index of the first glyph whose pen position is at or past `x`.
    pub fn glyph_at_x(&self, x: f32) -> usize {
        self.glyphs.partition_point(|g| g.x < x)
    }
}

/// Vertical metrics for a face at a size, in pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FontMetrics {
    pub ascent: f32,
    pub descent: f32,
    pub line_gap: f32,
    /// Recommended distance between baselines.
    pub line_height: f32,
    /// Height of a lowercase `x`, used to centre text optically within a row.
    pub x_height: f32,
}

/// Owns the parsed faces and the scratch buffer.
///
/// `&mut self` on [`Shaper::shape`] is deliberate. Shaping mutates a face cache and reuses
/// one `UnicodeBuffer` allocation across calls, and a `Shaper` is therefore owned by
/// exactly one thread. That is Constitution II expressed in a signature: there is no
/// interior mutability here to make sharing look safe when it would not be cheap.
pub struct Shaper {
    db: Arc<dyn FontDb>,
    faces: HashMap<FontId, rustybuzz::Face<'static>>,
    /// Reused across calls. Steady-state shaping allocates nothing.
    scratch: Option<UnicodeBuffer>,
    /// Reused segmentation buffer, same reason. `(face, start_byte, end_byte)`.
    segments: Vec<(FontId, usize, usize)>,
}

impl std::fmt::Debug for Shaper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shaper")
            .field("faces_cached", &self.faces.len())
            .finish()
    }
}

impl Shaper {
    pub fn new(db: Arc<dyn FontDb>) -> Self {
        Self {
            db,
            faces: HashMap::new(),
            scratch: Some(UnicodeBuffer::new()),
            segments: Vec::new(),
        }
    }

    pub fn db(&self) -> &Arc<dyn FontDb> {
        &self.db
    }

    /// Parse and cache a face. Returns `None` if the face is missing or unparseable, which
    /// is survivable -- the caller renders nothing for that run rather than panicking.
    fn face(&mut self, id: FontId) -> Option<&rustybuzz::Face<'static>> {
        if !self.faces.contains_key(&id) {
            let data = self.db.face_data(id)?;
            let face = rustybuzz::Face::from_slice(data.bytes, data.index)?;
            self.faces.insert(id, face);
        }
        self.faces.get(&id)
    }

    pub fn metrics(&mut self, font: FontId, size: PxSize) -> Option<FontMetrics> {
        let data = self.db.face_data(font)?;
        let face = ttf_parser::Face::parse(data.bytes, data.index).ok()?;
        let upem = f32::from(face.units_per_em());
        if upem <= 0.0 {
            return None;
        }
        let scale = size.to_f32() / upem;
        let ascent = f32::from(face.ascender()) * scale;
        let descent = -f32::from(face.descender()) * scale;
        let line_gap = f32::from(face.line_gap()) * scale;
        Some(FontMetrics {
            ascent,
            descent,
            line_gap,
            line_height: ascent + descent + line_gap,
            x_height: face
                .x_height()
                .map_or(ascent * 0.52, |x| f32::from(x) * scale),
        })
    }

    /// Shape one name.
    ///
    /// `font` is the *requested* face; individual clusters may be shaped with another if
    /// fallback selects one.
    pub fn shape(
        &mut self,
        text: &str,
        font: FontId,
        size: PxSize,
        features: Features,
    ) -> ShapedRun {
        if text.is_empty() {
            return ShapedRun::default();
        }

        let feature_list = feature_list(features);

        // Fast path: ASCII text has no bidi reordering to do (every character is strongly
        // LTR or neutral with an LTR base) and no combining marks whose clusters could
        // merge. If the requested face also covers all of it, segmentation is the identity.
        if text.is_ascii() && self.covers_ascii(font, text) {
            let mut run = ShapedRun::default();
            self.shape_segment(text, 0, font, size, &feature_list, false, &mut run);
            return run;
        }

        let resolved = resolve_paragraph(text, BaseDirection::Auto);
        let mut run = ShapedRun {
            has_directional_override: resolved.has_directional_override,
            ..ShapedRun::default()
        };

        for visual in &resolved.runs {
            let Some(slice) = text.get(visual.range.clone()) else {
                continue;
            };
            let rtl = visual.level.is_rtl();

            // Split the directional run further, by which face covers each character.
            self.segment_by_font(slice, visual.range.start, font, &mut run);

            // `segments` was filled by `segment_by_font`; move it out so the loop can take
            // `&mut self`, then move it back. The allocation survives the round trip, which
            // is what keeps steady-state shaping free of allocation.
            let mut segments = std::mem::take(&mut self.segments);
            for &(seg_font, start, end) in &segments {
                let Some(sub) = text.get(start..end) else {
                    continue;
                };
                self.shape_segment(sub, start, seg_font, size, &feature_list, rtl, &mut run);
            }
            segments.clear();
            self.segments = segments;
        }

        run
    }

    fn covers_ascii(&self, font: FontId, text: &str) -> bool {
        // One representative check is not enough -- a face can carry Latin letters and lack
        // e.g. the underscore. Names are short, so checking every distinct byte is cheap,
        // and `fallback` memoizes per codepoint.
        text.chars()
            .all(|c| c == ' ' || self.db.fallback(c, font) == Some(font))
    }

    /// Fill `self.segments` with maximal spans sharing one face.
    fn segment_by_font(
        &mut self,
        slice: &str,
        base_offset: usize,
        font: FontId,
        run: &mut ShapedRun,
    ) {
        self.segments.clear();
        let mut current: Option<(FontId, usize, usize)> = None;

        for (i, ch) in slice.char_indices() {
            let at = base_offset + i;
            let end = at + ch.len_utf8();

            // Spaces and other neutrals should not force a face change -- doing so splits
            // "日本 語" into three shaping runs and loses any kerning across the space.
            let resolved = if ch.is_whitespace() {
                current.map(|(f, _, _)| f).unwrap_or(font)
            } else {
                match self.db.fallback(ch, font) {
                    Some(f) => f,
                    None => {
                        // No installed face covers this codepoint. Shape it with the
                        // requested face anyway -- it will produce `.notdef` -- and count
                        // it, because a silent box is the failure SC-006 names.
                        run.missing_glyphs += 1;
                        font
                    }
                }
            };

            current = match current {
                Some((f, s, _)) if f == resolved => Some((f, s, end)),
                Some((f, s, e)) => {
                    self.segments.push((f, s, e));
                    Some((resolved, at, end))
                }
                None => Some((resolved, at, end)),
            };
        }
        if let Some((f, s, e)) = current {
            self.segments.push((f, s, e));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn shape_segment(
        &mut self,
        text: &str,
        byte_offset: usize,
        font: FontId,
        size: PxSize,
        features: &[Feature],
        rtl: bool,
        out: &mut ShapedRun,
    ) {
        let px = size.to_f32();
        // Populate the face cache first. The borrow from `face()` ends here, which is what
        // lets the scratch buffer be taken out of `self` on the next line.
        if self.face(font).is_none() {
            return;
        }

        let mut buffer = self.scratch.take().unwrap_or_default();
        buffer.push_str(text);
        buffer.set_direction(if rtl {
            Direction::RightToLeft
        } else {
            Direction::LeftToRight
        });

        let Some(face) = self.faces.get(&font) else {
            self.scratch = Some(buffer);
            return;
        };
        let upem = face.units_per_em();
        if upem <= 0 {
            self.scratch = Some(buffer);
            return;
        }
        let scale = px / upem as f32;

        let glyphs = rustybuzz::shape(face, features, buffer);

        let mut pen = out.width;
        for (info, pos) in glyphs
            .glyph_infos()
            .iter()
            .zip(glyphs.glyph_positions().iter())
        {
            let advance = pos.x_advance as f32 * scale;
            out.glyphs.push(ShapedGlyph {
                font,
                // rustybuzz guarantees the value fits in u16 after shaping.
                glyph_id: info.glyph_id as u16,
                x: pen + pos.x_offset as f32 * scale,
                y: -(pos.y_offset as f32) * scale,
                advance,
                cluster: (byte_offset + info.cluster as usize) as u32,
            });
            pen += advance;
        }
        out.width = pen;

        // Hand the allocation back for the next call.
        self.scratch = Some(glyphs.clear());
    }
}

fn feature_list(features: Features) -> Vec<Feature> {
    let mut list = Vec::new();
    if features.tabular_figures {
        // `tnum`: every digit gets the same advance, so a column of file sizes forms a
        // grid instead of a ragged edge (FR-013).
        list.push(Feature::new(
            rustybuzz::ttf_parser::Tag::from_bytes(b"tnum"),
            1,
            ..,
        ));
    }
    if features.slashed_zero {
        // `zero`: the slashed alternate of `0`, where the face draws one.
        list.push(Feature::new(
            rustybuzz::ttf_parser::Tag::from_bytes(b"zero"),
            1,
            ..,
        ));
    }
    list
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;
    use crate::fontdb::SystemFontDb;

    /// Each flag maps to exactly its tag, and the default asks the face for nothing.
    #[test]
    fn features_map_to_their_tags() {
        let tags = |f: Features| -> Vec<rustybuzz::ttf_parser::Tag> {
            feature_list(f).iter().map(|feature| feature.tag).collect()
        };
        assert!(tags(Features::default()).is_empty());
        let tnum = rustybuzz::ttf_parser::Tag::from_bytes(b"tnum");
        let zero = rustybuzz::ttf_parser::Tag::from_bytes(b"zero");
        assert_eq!(
            tags(Features {
                tabular_figures: true,
                slashed_zero: false,
            }),
            vec![tnum]
        );
        assert_eq!(
            tags(Features {
                tabular_figures: false,
                slashed_zero: true,
            }),
            vec![zero]
        );
        assert_eq!(
            tags(Features {
                tabular_figures: true,
                slashed_zero: true,
            }),
            vec![tnum, zero]
        );
    }

    fn shaper() -> Option<(Shaper, FontId)> {
        let db = SystemFontDb::scan();
        if db.is_empty() {
            return None;
        }
        let ui = db.ui_font();
        Some((Shaper::new(Arc::new(db)), ui))
    }

    #[test]
    fn latin_shapes_to_one_glyph_per_character() {
        let Some((mut s, ui)) = shaper() else { return };
        let run = s.shape("readme.md", ui, PxSize::new(14.0), Features::default());
        assert_eq!(run.glyphs.len(), 9);
        assert!(run.width > 0.0);
        assert_eq!(run.missing_glyphs, 0);
    }

    #[test]
    fn pen_positions_are_monotonic_and_match_the_total_width() {
        let Some((mut s, ui)) = shaper() else { return };
        let run = s.shape(
            "a quick brown fox.txt",
            ui,
            PxSize::new(14.0),
            Features::default(),
        );
        let mut last = f32::NEG_INFINITY;
        for g in &run.glyphs {
            assert!(
                g.x >= last - 0.001,
                "glyph pen positions must not go backwards"
            );
            last = g.x;
        }
        assert!(run.width >= last);
    }

    #[test]
    fn clusters_index_the_original_string() {
        let Some((mut s, ui)) = shaper() else { return };
        let text = "abc.rs";
        let run = s.shape(text, ui, PxSize::new(14.0), Features::default());
        for g in &run.glyphs {
            assert!(
                text.is_char_boundary(g.cluster as usize),
                "cluster {} is not a char boundary in {text:?}",
                g.cluster
            );
        }
    }

    #[test]
    fn the_ascii_fast_path_and_the_general_path_agree() {
        // The whole justification for the short-circuit is that it changes nothing. If
        // this ever fails, the US1 frame-time number stops describing the US2 renderer.
        let Some((mut s, ui)) = shaper() else { return };
        let text = "budget-2026.xlsx";

        let fast = s.shape(text, ui, PxSize::new(14.0), Features::default());

        // Force the general path by appending and removing a non-ASCII character.
        let general = {
            let resolved = resolve_paragraph(text, BaseDirection::Auto);
            let mut run = ShapedRun::default();
            for visual in &resolved.runs {
                let slice = &text[visual.range.clone()];
                s.segment_by_font(slice, visual.range.start, ui, &mut run);
                let segments = std::mem::take(&mut s.segments);
                for &(f, start, end) in &segments {
                    s.shape_segment(
                        &text[start..end],
                        start,
                        f,
                        PxSize::new(14.0),
                        &[],
                        false,
                        &mut run,
                    );
                }
                s.segments = segments;
            }
            run
        };

        assert_eq!(fast.glyphs.len(), general.glyphs.len());
        assert!((fast.width - general.width).abs() < 0.01);
    }

    #[test]
    fn an_empty_name_shapes_to_nothing_without_panicking() {
        let Some((mut s, ui)) = shaper() else { return };
        let run = s.shape("", ui, PxSize::new(14.0), Features::default());
        assert!(run.glyphs.is_empty());
        assert_eq!(run.width, 0.0);
    }
}
