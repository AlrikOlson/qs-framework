//! What the application knows about a row that the [`RowSource`] does not.
//!
//! Today that is one thing: a directory row can have agent sessions filed under it, and the
//! list says so. The type is deliberately narrow — a count, a confidence and a token name —
//! because of where the boundary is.
//!
//! # This crate does not learn what a session is, and that is the whole of the design
//!
//! `qs-ui` depends on `qs-gpu`, `qs-text` and `accesskit`. It does not depend on `qs-term`,
//! so [`qs_term::lifecycle::Lifecycle`] is not nameable here and must not become nameable:
//! the crate docs' boundary is that nothing below this crate knows what a file is and nothing
//! above it knows what a glyph is, and "what a shell is doing" is squarely above.
//!
//! So the mark carries what `crate::terminal::Status` already carries one surface up — a
//! **token name** rather than a colour, and the resolved state rather than the state machine.
//! The mapping from a lifecycle to a rail is made exactly once, in `qs`, and both the terminal
//! pane and this indicator read that one answer. A `Lifecycle` copied into this crate would be
//! a second mapping, and the two would drift the first time a state was added.
//!
//! # Marks travel on [`Interaction`], and that is not an accident of convenience
//!
//! [`Interaction`] is the one per-frame value that reaches **both**
//! [`ListRenderer::render`](crate::row::ListRenderer::render) and
//! [`SemanticTree::for_frame`](crate::a11y::SemanticTree::for_frame). The indicator has to
//! appear in both: a confidence rendered in ink and absent from the accessible name is a claim
//! made to sighted users only, which is the half-published state ADR 014 refuses. Any other
//! transport — a field on the renderer, an argument to `render` — reaches one consumer, and
//! the second channel then gets an answer of its own.
//!
//! [`RowSource`]: crate::row_source::RowSource
//! [`Interaction`]: crate::row::Interaction

/// The sessions filed under one directory, as a row is allowed to state them.
///
/// Constructed only through [`SessionMark::new`], which refuses a count of zero: "no sessions
/// here" is the *absence* of a mark, and a `SessionMark { count: 0 }` would be a row saying it
/// has zero of something — the blank-that-looks-like-a-state defect `Lifecycle` was built to
/// abolish, in a fourth place.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SessionMark {
    count: usize,
    vouched: bool,
    rail: Option<&'static str>,
}

impl SessionMark {
    /// A mark for `count` sessions, or `None` when there are none to mark.
    ///
    /// `vouched` is whether **any** of them had its directory vouched for by its shell (OSC 7),
    /// which is the strongest claim the filing earns — one reported session makes the folder a
    /// place a shell said it was, and the rest being inherited does not weaken that.
    ///
    /// `rail` is the token for the worst lifecycle among them, and `None` means the worst was a
    /// live shell. That is not a missing value: a running session is the state with nothing to
    /// report, and drawing a neutral rail for it would need `border/subtle`, which
    /// `design/tokens.json` states is deliberately outside the contrast gate because a boundary
    /// that identifies nothing must not be held to 3:1. Presence is already carried by the
    /// count; the rail is reserved for the two outcomes worth interrupting for.
    #[must_use]
    pub fn new(count: usize, vouched: bool, rail: Option<&'static str>) -> Option<Self> {
        (count > 0).then_some(Self {
            count,
            vouched,
            rail,
        })
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether any session here was vouched for by its shell.
    #[must_use]
    pub fn vouched(&self) -> bool {
        self.vouched
    }

    /// The rail token for the worst state, or `None` when every session is alive.
    #[must_use]
    pub fn rail(&self) -> Option<&'static str> {
        self.rail
    }

    /// What the row draws, in the slot a folder leaves empty.
    ///
    /// Short by construction: the size column is 72 logical pixels and this is measured against
    /// it, so the words are "2 sessions" rather than "2 agent sessions running here". The
    /// confidence is **not** in these words — it is in the ink, because a folder with four
    /// sessions and one with four inherited sessions must be distinguishable at a glance
    /// without the column growing. The accessible name says it in words instead; see
    /// [`SessionMark::spoken`].
    #[must_use]
    pub fn words(&self) -> String {
        let noun = if self.count == 1 {
            "session"
        } else {
            "sessions"
        };
        format!("{} {noun}", self.count)
    }

    /// What a screen reader hears, which is where the confidence becomes words.
    ///
    /// The distinction the ink carries has to be said out loud too, and in the pane's own
    /// vocabulary: [`crate::row::Interaction`]'s consumer in `qs` already tells a reader
    /// whether a *pane's* directory was reported or inherited, and a row that invented a third
    /// phrasing for the same fact would teach the distinction twice.
    #[must_use]
    pub fn spoken(&self) -> String {
        let confidence = if self.vouched {
            "reported by a shell"
        } else {
            "inherited, not confirmed"
        };
        format!("{} ({confidence})", self.words())
    }
}

/// This frame's marks, by corpus index.
///
/// A short association list rather than a map: it holds the *visible* rows that carry a mark,
/// which is bounded by a screenful and in practice is nearly always zero or one. A `HashMap`
/// here would allocate a table per frame to answer a question a linear scan over three entries
/// answers first.
///
/// Keyed by **corpus index** and not by [`RowId`](crate::row_source::RowId), because that is
/// the key `Interaction`'s other fields already use — `hovered`, `focused`, `pressed` and
/// `Selection` are all corpus indices, and a mark keyed differently would be the one field of
/// the frame's row state that a caller had to convert for.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct SessionMarks {
    by_index: Vec<(u64, SessionMark)>,
}

/// No marks at all: the frame every caller that has never heard of a session draws.
///
/// A `static` and not a `const`, for the reason [`crate::selection::NOTHING`] is one: a `const`
/// is materialized as a temporary at each use, so `Interaction::default` cannot return a
/// reference to it and `Interaction` would have to stop being `Copy` to hold it any other way.
pub static NO_MARKS: SessionMarks = SessionMarks {
    by_index: Vec::new(),
};

impl SessionMarks {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark the row at `index`.
    pub fn insert(&mut self, index: u64, mark: SessionMark) {
        self.by_index.push((index, mark));
    }

    /// The mark for `index`, if it has one.
    #[must_use]
    pub fn get(&self, index: u64) -> Option<&SessionMark> {
        self.by_index
            .iter()
            .find(|(at, _)| *at == index)
            .map(|(_, mark)| mark)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_index.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.by_index.len()
    }

    /// Empty it, keeping the allocation. Called once per frame by the builder.
    pub fn clear(&mut self) {
        self.by_index.clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn a_directory_with_no_sessions_has_no_mark_at_all() {
        // Acceptance 1, the half that is easy to get wrong: the absence has to be
        // unconstructible rather than merely unusual, or a row eventually draws "0 sessions".
        assert!(SessionMark::new(0, false, None).is_none());
        assert!(SessionMark::new(0, true, Some("rail/conflict")).is_none());
        assert!(SessionMark::new(1, false, None).is_some());
    }

    #[test]
    fn the_count_is_singular_at_one_and_plural_after() {
        assert_eq!(
            SessionMark::new(1, false, None).unwrap().words(),
            "1 session"
        );
        assert_eq!(
            SessionMark::new(2, false, None).unwrap().words(),
            "2 sessions"
        );
    }

    #[test]
    fn an_inherited_folder_and_a_vouched_one_say_different_things_out_loud() {
        // Acceptance 2, the accessible-name half. The ink half is asserted in `row.rs`, where
        // the colours are; both have to be asserted, because either channel alone silently
        // stops carrying the distinction for the users who depend on the other.
        let inherited = SessionMark::new(2, false, None).unwrap();
        let vouched = SessionMark::new(2, true, None).unwrap();
        assert_ne!(inherited.spoken(), vouched.spoken());
        // And both still say how many, so the confidence is added to the count rather than
        // replacing it.
        assert!(inherited.spoken().contains("2 sessions"));
        assert!(vouched.spoken().contains("2 sessions"));
    }

    #[test]
    fn the_drawn_words_are_short_enough_for_the_slot_they_go_in() {
        // The size column is 72 logical px at `ui/sm`. Asserted in characters rather than in
        // pixels for the reason the tab strip asserts its label the same way: the failure mode
        // is a label that GROWS, not a font that changed.
        for count in [1usize, 9, 10, 99] {
            let mark = SessionMark::new(count, true, None).unwrap();
            assert!(
                mark.words().chars().count() <= 11,
                "{:?} will be ellipsized in the size column",
                mark.words()
            );
        }
    }

    #[test]
    fn marks_answer_for_the_index_they_were_filed_under_and_no_other() {
        let mut marks = SessionMarks::new();
        marks.insert(7, SessionMark::new(3, true, Some("rail/conflict")).unwrap());
        assert_eq!(marks.get(7).map(SessionMark::count), Some(3));
        assert_eq!(marks.get(6), None);
        assert_eq!(marks.get(8), None);
        assert_eq!(marks.len(), 1);

        marks.clear();
        assert!(marks.is_empty());
        assert_eq!(marks.get(7), None);
    }

    #[test]
    fn the_empty_marks_const_is_empty() {
        assert!(NO_MARKS.is_empty());
        assert_eq!(NO_MARKS.get(0), None);
    }
}
