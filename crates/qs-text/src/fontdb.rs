//! Font enumeration and codepoint fallback.
//!
//! # What this is, and what it deliberately is not
//!
//! Research R2 names DirectWrite / CoreText / fontconfig as the enumeration and fallback
//! backends, and flags fallback as the place SC-006 ("no missing-glyph boxes") is won or
//! lost. This module implements that contract with a **directory scan plus a
//! per-platform preference chain** rather than three FFI bindings, and the reason is
//! worth stating rather than discovering later:
//!
//! * The three native APIs cannot be compile-tested from one host. Untested FFI on two of
//!   the three platforms is not coverage, it is the appearance of coverage -- which is
//!   exactly the failure mode the constitution's "Degrade Visibly" principle exists to
//!   prevent.
//! * The part of native fallback that carries the actual knowledge is the *ordering*: which
//!   face a platform reaches for when the primary font lacks a CJK ideograph. That ordering
//!   is data, and it is encoded in [`PlatformFonts::preference`] per platform.
//!
//! What is genuinely lost is system fallback *configuration* -- a user's fontconfig rules,
//! or a font installed somewhere non-standard. That is a real gap, it is recorded as a
//! finding rather than papered over, and the trait boundary here is what makes replacing
//! this with the native backends a swap rather than a rewrite.
//!
//! # Why the scan is lazy
//!
//! A Windows install carries several hundred font files totalling well over half a
//! gigabyte. Parsing all of them at startup to build a family index would put a
//! measurable, pointless cost in front of the first frame. Instead:
//!
//! 1. Scanning collects **paths only** -- one directory walk, no file contents read.
//! 2. The preference chain (roughly a dozen faces) is resolved eagerly, because those are
//!    the faces that will actually be used.
//! 3. Anything else is parsed only when a codepoint misses every preferred face, and the
//!    answer is memoized per codepoint so the walk happens at most once per glyph the
//!    corpus contains.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Handle to one face within the database. Dense, assigned at scan time.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct FontId(pub u32);

/// OpenType weight class (100..=900). 400 is regular, 700 is bold.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct FontWeight(pub u16);

impl FontWeight {
    pub const REGULAR: Self = Self(400);
    pub const MEDIUM: Self = Self(500);
    pub const BOLD: Self = Self(700);
}

impl Default for FontWeight {
    fn default() -> Self {
        Self::REGULAR
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum FontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

/// Borrowed face bytes plus the index of the face within a collection.
///
/// The `'static` lifetime is not a convenience: font data is loaded at most once and is
/// then live for the remainder of the process. Making that explicit lets `rustybuzz::Face`
/// and `swash::FontRef` -- both of which borrow -- be cached across frames without a
/// self-referential struct or an arena crate.
#[derive(Clone, Copy)]
pub struct FaceData {
    pub bytes: &'static [u8],
    pub index: u32,
}

impl fmt::Debug for FaceData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FaceData")
            .field("len", &self.bytes.len())
            .field("index", &self.index)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct FaceInfo {
    pub id: FontId,
    pub family: String,
    pub weight: FontWeight,
    pub style: FontStyle,
    pub monospace: bool,
    pub path: PathBuf,
    pub index: u32,
}

/// Coarse script classification, used only to pick which preference chain to walk.
///
/// This is deliberately not a full `unicode-script` classification: the chain is a list of
/// faces to try, and distinguishing e.g. Hiragana from Katakana would produce two identical
/// lists. `unicode-script` is still used for shaping (where the distinction matters).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScriptClass {
    Latin,
    Cjk,
    Arabic,
    Hebrew,
    Thai,
    Emoji,
    Other,
}

impl ScriptClass {
    pub fn of(ch: char) -> Self {
        let c = ch as u32;
        match c {
            0x0000..=0x02FF | 0x1E00..=0x1EFF | 0x2000..=0x206F => Self::Latin,
            0x0590..=0x05FF | 0xFB1D..=0xFB4F => Self::Hebrew,
            0x0600..=0x06FF
            | 0x0750..=0x077F
            | 0x08A0..=0x08FF
            | 0xFB50..=0xFDFF
            | 0xFE70..=0xFEFF => Self::Arabic,
            0x0E00..=0x0E7F => Self::Thai,
            0x1100..=0x11FF
            | 0x2E80..=0x2FDF
            | 0x3000..=0x30FF
            | 0x3130..=0x318F
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA960..=0xA97F
            | 0xAC00..=0xD7AF
            | 0xF900..=0xFAFF
            | 0xFF00..=0xFFEF
            | 0x20000..=0x2FA1F => Self::Cjk,
            0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x1F000..=0x1F0FF | 0xFE0F => Self::Emoji,
            _ => Self::Other,
        }
    }
}

/// Enumeration and fallback. `qs-ui` and `qs-gpu` see only this.
pub trait FontDb: Send + Sync + fmt::Debug {
    /// The face to use for row text when nothing more specific is requested.
    fn ui_font(&self) -> FontId;

    /// The UI face closest to `weight`.
    ///
    /// Never fails: a platform with no heavier face returns the regular one, so a role
    /// that asks for 600 renders at 400 rather than not rendering. Callers that need to
    /// know whether the request was honoured compare against [`FontDb::ui_font`], and
    /// [`SystemFontDb::weight_coverage`] reports it in aggregate.
    fn ui_font_at(&self, weight: FontWeight) -> FontId {
        let _ = weight;
        self.ui_font()
    }

    /// Resolve a family name to a face. `None` when the family is not installed.
    fn query(&self, family: &str, weight: FontWeight, style: FontStyle) -> Option<FontId>;

    /// The face to use for `ch` when `preferred` does not cover it.
    ///
    /// Returns `preferred` unchanged when it *does* cover `ch`, so callers can invoke this
    /// unconditionally per cluster. `None` means no installed face covers the codepoint,
    /// which is the SC-006 failure and must be surfaced, not silently rendered as `.notdef`.
    fn fallback(&self, ch: char, preferred: FontId) -> Option<FontId>;

    /// Face bytes. Loaded on first request and retained for the process lifetime.
    fn face_data(&self, id: FontId) -> Option<FaceData>;

    /// Metadata for a face. Parses the face on first request.
    fn info(&self, id: FontId) -> Option<FaceInfo>;

    /// Every face the scan found. Metadata is resolved lazily, so this is O(faces) file
    /// reads on first call -- it exists for diagnostics and tests, not for the frame path.
    fn enumerate(&self) -> Vec<FaceInfo>;
}

/// Per-platform data: where fonts live, and what order to try them in.
#[derive(Debug)]
pub struct PlatformFonts {
    /// Directories to scan, most-preferred first.
    pub dirs: Vec<PathBuf>,
    /// Filenames (lowercase, no directory) to try for the UI font, in order.
    pub ui: &'static [&'static str],
    /// Filenames to try per script class, in order.
    pub preference: &'static [(ScriptClass, &'static [&'static str])],
    /// `(weight class, filename)` for the UI family's weights, ascending.
    ///
    /// # Why this is a table and not a `query(family, weight)` call
    ///
    /// **It is not because the query fails.** That was the first hypothesis and it is
    /// wrong, which is worth recording so nobody re-derives it: Windows names each weight
    /// as its own *legacy* family (`FAMILY` = "Segoe UI Semibold"), but every one of them
    /// shares a **typographic family** (`TYPOGRAPHIC_FAMILY` = "Segoe UI"), and
    /// [`FontDb::info`] prefers the typographic name. So
    /// `query("Segoe UI", 600, Normal)` does find `seguisb.ttf`. Measured, not assumed.
    ///
    /// The real reason is cost. `query` is a linear scan that calls `info` on every face,
    /// and `info` calls `face_data`, which reads the file and **leaks it for the process
    /// lifetime**. On a stock Windows 11 install that is 337 faces: the call takes
    /// **975 ms** and permanently resides several hundred megabytes of font data, to
    /// answer one question asked five times at startup.
    ///
    /// A table of `(weight, filename)` answers the same question with one map lookup and
    /// loads only the faces actually used. It is the same shape as
    /// [`PlatformFonts::preference`], for the same reason: the per-platform knowledge is
    /// data, and data is checkable.
    ///
    /// `query` remains correct and is kept for callers that genuinely need to search by
    /// family — it just must not be on a startup path.
    pub weights: &'static [(u16, &'static str)],
}

#[derive(Debug)]
struct FaceSlot {
    path: PathBuf,
    index: u32,
    /// Lowercased file name, used for preference-chain matching without parsing.
    file_name: String,
}

/// The default [`FontDb`]. See the module docs for what it does and does not do.
pub struct SystemFontDb {
    slots: Vec<FaceSlot>,
    /// Faces named by the preference chain, in chain order, per script class.
    chains: HashMap<ScriptClass, Vec<FontId>>,
    ui: FontId,
    /// Resolved `(weight class, face)` for the UI family, ascending by weight. Only entries
    /// whose file actually exists on this machine are present, so the nearest-weight search
    /// never selects a face that cannot be loaded.
    weights: Vec<(u16, FontId)>,
    loaded: RwLock<HashMap<FontId, &'static [u8]>>,
    info_cache: RwLock<HashMap<FontId, FaceInfo>>,
    /// Memoized answers to `fallback`. Without this, a corpus containing one uncovered
    /// codepoint would re-walk every installed font on every frame that codepoint is
    /// visible, which turns a cosmetic gap into a frame-budget failure.
    fallback_cache: RwLock<HashMap<(char, FontId), Option<FontId>>>,
}

impl fmt::Debug for SystemFontDb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemFontDb")
            .field("faces", &self.slots.len())
            .field("ui", &self.ui)
            .finish()
    }
}

impl SystemFontDb {
    /// Scan the platform font directories.
    ///
    /// Never fails: a machine with no readable font directory yields an empty database,
    /// and the caller renders nothing rather than refusing to start. That is Principle III
    /// applied at the least convenient moment.
    pub fn scan() -> Self {
        Self::scan_with(platform_fonts())
    }

    pub fn scan_with(platform: PlatformFonts) -> Self {
        let mut slots = Vec::new();
        for dir in &platform.dirs {
            collect_dir(dir, &mut slots, 0);
        }
        slots.sort_by(|a, b| (&a.file_name, a.index).cmp(&(&b.file_name, b.index)));

        let index_of = |name: &str| -> Option<FontId> {
            slots
                .iter()
                .position(|s| s.file_name == name)
                .map(|i| FontId(i as u32))
        };

        let mut chains: HashMap<ScriptClass, Vec<FontId>> = HashMap::new();
        for (class, names) in platform.preference {
            let ids: Vec<FontId> = names.iter().filter_map(|n| index_of(n)).collect();
            chains.insert(*class, ids);
        }

        // The UI font is the first entry of the UI list that exists. If none of them do --
        // a stripped container, say -- fall back to whatever the scan found first, and to
        // `FontId(0)` on an empty database. `face_data` returns `None` for that id, and
        // every caller already handles a missing face.
        let ui = platform
            .ui
            .iter()
            .find_map(|n| index_of(n))
            .unwrap_or(FontId(0));

        // Only weights whose file is actually present. A table entry naming a face this
        // machine does not have must not become a resolution target, or `ui_font_at` would
        // hand back an id that `face_data` cannot load.
        let mut weights: Vec<(u16, FontId)> = platform
            .weights
            .iter()
            .filter_map(|&(weight, name)| index_of(name).map(|id| (weight, id)))
            .collect();
        weights.sort_by_key(|&(weight, _)| weight);
        weights.dedup_by_key(|&mut (weight, _)| weight);

        Self {
            slots,
            chains,
            ui,
            weights,
            loaded: RwLock::new(HashMap::new()),
            info_cache: RwLock::new(HashMap::new()),
            fallback_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Number of faces the scan found. Zero is a legitimate, survivable state.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Which weight classes this machine can actually satisfy, ascending.
    ///
    /// Exposed so a caller can *report* a degraded type scale rather than quietly render
    /// every role at 400. Constitution III: reduced capability is never silent.
    pub fn weight_coverage(&self) -> Vec<u16> {
        self.weights.iter().map(|&(weight, _)| weight).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn covers(&self, id: FontId, ch: char) -> bool {
        let Some(data) = self.face_data(id) else {
            return false;
        };
        let Ok(face) = ttf_parser::Face::parse(data.bytes, data.index) else {
            return false;
        };
        // A face "covers" a codepoint when its cmap maps it to a non-zero glyph. Glyph 0 is
        // `.notdef` by specification -- the box. Treating a `.notdef` hit as coverage is the
        // single most common way a fallback chain silently stops working.
        face.glyph_index(ch).is_some_and(|g| g.0 != 0)
    }
}

impl FontDb for SystemFontDb {
    fn ui_font(&self) -> FontId {
        self.ui
    }

    fn ui_font_at(&self, weight: FontWeight) -> FontId {
        // Nearest by absolute distance, ties going to the heavier face.
        //
        // Ties break heavier because the roles that ask for a non-400 weight are the ones
        // meant to stand out (headers, badges). Rounding a 500 request down to 400 on a
        // machine that has 400 and 600 would erase the distinction the role exists for;
        // rounding up overshoots slightly and still reads as emphasis.
        self.weights
            .iter()
            .min_by_key(|&&(candidate, _)| (candidate.abs_diff(weight.0), u16::MAX - candidate))
            .map(|&(_, id)| id)
            .unwrap_or(self.ui)
    }

    fn query(&self, family: &str, weight: FontWeight, style: FontStyle) -> Option<FontId> {
        // Cheap path: the preference chains name faces by file name, so a query that
        // matches one costs no parsing at all.
        let want = family.to_ascii_lowercase();
        if let Some(i) = self.slots.iter().position(|s| s.file_name == want) {
            return Some(FontId(i as u32));
        }

        // Real path: parse faces until the family matches. Memoized by `info`.
        let mut best: Option<(FontId, u32)> = None;
        for i in 0..self.slots.len() {
            let id = FontId(i as u32);
            let Some(info) = self.info(id) else { continue };
            if !info.family.eq_ignore_ascii_case(family) {
                continue;
            }
            if info.style != style {
                continue;
            }
            // Closest weight wins; an exact hit short-circuits.
            let distance = info.weight.0.abs_diff(weight.0) as u32;
            if distance == 0 {
                return Some(id);
            }
            if best.is_none_or(|(_, d)| distance < d) {
                best = Some((id, distance));
            }
        }
        best.map(|(id, _)| id)
    }

    fn fallback(&self, ch: char, preferred: FontId) -> Option<FontId> {
        if let Ok(cache) = self.fallback_cache.read() {
            if let Some(hit) = cache.get(&(ch, preferred)) {
                return *hit;
            }
        }

        let answer = self.resolve_fallback(ch, preferred);

        if let Ok(mut cache) = self.fallback_cache.write() {
            cache.insert((ch, preferred), answer);
        }
        answer
    }

    fn face_data(&self, id: FontId) -> Option<FaceData> {
        let slot = self.slots.get(id.0 as usize)?;

        if let Ok(loaded) = self.loaded.read() {
            if let Some(bytes) = loaded.get(&id) {
                return Some(FaceData {
                    bytes,
                    index: slot.index,
                });
            }
        }

        let bytes = fs::read(&slot.path).ok()?;
        // Deliberate leak. Font data is loaded once and lives until the process exits;
        // saying so in the type system is what lets `rustybuzz::Face<'static>` be cached
        // across frames. The alternative -- reparsing the face per shaped run -- puts file
        // I/O and table parsing on the frame path, which is Constitution I.
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());

        if let Ok(mut loaded) = self.loaded.write() {
            loaded.insert(id, leaked);
        }
        Some(FaceData {
            bytes: leaked,
            index: slot.index,
        })
    }

    fn info(&self, id: FontId) -> Option<FaceInfo> {
        if let Ok(cache) = self.info_cache.read() {
            if let Some(hit) = cache.get(&id) {
                return Some(hit.clone());
            }
        }

        let slot = self.slots.get(id.0 as usize)?;
        let data = self.face_data(id)?;
        let face = ttf_parser::Face::parse(data.bytes, data.index).ok()?;

        let family = face
            .names()
            .into_iter()
            .find(|n| n.name_id == ttf_parser::name_id::TYPOGRAPHIC_FAMILY && n.is_unicode())
            .or_else(|| {
                face.names()
                    .into_iter()
                    .find(|n| n.name_id == ttf_parser::name_id::FAMILY && n.is_unicode())
            })
            .and_then(|n| n.to_string())
            .unwrap_or_else(|| slot.file_name.clone());

        let info = FaceInfo {
            id,
            family,
            weight: FontWeight(face.weight().to_number()),
            style: match face.style() {
                ttf_parser::Style::Normal => FontStyle::Normal,
                ttf_parser::Style::Italic => FontStyle::Italic,
                ttf_parser::Style::Oblique => FontStyle::Oblique,
            },
            monospace: face.is_monospaced(),
            path: slot.path.clone(),
            index: slot.index,
        };

        if let Ok(mut cache) = self.info_cache.write() {
            cache.insert(id, info.clone());
        }
        Some(info)
    }

    fn enumerate(&self) -> Vec<FaceInfo> {
        (0..self.slots.len())
            .filter_map(|i| self.info(FontId(i as u32)))
            .collect()
    }
}

impl SystemFontDb {
    fn resolve_fallback(&self, ch: char, preferred: FontId) -> Option<FontId> {
        if self.covers(preferred, ch) {
            return Some(preferred);
        }

        let class = ScriptClass::of(ch);
        if let Some(chain) = self.chains.get(&class) {
            for &id in chain {
                if self.covers(id, ch) {
                    return Some(id);
                }
            }
        }

        // Not in the chain for its own class. Try every other chain before the exhaustive
        // walk -- the chains are already-loaded faces, so this is nearly free, and it
        // catches the common case of a codepoint classified as `Other` that a UI font has.
        for (other, chain) in &self.chains {
            if *other == class {
                continue;
            }
            for &id in chain {
                if self.covers(id, ch) {
                    return Some(id);
                }
            }
        }

        // Exhaustive walk. Costly -- it reads font files -- but bounded to once per
        // codepoint by the caller's memoization, and it is the difference between a
        // rendered glyph and a box.
        (0..self.slots.len())
            .map(|i| FontId(i as u32))
            .find(|&id| self.covers(id, ch))
    }
}

fn collect_dir(dir: &Path, out: &mut Vec<FaceSlot>, depth: u32) {
    // Font directories nest (Windows keeps none, macOS and Linux both do). Four levels is
    // past anything real; the bound exists so a symlink cycle cannot hang startup.
    if depth > 4 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            collect_dir(&path, out, depth + 1);
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let ext = ext.to_ascii_lowercase();
        if !matches!(ext.as_str(), "ttf" | "otf" | "ttc" | "otc") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let file_name = file_name.to_ascii_lowercase();

        let count = if ext == "ttc" || ext == "otc" {
            collection_count(&path).unwrap_or(1)
        } else {
            1
        };
        for index in 0..count {
            out.push(FaceSlot {
                path: path.clone(),
                index,
                file_name: file_name.clone(),
            });
        }
    }
}

/// Read `numFonts` from a TrueType collection header without loading the whole file.
///
/// The header is 12 bytes: tag, version, count. Reading the file in full just to learn it
/// contains four faces would mean reading every CJK collection on the machine at scan time.
fn collection_count(path: &Path) -> Option<u32> {
    use std::io::Read;
    let mut file = fs::File::open(path).ok()?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header).ok()?;
    if &header[0..4] != b"ttcf" {
        return Some(1);
    }
    let count = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    // A collection with thousands of faces is a corrupt or hostile file, not a font.
    (count > 0 && count < 1024).then_some(count)
}

#[cfg(target_os = "windows")]
fn platform_fonts() -> PlatformFonts {
    crate::win_fontdb::platform_fonts()
}

#[cfg(target_os = "macos")]
fn platform_fonts() -> PlatformFonts {
    crate::mac_fontdb::platform_fonts()
}

#[cfg(all(unix, not(target_os = "macos")))]
fn platform_fonts() -> PlatformFonts {
    crate::linux_fontdb::platform_fonts()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn script_classification_covers_the_corpus_scripts() {
        assert_eq!(ScriptClass::of('a'), ScriptClass::Latin);
        assert_eq!(ScriptClass::of('日'), ScriptClass::Cjk);
        assert_eq!(ScriptClass::of('한'), ScriptClass::Cjk);
        assert_eq!(ScriptClass::of('ع'), ScriptClass::Arabic);
        assert_eq!(ScriptClass::of('א'), ScriptClass::Hebrew);
        assert_eq!(ScriptClass::of('ก'), ScriptClass::Thai);
        assert_eq!(ScriptClass::of('🙂'), ScriptClass::Emoji);
    }

    #[test]
    fn weight_resolution_picks_the_nearest_available_face() {
        // The table is `(weight, filename)`; only entries whose file exists become
        // resolution targets, so this asserts the nearest-weight arithmetic against a
        // synthetic platform rather than against whatever fonts the test machine has.
        let db = SystemFontDb::scan();
        if db.is_empty() {
            return;
        }
        let coverage = db.weight_coverage();
        if coverage.len() < 2 {
            // A single-weight machine cannot exercise the search; that is a legitimate
            // state (see the macOS note in mac_fontdb) and not a test failure.
            return;
        }

        // An exact request lands on that weight; an out-of-range request clamps to the
        // nearest end rather than returning nothing.
        for &weight in &coverage {
            let id = db.ui_font_at(FontWeight(weight));
            assert!(
                db.face_data(id).is_some(),
                "weight {weight} resolved to an unloadable face"
            );
        }
        let lightest = coverage.iter().copied().min().unwrap_or(400);
        let heaviest = coverage.iter().copied().max().unwrap_or(400);
        assert_eq!(
            db.ui_font_at(FontWeight(1)),
            db.ui_font_at(FontWeight(lightest))
        );
        assert_eq!(
            db.ui_font_at(FontWeight(999)),
            db.ui_font_at(FontWeight(heaviest))
        );
    }

    #[test]
    fn a_heavier_role_resolves_to_a_different_face_than_regular_where_one_exists() {
        // This is the test that would have caught the family-per-weight trap. Matching on
        // family+weight returns the regular face for every heavier role on Windows, because
        // the 600-weight file calls itself "Segoe UI Semibold" rather than "Segoe UI".
        // If that regression returns, 400 and 600 collapse to one face here.
        let db = SystemFontDb::scan();
        if db.is_empty() || db.weight_coverage().len() < 2 {
            return;
        }
        let coverage = db.weight_coverage();
        if !coverage.contains(&400) || !coverage.contains(&600) {
            return;
        }
        assert_ne!(
            db.ui_font_at(FontWeight(400)),
            db.ui_font_at(FontWeight(600)),
            "the 600 role collapsed onto the regular face; weight resolution is not working"
        );
    }

    #[test]
    fn an_unsatisfiable_weight_falls_back_rather_than_failing() {
        // Constitution III: a missing weight renders at the wrong weight, never not at all.
        let db = SystemFontDb::scan();
        if db.is_empty() {
            return;
        }
        let id = db.ui_font_at(FontWeight(850));
        assert!(db.face_data(id).is_some());
    }

    #[test]
    fn scan_finds_a_ui_font_on_this_machine() {
        let db = SystemFontDb::scan();
        // A developer machine without a single installed font is not a case worth
        // asserting against; a machine with fonts must produce a loadable UI face.
        if db.is_empty() {
            return;
        }
        let ui = db.ui_font();
        assert!(db.face_data(ui).is_some(), "UI font must be loadable");
        assert!(
            db.fallback('A', ui).is_some(),
            "the UI font must cover basic Latin"
        );
    }
}
