//! macOS font locations and fallback ordering.
//!
//! Encodes the ordering CoreText's cascade list produces for an en-US system. See the
//! [`crate::fontdb`] module docs for why this is a directory scan rather than a
//! `CTFontCreateForString` binding.

use std::path::PathBuf;

use crate::fontdb::{PlatformFonts, ScriptClass};

/// `SFNS.ttf` is San Francisco, the system UI face from macOS 10.11. It is not installable
/// as a normal family and is only reachable by path, which is one reason the scan works by
/// file name. Helvetica is the pre-SF floor.
const UI: &[&str] = &[
    "sfns.ttf",
    "sfnsdisplay.ttf",
    "sfnstext.ttf",
    "helvetica.ttc",
    "helveticaneue.ttc",
];

const PREFERENCE: &[(ScriptClass, &[&str])] = &[
    (
        ScriptClass::Latin,
        &["sfns.ttf", "helvetica.ttc", "arial.ttf"],
    ),
    (
        ScriptClass::Cjk,
        &[
            "pingfang.ttc", // Simplified + Traditional Chinese
            "hiragino sans gb.ttc",
            "hiraginosans.ttc",     // Japanese
            "applesdgothicneo.ttc", // Korean
            "songti.ttc",
        ],
    ),
    (
        ScriptClass::Arabic,
        &["geezapro.ttc", "sfns.ttf", "arialuni.ttf"],
    ),
    (ScriptClass::Hebrew, &["arialhb.ttc", "sfns.ttf"]),
    (
        ScriptClass::Thai,
        &["thonburi.ttc", "ayuthaya.ttf", "sfns.ttf"],
    ),
    (
        ScriptClass::Emoji,
        &[
            "apple color emoji.ttc",
            "applecoloremoji.ttc",
            "applesymbols.ttf",
        ],
    ),
    (
        ScriptClass::Other,
        &["sfns.ttf", "applesymbols.ttf", "helvetica.ttc"],
    ),
];

/// San Francisco's weights, by file.
///
/// Unlike Windows, macOS ships SF as variable fonts (`SFNS.ttf` carries a weight axis), so
/// the honest position is that this table is a *partial* solution here: it names the
/// separate Helvetica weights as a fallback and otherwise resolves everything to SF
/// Regular. A machine with only `SFNS.ttf` therefore renders every role at 400 until
/// variation-axis support lands, and `weight_coverage()` reports that truthfully rather
/// than implying a hierarchy that is not being drawn.
const WEIGHTS: &[(u16, &str)] = &[
    (400, "sfns.ttf"),
    (400, "sfnstext.ttf"),
    (400, "helvetica.ttc"),
    (700, "helveticabold.ttf"),
];

pub fn platform_fonts() -> PlatformFonts {
    let mut dirs = Vec::new();

    // User fonts first, then the two system locations. `/System/Library/Fonts` is on the
    // sealed system volume and holds the faces that are always present; `/Library/Fonts`
    // holds machine-wide installs.
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(&home).join("Library/Fonts"));
    }
    dirs.push(PathBuf::from("/Library/Fonts"));
    dirs.push(PathBuf::from("/System/Library/Fonts"));

    PlatformFonts {
        dirs,
        ui: UI,
        preference: PREFERENCE,
        weights: WEIGHTS,
    }
}
