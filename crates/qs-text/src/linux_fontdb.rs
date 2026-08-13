//! Linux font locations and fallback ordering.
//!
//! Linux is the platform where this module's approach loses the most, and it is worth being
//! precise about what: fontconfig is *configuration*, not just a font list. A user's
//! `fonts.conf` can substitute families, reorder the cascade, and set per-family rendering
//! options, none of which a directory scan can observe. What is preserved is the part that
//! decides whether a glyph renders at all -- which face covers the codepoint.
//!
//! The lists below assume the Noto family, which is what every mainstream distribution
//! ships as its fallback set, with DejaVu as the older floor.

use std::path::PathBuf;

use crate::fontdb::{PlatformFonts, ScriptClass};

const UI: &[&str] = &[
    "notosans-regular.ttf",
    "dejavusans.ttf",
    "liberationsans-regular.ttf",
    "cantarell-regular.otf",
    "ubuntu-r.ttf",
];

/// The monospaced faces a mainstream distribution actually ships, most-preferred first.
///
/// DejaVu Sans Mono is the one that is nearly always present -- it is the fontconfig
/// `monospace` alias's target on Debian, Fedora and Arch alike. Noto Sans Mono is the
/// modern preference where the Noto set is installed, and Liberation Mono is the
/// metric-compatible Courier substitute that comes with most office installs.
const MONO: &[&str] = &[
    "dejavusansmono.ttf",
    "notosansmono-regular.ttf",
    "liberationmono-regular.ttf",
    "ubuntumono-r.ttf",
    "freemono.ttf",
];

const PREFERENCE: &[(ScriptClass, &[&str])] = &[
    (
        ScriptClass::Latin,
        &[
            "notosans-regular.ttf",
            "dejavusans.ttf",
            "liberationsans-regular.ttf",
        ],
    ),
    (
        ScriptClass::Cjk,
        &[
            "notosanscjk-regular.ttc",
            "notosanscjksc-regular.otf",
            "notosanscjkjp-regular.otf",
            "notosanscjkkr-regular.otf",
            "wqy-zenhei.ttc",
            "droidsansfallbackfull.ttf",
        ],
    ),
    (
        ScriptClass::Arabic,
        &[
            "notosansarabic-regular.ttf",
            "notonaskharabic-regular.ttf",
            "dejavusans.ttf",
        ],
    ),
    (
        ScriptClass::Hebrew,
        &["notosanshebrew-regular.ttf", "dejavusans.ttf"],
    ),
    (
        ScriptClass::Thai,
        &[
            "notosansthai-regular.ttf",
            "notoserifthai-regular.ttf",
            "garuda.ttf",
        ],
    ),
    (
        ScriptClass::Emoji,
        &[
            "notocoloremoji.ttf",
            "notoemoji-regular.ttf",
            "opensymbol.ttf",
        ],
    ),
    (
        ScriptClass::Other,
        &["notosans-regular.ttf", "dejavusans.ttf", "opensymbol.ttf"],
    ),
];

/// Noto Sans and DejaVu weights, by file.
///
/// Noto ships one file per weight with the weight in the filename, which is the friendliest
/// of the three platforms for this table. DejaVu has only Book and Bold, so a machine with
/// DejaVu alone gets a two-step scale.
const WEIGHTS: &[(u16, &str)] = &[
    (300, "notosans-light.ttf"),
    (400, "notosans-regular.ttf"),
    (400, "dejavusans.ttf"),
    (500, "notosans-medium.ttf"),
    (600, "notosans-semibold.ttf"),
    (700, "notosans-bold.ttf"),
    (700, "dejavusans-bold.ttf"),
];

pub fn platform_fonts() -> PlatformFonts {
    let mut dirs = Vec::new();

    // XDG user fonts, then the legacy per-user location, then system-wide. Order matters:
    // a user-installed face should win over the distribution's.
    if let Some(data_home) = std::env::var_os("XDG_DATA_HOME") {
        dirs.push(PathBuf::from(data_home).join("fonts"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(&home).join(".local/share/fonts"));
        dirs.push(PathBuf::from(&home).join(".fonts"));
    }
    dirs.push(PathBuf::from("/usr/local/share/fonts"));
    dirs.push(PathBuf::from("/usr/share/fonts"));
    dirs.push(PathBuf::from("/run/host/fonts")); // Flatpak

    PlatformFonts {
        dirs,
        ui: UI,
        mono: MONO,
        preference: PREFERENCE,
        weights: WEIGHTS,
    }
}
