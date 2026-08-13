//! Windows font locations and fallback ordering.
//!
//! The ordering below is the part of DirectWrite's behaviour that actually matters for
//! SC-006 -- when Segoe UI lacks a codepoint, which face does Windows reach for? These
//! lists encode that answer as data. See the [`crate::fontdb`] module docs for why this
//! is a directory scan rather than an `IDWriteFontFallback` binding.

use std::path::PathBuf;

use crate::fontdb::{PlatformFonts, ScriptClass};

/// Segoe UI is the shell font from Vista onward and covers Latin, Greek, Cyrillic, Arabic
/// and Hebrew. Arial is the floor: present on every Windows install ever shipped.
const UI: &[&str] = &["segoeui.ttf", "arial.ttf", "tahoma.ttf"];

/// Cascadia Mono is the terminal face Windows has shipped since 2019 and is what a stock
/// Windows Terminal draws with. Consolas is the pre-Cascadia floor and is on every install
/// from Vista onward; Courier New is the floor beneath that.
///
/// `cascadiamono.ttf` before `cascadiacode.ttf`: Code is the same design with programming
/// ligatures, and a ligature in a terminal grid puts two cells' worth of ink in one cell.
const MONO: &[&str] = &[
    "cascadiamono.ttf",
    "consola.ttf",
    "cascadiacode.ttf",
    "lucon.ttf",
    "cour.ttf",
];

const PREFERENCE: &[(ScriptClass, &[&str])] = &[
    (
        ScriptClass::Latin,
        &["segoeui.ttf", "arial.ttf", "tahoma.ttf"],
    ),
    // Simplified Chinese, Traditional Chinese, Korean, Japanese -- in the order a
    // en-US install resolves them. A locale-aware build would reorder these three;
    // doing so needs the user's locale, which is an M1 input.
    (
        ScriptClass::Cjk,
        &[
            "msyh.ttc",    // Microsoft YaHei UI  (Simplified Chinese)
            "msjh.ttc",    // Microsoft JhengHei  (Traditional Chinese)
            "yugothm.ttc", // Yu Gothic Medium    (Japanese)
            "malgun.ttf",  // Malgun Gothic       (Korean)
            "simsun.ttc",  // fallback for older installs
            "msgothic.ttc",
        ],
    ),
    (
        ScriptClass::Arabic,
        &["segoeui.ttf", "tahoma.ttf", "arial.ttf"],
    ),
    (
        ScriptClass::Hebrew,
        &["segoeui.ttf", "david.ttf", "arial.ttf"],
    ),
    (
        ScriptClass::Thai,
        &["leelawui.ttf", "leelawad.ttf", "tahoma.ttf"],
    ),
    (
        ScriptClass::Emoji,
        &["seguiemj.ttf", "seguisym.ttf", "segoeui.ttf"],
    ),
    (
        ScriptClass::Other,
        &["segoeui.ttf", "seguisym.ttf", "arial.ttf"],
    ),
];

/// Segoe UI's weights, by file.
///
/// Verified against a stock Windows 11 install rather than assumed:
///
/// | File | legacy `FAMILY` | `TYPOGRAPHIC_FAMILY` | weight |
/// |---|---|---|---|
/// | `segoeuil.ttf` | Segoe UI Light | Segoe UI | 300 |
/// | `segoeuisl.ttf` | Segoe UI Semilight | Segoe UI | 350 |
/// | `segoeui.ttf` | Segoe UI | Segoe UI | 400 |
/// | `seguisb.ttf` | Segoe UI Semibold | Segoe UI | 600 |
/// | `segoeuib.ttf` | Segoe UI | Segoe UI | 700 |
///
/// Note the split between the two family names — the legacy one differs per weight, the
/// typographic one does not. Reading the wrong one is how you conclude that weight
/// resolution is impossible when it is merely expensive; see
/// [`crate::fontdb::PlatformFonts::weights`].
///
/// There is no 500 (Medium) face. A `ui/xs` role asking for 500 therefore resolves to 600,
/// which is the nearest available and the direction that preserves the emphasis the role
/// exists for.
///
/// `SegUIVar.ttf` (a variable font with a real weight axis) also exists on recent builds.
/// Using it would give every intermediate weight instead of these five steps, but it needs
/// variation-axis support in the shaper and rasterizer, which is a larger change than this
/// table and buys little for five fixed roles.
const WEIGHTS: &[(u16, &str)] = &[
    (300, "segoeuil.ttf"),  // Light
    (350, "segoeuisl.ttf"), // Semilight
    (400, "segoeui.ttf"),   // Regular
    (600, "seguisb.ttf"),   // Semibold
    (700, "segoeuib.ttf"),  // Bold
];

pub fn platform_fonts() -> PlatformFonts {
    let mut dirs = Vec::new();

    // Per-user fonts (installed without elevation) shadow machine fonts, so they are
    // scanned first -- a user who installed their own Noto build should get it.
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        dirs.push(PathBuf::from(local).join("Microsoft/Windows/Fonts"));
    }
    let windir = std::env::var_os("WINDIR").unwrap_or_else(|| "C:/Windows".into());
    dirs.push(PathBuf::from(windir).join("Fonts"));

    PlatformFonts {
        dirs,
        ui: UI,
        mono: MONO,
        preference: PREFERENCE,
        weights: WEIGHTS,
    }
}
