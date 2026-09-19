//! Measure lighting ranges that preserve text contrast in each theme.
//!
//! The example finds shadow and added-light limits separately for light and
//! dark themes. It also measures the dark theme with added light excluded
//! from text backgrounds.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use qs_ui::tokens::{
    ContrastResult, LitBounds, LitPolicy, Theme, TokenFile, Tokens, material_results_lit,
    material_results_policy,
};

/// Failures across both themes — the number R12 reported.
fn failures(file: &TokenFile, lit: LitBounds) -> usize {
    [Theme::Light, Theme::Dark]
        .iter()
        .map(|&t| failures_in(file, t, lit))
        .sum()
}

/// Failures in one theme. The split the whole probe turns on.
fn failures_in(file: &TokenFile, theme: Theme, lit: LitBounds) -> usize {
    material_results_lit(file, theme, lit)
        .expect("results")
        .iter()
        .filter(|r| !r.passes())
        .count()
}

fn worst(file: &TokenFile, theme: Theme, lit: LitBounds) -> Option<ContrastResult> {
    material_results_lit(file, theme, lit)
        .expect("results")
        .into_iter()
        .filter(|r| !r.passes())
        .min_by(|a, b| a.ratio.total_cmp(&b.ratio))
}

/// Bisect `f` over `0..=1` for the tightest value that still passes.
///
/// `rising` says which way "safer" lies: an attenuation floor is safer as it approaches 1 (no
/// shadow), an addition is safer as it approaches 0 (no bounce).
fn bisect(hi_bound: f32, rising: bool, mut ok: impl FnMut(f32) -> bool) -> f32 {
    let (mut lo, mut hi) = (0.0f32, hi_bound);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        // `rising`: larger is safer, so a passing mid means we can try smaller.
        if ok(mid) == rising {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if rising { hi } else { lo }
}

fn shadow_floor(file: &TokenFile, theme: Option<Theme>) -> f32 {
    bisect(1.0, true, |mid| {
        let lit = LitBounds {
            attenuation: (mid, 1.0),
            addition: (0.0, 0.0),
        };
        match theme {
            Some(t) => failures_in(file, t, lit) == 0,
            None => failures(file, lit) == 0,
        }
    })
}

fn addition_ceiling(file: &TokenFile, theme: Option<Theme>) -> f32 {
    bisect(0.5, false, |mid| {
        let lit = LitBounds {
            attenuation: (1.0, 1.0),
            addition: (0.0, mid),
        };
        match theme {
            Some(t) => failures_in(file, t, lit) == 0,
            None => failures(file, lit) == 0,
        }
    })
}

fn main() {
    let file: TokenFile =
        serde_json::from_str(include_str!("../../../design/tokens.json")).expect("tokens.json");

    println!(
        "unlit baseline failures: {}\n",
        failures(&file, LitBounds::UNLIT)
    );

    // ---- R12, reproduced. One range, both themes, both directions at once. ----
    println!("== one global range, both themes (R12) ==");
    println!(
        "  provisional bounds (0.35 / 0.12): {} failures",
        failures(&file, LitBounds::PROVISIONAL)
    );
    println!(
        "  deepest affordable shadow:    attenuation floor {:.3}  (1.000 = no shadow at all)",
        shadow_floor(&file, None)
    );
    println!(
        "  brightest affordable addition: {:.4} linear",
        addition_ceiling(&file, None)
    );
    for floor in [0.98f32, 0.95, 0.90, 0.85, 0.75] {
        let add = (1.0 - floor) * 0.25;
        let lit = LitBounds {
            attenuation: (floor, 1.0),
            addition: (0.0, add),
        };
        println!(
            "    floor {floor:.2} + addition {add:.4} -> {} failures",
            failures(&file, lit)
        );
    }

    // ---- The same four questions, asked separately. ----
    //
    // A shadow darkens a ground. That helps light-on-dark text and hurts dark-on-light text, so the
    // two themes do not constrain the same direction and a single range pays for both.
    println!("\n== the same questions, per theme and per direction ==");
    for theme in [Theme::Light, Theme::Dark] {
        println!(
            "  {theme:?}: shadow floor {:.3}   addition ceiling {:.4} linear",
            shadow_floor(&file, Some(theme)),
            addition_ceiling(&file, Some(theme))
        );
    }

    // ---- The source/receiver split. ----
    //
    // A meaning-bearing element emits and never receives, and neither does the ground directly
    // behind it -- so a text-bearing surface's `addition` is zero by construction. That leaves only
    // shadow on it, which is the direction the dark theme is *helped* by. This is the number that
    // decides whether the split buys anything real or only re-describes route 2.
    println!("\n== with addition held off text-bearing grounds (addition = 0) ==");
    for theme in [Theme::Light, Theme::Dark] {
        let floor = shadow_floor(&file, Some(theme));
        let lit = LitBounds {
            attenuation: (floor, 1.0),
            addition: (0.0, 0.0),
        };
        let n = failures_in(&file, theme, lit);
        println!("  {theme:?}: shadow may reach {floor:.3} with {n} failures");
        // And the honest question: what does a *full* shadow cost this theme?
        let full = LitBounds {
            attenuation: (0.0, 1.0),
            addition: (0.0, 0.0),
        };
        let n_full = failures_in(&file, theme, full);
        match worst(&file, theme, full) {
            None => println!("           full shadow (0.000) costs nothing in this theme"),
            Some(w) => println!(
                "           full shadow (0.000) -> {n_full} failures, worst {} on {} at {:.2}:1 (needs {:.1}:1)",
                w.foreground, w.background, w.ratio, w.required
            ),
        }
    }

    // ---- And the mirror: attenuation held off text-bearing grounds. ----
    println!("\n== with shadow held off text-bearing grounds (attenuation = 1) ==");
    for theme in [Theme::Light, Theme::Dark] {
        let ceiling = addition_ceiling(&file, Some(theme));
        println!("  {theme:?}: addition may reach {ceiling:.4} linear");
    }

    // ---- What the token file actually claims, checked against what it can afford. ----
    //
    // The bisections above answer "what could this palette carry". This answers "what has been
    // authored", which is the question that decides whether the build is green -- and the gap
    // between the two is the headroom a future palette edit gets to spend.
    println!("\n== the shipped allowance (design/tokens.json `lighting`) ==");
    for theme in [Theme::Light, Theme::Dark] {
        let tokens = Tokens::embedded(theme).expect("embedded tokens");
        let lighting = tokens.lighting();
        let ground = lighting.allowance.text_ground.for_theme(theme);
        let receiver = lighting.allowance.receiver.for_theme(theme);
        let n = material_results_policy(&file, theme, LitPolicy::PerMaterial)
            .expect("results")
            .iter()
            .filter(|r| !r.passes())
            .count();
        println!(
            "  {theme:?}: text ground attenuation >= {:.3}, addition <= {:.4}  ->  {n} failures",
            ground.attenuation_min, ground.addition_max
        );
        println!(
            "          receivers  attenuation >= {:.3}, addition <= {:.4}  (no text, not gated)",
            receiver.attenuation_min, receiver.addition_max
        );
        let affordable = shadow_floor(&file, Some(theme));
        println!(
            "          headroom left on the shadow: authored {:.3} vs affordable {affordable:.3}",
            ground.attenuation_min
        );
    }
}
