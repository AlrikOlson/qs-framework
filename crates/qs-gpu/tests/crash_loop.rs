//! T094 / RP-3, RP-4 — crash-loop demotion, with an injected startup failure.
//!
//! From the quickstart:
//!
//! > Expected: two injected startup crashes demote exactly one tier, and the pinned choice
//! > is overridable by `--force-render-path`.
//!
//! # What "injected startup crash" means here
//!
//! A real crash inside device creation cannot be simulated in-process -- the process is
//! gone, which is the entire difficulty the crash counter exists to handle. What *can* be
//! simulated faithfully is the observable consequence: a run that calls `begin_attempt` and
//! then never calls `mark_success`, because it died in between.
//!
//! That is exactly what [`crashed_startup`] does, and it is a faithful model precisely
//! because the counter is persisted to disk *before* the risky work. If the implementation
//! ever moved the write to after device creation, this test would still pass on the first
//! crash and fail on the demotion -- which is the failure mode being defended against.

// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the
// vocabulary of a test, not a hazard in one. The workspace lints deny them for
// production code, so each test binary opts out at its root.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
use std::path::PathBuf;

use qs_gpu::path::{
    CrashCounter, DEMOTION_THRESHOLD, PathReason, RenderPath, RenderPathSelector, Resolution,
    resolve,
};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "qs-crash-loop-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Model one startup that faulted inside device creation: the attempt is recorded, and the
/// process dies before it can be marked successful.
fn crashed_startup(counter: &CrashCounter, tier: RenderPath) -> (RenderPath, Option<PathReason>) {
    counter.begin_attempt(tier)
    // ... and then nothing. No `mark_success`.
}

/// Model a startup that reached its first rendered frame.
fn successful_startup(counter: &CrashCounter, tier: RenderPath) -> RenderPath {
    let (actual, _) = counter.begin_attempt(tier);
    counter.mark_success();
    actual
}

struct Selector {
    probe: RenderPath,
    forced: Option<RenderPath>,
    pinned: Option<RenderPath>,
}

impl RenderPathSelector for Selector {
    fn probe(&self) -> RenderPath {
        self.probe
    }
    fn forced(&self) -> Option<RenderPath> {
        self.forced
    }
    fn pinned(&self) -> Option<RenderPath> {
        self.pinned
    }
}

#[test]
fn two_consecutive_startup_crashes_demote_exactly_one_tier() {
    // RP-3. The specific thing being ruled out is demoting straight to `Cpu`, which would
    // strand a machine on software rasterization after one bad driver day.
    let dir = temp_dir("one-tier");
    let counter = CrashCounter::in_dir(&dir);

    let (first, _) = crashed_startup(&counter, RenderPath::Primary);
    assert_eq!(
        first,
        RenderPath::Primary,
        "one crash is not enough to demote"
    );

    let (second, _) = crashed_startup(&counter, RenderPath::Primary);
    assert_eq!(
        second,
        RenderPath::Primary,
        "the {DEMOTION_THRESHOLD}nd attempt still runs on the same tier"
    );

    let (third, reason) = crashed_startup(&counter, RenderPath::Primary);
    assert_eq!(third, RenderPath::Reduced, "must demote exactly one tier");
    assert_ne!(
        third,
        RenderPath::Cpu,
        "must not skip straight to the CPU tier"
    );

    match reason {
        Some(PathReason::DemotedNow { from, attempts }) => {
            assert_eq!(from, RenderPath::Primary);
            assert_eq!(attempts, DEMOTION_THRESHOLD);
        }
        other => panic!("the demotion must record why, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn demotion_cascades_one_tier_at_a_time() {
    let dir = temp_dir("cascade");
    let counter = CrashCounter::in_dir(&dir);

    // Crash out of Primary.
    crashed_startup(&counter, RenderPath::Primary);
    crashed_startup(&counter, RenderPath::Primary);
    let (tier, _) = crashed_startup(&counter, RenderPath::Primary);
    assert_eq!(tier, RenderPath::Reduced);

    // Now crash out of Reduced. The demoted tier gets its own full allowance rather than
    // inheriting the previous tier's sentence.
    let (tier, _) = crashed_startup(&counter, RenderPath::Reduced);
    assert_eq!(tier, RenderPath::Reduced, "a fresh count on the new tier");
    let (tier, _) = crashed_startup(&counter, RenderPath::Reduced);
    assert_eq!(tier, RenderPath::Cpu);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_counter_is_readable_from_disk_before_any_device_is_created() {
    // The whole contract: "the counter written after successful initialization cannot
    // observe a crash *during* initialization". If the write ever moved after device
    // creation, this would read `None`.
    let dir = temp_dir("persist-order");
    let counter = CrashCounter::in_dir(&dir);

    counter.begin_attempt(RenderPath::Primary);

    let observed = CrashCounter::in_dir(&dir)
        .read()
        .expect("the attempt must already be on disk");
    assert_eq!(observed.tier, RenderPath::Primary);
    assert_eq!(observed.attempts, 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_pinned_tier_is_overridable_by_explicit_configuration() {
    // RP-4: a user whose driver was fixed must be able to get their GPU back.
    let dir = temp_dir("forced-beats-pinned");
    let counter = CrashCounter::in_dir(&dir);

    crashed_startup(&counter, RenderPath::Primary);
    crashed_startup(&counter, RenderPath::Primary);
    assert_eq!(
        counter.pinned(),
        Some(RenderPath::Reduced),
        "two crashes must produce a pin"
    );

    // Without a force, resolution honours the pin.
    let pinned = resolve(&Selector {
        probe: RenderPath::Primary,
        forced: None,
        pinned: counter.pinned(),
    });
    assert_eq!(pinned.path, RenderPath::Reduced);

    // `--force-render-path primary` beats it.
    let Resolution { path, reason } = resolve(&Selector {
        probe: RenderPath::Primary,
        forced: Some(RenderPath::Primary),
        pinned: counter.pinned(),
    });
    assert_eq!(path, RenderPath::Primary);
    assert_eq!(reason, PathReason::Forced);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_successful_run_clears_the_pin() {
    // A machine that crashed twice and then had its driver fixed must not stay pinned
    // forever once it starts working again.
    let dir = temp_dir("success-clears");
    let counter = CrashCounter::in_dir(&dir);

    crashed_startup(&counter, RenderPath::Primary);
    crashed_startup(&counter, RenderPath::Primary);
    assert!(counter.pinned().is_some());

    let tier = successful_startup(&counter, RenderPath::Reduced);
    assert_eq!(tier, RenderPath::Reduced);
    assert_eq!(counter.pinned(), None, "success must clear the pin");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_intermittent_crash_never_accumulates_into_a_demotion() {
    // Crash, succeed, crash, succeed... A machine that mostly works must never be demoted,
    // or a transient driver hiccup permanently costs the user their GPU.
    let dir = temp_dir("intermittent");
    let counter = CrashCounter::in_dir(&dir);

    for _ in 0..10 {
        let (tier, reason) = crashed_startup(&counter, RenderPath::Primary);
        assert_eq!(tier, RenderPath::Primary);
        assert!(reason.is_none());
        successful_startup(&counter, RenderPath::Primary);
    }
    assert_eq!(counter.pinned(), None);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_cpu_tier_is_the_floor() {
    // There is no fourth tier. A machine that crashes in software rasterization must keep
    // trying and reporting rather than looping on a demotion that cannot happen.
    let dir = temp_dir("floor");
    let counter = CrashCounter::in_dir(&dir);

    for _ in 0..6 {
        let (tier, _) = crashed_startup(&counter, RenderPath::Cpu);
        assert_eq!(tier, RenderPath::Cpu);
    }
    assert_eq!(RenderPath::Cpu.demoted(), None);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_machine_with_no_state_file_starts_on_the_probed_tier() {
    let dir = temp_dir("fresh");
    let counter = CrashCounter::in_dir(&dir);
    assert_eq!(counter.read(), None);
    assert_eq!(counter.pinned(), None);

    let resolution = resolve(&Selector {
        probe: RenderPath::Primary,
        forced: None,
        pinned: counter.pinned(),
    });
    assert_eq!(resolution.path, RenderPath::Primary);
    assert_eq!(resolution.reason, PathReason::Probed);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_reason_is_reportable_to_the_user() {
    // RP-6: the chosen tier and the reason are visible to the user and recorded in every
    // bench report. A reason that cannot be rendered as text is not visible to anyone.
    for reason in [
        PathReason::Forced,
        PathReason::Probed,
        PathReason::NoAdapter,
        PathReason::PinnedAfterCrash {
            crashed: RenderPath::Primary,
        },
        PathReason::DemotedNow {
            from: RenderPath::Primary,
            attempts: 2,
        },
    ] {
        let text = reason.to_string();
        assert!(!text.is_empty(), "{reason:?} renders as an empty string");
        assert!(
            text.chars().next().is_some_and(|c| c.is_lowercase()),
            "reasons are sentence fragments meant to follow the tier name: {text}"
        );
    }
}
