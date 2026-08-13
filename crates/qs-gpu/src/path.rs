//! Rendering tiers: choosing one, surviving a crash in one, and getting un-stuck.
//!
//! Implements [`contracts/render-path.md`]. SDD R1 names GPU driver variance as the top
//! technical risk; everything here is its mitigation.
//!
//! # The ordering is the whole contract
//!
//! > read attempt_counter for the tier about to be initialized
//! > increment and **PERSIST** it  ◄── before the risky work, not after
//!
//! A counter written after successful initialization cannot observe a crash *during*
//! initialization, and a driver that faults inside device creation is precisely the failure
//! being defended against. [`CrashCounter::begin_attempt`] therefore flushes to disk before
//! returning, and the caller must not create a device until it has.
//!
//! # Resolution order: forced → pinned → probe
//!
//! Forced beats pinned so a user whose driver got fixed can have their GPU back. Pinned
//! beats probe so a crash loop terminates instead of probing into the same fault forever.

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Consecutive failed startups at one tier before demoting. Two, not one: a single crash is
/// as likely to be a transient driver hiccup as a real incompatibility, and demoting on it
/// would strand users on the CPU path after one bad day.
pub const DEMOTION_THRESHOLD: u32 = 2;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub enum RenderPath {
    /// D3D12 / Metal / Vulkan. Gated on the frame budget *and* reference images.
    #[default]
    Primary,
    /// GL 4.3 / GLES 3.1. Gated on reference images only.
    Reduced,
    /// `tiny-skia` software raster. Gated on reference images and not crashing
    /// (decision A-3); frame times recorded but unbounded.
    Cpu,
}

impl RenderPath {
    /// The next tier down. `Cpu` has nowhere to go -- there is no fourth tier, and
    /// pretending otherwise would mean a machine that crashes on software rasterization
    /// loops forever instead of reporting.
    pub fn demoted(self) -> Option<Self> {
        match self {
            Self::Primary => Some(Self::Reduced),
            Self::Reduced => Some(Self::Cpu),
            Self::Cpu => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Reduced => "reduced",
            Self::Cpu => "cpu",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "primary" => Some(Self::Primary),
            "reduced" => Some(Self::Reduced),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    /// Whether results from this tier may gate a benchmark run (decision A-3).
    pub fn gates_frame_budget(self) -> bool {
        matches!(self, Self::Primary)
    }
}

impl fmt::Display for RenderPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why the running tier is the running tier.
///
/// RP-6 requires this to be visible to the user and recorded in every bench report.
/// Constitution III: reduced capability is never silent, and "silent" includes "shown
/// without saying why", because a user who cannot tell a deliberate choice from a
/// fallback cannot act on either.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PathReason {
    /// `--force-render-path`, or configuration.
    Forced,
    /// Probing found this to be the best usable tier.
    Probed,
    /// A previous tier crashed twice on startup and this one was pinned.
    PinnedAfterCrash { crashed: RenderPath },
    /// Demoted during *this* startup because the attempt counter was already at the
    /// threshold.
    DemotedNow { from: RenderPath, attempts: u32 },
    /// No usable adapter at all.
    NoAdapter,
}

impl fmt::Display for PathReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forced => write!(f, "forced by configuration"),
            Self::Probed => write!(f, "selected by capability probe"),
            Self::PinnedAfterCrash { crashed } => {
                write!(f, "pinned after {crashed} failed to start twice")
            }
            Self::DemotedNow { from, attempts } => {
                write!(f, "demoted from {from} after {attempts} failed starts")
            }
            Self::NoAdapter => write!(f, "no usable graphics adapter was found"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Resolution {
    pub path: RenderPath,
    pub reason: PathReason,
}

/// Contract interface. `probe` **must not panic** on a broken driver -- a probe failure is
/// a return value, because a panic here is a startup crash on exactly the machines this
/// mechanism exists to serve.
pub trait RenderPathSelector {
    fn probe(&self) -> RenderPath;
    fn forced(&self) -> Option<RenderPath>;
    fn pinned(&self) -> Option<RenderPath>;
}

/// Apply the contract's resolution order.
pub fn resolve(selector: &dyn RenderPathSelector) -> Resolution {
    if let Some(path) = selector.forced() {
        return Resolution {
            path,
            reason: PathReason::Forced,
        };
    }
    if let Some(path) = selector.pinned() {
        // The pinned tier is the one that *works*; the crashed one is the tier above it.
        let crashed = match path {
            RenderPath::Primary => RenderPath::Primary,
            RenderPath::Reduced => RenderPath::Primary,
            RenderPath::Cpu => RenderPath::Reduced,
        };
        return Resolution {
            path,
            reason: PathReason::PinnedAfterCrash { crashed },
        };
    }
    Resolution {
        path: selector.probe(),
        reason: PathReason::Probed,
    }
}

/// The conventional per-user state directory for this platform.
///
/// One function rather than one per thing that persists. SDD §13 gives this directory a
/// SQLite store eventually; until then the two small files that live here (the crash counter
/// and the window's split) at least agree about *where* here is — two functions deriving the
/// platform directory separately is the drift that puts a user's state in two places, and
/// only one of them gets migrated.
#[must_use]
pub fn state_dir() -> PathBuf {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
    };
    base.unwrap_or_else(std::env::temp_dir).join("quicksilver")
}

// -- crash counter -------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CrashState {
    pub tier: RenderPath,
    pub attempts: u32,
}

/// Persisted startup-attempt counter.
///
/// Deliberately a hand-rolled two-line file rather than JSON. It is written on the startup
/// path before any device exists, it must survive a process that is about to be killed by a
/// driver fault, and a partially-written file must parse as "no information" rather than as
/// a corrupt-state error that itself blocks startup. Fewer moving parts is the feature.
#[derive(Clone, Debug)]
pub struct CrashCounter {
    file: PathBuf,
}

impl CrashCounter {
    /// Store the counter in `dir`. The directory is created if needed; a failure to create
    /// it is not fatal -- the counter degrades to "always zero", which means no pinning,
    /// which is the same behaviour as a first-ever run.
    pub fn in_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let _ = fs::create_dir_all(dir);
        Self {
            file: dir.join("render-path.state"),
        }
    }

    /// The conventional per-user state location for this platform.
    pub fn default_location() -> Self {
        Self::in_dir(state_dir())
    }

    pub fn read(&self) -> Option<CrashState> {
        let text = fs::read_to_string(&self.file).ok()?;
        let mut tier = None;
        let mut attempts = None;
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            match key.trim() {
                "tier" => tier = RenderPath::parse(value),
                "attempts" => attempts = value.trim().parse::<u32>().ok(),
                _ => {}
            }
        }
        // A half-written file yields `None` and the caller proceeds as though this were a
        // first run. That is the safe direction: it can cost one extra crash, never a
        // machine wedged on the CPU path by a torn write.
        Some(CrashState {
            tier: tier?,
            attempts: attempts?,
        })
    }

    fn write(&self, state: CrashState) -> std::io::Result<()> {
        // Write-and-sync, not write-and-hope. The whole mechanism assumes the bytes are on
        // disk before the process faults, and without the flush they may not be.
        let mut file = fs::File::create(&self.file)?;
        writeln!(file, "tier={}", state.tier)?;
        writeln!(file, "attempts={}", state.attempts)?;
        file.flush()?;
        file.sync_all()
    }

    /// The tier pinned by previous crashes, if any.
    pub fn pinned(&self) -> Option<RenderPath> {
        let state = self.read()?;
        (state.attempts >= DEMOTION_THRESHOLD)
            .then(|| state.tier.demoted())
            .flatten()
    }

    /// Record that initialization of `tier` is about to be attempted, and return the tier
    /// that should actually be attempted.
    ///
    /// **Persists before returning.** The caller must not touch a graphics API until this
    /// has returned -- that ordering is the contract.
    pub fn begin_attempt(&self, tier: RenderPath) -> (RenderPath, Option<PathReason>) {
        let previous = self.read().unwrap_or(CrashState { tier, attempts: 0 });

        // Attempts only accumulate against the same tier. Switching tiers -- because the
        // user forced one, or because the probe changed after a driver update -- starts a
        // fresh count, or a machine would carry a grudge from an unrelated configuration.
        let attempts = if previous.tier == tier {
            previous.attempts
        } else {
            0
        };

        if attempts >= DEMOTION_THRESHOLD {
            // Exactly one tier down (RP-3), counter reset so the new tier gets its own
            // two chances rather than inheriting a sentence.
            if let Some(demoted) = tier.demoted() {
                let _ = self.write(CrashState {
                    tier: demoted,
                    attempts: 1,
                });
                return (
                    demoted,
                    Some(PathReason::DemotedNow {
                        from: tier,
                        attempts,
                    }),
                );
            }
            // Already on `Cpu`. There is nowhere to demote to; try again and report.
            let _ = self.write(CrashState { tier, attempts: 1 });
            return (tier, None);
        }

        let _ = self.write(CrashState {
            tier,
            attempts: attempts + 1,
        });
        (tier, None)
    }

    /// Called once the first frame has actually rendered.
    ///
    /// Not "once the device was created" -- a driver that creates a device and then faults
    /// on the first submit would otherwise never be detected.
    pub fn mark_success(&self) {
        if let Some(state) = self.read() {
            let _ = self.write(CrashState {
                tier: state.tier,
                attempts: 0,
            });
        }
    }

    /// Forget everything. Exposed so `--force-render-path` can clear a pin the user has
    /// decided is stale (RP-4).
    pub fn reset(&self) {
        let _ = fs::remove_file(&self.file);
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use super::*;

    struct Fixed {
        probe: RenderPath,
        forced: Option<RenderPath>,
        pinned: Option<RenderPath>,
    }

    impl RenderPathSelector for Fixed {
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

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("qs-path-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn forced_beats_pinned_and_probe() {
        // RP-4: a user whose driver was fixed must be able to get their GPU back.
        let r = resolve(&Fixed {
            probe: RenderPath::Reduced,
            forced: Some(RenderPath::Primary),
            pinned: Some(RenderPath::Cpu),
        });
        assert_eq!(r.path, RenderPath::Primary);
        assert_eq!(r.reason, PathReason::Forced);
    }

    #[test]
    fn pinned_beats_probe_so_a_crash_loop_terminates() {
        let r = resolve(&Fixed {
            probe: RenderPath::Primary,
            forced: None,
            pinned: Some(RenderPath::Reduced),
        });
        assert_eq!(r.path, RenderPath::Reduced);
        assert!(matches!(r.reason, PathReason::PinnedAfterCrash { .. }));
    }

    #[test]
    fn demotion_goes_exactly_one_tier() {
        // RP-3. Two crashes on Primary land on Reduced, never straight to Cpu.
        let dir = temp_dir("one-tier");
        let counter = CrashCounter::in_dir(&dir);

        let (t1, _) = counter.begin_attempt(RenderPath::Primary);
        assert_eq!(t1, RenderPath::Primary);
        // ... process dies here, no mark_success ...
        let (t2, _) = counter.begin_attempt(RenderPath::Primary);
        assert_eq!(t2, RenderPath::Primary, "one crash is not enough to demote");
        // ... dies again ...
        let (t3, reason) = counter.begin_attempt(RenderPath::Primary);
        assert_eq!(t3, RenderPath::Reduced);
        assert!(matches!(reason, Some(PathReason::DemotedNow { .. })));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_counter_is_persisted_before_the_risky_work() {
        // The contract's whole point: after begin_attempt returns, and before anything
        // touches a driver, the incremented count must already be readable from disk.
        let dir = temp_dir("persist-first");
        let counter = CrashCounter::in_dir(&dir);

        counter.begin_attempt(RenderPath::Primary);

        let reread = CrashCounter::in_dir(&dir).read().expect("state must exist");
        assert_eq!(reread.tier, RenderPath::Primary);
        assert_eq!(reread.attempts, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_successful_start_clears_the_count() {
        let dir = temp_dir("success-clears");
        let counter = CrashCounter::in_dir(&dir);

        counter.begin_attempt(RenderPath::Primary);
        counter.mark_success();
        counter.begin_attempt(RenderPath::Primary);
        counter.mark_success();

        // Two successful runs must leave no pin behind.
        assert_eq!(counter.pinned(), None);
        let (tier, _) = counter.begin_attempt(RenderPath::Primary);
        assert_eq!(tier, RenderPath::Primary);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn switching_tiers_starts_a_fresh_count() {
        let dir = temp_dir("fresh-count");
        let counter = CrashCounter::in_dir(&dir);

        counter.begin_attempt(RenderPath::Primary);
        counter.begin_attempt(RenderPath::Primary);
        // The user forces Cpu; the Primary grudge must not carry over.
        let (tier, reason) = counter.begin_attempt(RenderPath::Cpu);
        assert_eq!(tier, RenderPath::Cpu);
        assert!(reason.is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cpu_tier_never_demotes_below_itself() {
        let dir = temp_dir("cpu-floor");
        let counter = CrashCounter::in_dir(&dir);
        for _ in 0..5 {
            let (tier, _) = counter.begin_attempt(RenderPath::Cpu);
            assert_eq!(tier, RenderPath::Cpu);
        }
        assert_eq!(RenderPath::Cpu.demoted(), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_state_file_reads_as_no_information() {
        let dir = temp_dir("corrupt");
        let counter = CrashCounter::in_dir(&dir);
        fs::write(dir.join("render-path.state"), "tier=nonsense\nattempts=x\n").unwrap();

        assert_eq!(counter.read(), None);
        assert_eq!(
            counter.pinned(),
            None,
            "a torn write must not pin a machine"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_the_primary_tier_gates_the_frame_budget() {
        assert!(RenderPath::Primary.gates_frame_budget());
        assert!(!RenderPath::Reduced.gates_frame_budget());
        assert!(!RenderPath::Cpu.gates_frame_budget());
    }

    #[test]
    fn tier_names_round_trip() {
        for tier in [RenderPath::Primary, RenderPath::Reduced, RenderPath::Cpu] {
            assert_eq!(RenderPath::parse(tier.as_str()), Some(tier));
        }
        assert_eq!(RenderPath::parse("PRIMARY"), Some(RenderPath::Primary));
        assert_eq!(RenderPath::parse("gpu"), None);
    }
}
