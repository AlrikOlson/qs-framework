//! T051 / RS-1 — `rows()` is never entered from a frame thread in a blocking way.
//!
//! From the `RowSource` contract's test obligations:
//!
//! > A `SlowSource` that would block if called synchronously is used to assert RS-1 -- the
//! > test fails if the frame thread ever enters it.
//!
//! # How the assertion actually works
//!
//! `qs_gpu::affinity` records which thread is the UI thread and which is the Render thread.
//! `SlowSource::rows` calls `assert_may_block`, which panics in debug builds if it is
//! running on either. So the test does not measure elapsed time and guess -- it observes
//! the *thread*, which is the property the constitution actually states.
//!
//! The registration is process-wide and one-shot (`OnceLock`), which is why the positive
//! and negative cases are driven from separate spawned threads within one test rather than
//! from separate `#[test]` functions: cargo runs tests in one process, and whichever test
//! ran first would otherwise decide what "the UI thread" means for all of them.

// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the
// vocabulary of a test, not a hazard in one. The workspace lints deny them for
// production code, so each test binary opts out at its root.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
use std::sync::Arc;
use std::time::Duration;

use qs_bench::scenario::named_corpus;
use qs_bench::synthetic_source::SlowSource;
use qs_ui::row_source::{RowBuf, RowSource};

fn corpus() -> Arc<qs_bench::Corpus> {
    let mut manifest = named_corpus("flat-1m").expect("flat-1m must exist");
    manifest.row_count = 2_000;
    Arc::new(qs_bench::generate(&manifest).expect("generation must succeed"))
}

#[test]
fn a_blocking_source_is_rejected_on_the_ui_thread_and_allowed_off_it() {
    let source = Arc::new(SlowSource::new(corpus(), Duration::from_millis(5)));

    // -- the negative case: a worker thread may block --------------------------------
    // This is where a real source's I/O belongs, and it must be allowed.
    let worker = {
        let source = Arc::clone(&source);
        std::thread::spawn(move || {
            let mut buf = RowBuf::new();
            source.rows(0..40, &mut buf);
            buf.len()
        })
    };
    let filled = worker
        .join()
        .expect("a worker thread must be allowed to block");
    assert_eq!(filled, 40, "the source must actually have been entered");
    assert_eq!(source.entry_count(), 1);

    // -- the positive case: the UI thread may not -------------------------------------
    let ui = {
        let source = Arc::clone(&source);
        std::thread::spawn(move || {
            qs_gpu::affinity::register_ui_thread();
            assert!(qs_gpu::affinity::is_ui_thread());

            let mut buf = RowBuf::new();
            // In a debug build this panics; the panic is the assertion.
            source.rows(40..80, &mut buf);
        })
    };
    let outcome = ui.join();

    if cfg!(debug_assertions) {
        assert!(
            outcome.is_err(),
            "RS-1 violated: a blocking source ran on the UI thread without tripping the \
             affinity guard"
        );
    } else {
        // The guards are `debug_assert!` and compile out in release, so the frame path pays
        // nothing for them. The test correspondingly only asserts the guard in debug.
        assert!(outcome.is_ok());
    }

    // Either way, the source was entered a second time -- so the test is observing a real
    // call and not a call that never happened.
    assert_eq!(source.entry_count(), 2);
}

#[test]
fn a_non_blocking_source_is_fine_anywhere() {
    // The synthetic source does no I/O and takes no locks, so it is legal on any thread.
    // If this ever starts failing, something has grown a blocking call on the hot path.
    let source = qs_bench::synthetic_source::SyntheticSource::new(corpus());
    let mut buf = RowBuf::new();
    source.rows(0..40, &mut buf);
    assert_eq!(buf.len(), 40);
}

#[test]
fn the_affinity_guard_is_inert_when_no_thread_is_registered() {
    // Tests, the bench harness and any tool that is not the application run without a
    // registered UI thread. The guards must not fire there, or every one of them would
    // need a setup ritual and people would stop calling them.
    qs_gpu::affinity::assert_may_block("test harness");
    qs_gpu::affinity::assert_ui_thread("test harness");
    qs_gpu::affinity::assert_render_thread("test harness");
}
