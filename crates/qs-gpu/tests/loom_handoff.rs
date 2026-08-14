//! Model-checks the UI→Render draw-list handoff under `loom`.
//!
//! Run with:
//!
//! ```text
//! RUSTFLAGS="--cfg loom" cargo test -p qs-gpu --test loom_handoff --release
//! ```
//!
//! # Why this test exists
//!
//! `DrawListChannel` carries an `unsafe impl Sync` and two raw slot accesses justified by a
//! prose argument about which side owns which index. Prose arguments about lock-free code
//! are wrong at a rate that does not depend on how carefully they are written. `loom`
//! enumerates the interleavings and the memory orderings that a real weak-memory machine
//! could produce, so the two claims that matter get *checked*:
//!
//! 1. **No aliasing.** The producer and the consumer never hold the same slot index. loom's
//!    `UnsafeCell` tracks concurrent access and fails the model if they overlap -- which is
//!    the failure a stress test on x86 would essentially never reproduce, because x86's
//!    strong ordering hides it.
//! 2. **Visibility.** A `DrawList` fully written before `publish` is fully visible after
//!    `acquire`. This is what the `AcqRel` pair buys, and dropping either to `Relaxed`
//!    should make this test fail. It is worth trying that once, by hand, to confirm the
//!    test can actually detect the bug it is here for.
//!
//! Without `--cfg loom` the file compiles to nothing, so it costs a normal `cargo test`
//! nothing at all.

#![cfg(loom)]
// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the
// vocabulary of a test, not a hazard in one. The workspace lints deny them for
// production code, so each test binary opts out at its root.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use qs_gpu::color::Srgba;
use qs_gpu::frame::draw_list_channel;
use qs_gpu::scene::{Environment, SceneList, Slab};

/// Frames published by the producer. Kept small deliberately: loom explores interleavings
/// combinatorially, and a third publish takes this model from seconds to minutes without
/// covering a materially different shape.
const FRAMES: u64 = 2;

#[test]
fn the_handoff_never_aliases_a_slot() {
    loom::model(|| {
        let (mut producer, mut consumer) = draw_list_channel();

        let writer = loom::thread::spawn(move || {
            for generation in 1..=FRAMES {
                let slot = producer.slot();
                slot.reset([1920, 1080], Srgba::TRANSPARENT, generation);
                // Write something derived from the generation, so a torn or stale read is
                // detectable as an inconsistency rather than only as a wrong number.
                slot.stats.rows_laid_out = generation as u32;
                slot.stats.instances = generation as u32 * 10;
                // The scene rides in the same slot (scene-handoff rule 6), so the same
                // publish must make it visible with the same guarantee. Alternate lit and
                // unlit frames so the model also covers a recycled slot going back to None.
                *producer.scene_slot() = (generation % 2 == 1).then(|| {
                    let mut scene = SceneList::default();
                    scene.reset(generation, Environment::default());
                    for _ in 0..generation {
                        scene.push(Slab::default());
                    }
                    scene
                });
                producer.publish();
            }
        });

        let mut last = 0;
        for _ in 0..FRAMES {
            if let Some(list) = consumer.acquire() {
                // Generations must never go backwards: a triple buffer may skip a frame,
                // never replay one.
                assert!(
                    list.generation > last,
                    "generation went backwards: {} after {last}",
                    list.generation
                );
                last = list.generation;

                // Visibility. If the release/acquire pair were weakened, these fields could
                // be observed as their pre-`reset` values while `generation` was already
                // the new one.
                assert_eq!(
                    list.stats.rows_laid_out, list.generation as u32,
                    "a partially visible draw list was acquired"
                );
                assert_eq!(list.stats.instances, list.generation as u32 * 10);
                assert_eq!(list.viewport, [1920, 1080]);

                let generation = list.generation;
                match consumer.scene() {
                    // A lit frame's scene must be the one written before the same publish:
                    // same generation, and contents derived from it fully visible.
                    Some(scene) => {
                        assert_eq!(generation % 2, 1, "an unlit frame handed out a scene");
                        assert_eq!(scene.generation, generation);
                        assert_eq!(
                            scene.slabs.len(),
                            generation as usize,
                            "a partially visible scene was acquired"
                        );
                    }
                    // An unlit frame reads as no scene -- including when a previous lit
                    // frame used this slot, which is the recycled-slot hazard.
                    None => assert_eq!(generation % 2, 0, "a lit frame lost its scene"),
                }
            }
        }

        writer.join().expect("producer thread panicked");
    });
}

#[test]
fn a_consumer_that_never_polls_does_not_block_the_producer() {
    // The property that makes this a triple buffer rather than a queue: the producer
    // completes every publish regardless of what the consumer does. If `publish` could ever
    // wait, a slow GPU frame would stall input handling -- Constitution I.
    loom::model(|| {
        let (mut producer, _consumer) = draw_list_channel();
        for generation in 1..=FRAMES {
            producer
                .slot()
                .reset([1, 1], Srgba::TRANSPARENT, generation);
            producer.publish();
        }
    });
}
