//! GPU timestamp queries, with graceful absence.
//!
//! Research R3 splits the measurement in two and forbids conflating them. This module owns
//! the half that is not portable: GPU execution time. When the backend has no timestamp
//! query support -- which is common on GL and on some older D3D12 drivers -- every method
//! here becomes a no-op and [`GpuTimer::take`] returns `None`.
//!
//! `None` propagating all the way into `FrameSample::gpu_ms` and then into the bench report
//! as a null is the entire point. The alternative -- substituting the CPU frame span and
//! labelling it `gpu_ms` -- would produce a report that compares cleanly against a machine
//! where the number means something else, which is worse than having no number at all.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Two timestamps per frame: pass start and pass end.
const QUERY_COUNT: u32 = 2;
const RESULT_BYTES: u64 = QUERY_COUNT as u64 * 8;

/// Frames of latency before a timestamp result is read back.
///
/// The GPU has not finished the frame when it is submitted, so reading immediately would
/// mean blocking on the GPU -- which is the one thing the render thread must never do. Two
/// frames of slack is enough for any realistic pipeline depth, and it means the reported
/// GPU time is two frames stale. For a p99 over thousands of frames that is irrelevant; for
/// a live overlay it means the number lags slightly, which is worth the alternative.
const READBACK_LATENCY: usize = 2;

pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback: Vec<Readback>,
    cursor: usize,
    period_ns: f32,
    latest: Option<f32>,
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU64>,
}

/// `state` values for a readback buffer.
const IDLE: u64 = 0;
const IN_FLIGHT: u64 = 1;
const READY: u64 = 2;

impl std::fmt::Debug for GpuTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTimer")
            .field("period_ns", &self.period_ns)
            .field("latest_ms", &self.latest)
            .finish()
    }
}

impl GpuTimer {
    /// `None` when the device has no timestamp query support -- which is a supported
    /// configuration, not an error.
    pub fn new(ctx: &crate::device::GpuContext) -> Option<Self> {
        if !ctx.capabilities.timestamps {
            tracing::info!(
                target: "qs::gpu",
                backend = ?ctx.capabilities.backend,
                "no timestamp query support; gpu_ms will be reported as null"
            );
            return None;
        }

        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("qs-frame-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });

        let resolve_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qs-timestamp-resolve"),
            size: RESULT_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback = (0..=READBACK_LATENCY)
            .map(|_| Readback {
                buffer: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("qs-timestamp-readback"),
                    size: RESULT_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU64::new(IDLE)),
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(readback.len(), READBACK_LATENCY + 1);

        Some(Self {
            query_set,
            resolve_buffer,
            readback,
            cursor: 0,
            period_ns: ctx.queue.get_timestamp_period(),
            latest: None,
        })
    }

    /// The timestamp writes to attach to this frame's render pass.
    pub fn begin(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        })
    }

    /// Resolve this frame's queries into a readback buffer. Call after the render pass has
    /// been encoded and before the encoder is finished.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let slot = self.cursor % self.readback.len();
        let Some(readback) = self.readback.get(slot) else {
            return;
        };
        if readback.state.load(Ordering::Acquire) != IDLE {
            // Still mapped or in flight. Skipping a frame's timing is fine; waiting is not.
            return;
        }

        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &readback.buffer, 0, RESULT_BYTES);
        readback.state.store(IN_FLIGHT, Ordering::Release);
    }

    /// Kick off the asynchronous map for a frame submitted earlier, and collect any result
    /// that has arrived. Call once per frame, after submission.
    ///
    /// Never blocks: everything here is either an atomic load or a callback registration.
    pub fn poll(&mut self) {
        let len = self.readback.len();
        for slot in 0..len {
            let Some(readback) = self.readback.get(slot) else {
                continue;
            };
            match readback.state.load(Ordering::Acquire) {
                IN_FLIGHT => {
                    let state = Arc::clone(&readback.state);
                    readback
                        .buffer
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            state.store(
                                if result.is_ok() { READY } else { IDLE },
                                Ordering::Release,
                            );
                        });
                    // Mark as pending-map so the next poll does not register a second
                    // callback for the same buffer.
                    readback.state.store(IN_FLIGHT + 10, Ordering::Release);
                }
                READY => {
                    let elapsed = match readback.buffer.slice(..).get_mapped_range() {
                        Ok(view) => read_span(&view, self.period_ns),
                        // The map succeeded but the range is unreadable. One lost sample is
                        // not worth a branch anywhere else in the frame path.
                        Err(_) => None,
                    };
                    readback.buffer.unmap();
                    readback.state.store(IDLE, Ordering::Release);
                    if let Some(ms) = elapsed {
                        self.latest = Some(ms);
                    }
                }
                _ => {}
            }
        }
        self.cursor = self.cursor.wrapping_add(1);
    }

    /// The most recent GPU frame time in milliseconds, if one has been read back.
    pub fn take(&mut self) -> Option<f32> {
        self.latest.take()
    }

    /// Wait for the frame just submitted and return **its** GPU span.
    ///
    /// # This blocks, and only one caller may
    ///
    /// [`poll`](Self::poll) exists because the render thread must never wait on the GPU, and
    /// the price it pays is that its answer is two frames stale and missing on any frame whose
    /// buffer was still mapped. For a live overlay that is the right trade. For the measurement
    /// harness it is the wrong one twice over: a two-frame-stale span attributed to this frame
    /// is a number about a different frame, and a frame with no span at all becomes a
    /// `total_ms` with no GPU half in it -- so a run would report a distribution that mixes
    /// whole frames and half ones under a single name.
    ///
    /// The harness has nothing to do while the GPU works, so it can afford to wait, and
    /// waiting is what buys an exact per-frame span. **Calling this from the application's
    /// frame path would reintroduce the stall the whole ring exists to avoid.**
    ///
    /// Two rounds because the readback needs both: the first `poll` cannot register the map
    /// until the copy has landed, and the callback that marks the buffer readable cannot run
    /// until the map itself completes.
    pub fn read_blocking(&mut self, device: &wgpu::Device) -> Option<f32> {
        for _ in 0..2 {
            // A poll that errors means the device is gone; the caller sees `None` and the
            // frame is recorded without a GPU half, which is the honest outcome.
            if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                return None;
            }
            self.poll();
        }
        self.take()
    }

    /// The most recent value without consuming it -- for the live overlay.
    pub fn peek(&self) -> Option<f32> {
        self.latest
    }
}

fn read_span(bytes: &[u8], period_ns: f32) -> Option<f32> {
    let start = u64::from_le_bytes(bytes.get(0..8)?.try_into().ok()?);
    let end = u64::from_le_bytes(bytes.get(8..16)?.try_into().ok()?);
    // A wrapped or unwritten counter reads as a nonsense span. Discarding it is right:
    // one missing sample costs nothing, and a bogus 4-second frame in the distribution
    // moves the max and destroys the report's credibility.
    if end <= start {
        return None;
    }
    Some((end - start) as f32 * period_ns / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn a_span_converts_ticks_to_milliseconds() {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&1000u64.to_le_bytes());
        bytes[8..16].copy_from_slice(&2000u64.to_le_bytes());
        // 1000 ticks at 1000 ns per tick = 1 ms.
        let ms = read_span(&bytes, 1000.0).unwrap();
        assert!((ms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_backwards_counter_is_discarded_rather_than_reported() {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&5000u64.to_le_bytes());
        bytes[8..16].copy_from_slice(&1000u64.to_le_bytes());
        assert_eq!(read_span(&bytes, 1.0), None);

        // Equal timestamps mean the query never landed, not a zero-cost frame.
        let zero = [0u8; 16];
        assert_eq!(read_span(&zero, 1.0), None);
    }

    #[test]
    fn a_truncated_buffer_does_not_panic() {
        assert_eq!(read_span(&[0u8; 4], 1.0), None);
        assert_eq!(read_span(&[], 1.0), None);
    }
}
