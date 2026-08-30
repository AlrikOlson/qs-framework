# qs-framework

The GPU UI framework under [quicksilver-file-explorer](https://github.com/AlrikOlson/quicksilver-file-explorer),
extracted so that quicksilver and [magistr](https://github.com/magistr-dev/magistr) can both build on it.

| crate | what it is |
|---|---|
| `qs-platform` | windowing, input, display enumeration and present tuning behind traits (winit 0.30) |
| `qs-gpu` | device + surface management, draw lists, instanced pipelines, glyph atlas, three render tiers (wgpu / GL / CPU), lighting (wgpu 30) |
| `qs-text` | shaping (rustybuzz), glyph raster (swash), bidi, per-platform font handling |
| `qs-ui` | tokens, motion, density, the accesskit tree, scene, row recycler |

Extracted from quicksilver at `6dd3d8b` (2026-08-30) with `git filter-repo`, so every
file carries its original history. Nothing file-explorer-specific lives here and nothing
here may come to depend on any of it.

**Pin discipline:** `wgpu`, `winit` and `accesskit`/`accesskit_winit` move together — see
the comment in `Cargo.toml`. Consumers pin the same trio.

```
cargo build --workspace --all-targets
cargo nextest run --workspace
```

Apache-2.0 — see `LICENSE` and `NOTICE`.
