# qs-framework

Rust libraries for drawing desktop interfaces. They handle windows, GPU and CPU
rendering, text, and virtualized lists. They are used by
[Quicksilver](https://github.com/AlrikOlson/quicksilver-file-explorer) and
[Magistr](https://github.com/AlrikOlson/magistr).

| Crate | Contents |
| --- | --- |
| `qs-platform` | Windows, input, monitor information, and presentation settings using winit |
| `qs-gpu` | GPU setup, draw lists, glyph and image atlases, effects, and a CPU renderer |
| `qs-text` | Text shaping, font lookup, bidirectional text, and glyph rasterization |
| `qs-ui` | List layout, scrolling, selection, materials, animation, and accessibility |

## Build

The crates declare Rust 1.87 as their minimum version. With rustup installed,
`rust-toolchain.toml` selects Rust 1.87.0.

Build the rendering, platform, and text libraries with:

```sh
cargo build --locked -p qs-gpu -p qs-platform -p qs-text
```

`qs-ui` currently fails to compile because its material renderer is missing a
`DashedStroke` match arm.

Generate the API documentation with:

```sh
cargo doc --workspace --no-deps --locked --open
```

## Use in a project

The crates are available as Git dependencies:

```toml
[dependencies]
qs-gpu = { git = "https://github.com/AlrikOlson/qs-framework" }
qs-platform = { git = "https://github.com/AlrikOlson/qs-framework" }
qs-text = { git = "https://github.com/AlrikOlson/qs-framework" }
```

Use the same Git revision for each `qs-*` crate. If your application also depends
on wgpu, winit, or AccessKit, use the versions in [Cargo.toml](Cargo.toml) to
avoid incompatible types between the application and these libraries.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
