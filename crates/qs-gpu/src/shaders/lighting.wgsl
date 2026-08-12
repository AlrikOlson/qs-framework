// The lighting pass. Reads the scene, writes attenuation and added light; the surface image is
// modulated by the result, and text is drawn afterwards and never lit.
//
// # No derivatives, ever
//
// `fwidth`, `dpdx` and `dpdy` are forbidden in this file, and `tier_parity` asserts their absence
// by reading the source. The reason is not style: every shading rule here has to be reproducible
// by the CPU-side transcription, and a screen-space derivative has no meaning on a rasterizer with
// no neighbouring fragment. The moment one appears, the parity suite stops being evidence about
// this shader and starts being a second opinion about geometry.
//
// Everything below is therefore analytic. The slab distance function has a closed-form gradient,
// so a normal costs no extra scene evaluations; the shadow term reads the distance field the march
// is already producing; the occlusion term samples a stated, bounded number of points. See
// `specs/002-ray-traced-mode/research.md` R4 and R5.
//
// # The camera is orthographic and that is the whole trick
//
// The ray for a pixel starts directly above it and travels straight down. No perspective divide,
// no field of view. Every slab therefore occupies exactly the screen rectangle its instance
// occupies, which is what lets the interface gain a third dimension without a person's hit targets
// moving a pixel.
//
// Implementation lands in T034-T036 (shadow, occlusion), T050 (bounce) and T067 (refraction).
