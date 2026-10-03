This directory contains the unmodified MIT-licensed Metal attention template
and its transitive headers from Apple's MLX, pinned in `manifest.json`.
`LICENSE` must accompany redistributions. No MLX runtime or Python package is
required. `scripts/embed_metal_attention.py` verifies every pinned file and
embeds attention specializations in the native executable.
The H3 compilation entry point is `src/native_attention.metal`. The original
prototype uses the unchanged template with BF16 storage and FP32 matrix,
softmax and output accumulation. M1 additionally provides an opt-in
hybrid DiT path; it has not passed production qualification.
M2's FP16 candidate is a local adaptation in `src/routed_attention.metal`, used
standalone and through an explicit hybrid:
half Q/K/P/V matrix operands with FP32 accumulation, GPU range adaptation,
conditional BF16/FP32 recovery and a checked BF16 output boundary. It does not
modify these vendor headers or add an MLX runtime dependency. See
`docs/metal/m2-source-audit.json` for the supplementary VPIPE/NAX source audit.

The h3cli adapter supplies the small `METAL_FUNC`, `AccumHelper`, and float
`Limits` prelude normally supplied by MLX's generated kernel compilation unit.
Masks, causal attention, sinks, and grouped-query attention are disabled in
the H3 DiT specialization; the vendored template itself remains unchanged.

Source: https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/backend/metal/kernels/steel/attn
