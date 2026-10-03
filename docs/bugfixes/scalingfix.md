# Qwen attention scaling

`H3_QWEN_GQA_SCALE_MODE` selects the Qwen text/multimodal encoder's causal
attention arithmetic. It does not select DiT or VAE attention. Q/K/V activations
remain BF16 in memory; the QK contraction and softmax use FP32. Unset or empty
`H3_QWEN_GQA_SCALE_MODE` now selects `reference`.

| Mode | Arithmetic | Purpose |
| --- | --- | --- |
| `reference` | FP32 QK dot, then FP32 multiplication by `1/sqrt(head_dim)` | Intended production formulation |
| `scaled-q` | FP32 Q times scale, then FP32 QK dot | Diagnostic for operation order |
| `legacy` | FP32 Q times scale, rounded to BF16, converted back to FP32, then FP32 QK dot | Historical compatibility and regression bisects |

Configuration is case-sensitive. Unknown non-empty values fail with a named
configuration error. Empty values follow the unset default. Set the mode before
starting a generation; mutating process environment during an active generation
is unsupported. With `H3_PROFILE=1`, Qwen initialization reports the selected mode.
Conditioning and prepared-model cache keys include the resolved mode.

Three dedicated Metal pipelines specialize a shared template. Scaling in
`reference` occurs after the existing serial FMA contraction and before the
maximum reduction, exponentiation, normalization or attention weighting. The
threadgroup layout, causal mask, reduction order, V accumulation and final BF16
conversion are retained. No additional activation buffers are allocated.

The old `H3_MPS_GQA` override is rejected: opaque MPSGraph SDPA
cannot enforce these three arithmetic contracts and formerly bypassed direct
GQA preflight. Unset it and select an explicit Qwen scaling mode instead.

## Synchronization discovered during validation

The previous kernel read the maximum from `reductions[0]` and reused the same
buffer for softmax sums without a barrier between those operations. One SIMD
group could write its sum before another read the maximum. Repeated identical
inputs exposed occasional large errors, separately from attention scaling.

All three variants insert a threadgroup barrier after reading the maximum,
including `legacy`. Legacy preserves the historical BF16 scaling arithmetic,
but no longer reproduces the unsafe synchronization. Historical renders that
encountered the race can therefore differ even with explicit `legacy`; their
nondeterministic output cannot be reproduced reliably. Legacy remains a
debugging option, not a recommended quality mode. The earlier test-only
`legacy-synchronized` fixture now describes production legacy behavior.

All variants use the same dynamically sized score allocation and runtime
preflight. On the validated M4 Max the limit remains 7,936 tokens, with 1,024
bytes of static threadgroup storage, 16-byte dynamic alignment and a 32,768-byte
device limit. Future tiled/online causal GQA must preserve:

```
BF16 Q/K -> FP32 QK accumulation -> FP32 scale -> online softmax
```

## Release note: new reference default

The default has changed from legacy to reference after M4 numerical, encoder,
generation and performance validation. Identical historical seeds can now
produce different conditioning, latents, audio and video. Explicit `legacy`
retains the old arithmetic for troubleshooting, with the synchronization fix
described above. This is an attention-precision and repeatability fix; the small render
comparison does not establish a general perceptual quality improvement.

## Reproducibility and saved states

Changing scaling mode can change Qwen conditioning, denoising trajectories,
audio/video latents and rendered videos for identical historical seeds. The
mode does not change tokenization, multimodal presentation, reference video
preprocessing, weight selection, RNG initialization or scheduler configuration.

No saved-file schema changes are needed. Serialized BF16 conditioning remains
readable and is used as stored, without running Qwen again. Full `.h3sample`
resume still enforces the existing exact engine-build, backend, model and
numerical-environment checks; this change does not bypass those checks. Changing
`H3_QWEN_GQA_SCALE_MODE` while resuming a sampler checkpoint is therefore rejected
by the existing environment check. Older-build files can be inspected/decoded,
but arbitrary cross-build exact sampler resume remains unsupported.

AV continuation files remain compatible. Continuing from `.h3av` runs Qwen for
the new prompt, so the chosen mode affects the new generation. Reusing already
computed conditioning preserves that tensor; restarting before conditioning
uses the current mode. Same-format compatibility does not imply identical new
renders.

## Validation and future cleanup

See [scalingfix-validation.md](scalingfix-validation.md) for the M4 results,
reproduction commands, official-reference differences and quality-review limits.

Retain `scaled-q` through a compatibility period. Review its use after release
history and user regression reports are available; remove it only if operation
order no longer provides useful diagnostic information. Retain `legacy` longer
for historical bisects. T056 is a future review, not a claim that a compatibility
period has already elapsed.

* [DEFERRED] T056: After an appropriate compatibility period, evaluate removing the `scaled-q` mode if it no longer provides useful diagnostic value while retaining `legacy` for reproducibility and regression bisecting.
