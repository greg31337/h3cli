# Qwen scaling validation on M4 Max

This change removes an unnecessary BF16 rounding step from Qwen attention and
adds a reference formulation that scales the completed FP32 QK score. The
numerical evidence supports the attention correction. Full Qwen encoder parity
is close but mixed across prompts; this is not evidence of universally better
videos or monotonically lower final-layer error.

Baseline: `91546c572f8f6548ddd7da400442b0ffbf8eeae2`. Hardware: M4 Max, 128 GiB
unified memory. No M5 results are claimed. Detailed metrics and artifact
identities are in [scalingfix-validation.json](scalingfix-validation.json).
Large raw tensors, logs, renders and the synchronized review gallery live under
`outputs/scalingfix-validation/`.

The original measurements below describe the scaling rollout at `f74a37d`,
when legacy still retained the race. Production legacy now also includes the
barrier. See the [legacy synchronization follow-up](#legacy-synchronization-follow-up)
for validation of that change; the original legacy render and timing results
are retained as historical evidence.

## Arithmetic and synchronization

The old kernel converts BF16 Q to FP32, multiplies by the FP32 attention scale,
rounds back to BF16, converts back to FP32, and serially accumulates each QK dot
with FP32 FMA. It then performs a tree maximum reduction, FP32 exponentiation,
a tree sum reduction, FP32 weighted V accumulation and BF16 output conversion.
Only Qwen's `h3_gpu_gqa_causal_bf16` call uses this kernel. DiT, vision attention
and VAE kernels are outside the change.

Testing reproduced the shared-memory read/write race between maximum
consumption and sum reduction reported in [issue #52](https://github.com/antirez/h3.c/issues/52).
The initial rollout fixed it in the two corrected variants, with a test-only
synchronized legacy shader to separate synchronization from scaling precision.
The follow-up below applies that same barrier to production legacy as well.

The corrected kernels reproduced exactly across repeated synthetic dispatches
and complete encoder runs. Legacy showed intermittent changes on the 2,048-token
and BF16-boundary synthetic cases. Five of seven final legacy synthetic outputs
matched the old binary exactly; long and medium cases differed. Plain-text and
dialogue encoder outputs matched the old binary exactly at every saved layer;
image/video outputs varied. Historical nondeterministic output cannot be
promised bit-for-bit even when selecting legacy.

## Independent kernel tests

`tests/test_qwen_scaling.c` implements its own scalar FP32 causal attention
oracle, including post-dot scaling and stable softmax. BF16 inputs use
production dimensions: 64 query heads, 8 KV heads and head dimension 128.
Sequences 7, 128 and 2,048 cover small/medium/long execution. The long case
validates 17 distributed query rows while running the full GPU sequence.
Other cases exercise near ties, dominant logits, large positive/negative
logits and BF16 boundaries. Every GPU output is checked for NaN/Inf.

Representative RMSE against the BF16-rounded CPU oracle:

| Case | Legacy | Scaled-Q | Reference |
| --- | ---: | ---: | ---: |
| Random, 128 tokens | 4.78e-4 | 6.98e-6 | 4.98e-6 |
| Random, 2,048 tokens | 1.53e-4 | 1.92e-6 | 1.66e-6 |
| Dominant logits | 1.19e-3 | 1.32e-4 | 8.26e-6 |
| Large logits | 2.48e-2 | 2.64e-5 | 0 |

The near-tie BF16-rounded RMSE is slightly smaller for scaled-Q (2.65e-6 versus
5.92e-6); reference has fewer mismatched BF16 values, and the unrounded RMSE
is effectively identical (difference about 2e-12). Individual BF16 midpoint
crossings explain this tiny reversal; it is not a material accuracy regression.
Raw max/mean error, RMSE, relative L2 and mismatch rates are retained for every
case and every mode.

Captured real first-layer Q/K/V are also compared with an independent NumPy
FP32 attention implementation. All modes' inputs are byte-identical. Reference
attention RMSE is 35–290 times smaller than synchronized legacy across the four
presentations and is smaller than scaled-Q in all four cases. This isolates the
scaling improvement from synchronization and upstream normalization differences.

## Official full-encoder comparison

Real presentations contain 13 plain-text tokens, 33 dialogue tokens, 149 image
reference tokens and 829 video reference tokens. The image uses `inputs/2.jpg`;
the video is a camera-pan fixture derived from `inputs/body1.jpg`. Native
presentation builders and vision encoders supply the exact same IDs, mRoPE
positions, vision rows, deepstack rows and tags to every run. No conditioning
replay is used in encoder comparisons. All 14 FL2VA/Ref2VA text weight shards
were compared byte-for-byte and are identical; their SHA-256 values are retained.

The official runner uses Transformers 5.17.0 Qwen3-VL decoder layers with the
released weights, BF16 activations and official mRoPE, streamed one layer at a
time. It stops after layer 50 without final RMSNorm, matching H3's conditioning
boundary. The primary backend is official SDPA on MPS. A secondary runner calls
upstream eager attention with FP32 Q/K/V and casts its output to BF16; it exposes
the design's explicit attention precision while retaining other official
operation boundaries. This secondary runner is identified separately rather
than described as unmodified upstream execution.

Relative L2 at layer 50 against official SDPA:

| Presentation | Legacy | Scaled-Q | Reference |
| --- | ---: | ---: | ---: |
| Plain | 0.0776% | 0.0748% | 0.0802% |
| Dialogue | 0.1549% | 0.1789% | 0.1921% |
| Image | 0.6285% | 0.5809% | 0.2526% |
| Video | 0.6474% | 0.7768% | 0.9793% |

Reference's worst final relative L2 against the secondary FP32-eager runner is
1.1462%. Every saved checkpoint passes the report's 2% relative-L2 / 0.9998
cosine gate, tighter than the repository's existing 15% full multimodal parity
bound. This gate establishes close agreement, not an ordering claim. Reference
is not uniformly closer than the alternatives at the final conditioning layer.

Final conditioning is also measured by language rows and individual tokens,
so large hidden-state outliers cannot hide all smaller-row differences. Against
official SDPA, reference's last-token relative L2 is 0.98%, 1.29%, 0.92% and
0.90% for plain/dialogue/image/video; legacy is 0.91%, 1.50%, 1.18% and 1.09%.
The reference image language rows improve from 0.13% to 0.10% relative L2;
video language rows remain around 0.16%. The reference 95th-percentile per-token
relative error ranges from 1.52% to 4.60%, with dialogue the largest. Thus the
2% whole-tensor gate does not imply a 2% bound on every token. These diagnostics
support close conditioning agreement while preserving the mixed overall result.

Initial embedding/vision tensors match the official runner exactly. The first
post-RoPE Q/K tensors already differ by roughly 0.3–0.4% relative L2 before GQA.
Native norms, fused rotary arithmetic, plain-text FP32 rotary tables and fused
SwiGLU use different BF16 operation boundaries from upstream PyTorch. Those
pre-existing differences remain outside this scaling change. Late layers can
amplify small differences at sharp attention decisions. Files preserve initial
hidden states, first-layer Q/K/V/attention and layers 1–5, 10, 25, 40 and 50.

## Performance, memory and control flow

Warmed kernel GPU times (milliseconds; rotating mode order, median of ten
measured rounds after two warmup rounds):

| Tokens | Legacy | Scaled-Q | Reference | Reference vs legacy |
| --- | ---: | ---: | ---: | ---: |
| 13 | 0.02415 | 0.02349 | 0.02315 | −4.1% |
| 128 | 0.39490 | 0.39959 | 0.39523 | +0.1% |
| 829 | 11.82807 | 12.24259 | 11.97624 | +1.3% |
| 2,048 | 100.07318 | 100.17331 | 100.30481 | +0.2% |

No material kernel slowdown was measured. Small sequences batch 64 dispatches
per measurement and large sequences batch two, reducing timing overhead.

Full encoder runtimes (seconds; warm model file cache, same capture overhead):

| Presentation | Legacy | Scaled-Q | Reference |
| --- | ---: | ---: | ---: |
| Plain | 3.546 | 3.479 | 3.434 |
| Dialogue | 3.325 | 3.406 | 3.286 |
| Image | 3.350 | 3.442 | 3.452 |
| Video | 5.306 | 5.270 | 5.304 |

Reference is within −3.2% to +3.0% of legacy on these single encoder samples.
Tensor allocation counts, cumulative allocation bytes and peak live Metal bytes
are identical across all modes for each presentation. Process peak RSS varies
with loading/cache timing; reference differs from legacy by −4.8% to +1.0%.
The shader adds no activation or threadgroup storage.

All modes retain 1,024 static threadgroup bytes and the same 7,936-token M4
limit. Actual dispatches at 7,935/7,936 and rejection at 7,937, zero and SIZE_MAX
pass in every mode. Real-weight user cancellation and injected low-memory
cancellation stop after exactly one Qwen layer in every mode; normal and
ASan/UBSan runs pass. Old `.h3sample` conditioning and `.h3av` files load in every
mode. Full resume's existing build/environment restrictions remain enforced.

## Generation review and rollout

All nine full generations completed. Each case uses the same full 50-layer
model, 320×320 canvas, 124 frames, 20 steps and seed 72. Render times in seconds:

| Case | Legacy | Scaled-Q | Reference |
| --- | ---: | ---: | ---: |
| FL2VA dialogue, `face1.jpg` | 327.1 | 330.8 | 330.8 |
| Ref2VA image, `2.jpg` | 331.8 | 331.6 | 332.6 |
| Ref2VA video derived from `body1.jpg` | 389.7 | 409.7 | 413.8 |

The reference video render took 6.2% longer in this single-run comparison.
Its Qwen phase was 7.760 s versus legacy's 7.700 s; DiT denoising was
351.989 s versus 330.096 s and accounts for most of the difference. No DiT
implementation was changed. This does not establish the cause of the render
timing variation or a general end-to-end performance bound. The isolated,
rotating kernel benchmark and encoder measurements above are the more direct
evidence for the scaling change's cost. Generation Qwen times include
model-loading/cache variation and range from 3.080 to 7.760 s.

Seven matched timestamps were reviewed across all three modes, with denser
21-frame legacy/reference sheets for image and video cases. The synchronized
gallery is `outputs/scalingfix-validation/generation/index.html`.

- FL2VA: identity, backpack, background and framing remain coherent. All three
  transcribe to “Where are you going? To the garden.” with the same coarse
  timestamps (0–2 s, 3–5 s). Sampled mouth movement is consistent with the
  speaking intervals and pause.
- Image: all modes retain the main subject on the left, blue-shirted person
  on the right, table, red book and window. The requested hand gesture is weak,
  the third subject mostly occluded, and fine reference details unreliable in
  all modes. No new corrected-mode regression was apparent.
- Video: the main subject's glasses, bob and dress remain recognizable, with
  three people, window, forward movement and head turns. Unrequested moving
  colored-light spots persist in all three modes. The scaling correction does
  not fix those existing artifacts. No new subject loss, scene change or
  corrected-mode artifact was apparent in the reviewed frames.

The harness captures fresh text, every Qwen input, source pixels, VAE conditions,
initial video/audio noise, schedules, final latents and decoded RGB/PCM. It
verified all non-Qwen inputs match exactly across modes. All captured latents,
RGB and PCM are finite. Every output contains 124 frames and stereo 32 kHz
audio; the muxed audio/video duration difference is 8.33 ms, below one frame.
This small, unblinded comparison found no obvious new regression; it does not
establish universally better videos. Speech transcription and sampled frames
support coarse dialogue timing, not phoneme-level lip synchronization.

After reviewing the numerical, encoder, generation and performance evidence,
the unset/empty default was changed to `reference`. The final uninstrumented
production library reproduces explicit-reference conditioning byte-for-byte
for all four presentations. Synthetic dispatch tests also verify both unset
and empty environment values produce the explicit-reference output, and cache
tests verify those aliases share the same resolved-mode key. Identical
historical seeds may now produce different renders; see the release note in
[scalingfix.md](scalingfix.md).

## Final regression checks

`make test` passed after the default change, including all seven synthetic
scaling cases, 103,196 continuation checks, 1,331 sampler-state checks, GQA
boundary dispatches, tokenizer, tile policy, audio primitives and bridge tests.
Ten existing optional fixture groups were skipped because their fixture files
are not installed; these are not counted as passes. The separate scaling
validation above exercised released model weights, official Qwen comparisons
and full generations. Builds produced no compiler warnings.

T001–T055 are complete with the documented legacy reproducibility limitation.
T056 remains deferred: its compatibility period must occur after release, and
`scaled-q` still provides useful arithmetic-order diagnostics today.

## Legacy synchronization follow-up

The barrier after reading the maximum is now unconditional across all modes.
Legacy retains BF16 query re-rounding, while preventing softmax sums from
overwriting the maximum before every thread has read it. Reference and scaled-Q
arithmetic and synchronization are unchanged.

The mandatory synthetic test now requires exact repeatability in every mode
over 32 dispatches per case, including a new 662-token case matching the issue's
reported sequence length. It still checks the independent FP32 oracle, finite
outputs, memory use and default-mode aliases. The historical baseline build is
exempt from the repeatability assertion because it intentionally tests old code.

`tests/scalingfix_legacy_sync.py` runs the uninstrumented production encoder
twice per mode for all four real presentations. Legacy is checked against the
previously captured synchronized-legacy output; reference and scaled-Q are
checked against their unchanged captures. Logs and results are kept separately
under `outputs/scalingfix-validation/legacy-sync/`.

All 24 encoder runs passed with byte-identical expected outputs. All eight
synthetic cases passed in all three modes with zero repeat mismatches across
32 dispatches per mode/case. Unset-default encoder checks passed for all four
presentations. All-mode cancellation, low-memory guard, historical saved-state
loading, ASan/UBSan control checks, and actual sequence-boundary dispatches also
passed. Static threadgroup memory remains 1,024 bytes and the sequence limit
remains 7,936 on the M4 Max.

The complete `make test` run passed with no compiler warnings; the same ten
optional fixture groups remain unavailable and were skipped. These skips are
not counted as passes. Structured follow-up results are in
[scalingfix-legacy-sync-validation.json](scalingfix-legacy-sync-validation.json).

Warmed legacy kernel medians were 0.02473 / 0.39613 / 11.98898 / 100.25059 ms
at 13 / 128 / 829 / 2,048 tokens. These are 0.2–2.4% above the earlier
unsynchronized legacy measurements, with no large slowdown observed. The runs
were in separate sessions, so this comparison does not isolate the barrier's
exact cost.

To repeat the follow-up after preparing the original encoder fixtures:

```sh
make -j8 bin/h3cli bin/libh3.a
make test
python3 tests/scalingfix_legacy_sync.py
python3 tests/scalingfix_default.py
python3 tests/scalingfix_control.py
python3 tests/scalingfix_control.py --sanitize
```

## Reproduction

Run GPU jobs sequentially. The installed oracle environment is
`outputs/refvideo-encoder-validation/venv/bin/python`; dependencies are pinned in
`tests/requirements-scalingfix.txt`.

```sh
make -j8 bin/h3cli bin/libh3.a
python tests/scalingfix_prepare.py
python tests/scalingfix_identity.py
python tests/scalingfix_encoder.py
python tests/scalingfix_encoder.py --modes legacy-synchronized
python tests/scalingfix_oracle.py
python tests/scalingfix_oracle.py --backend fp32-eager
python tests/scalingfix_metrics.py
python tests/scalingfix_attention.py
python tests/scalingfix_checks.py
python tests/scalingfix_generation.py
python tests/scalingfix_gallery.py
python tests/scalingfix_speech.py
make bin/qwen_scaling_bench
./bin/qwen_scaling_bench > outputs/scalingfix-validation/benchmark.jsonl
python tests/scalingfix_default.py
python tests/scalingfix_report.py
make test
```

`scalingfix_prepare.py` creates a pinned baseline from commit `91546c5` under
`outputs/scalingfix-validation/baseline`. Speech
review additionally uses the public `openai/whisper-tiny.en` model downloaded
locally under `outputs/scalingfix-validation/whisper-tiny.en`. Test captures and
synchronized-legacy instrumentation exist only in isolated output builds.

## Scaling-fix validation notes

See [scalingfix-validation.md](scalingfix-validation.md) and its JSON metrics.
T005/T045 preserve legacy arithmetic and reproduce the old deterministic text
fixtures exactly. Historical multimodal execution contains a discovered shared
softmax reduction race, so bit-exact legacy replay is not guaranteed where the
old implementation itself was nondeterministic. Corrected modes fix that race.

T056 requires a future compatibility period. `scaled-q` still demonstrates
useful operation-order differences in current tests, so it remains available;
removal is deliberately deferred until release history supports it.

## Legacy synchronization follow-up

T005/T045 preserve the historical scaling arithmetic. At the user's request,
legacy now also fixes the shared softmax reduction race, so historical outputs
that encountered the race are not guaranteed to replay exactly. All modes must
pass the repeated-dispatch regression. See [scalingfix-validation.md](scalingfix-validation.md#legacy-synchronization-follow-up).
