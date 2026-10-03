# M3B/M4 single CUDA pipeline — implementation and qualification

Status: **M3B and M4 complete.** All twelve configurations executed successfully;
the final unchanged default gate passed all 204 artifacts. See the
[design](design-single-pipeline.md) and [checklist](single-pipeline-tasks.md).

## Scope and source lineage

Starting source digest:
`6154af2d168eebe0154fc34178d0c7a25f79e8ec33a2921abcc83a8625bd5926`.
Frozen manifest:
`4ac45c94f1cbec276333102676ce1600ead71da025df4ca60b9ae2343098e413`.
Expected outputs are recorded in `tests/cuda_reference/manifest.json`.
All 16 frozen files and 204 expected artifacts remain unchanged.

The first coherent patch removes public fast policy, permits explicit Sage/SOL
on the shared base, rejects old fast states and disables activation of old fast
arithmetic. Its default gate passed 204/204 in 126.732 seconds after build.
The next enables FP8/NVFP4, adds counters and cancellation recovery, and updates
mutable state/option tests. Its gate passed 204/204 in 120.154 seconds.
The mutable test/example cleanup passed 204/204 in 122.972 seconds. A final
audit removed the retired low-level request bit from arithmetic selection; that
source passed 204/204 in 130.704 seconds. Its digest is
`0dfe45c68d596d0f2ef5fd54ad268dbf9473e72c6a39c9f02e775a9b7df051d9`.
The complete qualification campaign used this fourth source without further
runtime edits. Its closing gate passed 204/204 in 123.624 seconds after build
(237.181 seconds including the fresh build).

## Consumer and compatibility audit

| Consumer | Current route / restriction |
| --- | --- |
| Default video and explicit `sglang` | Shared SGLang arithmetic 4, dense BF16, original full decoder |
| Sage2++ / Sage3 / SOL | Explicit main-DiT boundary; same preparation, normalization/RoPE, sampler and delivery |
| FP8 / NVFP4 | Eligible main-block QKV/output/MLP weight descriptors, shared surrounding operations |
| Qwen, refiners, patch/velocity heads, conditioning encoders | Original BF16/FP32 operators, no optional attention/quantization |
| First/last-frame, ordered image/video/audio, match/max | Shared conditioning/packing and protected ranges |
| LoRA/Turbo | Existing BF16 fold resolution before DiT quantized packing; no fast flag |
| Reuse/reduction | Existing explicit controls and restrictions on the shared sampler; outside default parity preset |
| Saved conditioning | Existing source/model metadata and encoder policy checks; retired fast identity is zero |
| Resume | Reference arithmetic/build/environment checks plus explicit attention/quant/SOL recipes; old fast/legacy CUDA trajectories rejected |
| Continuation and clean AV delivery | Existing container/presentation/decoder compatibility; no numerical relabeling |
| CUDA still/image VAE | Existing shared low-level primitives and image-specific restrictions; video approximation flags rejected; no SGLang still-parity claim |
| Preview VAE / explicit full-VAE variants | Independent presentation choices, no effect on denoising option selection |
| Metal | Existing math/dispatch/defaults retained |

Remaining `fast_cuda` fields are zero-only public ABI/state compatibility fields,
old-state detection, or the frozen low-level probe's telemetry echo. Production
policy has no mode/fast-recipe selector. Old fast kernels cannot be enabled by
CLI/API/environment. Unreachable legacy primitive cleanup remains M6, as planned.

## Qualification protocol

The predeclared matrix contains dense/Sage2++/Sage3/SOL × BF16/FP8/NVFP4.
Complete C0/R1 visual clips use 640×480, 124 frames, six evaluations, original
full VAE, identical prompt/seed/conditions. Supplemental image `match` and
continuation cover all four BF16 attention choices. Performance uses three
interleaved matched reference/candidate pairs per nondefault cell at 640×480 /
243 frames / six evaluations. Both requests bind CPU/memory to NUMA node 0 on
the qualification RTX PRO 5000 72GB. Report ratio-of-medians wall speedup and denoise
speedup separately; no change to existing memory admission limits.

The first complete quantized run for each model variant retains cold preparation
cost; repeated timing uses populated metadata caches. Caches live on the large
`/models` filesystem. Original weights are neither hashed nor modified. Weight
packing is independent of attention, so packed caches are shared between Sage,
SOL and dense configurations of the same precision. Payload header/size/scales
are checked; arbitrary finite bit corruption is not claimed detectable without
strict diagnostic hashing.

Quality changes are allowed for explicit approximate choices. Every-frame SSIM,
LPIPS, temporal and audio comparisons still run against the retained historical
thresholds, with failures reported as differences rather than waived parity.
Correctness requires finite stable execution, real selected dispatch and complete
valid AV output. Human review is not claimed and does not block the report.

## Early checks and retained failures

Local host policy, sampler and metadata-cache tests passed; the sampler exercised
1,773 checks and the cache 179 checks without GPU or weight hashing. Local Metal
builds. On CUDA, all twelve operator cells passed in both forward/reverse context
orders, including tail rows, finite output, actual dispatch, allocation plateau
and cancellation. Mapped/copied staging, isolated option policies, forced-dense
SOL and native FP8/NVFP4 range/scale cases passed.

The first external campaign stopped before real-QKV comparisons because it used
system Python without NumPy. Its failure record is retained. The driver now uses
the already-installed SGLang validation environment; no dependencies or frozen
suite files were changed. No render from that stopped campaign is counted as a
performance result.

The third campaign was stopped deliberately before full renders to remove the
last retired request-bit influence. Its in-flight CLI/delivery checks finished
successfully; its record remains incomplete, not a qualification pass. The fourth
campaign repeated all operator, policy, cache, cancellation, real-QKV, CLI,
decoder and still-host checks successfully before starting complete renders.

After all 98 renders, the first CPU quality comparison stopped because its
environment omitted the existing LPIPS provider path. FFmpeg cleanup waited
until its two child decoders were terminated; that failed comparison record and
log are retained. Comparisons resumed using `/path/to/qualification/metrics-site`,
the same LPIPS v0.1/AlexNet weights and unchanged thresholds, without rerendering,
changing source, installing packages or upgrading dependencies.

## Final performance and playback

[Open the playback and timing summary](../../outputs/cuda-single/review.html),
[text-only comparisons](../../outputs/cuda-single/review-C0.html), and
[mixed-reference comparisons](../../outputs/cuda-single/review-R1.html).
The gallery contains 22 side-by-side comparisons from 24 complete matrix clips,
plus eight match/continuation clips. Each comparison includes every-frame metrics,
worst frames, commands, identities, stage timings and memory.

Performance below is the ratio of medians from three matched interleaved pairs
per option: **640×480, 243 frames, six evaluations, original full VAE**. Timings
are complete request wall time on RTX PRO 5000, with warm packed caches and both
requests bound to NUMA node 0. Values below 1× are slower. Differences of a few
percent should not be interpreted as a reliable ordering between candidates.

| Attention | Projections | Reference wall (s) | Candidate wall (s) | Wall speedup | Denoise speedup | Peak VRAM (GiB) | Peak host RSS (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sage2++ | BF16 | 131.26 | 120.20 | 1.092× | 1.202× | 8.64 | 36.86 |
| sage3 | BF16 | 127.90 | 112.71 | 1.135× | 1.305× | 8.64 | 36.95 |
| sol | BF16 | 127.86 | 175.80 | 0.727× | 0.580× | 8.64 | 36.85 |
| default | FP8 | 129.26 | 128.91 | 1.003× | 0.990× | 8.28 | 18.92 |
| sage2++ | FP8 | 132.57 | 119.69 | 1.108× | 1.136× | 8.78 | 18.92 |
| sage3 | FP8 | 129.86 | 116.98 | 1.110× | 1.177× | 8.78 | 19.01 |
| sol | FP8 | 133.68 | 177.55 | 0.753× | 0.564× | 8.78 | 18.92 |
| default | NVFP4 | 130.71 | 98.87 | 1.322× | 1.329× | 8.08 | 11.07 |
| sage2++ | NVFP4 | 130.92 | 90.29 | 1.450× | 1.618× | 8.58 | 11.07 |
| sage3 | NVFP4 | 131.11 | 89.64 | 1.463× | 1.709× | 8.58 | 11.16 |
| sol | NVFP4 | 129.21 | 149.67 | 0.863× | 0.656× | 8.58 | 11.07 |

Sage3/NVFP4 and Sage2++/NVFP4 have similar complete wall time; Sage3/NVFP4
has the lowest median here and the largest measured denoising speedup. FP8 alone
is effectively tied with BF16. SOL uses the existing conservative defaults,
including `min_exact=0.75`; it is slower on this workload. These results do not
qualify different SOL routing settings, other GPUs or larger production geometry.

All memory values cover the whole request, including conditioning and full
decoding; projection compression need not reduce that overall peak. Full stage
medians and paired reference memory measurements are in the linked report.

### Visual differences and cold preparation

Quality degradation is permitted only for explicit options. The following retained
comparisons use 124-frame/six-evaluation clips, with the same seed and full decoder.
A historical-threshold failure is reported, not converted into a parity pass.
No human visual acceptance is claimed. Selected Sage3/NVFP4 worst-frame
inspection shows changes in the text-only clip’s framing and facial appearance;
the mixed-reference pair has smaller changes in facial detail, piano geometry
and lighting. These selected stills do not establish whole-clip visual equivalence.
All 22 comparisons fail at least one historical limit.

| Case / option | Min frame SSIM | Max frame LPIPS | Max temporal RMS | Audio relative L2 | Historical thresholds |
| --- | ---: | ---: | ---: | ---: | --- |
| C0-sage2++-off | 0.52638 | 0.41008 | 0.03425 | 0.26428 | not met |
| C0-sage3-off | 0.47440 | 0.64174 | 0.05009 | 0.55841 | not met |
| C0-sol-off | 0.63004 | 0.29887 | 0.04404 | 0.20745 | not met |
| C0-default-fp8 | 0.60015 | 0.30918 | 0.03778 | 0.36245 | not met |
| C0-sage2++-fp8 | 0.74850 | 0.20466 | 0.03080 | 0.70689 | not met |
| C0-sage3-fp8 | 0.52924 | 0.48394 | 0.04084 | 0.67988 | not met |
| C0-sol-fp8 | 0.61002 | 0.38426 | 0.03583 | 0.32327 | not met |
| C0-default-nvfp4 | 0.46932 | 0.66326 | 0.03307 | 0.62941 | not met |
| C0-sage2++-nvfp4 | 0.48486 | 0.59622 | 0.03406 | 0.64397 | not met |
| C0-sage3-nvfp4 | 0.50901 | 0.58122 | 0.02877 | 0.65622 | not met |
| C0-sol-nvfp4 | 0.47712 | 0.65946 | 0.03887 | 0.60850 | not met |
| R1-sage2++-off | 0.83482 | 0.04650 | 0.02445 | 0.14479 | not met |
| R1-sage3-off | 0.69460 | 0.16013 | 0.02585 | 0.44945 | not met |
| R1-sol-off | 0.84124 | 0.04281 | 0.02337 | 0.05982 | not met |
| R1-default-fp8 | 0.78337 | 0.13398 | 0.02559 | 0.29124 | not met |
| R1-sage2++-fp8 | 0.86358 | 0.04721 | 0.01881 | 0.28895 | not met |
| R1-sage3-fp8 | 0.74653 | 0.13656 | 0.02334 | 0.49327 | not met |
| R1-sol-fp8 | 0.80354 | 0.11978 | 0.02183 | 0.27454 | not met |
| R1-default-nvfp4 | 0.79015 | 0.09584 | 0.02265 | 0.76885 | not met |
| R1-sage2++-nvfp4 | 0.78572 | 0.11678 | 0.02400 | 0.76026 | not met |
| R1-sage3-nvfp4 | 0.72803 | 0.13017 | 0.02550 | 0.82800 | not met |
| R1-sol-nvfp4 | 0.76408 | 0.10410 | 0.02365 | 0.79311 | not met |

First complete quantized runs retain cold preparation rather than mixing it into
warm timing claims:

| Case | Quantization | Complete wall (s) | Transformer loading (s) | Cache hits / newly prepared descriptors |
| --- | --- | ---: | ---: | --- |
| C0 | fp8 | 121.70 | 47.93 | 0 / 200 |
| C0 | nvfp4 | 91.30 | 32.09 | 0 / 200 |
| R1 | fp8 | 209.42 | 46.20 | 0 / 200 |
| R1 | nvfp4 | 181.11 | 38.27 | 0 / 200 |

### Closing validation

The final source passes the original 204-artifact reference suite with its
recorded golden manifest and all sixteen frozen files unchanged. No case, threshold,
fixture, dependency, evaluation count or golden was modified. Final gate records:
[closing gate](../../outputs/cuda-single/evidence/p4-final-gate/result.json),
[option matrix and all timing pairs](../../outputs/cuda-single/evidence/p4-qualification/result.json),
[conditioning/resume/alias checks](../../outputs/cuda-single/evidence/p4-consumers/result.json),
and [artifact checksums](../../outputs/cuda-single/artifact-manifest.json).

Bounded 64×64/22-frame/two-evaluation consumer checks cover dense BF16,
Sage2++/FP8 and SOL/NVFP4. Saved-conditioning stop/resume reproduces direct
clean AV state and MP4 bytes exactly; changed attention on resume is rejected.
Explicit `--cuda-reference sglang` also reproduces default AV and MP4 bytes
exactly. These are compatibility checks, not additional quality benchmarks.

The final default log confirms dense attention, zero Sage/SOL/quantized calls,
quantization disabled, separate full-VAE casts, mapped DiT staging and arithmetic
4. Local Metal builds and bounded host tests passed. Optional-kernel capability
errors, twelve-cell context ordering, cancellation/recovery, real QKV, tails,
cache validation and bounded allocation checks passed independently of the frozen
suite. Residual unreachable helper deletion and broader held-outs remain M6/M7.

Tested CUDA executable: `/path/to/qualification/cuda-single/p4-gate/build/bin/h3cli`
on the configured CUDA host; the independent closing build is under
`p4-final-gate/build/bin/h3cli`. Source, binary and dependency identities are retained
in the linked records. Models and packed caches remain on the server.
