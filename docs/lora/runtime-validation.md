# Native runtime LoRA validation

Validation date: September 19, 2026. Implementation and required hardware
acceptance are complete within the scope below. The checklist is [../todo.md](../todo.md); architectural details are
in [design-runtime-lora.md](design-runtime-lora.md).

## Reproduction and scope

The native runtime has no Python dependency. Validation uses these independent
host and optional real-model drivers:

```sh
make test-lora-runtime
make test-lora-runtime-sanitize
make test-lora
make bin/sampler_tests && ./bin/sampler_tests
make bin/lora_runtime_generate

# Optional NumPy oracle, using the offline tooling's isolated environment.
lora/.venv/bin/python tests/lora_runtime_parity.py \
  --base models/MiniMax-H3/FL2VA/transformer \
  --native /path/to/cache/variants/KEY/transformer \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors \
  --offline lora/output/Turbo/FL2VA/transformer --output outputs/parity.json

python3 tests/lora_runtime_workflows.py --backend metal \
  --model models/MiniMax-H3 --cache outputs/runtime-lora/cache \
  --turbo lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors \
  --ordinary lora/downloads/h3-realism-people-t2v-i2v-r2v.safetensors \
  --output outputs/runtime-lora/local/workflows \
  --cases baseline resume ref-resume continuation
```

The real workflow driver records complete arguments, executable SHA-256, exit
status, wall/CPU time, child maximum RSS, and canonical AV-state hashes. It uses
`H3_CPU_SAMPLER=1` consistently for matched comparisons. The current driver uses `wait4` for each child
process's peak RSS and CPU time. Earlier artifacts without `rss_scope` used the
maximum child high-water mark so far; those are labeled cumulative bounds, not
isolated per-phase allocations. Numerical validation does not assess visual quality,
prompt fidelity, adapter strength preferences, or long-video quality.

The pinned adapters are the existing local Turbo v4 step600 EMA and Realism
People files used by the offline tooling. Runtime Turbo recognition is by SHA-256
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`;
recognition does not select the step count or alter inference settings.

## Host correctness and fault coverage

The native host suite has **37 passing tests**, also passing AddressSanitizer and
UndefinedBehaviorSanitizer on both macOS and Linux. It requires no model download or GPU
allocation. Coverage includes:

- Native, PEFT, Diffusers Q/K/V, and unambiguous ComfyUI names; BF16/F16/F32
  adapters; scalar/nested alpha metadata; rank conflicts; orphan/extra tensors;
  malformed descriptors, duplicate JSON keys, unsupported dtypes and shapes.
- Ordered repeated adapters, negative/fractional/zero strengths, one final BF16
  rounding, ties/subnormals/overflow, non-finites even at zero strength, and
  exact untouched bytes.
- Independent rank-order arithmetic at row/column/rank tile boundaries,
  different memory budgets, caller rounding/flush-mode restoration, and
  binary64 subnormal scale identity under caller flush modes.
- Independently encoded canonical cache keys, recipe-domain separation,
  equivalent decimal strengths, moved inputs, same-size/same-mtime changes,
  both modes, and folded parents.
- Content corruption, truncation, missing files, symlinks, traversal, malformed
  manifests, read-only hits, overlapping source/cache paths, and changed sources
  during folding and publication.
- Same-key/different-key concurrent writers, reader leases blocking repair,
  cancellation, process death at preparation stages, short writes, and injected
  selection-allocation and arithmetic-allocation failures, copy/permission/lock/
  write/ENOSPC/fsync/rename/repair failures, symlinked input isolation, and the
  checked full-copy fallback when cloned shards lack reported CoW headroom,
  including delayed quota release and cancellation while waiting for it.

The sampler host suite passes **1,525 checks**, including effective-transformer
substitution, unchanged logical names, equivalent physical/offline trees, and
changed transformer/shared-asset rejection. Runtime effective-model fingerprints
read file content directly: the CUDA node reproduced same-size writes with
identical ctime/mtime values, which are insufficient for persistent hash reuse.
Optional CPU SHA acceleration preserves the digest bytes. Existing no-LoRA and
offline fingerprint policy remains unchanged. Existing regressions passed:

| Suite | Result |
| --- | --- |
| Offline LoRA | 36 tests |
| Sampler CLI | 41 cases |
| Sampler containers | 16 tests and 135 adversarial cases |
| Continuation host | 103,204 checks |
| Bridge host | 16,544,096 checks |
| Preview VAE host/CLI | Host contract suite and 3 CLI tests |
| Quantization CLI | 6 tests |
| Model layout | FL2VA-only, Ref2VA-only, combined, missing modes, incomplete installation |
| Runtime LoRA CLI | 24 cases; `--info` and invalid requests created no cache |

Logs are retained under `outputs/runtime-lora/local/`, with transient build and
sanitizer logs under `/tmp/h3-runtime-*.log` during development. Real-backend
acceptance is tracked separately below; host fault injection is not a claim of
exhaustively testing every operating-system or filesystem failure.
Clang static analysis of the native folder and JSON parser is clean after
checking JSON file lengths once and validating both seek operations.

## Real-model arithmetic

The NumPy oracle independently recomputes selected complete rows from the
original BF16 model and ordered adapters. It checks every selected element
against FP32/BF16 error bounds and optionally compares the existing offline
fold. This is sampled numerical comparison, not an exhaustive NumPy comparison
of every matrix element. For the combined case, the offline planner, ordered
NumPy deltas, and final `round_bf16` conversion are evaluated directly on sampled
rows, avoiding a second full transformer tree. Maximum sampled BF16 difference
from this offline recipe is zero in all five selections.

| Selection | Mode | Coverage and result |
| --- | --- | --- |
| Turbo 1.0 | FL2VA | 259 targets, 1,036 rows, 7,257,600 elements; passed bounds; sampled BF16 bytes equal offline |
| Turbo 1.0 | Ref2VA | All 259 targets sampled; passed bounds; sampled BF16 bytes equal offline |
| Realism 1.0 | FL2VA | All 104 targets sampled; passed bounds and offline comparison |
| Realism 1.0 | Ref2VA | 104 targets, 416 rows, 2,609,152 elements; sampled BF16 bytes equal offline |
| Turbo 1.0 + Realism 0.6 | FL2VA | 259 combined targets, 1,036 rows, 7,257,600 elements; sampled BF16 values exactly equal the offline NumPy arithmetic and BF16 conversion |

Sampled native BF16 versus the offline FP32 accumulator:

| Selection/mode | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: |
| Turbo FL2VA | 0.000974864 | 0.000340792 |
| Turbo Ref2VA | 0.000974864 | 0.000340828 |
| Realism FL2VA | 0.002978683 | 0.001067495 |
| Realism Ref2VA | 0.002978683 | 0.001067428 |
| Turbo + Realism FL2VA | 0.002158880 | 0.000574591 |

These errors include the intentional BF16 rounding. All sampled elements stayed
within the independently computed FP32/BF16 bound; the largest bound fraction
was below 0.991.

There is also an **exhaustive cross-platform output identity check** for Turbo
FL2VA: all **16 files, 66,280,525,110 bytes**, have matching SHA-256 inventories
between macOS ARM64/Clang's initial scalar fold and Linux x86-64/GCC's SIMD fold.
This includes the deterministic in-transformer manifest. The common key is:

```text
39f92869e1dc61dcb4c7fc65a4f2c070d9e422864815219fa5e2d7e764408890
```

Evidence: `outputs/runtime-lora/cross-platform-turbo-fl2va.json`,
`outputs/runtime-lora/cuda/turbo-fl2va-manifest.json`, and the local
`*-parity.json` files. Source content was rehashed before publication; unchanged
output ranges were compared byte-for-byte by the native folder.
Realism FL2VA also passed an exhaustive comparison of all 16 output files
between the native Metal-host and CUDA-host folds; see
`outputs/runtime-lora/cross-platform-ordinary-fl2va.json`. Its key is
`da8c7d1a98d543e1fecef5241e6cbc197103ad9bd40edf4881cb7cbc53e3684f`.
The combined Turbo 1.0 + Realism 0.6 FL2VA fold likewise matched all 16 files
across both host platforms (`cross-platform-combined-fl2va.json`), with key
`e29570bd075cce5293b759c6673d3b46573892ed3c991bd84beff7a95a1f24d1`.

## Hardware and measured preparation

Local hardware is **Apple M4 Max, 128 GiB unified memory**, using Metal. CUDA
validation uses the same existing **RTX 5090, 32,607 MiB reported VRAM** node as
previous quantization work. The original models and adapters are retained.

The node initially had insufficient space. With explicit user approval, the
old generated `quant-turbo-05` test fold was removed. Runtime LoRA variants use
its private root filesystem; the separate packed-quantization cache uses the
larger workspace volume. That workspace ignores permission changes and is
therefore unsuitable for the runtime LoRA cache's private-directory contract.
The root filesystem charges reflinks against its logical quota and releases
that quota asynchronously after unlink. Validation exposed both behaviors.
The folder replaces modified staged clones with independent full copies when
needed, allowing up to five seconds of cancellable waiting for released quota.
This fallback never removes source files or published cache entries.

| Measured operation | Preparation | Details |
| --- | ---: | --- |
| Initial Turbo FL2VA, M4 Max | 742.46 s | Initial scalar kernel; input hash/plan 23.50 s, final verification 22.28 s |
| Turbo Ref2VA, M4 Max | 538.28 s | SIMD/tiled kernel; hash/plan 24.58 s, final verification 22.48 s; conservative owned scratch 8,958,576 bytes |
| Realism Ref2VA, M4 Max | 270.45 s | 15 APFS clones; fold 125.21 s; input/untouched validation 58.78 s; verification 34.55 s; owned scratch bound 8,489,828 bytes |
| Realism FL2VA, M4 Max | 278.17 s | 15 APFS clones, no full copies; fold 129.85 s; input/untouched validation 65.43 s; verification 31.15 s; owned scratch bound 8,489,798 bytes |
| Turbo + Realism FL2VA, M4 Max | 625.80 s | 15 APFS clones; fold 506.15 s; input/untouched validation 45.03 s; verification 23.25 s; owned scratch bound 9,408,582 bytes |
| Turbo FL2VA, 5090 node CPU | 809.70 s | Full-copy storage path; hash/plan 31.83 s, final verification 31.67 s; owned scratch bound 8,958,576 bytes |
| Turbo FL2VA, corrected 5090 quota fallback | 787.34 s | 13 full shard copies + 2 metadata clones; copy 20.97 s, fold 595.90 s, input/untouched validation 75.92 s, final verification 33.54 s; owned scratch bound 8,983,756 bytes |
| Realism FL2VA, 5090 node CPU | 290.52 s | 13 full shard copies + 2 metadata clones; hash/plan 30.09 s, copy 22.07 s, fold 110.35 s, input/untouched validation 59.64 s, verification 33.65 s; owned scratch bound 8,490,188 bytes |
| Turbo + Realism FL2VA, 5090 node CPU | 758.37 s | 13 full shard copies + 2 metadata clones; hash/plan 31.19 s, copy 21.99 s, fold 576.95 s, input/untouched validation 60.55 s, verification 33.80 s; owned scratch bound 9,408,972 bytes |
| Turbo warm hit, M4 Max | 52.75 s | Input hash/plan 26.18 s plus full output verification 26.58 s |

Cold timings include input and output reads, validation, and publication. These
are individual development integration runs, not controlled throughput benchmarks;
filesystem cache state and concurrent host validation can affect wall time. The
initial scalar measurement predates the column-vectorization improvement and is
not a current-kernel performance estimate. Outer cache manifests record detailed
phase times for entries built after timing instrumentation was added. Logical
transformer size is about 61.7 GiB; physical CoW growth depends on modified pages.
The original isolated Turbo fold on the CUDA node had **11,336 KiB peak RSS**
(`turbo-fl2va-fold-measurement.json`), separate from later GPU inference.
The corrected full-copy CUDA fold again matched every file hash in the Metal
variant (`cross-platform-turbo-final-cold.json`), independently of copy strategy.
Its matched 256×256, 56-frame, eight-step BF16 generation pair took **914.10 s
cold / 199.56 s warm**, with identical AV states. Warm preparation was 66.01 s.
Denoising took 83.295 / 89.993 s, including 82.861 / 89.544 s reported streaming
copy time. Individual-process peak RSS was 1,074,696,192 / 1,074,216,960 bytes;
both runs reported 1.822 GiB peak GPU allocation. These default-CUDA runs stream
the BF16 weights and are sensitive to storage/page-cache behavior.

Metal Realism Ref2VA generation passed at 128×128, 22 frames, 20 steps.
Turbo 1.0 + Realism 0.6 repeated Metal generation passed at 256×256, 22 frames,
eight steps in 86.52 / 86.32 s; both AV states share SHA-256
`b7407787bf34dfb4268634cf44d698487dc4763f37c58d1527dbc24e509e4455`.
The same combined selection passed cold/warm default CUDA BF16 generation in
**887.65 / 205.26 s**, with AV SHA-256
`331fb9ac4134a195283207931f60c2eb1f0beaf4c69bfda189be9ce2806b0588`.
Preparation was 758.37 / 66.81 s; denoising was 86.160 / 95.385 s, including
85.716 / 94.950 s reported streaming copy time. Individual-process peak RSS was
1,073,610,752 / 1,074,163,712 bytes, with 1.679 GiB reported peak GPU allocation.

The matched Metal Realism cold/warm generation pair used 256×256, 22 frames,
20 steps, seed 42, and preview VAE. The cold run took 248.07 s, including
193.07 s preparation (88.00 s arithmetic) and 32.304 s denoising. The warm run
took 103.36 s and produced the identical AV hash
`3385dd4af2d88392c5fa14761d7db44949ea0ec7c0039d3601a4c06509c9b7bf`.

The same Realism canvas, frame count, steps, and seed passed cold/warm default
CUDA BF16 generation in **529.54 / 322.76 s**, producing identical AV SHA-256
`406448693752ec3cc3346b3516066be995f5cefb926c219c754093e9f681e321`.
Preparation was 290.52 / 63.72 s; denoising was 196.412 / 216.200 s, largely
streaming copy time. Individual-process peak RSS was 1,206,931,456 /
1,206,517,760 bytes, with 1.896 GiB reported peak GPU allocation. The final CUDA
executable used for ordinary/combined acceptance has SHA-256
`7ef6b80ee907423d48edea7aef836ea65d8d34fe088c39eb9b19f98af9a24df3`.

A Metal Turbo warm render at **256×256, 22 frames, 8 steps, seed 42, preview VAE**
completed in **91.33 s**, including 52.75 s cache preparation, 11.151 s DiT
loading, and 12.999 s denoising. A cold Realism render at the same canvas/frame
count and **20 steps** completed in **347.85 s**, including 278.17 s preparation
and 34.723 s denoising. These are small integration workloads, not visual-quality
or 1344×768 throughput claims.

## Lifecycle and execution acceptance

The local real-model API test passed four requests through one context:
FL2VA, FL2VA, Ref2VA, then FL2VA. The second request hit the prepared DiT cache;
all three FL2VA results had identical canonical latent hashes. Caller-owned
option strings were overwritten/freed immediately after loading, proving context
ownership. Cancellation and saved informational provenance also passed.

Matched eight-step FL2VA runs passed on both backends. On Metal, uninterrupted,
repeated, and pause/resume AV states share SHA-256
`a4629e962ed6dd5f330b3b7bccf2000aca6be4458530277c590f632d8a03e4f0`.
On CUDA they share
`871f9df8f9666eda340e702b2d501010b570fc7bb345e56813ab8b60fec99155`.
Backend-specific hashes are expected; these comparisons hold settings and
backend fixed. Missing LoRA selection correctly rejects the sampler checkpoint. Metal also
rejected changed adapters, strengths, ordered selections, and base configuration
by the exact model-content fingerprint, including reversed zero-strength adapters
whose folded numerical weights are otherwise identical.
Metal Ref2VA uninterrupted and resumed states also match exactly:
`7f13192a7b0e82f68c739a7a8102c098efb426425516d202085782915ec5219b`.
After the filesystem/fingerprint fixes, a fresh CUDA checkpoint again resumed
exactly (243.38 s to prepare/run/save at step 3, 245.51 s to resume). Missing
selection rejected in 2.46 s. These runs retain the same CUDA AV hash above.

Metal also passed relocated-input generation and cold checkpoint resume after
explicit eviction of its generated Turbo variant. The rebuilt transformer's
complete file-hash inventory matched the pre-eviction inventory, and both runs
retained the original FL2VA AV hash. Rebuilding and resuming took 574.48 s;
fold preparation accounted for 523.62 s, including 406.08 s of arithmetic.

CUDA moved-input generation also retained the original AV hash and verified
cache key (213.32 s). After exclusive-lease eviction, the cold checkpoint resume
rebuilt the entry and produced the same AV state in **1,099.59 s**. Preparation
took 927.72 s: 22.85 s copying, 744.77 s arithmetic, 60.99 s input/untouched
validation, and 33.69 s final verification. All 16 rebuilt file hashes matched
the pre-eviction inventory. Copy strategy was 13 independent shard copies plus
two metadata clones, with an 8,984,746-byte owned scratch bound.
Fresh generation using that rebuilt CUDA entry also matched the original AV
state, taking 220.47 s. Thus cold, warm, relocated, evicted/rebuilt resume, and
rebuilt generation all agree within each backend.

Metal hard and bridge AV continuation both passed with the original Turbo
adapter and with Realism replacing it, confirming that the informational LoRA
sidecar does not impose transformer equality on AV continuation.
CUDA also passed both modes with the retained Turbo adapter and with Realism
replacing it. The final changed-adapter runs took 159.36 s (hard) and 158.99 s
(bridge), using 90 frames, four steps, and the saved Turbo baseline AV state.

CUDA fast BF16 repeated results matched exactly. FP8 cold preparation packed
200 matrices; its warm run reused all 200 with zero new preparations and produced
identical AV latents. NVFP4 likewise packed 200 matrices initially and reused
all 200 on the warm run with identical latents. Runtime LoRA reported the same
verified cache key throughout; execution policy changes caused no refolding.

| CUDA small integration run | First run | Repeated run | Repeated AV result |
| --- | ---: | ---: | --- |
| Fast BF16 | 101.70 s | 102.28 s | Exact match |
| Fast FP8 | 153.53 s | 142.44 s | Exact match; 200 packed-cache hits |
| Fast NVFP4 | 141.58 s | 129.47 s | Exact match; 200 packed-cache hits |

These runs use 256×256, 22 frames, eight steps, seed 42, preview VAE, and an
already folded Turbo variant. First FP8/NVFP4 runs include packed-weight creation;
all wall times include runtime LoRA source hashing and output verification.
The final combined Turbo 1.0 + Realism 0.6 NVFP4 run reused **100 unchanged
packed matrices** and prepared **100 changed matrices**, while retaining the
combined native LoRA key. The 400 pre-existing FP8/NVFP4 data files and their
locks kept the same size/mtime inventory; exactly 100 new NVFP4 data files and
100 companion locks were added. The ordinary BF16 runs did not change this
packed cache. Evidence is in `outputs/runtime-lora/cuda/packed-sharing-verified.json`
and the three `*-packed-inventory.json` snapshots.

That combined NVFP4 render passed in **246.51 s**, including 67.55 s native cache
verification/preparation and 68.325 s denoising. Reported GPU peak was 10.380 GiB;
individual-process peak RSS was 1,126,899,712 bytes. Its AV SHA-256 is
`8f4a0614bb8d009a56c58e011485e9f89adcdcd1b46f2968223537e504fb090a`.
It uses the final CUDA executable and exercises the fresh effective-model
fingerprint policy as well as BF16 folding before NVFP4 packing.
The NVFP4 full-VAE run (133.74 s) and paused/resumed run (109.37 s to resume)
produced the same canonical AV state as preview decoding and uninterrupted
sampling:
`af9a5e0a57a123dbdeaed110b42948d88cef9ad30f0905f5cd95bde6d6aa861b`.

The **1344×768, 362-frame, eight-step, seed-42 FL2VA** CUDA smoke passed with
Turbo 1.0, fast CUDA, NVFP4, and preview VAE. It took **812.45 s total**,
including **61.94 s** verified LoRA preparation (32.60 s input hashing and
29.34 s output verification) and **717.763 s denoising**. GPU peak was
**28.790 GiB**; individual-process peak CPU RSS was **11,890,376,704 bytes**.
All 200 packed matrices were cache hits, with zero new preparations. Attention
accounted for 615.814 s in the CUDA category profile, while GEMM accounted for
42.035 s. This explains why weight quantization alone leaves most of this
workload's denoising time intact; it is not an attention acceleration benchmark.
The canonical AV SHA-256 is
`d75490da74fe6cf95e78fbefe43ba7501d7256b6a3151c6d59244c0c008e78ea`.
This uses the requested canvas/schedule/execution settings with the acceptance
driver's prompt and FL2VA mode; it does not modify or execute `testf1.sh`.
The large run used executable `fb544c90ad3a9b96ded9f6a4c388b1c0358c54aa9150240603e68ed1e76b63c4`
before the final always-fresh effective-model fingerprint hardening. Final-build
quantization requests include that additional content read. Treat 812.45 s as
the recorded development smoke result, not a final-build startup benchmark;
the final combined NVFP4 case exercises the hardened path at the smaller canvas.

Legacy no-LoRA and offline-Turbo complete/pause/resume workflows passed on
Metal. The offline-folded model correctly rejected a runtime-fold checkpoint
whose deterministic identity manifest differs. A deliberate media-publication
failure on CUDA occurred after successful folding, denoising, and decoding; the
folded cache manifest remained unchanged and no completed media was published.
Invalid preview weights were separately rejected before LoRA preparation.

The final Metal fingerprint workflow passed baseline/repeat, pause/resume,
missing-selection rejection, and the late media-publication failure. Its
executable SHA-256 was
`4a1f925b3b946b0204e1f5e1ad6f62cefb869afbbca87cf2312f5590c642d710`.
Baseline/repeat took 102.39 / 102.57 s and retained the original FL2VA AV hash;
the baseline's individual peak RSS was 45,439,303,680 bytes. The final local
build, after the quota-wait and JSON error-path fixes, also passed all 1,525
sampler checks and 24 runtime CLI cases, with no compiler warnings. Its SHA-256
is `40ab0734372c4181c8dd467e1c91dddab024cf8470a0d35429adaa1cbac02b63`.
Those two later host-only error paths are covered by the native/sanitizer suites;
the CUDA cold-copy acceptance exercises the quota wait on the real filesystem.

Ref2VA GPU generation, mode switching, and exact resume were exercised on Metal;
CUDA GPU acceptance uses FL2VA. Native Ref2VA arithmetic for both pinned adapters
is covered separately above. This is not a claim that every matrix combination
was run through every backend and decoder.

All required acceptance gates are complete. Generated CUDA logs, argument and
timing JSON, cache manifests, and packed-cache inventories are retained locally
under `outputs/runtime-lora/cuda/`; the final coordinator records successful
completion in `cuda-remaining-complete.json`. Per-run executable hashes distinguish
development measurements from final-build checks.

No adapter quality or subjective equivalence conclusion is implied by these
arithmetic, cache-integrity, and lifecycle checks.
