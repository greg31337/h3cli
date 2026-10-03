# LoRA validation

Validation machine: Apple M4 Max, macOS. No M5 results are claimed.

## Implementation coverage

| Tasks | Implementation / evidence |
| --- | --- |
| T001–T005 | Isolated NumPy/safetensors tool, ignored environments/downloads/output, strict CLI and punctuation/scale tests. |
| T006–T023 | Header-only shard indexing, independent safetensors validation, canonical A/B targets, native/PEFT/Diffusers/ComfyUI mappings, exact QKV slices, metadata scaling, strict extra-tensor rejection, PDD diagnostics, inspection and dry run. |
| T024–T030 | APFS `clonefile`, explicit full-copy fallback, canonical path guards, protected overwrite, staging/finalization rollback, byte-preserving writes. Synthetic fault tests cover numerical, write and final-rename failures. |
| T031–T041 | Bounded row blocks, FP32 update accumulation and one BF16 rounding, independent scalar BF16 oracle, runtime-style pre/post probes, exact output tensor hashes and comparison of every unchanged byte range. |
| T042–T048 | Adapter/base SHA-256 identities, manifest, hash-selected or explicit runtime profiles, unknown ordinary adapters supported. |
| T049–T057 | Synthetic test suite: naming conventions, multiple adapters/scales, alpha/rank, malformed payloads, orientation, duplicate/unknown targets, CoW/fallback, source immutability, output cleanup and rollback. |
| T058–T066 | Full-checkpoint integration suite in `tests/integration.py`; passed on M4, with results below. Optional `.h3av.lora.json` provenance keeps AV continuation independent of transformer identity. |
| T067–T090 | README environment/licensing/download/fold/assembly/generation/A-B examples, separate ordinary/Turbo usage, Ref2VA, multi-adapter caveats, provenance, disk and precision guidance, troubleshooting and scope. Top-level README links to the guide. |

## Synthetic and existing regression tests

`make test-lora`: **36 tests passed**, including parametrized subcases.
`make test`: **passed**; ten existing optional fixture groups skipped because their
external reference fixtures are not installed. The suite exercised memory,
GQA/scaling/race, video posterior, Ref2VA layout, sampler/preview, continuation,
bridge, tokenizer, AudioVAE primitives and FFmpeg muxing. No C/Metal inference
implementation changed for LoRA support.

Logs: `outputs/lora-validation/unit.log`, `build.log`, `existing-tests.log`.
After changing probe statistics to stream instead of retaining all output vectors,
the full-sized block-0 QKV matrix was folded again for both published adapters:
BF16 bytes matched the full folds exactly, and metrics agreed within `1e-12`
(`final-math.log`).

## Published adapters and full folds

Pinned inputs:

- LarryVRH v4 step-600 EMA, repository revision
  `43a74557ac3f6539db8e0f2a959d03feb7a81480`, SHA-256
  `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.
- fal Realism People, repository revision
  `039cc8579d7aa357a882d7f4111b25da4f72dccc`, SHA-256
  `acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e`.

All folds used scale 1.0, native scale 1, APFS clones, the default 512 MiB planned
matrix working set, and unchanged released base shards. Every target matched;
zero tensors were ignored. Turbo's 51 rank-16 and 208 rank-64 pairs modified 259
matrices. Realism's 104 rank-32 pairs modified 104 matrices in each mode. Each
fold touched 13 shards.

Worst per-matrix post-BF16 probe metrics (different columns can come from
different matrices):

| Fold | Max absolute error | Max mean absolute error | Max relative L2 | Minimum cosine |
| --- | ---: | ---: | ---: | ---: |
| Turbo FL2VA | 0.001476 | 0.00009912 | 0.0009794 | 0.999999520 |
| Realism FL2VA | 0.001301 | 0.00018493 | 0.0015051 | 0.999998867 |
| Realism Ref2VA | 0.001275 | 0.00018470 | 0.0015047 | 0.999998868 |

All FP32 and on-disk BF16 bounds passed. Headers, tensor declarations, ordering,
offsets, file lengths and unmodified bytes were preserved. The full-fold logs
are `fold-turbo.log`, `fold-realism.log`, and `fold-ref2va.log` in the validation
output directory; individual manifests contain every matrix's metrics and hashes.

## Full-model execution

`make test-lora-models`: **passed** on M4. Generation settings: seed 72,
128×128, 56 frames, all 50 DiT blocks, CPU Euler, no reuse/core reuse,
`inputs/face1.jpg`, and identical prompts within each comparison.

| Check | Result |
| --- | --- |
| Ordinary Realism vs base, 20 steps | Different outputs under otherwise identical settings. |
| Repeated Realism, 20 steps | MP4 and full AV state byte-identical. |
| Turbo, 8 steps, repeated | MP4 and full AV state byte-identical; both audio and video streams present. |
| Separate Realism Ref2VA, 8 steps | Successful image-conditioned generation with the separately folded Ref2VA transformer. |
| Hard and bridge continuation | Both generated valid audio/video streams. |
| Realism AV state continued under Turbo | Passed; VAE compatibility is unchanged. |
| Stop after step 3 of 8, then resume | Final MP4 and full AV state byte-identical to uninterrupted Turbo. |
| Resume Turbo state against base or Realism | Both rejected with model content fingerprint mismatch. |
| Pause preview and per-step denoised preview | Passed. |
| Folded-model memory-floor rejection | Aborted with the expected safety-floor diagnostic and no output MP4. |
| Existing memory and Qwen GQA checks | Passed, including deterministic repeated GPU dispatches. |
| Continuation provenance | Sidecar records state/manifest/base/adapter hashes; state bytes unchanged. |

MP4 SHA-256:

```text
Base / 20:    2024504ba938f1bc7bc9aedbbce925315ffa42d25c0e04550d5d79ca7e794931
Realism / 20: f00955cedecbb8e66f90f29c7ccf66e8afcd77c42d8b9e0343b756b5a59e21e0
Turbo / 8:    612fd926085f81089edf6a37881f99bcc5e2cc5805f23bdfc5462d1f30691e98
```

Detailed timings, fingerprints and state hashes are recorded in
`outputs/lora-validation/integration-results.json`; the command log ends with
`PASS all full-checkpoint LoRA integration tests`.

The 128×128, 56-frame smoke tests exercise compatibility and reproducibility.
A contact sheet (`outputs/lora-validation/style-comparison.png`) shows severe
artifacts in both base and Realism renders at this size. Their deterministic
outputs are not visual-quality acceptance. A separate 256×256 matched-seed A/B
comparison passed for base/Realism at 20 steps and base/Turbo at 8 steps. All four
files have 56 video frames and an audio stream. Sampled frames were inspected:
faces and scenes are coherent, without the severe grid artifacts observed in the
128px tests. The adapters alter the outputs, but this one portrait does not
establish a general perceptual improvement or motion-quality advantage.

| 256px render | Wall time (including process startup/exit) | MP4 SHA-256 prefix |
| --- | ---: | --- |
| Base, 20 steps | 116.44 s | `ebb21c60d3be92fa` |
| Realism, 20 steps | 116.67 s | `284c8c65c3f48890` |
| Base, 8 steps | 65.14 s | `d043aeeedaa32e2f` |
| Turbo, 8 steps | 65.20 s | `150c114bd77398a0` |

The equal-step base/Turbo timings are similar, as expected for ordinary folded
weights. Any acceleration comes from choosing fewer sampling steps, not cheaper
inference kernels. These timings are single runs, not a benchmark.

Artifacts: `outputs/lora-validation/quality-256/results.json`, four MP4s, and
[`comparison.png`](../outputs/lora-validation/quality-256/comparison.png). Rows
are base-20, Realism-20, base-8, Turbo-8; columns are frames 0, 14, 28 and 42.
No production-resolution or combined-adapter visual-quality claim is made.
