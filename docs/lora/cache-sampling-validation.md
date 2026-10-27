# LoRA sampled-cache validation

Validated October 26, 2026 on NVIDIA RTX PRO 5000 72GB Blackwell, with host and
sanitizer coverage also run on macOS arm64. This supersedes the warm-cache
hashing timings and locking behavior in the [September report](runtime-validation.md).

## Behavior

A sampled cache hit reads at most 768 KiB of model, adapter and cached data:
384 KiB for the inputs and 384 KiB for the cached transformer and manifest.
Every file participates. Small files are read completely; large files use up
to 32 windows distributed from the beginning to the end. The lookup metadata
is bounded separately to 4 KiB, keeping data plus that record below 1 MB.
Warm hits skip full-file hashing and tensor planning.

The disposable `samples/` record maps sampled inputs to the original full-content
fold key. Cold folding still fully hashes and validates inputs and outputs.
A legacy cache is fully verified once to create the record, with no refolding
or weight changes. Missing or invalid records fall back to full verification.
Deleting a sample record forces a full check on the next request.

LoRA ownership checks, private-directory restrictions, lock files and reader
leases are removed. Ordinary filesystem permissions apply. Cache-root aliases
are supported; source/cache overlap and generated-file isolation checks remain.
The caller must serialize cache mutation and avoid changing weights in use.
Sampling can miss same-size changes outside its windows. These limits and
cleanup behavior are documented in the [LoRA guide](../../lora/README.md#cache-and-resource-costs).

## Measured existing-cache comparison

The same existing Ref2VA cache, using one adapter with 208 targets, was used
for both paths. The original cache key remained
`4654b4eb1bd46f3ecb830413fd8d54f9961cd9a65cbcb5bfdf77370a6c747b73`.

| Measurement | Result |
| --- | --- |
| Full input hashing and planning during legacy migration | 168.15 s |
| Full cached-output verification | 149.09 s |
| Complete legacy preparation | 317.26 s |
| Sampled preparation reported by the application/host probe | 0.00–0.03 s, rounded to two decimals |
| Five independent sampled host processes, including process startup | 32.55, 16.22, 7.96, 7.92, 8.04 ms |
| Data sampled on each hit | 716,656 bytes |
| Complete CUDA render with legacy verification | 363.87 s |
| Complete CUDA render with sampled verification | 40.46 s |

Render settings were Ref2VA, 128×128, 22 frames, two denoiser steps, seed 42,
the bundled `first.png` reference image, original VAE decoding, and the same neutral
teapot prompt. Timing includes normal filesystem-cache effects. Sampling speeds
up LoRA preparation; inference still loads and uses the model weights normally.

Both renders produced byte-identical saved AV states:
`973a86bcfc69d7525ce8673fa2445fbbde63df4b30ce848e5b269142ca297ff0`.
All 17 existing cache files retained their sizes, modification times and inodes.
The original manifest remained byte-identical. Each sampled host run enabled
the test-only full-hash failure injection: any full-file hash attempt would
have failed the run.

## Checks

| Check | Result |
| --- | --- |
| LoRA host suite, macOS and PRO 5000 host | 42 tests passed on each |
| Address/undefined-behavior sanitizers, both hosts | Same 42 tests passed on each |
| LoRA CLI validation and model selection | 24 cases passed |
| Sampler state suite | 4,305 checks passed |
| CUDA reference runner host suite | 18 tests passed |
| Complete immutable CUDA reference regression | All 204 output hashes matched |

Host coverage includes the aggregate sample budget, changes at beginning/interior/
end windows across multiple large files, explicit unread-byte blind spots,
legacy migration, malformed lookup records, cache repair, readonly sources and
cache hits, ordinary directory permissions including `2775`, cache-root aliases,
source isolation, cancellation, and ignored staging trees left by killed writers.
CLI fixtures now target missing transformers under the current offline preflight
and check the current empty-cache-argument diagnostic.

The initial Linux runs used ext4 temporary storage and skipped the reflink quota
cancellation case. Both host suites were subsequently rerun with `TMPDIR` on XFS:
all 42 cases, including that case, passed normally and under sanitizers without
skips. The follow-up logs are retained with the main-checkout update evidence.

The CUDA regression completed in 133.53 s after its isolated build, 234.65 s
including that build. No golden hashes, fixtures, tolerance or case selection
were changed.

- Tested source fingerprint: `99240a6e604fa6a669f97af59e6633dff6a253f344631b723bd58ac999917edd`.
- Tested CUDA application SHA-256: `7ab64b925d85832f42964ce2a7fd9adb4ef16ebca4499a27e0dbfe9db3bbf2a8`.
- Golden manifest SHA-256: `fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.

Private command records, failed historical CLI assertions, final passing logs,
state artifacts and the server evidence archive are retained under
`outputs/lora-sampling-20261026/`.

## Reproduction

Use the [qualified CUDA environment](../cuda/cuda-sglang-reference.md) and
existing model/adapter files. No models need to be downloaded for these tests.
Set `TMPDIR` to an existing writable directory on a reflink-capable filesystem
to execute the clone-specific host cases rather than skip them.

```sh
make -j8 all bin/sampler_tests test-lora-runtime test-lora-runtime-sanitize
H3_MODEL_DIR=/path/to/model H3_LORA_ADAPTER=/path/to/adapter.safetensors \
  python3 tests/test_lora_cli.py
./bin/sampler_tests
python3 tests/test_cuda_reference_gate.py
H3_REFERENCE_MODEL=/path/to/model \
  H3_REFERENCE_MEDIA=/path/to/reference-media/profile.json \
  H3_REFERENCE_REGRESSION_OUT=outputs/cuda-reference-regression/lora-sampling \
  make test-cuda-reference-regression
```

After creating or migrating a cache, measure a standalone sampled hit with:

```sh
H3_TEST_LORA_FAULT=full-hash ./bin/lora_host \
  /path/to/model/Ref2VA/transformer /path/to/lora-cache 512 \
  --ref2va /path/to/adapter.safetensors
```

This fault injection exists only in the host test binary. Normal h3cli cache
hits use sampling automatically without additional flags.
