# CUDA change validation

Before publishing code or reports, follow the [publication privacy notes](docs/publishing.md).

Run the complete CUDA reference regression after every coherent code change,
including changes to generators, build flags and test tooling. Fix failures and
rerun the whole test. Documentation-only edits need document/link validation.
Unavailable hardware, missing models, timeouts and stale artifacts are not passes.

The test runs the current inference code with historical media tools and compares
its output directly with the recorded golden hashes in
`tests/cuda_reference/manifest.json`. The 12 recorded input fixtures are bundled
in `tests/fixtures/cuda-reference`; all 204 expected output hashes are retained.
Do not regenerate golden outputs to hide a regression. The test has no
case-selection, tolerance or skip switches.

Configure your CUDA build and media libraries using the [CUDA reference
guide](docs/cuda/cuda-sglang-reference.md). The runner inherits the calling shell's
build/runtime environment for CUDA, CUTLASS and cuDNN. The gate explicitly
selects its verified test-only FFmpeg 4.2.2 encoder and 6.1.1 decoder; normal
rendering and current-feature tests use FFmpeg/FFprobe 9.0.2. Exact comparisons require compatible library versions; a different
installation path does not require changing the recorded golden state.

From the checkout root:

```sh
python3 scripts/setup_reference_media.py
export H3_REFERENCE_MODEL=/path/to/MiniMax-H3
export H3_REFERENCE_REGRESSION_OUT=outputs/cuda-reference-regression/run-001
make test-cuda-reference-regression
```

`H3_REFERENCE_MEDIA` may point to a relocated `profile.json` from that installer.
Packaged validation obtains its hash-bound profile from `linux-validation/media`;
historical codecs are excluded from the standalone executable. The gate checks
all selected tools before and after execution and never falls back to PATH.

The default model path is `models/MiniMax-H3` in the source checkout. The bundled
fixtures are used by default; `H3_REFERENCE_FIXTURES` may select another copy.
The runner also accepts explicit paths without going through Make:

```sh
python3 tests/cuda_reference_regression.py --source . \
  --model /path/to/MiniMax-H3 \
  --out outputs/cuda-reference-regression/run-002
```

Use a fresh output directory. The runner verifies fixture contents, creates an
isolated build, and records source, binary and golden-manifest identities. It
rejects source or fixture changes during the run. Its test deadline is 12 minutes
after build. It includes one complete 640×480 / 124-frame / six-step video, plus
bounded component probes. No model weights are hashed. The final result passes
only when every expected artifact is present and matches its recorded hash.

Run the CPU-only runner checks with `python3 tests/test_cuda_reference_gate.py`.
Current CLI/state round-trip and rejection tests and explicit attention/precision tests supplement the
reference regression. The [single-pipeline design](docs/cuda/design-single-pipeline.md)
describes the pipeline; the [state contract](docs/features/current-state-contract.md)
defines supported saved formats; the [M0 record](docs/cuda/two-modes-m0-m1.md) describes
reference coverage.

The retained current-feature suite is separate from the historical numerical
contract. With the model symlink installed, `make test` runs host/API, container,
CLI, tokenizer and platform operator coverage. `make test-current-host` is the
host subset. On the qualified CUDA build (Sage, SOL and SubBlock enabled), run
real-model integration, placement, conditioning and decoder recovery probes:

```sh
export H3_MODEL_DIR=/path/to/MiniMax-H3
export H3_CUDA_CURRENT_OUT=outputs/current-cuda/run-001
export H3_CUDA_CURRENT_STATE=/path/to/fresh-current-cuda.h3av
export H3_CUDA_COW_TMP=/path/on/reflink-filesystem/h3cli-test-scratch
make test-current-cuda
```

The state must have a matching current presentation sidecar. Use a fresh output
directory. `tests/current_cuda_suite.py` records every supplementary command,
status and duration; `tests/legacy_cleanup_render.py` defines the bounded
cleanup render matrix. Archived MLX/Metal and pre-SGLang CUDA golden targets
are removed; their numerical baselines are not prerequisites of the current
suite. The immutable 204-output gate remains mandatory and separate.
