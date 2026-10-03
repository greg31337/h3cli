> Historical kernel/qualification guide. M3B/M4 remove `--fast-cuda` and use
> [one CUDA pipeline with explicit attention/precision options](design-single-pipeline.md).
> Commands and old acceptance statements below describe their recorded recipes.
> Use the current design for supported commands and the
> [single-pipeline report](single-pipeline-results.md) for new comparisons.
> Explicit approximate options may degrade quality; default reference parity remains frozen.


# CUDA attention policies

`--cuda-attention default|sage2++|sage3` selects the main DiT attention
implementation. The default preserves the existing cuDNN/native routes.
The two Sage modes are unqualified, opt-in native CUDA implementations for
RTX 5090 / SM120. See [validation results](attention-validation.md) for measured scope
and quality status. Compilation alone does not qualify another GPU.

| Selection | Main attention arithmetic |
| --- | --- |
| `default` | Existing BF16 cuDNN/native attention |
| `sage2++` | Per-thread INT8 Q/K, FP8 P/V, FP16 partial accumulation with FP32 buffering |
| `sage3` | NVFP4 Q/K/P/V with block scales, K/Q smoothing and FP32 score correction |

All modes consume the existing BF16 post-normalization/RoPE Q/K/V and produce
BF16 outputs. Text refiners, Qwen, reference encoders and VAEs retain their
existing attention implementations. Metal attention is unchanged.

Attention quantization and projection-weight quantization are independent.
For example, `--cuda-attention sage2++ --cuda-denoise-quant nvfp4` combines
Sage2++ attention with NVFP4 denoiser projections. Changing only
`--cuda-denoise-quant` leaves attention arithmetic unchanged.

## Build

Use the normal native CUDA dependencies, CUDA 12.8 or newer, and an existing
CUTLASS v4.2.1 checkout at commit
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`. Imported SageAttention sources are
pinned at `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`.

```sh
make clean
make -j3 CUDA_ARCH=120 CUDA_SAGE=1 \
  SAGE_CUTLASS_PATH=/path/to/cutlass \
  CUDA_CUDNN=1 CUDNN_FRONTEND_PATH=/path/to/cudnn-frontend/include
```

The build checks the imported source and CUTLASS header hashes and compiles
Sage instruction families separately for `sm_120a`. Omit the cuDNN options
for a native-only default backend. Ordinary builds omit `CUDA_SAGE=1` and
retain guarded capability errors for explicit Sage requests. Clean when
changing optional build flags.

Inference links native CUDA libraries, including cuBLAS and the CUDA driver.
It does not import Python, PyTorch or Triton, invoke a Python worker, or
download dependencies. Python/Torch/Sage packages belong only in the
isolated test-oracle environment. Dependency licenses, file hashes and local
adaptations are listed in
[the source inventory](../../third_party/sageattention/README.md).

## Run and observe

```sh
./bin/h3cli -d /path/to/model --fast-cuda --cuda-attention sage2++ \
  --cuda-denoise-quant nvfp4 --cuda-denoise-quant-cache /path/to/packed-cache \
  --width 1344 --height 768 --frames 362 --steps 8 \
  --lora /path/to/turbo.safetensors:1 --lora-cache /path/to/private-lora-cache \
  --preview-vae --preview-vae-model /path/to/taeh3.safetensors \
  --profile -p 'A person walks through a sunny park. Birds sing softly.' \
  --save-av-state result.h3av -o result.mp4
```

Use eight steps with the Turbo adapter; retain the model's ordinary schedule
when Turbo is absent. Ref2VA references and AV continuation use the same
attention option.

The runtime reserves at most 512 MiB of additional attention workspace
before optional weight-cache admission. It groups heads and, where needed,
queries using a shape-dependent fixed plan. Every query still attends to
all valid keys. Sage3 never allocates the full approximately 19.43 GiB
production score-correction tensor.

`--profile` reports Sage dispatches, head-group count, high-water workspace,
reduction, packing, correction, kernel and output-conversion times. The
normal attention category includes the complete operation. Compare complete
denoising time as well: extra workspace can reduce cached weights, and short
sequences may not benefit.

## Saved state and compatibility

Prepared DiT contexts include attention policy, recipe and arithmetic plan
in their identity. Default identities and LoRA/packed-projection cache keys
remain unchanged. Activation packs are regenerated for each evaluated block.

Non-default sampler checkpoints contain required extension 35 with attention
policy, recipe and plan. Exact resume restores this selection. An explicit
conflicting `--cuda-attention`, an unavailable Sage build/device, or
`--resume-default-cuda` on a Sage checkpoint fails. To change attention for
a new segment, use a completed `.h3av` state with `--continue-from`.

Completed states record attention in presentation sidecar version 3, with
projection provenance when selected. Readers still accept versions 1 and 2.
Decode replay uses saved latents and does not require Sage kernels or DiT
weights; decoder assets are still required. Do not pass `--cuda-attention`
to a decode-only command.

## Failures and validation

Explicit Sage requests fail before model loading/folding on Metal, an
unavailable build/device, or conflicting `H3_CUDA_REFERENCE=1` /
`H3_FAST_CUDA_ATTENTION` overrides. Unset the latter or use `auto`.
Unsupported main-attention shapes, insufficient workspace, non-finite inputs
and CUDA launch failures are errors; they do not switch to another backend.

The native harness checks both output layouts, tails, input immutability and
output guard regions. `tests/attention_oracle.py` compares native outputs
with pinned upstream kernels and independent blocked FP32 SDPA.
For short cases, its temporary native interchange files use at most 224 MiB
of `/dev/shm` when at least 512 MiB is free. Results, logs and checksums stay
in the output directory; long cases and unavailable RAM scratch use disk.
`tests/attention_packs.py` checks both Sage modes' intermediate tensors.
`tests/attention_workflows.py` records real model commands, binary hashes,
dispatch/timing logs, per-process RSS/VRAM, media contracts and resume results.
`tests/attention_qualify.py` runs the short calibration/held-out matrix and
analyzes all 50 blocks at the selected capture seed. `attention_freeze.py`
freezes numerical limits from calibration before any held-out evaluation.
`tests/attention_report.py` aggregates the evidence, preserving failed,
incomplete and user-skipped results separately from passed checks. The final
qualification record identifies checks stopped by the user at closeout.
Its production summary separates startup, denoising and decoder phases and
subtracts the DiT loading snapshot from cumulative transfer and weight-cache
counters. Peak device memory remains an absolute high-water mark. Cache
warmups are recorded separately from measured target renders.
Projection-quantization snapshots separately record disk artifact hits and
compressed weight streaming. The existing transfer-event/source-upload timers
omit compressed uploads on the compute stream even though byte counters
include them; use denoising/process wall time for their inclusive cost and do
not calculate compressed-upload bandwidth from those timers.
The qualification drivers fix `H3_FAST_CUDA_GEMM_TUNE=0` so independently
timed projection choices cannot confound attention comparisons or exact resume.
They select GPU-state Euler by default; `--sampler cpu` adds CPU-state coverage.
Keep numerical port parity separate from approximate-attention drift and
human video/audio review.

`tests/attention_review.py --root /path/to/evidence` creates an offline paired
playback gallery for completed held-out, target-setting and full-VAE clips,
with links to continuation playback pages. Reviews start as
pending, bind to clip hashes, and can be exported as JSON. The gallery does
not turn numerical results into a perceptual pass. Add `--skip-human-review`
to generate playback pages without verdicts, notes or completion checks.
The bounded mixed-recipe
study uses `tests/attention_sensitivity.py` and its separate diagnostic patch;
plan 9001 is not a supported runtime policy.
