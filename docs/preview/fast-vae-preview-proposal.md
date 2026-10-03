Fast VAE previews — research and feature proposal, 2026-09-18

Implementation status: see [the feature guide](preview-vae.md),
[Metal validation](preview-vae-validation.md) and
[RTX 5090 validation](preview-vae-cuda-validation.md). The text below preserves
the original research proposal. Native Metal and CUDA decoding are implemented;
each report records its own hardware coverage and quality acceptance.

Recommend an opt-in `--preview-vae` mode using **Ollin's H3-specific TAEH3
decoder**, together with a decode-only command that can turn saved final
latents into a full-quality video later. This would accelerate both a preview
MP4 and live `--show` images. It would work on CUDA and Metal, independently of
`--fast-cuda`, step count and Turbo adapters. This is a proposal, not an
implemented or locally benchmarked decoder.

The expected tradeoff is cheaper reconstruction with less reliable fine detail.
Its acceptance criterion should be whether a preview usefully represents the
scene, subject, action and timing. Full-VAE pixel similarity is not a quality
gate. Structural correctness, finite output, frame timing and state integrity
remain requirements.

**Why this addresses the current bottleneck**

Our [B200 qualification](cuda-b200-qualification.md) records 4.70 seconds of
fast VideoVAE decoding versus approximately 3.46 seconds of denoising for the
288×384, 56-frame, 20-step pilot. The complete process takes 47.45 seconds,
with substantial model/conditioning startup costs. These are different timing
boundaries: eliminating those 4.70 seconds alone could save at most about 10%
of that cold process. Repeated requests and lower-step workflows can make the
decoder a much larger fraction of elapsed time.

The [current decoder](../../src/vae/video_vae.c) is a 36-layer, width-2048 transformer.
It repeats that work across spatial tiles and overlapping temporal chunks.
Weights and activations are primarily F32; fast CUDA already accelerates its
attention with TF32 GEMMs. A live preview still executes the complete decoder
for one temporal chunk at every spatial tile, then extracts one frame. It
does not execute only a single frame's worth of transformer work.

The existing `--show` path also retains the full decoder while denoising.
A tiny decoder could reduce both preview latency and competition with the
DiT cache. Fewer denoising steps alone do not reduce this final decode cost.

**Candidates and priorities**

| Option | Evidence and tradeoff | Recommendation |
| --- | --- | --- |
| Ollin TAEH3 | Released H3-specific weights and temporal mapping; approximate reconstruction. | First preview candidate. Port only its decoder. |
| Earlier Kijai 2D TAE / independent preview decoder | Available H3 preview alternatives, but less compelling as the main video candidate. | Reserve as comparisons if TAEH3 disappoints. |
| Reduced-precision full VAE | Retains the original architecture; potentially useful for higher-quality output. | Separate second track, starting with decoder-block mixed precision. |
| TensorRT full VAE | Existing H3 integration reports up to 1.7×; adds CUDA-specific engine/build requirements. | Optional experiment after the native tiny path. |
| Batch full-VAE tiles | More concurrency, almost unchanged arithmetic and greater memory demand. | Benchmark only; do not assume a gain on fast nodes. |
| Quantized full VAE | Released INT8 checkpoint exists; requires compatible operators and weight loading. | Later, if full-quality decode remains a priority. |
| Train a new distilled/pruned decoder | Could offer a better quality/speed point, but adds training and evaluation work. | Defer until existing H3 weights have been evaluated. |

[TAEHV's current documentation](https://github.com/madebyollin/taehv/blob/011dfc2112197741c540e0bdd5b7b67bcc930771/README.md)
explicitly lists MiniMax H3. Its advertised approximately 2–3 seconds versus
0.5 seconds benchmark is for **Hunyuan on GH200**, not H3 or our implementation.
It establishes precedent, not a speedup forecast. The project's higher-quality
`super` variants are documented for other models; they should not be treated
as H3-compatible substitutes.

[Kijai's model card](https://huggingface.co/Kijai/MiniMax-H3-TAE) now directs
users to Ollin's newer weights. Another
[independent H3 preview decoder](https://github.com/simsim9-stack/ComfyUI-MiniMaxH3-PreviewOverride)
reports training on roughly 100 generated videos. Similar filenames do not
establish identical architectures, training targets or temporal behavior.
[ComfyUI merged TAEH3 support](https://github.com/Comfy-Org/ComfyUI/pull/15695)
on August 18, 2026.

[vLLM-Omni's H3 decoder optimization](https://github.com/vllm-project/vllm-omni/pull/6607)
reports H200 video decode improving from 6.066 to 3.546 seconds, with its
single-GPU end-to-end measurement improving 28.449 to 25.920 seconds. It
combines FP16 decoder-block weight materialization and fused operators.
Our native code already has fused QKV/RoPE, SwiGLU and scaled-add operations,
so its full gain cannot be assumed here. Mixed-precision weights/activations
remain a worthwhile independent candidate.

[ComfyUI-H3VAE_TRT](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT)
reports up to 1.7×. Conversely, the
[batched FastVAE project's own comparison](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE)
takes 11.15 seconds against 10.36 seconds for ordinary decoding. These are
framework-specific observations, not matched h3cli results. Our prior workspace
and attention sweeps also do not justify expecting a large gain from changing
a workspace setting.

[Flash-VAED](https://arxiv.org/abs/2602.19161) reports about 6× decoder gains
on Wan/LTX experiments, while [Turbo-VAED](https://arxiv.org/abs/2508.09136)
studies decoder distillation for mobile execution. Neither supplies evidence
that its published speedups transfer to H3's transformer decoder. No compatible
H3 checkpoint was established from the releases reviewed here; adaptation would
be a training project.

**A concrete proposed workflow**

The following flags are proposed; only `--save-av-state` already exists:

```sh
# Generate a quick MP4 and retain the complete final video/audio latents.
./bin/h3cli -d "$MODEL" --fast-cuda \
  --preview-vae --preview-vae-model models/taeh3.safetensors \
  --ref-image inputs/1.jpg -p 'The person in <Picture 1> waves.' \
  --width 480 --height 640 --frames 56 --steps 6 \
  --save-av-state outputs/draft.h3av -o outputs/draft.mp4

# Later, decode the same result with the original VAE; no denoising rerun.
./bin/h3cli -d "$MODEL" --fast-cuda \
  --decode-av-state outputs/draft.h3av -o outputs/final.mp4
```

`--preview-vae` would select approximate video reconstruction for the final
MP4 and, when requested, `--show` previews. Default output continues to use
the original VAE. Log the selected decoder and include its identity in output
metadata. An explicit tiny-decoder request with missing or incompatible weights
should fail before generation instead of silently performing a slow full decode.

Normal audio decoding and 24-fps presentation remain in use. Preview mode must
not change conditioning, sampling, seeds, original final latents or continuation
state. Reference images/video still use the existing reference encoders; the
tiny encoder is outside this proposal. Six ordinary steps in the example are
simply a preview choice, not a claim that a particular Turbo model is selected.

The decode-only command should load only the components required for decoding,
verify the state's model identity and produce the same delivery geometry as the
generation that saved it. Retain presentation metadata beside the existing
`.h3av`: output size, continuation-prefix trimming and relevant codec settings.
The state contains the complete untrimmed latents, so treating it as an ordinary
new clip would incorrectly restore protected continuation frames. A low-step
sample remains a low-step sample after full decoding; this improves
reconstruction, not the underlying generation.

**Compatibility details that determine whether this works**

Use the pinned [TAEH3 checkpoint](https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors),
revision `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c`. Direct inspection of its
safetensors header found 64 decoder tensors, 9,868,236 parameters and 19,736,472
bytes of F16 decoder weights, approximately 18.82 MiB. The encoder is additional
and unnecessary for this feature. This is weight storage, not peak runtime
memory. The upstream repository uses the
[MIT license](https://github.com/madebyollin/taehv/blob/011dfc2112197741c540e0bdd5b7b67bcc930771/LICENSE).

[FastVideo's PyTorch adapter](https://github.com/hao-ai-lab/FastVideo/blob/430e52154e76b902c3cc17a16b3edc1fad790012/fastvideo/models/vaes/minimax_h3_taeh3.py)
pins the same checkpoint and records SHA-256
`4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13`.
Only the header was fetched in this research, so the whole-file hash has not
been independently verified here. Verify it when obtaining weights for the
prototype. Its separate
[MLX adapter](https://github.com/hao-ai-lab/FastVideo/blob/430e52154e76b902c3cc17a16b3edc1fad790012/fastvideo/mlx_runtime/minimax_h3_taeh3.py)
provides useful Apple Silicon implementation evidence, not proof of native
Metal performance.

The [upstream architecture](https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/taehv.py)
uses small convolutions, temporal feature memory, spatial/temporal upsampling
and pixel shuffle. It consumes normalized diffusion latents and produces RGB
in `[0,1]`. Feeding it the full VAE's denormalized latent input, or applying the
full VAE's RGB conversion afterward, would change contrast and color.

H3 timing needs an explicit adapter. For `T = 5n + 2` latent frames, retain
`17n + 5` output frames. The inspected FastVideo implementation keeps raw
decoder frame indices satisfying `index % 20 >= 3`; this yields 22, 56 and 90
frames for latent lengths 7, 17 and 27. Preserve temporal feature memory across
execution batches and use global frame indices when dropping frames. Do not
reset it at each batch or reuse the generic streaming wrapper's single startup
trim as an H3 implementation. Validate first/last frame alignment and joins
visually as well as counting frames.

For live `--show`, each denoising step is a fresh decode of that step's clean
estimate. Reset tiny-decoder feature memory between steps, clips, cancellation
and requests. Keep sufficient preceding temporal context when showing a middle
frame; decoding isolated, discontinuous latent slices can produce a misleading
preview. Preserve the existing distinction between clean display estimates and
authoritative raw sampler checkpoints, including on-stop preview semantics.

**Implementation order and bounded evaluation**

1. Prototype TAEH3 reconstruction of existing local final AV states before
   adding native kernels. Evaluate faces/body, `1.jpg`, `2.jpg`, motion,
   multi-reference output and an existing continuation pair. Compare the same
   latent arrays through full and tiny decoders. No new denoising is needed.
2. Establish the H3 temporal and color contract against the pinned decoder.
   Check finite frames, frame counts, shape/layout, chunk boundaries and audio
   sync. Human review determines whether quality is useful for previews;
   pixel-error thresholds against the full VAE are not acceptance criteria.
3. Port decoder-only inference to the shared C interface and CUDA/Metal
   backends. Use bounded temporal batches with retained feature memory. Profile
   the tiny network's convolution implementation; a small model does not
   guarantee fast execution with unsuitable kernels. Keep Python/MLX/PyTorch
   as development references, not production runtime dependencies.
4. Add opt-in generation/live-preview selection and decode-only finalization.
   Keep model identity and decoder identity separate, include decoder/policy
   in decoder-cache keys, and size memory headroom for the selected decoder
   rather than always reserving space for the full VAE.
5. Stream completed RGB8 frame batches into bounded encoding buffers. Measure
   first-frame delivery and finished MP4 time. This complements the earlier
   [chunked-output recommendation](minimax-official-review.md), especially once
   decoding becomes cheap; tiny weights do not eliminate full-clip RGB memory.

Start at 288×384/56 frames and 480×640/22 frames, then add a bounded saved-state
case with more spatial tiles if initial results warrant it. Reuse decoder
instances for warm measurements; record cold loading separately. Keep the same
audio and codec settings. Charge decode/replay/mux work to a new explicit
experiment budget rather than the already exhausted B200 qualification round.
Use short per-case timeouts and stop an unpromising candidate early.

Measure video decoder time, load time, audio time, conversion/mux time, complete
operation time and peak host/device memory separately. A reasonable development
target is at least **3× faster video decoding** with useful preview quality;
this is a target, not a measured prediction. For illustration, a 5× decoder
speedup cuts total time by only 8% when decoding occupies 10% of a request,
but by 48% when it occupies 60%.

After full-frame-rate TAEH3 is useful, investigate reduced spatial upsampling
for a cheaper preview tier. Upstream supports disabling upsampling, but H3
frame trimming must be revalidated for temporal changes. Preserve normal
timing/audio in the initial feature. Arbitrary latent resizing, omitted full
VAE layers, larger full-VAE tiles and frame repetition are not substitutes for
a validated decoder; they can distort the very motion and identity the preview
is meant to assess.

This research changed no inference code and ran no new rendering workload.
Existing full-VAE quality approvals do not constitute approval of TAEH3 output.
