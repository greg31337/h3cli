# Latent upscale qualification record

Native implementation, CUDA/Metal qualification and the fixed 14-video comparison
are complete. See the [results](latent-upscale-results.md) and
[completed task list](latent-upscale-tasks.md). The user accepted the results and promoted
latent upscaling to a regular CUDA/Metal feature. The
[acceptance record](latent-upscale-acceptance.json) identifies the reviewed artifacts.

## Reference freeze

The [operator contract](../../tests/upscale/contract.json) pins the LBH source,
BF16 artifact, every tensor name/shape/offset, and normalization constants. The
artifact contains 345,280,216 BF16 parameters in 690,592,992 bytes. Acquisition
verified SHA-256
`4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`
against the pinned Hugging Face LFS identity. The production loader reads
only safetensors. Code and weight notices are separate in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

The unchanged baseline passed all **204 CUDA golden outputs** using CUDA 13.0.3,
cuDNN 9.20 and the qualified SM120 environment. Baseline source identity:
`cfc942a59e984cbf39bb7ff11728a892d50a12c52a95602e5a68575bad860c5e`.
Golden manifest:
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
The baseline inference/check phase took 96.221 seconds; this excludes build.
Local evidence: `outputs/latent-upscale/m0/gate/result.json`. H3 base weights
were identified by existing runtime metadata, not scanned or hashed.

## Domain decision: recipe 1

Pinned ComfyUI commit `4ef23c34d950eecc37040a21ee1741a49d2e44b1` puts
normalization inside `MiniMaxH3VideoVAE.encode` and its inverse inside `decode`.
`MiniMaxH3Video.process_in/process_out` use the base identity scale. The native
decoder's `prepare_input` likewise applies `z*std+mean`. Therefore the adapter
between **sampler-facing** native and ComfyUI video is identity, with shape
`[1,24,T,H,W]`; it is not a conversion back to raw VAE moments.

The LBH wrapper casts input and constants to the compute dtype, subtracts mean,
divides by std, runs the network, multiplies by std, then adds mean. Recipe 1
retains all those operations, despite the additional normalization of an
already normalized sampler latent. Each BF16 tensor operation rounds separately;
the pointwise expressions must not be fused across those boundaries. Network
scale embedding input is `2−1=1`, not zero. Audio `[64,T]` is a byte-preserving
reshape of `[1,32,2,T]`, ordered channel/stereo/time. No audio normalization or
stereo transpose belongs in the upscaler.

[Pinned video VAE](https://github.com/Comfy-Org/ComfyUI/blob/4ef23c34d950eecc37040a21ee1741a49d2e44b1/comfy/ldm/minimax/vae.py),
[pinned audio VAE](https://github.com/Comfy-Org/ComfyUI/blob/4ef23c34d950eecc37040a21ee1741a49d2e44b1/comfy/ldm/minimax/audio_vae.py),
[pinned wrapper](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/40316cf008b2fd8663263270669eb4da23f89d2c/nodes/minimax_h3_latent_upscaler_3d.py).

`tests/upscale_reference.py` is an opt-in fixture generator. It selects just the
pinned network definitions, loads BF16 safetensors, and runs BF16 and FP32
oracles from the same weights. It captures layers and output for T=1, T=2 and
T=27 inputs, including constant, channel ramp, edge impulse, random and a real
clean crop. It also retains paired full real video and stereo-audio domain
fixtures. This is test tooling, never a native inference dependency.

Tolerances were frozen before candidate operators: layer BF16 NRMSE ≤0.015 and
max error/reference RMS ≤0.12; whole-network BF16 ≤0.04 and ≤0.35 respectively.
The denominator is `max(reference RMS,0.1)`. FP32 layer limits are 0.0001 and
0.001. Exact identities (audio, temporal sizes, state integrity and resume) do
not use numerical tolerances. These thresholds do not modify the CUDA golden
regression or imply perceptual acceptance.

The reference fixture generation and a subsequent unchanged 204-output gate
both passed at source `fdfed5b835ec592d9de0978b4653d39332fab97575a2659c2aa105ef6e911787`.
Evidence is under `outputs/latent-upscale/m0-contract/`: `fixtures/manifest.json`,
`oracle.log` and `gate/result.json`. These are the frozen oracle outputs used by the subsequent native comparisons.

## Conditioning decision

The existing engine feeds Qwen the source conditioning pixels, whose spatial
grid can depend on the render canvas. Such embeddings are **not inherently
independent of canvas**. Recipe 1 explicitly freezes the semantic view captured
at source generation: saved BF16 Qwen output, tags, token IDs, position IDs,
vision spans and source semantic grid. It does not claim equality with Qwen
re-encoding a larger reference. This immutable source semantic view is the input
to the later upscale stage; only the raw visual DiT conditions are retargeted.

The source record must distinguish this policy from a target-encoded semantic
view. Record raw visual/audio rows before augmentation; first/last keyframe
indices; reference order/kind, original dimensions, actual source latent grid,
image `match`/`max` policy, video pipeline/frame phase and soundtrack pairing.
Reuse of a conditioning cache that lacks these records requires recovering the
metadata from the still-available original media during capture or rejecting
the capture. Later loading must not reopen those files.

FL2VA anchors enlarge as independent T=1 volumes. Ref2VA image `match` derives
its new native canvas using the saved original dimensions and original-size
cap. Image `max` and intrinsic video grids retain their native policy; unchanged
grids are byte copies. Audio rows remain exact. Visual volumes are unpacked,
transferred and repacked in order; the packed row sequence is never resized.
Augmentation runs once on the transformed raw visual rows. Rebuild DiT positions,
segment offsets and schedule maps against those rows and the frozen semantic
view. Clear all prepared and approximation histories.

## Geometry and resource audit

The explicit larger profile has a 2,088,960-pixel and 1,920-per-axis ceiling,
32-pixel alignment, unchanged frame-grid rules and checked products. Ordinary
generation retains its 1,032,192-pixel cap and adaptation behavior. AV v2 and a
required sampler geometry/stage section must distinguish this profile from old
states. Presentation validation must use the same explicit profile. A dedicated
comparison-only direct render uses that profile; it must not alter defaults.

At 90 frames, video T=27 and audio T=150. The 1344×768 target has 27,216 video
rows; 1920×1088 has 55,080. Each adds 300 audio rows and the actual text/reference
rows. The larger unconditioned QKV activation has about 894 million BF16
elements, below the 32-bit element-index ceiling. Reference/text additions
still require checked admission; a geometric cap is not a memory guarantee.

The 512-channel target upscaler BF16 feature volumes are 106.3125 MiB and
215.15625 MiB respectively. A full graph needs several such buffers plus
658.57 MiB of weights, scalar/embedding buffers and bounded convolution
workspace. Whole-volume normalization and symmetric convolution require full
temporal context. No temporal chunk fallback is permitted. Release this model
and scratch before DiT admission, and DiT before VAE admission. Decoder tiling
already has checked element products; its grid/stitch limits require actual
probes on both canvases and both backends before the campaign.

Bounded qualification progresses from host-only geometry/layout/format checks,
to small operator fixtures, full T=27 upscaler transfer, one/few-step native DiT
probes (at most six evaluations), and full decoder tiles/edges. Record backend
peak allocations and admission failures. A failed or unavailable probe is not
a pass and cannot be replaced by a smaller campaign canvas.

## Comparison freeze

[The manifest](latent-upscale-manifest.json) defines exactly 14 videos: seven
per canvas pair, with a fixed piano prompt, seed 42, 90 frames, 24 FPS and
50-step source/direct baselines. I4/U2/U4 share target noise and sigma 0.25;
U0/U2/U4 share the learned transfer. Charge shared source/transfer work to each
independent-job estimate and retain raw stage timings. The manifest specifies
stage accounting, deadlines, no successful repeats and failure retention.
K=0/2/4 and sigma 0.25 were frozen comparison settings and have now received
user acceptance. The original manifest status describes its initial freeze;
execution state and visual acceptance are recorded separately without editing
that immutable protocol.


## Source capture and initial native implementation

The source-capture patch passed all 204 unchanged CUDA outputs at source
`3619de6f95708995d54c02d622be317460c6457b57e0c80c3dd9398b94f72dcc`.
The bounded CUDA 128×64 / 22-frame / two-evaluation probe saved a clean source
and AV state without decoding. Shared state tests include source roundtrips,
malformed checksummed sections, nonfinite payloads, atomic replacement failure
and nine CLI preflight conflicts. Evidence: `outputs/latent-upscale/m1/`.
Integration and lifecycle checks completed in the subsequent milestones below.

The native graph uses full-context NDHWC BF16 features, F32 reductions and a
256-row convolution packing tile. Explicit packing/output scratch is 7,340,032
bytes, independent of full spatial/temporal volume. The model and operator paths
are isolated from ordinary inference; Python/Torch only produce the oracle.

`outputs/latent-upscale/m2-metal-004/comparison.json` passes all 210 frozen BF16
checks. Isolated block/head checks feed the recorded reference input to each
production layer. The five complete graph outputs have worst NRMSE 0.022122
and max error/reference RMS 0.153304 (edge impulse). The isolated layers have
worst NRMSE 0.001182 and max error/reference RMS 0.099988. Cancellation before,
during and after transfer, reuse after admission failure, two interleaved
contexts and unchanged caller input all passed. Small-fixture peak allocation
was 702,215,856 bytes; this is not a full-canvas resource measurement.

Initial runs are retained. Metal 001 incorrectly applied isolated-layer bounds
to accumulated whole-network intermediates; 002 used the correct isolated
procedure and exposed temporal depthwise bias rounding. Bias must be added
before the final BF16 cast in that operator. Runs 003/004 include the correction.
No oracle bytes or frozen thresholds changed. FP32 oracle captures are
available for diagnosis; there is no production FP32 upscaler claim.

The first CUDA native run passed the ordinary 204-output gate at source
`7c616d1144c8756e6e5f3b8960a84c44b3f6919881020ad7845bc612b6f8f1f2`,
but failed several isolated residual-block maximum-error checks. This is a
failed operator qualification, retained under `outputs/latent-upscale/m2/`.
The accumulation issue was corrected and requalified before higher-resolution
integration, as recorded below.


The CUDA correction disables reduced-precision partial accumulation for the
upscaler convolution GEMMs, using an isolated cuBLAS handle with F32 reductions.
`outputs/latent-upscale/m2b/native/comparison.json` passes all 210 checks;
`m2b/network.log` records the lifecycle probes. The unchanged 204-output gate
passed at source `763f701ef7286924a72ee1bcdc8c4e9302dc08c0e4d4519234b1e094364c3508`
in 96.630 seconds excluding build. The matching Metal 005 fixtures also pass.
The failed CUDA run remains retained; the ordinary GEMM paths are unchanged.


The geometry/retarget patch passes the full 204-output CUDA gate at source
`96e83d94e787a7dacc431659e8baf2ab609089fa9b7c61fb7bb52eb169c8880c`
(95.878 seconds excluding build). Host tests cover the explicit large profile,
AV v2/presentation v7 roundtrips, ordinary cap retention, 90-frame target row
counts, no-reference/first/last/both anchors, ordered image/video/audio
retargeting, exact intrinsic-video/audio copies and stale plans. Existing
210 adversarial sampler cases and 52 sampler CLI cases still pass.

A Metal 128×64 / 22-frame / two-evaluation state-only source capture passed,
including reload of its source bundle and saved AV/presentation pair. The AV
compatibility metadata check took below one millisecond; no H3 weight hashing
or VAE decoding was performed. Evidence: `outputs/latent-upscale/m3-metal/`.
Larger-canvas GPU/decoder qualification is recorded below.

## Refinement and exact resume

The refinement patch retained all 204 CUDA golden outputs at source
`94ca1aba3278bad1197d2fbfc92e266aa63e80531337b28cbb3711102d9cb91f`
(95.825 seconds excluding build). Its first CUDA-specific initialization probe
correctly rejected a boolean in place of the versioned SGLang recipe tag. That
failure is retained under `outputs/latent-upscale/m4/`; it was not a refinement
pass. The corrected patch passed the full gate at source
`1d87911278b1de08ab78d1a7860d5c19f0856d1928ab2838f18fd2ae6b27b1e3`
(98.060 seconds excluding build).

Both backends pass actual-model four-step refinement, fresh-process exact
resume at boundaries 0, 1 and 3, and cancellation/recovery at the same
boundaries. Every callback compares the full AV trajectory and exact clean
audio bytes; every recovered final AV file matches its uninterrupted run.
K=0 passes direct transfer/audio identity with no noise draws or DiT calls.
The initialized boundary-zero checkpoint is saved before transformer loading.

Metal evidence: `outputs/latent-upscale/m4-metal/lifecycle-002/`. Final AV
SHA-256 is `973a7e8a574b1d81135ea3320075c81e2960a3be9fa06992b87125492643b135`.
CUDA evidence: `outputs/latent-upscale/m5/lifecycle/`. Final AV SHA-256 is
`b3205c89fc47e53f57d076a73a462bfeac0aa749f1fe8a14961e4771b4509338`.
These are same-backend exactness checks, not cross-backend equality claims.
The first Metal lifecycle run stopped at a test assertion expecting the word
“cancel”; the existing API reports “latent callback stopped.” The saved
checkpoint was intact. The corrected full run is retained separately.

## Full-canvas probes

CUDA has passed both **1344×768** and **1920×1088** targets at 90 frames. Each
source→upscale→source process uses six evaluations total: two source, two
refinement and two source regeneration. Same-context regeneration matches
the original source AV bytes exactly. The source/target audio decodes match
byte for byte (stereo F32, 120,000 samples/channel at 32 kHz). Full tiled
decoders deliver all 90 frames at 24 FPS and retain the exact target canvas.
Separate one-evaluation direct-profile probes also pass on CUDA.

Metal passes the 1344×768 probe with the same isolation, PCM, full-decoder and
direct-profile checks. Its learned transfer took 8.699 seconds including output
handling, with a tracked GPU tensor peak of 1,037,558,448 bytes. Full decoder
checks took 37.290 seconds for the source and 124.962 seconds for the target.
These bounded two-step states are resource/correctness probes, not comparison
quality or 50-step performance observations. Both Metal canvases now pass. The 1920×1088 transfer took 17.579 seconds;
its source/target full decodes took 68.250/266.556 seconds. The complete
source→upscale→source probes took 652.684 seconds for the smaller pair and
1,712.068 seconds for the larger pair. Separate one-evaluation direct probes
took 186.347/550.180 seconds. Evidence:
`outputs/latent-upscale/m5-metal/probes-001/result.json`.

The qualified Metal device is an Apple M4 Max with 128 GiB unified memory
(`m6a-metal/device.txt`, header-only model inventory). Metal peak process
footprint was 47,902,126,112 bytes at the smaller target
and 93,066,702,928 bytes at the larger target; peak Metal allocated bytes were
47,400,419,328 and 95,064,309,760 respectively. These measurements require
substantial unified-memory headroom and do not qualify smaller-memory Macs.

CUDA first/last/both-keyframe sources survive decoder cancellation, moving the
bundle and removal of the original reference files. Learned T=1 retargeting and
two refinement steps then pass with unchanged generated audio. The first
conditioning orchestration attempt lacked copied media in the isolated build;
restoring the unchanged fixtures resolved that setup error. The first mixed
fixture was shorter than the existing two-second native reference-audio
minimum. Its failed attempt is retained; the revised fixture explicitly loops
owned copies to 56 frames without altering the bundled golden inputs.


## Frozen comparison build

The final implementation/tool revision is
`d7d8638cc6b99259ab304b3aae66375cf86bf26f55b45eac2ad24eeb5209ffb6`.
`outputs/latent-upscale/m6a/status.json` records all 204 unchanged CUDA golden
outputs, all 210 native BF16 operator checks, 2,930 sampler host checks, ten
source-file test groups, 24 upscale CLI conflicts, four campaign accounting
tests, two report diagnostic tests, and all 14 actual-model lifecycle runs.
First/last/both keyframes, ordered mixed image/video/audio, `max` image sizing,
decoder cancellation, moved bundles without original media, and cancellation
before source publication pass on CUDA. The previous reader rejects the new
required refinement section. Campaign smoke checks prove learned/bilinear
transfer, persisted shared noise, zero redraws for U2/U4, and native PCM delivery.

The first `max` fixture combined four large references and exceeded the existing
packed-vision capacity; it remains a recorded capability failure. The bounded
`max` sizing check uses one image; the separate ordered mixed case retains all
four reference entries. No runtime limit or golden fixture was relaxed.

Ordinary CUDA host, continuation/bridge/reference layout, posterior, memory,
progress, saved-state, preview and full-VAE suites pass in
`m6a/ordinary-host-002.log`. This includes 210 adversarial sampler-container
cases, 52 sampler CLI cases and 1,270 binary-mask oracle rows. The first attempt
lacked binary reference-layout fixtures in the isolated build; the rerun used
the unchanged bundled fixture files. The final source fingerprint remains exact.


Final Metal evidence is under `outputs/latent-upscale/m6a-metal/`: all 210
operator checks, every reference case, and all 14 lifecycle checks pass. The
final AV and trajectory hashes exactly match the earlier Metal lifecycle run.
`identity.json` binds the binaries to the same final source fingerprint as CUDA.
The ordinary host/preview/full-VAE suites also pass
(`outputs/latent-upscale/work/m6a-metal-ordinary.log`). Both complete full-canvas
probe results remain in `m5-metal/probes-001/`; subsequent changes added campaign
noise sharing and bounded fixtures without changing the qualified canvas math.

The final CUDA golden check took 100.021 seconds excluding build, with CLI
binary SHA-256
`e5c8a972ccbcc918ed150755c78d031b7606c2cb3b25f7b7f032acb596f8a037`.
Campaign dry run: exactly 14 outputs and 220 denoising evaluations. The four
50-step source/direct jobs alone receive that budget; all other subprocesses
retain the ordinary six-evaluation maximum. The fixed comparison starts only
after all backend, canvas and reference qualification gates above have passed.


## Final comparison and closure

All 14 fixed videos and 36 recorded stages passed without a failed campaign
attempt or successful repeat. Both pairs preserve exact source audio through
latent transfer, every refinement transition, native PCM decoding and the
observed AAC outputs. All media retain 90 frames, 24 FPS and exact dimensions.
The [results record](latent-upscale-results.md) contains timings, diagnostics,
frame observations, memory, guidance and limitations. Local evidence is
`outputs/latent-upscale/comparison-001/`; the synchronized report has 42 crops.

`report-audit.json` and `local-audit.json` both pass. All stage artifact hashes,
video hashes, media readability, counts and local links were checked after
copying the evidence. The current source still matches the final CUDA gate's
`d7d8638cc6b99259ab304b3aae66375cf86bf26f55b45eac2ad24eeb5209ffb6`
identity. Later edits changed documentation and publication status only. The
user explicitly accepted this comparison and requested promotion to a regular
feature. The acceptance record binds that decision to all fourteen video hashes
and the original report SHA-256. It is separate from the prior caching reviews.
