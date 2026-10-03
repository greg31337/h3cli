# MiniMax H3 official implementation and ecosystem review

Reviewed **2026-09-18**, against local h3cli commit `2a4a750`.

The most actionable correctness finding is a **mixed-reference audio-budget mismatch in our code**: video soundtracks consume the allowance intended for standalone audio. The strongest new implementation opportunity is **streaming completed VAE chunks into FFmpeg**, reducing full-clip memory retention and potentially overlapping decode with encoding. Newer distilled adapters also need explicit sampler contracts before we can claim support.

This is a research and source review, with small local geometry probes. No inference implementation was changed and no new model renders were performed. Recommendations extend the separate [antirez/h3.c fork review](upstream-fork-review.md).

## Scope and source versions

| Source | Reviewed snapshot and coverage |
| --- | --- |
| [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438) | Main `d21241f0a4b3acbb34c97dae47fa417b7065e438`; model configuration, original VideoVAE/AudioVAE source, prompt guidance, recent history. |
| Official repository issues and PRs | Screened all 40 issues and 34 PRs returned by the API, including closed items; 31 issues and 5 PRs open, 12 PRs merged. Read relevant discussion and downloaded all PR diffs. |
| Official repository forks | Enumerated 656 distinct public forks across seven API pages. Attempted default-branch comparisons for 25 recently pushed forks plus one large older fork; 25 comparisons succeeded, one returned 404. Inspected substantive changes selectively. |
| [Diffusers](https://github.com/huggingface/diffusers/tree/a3e0b8ec235c27a6c17a21976daf7fd32d819d05) | Main `a3e0b8ec235c27a6c17a21976daf7fd32d819d05`; H3 converter, transformer, scheduler, VAEs, modular pipeline and relevant tests. |
| Linked integrations | Followed concrete H3 issues into ComfyUI/PyTorch and vLLM-Omni, including superseding PRs. These are integration implementations, not MiniMax's unpublished internal pipeline. |

The fork inventory is broad, but this is not an exhaustive audit of every branch in every fork. Selection favored pushes after the official main's last push. GitHub's fork counter differed from the enumerated list; the inventory records observed entries. The large `benriworks` comparison hit GitHub's 300-file listing cap. Unsubmitted changes on older/nondefault branches may have been missed.

The official repository supplies VAE code, configurations and integration guidance; the transformer and runnable framework pipeline require following its downstream implementations. Its public release provides H3-Base. **Context-IR preprocessing and Regenerate-2K are not open-sourced**, and the README still describes MiniMax's native sparse-attention release as future work. Community FastH3/VSA is a separate development, discussed below. Hosted 2K output is therefore not a controlled baseline for the native base model. [Official architecture and release boundaries](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/README.md#model-architecture).

## Recommended work, in priority order

| ID | Priority | Recommendation | Evidence and expected value |
| --- | --- | --- | --- |
| R01 | High | Separate standalone-audio and video-soundtrack input budgets. | Direct local source match to a downstream bug fix; restores valid mixed-reference input combinations. |
| R02 | High for memory; profile speed benefit | Feed committed VAE chunks into a bounded FFmpeg encoder pipeline. | Our full-clip RGB retention is confirmed; downstream H3 implementations now expose and consume chunks. |
| R03 | High before claiming newer Turbo support | Carry artifact-specific task, layout, alpha and dual-sigma schedules through folding and generation. | Published LightX2V artifacts have different sampling contracts; successful tensor mapping alone is insufficient. |
| R04 | Medium, quality | Add optional structured prompt templates and validation using the official task-specific guidance. | Low runtime cost; directly addresses ambiguous reference roles and speaker descriptions. |
| R05 | Medium, quality | Extend short A/V review cases for boundary pops, speaker identity, dialogue leakage and timing. | Multiple upstream reports; no demonstrated native root cause or general upstream fix. |
| R06 | Medium, test reliability | Harden reference-fixture provenance and reject known-bad framework configurations. | MPS padding corruption and checkpoint-layout confusion can invalidate an external comparison. Existing version pins/hashes should be extended, not replaced. |
| R07 | Research | Evaluate a dedicated FastH3 Dense preview profile, then VSA only if justified. | Four-forward students and learned sparse attention exist downstream; significant new model/loader/kernel work, with limited task coverage. |
| R08 | Medium, profile-driven | Prototype reduced-precision VideoVAE execution while retaining released tiling. | Current Diffusers CUDA decode uses FP16 autocast. Strengthens the VAE precision recommendation in the earlier fork review. |

### R01 — Mixed-reference limits currently reject valid combinations

[vLLM-Omni PR #7281](https://github.com/vllm-project/vllm-omni/pull/7281), merged September 11, fixes a request containing an image, a video with a soundtrack, and a separate 15-second WAV. The framework incorrectly counted the soundtrack against the standalone-audio allowance. The fix keeps the two duration budgets separate and preserves both audio blocks. Its successful render demonstrates input acceptance, not parity with hosted MiniMax output.

Our [src/engine.c](../src/engine.c) has the same restriction:

- Around line 1364, `total_audio_samples` is shared across standalone audio, embedded soundtracks, and explicit video+audio pairs.
- Around line 1405, each is tested against the same `32000 * 15` allowance; around line 1462, all increment the same counter.
- Around lines 731–763, `audio_inputs` also includes video soundtracks when enforcing the three-audio-input limit. This deserves a corresponding count-policy audit.

For example, an accepted two-second video soundtrack plus a 15-second standalone WAV reaches 17 seconds and is rejected. Both source categories independently fit the published limits. This is a **source-confirmed contract mismatch**; I checked the rejection arithmetic without running the full model.

Implement separate category counters, preserving the individual clip limits, total video limit, standalone-audio limit, overall input-file limit, ordering, and resource checks. Define explicitly how a separate soundtrack supplied with a video is classified. Do not merely remove all limits or discard the soundtrack. Validate both reference orders, boundary durations, embedded/explicit soundtracks, and preservation of both packed audio blocks. Start with host/preparation tests; a long render is unnecessary to prove the acceptance fix.

### R02 — Decode-to-encode streaming is genuinely missing

Our [VideoVAE decoder](../src/vae/video_vae.c) already tiles and decodes temporal chunks. However, `h3_video_vae_decoder_decode_progress()` and `decode_chunked()` allocate the entire `final_rgb` clip. [src/engine.c](../src/engine.c) then creates another full RGB8 clip, delivers final frame callbacks, and only afterward calls [FFmpeg](../src/media/ffmpeg.c). Its audio/video pipe writers run concurrently, but they receive already-complete buffers.

At 1344×768 and 362 frames, the retained float RGB buffer alone is **4.176 GiB**, plus **1.044 GiB** for RGB8 before tile scratch, resize buffers and audio. These are calculated buffer sizes, not measured peak RSS. Streaming can remove most of that retention and provide earlier completed frames even if total latency improves only modestly.

[Official PR #71](https://github.com/MiniMax-AI/MiniMax-H3/pull/71) is closed unmerged. Its useful idea is to publish only frames whose overlap blending and padding removal are complete. Its reported 3.2513→2.3374-second result is a bundled VAE-through-MP4 experiment on four L20X GPUs, not a 28% full-generation speedup for h3cli. The superseding integration work is now merged: [vLLM-Omni #7017](https://github.com/vllm-project/vllm-omni/pull/7017) adds H3 chunk callbacks and [#7018](https://github.com/vllm-project/vllm-omni/pull/7018) connects them to bounded encoding. The latter's headline approximately 28% timing example concerns Wan; do not attribute it to H3.

Recommended native design: emit each committed 17-frame region after blending, retain the five-frame overlap until final, and queue bounded RGB8 batches to FFmpeg. On CUDA, evaluate pinned buffers and event-ordered asynchronous transfer; on Metal, use an appropriate shared-buffer handoff. Preserve current resizing, continuation trimming, audio timestamps, codec options, cancellation and callback lifetimes. Avoid declaring a reusable buffer free while FFmpeg or a transfer still reads it.

Benchmark existing final latents through decode and mux separately from denoising. Measure peak host/device memory, time to first delivered frame, decode wall time and MP4 completion. A single saved-latent pair can answer this without repeating a 50-step generation.

### R03 — Distilled adapters need more than compatible tensor names

Our [LoRA workflow](../lora/README.md) already folds adapters offline, checks shapes/scales, and has a pinned LarryVRH Turbo profile. Keep that accepted behavior. Its LightX2V entry is only a conditional compatibility statement, and the native sampler still uses its existing dual schedules.

New evidence from [vLLM-Omni #7062](https://github.com/vllm-project/vllm-omni/pull/7062): LightX2V's artifact family includes FL2VA and Ref2VA, four- and eight-forward variants, video shifts of 6 or 12, and differing alpha values. Some rank-128 artifacts declare alpha 8; assuming alpha 128 would multiply their intended update by 16. Different layouts also require different FFN/QKV mappings. An earlier integration, [#6476](https://github.com/vllm-project/vllm-omni/pull/6476), explicitly enforced video/audio shifts 6/3 for its supported four-forward artifact.

Add reviewed artifact profiles keyed by content identity and metadata, covering model family, tensor conventions, actual transformer-forward count, and both sigma grids. Persist the contract in the folded-model manifest and enforce it at generation/resume/cache boundaries. Reject unsupported combinations explicitly. Our existing alpha handling should be retained; I did not establish an alpha bug in it. The missing qualification is the complete artifact-plus-sampler contract.

Also normalize benchmark terminology. [Diffusers' H3 scheduler](https://github.com/huggingface/diffusers/blob/a3e0b8ec235c27a6c17a21976daf7fd32d819d05/src/diffusers/schedulers/scheduling_minimax_h3.py) accepts sigma-point counts: `50` means 49 transformer evaluations. Our oracle already calls `set_timesteps(steps + 1)` in [tests/refvideo_dit_oracle.py](../tests/refvideo_dit_oracle.py). Compare actual evaluations, grids and reuse settings, not a shared number called “steps.” There is no reason to change the accepted native step convention merely to match a framework CLI.

### R04 — Better prompt preparation without mandatory hosted services

The official [base-mode guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/references/base-en.txt) gives task-specific first/last-frame alignment and separate visual/dialogue, soundscape and music fields. Its [reference-mode guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/references/ref-en.txt) adds subject definitions, retention analysis, asset roles and stable speaker identities. Our README has useful general prompting advice but not this fuller workflow.

Add optional templates or a small preprocessing tool that validates reference labels, speaker IDs, shot times and anchor roles against the actual CLI inputs and aligned duration. Preserve original dialogue language and permit plain prompts unchanged. If an LLM rewrite is offered, show/save the final prompt and make it optional; deterministic validation is independently useful. [Issue #41](https://github.com/MiniMax-AI/MiniMax-H3/issues/41) describes a community typed-draft/validation/repair approach worth evaluating, without claiming to reproduce private Context-IR.

Do not derive a hard 7,000-character cap from the unanswered [prompt-length issue #61](https://github.com/MiniMax-AI/MiniMax-H3/issues/61). Actual token counts and supported model/resource limits are the relevant quantities.

### R05 — Add targeted quality cases before changing audio or conditioning

| Reports | Interpretation for this codebase | Recommended check |
| --- | --- | --- |
| [#50](https://github.com/MiniMax-AI/MiniMax-H3/issues/50), [#51](https://github.com/MiniMax-AI/MiniMax-H3/issues/51): initial pop or fragment of speech | A recurring reported audio-boundary problem; #51 has an acknowledgment and is closed, but no verified correcting patch was identified. It is not evidence of a CUDA indexing failure here. | Listen to the first/last 200 ms, silence-to-speech onset, and continuation seams. Compare raw PCM and muxed AAC to distinguish generation from codec priming/container effects. |
| [#17](https://github.com/MiniMax-AI/MiniMax-H3/issues/17): voice identity crosses subjects | Reported across reference modes, including community comparisons with the hosted service. | Two clearly named speakers with distinct references; inspect voice consistency, speaker turns and lip sync. |
| [#68](https://github.com/MiniMax-AI/MiniMax-H3/issues/68), [#80](https://github.com/MiniMax-AI/MiniMax-H3/issues/80), [#81](https://github.com/MiniMax-AI/MiniMax-H3/issues/81) | Reports of sung dialogue, failed language conversion, and spoken nouns leaking into visuals. These remain behavior reports, not isolated native defects. | Short dialogue/language cases with explicit roles; compare plain and structured prompts without promising a universal fix. |
| [#74](https://github.com/MiniMax-AI/MiniMax-H3/issues/74), [#73](https://github.com/MiniMax-AI/MiniMax-H3/issues/73) | Fast-motion detail and clean line art can deteriorate despite more steps or expanded prompts. | Include motion and thin-line cases when qualifying previews, precision changes or new adapters. |
| [#69](https://github.com/MiniMax-AI/MiniMax-H3/issues/69): 7.5-second reference produces an 8-second output with timing complaints | Our frame alignment can explain the duration increase; it does not explain every reported change in voice or motion. | Report requested and effective duration before generation; check both A/V clocks and reference timing. |

A compiled local probe of `h3_temporal(180)` returns **192 frames, 320 audio ticks: 8.000 seconds for both**. Our FFmpeg path uses fixed 24 fps and 32-kHz audio and has no `atempo` stretching. Thus the 7.5→8-second expansion alone is expected alignment, not proof of an A/V synchronization bug. Audio latent rounding for other lengths should be distinguished from larger drift.

Do not silently trim 50–200 ms or add a fade as a supposed root-cause fix: that can remove speech and alter continuation timing. A user-selectable boundary treatment could be considered separately after diagnosis.

[PR #67](https://github.com/MiniMax-AI/MiniMax-H3/pull/67) offers a useful evaluation idea: assess explicit identity traits, first checking that the reference actually shows them, and include negative controls. Use this as review assistance on a small corpus; automated VLM scores should not replace playback, listening and user acceptance.

### R06 — Keep external reference implementations trustworthy

**Checkpoint format.** [Issue #48](https://github.com/MiniMax-AI/MiniMax-H3/issues/48) reports original-versus-Diffusers QKV/FFN differences. The [current converter](https://github.com/huggingface/diffusers/blob/a3e0b8ec235c27a6c17a21976daf7fd32d819d05/scripts/convert_minimax_h3_to_diffusers.py) explicitly reorders per-head interleaved QKV and swaps fused SwiGLU gate/value halves. Raw tensor inequality is therefore not enough to establish bad weights. Preserve original-format loading and extend explicit format detection/provenance when supporting converted or quantized artifacts; do not “correct” rows by copying them between formats.

**Framework padding.** The discussion under [H3 issue #56](https://github.com/MiniMax-AI/MiniMax-H3/issues/56) leads to [PyTorch #194922](https://github.com/pytorch/pytorch/issues/194922), which reports size-dependent silent MPS temporal-padding corruption, and [ComfyUI #15902](https://github.com/Comfy-Org/ComfyUI/pull/15902), an open workaround replacing temporal constant padding with concatenation. Our native [encoder-padding kernel](../src/metal/shaders.metal) does not call PyTorch `F.pad`, so this is not evidence that our Metal path shares the bug. It does justify a known-pattern padding check before trusting an MPS-based encoder oracle after framework changes. The separate `aten::_int_mm` failure also does not apply to our custom native kernels.

**Dependency and source identity.** [Official PR #37](https://github.com/MiniMax-AI/MiniMax-H3/pull/37) repairs stale Diffusers installation links and insufficient dependency declarations. We already pin oracle packages in [tests/requirements-refvideo-oracle.txt](../tests/requirements-refvideo-oracle.txt) and record source/weight hashes in several oracle scripts. Extend that coverage consistently to full generation and downloaded conversion code. Record GitHub source revision, Hugging Face weight/remote-code revision, converter hash, device and dtype separately; a GitHub code merge does not establish that the loaded Hub snapshot contains it.

### R07 — More aggressive previews: distinguish Dense FastH3 from VSA

[vLLM-Omni #6714](https://github.com/vllm-project/vllm-omni/pull/6714) implements FastH3 Dense/Data-Free, a four-forward T2VA student. Its artifact includes low-rank updates **and full-rank deltas/biases/replacement tensors**. Our [fold_lora.py](../lora/fold_lora.py) rejects unrecognized tensor types, so this is not an adapter we can currently promise to fold. A dedicated reconstruction profile must apply the artifact's complete update order and schedule. Start with offline fusion and T2VA only; evaluate visual/audio quality against both the base model and our accepted Turbo option. Published multi-GPU timings do not predict single-GPU or Metal performance.

[vLLM-Omni #6909](https://github.com/vllm-project/vllm-omni/pull/6909), merged September 2, adds FastH3 VSA using learned compression gates in all 50 DiT blocks, 64-token video tiles and top-k block selection. This requires the matching student and a different attention implementation. It is not a switch that can safely turn our existing dense Ref2VA model into sparse attention. Its reported 10-second case improved MP4-inclusive time from 9.838 to 7.278 seconds on **eight B300 GPUs**, comparing different Dense/VSA student artifacts; peak memory increased slightly. Treat those results as scoped feasibility evidence.

If Dense previews prove worthwhile, consider CUDA VSA as a separate experiment with explicit model, architecture and task qualification. A Metal implementation requires a kernel port and its own measurements. Preserve dense fallback and qualify these experiments separately from the existing CUDA and Metal workflows.

### R08 — Reduced-precision VAE, keeping the trained tiling behavior

[Diffusers' decode step](https://github.com/huggingface/diffusers/blob/a3e0b8ec235c27a6c17a21976daf7fd32d819d05/src/diffusers/modular_pipelines/minimax_h3/decoders.py) uses CUDA FP16 autocast with float32 VideoVAE weights. Our decoder still primarily uses F32 operations. This supports experimenting with lower-precision convolution/activation storage, retaining higher-precision reductions where needed, and qualifying by visible output and listening rather than numerical identity.

[Issue #46](https://github.com/MiniMax-AI/MiniMax-H3/issues/46) reports shading/block changes when tile settings change. Our existing tile work already addresses released behavior. Keep those tile sizes, overlap and blending fixed while changing execution precision. Large VRAM is not a reason to substitute untiled/full-frame VAE inference. Prioritize this after profiling; it overlaps the earlier fork review's recommendation and should not become a duplicate task series.

## Changes that should not be copied as fixes

**Official PR #19 is unsafe to adopt as a bundle.** Despite its “Fix login bug” title, its [diff](https://github.com/MiniMax-AI/MiniMax-H3/pull/19/files) changes AudioVAE inference geometry. It replaces `ceil(stride/2)` padding with `floor(stride/2)` and removes attention head pooling while changing the projection shape. A small calculation using the released strides `[2,4,4,5,5]` gives:

| One-second, 32,000-sample input | Length after each downsample |
| --- | --- |
| Released/native padding | 16,000 → 4,000 → 1,000 → 200 → **40** |
| Proposed PR #19 padding | 16,000 → 4,000 → 1,000 → 199 → **39** |

For our released encoder dimensions, its projection also changes from `[32,32]` to `[32,2048]`, incompatible with the existing weight shape. The [current Diffusers AudioVAE](https://github.com/huggingface/diffusers/blob/a3e0b8ec235c27a6c17a21976daf7fd32d819d05/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py) deliberately retains ceil padding and pooled heads. PR #19's source-string checks do not establish released-checkpoint compatibility. Evaluate any independently useful cleanup separately.

Other exclusions:

- Merged [#10](https://github.com/MiniMax-AI/MiniMax-H3/pull/10) and [#14](https://github.com/MiniMax-AI/MiniMax-H3/pull/14) concern Python weight initialization under weight normalization. Native inference loads pretrained tensors; these are not a demonstrated correction for our generated-audio boundaries.
- [Issue #65](https://github.com/MiniMax-AI/MiniMax-H3/issues/65) proposes selecting modality rows before output projection. [src/denoise/dit.c](../src/denoise/dit.c) already projects `dit->video_rows` and `dit->audio_rows` separately. No new speedup to implement here.
- AdaLN precomputation is already present. Official discussion of caching those branches does not identify missing work in h3cli.
- [PR #38](https://github.com/MiniMax-AI/MiniMax-H3/pull/38) fixes polling of asynchronous API examples. Useful for a future service client, not a defect in the synchronous native CLI.
- [Issue #62](https://github.com/MiniMax-AI/MiniMax-H3/issues/62) reports a SageAttention integration failure; [#70](https://github.com/MiniMax-AI/MiniMax-H3/issues/70) reports a PyAV encoder error with little diagnosis. Neither establishes a failure in our cuDNN attention or FFmpeg subprocess path.
- [Issue #60](https://github.com/MiniMax-AI/MiniMax-H3/issues/60) reports failed image-conditioned second passes/latent upscaling. A refinement path must preserve anchors and reference conditioning; simply resizing latents is not the unreleased Regenerate-2K workflow.

## Forks worth knowing about

| Fork and inspected head | Useful idea | Applicability |
| --- | --- | --- |
| [winbeau/MiniMax-H3](https://github.com/winbeau/MiniMax-H3/tree/dfe18a4e55fff32be4a135fcc1bff3af7af5f4ef), 13 commits ahead | Separate text conditioning onto another device; component/block offload; benchmark output metadata and run exclusion. | Reinforces independent/persistent conditioning work. Two-GPU component placement is not evidence of two-GPU DiT acceleration. Its “lossless” profile means the base schedule, not a measured universal quality guarantee. |
| [shejitu/MiniMax-H3](https://github.com/shejitu/MiniMax-H3/tree/78ff58256be300a19eea34ef698012d5d0508521), 8 ahead | Kaggle dual-T4 deployment guidance and quantized ComfyUI workflows. | Ideas for constrained-memory setup, not native CUDA kernel patches or transferable performance evidence. |
| [desmondzee/WAM-H3](https://github.com/desmondzee/WAM-H3/tree/e661b1519cf1ba7068081712199895df2b44c846), 47 ahead | Robotics adaptation replaces the audio role with actions, changes attention masks, and caches context for action-only denoising. | Interesting architectural research, not compatible audio/video inference. Ordinary H3 has bidirectional joint attention; caching reference K/V across denoising steps is not automatically equivalent. |
| [benriworks/MiniMax-H3](https://github.com/benriworks/MiniMax-H3/tree/1f262cead5fcd258ee5c276e4ae5383f2906c617), 64 ahead, diverged | Windows Python/API tooling and a substantial production/shot/timeline UI. | Optional application-layer inspiration. It does not establish a native Windows CUDA build of h3cli; not exhaustively audited. |
| [wmd-1/MiniMax-H3](https://github.com/wmd-1/MiniMax-H3/tree/01db239f9311c095e0c953178dc478e672d38083), 11 ahead | Container/service setup and batch request scripts. | Packaging and orchestration ideas, not model-level optimizations. |

Other inspected differences were documentation, prompt assets, administrative files or merges without substantive changes. Several recently pushed defaults were identical to official main. PR source branches were considered separately, so an unchanged default branch was not treated as proof that a contributor had no useful work.

## Bounded validation and implementation order

1. **R01 first:** host/preparation acceptance and packing tests, followed by one short mixed-reference smoke. Include valid separate budgets and invalid per-category excess. Do not run a 15-second, 50-step generation merely to test input admission.
2. **R02 and R08 independently:** replay existing final latents to isolate decode/mux memory and timing. Cover a temporal seam, a spatial seam and final-tail handling. Hold codec settings fixed.
3. **R03/R04/R05:** use existing face/body/`2.jpg`/`1.jpg` inputs where appropriate, plus short audio/video fixtures. Use 56 frames for mechanical smokes; use approximately 107 frames for dialogue and motion quality because very short clips poorly represent the released duration range. Limit the first comparison to a few representative pairs and a preset time budget.
4. **R07 last:** one dedicated Dense student profile before sparse kernels; four-forward student runs should be compared with its required schedule. Extend only after the first quality and timing results justify it.

Structural acceptance includes playable MP4s, expected frame counts, intact stereo audio, correct timing and reference retention. Performance acceptance is measured wall time and memory on the actual target device. Quality acceptance is matched playback/listening, including identity, motion, tile seams and speech; tensor equality is not the quality gate. No `test2.sh`-scale or nine-hour render campaign is proposed.

The local review artifacts are in [outputs/minimax-official-review/2026-09-18](../outputs/minimax-official-review/2026-09-18/): an issue/PR/fork inventory, downloaded-source hashes, and reproducible host/arithmetic probes. `sh outputs/minimax-official-review/2026-09-18/reproduce.sh` compiled and passed locally. Those probes establish frame alignment, buffer sizes and the PR #19 geometry problem; they do not constitute a model-quality or GPU-performance test.
