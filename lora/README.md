# LoRA adapters for h3cli

A LoRA stores a learned change to a model's weight matrices. H3 can fold it into
BF16 weights automatically at startup and cache the result, or load a model
prepared with the offline Python workflow. Neither path adds LoRA matrix
multiplications to the denoising loop.

## 1. Check licensing

Read the current [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
and every adapter's license **before downloading, folding, running, or distributing
weights**. The current base license lists the United States, European Union,
United Kingdom, and Republic of Korea as excluded territories and provides a
separate authorization route. Consult the terms directly; this tool provides no
additional rights or legal interpretation. Adapter availability does not override
the base license.

## Runtime folding (Metal and CUDA)

Pass adapters directly to the native executable. On the first request for a
mode, H3 folds the selected adapters into a separate BF16 transformer, validates
it, and continues generation. Later requests verify and reuse that variant.
Python, NumPy, safetensors Python bindings, and an additional BLAS library are
not runtime dependencies.

```sh
# Turbo: select eight steps explicitly; detecting the adapter never changes steps.
./bin/h3cli -d models/MiniMax-H3 \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors \
  --steps 8 --width 256 --height 256 --frames 56 --seed 42 \
  --preview-vae -p 'A person walks through a sunny park.' -o outputs/turbo.mp4

# Ordered combination; each adapter retains its own rank/alpha scaling.
./bin/h3cli -d models/MiniMax-H3 \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1 \
  --lora lora/downloads/h3-realism-people-t2v-i2v-r2v.safetensors:0.6 \
  --steps 8 -p 'r34l1sm. A person walks through a sunny park.' -o outputs/combined.mp4
```

| Option | Meaning |
| --- | --- |
| `--lora PATH[:SCALE]` | Repeatable, ordered adapter selection; default scale `1`. |
| `--lora-cache DIR` | Override the variant cache location. |
| `--lora-memory-mib N` | Positive integer folding allocation budget; default `512` MiB. This does not change inference memory. |

A readable literal filename takes precedence over splitting its final colon.
Quote paths containing spaces. Symlinks resolve to their regular-file target.
Scales use decimal syntax independent of locale; finite negative and fractional
values are supported. Zero strengths still validate the adapter and its target
weights. Duplicate arguments remain separate updates. `0` and `-0`, or `0.5`
and `5e-1`, select the same scale. Changing adapter order can change both the
arithmetic result and the cache identity.

FL2VA serves prompt-only and first/last-frame requests. Ordered references select
Ref2VA. Only the selected mode is prepared; a library context
prepares the other mode on its first use. Standalone installations containing
just one mode are supported. `--info` validates and reports the selection without
folding or creating a cache. Decode-only requests reject LoRA flags because saved
AV latents already contain the denoised result. Cache/memory flags require at
least one `--lora`.

The native mapping supports the same ordinary H3 A/B, down/up, PEFT, supported
Diffusers Q/K/V slices, and unambiguous ComfyUI names described below. Unsupported
architectural adapters, unknown tensors, missing pairs, conflicting metadata,
non-finite values, and incompatible shapes fail explicitly. A detected Turbo
content hash is informational. Already folded BF16 bases are accepted and their
parent manifest participates in the new cache identity; selecting the same
adapter again applies it again.

### Cache and resource costs

The default cache is `$XDG_CACHE_HOME/h3/lora` when `XDG_CACHE_HOME` is absolute,
otherwise `$HOME/.cache/h3/lora`. An explicit cache path may be relative. Cache
roots must be private directories owned by the running user; source/cache overlap
and cache-entry symlinks are rejected.

The key covers the selected mode, all original transformer files, parent fold
identity, ordered adapter contents, binary64 strengths, and native mapping/math
versions. It excludes paths, timestamps, memory limits, prompts, seeds, step
counts, canvas sizes, fast-CUDA policy, quantization, and decoder settings.
Moving inputs with identical content therefore reuses the same entry. Original
model and adapter files must remain readable: H3 rehashes them to establish the
selection on each request.

Each mode/combination has a complete transformer directory (about **61.7 GiB
logical size** for the tested model). APFS clones or Linux reflinks reduce the
initial copy cost where supported, but writing folded tensors consumes storage.
A full copy is automatic when cloning is unavailable, after a disk-space check.
If a filesystem charges clones against a logical-size quota and reports too
little room for their modified pages, H3 replaces its unpublished modified
clones with checked independent copies, one shard at a time.
Multiple strengths/combinations can use substantial additional disk space.
The cache never writes to the source weights and never automatically evicts
variants.

The memory limit includes conservative metadata accounting and bounded matrix,
conversion, I/O, and validation scratch. The default uses only a fraction of its
512 MiB allowance for the current H3 adapters. Small budgets may reject a large
header/plan before copying. OS file-cache residency, process libraries, and later
inference allocations are separate from this folding budget. H3's normal
available-memory floor remains active.

A hit verifies every cached file's contents, not just its timestamp. Consequently
warm preparation still includes reading and hashing large model files; it is
not instantaneous. Logs separate input hashing/planning, copying, folding,
validation, and cache verification from the existing inference profile.

Stable per-key locks coordinate processes. Readers hold shared leases while a
context can use the weights; writers wait cancellably, build in private staging,
and publish atomically. A corrupt entry is moved to `invalid/` and rebuilt once.
Read-only hits work when the existing lock and complete cache directories are
present. A miss or repair on an unwritable/full cache fails with an error.
There is no fallback that silently omits adapters.

Cancellation removes the current unpublished staging directory. After a killed
writer, the next writer for that key cleans up abandoned staging under the key
lock. A completed entry survives later encoder, quantization, or generation
failures. For manual pruning, stop all H3 processes using the cache, remove the
unneeded directories under `variants/` (and old `invalid/`/`staging/` entries),
and leave `locks/` intact. The next request rebuilds an evicted variant.

### Checkpoints, continuation, and quantization

Repeat the original ordered LoRA flags when resuming. The checkpoint selects its
saved FL2VA/Ref2VA mode, so reference flags are not repeated. Exact compatibility
uses the effective model fingerprint. Rebuilding an evicted native variant keeps
that fingerprint; different/missing adapters, strengths, order, base contents,
or an independently produced offline manifest do not.

```sh
./bin/h3cli -d models/MiniMax-H3 \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors \
  --steps 8 --width 256 --height 256 --frames 56 --seed 42 \
  -p 'A person walks through a sunny park.' \
  --stop-after-step 3 --save-sampler-state outputs/paused.h3sample

./bin/h3cli -d models/MiniMax-H3 \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors \
  --resume-sampler-state outputs/paused.h3sample \
  --preview-vae --save-av-state outputs/complete.h3av -o outputs/complete.mp4
```

Sampler format and ordinary no-LoRA/offline model fingerprints are unchanged.
Saved states receive an informational `.lora.json` sidecar containing the native
key and ordered selection; the sidecar does not replace compatibility checks.
Hard and bridge `.h3av` continuation retain their existing VAE/latent contract,
so the next segment may select another adapter. Decoder-only finalization needs
no LoRA flags.

Runtime folding always produces BF16 weights **before** existing CUDA FP8/NVFP4
preparation. The LoRA and packed-weight caches are independent; unchanged folded
tensors retain their content-based packed identities. Fast CUDA and preview VAE
remain independent execution/decoder choices and do not trigger refolding.

```sh
# RTX 5090 example; keep the two caches on volumes with sufficient free space.
./bin/h3cli -d models/MiniMax-H3 \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-cache /path/to/private/lora-cache --lora-memory-mib 512 \
  --width 1344 --height 768 --frames 362 --steps 8 --seed 42 \
  --cuda-denoise-quant nvfp4 \
  --cuda-denoise-quant-cache /path/to/quant-cache --preview-vae \
  -p 'A person walks through a sunny park.' -o outputs/turbo-large.mp4
```

Library callers use `h3_load_dir_with_lora(model_dir, &options)` with
`h3_lora_options`. The context copies the options and adapter paths. Preparation
uses each request's progress/cancellation callback, and `h3_free` releases cached
model objects, selection metadata, and reader leases. `h3_load_dir` retains its
previous behavior.

See [runtime validation](../docs/lora/runtime-validation.md) for measured costs,
exact test settings, numerical coverage, and remaining assessment limits. The
Python workflow below remains useful for exporting standalone folded models.

## 2. Create the optional offline Python environment

From the repository root:

```sh
python3 -m venv lora/.venv
source lora/.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r lora/requirements.txt
mkdir -p lora/downloads lora/output outputs/lora-ab
```

The virtual environment avoids macOS/Homebrew's `externally-managed-environment`
pip error. Only NumPy and safetensors are required. PyTorch, Diffusers, ComfyUI,
and MLX are not dependencies. The C build works independently of this environment.

## 3. Select and download an adapter

Sources checked September 16, 2026; check the linked cards for later updates.

| Adapter | Intended use and compatibility |
| --- | --- |
| [LarryVRH MiniMax-H3 Turbo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | Recommended acceleration adapter: `minimax_h3_turbo_v4_step600_ema.safetensors`, strength 1.0. Its plain BF16 updates use native scale 1. Use 6–8 steps, with 8 as the quality example. Four-step fast-motion smear is an adapter-specific caveat. The tool recognizes this release by SHA-256. |
| [fal Realism People](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA) | Ordinary fused-attention adapter; trigger `r34l1sm`. Start at 1.0, or 0.6–0.8 for a lighter effect. The publisher describes text, image, and reference workflows. FL2VA and Ref2VA must still be folded and tested separately. |
| [LightX2V MiniMax-H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | Alternative few-step adapters. Select the matching FL2V/Ref2V variant. Compatibility depends on the actual keys/layout accepted by the normalizer; the repository name alone is not a compatibility guarantee. Run `--inspect` first. |
| [Alibaba PAI Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) | **Unsupported:** PDD adds output-head and inference behavior that cannot be represented by folding the backbone A/B matrices alone. Use its dedicated inference implementation. |

The [MiniMax integration index](https://github.com/MiniMax-AI/awesome-minimax-h3-integration)
is useful for discovery; inclusion there does not establish folding compatibility.

After checking the licenses, download the two example adapters explicitly:

```sh
curl -fL --retry 3 \
  https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/43a74557ac3f6539db8e0f2a959d03feb7a81480/minimax_h3_turbo_v4_step600_ema.safetensors \
  -o lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors
curl -fL --retry 3 \
  https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/039cc8579d7aa357a882d7f4111b25da4f72dccc/h3-realism-people-t2v-i2v-r2v.safetensors \
  -o lora/downloads/h3-realism-people-t2v-i2v-r2v.safetensors
```

Pinned release SHA-256 values:

```text
Turbo:   5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3
Realism: acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e
```

The folding tool itself never downloads weights.

## 4. Inspect and fold one adapter

`--checkpoint` takes a **transformer directory**. `h3cli -d` takes the **complete model
directory** containing FL2VA and/or Ref2VA plus their unchanged assets.

```sh
python lora/fold_lora.py \
  --checkpoint models/MiniMax-H3/FL2VA/transformer \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 \
  --dry-run

python lora/fold_lora.py \
  --checkpoint models/MiniMax-H3/FL2VA/transformer \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 \
  --out lora/output/Turbo/FL2VA/transformer
```

Typical dry-run summary for the pinned Turbo release:

```text
Adapters: 1
LoRA pairs: 259
Targets matched: 259
Targets unsupported: 0
Shards modified: 13
... effective_scales=[1.0] ... profile=larryvrh-turbo-v4-step600-ema
No alpha metadata for some/all targets: using alpha=rank, native scale=1 (plain BA).
Dry run: validation and mapping PASS; no files written. Numerical probes run during folding.
```

A successful fold adds per-matrix numerical metrics and finishes with:

```text
BF16 validation: PASS
Manifest written: .../transformer/h3_lora_manifest.json
```

`--inspect` is a verbose dry run: it prints all tensor names, shapes, dtypes,
metadata, normalized targets, ranks, alpha values, formats and profiles. Unknown
keys and unsupported tensors are errors, never ignored. `--verbose` provides the
same detailed mapping during a fold. Numerical probes require an actual fold.

The default scale is 1.0. A final `:SCALE` may be zero, negative, fractional, or
above one, but must be finite and representable in FP32. Existing literal
filenames take priority (including colons); otherwise the last colon separates
the scale. Quote paths containing spaces or shell punctuation.

Effective scale is `user_scale * alpha / rank`. Scalar `.alpha` tensors and
ordinary `alpha`/`lora_alpha`/`ss_network_alpha`/`network_alpha` metadata are supported. Conflicting
alpha or rank metadata is rejected. When alpha is absent, the tool explicitly
reports and records **alpha=rank, native scale=1, plain BA**. Confirm that this
matches the adapter's training convention; it is not a guess inferred from its
filename. Both pinned examples publish plain-BA updates.

## 5. Assemble the complete model tree

This command symlinks unchanged tokenizer, processor, text encoder, video VAE,
audio VAE, and model-index assets from the base. It keeps the folded transformer
as an independent copy and refuses to replace conflicting assets.

```sh
python lora/workflow.py assemble \
  --base models/MiniMax-H3 --model lora/output/Turbo --mode FL2VA
```

It assembles only the requested mode. It does not silently supply an unfurled
base transformer for other modes. Keep the original model in place: shared asset
symlinks depend on it.

For a separate ordinary style model, including Ref2VA:

```sh
for mode in FL2VA Ref2VA; do
  python lora/fold_lora.py \
    --checkpoint "models/MiniMax-H3/$mode/transformer" \
    --lora lora/downloads/h3-realism-people-t2v-i2v-r2v.safetensors:1.0 \
    --out "lora/output/Realism/$mode/transformer"
  python lora/workflow.py assemble \
    --base models/MiniMax-H3 --model lora/output/Realism --mode "$mode"
done
```

A successful FL2VA mapping proves neither Ref2VA layout nor adapter quality.
Test the Ref2VA fold independently with references.

## 6. Run and compare

For the distilled Turbo model, start without `--reuse` or `--core-reuse`. Neither
approximation is validated by this profile; folding does not make it compatible.
The native H3 sampler retains its existing dual audio/video schedules.

```sh
./bin/h3cli -d lora/output/Turbo \
  -p 'A woman smiles and turns toward the camera. Quiet room ambience.' \
  --first-frame inputs/face1.jpg --seed 72 \
  --width 256 --height 256 --frames 56 --steps 8 \
  -o outputs/lora-ab/turbo.mp4
```

Use 6 steps as the faster alternative. Do not apply Turbo's few-step settings to
ordinary style/content adapters. Here is an ordinary Realism A/B test at the
normal 20-step schedule, with the trigger present in **both** prompts:

```sh
for variant in base realism; do
  model=models/MiniMax-H3
  if [ "$variant" = realism ]; then model=lora/output/Realism; fi
  ./bin/h3cli -d "$model" \
    -p 'r34l1sm, a woman smiles and turns toward the camera. Quiet room ambience.' \
    --first-frame inputs/face1.jpg --seed 72 \
    --width 256 --height 256 --frames 56 --steps 20 \
    -o "outputs/lora-ab/$variant.mp4"
done

./bin/h3cli -d lora/output/Realism \
  -p 'r34l1sm, the person in <Picture 1> smiles at the camera.' \
  --ref-image inputs/face1.jpg --seed 72 \
  --width 256 --height 256 --frames 56 --steps 20 \
  -o outputs/lora-ab/realism-ref.mp4
```

Keep prompt, seed, resolution, frame count, steps, sampler settings, and references
identical in each A/B pair. Repeat the Turbo command with `-d models/MiniMax-H3` and a
new output name for an eight-step base/Turbo comparison. Inspect motion, faces,
reference fidelity, and audio; a numerical fold test cannot certify visual quality.

## 7. Combine adapters in one operation

```sh
python lora/fold_lora.py \
  --checkpoint models/MiniMax-H3/FL2VA/transformer \
  --lora lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 \
  --lora lora/downloads/h3-realism-people-t2v-i2v-r2v.safetensors:0.7 \
  --out lora/output/Turbo-Realism/FL2VA/transformer
python lora/workflow.py assemble \
  --base models/MiniMax-H3 --model lora/output/Turbo-Realism --mode FL2VA
./bin/h3cli -d lora/output/Turbo-Realism \
  -p 'r34l1sm, a woman smiles in soft window light. Quiet ambience.' \
  --first-frame inputs/face1.jpg --seed 72 \
  --width 256 --height 256 --frames 56 --steps 8 \
  -o outputs/lora-ab/turbo-realism.mp4
```

For each tensor, the tool evaluates `W + sum(scale_i * B_i @ A_i)` in FP32, then
rounds to BF16 **once**. There are no intermediate folded checkpoints. Adapter
addition follows command-line order, with lexical target order within each
adapter; use the same order for bitwise reproducibility on the same NumPy/BLAS
platform. Duplicate or overlapping mappings **within one adapter** are rejected;
overlap between separately supplied adapters is intentional addition.

Individually valid adapters can degrade one another's quality. Compare the
combination against each individual fold using identical seeds and settings.

## Storage, integrity, and numerical validation

Shard copies use `clonefile` on APFS/macOS and `FICLONE` reflinks on supporting
Linux filesystems, such as XFS with reflinks enabled or Btrfs. Keep the source
checkpoint and output on the same filesystem to use CoW. The fold manifest's
`copy_methods` records `clonefile`, `reflink`, or `full-copy` for each file.
Logical size stays about the size of the transformer, while physical storage
grows as modified ranges are written.
Turbo touches many large matrices, so its real footprint can still be substantial.
Copying a folded model to another filesystem may materialize its entire logical
size. CoW is not a substitute for available free space.

If cloning is unavailable, including across filesystems, folding stops unless
`--allow-full-copy` is supplied. Failed Linux clones leave no destination file.
That option prints a prominent per-file full-copy size and checks free space.
Neither mode uses hard links to writable checkpoint files.

Source/output aliases and nested source/output directories are rejected. Existing
outputs require `--overwrite`; the previous output stays intact during validation.
All work happens in a temporary sibling directory. Finalization replaces the
output only after structural checks, numerical probes, hashes, and manifest
creation succeed. Exceptions clean the stage; a forced process kill can leave a
hidden `.NAME.tmp-*` directory, never a finalized model. During overwrite, a
forced kill between renames can leave `.NAME.backup-*`, from which the old output
can be recovered.

Headers, ordering, declared shapes/dtypes, absolute offsets, and unaffected bytes
are preserved exactly. Existing payload alignment is retained; unaligned released
shards continue to use h3cli's safe fallback. Validation checks duplicate keys,
bounds, integer sizes, overlaps, holes, shard-index agreement, and file stability.
All modified shards are reopened; all unchanged byte ranges are compared.

The default `--memory-mib 512` limits the planned matrix working set. Folding
processes one tensor at a time, subdividing large tensors into row blocks. Each
row still gets every applicable FP32 update before its only BF16 rounding. Mapped
file pages and NumPy/BLAS overhead are additional to this budget; it is not an OS
resident-memory cap.

Three deterministic probe vectors compare the FP32 fold and the actual BF16
weights read from disk to `W@x + sum(scale * B@(A@x))`. The manifest reports max
absolute error, mean absolute error, relative L2, and cosine similarity. Acceptance
uses a componentwise bound: FP32 reduction allowance plus BF16 round-to-nearest's
`2^-8 * abs(W_folded) @ abs(x)` bound and a small absolute floor. This handles
cancellation without treating ordinary BF16 quantization as a mapping failure.
Non-finite inputs/results and failed bounds stop finalization. Probes supplement
strict mapping/shape checks; they do not replace them.

## Manifest and continuation provenance

`transformer/h3_lora_manifest.json` records the tool version, source transformer,
SHA-256 of every base file, deterministic base identity, every adapter hash,
user/native/effective scales, alpha origin, normalized mappings, modified
shards/tensors, copy method, numerical metrics, and runtime profile advice.
Inspect it with:

```sh
python -m json.tool lora/output/Turbo/FL2VA/transformer/h3_lora_manifest.json
```

The Turbo profile is selected by the pinned hash, never its filename. Explicit
selection is available with `--profile larryvrh-turbo-v4-step600-ema`; with multiple
adapters provide one `--profile` per `--lora`, using `auto` for the others.
Explicit selection records advice, not a compatibility certification. Structurally
valid unknown adapters remain foldable.

Full `.h3sample` resume fingerprints include the folded model contents and manifest;
resuming against the base or another variant is rejected. Keep that tree and
manifest unchanged while a sampler checkpoint is in use. Completed `.h3av`
continuations retain their existing VAE/latent compatibility and may intentionally
continue under another folded transformer. To record their adapter provenance
without changing or binding the state format:

```sh
python lora/workflow.py provenance --model lora/output/Realism --mode FL2VA \
  --av-state outputs/my-shot.h3av
```

This writes `my-shot.h3av.lora.json` beside an existing state, with its SHA-256,
manifest hash, base identity and adapter hashes/scales. It is informational;
continuation latents are not permanently tied to a LoRA variant.

To remove an adapter, select the original base. To change strength, rebuild from
the **original** checkpoint. Applying the same adapter at a negative scale to an
already folded model cannot reliably undo BF16 rounding.

## Supported layouts and troubleshooting

Supported naming includes direct native H3 modules, `diffusion_model.` and
`model.diffusion_model.` prefixes, PEFT `base_model.model.`/`transformer.` prefixes,
`.lora_A[.default].weight` / `.lora_B[.default].weight`, ComfyUI
`lora_unet_blocks_*` with up/down pairs, and scalar `.alpha` values. Native H3
matrix modules include the token refiner. Separate `blocks.N.attn.to_q/to_k/to_v`
(or `transformer_blocks.N`) map to the equal Q/K/V row slices of H3's fused
projection; `to_out.0` maps to `out_proj`. Config dimensions are checked when
available. Other Diffusers layouts must receive an explicit tested normalizer.

| Problem | Action |
| --- | --- |
| Unmatched key, extra tensor, missing pair | Stop. Inspect every key; use a supported export or add a tested mapping. Do not remove inconvenient keys to force a partial fold. |
| PDD, VDN, ControlNet, DoRA, RS-LoRA | Use dedicated implementations. Runtime architecture changes and alternative scaling conventions are outside this ordinary-LoRA tool. |
| Shape/rank/orientation mismatch | Check base mode and adapter architecture. Matrices are never silently transposed or reshaped. |
| Insufficient disk space | Free space or choose a filesystem with sufficient capacity; partial stages are removed on failure. |
| CoW unavailable | Choose APFS, or deliberately use `--allow-full-copy` after checking the logical size. |
| Safetensors validation failure | Verify downloads/checksums; do not repair offsets by guessing. Relative misalignment and malformed payloads are errors. |
| Python dependency/import error | Activate `lora/.venv` and install `lora/requirements.txt` with that environment's Python. |
| Numerical validation failure | Inspect scales and finite values; retain the original model and report the adapter identity and failing target. |

Runtime LoRA execution, dynamic adapter switching during generation, training,
PDD/VDN, ControlNet and other architectural adapters are intentionally out of scope.

## Tests

```sh
make test-lora
make test                         # existing C/Metal tests
make test-lora-models             # opt-in full checkpoints and downloaded adapters
```

`LORA_PYTHON=/path/to/python` selects another environment. Normal LoRA tests create
small aligned H3-style checkpoints and require no real model. Full integration
uses the pinned adapters above, both base modes, `inputs/face1.jpg`, FFmpeg, and an
Apple GPU. It verifies all targets, base/output hashes, deterministic A/B renders,
audio/video generation, Ref2VA, hard/bridge continuation, cross-variant latent
continuation, denoised previews, memory/GQA checks, and exact stop/resume with
wrong-model rejection. Its small 128px renders validate execution and determinism;
use production-resolution A/B tests to judge appearance. Results and generated
models live in `outputs/lora-validation/`.

For larger, matched-setting base/Realism and base/Turbo comparisons after full
integration, run `python lora/tests/quality.py --size 256` (or a larger supported
multiple of 32). Inspect the generated contact sheet and videos; this script does
not assign a visual-quality pass. See [the validation report](VALIDATION.md).
