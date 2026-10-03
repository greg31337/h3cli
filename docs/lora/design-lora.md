## Offline LoRA Folding and Workflow Support

### Objective

Add a self-contained `lora/` directory that provides:

```text
lora/
├── README.md
├── fold_lora.py
├── requirements.txt
└── tests/
    ├── test_key_mapping.py
    ├── test_fold_math.py
    └── test_safetensors.py
```

The feature should support applying one or more ordinary MiniMax-H3 LoRA adapters to a BF16 H3 transformer checkpoint **offline**.

The output should be a normal H3 transformer checkpoint that requires no LoRA-specific code in the C/Metal inference engine.

The architecture should remain:

```text
base H3 transformer
        +
one or more LoRAs
        ↓
lora/fold_lora.py
        ↓
ordinary folded BF16 transformer
        ↓
unchanged h3cli inference engine
```

This provides LoRA support while avoiding changes to:

```text
Metal matmul execution
SSD weight streaming
weight prefetch
zero-copy loading
DiT layer scheduling
velocity reuse
continuation
bridge continuation
stop/resume
CUDA implementation, when added
```

---

# 1. Scope

The initial implementation should support **ordinary low-rank weight adapters** whose effective operation is:

```text
W_effective =
    W +
    scale × B × A
```

or, for multiple adapters:

```text
W_effective =
    W +
    Σ(scale_i × B_i × A_i)
```

The folding operation should be performed once and the resulting matrix rounded to BF16 once.

The tool should not implement runtime LoRA execution.

It should also reject adapter formats that contain additional runtime architecture or inference semantics that cannot be represented solely as modifications to existing H3 weight matrices.

---

# 2. Folder Layout

Add:

```text
lora/
├── README.md
├── fold_lora.py
├── requirements.txt
└── tests/
```

Do not place generated models, downloaded LoRAs, virtual environments, or temporary checkpoint data under version control.

Add relevant patterns to `.gitignore`, for example conceptually:

```text
lora/.venv/
lora/downloads/
lora/output/
lora/*.safetensors
```

unless the repository has an existing preferred convention.

---

# 3. Python Environment

Keep the folding tool independent of the main C build.

`requirements.txt` should contain only the dependencies necessary for checkpoint manipulation.

Prefer:

```text
numpy
safetensors
```

and avoid requiring:

```text
PyTorch
Diffusers
ComfyUI
MLX
```

for ordinary native-H3 folding.

A minimal macOS setup documented in the README should be:

```text
cd h3cli

python3 -m venv lora/.venv
source lora/.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r lora/requirements.txt
```

This also avoids macOS/Homebrew's externally-managed Python installation problem.

---

# 4. Command-Line Interface

The basic CLI should be:

```text
python3 lora/fold_lora.py \
    --checkpoint <TRANSFORMER_DIRECTORY> \
    --lora <LORA_FILE>[:SCALE] \
    --out <OUTPUT_TRANSFORMER_DIRECTORY>
```

Example conceptually:

```text
python3 lora/fold_lora.py \
    --checkpoint ./models/MiniMax-H3/FL2VA/transformer \
    --lora ./loras/turbo.safetensors:1.0 \
    --out ./MiniMax-H3-Turbo/FL2VA/transformer
```

### Multiple LoRAs

Support repeated `--lora` options:

```text
--lora turbo.safetensors:1.0
--lora realism.safetensors:0.7
--lora another.safetensors:0.5
```

The tool must combine all applicable updates in FP32 before performing the final BF16 rounding.

Do **not** internally implement multiple adapters as:

```text
fold adapter 1
write BF16
reload
fold adapter 2
write BF16
```

because that introduces unnecessary cumulative BF16 rounding.

Instead:

```text
load W BF16
   ↓
convert W → FP32
   ↓
+ update adapter 1
+ update adapter 2
+ update adapter 3
   ↓
round once → BF16
```

---

# 5. LoRA Scale

Each adapter should support an explicit user multiplier.

The effective scale should be:

```text
user_scale × adapter_native_scale
```

where the native scale is derived from LoRA metadata where available:

```text
alpha / rank
```

If:

```text
alpha == rank
```

the native scale is:

```text
1.0
```

which is the case for the currently recommended LarryVRH Turbo adapter. Its model card describes the weights as plain BF16 low-rank updates and recommends the v4 step-600 EMA checkpoint; 6–8 inference steps are currently recommended for its best quality.

The tool must report the effective scale rather than applying implicit scaling silently.

---

# 6. Adapter Input Formats

Internally normalize adapter keys into a canonical representation:

```text
target checkpoint tensor
LoRA A tensor
LoRA B tensor
alpha
rank
```

Keep parsing/naming conventions separate from folding mathematics.

Conceptual architecture:

```text
.safetensors adapter
        ↓
format detector
        ↓
key normalization
        ↓
canonical LoRA targets
        ↓
shape validation
        ↓
folding engine
```

### Native H3 Layout

The first-class format should support targets corresponding directly to H3 checkpoint keys such as:

```text
blocks.N.attn.qkv_proj
blocks.N.attn.out_proj
blocks.N.mlp.*
...
```

with paired tensors such as:

```text
<target>.lora_A.weight
<target>.lora_B.weight
```

The Realism People adapter from fal is useful as a compatibility fixture because its published weights use H3's fused attention layout and are described as ordinary LoRA weights applicable to T2V, I2V and reference-video generation.

### Diffusers / PEFT Layout

Support commonly encountered Diffusers/PEFT forms where practical.

These may expose:

```text
to_q
to_k
to_v
```

separately even though the H3 checkpoint stores a fused:

```text
qkv_proj
```

The normalization layer should map such updates into the corresponding slices of the fused checkpoint tensor.

Do not place this conversion logic inside the matrix-folding core.

### ComfyUI-Style Names

Support common H3 adapter names such as:

```text
lora_unet_blocks_...
```

through a dedicated key-normalization layer where mappings can be established unambiguously.

Unknown keys must not be guessed.

---

# 7. Strict Target Validation

The tool must never silently ignore adapter tensors.

For every adapter, report:

```text
number of LoRA A tensors
number of LoRA B tensors
matched pairs
matched checkpoint targets
unsupported targets
missing pairs
shape mismatches
```

Before any checkpoint modification:

```text
all required adapter targets must be accounted for
```

unless the user explicitly selects a future diagnostic partial-fold mode.

Production behavior should therefore be:

```text
unknown adapter tensor
        ↓
error
```

rather than:

```text
unknown adapter tensor
        ↓
silently ignore
        ↓
apparently successful but incomplete model
```

This is especially important because some H3 adapters contain architecture-specific additions that cannot be represented as ordinary matrix deltas.

---

# 8. Unsupported Advanced Adapters

Not every file marketed as an H3 "LoRA" is a simple foldable LoRA.

For example, Alibaba PAI's PDD H3 acceleration adapters contain ordinary rank-64 backbone LoRA data **plus additional output heads and special inference behavior**. A loader that treats the file as a conventional `W + BA` adapter would omit important parts of the method.

The folding tool should detect known unsupported structures and fail with a message such as:

```text
This adapter contains tensors that cannot be represented as
ordinary H3 weight deltas.

Offline fold_lora.py supports standard weight LoRAs only.
This adapter requires its dedicated inference implementation.
```

Do not partially fold PDD, VDN, ControlNet, or similar architectural adapters.

---

# 9. Checkpoint Discovery

The input should point to an H3 transformer directory rather than requiring the user to enumerate safetensors shards.

The tool should:

1. discover all safetensors shards;
2. parse their headers;
3. build:

```text
checkpoint tensor name
    → shard
    → byte offset
    → shape
    → dtype
```

4. map adapter targets onto checkpoint tensors;
5. determine which shards actually require modification.

Do not load all 33B model weights into RAM.

---

# 10. Copy-on-Write Output

On APFS/macOS, prefer copy-on-write cloning of checkpoint shard files.

The design used by PR #14 is appropriate: clone each source shard and modify only affected tensor byte ranges. The PR reports that this preserves headers, tensor ordering and alignment and can reduce actual disk use of a folded variant to a few GB on APFS despite the much larger logical checkpoint size.

The implementation should:

```text
source transformer/
        ↓
clone directory/files
        ↓
modify affected BF16 byte ranges
        ↓
output transformer/
```

### Cross-Filesystem Behavior

If copy-on-write cloning is unavailable:

* either fall back to a normal copy with a prominent disk-space warning;
* or require a deliberate option such as `--allow-full-copy`.

The README must explain that moving a CoW-cloned folded model to another filesystem may materialize the full logical model size.

---

# 11. Safetensors Preservation

The folding tool should preserve, whenever possible:

```text
JSON header bytes
header length
tensor order
tensor absolute offsets
payload alignment
unmodified tensor bytes
```

Only the byte ranges belonging to tensors actually changed by LoRA folding should differ.

This aligns with the hardened safetensors behavior designed elsewhere in the codebase and avoids accidentally generating a checkpoint whose payload alignment makes it ineligible for zero-copy loading.

After writing, reopen and validate every modified shard using the same structural checks the tool applied before folding.

---

# 12. Folding Precision

For an affected BF16 matrix:

```text
W_bf16
```

perform:

```text
W_f32 = BF16_to_FP32(W)

for adapter in applicable_adapters:
    update =
        effective_scale ×
        FP32(B) @ FP32(A)

    W_f32 += update

W_output =
    FP32_to_BF16(W_f32)
```

The matrix multiplication should be FP32.

Do not calculate the LoRA update in BF16.

---

# 13. Post-BF16 Numerical Validation

PR #14 verifies the algebraic operation before writing, but the production checkpoint ultimately executes the BF16-rounded merged weights.

Add a stronger validation probe.

For selected random vectors `x`, compare:

```text
runtime_reference =
    W_base @ x +
    Σ scale_i × B_i @ (A_i @ x)
```

against:

```text
folded_result =
    BF16(W_base + Σ scale_i × B_i A_i) @ x
```

Report:

```text
max absolute error
mean absolute error
relative L2
cosine similarity
```

The validation should measure the actual BF16 checkpoint that H3 will execute.

Allow tolerances appropriate for BF16, but fail on gross mismatches.

---

# 14. Output Manifest

Create a manifest beside the folded transformer, for example:

```text
h3_lora_manifest.json
```

The inference engine does not need this file for model execution.

Its purpose is provenance and workflow safety.

Include:

```text
schema_version
tool_version
base checkpoint identity/fingerprint
base transformer mode if known: FL2VA / Ref2VA
creation time
adapter filenames
adapter SHA-256 hashes
adapter scales
native alpha/rank metadata
key-format detector used
number of modified tensors
number of modified shards
recommended runtime profile, if recognized
```

For a known adapter profile, also record:

```text
recommended_steps
supported_step_range
recommended_scheduler
reuse_compatible
core_reuse_compatible
notes
```

---

# 15. Known Adapter Profiles

Keep adapter-specific runtime advice outside the generic folding mathematics.

A small profile table can recognize well-known adapter hashes or explicit profile names.

For example:

### LarryVRH MiniMax-H3 Turbo

Recognize the currently recommended:

```text
minimax_h3_turbo_v4_step600_ema.safetensors
```

The current model card describes it as the preferred checkpoint, with 6–8 steps recommended for best v4 behavior and possible trailing/smear at four steps under large fast motion.

Recommended profile:

```text
type: turbo
recommended steps: 6–8
default example: 8
reuse: disabled
core-reuse: disabled
```

The tool itself should not modify the H3 sampler.

The README should instruct users not to combine distilled few-step adapters with reuse/core-reuse unless that combination has been explicitly validated.

---

# 16. Full Model Directory Workflow

The folding tool should modify only the transformer directory.

The README should recommend constructing an H3 model directory by reusing the unchanged model components.

Example conceptually:

```text
MiniMax-H3-Turbo/
├── FL2VA/
│   ├── transformer/   ← folded output
│   ├── tokenizer/     → original
│   └── ...
├── Ref2VA/
│   └── ...
├── video_vae/         → original
└── audio_vae/         → original
```

Where safe and convenient on macOS, symlinks may be used for unchanged directories.

The documentation must clearly distinguish:

```text
transformer checkpoint
```

from:

```text
complete model directory accepted by h3cli -d
```

because users should not needlessly duplicate tokenizer/VAE/text-encoder data.

---

# 17. README Step-by-Step Workflow

`lora/README.md` should be written for users who do not already understand LoRA.

It should begin with a short explanation:

```text
A LoRA stores a small learned change to some of the model's
large weight matrices.

Instead of changing h3cli so it computes the LoRA on every
generation, this tool adds those changes to a copy of the
BF16 H3 transformer once.

After folding, h3cli sees an ordinary model.
```

Then provide explicit steps.

---

## README: Step 1 — Verify Licensing

Before downloads, link to the current MiniMax-H3 license.

The current MiniMax-H3 Community License defines the United States, European Union, United Kingdom and Republic of Korea as excluded territories under the open-weight license and directs users in those locations toward obtaining separate authorization.

The README should say:

```text
Check the current MiniMax-H3 license and the license of every
adapter before downloading, folding, running or distributing
weights.

The tool does not grant additional rights to use either the
base model or an adapter.
```

Do not attempt to provide legal interpretation beyond linking to the current terms.

---

## README: Step 2 — Create Python Environment

Document:

```text
python3 -m venv lora/.venv
source lora/.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r lora/requirements.txt
```

Include a note explaining that a virtual environment avoids macOS's:

```text
externally-managed-environment
```

pip error.

---

## README: Step 3 — Download an Adapter

Provide a few useful current examples.

### A. LarryVRH MiniMax-H3 Turbo LoRA

Recommended general-purpose acceleration adapter:

[LarryVRH MiniMax-H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora?utm_source=chatgpt.com)

Recommend:

```text
minimax_h3_turbo_v4_step600_ema.safetensors
```

Its current model card identifies that file as the preferred version and recommends 6–8 steps for best v4 quality.

Compatibility:

```text
fold_lora.py: supported target
use case: acceleration
typical scale: 1.0
```

### B. fal MiniMax-H3 Realism People LoRA

Useful style/content specialization:

[fal MiniMax-H3 Realism People LoRA](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA?utm_source=chatgpt.com)

It targets realistic human footage, uses ordinary H3 fused attention LoRA weights, supports text/image/reference workflows, and currently uses the trigger word `r34l1sm`.

Compatibility:

```text
fold_lora.py: supported target
use case: realistic people / portrait / cinematic footage
suggested scale from model card: 1.0
lighter effect: approximately 0.6–0.8
prompt trigger: r34l1sm
```

### C. LightX2V MiniMax-H3 Turbo

Alternative acceleration adapters:

[LightX2V MiniMax-H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo?utm_source=chatgpt.com)

The repository currently includes several FL2V/Ref2V few-step variants, including an 8-step FL2V model used by LightX2V Studio.

Compatibility:

```text
fold_lora.py:
    supported only for layouts recognized by the
    implemented PEFT/Diffusers key normalizer

use case:
    alternative few-step acceleration

important:
    select an adapter matching FL2VA versus Ref2VA
```

### D. Alibaba PAI PDD Acc-LoRAs — reference only, not foldable

[Alibaba PAI MiniMax-H3 Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs?utm_source=chatgpt.com)

These are useful 8-step acceleration adapters, but they use PDD and contain additional output-head/inference machinery beyond an ordinary weight LoRA. They must be explicitly listed as **unsupported by `fold_lora.py`** unless dedicated PDD support is implemented.

This example is valuable because it teaches users that:

```text
".safetensors LoRA"
does not automatically mean
"safe to fold with W + BA"
```

### More adapters

Also link to MiniMax's maintained H3 integration index:

[Awesome MiniMax-H3 Integration index](https://github.com/MiniMax-AI/awesome-minimax-h3-integration?utm_source=chatgpt.com)

and remind users to check compatibility before folding.

---

## README: Step 4 — Fold One LoRA

Show:

```text
python3 lora/fold_lora.py \
  --checkpoint ./models/MiniMax-H3/FL2VA/transformer \
  --lora ./downloads/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 \
  --out ./MiniMax-H3-Turbo/FL2VA/transformer
```

Explain each argument.

---

## README: Step 5 — Fold Multiple LoRAs

Example conceptually:

```text
python3 lora/fold_lora.py \
  --checkpoint ./models/MiniMax-H3/FL2VA/transformer \
  --lora ./downloads/turbo.safetensors:1.0 \
  --lora ./downloads/realism.safetensors:0.7 \
  --out ./MiniMax-H3-Turbo-Realism/FL2VA/transformer
```

Explain that all updates are accumulated before one BF16 rounding.

Also warn that adapters trained independently may interact badly; mathematical compatibility does not guarantee good combined output.

---

## README: Step 6 — Assemble the Model Directory

Show how to reuse unchanged model components and place/symlink the folded transformer into a complete directory accepted by:

```text
./bin/h3cli -d ...
```

Provide separate FL2VA and Ref2VA examples.

---

## README: Step 7 — Run H3

For normal style/content LoRAs:

```text
./bin/h3cli \
  -d ./MiniMax-H3-Realism \
  -p "r34l1sm, ..." \
  ...
```

For the Turbo profile, provide an example using 6–8 steps and no velocity/core reuse.

Use 8 steps as the quality-oriented README example, with 6 as the faster alternative.

---

## README: Step 8 — Verify the Fold

Document expected `fold_lora.py` summary output such as:

```text
Adapters:               1
LoRA pairs:           173
Targets matched:      173
Targets unsupported:    0
Shards modified:        8
Effective scale:       1.0
BF16 validation:      PASS
Manifest written:     ...
```

Tell users to treat:

```text
unmatched keys
unsupported targets
shape mismatches
missing A/B pairs
```

as errors, not warnings to ignore.

---

## README: Step 9 — A/B Test

Recommend comparison using identical:

```text
prompt
seed
resolution
frames
steps
scheduler
references
```

between base and folded models.

For a style LoRA, normally keep the model's normal sampling schedule.

For distilled Turbo LoRAs, use their documented few-step schedule.

---

## README: Step 10 — Remove a LoRA

Explain that folding is not intended to be reversed numerically.

To return to the base model:

```text
point h3cli back at the original checkpoint
```

To change adapter scale:

```text
create another folded variant from the original base
```

Do not encourage:

```text
fold +1.0
then fold -1.0
```

because BF16 rounding prevents that from being a perfect inverse.

---

# 18. Dry-Run Mode

Add:

```text
--dry-run
```

This should:

* load adapter metadata;
* detect format;
* map targets;
* validate pairs and shapes;
* locate checkpoint tensors;
* calculate expected modifications;
* estimate output disk behavior;
* print the plan;

but perform no file modifications.

Users should be encouraged to run:

```text
--dry-run
```

when trying an unfamiliar community adapter.

---

# 19. Inspect Mode

Optionally add:

```text
--inspect <LORA>
```

which reports:

```text
detected format
number of tensors
number of pairs
rank distribution
alpha values
target module families
known profile
unsupported extra tensors
likely FL2VA/Ref2VA compatibility
```

This is particularly useful because H3's LoRA ecosystem contains several incompatible conventions.

---

# 20. Atomicity

Do not leave an apparently valid folded checkpoint after a partial failure.

Use an output staging location:

```text
<output>.tmp
```

and only rename/finalize after:

```text
all adapters validated
all shards modified
all post-write safetensors checks pass
all numerical probes pass
manifest written
```

If folding fails, clean the incomplete staging output.

Never modify the source checkpoint in place by default.

---

# 21. Existing Output Protection

Refuse to overwrite an existing output directory unless:

```text
--overwrite
```

is supplied explicitly.

Even with `--overwrite`, never overwrite the source checkpoint if input and output resolve to the same filesystem location.

---

# 22. Hashing and Reproducibility

Calculate SHA-256 for every adapter.

For the base checkpoint, use the existing H3 model fingerprint mechanism where practical or record deterministic shard identities.

The manifest should make it possible to determine:

```text
exact base
+
exact adapter(s)
+
scale(s)
+
folding-tool version
```

that produced a checkpoint.

This also integrates naturally with the stop/resume model fingerprinting already designed elsewhere.

---

# 23. Tests

Testing should not require the full 33B H3 model for basic CI.

Construct miniature safetensors fixtures whose tensor names and matrix shapes mimic H3.

Test:

```text
single LoRA
multiple LoRAs
non-unit scale
alpha/rank scale
fused QKV update
unknown key
missing A
missing B
shape mismatch
unsupported extra tensors
aligned safetensors
unaligned safetensors
CoW/fallback copy behavior
atomic failure
post-BF16 numerical probe
```

Add optional full-H3 integration tests outside normal CI.

---

# 24. Out of Scope

The initial feature does not include:

```text
runtime --lora support in h3cli
dynamic adapter switching during generation
GPU LoRA kernels
Metal LoRA matmuls
LoRA-aware SSD streaming
LoRA training
PDD runtime support
VDN support
ControlNet support
architectural adapters
automatic downloading of arbitrary adapters
```

The important design principle is:

```text
LoRA complexity ends at checkpoint preparation.
```

The inference runtime remains an ordinary H3 runtime.
