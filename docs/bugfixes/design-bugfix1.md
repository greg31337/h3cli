## Runtime Robustness: Shader Lookup, Safetensors Alignment, and VAE Tile Limits

### Objective

Improve runtime robustness and failure diagnostics in three independent areas:

1. resolve `src/metal/shaders.metal` relative to the executable when it is not available in the current working directory;
2. prevent silent model corruption when loading safetensors whose tensor data is unsuitable for the current zero-copy/aligned weight path, while ultimately supporting such files through a copied-loading fallback;
3. restrict explicit VideoVAE decoder tile sizes to the validated 256–320 pixel range and reject unsupported larger tiles that can produce quilt/grid artifacts.

These changes should not alter normal model output when using:

* the existing official MiniMax-H3 weight files;
* the default 256-pixel VideoVAE tile size;
* correctly aligned safetensors;
* execution from the source directory.

---

# 1. Executable-Relative Shader Lookup

## Problem

The current runtime receives a shader path such as:

```text
src/metal/shaders.metal
```

and resolves it relative to the process's current working directory.

This means:

```bash
cd /path/to/bin/h3cli
./bin/h3cli ...
```

works, while:

```bash
cd ~/videos
/path/to/bin/h3cli/bin/h3cli ...
```

may fail with an error equivalent to:

```text
cannot compile src/metal/shaders.metal
```

even though the shader file exists next to the executable.

This makes CLI behavior unnecessarily dependent on the directory from which `h3cli` was launched.

## Resolution Policy

Implement centralized shader-path resolution using the following precedence:

```text
1. explicit absolute path, if one is supplied;
2. supplied relative path resolved in the current working directory;
3. supplied relative path resolved against the directory containing the executable;
4. fail with an actionable error listing the locations that were attempted.
```

This preserves current developer behavior while making installed or scripted use independent of the current directory.

### Example

Given:

```text
/opt/bin/h3cli/bin/h3cli
/opt/bin/h3cli/src/metal/shaders.metal
```

running:

```bash
cd ~/renders
/opt/bin/h3cli/bin/h3cli ...
```

should resolve:

```text
/opt/bin/h3cli/src/metal/shaders.metal
```

when:

```text
~/renders/src/metal/shaders.metal
```

does not exist.

## macOS Executable Directory

On macOS, obtain the executable path using an appropriate platform API such as:

```text
_NSGetExecutablePath()
```

and canonicalize the result before extracting the parent directory.

The implementation should correctly handle:

* relative executable invocation;
* symbolic links where practical;
* paths containing spaces;
* long paths requiring a dynamically sized buffer.

## Explicit Override

Optionally support an explicit shader override such as:

```text
H3_SHADER_PATH
```

with the highest precedence.

Recommended final ordering:

```text
H3_SHADER_PATH
    ↓
supplied absolute path
    ↓
CWD-relative path
    ↓
executable-relative path
```

If an explicit override is provided but invalid, fail rather than silently falling through to another shader.

## Scope

Shader-path resolution should be performed once in a shared helper rather than independently by:

```text
DiT
Qwen
VideoVAE
AudioVAE
vision encoder
other Metal-backed components
```

All Metal model components should receive the same resolved shader location.

---

# 2. Safetensors Alignment Safety and Compatibility

## Problem

The current safetensors loader computes the tensor-data base as:

```text
data_start =
    8 + header_size
```

and accepts the result even when the resulting file offsets are incompatible with the assumptions of the zero-copy Metal weight path.

Such safetensors can be specification-valid.

Some serializers or checkpoint-conversion workflows may produce a JSON header whose size is not padded to an 8-byte boundary.

The current behavior can therefore become:

```text
valid-looking model load
        ↓
misaligned zero-copy/typed weight access
        ↓
incorrect tensor values
        ↓
generation continues
        ↓
black or corrupted video
```

This is a severe failure mode because the loader does not necessarily produce an immediate error.

## Design Principle

Correctness takes precedence over zero-copy loading.

The loader should therefore distinguish:

```text
file format validity
```

from:

```text
eligibility for a specific optimized loading path
```

A safetensors file should not ultimately be rejected solely because its tensor data is unaligned if the runtime can correctly load it through a copied path.

---

## Phase 1: Immediate Safety Guard

As an initial correctness fix, detect unsafe alignment before tensor execution begins.

At minimum inspect:

```text
data_start = 8 + header_size
```

and each actual tensor file offset used by the loader.

If a file cannot currently be loaded correctly:

```text
fail immediately
```

with an actionable message instead of allowing inference to proceed.

Example:

```text
model.safetensors:
tensor data begins at unaligned file offset 123457.

This file is valid safetensors but cannot be safely used
with the current zero-copy loader.

Re-save/pad the checkpoint or use a build with unaligned
safetensors fallback support.
```

This safety guard should land before any broader compatibility work if necessary.

---

## Phase 2: Correct Copied-Loading Fallback

The final implementation should accept otherwise valid unaligned safetensors.

For every tensor/shard, determine whether it is eligible for zero-copy mapping.

Conceptually:

```text
safetensors tensor
      ↓
validate bounds/type
      ↓
is file/tensor offset suitable for zero-copy?
      ├─ yes → existing zero-copy path
      └─ no  → allocate correctly aligned destination
               + pread/copy tensor bytes
```

The existing copied Metal-buffer loading path should be reused where practical.

Do not modify or rewrite the user's safetensors file.

## Alignment Eligibility

Zero-copy eligibility should be defined in a central helper.

For example:

```c
int h3_tensor_zero_copy_eligible(
    uint64_t file_offset,
    size_t tensor_bytes,
    h3_dtype dtype);
```

The helper should consider:

* required alignment for the target loading mechanism;
* tensor dtype alignment;
* Metal buffer requirements;
* integer overflow;
* file bounds.

Do not rely only on:

```text
data_start % 8
```

once tensor-specific fallback exists.

The actual tensor address/offset is what matters.

## Shard-Level vs Tensor-Level Fallback

Prefer tensor-level fallback if the architecture supports it cleanly:

```text
aligned tensors   → mapped
unaligned tensors → copied
```

If mixing mapped and copied tensors significantly complicates model ownership/lifetime, shard-level fallback is acceptable:

```text
any unsafe tensor in shard
    ↓
disable zero-copy for entire shard
```

The implementation should document which granularity is used.

## Diagnostics

When fallback occurs, emit a verbose message such as:

```text
model-00002-of-00008.safetensors:
unaligned tensor storage detected;
zero-copy disabled for this shard.
```

Do not treat this as an error once the copied fallback is implemented.

Normal released aligned models should not produce extra messages at ordinary verbosity.

---

## Safetensors Validation

Before either mapped or copied loading, validate:

```text
header length
tensor offsets
tensor lengths
offset overflow
file bounds
dtype
shape-derived expected byte count
overlapping or invalid ranges where applicable
```

A malformed file should still fail even if the copied path could technically read the referenced bytes.

The unaligned fallback applies only to otherwise valid safetensors files.

---

## Performance and Memory Considerations

Mapped zero-copy loading remains preferred when safe.

Copied fallback can increase:

```text
load time
resident memory
memory bandwidth
Metal-buffer allocation
```

especially for large model shards.

This is acceptable because:

```text
correct copied loading
```

is preferable to:

```text
fast corrupt loading
```

The loader should not silently sacrifice correctness to preserve zero-copy performance.

---

# 3. VideoVAE Explicit Tile Maximum

## Problem

The existing explicit `H3_VAE_TILE_PIXELS` configuration permits tile sizes larger than the range used by the automatic tile selector.

For example:

```text
H3_VAE_TILE_PIXELS=512
```

or larger values may be accepted even though observed output can exhibit strong quilt/grid artifacts.

The automatic policy itself only selects values through:

```text
320
```

and the released MiniMax-H3 configuration uses:

```text
256
```

The supported production policy should therefore not imply that values such as 384, 512, or 1088 are valid quality-preserving configurations.

---

## Final Tile Policy

The VideoVAE decoder should support:

```text
default:
    256

auto:
    existing automatic search,
    restricted to 256..320

explicit:
    valid numeric values in 256..320
    satisfying existing alignment constraints
```

Unsupported values above 320 should fail configuration validation.

Recommended behavior:

```bash
H3_VAE_TILE_PIXELS=512 ./bin/h3cli ...
```

produces:

```text
Invalid H3_VAE_TILE_PIXELS=512.
Supported VideoVAE decoder tile range is 256..320 pixels.
The released MiniMax-H3 default is 256.
```

## Reject Instead of Clamp

Do not silently convert:

```text
512 → 320
```

because this makes benchmarking and debugging misleading.

An explicit environment value should mean exactly what it says or produce an error.

Therefore:

```text
>320
    → error
```

is preferred over:

```text
>320
    → silent clamp
```

If backwards compatibility requires a transitional period, a warning-plus-clamp mode may temporarily exist, but it should not be the long-term behavior.

---

## Relationship to Existing 256-Default Design

This work complements the previously designed VideoVAE tile policy:

```text
unset
    → 256

H3_VAE_TILE_PIXELS=auto
    → legacy geometry optimizer

H3_VAE_TILE_PIXELS=<N>
    → explicit tile
```

The new validation simply restricts `<N>` to:

```text
256 <= N <= 320
```

plus the existing tile alignment/multiple requirements.

No changes are required to the default 256 mode.

---

# Combined Failure Philosophy

These three fixes should follow a common rule:

```text
do not continue with a configuration that is known
or strongly expected to produce invalid output
```

Examples:

```text
missing shader in CWD
    → search executable directory

unaligned safetensors incompatible with zero-copy
    → use safe copied path

unsupported VAE tile > 320
    → reject configuration
```

Silent corruption or silent reinterpretation should be avoided.

---

# Compatibility

For standard current usage:

```text
official aligned MiniMax-H3 weights
256-pixel VAE tiles
source-directory execution
```

there should be no numerical change.

The shader lookup affects resource discovery only.

Safetensors fallback affects tensor loading only for files that cannot safely use the current zero-copy path.

The VAE validation affects only explicitly requested unsupported tile sizes.

---

# Testing

## Shader Lookup

Test:

```text
shader in CWD
shader only beside executable
absolute shader path
explicit H3_SHADER_PATH
missing shader everywhere
path containing spaces
executable launched through relative path
```

Verify precedence exactly.

## Safetensors

Construct equivalent safetensors files containing identical tensor payloads but different header padding:

```text
aligned header
unaligned header
```

Verify:

```text
tensor values after load are identical
```

under the final implementation.

Also test:

```text
aligned zero-copy
unaligned copied fallback
truncated tensor data
offset overflow
invalid header
tensor beyond EOF
```

## VideoVAE Tiles

Verify:

```text
unset → 256
auto → 256..320
256 → accepted
272 → accepted
288 → accepted
304 → accepted
320 → accepted
336 → rejected
384 → rejected
512 → rejected
1088 → rejected
```

Also preserve existing multiple/alignment validation.

---

# Rollout

## Stage 1

Implement:

```text
executable-relative shader fallback
safetensors unsafe-alignment rejection
VAE >320 validation error
```

This immediately removes the most dangerous failure behaviors.

## Stage 2

Implement safetensors copied fallback for valid files that cannot safely use zero-copy.

Once that exists, replace the alignment rejection with:

```text
safe fallback + diagnostic
```

## Stage 3

Add parity/performance tests for copied versus mapped safetensors loading.

The final desired behavior is:

```text
aligned
    → zero-copy where supported

unaligned but valid
    → copied loading

malformed
    → reject
```
