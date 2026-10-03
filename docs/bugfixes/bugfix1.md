# Runtime robustness: release notes and validation

Shader discovery, safetensors storage, and explicit VideoVAE tile validation now
follow `design-bugfix1.md`. Normal M4 generation retains its numerical behavior.

## Release notes

- Launching `h3cli` from another working directory locates `src/metal/shaders.metal` beside
  the executable when the file is absent from the current directory.
- Valid unaligned safetensors use correctly aligned copied buffers when they
  cannot safely be mapped. Input checkpoints are never rewritten. Malformed
  ranges, lengths, headers, and metadata fail before payload loading.
- `H3_VAE_TILE_PIXELS` supports unset = 256, `auto` = 256..320, and explicit
  256/272/288/304/320. Other explicit values, including empty strings, fail with
  the requested value, supported range, and recommended/released default.
  Tiles above 320 are unsupported because larger tiles have demonstrated severe
  grid/quilt reconstruction artifacts. No silent clamping or automatic fallback
  occurs for invalid explicit values.

## Shader discovery

`h3_shader_resolve()` is called centrally by `h3_gpu_create()`. Qwen text,
Qwen vision/multimodal conditioning, DiT, VideoVAE encoding/decoding and resident
previews, and AudioVAE encoding/decoding all use this entry point. No subsystem
implements a competing lookup policy; the device-only Metal probe reads no shader.

Precedence is `H3_SHADER_PATH`, supplied absolute path, supplied relative path in
CWD, then that relative path beside the executable. The default name is
`src/metal/shaders.metal`. Explicit overrides (including empty values) fail without
falling through. Absolute paths are preserved exactly. A present but unreadable
CWD file also fails rather than selecting different shader contents.

The helper dynamically sizes `_NSGetExecutablePath()` storage, canonicalizes
the executable before extracting its directory, and allocates joined paths
without a fixed-size pathname buffer. Executable symlinks resolve to the actual
binary's directory. Each context resolves its path through this helper; there
is no process-global cache that could become stale after an environment/CWD
change. Missing-file errors include both attempted locations.

## Safetensors audit and ownership

The parser validates `8 + header_size`, signed file-offset representability,
absolute tensor bounds, shape products, dtype byte lengths, duplicate fields and
names, contiguous nonoverlapping ranges, JSON syntax/UTF-8, and string-valued
metadata. Low-level copied/mapped GPU loaders and streaming BF16 reads also
`fstat()` the opened descriptor and validate ranges before touching storage.
The host read API checks the current file size, including files truncated after
header parsing. Empty tensors and scalars remain supported.

Safetensors offsets describe bytes relative to the data section. Header padding
is independent of optimization eligibility. The file format permits JSON padding
and does not require Metal-compatible tensor addresses. See the
[official safetensors format](https://github.com/safetensors/safetensors/blob/main/README.md).

Previously, the loader page-aligned the mmap offset but passed `mapping + delta`
and the unrounded tensor length to Metal. That did not satisfy
[Metal's no-copy buffer requirements](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)).
`h3_st_zero_copy_plan()` now checks each actual tensor offset, dtype, size,
file bounds, page alignment, rounded allocation length, and device buffer limit.
An eligible buffer begins at a page-aligned tensor offset; the mapping and Metal
allocation cover whole pages, while all tensor operations retain their original
logical byte count. The final partial file page is never consumed as tensor data.

Fallback is **per tensor**. A shard may own both mapped buffers (released by the
Metal buffer's `munmap` deallocator) and ordinary allocated buffers. An ineligible
mapping or unavailable mapping allocation uses the existing direct `pread` into
aligned Metal storage, without an intermediate weight array. `H3_PROFILE=1`
prints the file, tensor offset, and reason for per-tensor fallback. Ordinary
verbosity remains quiet. Checkpoint files must remain unchanged while loaded,
as with any file-backed mapping.

`H3_ZERO_COPY_WEIGHTS` selection remains unchanged: `1` requests mapping for all
weights, `transformer` for transformer weights, and `0` disables it. Unset requests
transformer mapping on M5 and copying elsewhere. Eligibility is now enforced
for every request. The audited FL2VA checkpoints contain 3 eligible Qwen tensors
(94,382,080 bytes), and no page-aligned transformer, VideoVAE, or AudioVAE tensors.
Thus even an 8-byte-padded official file may need copying. This is an intentional
correctness tradeoff; M5 memory/performance may change and was not measured.

The initial safety-only implementation rejected the odd-offset fixture with
“may be valid safetensors but cannot safely use the current zero-copy loader”;
the padded fixture mapped and returned exact values. That guard was tested and
its source/logs retained under `outputs/bugfix1-validation/stage1-*`, then replaced
by the final copied fallback. Shard-wide rejection/fallback was not selected.

## Reproduction and evidence

Use `make test-bugfix1`, `make test-bugfix1-sanitize`,
`make test-video-vae-tiles`, `make test-video-vae-tiles-sanitize`, and `make test`.
GPU tests need a Metal-capable process. Full checkpoint copies and expensive
quality runs are opt-in development utilities:

```sh
python3 tests/bugfix1_repack.py models/MiniMax-H3/FL2VA outputs/bugfix1-validation/unaligned-model/FL2VA
python3 tests/bugfix1_integration.py
python3 tests/bugfix1_integration.py --unaligned
python3 tests/bugfix1_weights.py
outputs/refvideo-encoder-validation/venv/bin/python tests/bugfix1_quality.py \
  --baseline outputs/bugfix1-validation/baseline/bin/tilefix_decode
```

The integration/quality scripts require a separately captured pre-change baseline;
they never synthesize expected outputs using the implementation under test.
Baseline binaries here came from commit `1959b14` before edits. The quality script
uses that isolated old decoder for 512-pixel evidence; production has no bypass
for oversized tiles. `bugfix1_repack.py` writes a separate explicitly supplied
destination and adds JSON padding only, retaining original tensor descriptors
and payloads. Its manifest records payload SHA-256 and file sizes.

Validation details and measurements are recorded in [the M4 results](bugfix1-validation.md).
