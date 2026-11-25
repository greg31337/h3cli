# CUDA 1,450-frame and bounded NVFP4 qualification

Completed **2026-11-24**. All 32 tasks in the [task list](../todo.md) are complete
within the authorized test envelope. The final standalone Linux artifact passed
**204/204 exact golden output hashes** in **172.04 seconds**. The implementation
and supported combinations are described by the [contract](long-video-1450-contract.md).

The temporal implementation and bounded NVFP4 implementation each passed the
unchanged 204-output regression. Golden manifest SHA-256 remains
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.
All 8,084 AudioVAE descriptor queries through 1,450 frames passed. The host-only
[capacity report](long-video-1450-memory-estimates.md) is separate from measurements.

On RTX PRO 5000 SM120, four bounded projection families and the two-pass MLP
matched retained NVFP4 operator outputs exactly for the tested fixtures in
resident and streamed modes. Tests also covered global outliers across chunks,
tails, zero/nonfinite inputs, cache-header corruption, cancellation recovery
and stable allocation counts. Actual production GEMM descriptors were queried through the
maximum logical sequence without full-size tensor allocation.

Initial 256×256, 464-frame, two-evaluation results:

| Mode | Generation wall time | Peak device use | Full replay | Preview replay |
| --- | ---: | ---: | ---: | ---: |
| BF16 streamed | 67.13 s | 5,568 MiB | 7.69 s | 3.95 s |
| NVFP4 streamed, cold packed cache | 87.34 s | 5,570 MiB | 7.72 s | 3.95 s |
| NVFP4 resident, warm packed cache | 43.31 s | 13,594 MiB | 7.67 s | 3.98 s |

The same runs give the following phase measurements. CUDA tracked peaks are
per backend context; the device-use table above includes other live allocations.
Device use and process RSS are sampled every 0.5 seconds. RSS is summed over
descendants and can count shared pages more than once.

| Mode | Core preparation | Two denoising evaluations | Tracked DiT peak | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| BF16 streamed | 26.30 s | 7.78 s | 3.932 GiB | 36.80 GiB |
| NVFP4 streamed, cold | 36.12 s | 19.43 s | 2.584 GiB | 11.02 GiB |
| NVFP4 resident, warm | 11.63 s | 4.24 s | 12.597 GiB | 1.34 GiB |

Cold NVFP4 prepared all 200 projection matrices; the warm run loaded all 200
from the unchanged packed-weight cache. Each NVFP4 run dispatched 1,000 native
NVFP4 slab GEMMs under execution recipe 5, including FC1 recomputation. Audio
VAE took 0.55–0.57 s with a 0.578 GiB tracked peak. Full video VAE took
6.00–6.55 s with a 4.998 GiB tracked peak independently of the DiT context.
The decoder therefore determines the device peak for streamed runs here.

These rows include preparation and decoding and have different cache/placement
conditions; they are not a controlled speedup comparison. Resident and streamed
NVFP4 AV latents were byte-identical. Full replay matched each source MP4.
Local Metal 256×256/22-frame generation and AV replay matched byte-for-byte.

## Connected workflow checks

BF16 and NVFP4 passed 464-frame prompt-only, first/last-anchor and mixed
image/audio/silent-video generation, plus video-with-soundtrack conditioning.
The final native build also repeated each precision’s exact pause/resume check.
Native full replay reproduces each MP4; preview replay retains the same frame count and timing. Each precision's
pause/resume result matches its uninterrupted AV latents exactly.

An injected 8 GiB tensor-allocation budget forced resident NVFP4 loading to
fail during block 39. The existing automatic retry released that attempt and
completed with compressed streaming. Its video and audio latents matched the
normal streamed NVFP4 run exactly; observed device use peaked at 8,806 MiB.
Cache-header corruption, source stamp immutability, load cancellation and subsequent recovery were also checked without modifying
model files.

Both hard and bridge continuation preserve the protected source rows exactly
for a 447-frame context and 464-frame target, in prompt-only and mixed-reference
BF16/NVFP4 cases. Reference continuation uses matching Ref2VA source states;
the existing incompatible model/VAE-signature rejection is preserved. Delivery
contains 17 new frames; joining it to the 447-frame source produces 464 frames, with complete audio.
Script jobs retain NVFP4 in child commands and presentation metadata, including
four short seed variants and their bounded merges.

The API tests reuse a context, compare repeated seeded AV latents and separately
decoded finite PCM, enforce allocation plateaus and host-memory denial, cancel
weight loading, and recover in the same context. NVFP4 additionally switches to
a short BF16 request in that context. Device memory stayed below the development
ceiling. The NVFP4 API sequence peaked at 38,090 MiB during its switch to
resident BF16; the repeated NVFP4 requests held RSS at 964,005,888 bytes and
changed free device memory by only about 2 MiB. The HTTP fixture uses a relative
cache name, as required by the server's job-local artifact policy. Both HTTP video requests and the
legacy still control completed successfully. Full/preview decoder cancellation,
encoder failure, output preservation, and subsequent recovery passed for each
precision. Completed sampler states remained loadable after post-denoise
encoding failure.

BF16 256×256→512×512 K=0 upscale compatibility passed for both hard and bridge
continuations, with 39-frame sources, 56-frame targets and 56-frame final merges.
Audio and source identity checks remained exact, native states remained the
continuation inputs, and K=0 workers ran no DiT refinement evaluations. NVFP4
AV-state K=0 transfer and invalid AV refinement were rejected before model loading.
Independent K=2 refinement also passed on two 22-frame clips, with a 44-frame
merge. No long or production-resolution upscale was executed.

## Host and descriptor coverage

The local and CUDA host suites cover frame/seconds boundaries through 1,450,
1,416-frame continuation contexts, maximum native/upscaled sequence counts,
NVFP4 padding and capacity arithmetic, and unsupported backend/precision
combinations. Fifty render-parser tests cover duration planning, reference
inheritance and comma lists, seed sets, and NVFP4/upscale exclusions. Six guard
tests reject hidden oversized canvases and checkpoint durations. The sampler
suite passed 22 tests containing 257 adversarial container cases; temporal
section versions retain their fixed v1/v2/v3 sizes.

Audio descriptor queries cover 86 aligned durations, 94 convolution shapes each.
The [descriptor record](long-video-1450-audio-descriptors.json) retains all
86 pre-convolution workspace queries, including their stepwise changes. The
qualified algorithm choices remain fixed; no timing-based selection was added.
The peak requested workspace is 2,176,942,080 bytes, within the retained 3 GiB
cap; the peak layout is 123,750,400 bytes, within the new 128 MiB cap. No maximum
movie/PCM/decoder-workspace allocation was made. NVFP4 queries use the four real
H3 projection N/K pairs and bounded M/tail descriptors through a one-million-row
logical sequence. Operator fixtures use the production 8,192-row chunk with
small feature widths; all tested outputs matched exactly. The operator contract
permits a final BF16 rounding difference from a different cuBLASLt M descriptor;
this tolerance is separate from the unchanged exact 204-output golden gate.

Local Metal passed 20 early CLI/indirect-state rejections without models,
HTTP rejection of CUDA-only durations, and script validation. The final offline
macOS package rendered/replayed a 256×256/22-frame/two-evaluation clip exactly
and rejected long duration/NVFP4 requests before model loading. The development
package is ad-hoc signed and unnotarized.

## Artifact provenance

The final inference/test-source fingerprint is
`64586f9e9bf138ac6da1da50e7b623575df69302439f8c91f859cce2be20fc64`.
Packages use the repository's locked, previously qualified dependencies. Models
and packed-cache locations are supplied explicitly for offline smoke checks.

- Linux standalone SHA-256:
  `c97421d1521b7c7a5773ca1f5099f57633477706d29cf50e5575c6b7a9fa60e6`.
- macOS standalone SHA-256:
  `840e89ec3ec5e0249d59de335a8bdbf3e73bc520e4c566ff17d43be73599d556`.

Both artifacts passed offline native smoke tests. Linux additionally generated
464-frame NVFP4 output and passed exact full replay plus preview replay.
Packaged K=0 continuation upscaling also passed with exact audio and 56-frame
merges. Packaged and native NVFP4 AV latents matched byte-for-byte.
The final unchanged 204-output gate passed in single-file mode: 204 expected
artifacts were present and matched exactly, with no skips, tolerances or golden
changes. Final documentation is included in the outer source archives; the
qualified executables remain byte-identical.

## Runtime coverage limit

General runtime feature tests used 256×256 and at most 464 frames. The larger
runtime canvases are the separately approved immutable regression and short
BF16 upscale compatibility tests. **No end-to-end 1,450-frame render or decode
has been performed.** Motion quality and scene consistency are outside acceptance.
Raw machine records and logs remain under ignored `outputs/`.
