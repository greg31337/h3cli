# Long-video capacity estimates

These are conservative host-only estimates, not measured minimums or promises
of a successful render. Reproduce them without loading models or allocating
movie tensors:

```sh
make bin/long_video_capacity
./bin/long_video_capacity
```

The [checked JSON report](long-video-1450-capacity.json) records every byte
component and capacity estimate from this host-only program.

The report covers five canvases at 379, 464, 736, 1,433 and 1,450 frames, all
using the new bounded NVFP4 recipe. Existing short-video recipe 2 retains its
original allocation policy and is not estimated by this new planner. The report
uses 512 text rows, no reference rows, all 50 blocks, 2 GiB for other allocations
and 10% total-capacity headroom. Actual admission additionally sees live
conditioning/model allocations and actual free VRAM, and keeps at least 1 GiB
or 10% of physical VRAM free. Do not add that reserve twice.

At **1,450 frames / 60.416667 seconds**, capacities round upward to whole GiB:

| Canvas | Sequence rows | BF16 resident / streamed | NVFP4 cold resident / streamed | NVFP4 warm resident / streamed |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 32,674 | 56 / 18 | 29 / 18 | 29 / 17 |
| 640×480 | 133,446 | 93 / 55 | 66 / 55 | 66 / 55 |
| 1024×768 | 333,282 | 167 / 129 | 140 / 129 | 140 / 129 |
| 1344×768 | 435,762 | 205 / 167 | 178 / 167 | 178 / 167 |
| 1920×1088 | 876,426 | 369 / 330 | 342 / 331 | 341 / 330 |

NVFP4 reduces resident core weights from 35.89 GiB to 10.09 GiB. BF16 streaming
already retains only about 1.44 GiB of core slots. Full sequence activations,
QKV tensors and attention remain live, so dividing total VRAM by four is
incorrect. The conservative common DiT allowance remains
`358400 * sequence_rows + 1 GiB`; this milestone does not redesign QKV storage.

## Per-phase buffers

The checked NVFP4 planner reports packed core and largest streamed operand,
the largest BF16-to-packed cold conversion, padded activation/output scratch,
MLP slabs, global maxima, host packed-cache storage and the conservative live
DiT allowance. GEMM workspace (32 MiB) and library overhead fit inside the
retained 1 GiB common allowance. The fixed 8,192-row recipe bounds scratch
independently of video duration:

- FC1 plus activated MLP slabs: 0.65625 GiB.
- Maximum padded BF16 output slab: 0.4375 GiB.
- Activation packing: VEC16 scales, aligned payload and global scalars for
  8,192×14,336 values; about 0.0616 GiB.
- Packed stream buffer: largest 28,672×5,376 matrix, about 0.0808 GiB.
- Cold conversion: that packed matrix plus its BF16 source, about 0.3679 GiB.
- Global reductions: 32 bytes, separate from each packed activation's scales.

Admission conservatively sums cold conversion with the future execution
budget even though preparation and MLP execution have different lifetimes.
Warm cache loading avoids conversion but retains this safe admission bound.
The warm column describes lifetimes, not a relaxed warm-only admission rule.
No complete BF16 device core accompanies packed weights.

Compressed streaming retains about 10.09 GiB of host weights plus one packed
staging operand. Cold host admission adds the largest conversion operand.
Sampler snapshots, conditioning copies, pinned BF16 caches and media delivery
have independent host checks. Full VAE delivery retains 22-frame tiles, five
overlap frames and ordered stitching, rather than a complete RGB movie.
The decoder allowance remains 7 GiB. Generated AudioVAE's descriptor peak is
2,176,942,080 workspace bytes and 123,750,400 layout bytes per adapter; production
caps are 3 GiB and 128 MiB, with only requested storage allocated. The retained
FFT pre-convolution workspace jumps from 286,883,840 to 2,176,942,080 bytes at
566 frames and remains at that level through 1,450; the
[86-duration query record](long-video-1450-audio-descriptors.json) preserves
this discontinuity.

Maximum native 1920×1088 has 871,080 video rows. References increase the total
and can exceed the one-million-row limit. Doubling that canvas at 1,450 frames
requires 3,484,320 video rows before text/audio, so learned upscaling is rejected
regardless of available memory or refinement steps.

Measured bounded peaks and phase times belong in the
[acceptance report](long-video-1450-results.md). They do not establish a
60-second speedup or full-duration memory minimum.
