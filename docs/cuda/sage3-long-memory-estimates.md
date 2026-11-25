# Sage3 long-video capacity estimates

These are host-only planning estimates, not measured allocations or fit guarantees.
They use 512 text rows, no reference rows, all 50 blocks, two GiB of other live
storage and the existing conservative future-buffer model. Sage3 adds its full
512 MiB workspace. Dividing live bytes by 0.9 budgets the physical 10% reserve
once; the actual admission code uses current free VRAM and max(1 GiB, 10%).

Projection formats retain BF16 Q/K/V. NVFP4 bounds projection/MLP slabs; it does
not quantize retained full-sequence tensors. Cold columns include conversion
scratch. Warm columns assume valid packed artifacts. Compressed streaming also
retains approximately 10.54 GiB of packed host weights, plus host conditioning,
sampler arrays and decoder buffers; host admission is independent of VRAM.

## Maximum-duration planning (1,450 frames)

| Canvas | Joint rows | BF16 resident / stream GiB | NVFP4 warm resident / stream GiB | NVFP4 cold resident / stream GiB |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 32,674 | 55.9 / 17.6 | 28.6 / 17.4 | 29.0 / 17.8 |
| 640×480 | 133,446 | 93.3 / 55.0 | 66.0 / 54.8 | 66.4 / 55.2 |
| 960×544 | 223,116 | 126.5 / 88.2 | 99.2 / 88.0 | 99.6 / 88.4 |
| 1024×768 | 333,282 | 167.4 / 129.1 | 140.1 / 128.9 | 140.5 / 129.3 |
| 1344×768 | 435,762 | 205.4 / 167.1 | 178.1 / 166.9 | 178.5 / 167.3 |
| 1920×1088 | 876,426 | 368.8 / 330.5 | 341.5 / 330.3 | 341.9 / 330.7 |

The 72 GB card does not fit every supported combination. Reference rows add
both retained activations and attention work; a single reference can change
admission. Do not subtract memory based on the internal FP4 operand width.

## Workspace and intermediate-duration coverage

The [machine-readable report](sage3-long-memory-estimates.json) contains all
30 shape plans: six canvases at 379, 464, 736, 1,433 and 1,450 frames. It records
padding, query chunk, head group, planned workspace and BF16 Q/K/V bytes,
plus sampler/decoder host storage, BF16 and packed host weight caches,
NVFP4 cold host preparation and bounded activation/MLP/output scratch.
Generate it with `make bin/sage3_capacity && ./bin/sage3_capacity`.

At 1920×1088 / 1,450 frames the plan has 876,426 joint rows, 876,544 padded
rows, 6,400 query rows and one head per group. Its planned workspace is
534,845,696 bytes. Three retained BF16 Q/K/V tensors alone require about
35.1 GiB; the conservative budget includes many other simultaneously live
buffers. At the one-million-row boundary all pointer offsets and descriptors
are checked using metadata, without allocating those tensors.

The planner repacks complete K/V once per head group and visits every logical
key for every query chunk. This bounds scratch, not quadratic arithmetic cost.
The original weight-only NVFP4 estimates in [the task list](../todo.md) remain historical.
See the [contract](sage3-long-contract.md) and [bounded results](sage3-long-results.md).
