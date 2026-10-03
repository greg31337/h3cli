# M5 memory lifetime validation

**M5 complete.** Default BF16 and cached SOL B5, including production VAE,
and the final high-resolution single-step safety check all completed. The user clarified the reproduction as **640×480,
362 frames, `inputs/2.jpg`, `--ref-image-size max`, default BF16/MPSGraph, no
memory environment overrides**. Tests use a benign walking-in-a-park prompt.
Every 1344×768 test is restricted to at most two evaluations.

[Playback and memory plots](../../outputs/metal-native-m5-362/review.html) ·
[Verified measurements](../../outputs/metal-native-m5-362/summary.json) ·
[Operator/cancellation records](../../outputs/metal-native-m5-362/operator-records.json).

## Cause and implemented changes

The old common guard checked only whether free plus inactive memory exceeded
10 GiB. It did not cap process memory at 110 GB, ignored compressed process
footprint, and allowed execution if the query failed. Metal cache admission
also always returned success. Those checks could not enforce the limit the
user expected.

The captured packed sequence contains **38,249 rows**, including maximum-size
reference conditioning. On this sequence, MPSGraph graph work reserves approximately **3.29 GB per
encoded DiT block**, beyond the persistent model/activation buffers. The old
M4 full-transformer default encoded 30 blocks per batch. A batch could therefore
build a large temporary working set before a denoising-boundary guard ran.
Completed root command objects were also retained until the final submission.

Repeated-load testing exposed another lifetime defect: copied weight loading
created autoreleased Metal buffer references outside a local pool. Freeing the
tensor purged its physical pages, but those references kept its Metal reservation
alive. In the original cancellation sequence reservations accumulated to
86.050 GB with only about 1 GB physical footprint, causing the new guard to
reject a later load safely. The final `final-v3` build scopes tensor allocation
and loading, scopes additional MPS operations, and explicitly clears cached
tensor graph wrappers on release. The same entire cancellation/reuse sequence
now passes; post-DiT teardown reservations fall to about 0.025 GiB. An isolated
8 MiB copied-weight regression fails against the previous GPU object and passes
against the final build. The cap was not weakened to make it pass.

Long Metal sequences now use five-block batches, or one block above 65,536 rows.
At most two tracked committed batches are retained; completed commands are
retired promptly. If the next workspace would exceed budget, safe queued work
is drained and the budget is checked again. In the protected 362-frame run,
a pressure drain reduced footprint from approximately **69 to 49 GB** and Metal
reservations from approximately **104 to 86 GB**. These are overlapping measures
and must not be added.

The default cap is **110,000,000,000 bytes** (110 GB decimal, 102.45 GiB), lowered
on smaller machines. It checks the maximum of compressed-inclusive process
footprint and Metal reservations, plus known upcoming buffer/workspace reserve.
The separate 10 GiB headroom floor remains. Checks run during planning,
allocation, block execution, encoding and submission. Monitoring failures return
normal errors. `H3_MEMORY_LIMIT_BYTES` can only lower the cap. These are
conservative checks, not an OS quota or a promise that another process cannot
exhaust memory between checks.

Native attention/SOL scratch now has an explicit typed, geometry-checked owner.
Slots and pipeline states are reused across layers/steps. Local autorelease pools
bound encoder and temporary Objective-C lifetimes. A head-major FP16 SOL output
can reuse its exact partial buffer; row-major output retains a separate buffer
to avoid an in-place transpose race. Kernel arithmetic and model precision are
unchanged.

Completed production renders release the DiT context before VideoVAE loading,
even when prepared caching is enabled. Paused renders retain their context for
resume; pressure can evict it before optional preview decoding. The original
one-shot CLI already freed its noncached DiT before VAE, so the reported
`test.sh` growth must not be attributed solely to overlapping transformer/VAE
weights. Production VAE's 256-pixel tiling is unchanged.

## Retained checks

- Host guard: 41 cap, headroom, overflow, lower-only override, failed-query and
  cancellation assertions pass.
- Decoder: 86 before/after cases pass with exact F32 parity, including failure
  cleanup and child cancellation; the same cases pass ASan/UBSan.
- Native lifetime: 96 queued SOL layers across eight synthetic steps pass with
  both output layouts and unchanged warm allocation/pipeline counts. The final
  test also verifies immediate release across eight copied-weight loads.
- Real installed-model cancellation: text, vision, audio/video encoders and
  decoders, three denoiser paths, and populated public-generation caches all
  stop before the next execution and remain reusable after cancellation.
- Independent SOL operator: 513 rows, four heads, head-major input, both output
  layouts; scalar oracle relative L2 0.00183006, protected-row difference zero,
  logical outputs bit-identical between layouts. The aggressive synthetic
  approximation does not pass the separate conservative-quality threshold;
  this test verifies implementation and storage reuse, not a new SOL preset.
- Existing conditioning, sampler, native CLI, protection/layout and record
  validation suites pass. Historical 243-frame unit fixtures remain retained;
  new representative model renders use 362 frames.

Frozen builds, commands, checksums, logs and independent 50 ms footprint samples
are retained under `outputs/metal-native-m5-362/`. The guard-only failure at a
90 GB lower test cap exited normally before unsafe work; the external watchdog
did not terminate it. Default-cap BF16 and cached SOL B5 results follow below.

## Default BF16/MPSGraph B5 (362 frames)

Frozen `candidate2` build, seed 12001, 50 DiT blocks, five evaluated steps,
maximum-size `2.jpg` reference and no memory-limit override. The independent
watchdog uses a 105 GB cap; renderer cap is the default 110 GB. No watchdog
intervention occurred during denoising.

| Evaluation | Wall seconds | End footprint GB |
| --- | ---: | ---: |
| 1 | 300.669 | 47.644 |
| 2 | 298.037 | 47.644 |
| 3 | 299.423 | 47.644 |
| 4 | 299.936 | 47.644 |
| 5 | 299.904 | 47.644 |

There were 640 tensor allocations and 43.217 GB of logical tensor storage
throughout all five steps: no per-step increase. The tracked in-flight command
peak was two. End-step Metal reservations were 82.762 GB; they overlap the
footprint and include reserved workspace. Process compression and system swap
were zero in every end-step record.

Transformer teardown took **0.768 seconds**, reducing footprint from **47.644
to 2.194 GB**. At production VideoVAE load, footprint was **0.721 GB**; loading
took **1.530 seconds** and ended at **9.891 GB**. Decode stayed around 11–12 GB. The complete render finished in **1983.138
seconds**, with **72.553 GB peak sampled footprint** and no watchdog stop. The candidate2 transition records used zero for Metal allocation when
no GPU context was supplied; those zeros are not measurements of device-wide
reservations. The final build queries the device for these transition records.

These five steps demonstrate bounded memory at the requested geometry. They
are not a 50-step performance/quality qualification, nor a matched timing
comparison against an unsafe original render.

## Cached mixed-FP16 SOL B5 (362 frames)

Frozen `final-v2` build, the same saved conditioning and seed, 50 blocks and five
evaluations. Configuration: adapter layout, Q64/KV64, tau 1, minimum exact 0.75,
one dense step, one dense layer and local radius 1; BF16 QKV linears remain on
MPSGraph. A zero-step pause retains the prepared transformer and saves a sampler
checkpoint. Resuming on the same context logs a prepared-cache hit.

| Evaluation | Wall seconds | End footprint GB |
| --- | ---: | ---: |
| 1 | 294.415 | 50.501 |
| 2 | 298.527 | 52.641 |
| 3 | 298.404 | 52.641 |
| 4 | 298.059 | 52.641 |
| 5 | 281.183 | 52.641 |

Dense initialization uses nine native scratch slots (2.743 GB). The first
routed evaluation expands this once to **20 slots / 5.016 GB**, with 16 native
pipelines. Subsequent routed blocks and steps reuse that storage: the tensor allocation
counter stays at 631 and logical storage at 48.233 GB, with no per-step increase. The independent peak
sampled footprint is **77.663 GB**; no watchdog intervention or process
compression occurs. End-step system swap is zero. Tracked committed batches
peak at two.

Every production VAE load callback asserts that the cached transformer has
already been detached. Teardown takes **0.701 seconds**, reducing footprint from
**52.641 to 2.528 GB**; footprint is **0.765 GB** before VAE loading and **9.993 GB**
after its **1.521-second** load. The complete pause/resume/render test takes
**1661.217 seconds**, saving both production MP4 and AV state. Both B5 MP4s have
362 video frames at 640×480/24 fps plus audio.

The `final-v2` build also measures device-wide Metal reservations between contexts:
71.854 GB immediately after DiT teardown despite the 2.528 GB physical footprint.
The later repeated-load regression identified retained copied-weight references
behind this discrepancy; `final-v3` adds the allocation/loading lifetime fix
described above. These B5 records predate that final fix and are retained with
their exact source provenance. The SOL run reuses saved
conditioning, so its total wall time is not comparable to the BF16 run that
generates conditioning. These records qualify memory lifetime; they do not
establish a new SOL speedup.

## Final 1344×768 safety test (362 frames)

Frozen **`final-v3`**, including copied-weight lifetime cleanup; original BF16 /
MPSGraph, maximum-size `inputs/2.jpg`, seed 12001, **114,005 packed rows** and
no memory-limit override. `--steps 2 --stop-after-step 1 --save-sampler-state …`
executes exactly one evaluation across all 50 blocks, then pauses. The renderer
uses its default 110 GB budget and the independent watchdog uses 105 GB.

- **Completed normally:** exit 0, checkpoint saved at completed=1 / schedule=2.
- **Peak sampled footprint: 76.621 GB**, across 44,593 independent samples;
  no watchdog intervention.
- **Step wall time: 2160.394 seconds**; total command wall: 2473.217 seconds.
- From block 2 onward, encoding-boundary footprint stays about **69.8 GB** and
  Metal reservations at **75.689 GB**. The tensor allocation counter stays at
  **631**, with **52.177 GB** logical storage. At most two committed batches
  are tracked; this geometry uses one block per batch.
- End-step footprint is **57.950 GB**, including **1.863 GB** process compression;
  system swap is zero. The guard counts the compressed memory.
- Teardown takes **1.016 seconds**, reducing footprint from **57.905 to 0.713 GB**
  and device reservations from **63.934 to 11.756 GB**. Tensor storage reaches
  zero. This confirms release of the 52.177 GB transformer allocation; framework
  workspace reservations can remain temporarily after component teardown.

The 177,971,193-byte sampler checkpoint is retained with its checksum. This was
a **single-step safety/cleanup check**, not a completed high-resolution video:
it skips the second denoising evaluation and final VAE decode. The two complete
640×480 B5 runs cover production decoding. No unsafe original 135–200+ GB render
was repeated, and no 50-step memory qualification is claimed.

### Additional final-build operator checks

Metal API validation passes the expanded lifetime test, including 12 dependent
MPSGraph projections, queue bounds, resident-plan overflow/rejection and recovery
after a guard failure. The retained mixed-SOL recovery fixture (513 rows, two
heads, Q32/KV64, minimum exact 0.5) passes with scalar-oracle L2 0.00079320 and
zero protected-row difference. Invalid-input commit rejection also passes.

An extra extreme-value fixture (513 rows, four heads, Q64/KV64, minimum exact
0.75, row-major input, `recovery` pattern) fails the existing scalar-oracle gate
in **both** the frozen guard-only build and the final build. Both failures are
retained in `operator-records.json`; this pre-existing corner case is not
relabeled as passing or used to qualify a new SOL preset.
