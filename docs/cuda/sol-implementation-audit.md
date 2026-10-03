# CUDA SOL implementation audit

Base revision: `9b971fb33fd8ddbc2a88766b84c366e6bb152b4f`. Implementation changes are recorded by the campaign source manifest and binary SHA-256.

Only `h3_gpu_dit_sdpa_bf16` selects SOL. Text, vision, audio, VAE and auxiliary attention continue through their existing paths. Default/Sage IDs 0/1/2 remain unchanged; SOL is 3, with independent recipe/plan version 1 and `CUDA_SOL=1` build support.

The MMA, shared-memory loads and online-softmax update derive from this repository's `h3_cuda_fast_attention.cuh`, itself attributed to QuixiAI/h3.c commit `69172740de9cf1bb12c8718479cefe47eeaf7a19`. Its MIT notice is retained in `src/cuda/cuda_sol.cu`. No additional third-party kernel sources or runtime were imported.

CUDA uses its own policy defaults: Q32/KV64, tau 1, minimum exact fraction 0.75, one dense layer and evaluation, local radius 1, sigma disabled. Q64 is also supported; other tiles, reduction, layer skipping and reuse are rejected. `h3_sol_layout_build` provides the same protected reference, text, audio, target-anchor and continuation ranges used by Metal, with no changes to Metal arithmetic or serialization.

One immutable context reserves 512 MiB before weight-cache admission. The deterministic planner subdivides heads and query blocks; centroids, routing and output scratch are reused. Summaries regenerate each call. Exact and centroid traversals share a single FP32 online-softmax state inside one fused kernel; profiles report their inclusive combined time rather than misleading separate timings. BF16 centroid and probability rounding are part of recipe 1; FP32 accumulators do not make this FP32 SDPA.

Inputs are scanned before any output copy. Nonfinite input prevents every output slab from committing. Summary/output numerical faults stop subsequent commits and fail the request at the existing synchronization boundary. A later numerical fault can leave earlier valid slabs in private scratch/output storage, but no failed render is published as valid. The runtime does not promise transaction rollback of earlier valid output slabs. No host synchronization is added per layer.

Checkpoint extension 38 is required only for SOL and contains every policy value plus recipe/plan. Extension 35 retains the mode identity. Completed AV sidecars use version 4 for SOL; previous formats remain readable. Prepared keys include policy values while default/Sage identities remain unchanged.

The existing still-generation CLI continues to reject acceleration.
The fully protected still-layout check exercises the CUDA operator directly;
it does not widen the public still-rendering support contract.

Validation is frozen in `tests/cuda_sol_acceptance.json` and `tests/cuda_sol_manifest.json`. The eight-hour clock begins before the first remote build verification, not during local implementation. Raw diagnostics remain bounded and are distinct from unfenced whole-render performance results.

The matched render matrix explicitly selects CPU Euler in both arms. This allows R1 to retain inputs and velocities at existing step boundaries, without introducing attention-level diagnostic fences. Its first evaluation is dense in both arms; byte-identical second-evaluation inputs establish a held-out teacher-forced comparison without another render. CUDA device Euler is covered separately by exact checkpoint-resume tests. Calibration QKV captures remain diagnostic and do not substitute for matrix timing.

Development validation found a potential divergent-barrier error path when another CTA reports overflow. The bounded repair snapshots the device fault once per CTA before its first barrier. The pre-repair identity and synthetic results are retained; final qualification uses the repaired binary. Existing sampler tests also required separating rapid filesystem fixture rewrites by 10 ms to avoid same-tick cached identities; production fingerprinting was not changed.

The final continuation audit compares the saved video prefix to the existing
seeded augmentation contract, not directly to its parent. A CPU-only checker
reconstructs the complete original noise and checks separate F32 products
`0.999 * parent + 0.001 * noise` byte for byte. Both render modes pass; the
original incorrect expectation is retained. The initial checkpoint pause
timed out during cold model fingerprinting. A separately recorded preparation
and tiny retry pass with unchanged production binaries, caps and criteria.
Neither correction required rerendering a large case.

Final automated coverage and playback delivery pass. The report preserves
unknown resume phase timings as unavailable, scopes real routing comparisons
to exported windows, and does not infer full-mask coverage from the original
oracle's placeholder flag. Two-step visual quality, other architectures and
Sage/SOL composition remain outside the resulting qualification.
