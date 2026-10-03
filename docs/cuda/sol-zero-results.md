# SOL minimum exact fraction: default zero

`--sol-min-exact` now defaults to **0 on Metal and CUDA** (previously 0.1 and
0.75 respectively). CLI help and the default-policy tests are updated. Zero
removes the minimum exact-block quota; content-based routing, local exact regions,
protected conditioning/continuation ranges and existing early dense policies remain.
Explicit values and saved sampler policies keep their existing semantics.

[Paired playback and detailed timings](../../outputs/sol-zero/review.html) ·
[Machine-readable summary](../../outputs/sol-zero/summary.json) ·
[Checksums/report audit](../../outputs/sol-zero/report-audit.json).

## Two quick CUDA comparisons

RTX PRO 5000 72 GB on the qualification host, NUMA0, BF16 weights, 243 frames, two
actual denoising evaluations, full VAE, no references, same piano prompt/seed.
One adjacent pair per resolution: explicit previous `0.75`, then the new default
with `--sol-min-exact` omitted. No quantization or other parameter changes.
These timings compare SOL settings, not SOL versus default dense attention.

| Resolution | Wall, 0.75 → 0 | Wall speedup | Denoising, 0.75 → 0 | Denoising speedup |
| --- | ---: | ---: | ---: | ---: |
| 640×480 | 101.51 → 93.11 s | 1.090×; 8.3% less time | 30.42 → 23.68 s | 1.285×; 22.2% less time |
| 1344×768 | 339.80 → 250.57 s | 1.356×; 26.3% less time | 237.70 → 150.15 s | 1.583×; 36.8% less time |

The first CUDA evaluation stays dense. Its times were 10.73 → 11.01 seconds
at 640×480 and 66.30 → 66.75 seconds at 1344×768. The routed second evaluation
improved from **19.69 → 12.67 seconds (1.554×)** and
**171.41 → 83.40 seconds (2.055×)** respectively. Step times are rounded console
values; total denoising times use stage telemetry.

Both candidates executed all 49 eligible routed layers, reported zero recovery
fallbacks and zero quota-forced counts. The exact fraction within routed work
fell from 83.9% to 34.7% at 640×480 and from 82.4% to 29.4% at 1344×768.
Protected/local counts matched across each pair. Zero does not mean zero exact
attention. The full render output and actual dispatch counters passed validation.

These are single-pair observations, not repeated-run statistics. Two-step clips
check execution and timing, not converged visual quality. No new quality-equivalence
claim is made. Metal defaults/build/policy checks passed, but rendering speed was
measured only on CUDA.

## Validation and provenance

Local Metal executable and CUDA executable built successfully. Both host policy
suites passed: 114 CUDA SOL policy/planner assertions and 430,919 shared Metal
policy/protected-layout assertions. All sixteen frozen suite files remain unchanged.
The full BF16 regression gate was **skipped at the user's explicit request**;
the previous M6/M7 gate does not qualify this new source. That one-run exception
does not change the repository's normal regression requirement.

Source: `63a4800f72e13921d670bf383ae334e15f31d55deba1857297bd2d5c7e54bbad`.
The independent full-render guard permits a ceiling of six; each command still
requests and validates exactly two evaluations. All 40 downloaded records and
playback assets passed checksum verification, with no missing report links.
