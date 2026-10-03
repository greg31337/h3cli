# External native-Metal reference

The upstream [VPIPE H3 guide](https://github.com/tgo-app-dev/vpipe/blob/main/docs/MINIMAX-H3.md)
(accessed 2026-09-20) reports the following M4-family H3 measurements:

| Hardware | Workload | Reported denoising | Reported wall |
| --- | --- | --- | --- |
| M4 Pro MacBook Pro, 64 GB | 960×544, 124 frames, 4 steps, 8-bit checkpoint, dense, GPU alone | 602 s | 857 s |
| Same machine/configuration plus ANE FFN/QKV | Same geometry and steps | 443 s | 706 s |
| M4 Pro Mac mini, 64 GB, external Thunderbolt model SSD | 960×544, 124 frames, 6 steps, Turbo LoRA, 8-bit | Not separated in that row | 1304 s |

These establish published native-Metal reference workloads, not a speed ratio
against this repository's M4 Max 128 GiB, BF16, 640×480, 243-frame B1/B5/B6
workload (the original M0 evidence used 362 frames).
Hardware, quantization, model/adapter, frame count, decoder and startup differ.
The ANE row additionally changes arithmetic. SOL and Sage must be disabled for
any dense BF16 comparison. A local matched VPIPE run would need identical
weights, prompt/seed, geometry, full layer/evaluation counts and decoder, with
conditioning and denoising timing reported separately. No local VPIPE timing is
claimed by this report.
