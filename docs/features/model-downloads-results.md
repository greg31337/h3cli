# Native model download qualification

Status: **complete — 36/36 tasks**. Qualified on local M4 and RTX PRO 5000 on
2026-09-30. See the [user guide](model-downloads.md), [dependency inventory](model-download-dependencies.md)
and [completed checklist](model-downloads-tasks.md). No cloud node was used.

## Catalog and installation

The embedded catalog contains 119 pinned artifacts. Catalog SHA-256:
`3320575ec56558aa54d39bb7473ff721bd0b30cbfb611eb81e0fce9f6ceb72c4`. All 115 files in the previously qualified main model
matched the published revision by full SHA-256 (198.484 seconds). Existing
models were retained as read-only inputs. The new PRO 5000 installation is
`/path/to/h3cli-download-work/final-models`; pass that directory to `--models-path`.

| Component | Source revision | Bytes |
| --- | --- | ---: |
| MiniMaxAI/MiniMax-H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` | 288,102,039,815 |
| madebyollin/taehv | `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c` | 22,709,752 |
| Mamad8/MiniMax-H3-Image-VAE | `c7b9252c73707dba494cf4d99ca45d3f33f561b3` | 5,207,808,784 |
| LBH-123-AI/Minimax_h3_latent_Upscaler | `3f941d5d182014dd5c0a5e16330420ee2d4aa0c6` | 690,592,992 |

The [catalog](../../src/models/catalog.json) records every source URL and complete-file hash.
The unique payload budget is 216,229,637,924 bytes; the complete installation is
**294,023,151,343 logical bytes**. The independent post-download audit checked
all 119 full hashes and sizes, and confirmed no writable hardlinks. Reported
allocated blocks total 294,023,417,856 bytes when shared CoW extents are counted
for each file; this is not unique physical disk consumption.

## Real transfer and recovery

The all-group acquisition started from an isolated empty tree. A native build
was interrupted after 60.125 seconds, preserving large partials. The packaged
resume started with 6,084,319,591 partial bytes and completed in
1963.516 seconds, transferring **209,528,385,976 additional bytes**
and reusing **77,791,836,081 bytes** through verified local CoW/copy.
Peak process RSS was 192,664 KiB. Counters measure payload delivered by curl,
excluding HTTP/TLS overhead. Earlier completed files and saved prefixes explain
why this resumed transfer is smaller than the empty-tree unique budget.

Real traffic exposed and fixed redirect-body size handling, redirect Location
capture after an aborted body, and curl 7.81 Range errors on an origin 302 before
the CDN. An earlier package failed that redirect-resume case; retained partials
were reused successfully after the fix. No failed attempt is counted as a pass.

After the final custom-mode adoption fix, the final executable performed an
additional real interruption/resume of the 5.2 GB image VAE, while reusing every
other installed catalog group. This preserves the main installation and proves
the final artifact’s actual transfer path:

- Interrupted after 22.290 seconds; retained 293,077,609 bytes.
- Resume: 62.529 seconds, 4,914,731,175 additional bytes, peak RSS 242,528 KiB.
- Full image-VAE hash matched; warm offline verification completed in 0.201 seconds with invalid CA paths and no visible GPU.

Evidence: [full acquisition](../../outputs/model-downloads/pro5000-evidence/download-all-v2-result.json),
[final artifact resume](../../outputs/model-downloads/pro5000-evidence/final3-resume/results.json).

## Build and automated coverage

Application, library and test builds passed without project compiler warnings
on M4 and PRO 5000. Complete M4 host/Metal and PRO 5000 current-host suites passed.
Both platforms passed **25 downloader tests** and **23 HTTP/server tests**. The
preceding 24-test downloader suite also passed AddressSanitizer and UndefinedBehaviorSanitizer
on M4 (87.996 seconds); the added custom-neighbor adoption test passed normally
on both platforms. Actual APFS and XFS CoW probes preserved source timestamps
and used separate inodes.

Coverage includes catalog/operation closure, all model-root overrides, offline
behavior, TLS/CA/proxy/auth containment, redirects, retries, Range, corruption,
cancellation, publication/crash recovery, concurrent locks, read-only reuse,
receipts, CoW/copy fallback and custom-mode conflicts. Server tests cover queued
CPU preparation, progress, cancellation/restart, timeouts, managed write roots,
idempotency and resource mutation. Adding unrelated components leaves required
model identities unchanged. Offline policy is excluded from numerical sampler
identity while the existing model/saved-state checks remain enforced.

The complete [current CUDA feature suite](../../outputs/model-downloads/pro5000-evidence/current-cuda/result.json)
passed all 19 commands, including integration, conditioning, placement,
continuation/bridge, memory, operator and decoder recovery checks. The final
change only strengthened installation adoption; its 25-test suite and complete
source gate were rerun afterward.

## Final standalone artifact

The package retains CUDA 13.0, cuDNN 9.20 and production FFmpeg/FFprobe 9.0.2.
Historical 4.2.2 output and 6.1.1 input codecs remain in the private parity kit.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `h3cli-linux-x86_64` | 1,309,517,029 | `75101aa0a88bea4cbbf638a469040e824bf557f211d662952c5a0fedc77a415b` |
| `h3cli-linux-x86_64-sources.tar.gz` | 418,396,175 | `882efdec80276dac9bba3fcfbcc5bf0efebac8a785d676c9238d9e4380197e98` |

Final code/test/build source identity: `6a12713950765ab81e9fd487ca1eb02cfff1109802727823891a81c95b52863c`.
Artifacts, checksum sidecars and `libh3.a` are in ignored
`bin/linux-model-download-release/`. The [source/payload audit](../../outputs/model-downloads/local-release-audit.json)
verified catalog embedding, source identity, fixture retention and exclusions.
The runtime contains 354 manifest entries plus its manifest and materials for
56 shipped components; it includes no model weights, Python interpreter, HF CLI
or external curl executable. Build-tool license notices are retained as notices.
The source archive excludes downloaded weights, credentials and generated build
outputs, while retaining seven small checked-in safetensors test fixtures.

The final rebuild reused unchanged objects and dependencies only from the same
pinned compiler environment. Changed sources and embedded build identity were
rebuilt. Snapshot regression tests ensure `src/models/` is included and generated
`.cuda-build-config` files/top-level models are excluded.

## Preserved SGLang contract

All 12 fixtures and 204 recorded expected hashes remain unchanged. Manifest:
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`. Every gate ran offline without relaxed or skipped cases.

| Gate | Matches | Execution/check seconds |
| --- | ---: | ---: |
| [Final source](../../outputs/model-downloads/pro5000-evidence/parity-final3-source/result.json) | 204/204 | 113.142 |
| [Final standalone](../../outputs/model-downloads/pro5000-evidence/parity-release-artifact/result.json) | 204/204 | 107.312 |

Earlier coherent implementation gates also passed 204/204 (116.398, 105.816,
103.352 and 130.005 seconds). The final standalone gate uses the newly downloaded
weights. Ordinary feature samples use production 9.0.2 codecs.

## Render and server qualification

The [M4 gallery](../../outputs/model-downloads/m4-final-render/review.html) and
[CUDA gallery](../../outputs/model-downloads/pro5000-evidence/cuda-render/review.html)
retain short videos and stills. Both matrices use 256×256 and two steps, with
22 ordinary frames, 56 source/reference/continuation frames, 17 delivered frames
for continuation, and 512×512 upscale output. Media was probed and fully decoded.
These are functional qualification timings with concurrent work, not isolated benchmarks.

| Case | M4 wall seconds | PRO 5000 wall seconds |
| --- | ---: | ---: |
| text-full | 30.110 | 67.515 |
| preview | 21.303 | 29.370 |
| first-last | 27.138 | 34.842 |
| references | 57.747 | 111.675 |
| still | 32.766 | 148.172 |
| still-decode | 8.761 | 76.286 |
| av-decode | 4.828 | 6.181 |
| av-preview-decode | 0.720 | 1.870 |
| conditioning | 19.828 | 19.234 |
| minimal-av-decode | 3.944 | 3.978 |
| continue-hard | 30.485 | 43.775 |
| continue-bridge | 30.404 | 34.788 |
| sampler-uninterrupted | 24.774 | 31.780 |
| sampler-pause | 20.030 | 28.419 |
| sampler-resume | 8.297 | 15.728 |
| upscale | 64.726 | 28.664 |
| upscale-pause | 24.444 | 21.139 |
| upscale-resume | 44.197 | 20.837 |
| server-cli-equivalent | 21.687 | 39.494 |
| manual-identical-weights | — | 89.748 |

Still/full AV/minimal-eight-file decode, sampler resume and upscale resume
outputs matched their direct outputs byte for byte on both backends. Managed
versus manually provisioned identical CUDA weights also produced identical video.

Both real servers automatically acquired an absent 22,709,752-byte preview VAE,
reported preparation progress and completed rendering. After restart offline,
warm jobs passed. Cold server, warm server and equivalent CLI videos were byte
identical on each backend.

| Server run | Complete wall seconds | Model preparation seconds |
| --- | ---: | ---: |
| M4, cold | 22.568 | 0.839 |
| M4, warm offline | 21.509 | 0.031 |
| PRO 5000, cold | 45.643 | 1.361 |
| PRO 5000, warm offline | 38.097 | 0.269 |

M4 additionally passed a warm offline render with both CA overrides set to a
nonexistent file; its video matched the normal preview byte for byte. Saved
states remain tied to their original build/model identity; downloading identical
weights into a new tree does not migrate old checkpoints.

## Linux userlands and CA prerequisite

The same final standalone executable passed under pinned Ubuntu 22.04, Ubuntu
24.04 and Debian 12 userlands on the PRO 5000 host. These are clean userlands
sharing the host NVIDIA driver, not separate kernel/driver qualifications. Each
ran outside a checkout, downloaded the preview VAE over HTTPS into a path with
spaces, then rendered offline using read-only CoW copies of the complete model.
All three output videos were byte-identical. No Python, HF CLI, external curl,
system FFmpeg or CUDA toolkit was installed into those userlands.

The stock images have no CA store. All three first rejected HTTPS and published
no model. Only pinned OS trust data from `ca-certificates_20260601~22.04.1_all.deb`
was then supplied. Bundle SHA-256:
`ecd9dc38bc3efb7dbd6431f57e29d2f8d6a0f0d211e1464b3fef2cbfe266fcd2`.
This is the documented system-CA prerequisite; TLS verification was never
disabled and the application embeds no aging CA bundle.

Evidence: [CA-negative checks](../../outputs/model-downloads/pro5000-evidence/ca-userlands/results.json),
[all distro checks](../../outputs/model-downloads/pro5000-evidence/distro-models/results.json).

## Completion

Catalog regeneration, artifact checksums, documentation links, final diff
whitespace and credential/artifact exclusions passed. Evidence is retained in
ignored `outputs/model-downloads/` and the isolated PRO 5000 work directory.
No release was published and no existing qualified model tree was replaced.
