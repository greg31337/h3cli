# Build warning cleanup validation

Validated on 2026-09-29 using the local M4 Max and RTX PRO 5000.
The application and test builds retain their existing warning flags.

The cleanup separates misleading one-line control flow, makes intended numeric
conversions explicit, removes shadowed locals, uses compiler-compatible pragmas,
and selects the curl protocol API supported by the installed headers. JPEG symbol
loading now uses the POSIX function-pointer representation without an ISO C
pointer conversion. Device-name truncation is explicit; server orphan cleanup
compares complete IDs instead of silently truncating directory names.

Sage builds retain fast math and FMA without passing conflicting FMA options.
The vendored headers use qualified dependent template calls, unsigned shuffle
masks, and C++17-compatible lambda captures. Their adaptation hashes and notices
are updated. Torch RNG constants retain the original double-to-float rounding.
The static launcher uses the direct SHA-256 API consistently, avoiding OpenSSL's
one-shot EVP dependency on dynamic loading and network helpers.

Testing also exposed an event-stream race. The server now reads job status before
draining events, so a concurrent completion cannot close the stream before its
completion event is sent.

| Validation | Result |
| --- | --- |
| Clean Metal build, Apple Clang 21 | Zero compiler warnings |
| Metal `make -j8 all test test-server-http` | Passed; all 21 HTTP cases passed |
| Clean native CUDA build, GCC 13.3 / CUDA 13.0.88 | Zero compiler warnings |
| Native CUDA `make -j8 all test test-server-http` | Passed; zero compiler warnings |
| Pinned GCC 11 application/parity/feature-probe build | Zero compiler warnings |
| Static launcher | All 16 tests passed, including a real `noexec` mount |
| Sage2++ / Sage3 before/after comparison | All 16 BF16 outputs byte-identical |
| Complete SGLang regression | All 204 recorded outputs matched |
| Current CUDA feature suite | All 18 functional checks plus build passed; 1,884.42 seconds |
| Real CUDA server jobs | All 16 expected outcomes passed; 594.20 seconds |

The Sage comparison covers random inputs at 17, 129, and 257 tokens in both
head layouts, plus constant and zero inputs. Its baseline is the previously
qualified library. The SGLang gate rebuilt all its probes in isolation with zero
compiler warnings, then completed GPU execution in 95.53 seconds; total wall
time including build was 189.33 seconds. Its fixtures and expected hashes were
not changed.

Normal rendering uses FFmpeg/FFprobe 9.0.2. Only the recorded regression uses
its verified FFmpeg 4.2.2 output encoder and FFmpeg/FFprobe 6.1.1 input tools.

The current CUDA suite covers text/anchor/image/video/audio renders, continuation,
sampler restart, reuse/core-reuse/reduction recovery, quantization/attention
combinations, SOL and SubBlock lifetimes, reference-conditioning caches, streamed
and partial/resident weight transitions, placement changes across resume, GPU
VAE/still/patch operators, memory accounting, state cropping, and decoder failure
recovery. Its build log contains no compiler warnings.

Real server cases M01, M03, M05, M07, M08, M09, M12, and M13 cover queued jobs,
request overrides, artifact downloads, first/last anchors, conditioning reuse,
hard/bridge continuation, upscale inspection and refinement, still latent
round-trips, cancellation, and restart recovery. All 14 successful jobs completed;
the cancellation and interruption cases reached their expected terminal states.
The server's MP4 and AV-state outputs were byte-identical to a direct CLI render.
Every downloaded artifact passed its checksum check; rendered media passed
FFprobe and decoding checks. The smoke harness was given an explicit input root
after its default optional LoRA download directory was absent from the clean
checkout.

Two signal-driven LoRA tests initially failed under PRoot. The identical probe
passed all 37 cases directly on the host, confirming a PRoot stop/resume signal
limitation. Final CUDA host and feature checks run directly on the PRO 5000.

Validation source fingerprint:
`453fd04bbced1ce3ae5694a415b1269525c8ad0c68ba1e78847e39ee50465f23`.
Unchanged golden manifest SHA-256:
`fa6f86cfa9db94b98d599d9044fd22a4a6605191cb48e0b2ca34dd7e022aafe3`.

Local evidence is in `outputs/build-warnings/`, including the clean Metal build
log and the PRO 5000 native build/test logs and parity result. The isolated remote
checkout and complete evidence are under
`/path/to/h3cli-experiments/build-warnings/`.

The tested CUDA executable is `native-source/bin/h3cli` beneath that remote
directory, SHA-256
`168952b841ff1191b1345dd53aac942960b04982eb7f3d6d19bf45b775401d67`.
The local workspace's `bin/h3cli` was also rebuilt successfully without warnings.
