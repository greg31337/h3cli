# Long CUDA video capacity estimates

These are capacity budgets from `bin/long_video_tests`, not measured minimums.
The table uses dense BF16, 50 blocks, 512 text rows, no extra reference rows,
2 GiB for other allocations and 10% capacity headroom. Figures round **up** to
whole GiB. Actual admission uses the effective conditioning and current free
host/device memory; transformer streaming does not eliminate host weight storage.

| Canvas | Frames / duration | Video T / audio T | Resident capacity | Streamed capacity |
| --- | --- | --- | ---: | ---: |
| 256×256 | 464 / 19.333 s | 137 / 773 | 48 GiB | 9 GiB |
| 256×256 | 481 / 20.042 s | 142 / 802 | 48 GiB | 10 GiB |
| 256×256 | 736 / 30.667 s | 217 / 1,227 | 50 GiB | 12 GiB |
| 1344×768 | 719 / 29.958 s | 212 / 1,198 | 124 GiB | 86 GiB |
| 1344×768 | 736 / 30.667 s | 217 / 1,227 | 126 GiB | 88 GiB |
| 1920×1088 | 736 / 30.667 s | 217 / 1,227 | 209 GiB | 171 GiB |

For a width/height multiple of 32, video rows are `T*(width/32)*(height/32)`.
Total rows add two audio channels, text and reference rows. The conservative
DiT reserve is `rows*358400 + 1 GiB`; resident core weights use 38,535,168,000
bytes and two streamed slots use 1,541,406,720 bytes. Add other live allocations,
then divide by 0.9 for capacity. Do not add capacity headroom twice.

The runtime campaign separately preserves `max(1 GiB, physical VRAM/10)` free
reserve, checks actual free VRAM, rejects estimates above 60 GiB and requires
at least 32 GiB currently available host RAM. Plan around 64 GiB system RAM
when using the existing 40 GiB streamed host weight cache. Bounded full-VAE
host delivery estimates are about 2.04 GiB at 256×256, 2.79 GiB at 1344×768,
and 3.65 GiB at 1920×1088; they do not scale with full RGB movie length.

The 1344×768 and 1920×1088 near-30-second budgets exceed a 72 GB card, even with
fully streamed transformer weights. A larger frame ceiling does not promise
those combinations will fit. The 736-frame 3840×2176 upscale is additionally
rejected by the existing sequence ceiling, independently of memory.
Only the 256×256/464-frame row is exercised as a long runtime case in this
milestone. Measurements are reported separately in [the acceptance report](long-video-results.md).
