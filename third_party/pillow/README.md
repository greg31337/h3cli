The reference-only Lanczos resize in `src/sglang_media.c` adapts the coefficient
calculation and 22-bit, two-pass byte rounding from Pillow **11.3.0**:
https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c

Pillow's MIT-CMU license and attribution are retained in LICENSE. The production
renderer does not import Pillow or execute Python. JPEG input uses the stable
TurboJPEG v2 decoder ABI (accurate integer DCT and default fancy upsampling).
The qualification server uses Ubuntu's libturbojpeg 2.1.5 runtime, extracted
into the user-owned validation directory because system package installation
requires a password. No system library was replaced.
