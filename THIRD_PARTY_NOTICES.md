# Third-party notices

The dedicated reference-video posterior generator in `src/vae/video_posterior.c`
implements MT19937 and the PyTorch CPU normal-fill recipe, following PyTorch's
`ATen/core/MT19937RNGEngine.h`, `ATen/core/DistributionsHelper.h`, and
`ATen/native/cpu/DistributionTemplates.h`. The applicable notices are retained in
[licenses/MT19937.txt](licenses/MT19937.txt) and
[licenses/PyTorch.txt](licenses/PyTorch.txt).

The optional VAE oracle imports the official Apache-2.0 Diffusers implementation;
it does not bundle its source or model weights. Its package version and source
hash are recorded with each generated fixture.

The rectangular Morton decoder and the dynamic symmetric int8 quantization /
Metal 4 TensorOps scheduling design in `src/metal/shaders.metal` are adapted from
ccv's Metal FlashAttention `NAMatMulKernel` and `NAInt8MatMulKernel`,
distributed under the following BSD-3-Clause license:

Copyright (c) 2010, Liu Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the authors nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# PyTorch CPU normal compatibility helper

`third_party/torch-rng/avx_mathfun.h` retains Giovanni Garberoglio's zlib license
and attribution to Julien Pommier's SSE implementation. It is adapted from the
PyTorch 2.13.0 installed source header only to remove an ATen include dependency.
See that file and `third_party/torch-rng/README.md` for the notice and provenance.

The reference CUDA normalization kernels in `src/cuda/cuda_sglang.cuh` adapt the
operation/reduction order of PyTorch's `aten/src/ATen/native/cuda/layer_norm_kernel.cu`
and `aten/src/ATen/native/cuda/WeightNorm.cu`, together with
`group_norm_kernel.cu`, `block_reduce.cuh` and `SharedReduceOps.h`
at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`. PyTorch's notices are retained
in [licenses/PyTorch.txt](licenses/PyTorch.txt).

# FlashAttention CUDA reference inference

`third_party/flash-attention/` contains the inference-header dependency subset
from Dao-AILab/flash-attention commit `6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`,
pinned by the PyTorch oracle. BSD 3-Clause license and original attribution
are retained in `third_party/flash-attention/LICENSE` and individual headers.
CUTLASS is an external build dependency under NVIDIA's BSD 3-Clause license,
retained in `third_party/flash-attention/CUTLASS-LICENSE.txt` for binary distributions.
The native C ABI removes PyTorch-only integration; it does not link Python,
ATen, or Torch at runtime. See the directory README for the adaptations.

### Pillow reference image resampling

`src/sglang/sglang_media.c` adapts the Lanczos coefficient and fixed-point byte rounding
recipe from [Pillow 11.3.0 Resample.c](https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c).
Copyright © 1997–2011 Secret Labs AB; © 1995–2011 Fredrik Lundh and contributors;
© 2010 Jeffrey A. Clark and contributors. The full MIT-CMU terms are retained in
[third_party/pillow/LICENSE](third_party/pillow/LICENSE). This path is specific
to CUDA SGLang reference conditioning; Pillow is not a runtime dependency.

### Torchaudio reference soundtrack resampling

The reference soundtrack resampler in `src/sglang/sglang_media.c` follows the installed
Torchaudio 2.11.0 Hann-window sinc recipe in `functional/functional.py`.
Copyright (c) 2017 Facebook Inc. (Soumith Chintala). The BSD 2-Clause terms are
retained in [third_party/torchaudio/LICENSE](third_party/torchaudio/LICENSE).
Torchaudio is a test oracle only; the production implementation is C.

### LBH MiniMax H3 latent upscaler

The native latent-upscaler graph and its opt-in fixture generator follow
[LBH-123-AI's 3D inference implementation](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/tree/40316cf008b2fd8663270669eb4da23f89d2c),
copyright (c) 2026 LBH-123-AI, under the MIT license. The verbatim notice is
retained in [licenses/LBH-upscaler-MIT.txt](licenses/LBH-upscaler-MIT.txt).

The separately acquired BF16 model is published under Apache-2.0, according to
its [pinned model card](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/3f941d5d182014dd5c0a5e16330420ee2d4aa0c6/README.md).
Weights are not bundled with h3cli. The tensor inventory, artifact checksum,
normalization constants and source revisions are recorded in
[tests/upscale/contract.json](tests/upscale/contract.json). Code and model
licenses are separate; the inference repository's MIT notice does not relicense
the weights.
