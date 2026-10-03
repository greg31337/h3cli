// Native BF16 dense attention prototype. Pinned MIT implementation and license:
// third_party/mlx-attention. QK, softmax and PV accumulation remain FP32.
// Candidate A remains unchanged. DiT use requires explicit opt-in;
// standalone numerical qualification is not production performance acceptance.
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
