/* Generated portable kernel wrappers; see scripts/cuda_translate_reference.py. */
int h3_gpu_silu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, input, elements, "SiLU input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "SiLU output") ||
        output->dtype != H3_GPU_F32) return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_silu_f32<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (float *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_silu_f32");
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, input, elements, "cast input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "cast output") ||
        output->dtype != H3_GPU_BF16) return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_cast_f32_to_bf16<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (ushort *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_cast_f32_to_bf16");
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, input, elements, "cast input") ||
        input->dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, output, elements, "cast output") ||
        output->dtype != H3_GPU_F32) return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_cast_bf16_to_f32<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (float *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_cast_bf16_to_f32");
}

int h3_gpu_rms_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, "RMSNorm input") ||
        !h3_gpu_require_elements(gpu, weight, width, "RMSNorm weight") ||
        !h3_gpu_require_elements(gpu, output, count, "RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_vae) {
        h3_sg_vae_norm<<<dim3(rows),128,0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(weight, 0), (float *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_vae_norm");
    }
    h3_rms_norm_f32<<<dim3(rows),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(weight, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_rms_norm_f32");
}

int h3_gpu_adaln_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, "AdaLN input") ||
        !h3_gpu_require_elements(gpu, norm_weight, width, "AdaLN norm") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "AdaLN row map") ||
        !h3_gpu_require_elements(gpu, output, count, "AdaLN output") ||
        !modulation || shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_adaln_f32<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(norm_weight, 0), (const float *)tensor_pointer(modulation, 0), (const uint *)tensor_pointer(row_map, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_adaln_f32");
}

int h3_gpu_gate_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, "gate residual") ||
        !h3_gpu_require_elements(gpu, branch, count, "gate branch") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "gate row map") ||
        !h3_gpu_require_elements(gpu, output, count, "gate output") ||
        !modulation || gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_gate_f32<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const float *)tensor_pointer(residual, 0), (const float *)tensor_pointer(branch, 0), (const float *)tensor_pointer(modulation, 0), (const uint *)tensor_pointer(row_map, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_gate_f32");
}

int h3_gpu_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, "QKV input") ||
        !h3_gpu_require_elements(gpu, q_norm, head_dim, "Q norm") ||
        !h3_gpu_require_elements(gpu, k_norm, head_dim, "K norm") ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, "RoPE cosine") ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, "RoPE sine") ||
        !h3_gpu_require_elements(gpu, query, count, "query") ||
        !h3_gpu_require_elements(gpu, key, count, "key") ||
        !h3_gpu_require_elements(gpu, value, count, "value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_qkv_rope_f32<<<dim3((head_dim+7)/8,( heads+3)/4,( sequence+3)/4),dim3(8,4,4),0,gpu->compute>>>((const float *)tensor_pointer(qkv, 0), (const float *)tensor_pointer(q_norm, 0), (const float *)tensor_pointer(k_norm, 0), (const float *)tensor_pointer(rope_cos, 0), (const float *)tensor_pointer(rope_sin, 0), (float *)tensor_pointer(query, 0), (float *)tensor_pointer(key, 0), (float *)tensor_pointer(value, 0), args);
    return launch_status(gpu, "h3_qkv_rope_f32");
}

int h3_gpu_swiglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, fused, (size_t)rows * width * 2, "SwiGLU input") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * width, "SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_vae) {
        h3_sg_vae_swiglu<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const float *)tensor_pointer(fused, 0), (float *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_vae_swiglu");
    }
    h3_swiglu_f32<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const float *)tensor_pointer(fused, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_swiglu_f32");
}

int h3_gpu_scale_add_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, "scale-add residual") ||
        residual->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, branch, count, "scale-add branch") ||
        branch->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, scale, width, "scale-add scale") ||
        scale->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "scale-add output") ||
        output->dtype != H3_GPU_F32) return 0;
    scale_add_args args = {rows, width};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_scale_add_f32<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const float *)tensor_pointer(residual, 0), (const float *)tensor_pointer(branch, 0), (const float *)tensor_pointer(scale, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_scale_add_f32");
}

int h3_gpu_layer_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, "LayerNorm input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, width, "LayerNorm weight") ||
        weight->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, width, "LayerNorm bias") ||
        bias->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "LayerNorm output") ||
        output->dtype != H3_GPU_F32) return 0;
    norm_args args = {rows, width, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_vae || gpu->sglang_audio_encoder) {
        h3_sg_vae_layer_norm<<<dim3(rows),128,0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(weight, 0), (const float *)tensor_pointer(bias, 0), (float *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_vae_layer_norm");
    }
    h3_layer_norm_f32<<<dim3(rows),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(weight, 0), (const float *)tensor_pointer(bias, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_layer_norm_f32");
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    h3_gpu *gpu = opaque;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, "video QKV") ||
        qkv->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, "video RoPE cosine") ||
        rope_cos->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, "video RoPE sine") ||
        rope_sin->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, "video query") ||
        query->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, key, count, "video key") ||
        key->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, value, count, "video value") ||
        value->dtype != H3_GPU_F32 || rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_video_qkv_rope_f32<<<dim3((head_dim+7)/8,( heads+3)/4,( sequence+3)/4),dim3(8,4,4),0,gpu->compute>>>((const float *)tensor_pointer(qkv, 0), (const float *)tensor_pointer(rope_cos, 0), (const float *)tensor_pointer(rope_sin, 0), (float *)tensor_pointer(query, 0), (float *)tensor_pointer(key, 0), (float *)tensor_pointer(value, 0), args);
    return launch_status(gpu, "h3_video_qkv_rope_f32");
}

int h3_gpu_weight_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)outer * inner;
    if (!outer || !inner ||
        !h3_gpu_require_elements(gpu, vector, count, "weight-norm vector") ||
        vector->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, magnitude, outer,
                                 "weight-norm magnitude") ||
        magnitude->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "weight-norm output") ||
        output->dtype != H3_GPU_F32) return 0;
    weight_norm_args args = {outer, inner};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_reference) {
        h3_sg_weight_norm<<<outer,256,0,gpu->compute>>>((const float*)tensor_pointer(vector),
            (const float*)tensor_pointer(magnitude),(float*)tensor_pointer(output),inner);
        return launch_status(gpu,"reference weight norm");
    }
    h3_weight_norm_f32<<<dim3((outer+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(vector, 0), (const float *)tensor_pointer(magnitude, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_weight_norm_f32");
}

int h3_gpu_add_scaled_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, left, elements, "scaled-add left") ||
        left->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, right, elements, "scaled-add right") ||
        right->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "scaled-add output") ||
        output->dtype != H3_GPU_F32) return 0;
    add_scaled_args args = {elements, left_scale, right_scale};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_add_scaled_f32<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(left, 0), (const float *)tensor_pointer(right, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_add_scaled_f32");
}

int h3_gpu_alias_free_snake_f32(
                          h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels ||
        !h3_gpu_require_elements(gpu, input, count, "Snake input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "Snake output") ||
        output->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha_log, channels, "Snake alpha") ||
        alpha_log->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, beta_log, channels, "Snake beta") ||
        beta_log->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, upsample_filter, 12,
                                 "Snake upsample filter") ||
        upsample_filter->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, downsample_filter, 12,
                                 "Snake downsample filter") ||
        downsample_filter->dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_audio) {
        // The pinned JIT profiles two static shapes, then one dynamic shape.
        // Calls after that use a dynamic TensorExpr kernel (including FMA).
        if(gpu->sglang_audio_activation_shapes.size()>=8 &&
           !gpu->sglang_audio_activation_shapes.count({batch,length,channels}))
            return h3_gpu_set_error(gpu,"reference audio activation shape limit exceeded");
        bool fused=gpu->sglang_audio_activation_shapes.size()>=3 ||
            gpu->sglang_audio_activation_shapes.count({batch,length,channels});
        gpu->sglang_audio_activation_shapes.emplace(batch,length,channels);
    constexpr uint32_t time_chunk = 65535u * 4u;
    for (uint64_t time_base = 0; time_base < length; time_base += time_chunk) {
        uint32_t chunk_length = length - (uint32_t)time_base;
        if (chunk_length > time_chunk) chunk_length = time_chunk;
        h3_sg_alias_free_snake_f32<<<dim3((channels+7)/8,(chunk_length+3)/4,(batch+3)/4),dim3(8,4,4),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(alpha_log, 0), (const float *)tensor_pointer(beta_log, 0), (const float *)tensor_pointer(upsample_filter, 0), (const float *)tensor_pointer(downsample_filter, 0), (float *)tensor_pointer(output, 0), args, (uint32_t)time_base, fused);
        if (!launch_status(gpu, "h3_sg_alias_free_snake_f32")) return 0;
    }
    return 1;
    }
    constexpr uint32_t time_chunk = 65535u * 4u;
    for (uint64_t time_base = 0; time_base < length; time_base += time_chunk) {
        uint32_t chunk_length = length - (uint32_t)time_base;
        if (chunk_length > time_chunk) chunk_length = time_chunk;
        h3_alias_free_snake_f32<<<dim3((channels+7)/8,(chunk_length+3)/4,(batch+3)/4),dim3(8,4,4),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(alpha_log, 0), (const float *)tensor_pointer(beta_log, 0), (const float *)tensor_pointer(upsample_filter, 0), (const float *)tensor_pointer(downsample_filter, 0), (float *)tensor_pointer(output, 0), args, (uint32_t)time_base);
        if (!launch_status(gpu, "h3_alias_free_snake_f32")) return 0;
    }
    return 1;
}

int h3_gpu_snake1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, input, count, "Snake1d input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha, channels, "Snake1d alpha") ||
        alpha->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "Snake1d output") ||
        output->dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_audio_encoder) {
        h3_sg_snake1d_f32<<<dim3(((uint32_t)count+255)/256),dim3(256),0,gpu->compute>>>(
            (const float*)tensor_pointer(input),(const float*)tensor_pointer(alpha),
            (float*)tensor_pointer(output),args);
        return launch_status(gpu,"reference encoder Snake1d");
    }
    h3_snake1d_f32<<<dim3(((uint32_t)count+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (const float *)tensor_pointer(alpha, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_snake1d_f32");
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *opaque,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    h3_gpu *gpu = opaque;
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!batch || !length || !heads || !head_dim || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, qkv, count * 3, "audio QKV") ||
        qkv->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, q_bias, width, "audio Q bias") ||
        !h3_gpu_require_elements(gpu, k_bias, width, "audio K bias") ||
        !h3_gpu_require_elements(gpu, v_bias, width, "audio V bias") ||
        q_bias->dtype != H3_GPU_F32 ||
        k_bias->dtype != H3_GPU_F32 ||
        v_bias->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, "audio query") ||
        !h3_gpu_require_elements(gpu, key, count, "audio key") ||
        !h3_gpu_require_elements(gpu, value, count, "audio value") ||
        query->dtype != H3_GPU_F32 || key->dtype != H3_GPU_F32 ||
        value->dtype != H3_GPU_F32) return 0;
    audio_qkv_args args = {batch, length, heads, head_dim};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_audio_qkv_split_f32<<<dim3(((uint32_t)count+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(qkv, 0), (const float *)tensor_pointer(q_bias, 0), (const float *)tensor_pointer(k_bias, 0), (const float *)tensor_pointer(v_bias, 0), (float *)tensor_pointer(query, 0), (float *)tensor_pointer(key, 0), (float *)tensor_pointer(value, 0), args);
    return launch_status(gpu, "h3_audio_qkv_split_f32");
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *opaque,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    h3_gpu *gpu = opaque;
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, attended, input_count,
                                 "audio attended values") ||
        attended->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 "audio pooled values") ||
        output->dtype != H3_GPU_F32) return 0;
    audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_audio_encoder) {
        if(heads!=8)return h3_gpu_set_error(gpu,"reference audio pooling requires eight heads");
        h3_sg_audio_pool<<<dim3(((uint32_t)output_count+255)/256),256,0,gpu->compute>>>(
            (const float*)tensor_pointer(attended),(float*)tensor_pointer(output),args);
        return launch_status(gpu,"reference audio head mean and adaptive pooling");
    }
    h3_audio_attention_pool_f32<<<dim3(((uint32_t)output_count+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(attended, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_audio_attention_pool_f32");
}

int h3_gpu_geglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, gate, elements, "GeGLU gate") ||
        !h3_gpu_require_elements(gpu, linear, elements, "GeGLU linear") ||
        !h3_gpu_require_elements(gpu, output, elements, "GeGLU output") ||
        gate->dtype != H3_GPU_F32 ||
        linear->dtype != H3_GPU_F32 ||
        output->dtype != H3_GPU_F32) return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_audio_encoder) {
        h3_sg_audio_geglu<<<dim3((elements+255)/256),256,0,gpu->compute>>>(
            (const float*)tensor_pointer(gate),(const float*)tensor_pointer(linear),
            (float*)tensor_pointer(output),elements);
        return launch_status(gpu,"reference encoder GeGLU");
    }
    h3_geglu_f32<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(gate, 0), (const float *)tensor_pointer(linear, 0), (float *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_geglu_f32");
}

int h3_gpu_clip_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    h3_gpu *gpu = opaque;
    if (!(minimum <= maximum) ||
        !h3_gpu_require_elements(gpu, input, elements, "clip input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "clip output") ||
        output->dtype != H3_GPU_F32) return 0;
    clip_args args = {elements, minimum, maximum};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_clip_f32<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_clip_f32");
}

int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    h3_gpu *gpu = opaque;
    if (!batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) return 0;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 "VAE encoder pad input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 "VAE encoder pad output") ||
        output->dtype != H3_GPU_F32) return 0;
    vae_encoder_pad_args args = {
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after
    };
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_vae_encoder_pad_f32<<<dim3((channels+7)/8,( output_width+3)/4,(
                    (size_t)batch * output_depth * output_height+3)/4),dim3(8,4,4),0,gpu->compute>>>((const float *)tensor_pointer(input, 0), (float *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_vae_encoder_pad_f32");
}

int h3_gpu_silu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, input, elements, "SiLU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, "SiLU output")) return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_silu_bf16<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (ushort *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_silu_bf16");
}

int h3_gpu_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, "RMSNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, "RMSNorm weight") ||
        !h3_gpu_require_bf16(gpu, output, count, "RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_text) {
        h3_sg_text_norm<<<(rows+3)/4,128,0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (const ushort *)tensor_pointer(weight, 0), (ushort *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_text_norm");
    }
    if(gpu->sglang_reference) {
        h3_sg_dit_norm<<<dim3(rows),128,0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (const ushort *)tensor_pointer(weight, 0), (ushort *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_dit_norm");
    }
    h3_rms_norm_bf16<<<dim3(rows),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (const ushort *)tensor_pointer(weight, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_rms_norm_bf16");
}

int h3_gpu_layer_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, "LayerNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, "LayerNorm weight") ||
        !h3_gpu_require_bf16(gpu, bias, width, "LayerNorm bias") ||
        !h3_gpu_require_bf16(gpu, output, count, "LayerNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_reference) {
        h3_sg_vision_layer_norm<<<dim3(rows),128,0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (const ushort *)tensor_pointer(weight, 0), (const ushort *)tensor_pointer(bias, 0), (ushort *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_vision_layer_norm");
    }
    h3_layer_norm_bf16<<<dim3(rows),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (const ushort *)tensor_pointer(weight, 0), (const ushort *)tensor_pointer(bias, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_layer_norm_bf16");
}

int h3_gpu_gelu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, input, elements, "GELU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, "GELU output")) return 0;
    gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_reference) {
        h3_sg_vision_gelu<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (ushort *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_vision_gelu");
    }
    h3_gelu_bf16<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_gelu_bf16");
}

int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    h3_gpu *gpu = opaque;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if(gpu->sglang_reference) {
        if(!rope_cos||!rope_sin||rope_cos->dtype!=H3_GPU_F32||rope_sin->dtype!=H3_GPU_F32||
           !h3_gpu_require_elements(gpu,rope_cos,rope_count,"reference vision cosine")||
           !h3_gpu_require_elements(gpu,rope_sin,rope_count,"reference vision sine")||
           !h3_gpu_require_bf16(gpu,qkv,count*3,"reference vision QKV")||
           !h3_gpu_require_bf16(gpu,query,count,"reference vision Q")||
           !h3_gpu_require_bf16(gpu,key,count,"reference vision K")||
           !h3_gpu_require_bf16(gpu,value,count,"reference vision V")||
           head_dim!=72||rope_half!=36||!launch_ready(gpu))return 0;
        qkv_args args={sequence,heads,head_dim,rope_half,0,0.f};
        h3_sg_vision_rope<<<(count+255)/256,256,0,gpu->compute>>>((const ushort*)tensor_pointer(qkv),
            (const float*)tensor_pointer(rope_cos),(const float*)tensor_pointer(rope_sin),
            (ushort*)tensor_pointer(query),(ushort*)tensor_pointer(key),(ushort*)tensor_pointer(value),args);
        return launch_status(gpu,"reference vision RoPE");
    }
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, "vision QKV") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                              "vision RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                              "vision RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, "vision query") ||
        !h3_gpu_require_bf16(gpu, key, count, "vision key") ||
        !h3_gpu_require_bf16(gpu, value, count, "vision value") ||
        rope_half * 2 != head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, 0.0f};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_vision_qkv_rope_bf16<<<dim3((head_dim+7)/8,( heads+3)/4,( sequence+3)/4),dim3(8,4,4),0,gpu->compute>>>((const ushort *)tensor_pointer(qkv, 0), (const ushort *)tensor_pointer(rope_cos, 0), (const ushort *)tensor_pointer(rope_sin, 0), (ushort *)tensor_pointer(query, 0), (ushort *)tensor_pointer(key, 0), (ushort *)tensor_pointer(value, 0), args);
    return launch_status(gpu, "h3_vision_qkv_rope_bf16");
}

int h3_gpu_adaln_bf16_offset(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (input_offset > SIZE_MAX - count ||
        input_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, "AdaLN input offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, input, input_offset + count,
                             "AdaLN input") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width, "AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, "AdaLN modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "AdaLN row map") ||
        row_map->dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, "AdaLN output") ||
        shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_reference) {
        h3_sg_adaln_bf16<<<dim3(rows),128,0,gpu->compute>>>((const ushort *)tensor_pointer(input, input_offset * sizeof(uint16_t)), (const ushort *)tensor_pointer(norm_weight, 0), (const ushort *)tensor_pointer(modulation, 0), (const uint *)tensor_pointer(row_map, 0), (ushort *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_adaln_bf16");
    }
    h3_adaln_bf16<<<dim3(rows),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(input, input_offset * sizeof(uint16_t)), (const ushort *)tensor_pointer(norm_weight, 0), (const ushort *)tensor_pointer(modulation, 0), (const uint *)tensor_pointer(row_map, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_adaln_bf16");
}

int h3_gpu_gate_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, residual, count, "gate residual") ||
        !h3_gpu_require_bf16(gpu, branch, count, "gate branch") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, "gate modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "gate row map") ||
        row_map->dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, "gate output") ||
        gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_reference) {
        h3_sg_gate_bf16<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const ushort *)tensor_pointer(residual, 0), (const ushort *)tensor_pointer(branch, 0), (const ushort *)tensor_pointer(modulation, 0), (const uint *)tensor_pointer(row_map, 0), (ushort *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_gate_bf16");
    }
    h3_gate_bf16<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const ushort *)tensor_pointer(residual, 0), (const ushort *)tensor_pointer(branch, 0), (const ushort *)tensor_pointer(modulation, 0), (const uint *)tensor_pointer(row_map, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_gate_bf16");
}

int h3_gpu_swiglu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, fused, (size_t)rows * width * 2,
                              "SwiGLU input") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * width,
                              "SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_reference) {
        h3_sg_swiglu_bf16<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const ushort *)tensor_pointer(fused, 0), (ushort *)tensor_pointer(output, 0), args);
        return launch_status(gpu, "h3_sg_swiglu_bf16");
    }
    h3_swiglu_bf16<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const ushort *)tensor_pointer(fused, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_swiglu_bf16");
}

int h3_gpu_embedding_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, weight, (size_t)vocab_size * width,
                              "embedding weight") ||
        !h3_gpu_require_elements(gpu, token_ids, tokens, "token IDs") ||
        token_ids->dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, (size_t)tokens * width,
                              "embedding output")) return 0;
    embedding_args args = {tokens, vocab_size, width};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_embedding_bf16<<<dim3((width+15)/16,( tokens+15)/16),dim3(16,16),0,gpu->compute>>>((const ushort *)tensor_pointer(weight, 0), (const uint *)tensor_pointer(token_ids, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_embedding_bf16");
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *opaque,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    h3_gpu *gpu = opaque;
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query_input, query_count, "text query") ||
        !h3_gpu_require_bf16(gpu, key_input, key_count, "text key") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, "text Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, "text K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, "text RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, "text RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query_output, query_count, "text query output") ||
        !h3_gpu_require_bf16(gpu, key_output, key_count, "text key output")) return 0;
    text_rope_args args = {sequence, query_heads, kv_heads, head_dim, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_text_qk_rope_bf16<<<dim3((head_dim+7)/8,( query_heads+3)/4,( sequence+3)/4),dim3(8,4,4),0,gpu->compute>>>((const ushort *)tensor_pointer(query_input, 0), (const ushort *)tensor_pointer(key_input, 0), (const ushort *)tensor_pointer(q_norm, 0), (const ushort *)tensor_pointer(k_norm, 0), (const ushort *)tensor_pointer(rope_cos, 0), (const ushort *)tensor_pointer(rope_sin, 0), (ushort *)tensor_pointer(query_output, 0), (ushort *)tensor_pointer(key_output, 0), args);
    return launch_status(gpu, "h3_text_qk_rope_bf16");
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)sequence * heads * head_dim;
    if (!h3_gpu_require_bf16(gpu, tensor, count, "head norm tensor") ||
        !h3_gpu_require_bf16(gpu, weight, head_dim, "head norm weight")) return 0;
    head_norm_args args = {sequence, heads, head_dim, epsilon};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_text) {
        h3_sg_text_head_norm<<<(sequence*heads+3)/4,128,0,gpu->compute>>>((ushort *)tensor_pointer(tensor, 0), (const ushort *)tensor_pointer(weight, 0), args);
        return launch_status(gpu, "h3_sg_text_head_norm");
    }
    h3_head_rms_norm_bf16<<<dim3((sequence+15)/16,( heads+15)/16),dim3(16,16),0,gpu->compute>>>((ushort *)tensor_pointer(tensor, 0), (const ushort *)tensor_pointer(weight, 0), args);
    return launch_status(gpu, "h3_head_rms_norm_bf16");
}

int h3_gpu_rope_text_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    h3_gpu *gpu = opaque;
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query, query_count, "RoPE query") ||
        !h3_gpu_require_bf16(gpu, key, key_count, "RoPE key") ||
        !h3_gpu_require_elements(gpu, rope_cos_f32, rope_count, "RoPE cosine") ||
        rope_cos_f32->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin_f32, rope_count, "RoPE sine") ||
        rope_sin_f32->dtype != H3_GPU_F32) return 0;
    text_rope_inplace_args args = {sequence, query_heads, kv_heads, head_dim};
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_text) {
        h3_sg_text_rope<<<dim3((sequence+15)/16,( maximum_heads+15)/16),dim3(16,16),0,gpu->compute>>>((ushort *)tensor_pointer(query, 0), (ushort *)tensor_pointer(key, 0), (const float *)tensor_pointer(rope_cos_f32, 0), (const float *)tensor_pointer(rope_sin_f32, 0), args);
        return launch_status(gpu, "h3_sg_text_rope");
    }
    h3_rope_text_bf16<<<dim3((sequence+15)/16,( maximum_heads+15)/16),dim3(16,16),0,gpu->compute>>>((ushort *)tensor_pointer(query, 0), (ushort *)tensor_pointer(key, 0), (const float *)tensor_pointer(rope_cos_f32, 0), (const float *)tensor_pointer(rope_sin_f32, 0), args);
    return launch_status(gpu, "h3_rope_text_bf16");
}

int h3_gpu_add_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, left, elements, "add left") ||
        !h3_gpu_require_bf16(gpu, right, elements, "add right") ||
        !h3_gpu_require_bf16(gpu, output, elements, "add output")) return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_add_bf16<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(left, 0), (const ushort *)tensor_pointer(right, 0), (ushort *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_add_bf16");
}

int h3_gpu_sub_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, left, elements, "subtract left") ||
        !h3_gpu_require_bf16(gpu, right, elements, "subtract right") ||
        !h3_gpu_require_bf16(gpu, output, elements, "subtract output"))
        return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_sub_bf16<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(left, 0), (const ushort *)tensor_pointer(right, 0), (ushort *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_sub_bf16");
}

int h3_gpu_token_pool_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    h3_gpu *gpu = opaque;
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows ||
        baseline_rows > rows || !width ||
        elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !input || input->dtype != H3_GPU_BF16 ||
        input_offset > input->elements ||
        input_elements > input->elements - input_offset ||
        !original || original->dtype != H3_GPU_BF16 ||
        original_offset > original->elements ||
        input_elements > original->elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements, "token pool output") ||
        !baseline || baseline->dtype != H3_GPU_BF16 ||
        baseline_offset > baseline->elements ||
        baseline_elements > baseline->elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, rows,
                                 "token pool baseline indices") ||
        baseline_indices->dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, pairs, (size_t)rows * 2,
                                 "token pool pairs") ||
        pairs->dtype != H3_GPU_U32) return 0;
    
    h3_token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width
    };
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_token_pool_bf16<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const ushort *)tensor_pointer(input, 0), (const uint2 *)tensor_pointer(pairs, 0), (ushort *)tensor_pointer(output, 0), (ushort *)tensor_pointer(baseline, 0), (const uint *)tensor_pointer(baseline_indices, 0), (ushort *)tensor_pointer(original, 0), args);
    return launch_status(gpu, "h3_token_pool_bf16");
}

int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale) {
    h3_gpu *gpu = opaque;
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows ||
        baseline_rows > reduced_rows || !width ||
        exact_prefix_rows > reduced_rows || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !original || original->dtype != H3_GPU_BF16 ||
        original_offset > original->elements ||
        elements > original->elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             "token expand output") ||
        !h3_gpu_require_bf16(gpu, reduced, reduced_elements,
                             "token expand reduced") ||
        !baseline || baseline->dtype != H3_GPU_BF16 ||
        baseline_offset > baseline->elements ||
        baseline_elements > baseline->elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, reduced_rows,
                                 "token expand baseline indices") ||
        baseline_indices->dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, parents, rows,
                                 "token expand parents") ||
        parents->dtype != H3_GPU_U32) return 0;
    
    h3_token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale
    };
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_token_expand_delta_bf16<<<dim3((width+15)/16,( rows+15)/16),dim3(16,16),0,gpu->compute>>>((const ushort *)tensor_pointer(original, 0), (const ushort *)tensor_pointer(reduced, 0), (const ushort *)tensor_pointer(baseline, 0), (const uint *)tensor_pointer(baseline_indices, 0), (const uint *)tensor_pointer(parents, 0), (ushort *)tensor_pointer(output, 0), args);
    return launch_status(gpu, "h3_token_expand_delta_bf16");
}

int h3_gpu_euler_bf16(h3_gpu *opaque, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    h3_gpu *gpu = opaque;
    if (!sample || sample->dtype != H3_GPU_F32 ||
        sample_offset > sample->elements ||
        elements > sample->elements - sample_offset ||
        sample_offset > UINT32_MAX || elements > UINT32_MAX - sample_offset ||
        !h3_gpu_require_bf16(gpu, last, elements, "Euler last velocity") ||
        !h3_gpu_require_bf16(gpu, previous, elements,
                             "Euler previous velocity")) return 0;
    h3_euler_args args = {(uint32_t)sample_offset, elements, delta, ratio};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_euler_bf16<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((float *)tensor_pointer(sample, 0), (const ushort *)tensor_pointer(last, 0), (const ushort *)tensor_pointer(previous, 0), args);
    return launch_status(gpu, "h3_euler_bf16");
}

int h3_gpu_bridge_euler_bf16(h3_gpu *opaque, h3_gpu_tensor *sample,
    size_t sample_offset, const h3_gpu_tensor *last,
    const h3_gpu_tensor *previous, const h3_gpu_tensor *row_classes,
    const h3_gpu_tensor *strengths, uint32_t rows, uint32_t row_width,
    float delta, float ratio) {
    h3_gpu *gpu = opaque;
    if (!bridge_ranges(sample, sample_offset, row_classes, strengths, rows,
                       row_width) || !isfinite(delta) || !(delta > 0.0f) || !isfinite(ratio) ||
        !h3_gpu_require_bf16(gpu, last, rows * row_width, "bridge last velocity") ||
        !h3_gpu_require_bf16(gpu, previous, rows * row_width,
                             "bridge previous velocity")) return 0;
    h3_bridge_euler_args args = {(uint32_t)sample_offset, rows * row_width,
        row_width, (uint32_t)strengths->elements, delta, ratio};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_bridge_euler_bf16<<<dim3((args.elements+255)/256),dim3(256),0,gpu->compute>>>((float *)tensor_pointer(sample, 0), (const ushort *)tensor_pointer(last, 0), (const ushort *)tensor_pointer(previous, 0), (const uint *)tensor_pointer(row_classes, 0), (const float *)tensor_pointer(strengths, 0), args);
    return launch_status(gpu, "h3_bridge_euler_bf16");
}

int h3_gpu_bridge_check_exact(h3_gpu *opaque, const h3_gpu_tensor *sample,
    size_t sample_offset, const h3_gpu_tensor *initial,
    const h3_gpu_tensor *row_classes, const h3_gpu_tensor *strengths,
    h3_gpu_tensor *changed, uint32_t rows, uint32_t row_width) {
    h3_gpu *gpu = opaque;
    if (!bridge_ranges(sample, sample_offset, row_classes, strengths, rows,
                       row_width) || !initial ||
        initial->dtype != H3_GPU_F32 ||
        initial->elements < rows * row_width || !changed ||
        changed->dtype != H3_GPU_U32 ||
        changed->elements != 1) return 0;
    h3_bridge_euler_args args = {(uint32_t)sample_offset, rows * row_width,
        row_width, (uint32_t)strengths->elements, 0, 0};
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    h3_bridge_check_exact<<<dim3((args.elements+255)/256),dim3(256),0,gpu->compute>>>((const float *)tensor_pointer(sample, 0), (const float *)tensor_pointer(initial, 0), (const uint *)tensor_pointer(row_classes, 0), (const float *)tensor_pointer(strengths, 0), (unsigned *)tensor_pointer(changed, 0), args);
    return launch_status(gpu, "h3_bridge_check_exact");
}

int h3_gpu_silu_mul_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, gate, elements, "SiLU gate") ||
        !h3_gpu_require_bf16(gpu, up, elements, "SiLU up") ||
        !h3_gpu_require_bf16(gpu, output, elements, "SiLU product")) return 0;
    if (!launch_ready(gpu)) return 0;
    profile_scope timing(gpu,6);
    if(gpu->sglang_text) {
        h3_sg_silu_mul<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(gate, 0), (const ushort *)tensor_pointer(up, 0), (ushort *)tensor_pointer(output, 0), elements);
        return launch_status(gpu, "h3_sg_silu_mul");
    }
    h3_silu_mul_bf16<<<dim3((elements+255)/256),dim3(256),0,gpu->compute>>>((const ushort *)tensor_pointer(gate, 0), (const ushort *)tensor_pointer(up, 0), (ushort *)tensor_pointer(output, 0), elements);
    return launch_status(gpu, "h3_silu_mul_bf16");
}
