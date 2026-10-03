#include "src/weights/q8.h"
#include <float.h>
#include <math.h>
#include <string.h>

int h3_q8_layout(uint32_t rows, uint32_t columns, size_t *offset,
                 size_t *bytes) {
    if (!rows || !columns || columns > UINT32_MAX - (H3_Q8_GROUP - 1u) ||
        !offset || !bytes) return 0;
    uint64_t elements = (uint64_t)rows * columns;
    uint64_t groups = ((uint64_t)columns + H3_Q8_GROUP - 1) / H3_Q8_GROUP;
    /* Shader indexing and rounded dispatches use uint32_t. */
    if (elements > UINT32_MAX - 31u) return 0;
    uint64_t start = (elements + 3) & ~UINT64_C(3);
    uint64_t total = start + (uint64_t)rows * groups * sizeof(float);
    if (total > SIZE_MAX) return 0;
    *offset = (size_t)start; *bytes = (size_t)total;
    return 1;
}

static float unpack(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float value; memcpy(&value, &word, sizeof(value)); return value;
}

int h3_q8_pack_rows(const uint16_t *source, int8_t *weights, float *scales,
                    uint32_t rows, uint32_t columns) {
    size_t offset, bytes;
    if (!source || !weights || !scales ||
        !h3_q8_layout(rows, columns, &offset, &bytes)) return 0;
    uint32_t groups = (columns + H3_Q8_GROUP - 1) / H3_Q8_GROUP;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t group = 0; group < groups; group++) {
            uint32_t start = group * H3_Q8_GROUP;
            uint32_t end = columns - start < H3_Q8_GROUP ? columns : start + H3_Q8_GROUP;
            float maximum = 0;
            for (uint32_t k = start; k < end; k++) {
                float value = unpack(source[(size_t)row * columns + k]);
                if (!isfinite(value)) return 0;
                maximum = fmaxf(maximum, fabsf(value));
            }
            /* Keep scales normal on Metal; subnormal BF16 weights quantize to
             * zero instead of relying on device denormal behavior. */
            float scale = maximum == 0 ? 1 : fmaxf(maximum / 127.f, FLT_MIN);
            scales[(size_t)row * groups + group] = scale;
            for (uint32_t k = start; k < end; k++) {
                float value = unpack(source[(size_t)row * columns + k]) / scale;
                float floor_value = floorf(value), fraction = value - floor_value;
                int q = (int)floor_value;
                if (fraction > .5f || (fraction == .5f && q % 2)) q++;
                weights[(size_t)row * columns + k] = (int8_t)(q < -127 ? -127 : q > 127 ? 127 : q);
            }
        }
    }
    return 1;
}
