#include "src/host.h"
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

int h3_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output) {
    if (output) *output = NULL;
    if (!input || !output || frames < 1 || input_width < 1 ||
        input_height < 1 || output_width < 1 || output_height < 1)
        return 0;
    size_t input_area = (size_t)input_width * (size_t)input_height;
    size_t output_area = (size_t)output_width * (size_t)output_height;
    if (input_area > SIZE_MAX / 3 || output_area > SIZE_MAX / 3 ||
        (size_t)frames > SIZE_MAX / (output_area * 3)) return 0;
    size_t output_bytes = (size_t)frames * output_area * 3;
    uint8_t *pixels = malloc(output_bytes);
    if (!pixels) return 0;
    if (input_width == output_width && input_height == output_height) {
        if ((size_t)frames > SIZE_MAX / (input_area * 3)) {
            free(pixels);
            return 0;
        }
        memcpy(pixels, input, (size_t)frames * input_area * 3);
        *output = pixels;
        return 1;
    }
    if (input_area > SIZE_MAX / 4 || output_area > SIZE_MAX / 4) {
        free(pixels);
        return 0;
    }
    uint8_t *source_argb = malloc(input_area * 4);
    uint8_t *output_argb = malloc(output_area * 4);
    if (!source_argb || !output_argb) {
        free(source_argb); free(output_argb); free(pixels);
        return 0;
    }
    size_t input_frame_bytes = input_area * 3;
    size_t output_frame_bytes = output_area * 3;
    vImage_Buffer source_buffer = {
        source_argb, (vImagePixelCount)input_height,
        (vImagePixelCount)input_width, (size_t)input_width * 4
    };
    vImage_Buffer output_buffer = {
        output_argb, (vImagePixelCount)output_height,
        (vImagePixelCount)output_width, (size_t)output_width * 4
    };
    for (int frame = 0; frame < frames; frame++) {
        const uint8_t *source_frame = input + (size_t)frame * input_frame_bytes;
        uint8_t *output_frame = pixels + (size_t)frame * output_frame_bytes;
        for (size_t pixel = 0; pixel < input_area; pixel++) {
            source_argb[4 * pixel] = 255;
            source_argb[4 * pixel + 1] = source_frame[3 * pixel];
            source_argb[4 * pixel + 2] = source_frame[3 * pixel + 1];
            source_argb[4 * pixel + 3] = source_frame[3 * pixel + 2];
        }
        if (vImageScale_ARGB8888(
                &source_buffer, &output_buffer, NULL,
                kvImageHighQualityResampling | kvImageEdgeExtend) !=
            kvImageNoError) {
            free(source_argb); free(output_argb); free(pixels);
            return 0;
        }
        for (size_t pixel = 0; pixel < output_area; pixel++) {
            output_frame[3 * pixel] = output_argb[4 * pixel + 1];
            output_frame[3 * pixel + 1] = output_argb[4 * pixel + 2];
            output_frame[3 * pixel + 2] = output_argb[4 * pixel + 3];
        }
    }
    free(source_argb); free(output_argb);
    *output = pixels;
    return 1;
}

