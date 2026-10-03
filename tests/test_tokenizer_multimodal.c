#include "src/conditioning/multimodal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Real presentation builders with a capture-only decoder; no weights/Metal. */
static uint32_t expected[2048];
static size_t expected_count, expected_span_count;
static h3_text_vision_span expected_spans[4];
static int checks;
#define CHECK(c) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static void append(const h3_tokenizer *t, const char *text) {
    uint32_t *ids = NULL; size_t count = 0; char error[256];
    CHECK(h3_tokenizer_encode(t, text, 0, &ids, &count, error, sizeof(error)));
    CHECK(expected_count + count < sizeof(expected)/sizeof(*expected));
    memcpy(expected + expected_count, ids, count * sizeof(*ids));
    expected_count += count; h3_tokenizer_ids_free(ids);
}

static void vision(const h3_vision_output *v, uint32_t pad) {
    expected[expected_count++] = 151652;
    h3_text_vision_span *s = &expected_spans[expected_span_count++];
    s->start = expected_count; s->tokens = v->tokens; s->embeddings = v->merged;
    memcpy(s->deepstack, v->deepstack, sizeof(s->deepstack));
    for (size_t i = 0; i < v->tokens; i++) expected[expected_count++] = pad;
    expected[expected_count++] = 151653;
}

int h3_text_encode_multimodal_bf16(
        const char *weights, const char *shader, const uint32_t *ids, size_t count,
        const h3_text_vision_span *spans, size_t span_count,
        const uint32_t *positions, const uint8_t *tags,
        h3_text_progress progress, void *opaque, h3_text_embedding *output,
        char *error, size_t error_size) {
    (void)weights; (void)shader; (void)progress; (void)opaque;
    (void)error; (void)error_size;
    CHECK(count == expected_count);
    CHECK(!memcmp(ids, expected, count * sizeof(*ids)));
    CHECK(span_count == expected_span_count);
    CHECK(positions != NULL);
    for (size_t i = 0; i < span_count; i++) {
        CHECK(spans[i].start == expected_spans[i].start);
        CHECK(spans[i].tokens == expected_spans[i].tokens);
        CHECK(spans[i].embeddings == expected_spans[i].embeddings);
        for (size_t k = 0; k < 3; k++)
            CHECK(spans[i].deepstack[k] == expected_spans[i].deepstack[k]);
    }
    for (size_t i = 0; i < count; i++) {
        int visual = 0;
        for (size_t s = 0; s < span_count; s++)
            if (i >= spans[s].start-1 && i <= spans[s].start+spans[s].tokens) visual = 1;
        CHECK(tags[i] == (visual ? 0 : 1));
    }
    output->tokens = count;
    return 1;
}

static void release(h3_text_embedding *e) {
    CHECK(e->diagnostics != NULL);
    CHECK(!memcmp(e->diagnostics->ids, expected, expected_count*sizeof(*expected)));
    free(e->diagnostics->ids); free(e->diagnostics->positions);
    free(e->diagnostics->spans); free(e->diagnostics);
}

void h3_test_tokenizer_multimodal(h3_tokenizer *t) {
    uint16_t rows[4][16] = {{1}, {2}, {3}, {4}};
    h3_vision_output v = {.grid_h=4, .grid_w=4, .tokens=4,
        .merged=rows[0], .deepstack={rows[1], rows[2], rows[3]}};
    const char *prompt = "<d>Hello</d><|cutoff|><|lyrics_start|><|lyrics_end|><|caption_start|><|caption_end|>";
    char error[256]; h3_text_embedding output = {0};
    expected_count = expected_span_count = 0;
    append(t, "<Picture 1>: "); vision(&v, 151655); append(t, prompt);
    CHECK(h3_multimodal_encode_fl2va_bf16(t, "unused", "unused", prompt,
        &v, 1, NULL, NULL, &output, error, sizeof(error)));
    size_t predicted = 0;
    CHECK(h3_multimodal_count_fl2va(t, prompt, &v, 1, &predicted, error, sizeof(error)));
    CHECK(predicted == output.tokens);
    release(&output);
    h3_reference_presentation image = {.kind=H3_PRESENTATION_IMAGE, .vision=&v, .vision_count=1};
    CHECK(h3_multimodal_encode_ref2va_bf16(t, "unused", "unused", prompt,
        &image, 1, NULL, NULL, &output, error, sizeof(error)));
    release(&output);

    h3_vision_output blocks[] = {v, v};
    double times[] = {0.0, 0.5};
    h3_reference_presentation refs[] = {image,
        {.kind=H3_PRESENTATION_VIDEO, .has_audio=1, .vision=blocks,
         .vision_count=2, .timestamps=times},
        {.kind=H3_PRESENTATION_AUDIO, .has_audio=1}};
    expected_count = expected_span_count = 0;
    append(t, "<Picture 1>: "); vision(&v, 151655);
    /* Each modality prefix/timestamp is independently tokenized. Combining
     * these strings hides BPE boundary differences in a byte-only fixture. */
    append(t, "<Audio 1>: "); append(t, "<Video 1>: "); append(t, "<0.0 seconds>"); vision(&v, 151656);
    append(t, "<0.5 seconds>"); vision(&v, 151656);
    append(t, "<Audio 2>: "); append(t, prompt);
    CHECK(h3_multimodal_encode_ref2va_bf16(t, "unused", "unused", prompt,
        refs, 3, NULL, NULL, &output, error, sizeof(error)));
    CHECK(h3_multimodal_count_ref2va(t, prompt, refs, 3, &predicted, error, sizeof(error)));
    CHECK(predicted == output.tokens);
    v.merged = NULL; memset(v.deepstack, 0, sizeof(v.deepstack));
    blocks[0] = blocks[1] = v;
    CHECK(h3_multimodal_count_ref2va(t, prompt, refs, 3, &predicted, error, sizeof(error)));
    CHECK(predicted == output.tokens);
    release(&output);
    CHECK(rows[0][0] == 1 && rows[1][0] == 2 && rows[2][0] == 3 && rows[3][0] == 4);
    printf("ok: %d shared FL2VA/Ref2VA tokenizer and numeric vision insertion checks\n", checks);
}
