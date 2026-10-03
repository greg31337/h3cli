/* Host-only campaign geometry plus the existing independent ordinal fixture. */
#include "src/host.h"
#include "src/media/refvideo.h"
#include "src/conditioning/tokenizer.h"
#include <assert.h>
#include <stdio.h>
void h3_test_tokenizer_multimodal(h3_tokenizer *tokenizer);
/* This geometry/presentation test never decodes media or creates a GPU. */
int h3_sglang_requested(void) { return 0; }
int main(int argc, char **argv) {
    assert(argc == 2);
    char error[512];
    assert(h3_align_frame_count(72) == 73);
    assert(h3_align_frame_count(144) == 158);
    assert(h3_align_frame_count(226) == 226);
    assert(h3_align_frame_count(240) == 243);
    for (int count = 1; count <= 3; count++) {
        int frames = count == 3 ? 48 : 72, total = 0;
        for (int i = 0; i < count; i++) {
            h3_refvideo_plan p;
            assert(h3_refvideo_validate_duration(frames, &total, error, sizeof(error)));
            assert(h3_refvideo_plan_build(frames, &p));
            assert(p.vae_frames == (frames == 48 ? 39 : 56));
            assert(p.latent_t == (frames == 48 ? 12 : 17));
        }
        assert(total <= 144);
    }
    int total = 0;
    assert(!h3_refvideo_validate_duration(47, &total, error, sizeof(error)));
    int w, h;
    assert(h3_reference_image_canvas(1365, 1821, 640, 480, 0, &w, &h));
    printf("large image match 640: %dx%d\n", w, h);
    assert(w == 480 && h == 640);
    assert(h3_reference_image_canvas(1365, 1821, 1344, 768, 0, &w, &h));
    printf("large image match 1344: %dx%d\n", w, h);
    assert(w == 864 && h == 1184);
    assert(h3_reference_image_canvas(1365, 1821, 1344, 768, 2048, &w, &h));
    assert(w == 2048 && h == 2720);
    h3_tokenizer *t = h3_tokenizer_load(argv[1], error, sizeof(error));
    if (!t) { fprintf(stderr, "%s\n", error); return 1; }
    h3_test_tokenizer_multimodal(t);
    h3_tokenizer_free(t);
    puts("PASS: bounded multi-video geometry, image canvases and modality ordinals");
    return 0;
}
