#include <unistd.h>
#include "src/conditioning/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
void h3_test_tokenizer_multimodal(h3_tokenizer *tokenizer);
#define CHECK(c) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

/* Independent golden definitions: this test must fail if production mappings
 * change or disappear, even on a machine without released model files. */
static const char *markers[] = {"<d>", "</d>", "<|cutoff|>",
    "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>", "<|caption_end|>"};

/* Generate the same byte vocabulary on both backends without Foundation. */
static h3_tokenizer *load(const char *path, int marker, int vocab, int added,
                          int wrong_id, int conflict, int later, int succeeds) {
    FILE *f=fopen(path,"w");CHECK(f);
    fputs("{\"normalizer\":{\"type\":\"NFC\"},\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"vocab\":{",f);
    unsigned extra=0;
    for(unsigned byte=0;byte<256;byte++) {
        int visible=(byte>='!'&&byte<='~')||(byte>=0xa1&&byte<=0xac)||byte>=0xae;
        fprintf(f,"%s\"\\u%04x\":%u",byte?",":"",visible?byte:256+extra++,byte);
    }
    const char *content=marker<0?"":conflict?"<conflicting-token>":markers[marker];
    unsigned value=wrong_id?151680u:151669u+(unsigned)(marker<0?0:marker);
    if(vocab)fprintf(f,",\"%s\":%u",content,value);
    fputs("},\"merges\":[]},\"added_tokens\":[{\"content\":\"<|im_start|>\",\"id\":151644}",f);
    if(added)fprintf(f,",{\"content\":\"%s\",\"id\":%u}",content,value);
    if(later)fputs(",{\"content\":\"<later>\",\"id\":151700},{\"content\":\"<d>long\",\"id\":151701}",f);
    fputs("]}",f);CHECK(!fclose(f));
    char error[512];h3_tokenizer *t=h3_tokenizer_load(path,error,sizeof(error));
    CHECK((t!=NULL)==succeeds);
    if(!succeeds)CHECK(strstr(error,"collision"));
    return t;
}

static void check_ids(h3_tokenizer *t, const char *text,
                      const uint32_t *expected, size_t n) {
    uint32_t *ids = NULL; size_t count = 0; char error[512];
    CHECK(h3_tokenizer_encode(t, text, 0, &ids, &count, error, sizeof(error)));
    CHECK(count == n);
    CHECK(!n || !memcmp(ids, expected, n * sizeof(*ids)));
    /* Decode the independent expected IDs, not merely the encoder result. */
    char *decoded = h3_tokenizer_decode(t, expected, n, error, sizeof(error));
    CHECK(decoded && !strcmp(decoded, text));
    free(decoded); h3_tokenizer_ids_free(ids);
}

int main(void) {
    char path[]="/tmp/h3-tokenizer-XXXXXX";int fd=mkstemp(path);CHECK(fd>=0);close(fd);
        h3_tokenizer *t = load(path,-1,0,0,0,0,0,1);
        for (uint32_t i = 0; i < 7; i++) {
            uint32_t expected = 151669 + i;
            check_ids(t, markers[i], &expected, 1);
        }
        const uint32_t mixed[] = {'p','r','e','f','i','x',151669,'H','e','l','l','o',
                                 151670,'s','u','f','f','i','x'};
        check_ids(t, "prefix<d>Hello</d>suffix", mixed, sizeof(mixed)/sizeof(*mixed));
        const uint32_t adjacent[] = {151669,151670,151671,151672,151673,151674,151675,151669};
        check_ids(t, "<d></d><|cutoff|><|lyrics_start|><|lyrics_end|><|caption_start|><|caption_end|><d>",
                  adjacent, sizeof(adjacent)/sizeof(*adjacent));
        const uint32_t ordinary[] = {'<','c','u','t','o','f','f','>'};
        check_ids(t, "<cutoff>", ordinary, sizeof(ordinary)/sizeof(*ordinary));
        const uint32_t qwen = 151644;
        check_ids(t, "<|im_start|>", &qwen, 1);
        h3_test_tokenizer_multimodal(t);
        char error[512]; uint32_t unknown = 151676;
        CHECK(h3_tokenizer_decode(t, &unknown, 1, error, sizeof(error)) == NULL);
        unknown = 1000;
        CHECK(h3_tokenizer_decode(t, &unknown, 1, error, sizeof(error)) == NULL);
        h3_tokenizer_free(t);

        for (uint32_t i = 0; i < 7; i++) {
            /* Identical vocab and added-token mappings are idempotent. */
            t=load(path,(int)i,1,1,0,0,0,1);
            uint32_t expected=151669+i;
            check_ids(t,markers[i],&expected,1);h3_tokenizer_free(t);
            for(int in_vocab=0;in_vocab<2;in_vocab++)for(int wrong=0;wrong<2;wrong++)
                CHECK(!load(path,(int)i,in_vocab,!in_vocab,wrong,!wrong,0,0));
        }
        /* Longest existing special still wins at the same location. */
        t=load(path,-1,0,0,0,0,1,1);
        const uint32_t later[] = {151700,151675,151701};
        check_ids(t, "<later><|caption_end|><d>long", later, 3);
        h3_tokenizer_free(t);
        CHECK(!unlink(path));
        printf("ok: %d mandatory H3 tokenizer conformance/collision checks (no weights)\n", checks);
        return 0;
}
