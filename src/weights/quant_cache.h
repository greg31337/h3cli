#ifndef H3_QUANT_CACHE_H
#define H3_QUANT_CACHE_H
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Independent of the quantization arithmetic recipe. No tensor bytes are
 * hashed by this metadata cache, including on first preparation. */
#define H3_QUANT_CACHE_VERSION 2
#define H3_QUANT_CACHE_HEADER 128
typedef struct {
    char *source, *artifact;
    struct stat stamp;
    uint64_t offset, bytes, scale, global;
    unsigned mode, rows, columns;
    unsigned char key[32];
    int lock;
} h3_quant_cache;

/* The caller owns mkdir of directory. Holds a per-artifact writer lock until
 * close. Source metadata is checked again before accepting/publishing data. */
h3_quant_cache *h3_quant_cache_open(const char *directory,const char *source,
    uint64_t offset,unsigned rows,unsigned columns,int mode,
    uint64_t bytes,uint64_t scale,uint64_t global,char *error,size_t size);
/* 1 hit, 0 absent, -1 invalid/error. An existing malformed entry never falls
 * through to preparation. All payload buffers have exactly cache->bytes. */
int h3_quant_cache_read(h3_quant_cache *cache,void *payload,char *error,size_t size);
int h3_quant_cache_write(h3_quant_cache *cache,const void *payload,char *error,size_t size);
int h3_quant_cache_unchanged(const h3_quant_cache *cache,char *error,size_t size);
void h3_quant_cache_close(h3_quant_cache *cache);
#ifdef __cplusplus
}
#endif
#endif
