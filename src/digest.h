#ifndef H3_DIGEST_H
#define H3_DIGEST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __APPLE__
#define COMMON_DIGEST_FOR_OPENSSL
#include <CommonCrypto/CommonDigest.h>
typedef CC_SHA256_CTX h3_sha256_ctx;
typedef CC_LONG h3_sha256_size;
#define h3_sha256_init CC_SHA256_Init
#define h3_sha256_init_fast CC_SHA256_Init
#define h3_sha256_update CC_SHA256_Update
#define h3_sha256_final CC_SHA256_Final
#else
#ifdef H3_USE_OPENSSL
#include <openssl/sha.h>
#endif
typedef struct {
    uint32_t state[8];uint64_t bytes;uint8_t buffer[64];
#ifdef H3_USE_OPENSSL
    /* Stack storage preserves cleanup on callers' early error returns. */
    SHA256_CTX accelerated;
    int use_accelerated;
#endif
} h3_sha256_ctx;
typedef uint32_t h3_sha256_size;
void h3_sha256_init(h3_sha256_ctx *ctx);
/* Content-addressing may use the optional CPU accelerator independently of
 * inference policy. The resulting SHA-256 bytes are identical. */
void h3_sha256_init_fast(h3_sha256_ctx *ctx);
void h3_sha256_update(h3_sha256_ctx *ctx,const void *data,h3_sha256_size bytes);
void h3_sha256_final(uint8_t digest[32],h3_sha256_ctx *ctx);
#endif
#endif
