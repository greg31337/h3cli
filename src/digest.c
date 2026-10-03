/* FIPS 180-4 SHA-256. The serialized AV/sampler formats use the same digest on
 * both platforms; Darwin continues calling CommonCrypto through the header. */
#include "src/digest.h"
#include "src/execution.h"
#include "src/weights/quant.h"
#include <stdlib.h>
#include <string.h>
#ifndef __APPLE__
#ifdef H3_USE_OPENSSL
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
static uint32_t rotr(uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
static void block(h3_sha256_ctx*c,const uint8_t*p){
    static const uint32_t k[64]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];for(unsigned i=0;i<16;i++)w[i]=(uint32_t)p[4*i]<<24|(uint32_t)p[4*i+1]<<16|(uint32_t)p[4*i+2]<<8|p[4*i+3];
    for(unsigned i=16;i<64;i++){uint32_t x=w[i-15],y=w[i-2];w[i]=(rotr(x,7)^rotr(x,18)^(x>>3))+w[i-16]+(rotr(y,17)^rotr(y,19)^(y>>10))+w[i-7];}
    uint32_t a=c->state[0],b=c->state[1],v=c->state[2],d=c->state[3],e=c->state[4],f=c->state[5],g=c->state[6],h=c->state[7];
    for(unsigned i=0;i<64;i++){uint32_t t1=h+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&v)^(b&v));h=g;g=f;f=e;e=d+t1;d=v;v=b;b=a;a=t1+t2;}
    c->state[0]+=a;c->state[1]+=b;c->state[2]+=v;c->state[3]+=d;c->state[4]+=e;c->state[5]+=f;c->state[6]+=g;c->state[7]+=h;
}
void h3_sha256_init(h3_sha256_ctx*c){
    *c=(h3_sha256_ctx){.state={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
#ifdef H3_USE_OPENSSL
    if(h3_quant_current().mode)
        c->use_accelerated=SHA256_Init(&c->accelerated)==1;
#endif
}
void h3_sha256_update(h3_sha256_ctx*c,const void*data,h3_sha256_size bytes){
#ifdef H3_USE_OPENSSL
    if(c->use_accelerated){SHA256_Update(&c->accelerated,data,bytes);return;}
#endif
    const uint8_t*p=data;size_t used=(size_t)(c->bytes%64);c->bytes+=bytes;
    while(bytes){size_t n=64-used;if(n>bytes)n=bytes;memcpy(c->buffer+used,p,n);p+=n;bytes-=(uint32_t)n;used+=n;if(used==64){block(c,c->buffer);used=0;}}
}
void h3_sha256_init_fast(h3_sha256_ctx *c) {
    h3_sha256_init(c);
#ifdef H3_USE_OPENSSL
    c->use_accelerated=SHA256_Init(&c->accelerated)==1;
#endif
}
void h3_sha256_final(uint8_t digest[32],h3_sha256_ctx*c){
#ifdef H3_USE_OPENSSL
    if(c->use_accelerated){SHA256_Final(digest,&c->accelerated);memset(c,0,sizeof(*c));return;}
#endif
    uint64_t bits=c->bytes*8;uint8_t pad[128]={0x80};size_t used=(size_t)(c->bytes%64),n=used<56?56-used:120-used;
    h3_sha256_update(c,pad,(uint32_t)n);for(unsigned i=0;i<8;i++)pad[i]=(uint8_t)(bits>>(56-8*i));h3_sha256_update(c,pad,8);
    for (unsigned i = 0; i < 32; i++)
        digest[i] = (uint8_t)(c->state[i / 4] >> (24 - (i % 4) * 8));
    memset(c, 0, sizeof(*c));
}
#endif
