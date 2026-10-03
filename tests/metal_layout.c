/* M2E differential contract: unchanged BF16 norm/RoPE, range policy, final
 * output, protected SOL and failure/ownership behavior. Run serially. */
#include "src/metal/metal_fp16.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return (uint16_t)(u>>16);}
static uint64_t hash(const void *p,size_t n){const unsigned char *b=p;uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;i++){h^=b[i];h*=1099511628211ull;}return h;}
static float random_value(uint32_t *s){*s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return (float)((int)(*s%65537)-32768)/16384.f;}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);goto done;}}while(0)
/* Exercise the actual entry points above the old 100K-row cutoff without
 * allocating large tensors or executing quadratic attention. Valid shapes
 * must reach the deliberately undersized BF16 buffer check; overflow shapes
 * must fail before it. No dispatch should be encoded in either case. */
static int shape_guards(void) {
    char error[4096]={0};h3_gpu *g=NULL;h3_gpu_tensor *tiny=NULL;int pass=0;
    const struct {uint32_t seq,heads;int valid,fused;} cases[]={
        {99999,56,1,1},{100000,56,1,1},{100001,56,1,1},
        {107856,56,1,1},{114005,56,1,1},
        {0,56,0,0},{1,0,0,0},{1,257,0,0},{UINT32_MAX,UINT32_MAX,0,0},
        {(UINT32_MAX-255u)/128,1,1,0},{(UINT32_MAX-255u)/128+1,1,0,0},
        {(UINT32_MAX-255u)/(56*128),56,1,0},
        {(UINT32_MAX-255u)/(56*128)+1,56,0,0},
        {UINT32_MAX/(3*56*128),56,1,1},{UINT32_MAX/(3*56*128)+1,56,1,0}
    };
    h3_metal_attention_options o=H3_METAL_ATTENTION_DEFAULT;
    o.precision=1;o.candidate=1;o.layout_fusion=1;o.min_exact=0;
    uint16_t zero=0;const float scale=1.f/sqrtf(128.f);
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    tiny=h3_gpu_tensor_from_bf16(g,&zero,1);CHECK(tiny);
    for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
        uint32_t seq=cases[i].seq,heads=cases[i].heads;
        const char *expected=cases[i].valid?"mixed Q":"Invalid mixed attention";
        CHECK(h3_gpu_begin(g));
        for(int layout=0;layout<2;layout++) {
            CHECK(!h3_gpu_mixed_sdpa(g,tiny,tiny,tiny,tiny,seq,heads,scale,layout,layout,0));
            CHECK(strstr(h3_gpu_error(g),expected));
            CHECK(!h3_gpu_mixed_sol(g,tiny,tiny,tiny,tiny,seq,heads,scale,layout,layout,o,1,1));
            CHECK(strstr(h3_gpu_error(g),expected));
        }
        CHECK(!h3_gpu_mixed_prepared_attention(g,tiny,tiny,tiny,tiny,seq,heads,scale,0,o,1,1,1));
        CHECK(strstr(h3_gpu_error(g),expected));
        CHECK(!h3_gpu_mixed_prepare(g,tiny,tiny,tiny,tiny,tiny,tiny,tiny,tiny,seq,heads,128,48,1e-6f,scale));
        CHECK(strstr(h3_gpu_error(g),cases[i].fused?"fused preparation input":"Invalid or unconsumed fused preparation"));
        h3_gpu_cancel(g);
    }
    h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));CHECK(stats.direct_dispatches==0);
    puts("{\"shape_guards\":true,\"cases\":15,\"large_dispatches\":0,\"pass\":true}");
    pass=1;
done:
    if(g)h3_gpu_cancel(g);h3_gpu_tensor_free(tiny);h3_gpu_free(g);return pass;
}
static int run(unsigned seq,unsigned heads,int ol,int routed,const char *pattern) {
    char error[4096]={0};h3_gpu *g=NULL;h3_gpu_tensor *src[5]={0},*q[3]={0},*out=NULL;
    h3_sol_layout layout={0};uint16_t *data=NULL,*prepared[3]={0},*ref=NULL;
    uint64_t original[5];size_t sizes[5];int pass=0;unsigned recovered=0,invalid=0;
    h3_metal_attention_options o=H3_METAL_ATTENTION_DEFAULT;o.precision=1;o.candidate=1;o.layout_fusion=1;o.min_exact=routed==2?1.f:.75f;
    h3_backend_scope scope={H3_BACKEND_METAL,routed?H3_ATTN_SOL:H3_ATTN_DENSE,o};h3_backend_scope saved=h3_backend_exchange(scope);
    size_t n=(size_t)seq*heads*128,guard=128;const float scale=1.f/sqrtf(128.f);
    sizes[0]=n*3;sizes[1]=sizes[2]=128;sizes[3]=sizes[4]=(size_t)seq*48;
    h3_metal_fp16_range expected[56],actual[56];
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);CHECK(h3_gpu_dit_attention_configure(g,0));
    data=malloc((n*3+guard)*2);ref=malloc((n+guard)*2);CHECK(data&&ref);
    uint32_t rng=0x3153671;
    for(unsigned j=0;j<5;j++) {
        for(size_t i=0;i<sizes[j];i++)data[i]=bf(j==0?(!strcmp(pattern,"zero")?0:random_value(&rng)):j<3?1.f:j==3?cosf((float)(i%48)*.02f):sinf((float)(i%48)*.02f));
        if(!strcmp(pattern,"recovery")) {
            if(j==0)for(unsigned d=0;d<128;d++)data[d]=bf(0x1p-60f);
            if(j==1)for(size_t i=0;i<sizes[j];i++)data[i]=bf(0x1p20f);
        }
        if(j==0&&!strcmp(pattern,"nonfinite"))data[256]=0x7fc0;
        if(j==0&&!strcmp(pattern,"subnormal"))data[256]=1;
        original[j]=hash(data,sizes[j]*2);src[j]=h3_gpu_tensor_from_bf16(g,data,sizes[j]);CHECK(src[j]);
    }
    for(size_t i=0;i<n+guard;i++)data[i]=0x3f00;
    for(unsigned j=0;j<3;j++){q[j]=h3_gpu_tensor_from_bf16(g,data,n+guard);prepared[j]=malloc(n*2);CHECK(q[j]&&prepared[j]);}
    out=h3_gpu_tensor_from_bf16(g,data,n+guard);CHECK(out);
    if(routed) {
        layout.sequence=seq;layout.query_blocks=(seq+31)/32;layout.key_blocks=(seq+63)/64;
        layout.query=calloc(layout.query_blocks,sizeof(*layout.query));layout.key=calloc(layout.key_blocks,sizeof(*layout.key));CHECK(layout.query&&layout.key);
        for(unsigned r=0;r<seq;r++) {
            unsigned protected_row=r<7||seq-r<=7||(r>=seq/3&&r<seq/3+9);layout.protected_rows+=protected_row;
            h3_sol_block *meta[]={&layout.query[r/32],&layout.key[r/64]};
            for(unsigned j=0;j<2;j++){if(!meta[j]->rows)meta[j]->first_frame=(int)(r/16);meta[j]->last_frame=(int)(r/16);meta[j]->protect|=protected_row;meta[j]->rows++;}
        }
        CHECK(h3_gpu_native_attention_layout(g,&layout));
    }
    CHECK(h3_gpu_begin(g));
    CHECK(h3_gpu_grouped_qkv_rope_bf16(g,q[0],q[1],q[2],src[0],src[1],src[2],src[3],src[4],seq,heads,128,48,1e-6f));
    CHECK(routed?h3_gpu_mixed_sol(g,out,q[0],q[1],q[2],seq,heads,scale,1,ol,o,1,1):h3_gpu_mixed_sdpa(g,out,q[0],q[1],q[2],seq,heads,scale,1,ol,0));
    CHECK(h3_gpu_submit(g));CHECK(h3_gpu_mixed_range_report(g,expected,heads));
    for(unsigned j=0;j<3;j++)CHECK(h3_gpu_tensor_read_bf16(q[j],prepared[j],n));
    CHECK(h3_gpu_tensor_read_bf16(out,ref,n+guard));
    for(unsigned h=0;h<heads;h++){recovered+=(expected[h].flags&2)!=0;invalid+=(expected[h].flags&1)!=0;}
    CHECK(!strcmp(pattern,"recovery")?recovered>0:1);
    CHECK((!strcmp(pattern,"nonfinite")||!strcmp(pattern,"subnormal"))?invalid>0:invalid==0);
    if(invalid)for(size_t i=0;i<n;i++)CHECK(ref[i]==0x3f00);
    /* Prepare may span an existing submission boundary (e.g. QKV capture). */
    CHECK(h3_gpu_begin(g));
    CHECK(h3_gpu_mixed_prepare(g,q[0],q[1],q[2],src[0],src[1],src[2],src[3],src[4],seq,heads,128,48,1e-6f,scale));
    CHECK(h3_gpu_submit(g));
    for(unsigned j=0;j<3;j++){CHECK(h3_gpu_tensor_read_bf16(q[j],data,n+guard));CHECK(!memcmp(data,prepared[j],n*2));for(size_t i=n;i<n+guard;i++)CHECK(data[i]==0x3f00);}
    CHECK(h3_gpu_begin(g));
    CHECK(h3_gpu_mixed_prepared_attention(g,out,q[0],q[1],q[2],seq,heads,scale,ol,o,routed!=0,1,1));
    CHECK(h3_gpu_submit(g));CHECK(h3_gpu_mixed_range_report(g,actual,heads));
    CHECK(!memcmp(expected,actual,heads*sizeof(*actual)));
    CHECK(h3_gpu_tensor_read_bf16(out,data,n+guard));CHECK(!memcmp(data,ref,(n+guard)*2));
    for(unsigned j=0;j<3;j++){CHECK(h3_gpu_tensor_read_bf16(q[j],data,n+guard));if(routed)CHECK(!memcmp(data,prepared[j],n*2));for(size_t i=n;i<n+guard;i++)CHECK(data[i]==0x3f00);}
    for(unsigned j=0;j<5;j++)CHECK(original[j]==hash(h3_gpu_tensor_contents(src[j]),sizes[j]*2));
    h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));CHECK(stats.layout_prepares==1&&stats.layout_inplace_packs==(uint64_t)!routed);
    CHECK(stats.layout_scan_replaced_bytes==n*6&&stats.layout_partial_bytes==(uint64_t)seq*heads*28);
    /* Duplicate consumption, mismatched buffers, overwriting pending metadata,
     * and cancellation must never admit stale ranges or packed BF16 reads. */
    CHECK(h3_gpu_begin(g));CHECK(!h3_gpu_mixed_prepared_attention(g,out,q[0],q[1],q[2],seq,heads,scale,ol,o,routed!=0,1,1));h3_gpu_cancel(g);
    CHECK(h3_gpu_begin(g));CHECK(h3_gpu_mixed_prepare(g,q[0],q[1],q[2],src[0],src[1],src[2],src[3],src[4],seq,heads,128,48,1e-6f,scale));
    CHECK(!h3_gpu_mixed_prepared_attention(g,out,q[1],q[0],q[2],seq,heads,scale,ol,o,routed!=0,1,1));
    CHECK(!h3_gpu_mixed_sdpa(g,out,q[0],q[1],q[2],seq,heads,scale,1,ol,0));
    CHECK(!h3_gpu_mixed_prepare(g,q[0],q[1],q[2],src[0],src[1],src[2],src[3],src[4],seq,heads,128,48,1e-6f,scale));h3_gpu_cancel(g);
    CHECK(h3_gpu_begin(g));
    CHECK(!h3_gpu_mixed_prepare(g,q[0],q[1],q[2],src[0],src[1],src[2],src[3],src[4],100000,112,128,48,1e-6f,scale));
    CHECK(!h3_gpu_mixed_prepare(g,q[0],q[0],q[2],src[0],src[1],src[2],src[3],src[4],seq,heads,128,48,1e-6f,scale));
    CHECK(h3_gpu_mixed_prepare(g,q[0],q[1],q[2],src[0],src[1],src[2],src[3],src[4],seq,heads,128,48,1e-6f,scale));
    CHECK(h3_gpu_mixed_prepared_attention(g,out,q[0],q[1],q[2],seq,heads,scale,ol,o,routed!=0,1,1));CHECK(h3_gpu_submit(g));
    CHECK(h3_gpu_tensor_read_bf16(out,data,n+guard));CHECK(!memcmp(data,ref,(n+guard)*2));
    printf("{\"sequence\":%u,\"heads\":%u,\"output_layout\":%d,\"sol\":%d,\"pattern\":\"%s\",\"recovered_heads\":%u,\"invalid_heads\":%u,\"byte_identical\":true,\"pass\":true}\n",seq,heads,ol,routed,pattern,recovered,invalid);fflush(stdout);pass=1;
done:
    if(g)h3_gpu_cancel(g);for(unsigned j=0;j<5;j++)h3_gpu_tensor_free(src[j]);for(unsigned j=0;j<3;j++){h3_gpu_tensor_free(q[j]);free(prepared[j]);}
    h3_gpu_tensor_free(out);h3_gpu_free(g);h3_sol_layout_free(&layout);free(data);free(ref);h3_backend_exchange(saved);return pass;
}
int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--shape-only"))return !shape_guards();
    if(argc!=1)return 2;
    return !(shape_guards()&&run(17,4,0,0,"random")&&run(257,8,1,0,"random")&&run(513,56,0,0,"random")&&
        run(17,4,0,0,"zero")&&run(257,4,0,0,"recovery")&&run(17,4,0,0,"nonfinite")&&run(17,4,0,0,"subnormal")&&
        run(1025,4,0,1,"random")&&run(513,8,1,1,"random")&&run(257,4,0,2,"random")&&run(257,4,0,1,"recovery")&&run(17,4,0,1,"nonfinite"));
}
