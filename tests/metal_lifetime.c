/* M5: queued execution, stable scratch and failure cleanup. */
#include "src/gpu.h"
#include "src/metal/metal_fp16.h"
#include "src/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"line %d: %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);exit(1); } } while(0)
static void full(h3_memory_snapshot *s,void *unused) {
    (void)unused;memset(s,0,sizeof(*s));s->available_valid=s->process_valid=1;
    s->physical_total=128ull<<30;s->available=100ull<<30;s->physical_footprint=H3_MAX_PROCESS_MEMORY;
}
int main(void) {
    char error[512]={0};h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    CHECK(h3_gpu_plan_weights(g,1024,1024)==0);
    CHECK(h3_gpu_plan_weights(g,UINT64_MAX,1)==-1);
    /* A plain C caller has no surrounding autorelease pool. Copied weight
     * loads must release their Metal buffer references when the tensor dies. */
    char path[]="/tmp/h3-metal-weight-lifetime-XXXXXX";
    int fd=mkstemp(path);CHECK(fd>=0);CHECK(!ftruncate(fd,8*1024*1024+4));CHECK(!close(fd));
    uint64_t baseline=h3_gpu_device_allocated_bytes();
    for(unsigned i=0;i<8;i++) {
        h3_gpu_tensor *loaded=h3_gpu_tensor_load_f32(g,path,4,2*1024*1024);CHECK(loaded);
        h3_gpu_tensor_free(loaded);
        CHECK(h3_gpu_device_allocated_bytes()<=baseline+2*1024*1024);
    }
    CHECK(!unlink(path));
    enum {SEQ=257,HEADS=4,N=SEQ*HEADS*128};
    uint16_t *data=malloc(N*2),*reference=malloc(N*2);CHECK(data&&reference);
    for(unsigned i=0;i<N;i++)data[i]=0x3e80;
    h3_gpu_tensor *q=h3_gpu_tensor_from_bf16(g,data,N),*v=h3_gpu_tensor_from_bf16(g,data,N),*out=h3_gpu_tensor_new_bf16(g,N);
    CHECK(q&&v&&out);
    h3_metal_attention_options o=H3_METAL_ATTENTION_DEFAULT;
    o.precision=1;o.candidate=1;o.min_exact=.5f;o.local_radius=0;
    unsigned nq=(SEQ+o.q_block-1)/o.q_block,nk=(SEQ+o.kv_block-1)/o.kv_block;
    h3_sol_layout l={.sequence=SEQ,.query_blocks=nq,.key_blocks=nk};
    l.query=calloc(nq,sizeof(*l.query));l.key=calloc(nk,sizeof(*l.key));CHECK(l.query&&l.key);
    for(unsigned r=0;r<SEQ;r++) {
        h3_sol_block *b[]={&l.query[r/o.q_block],&l.key[r/o.kv_block]};
        for(unsigned i=0;i<2;i++){if(!b[i]->rows)b[i]->first_frame=r/16;b[i]->last_frame=r/16;b[i]->rows++;b[i]->protect|=r<7;}
    }
    CHECK(h3_gpu_native_attention_layout(g,&l));h3_gpu_stats warm={0},after={0};
    for(unsigned step=0;step<8;step++) {
        CHECK(h3_gpu_begin(g));
        for(unsigned block=0;block<12;block++) {
            CHECK(h3_gpu_mixed_sol(g,out,q,q,v,SEQ,HEADS,1.f/sqrtf(128),0,block%2,o,block,step));
            CHECK(h3_gpu_continue(g));
        }
        CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(out,data,N));CHECK(h3_gpu_get_stats(g,&after));
        for(unsigned i=0;i<N;i++)CHECK(data[i]==0x3e80);
        if(!step){warm=after;memcpy(reference,data,N*2);}
        else {
            CHECK(!memcmp(reference,data,N*2));CHECK(after.tensor_allocations==warm.tensor_allocations);
            CHECK(after.native_scratch_bytes==warm.native_scratch_bytes);CHECK(after.native_pipeline_count==warm.native_pipeline_count);
            CHECK(after.native_scratch_reuses>warm.native_scratch_reuses);
        }
        CHECK(after.inflight_peak<=2);
    }
    CHECK(after.command_retirements>=8*13);
    CHECK(h3_gpu_begin(g));h3_memory_set_test_snapshot(full,NULL);
    CHECK(h3_gpu_plan_weights(g,1,1)==-1);
    CHECK(!h3_gpu_tensor_new_bf16(g,N));CHECK(h3_memory_error(h3_gpu_error(g)));
    CHECK(!h3_gpu_add_bf16(g,out,q,v,N));CHECK(!h3_gpu_submit(g));
    h3_memory_set_test_snapshot(NULL,NULL);h3_gpu_cancel(g);
    CHECK(h3_gpu_begin(g));CHECK(h3_gpu_add_bf16(g,out,q,v,N));CHECK(h3_gpu_submit(g));
    CHECK(h3_gpu_tensor_read_bf16(out,data,N));for(unsigned i=0;i<N;i++)CHECK(data[i]==0x3f00);
    /* Input/output dependencies across real graph command batches:
     * each identity-times-two projection consumes the preceding result. */
    enum {ROWS=2049,DIM=256,COUNT=ROWS*DIM};
    uint16_t *matrix=calloc(DIM*DIM,2),*values=malloc(COUNT*2);CHECK(matrix&&values);
    for(unsigned i=0;i<DIM;i++)matrix[i*DIM+i]=0x4000;
    for(unsigned i=0;i<COUNT;i++)values[i]=0x3e80;
    h3_gpu_tensor *weight=h3_gpu_tensor_from_bf16(g,matrix,DIM*DIM);
    h3_gpu_tensor *a=h3_gpu_tensor_from_bf16(g,values,COUNT),*b=h3_gpu_tensor_new_bf16(g,COUNT);CHECK(weight&&a&&b);
    CHECK(h3_gpu_get_stats(g,&warm));CHECK(h3_gpu_begin(g));
    for(unsigned block=0;block<12;block++) {
        CHECK(h3_gpu_linear_bf16(g,b,a,weight,NULL,ROWS,DIM,DIM));CHECK(h3_gpu_continue(g));
        h3_gpu_tensor *swap=a;a=b;b=swap;
    }
    CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(a,values,COUNT));
    for(unsigned i=0;i<COUNT;i++)CHECK(values[i]==0x4480);
    CHECK(h3_gpu_get_stats(g,&after));CHECK(after.inflight_peak<=2);
    CHECK(after.mps_linear_dispatches==warm.mps_linear_dispatches+12);
    h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);h3_gpu_tensor_free(weight);free(matrix);free(values);
    printf("PASS: copied-weight release, 96 queued SOL layers, 8 steps, both output layouts, stable scratch/pipelines, 12 dependent MPSGraph projections, bounded commands, cap rejection and recovery; scratch=%llu, allocations=%llu, retired=%llu\n",
        (unsigned long long)after.native_scratch_bytes,(unsigned long long)after.native_scratch_allocations,(unsigned long long)after.command_retirements);
    h3_gpu_tensor_free(q);h3_gpu_tensor_free(v);h3_gpu_tensor_free(out);h3_gpu_free(g);
    free(l.query);free(l.key);free(data);free(reference);return 0;
}
