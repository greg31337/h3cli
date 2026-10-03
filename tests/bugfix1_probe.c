/* Development/test probe for real shader resolution and weight storage. */
#include "src/metal/shader.h"
#include "src/weights/safetensors.h"
#include "src/gpu.h"
#include "src/weights/weights.h"
#include "src/sampling/av_state.h"
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
static void die(const char *s) { fprintf(stderr,"%s\n",s); exit(1); }
static double now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9; }
static uint64_t resident(void) {
    struct mach_task_basic_info info;mach_msg_type_number_t n=MACH_TASK_BASIC_INFO_COUNT;
    return task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&info,&n)==KERN_SUCCESS?info.resident_size:0;
}
static uint64_t footprint(void) {
    task_vm_info_data_t info;mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&n)==KERN_SUCCESS?info.phys_footprint:0;
}
static void storage_check(const char *aligned, const char *unaligned) {
    char error[512],reason[128];size_t mapping;
#define CHECK(x) do { if(!(x)) die(#x); } while(0)
    CHECK(!h3_st_file_range(UINT64_MAX,1,UINT64_MAX));
    CHECK(!h3_st_file_range(INT64_MAX,1,UINT64_MAX));
    CHECK(!h3_st_zero_copy_plan(0,SIZE_MAX,H3_DTYPE_U8,UINT64_MAX,16384,SIZE_MAX,&mapping,reason,sizeof(reason)));
    CHECK(!h3_st_zero_copy_plan(0,4,H3_DTYPE_UNKNOWN,4,16384,SIZE_MAX,&mapping,reason,sizeof(reason)));
    CHECK(!h3_st_zero_copy_plan(0,4,H3_DTYPE_F32,4,16384,100,&mapping,reason,sizeof(reason)));
    CHECK(h3_st_zero_copy_plan(16384,4,H3_DTYPE_F32,16388,16384,SIZE_MAX,&mapping,reason,sizeof(reason)) && mapping==16384);
    h3_gpu *gpu=h3_gpu_create(NULL,error,sizeof(error));CHECK(gpu);
    h3_st_header headers[2];
    CHECK(h3_st_read_header(aligned,&headers[0],error,sizeof(error)));
    CHECK(h3_st_read_header(unaligned,&headers[1],error,sizeof(error)));
    for(int repeat=0;repeat<32;repeat++) {
        h3_gpu_tensor *tensors[4];
        for(int f=0;f<2;f++) {
            const h3_st_tensor *x=h3_st_find(&headers[f],"x"),*y=h3_st_find(&headers[f],"y");
            tensors[f*2]=h3_gpu_tensor_load_f32(gpu,headers[f].path,x->file_offset,4);
            tensors[f*2+1]=h3_gpu_tensor_load_bf16(gpu,headers[f].path,y->file_offset,4);
            CHECK(tensors[f*2]&&tensors[f*2+1]);
            CHECK(h3_gpu_tensor_is_file_mapped(tensors[f*2])==(f==0));
            CHECK(!h3_gpu_tensor_is_file_mapped(tensors[f*2+1]));
        }
        h3_gpu_tensor *out=h3_gpu_tensor_new_f32(gpu,4);CHECK(out);
        float values[4];const float expected[]={1,-2,3.25f,0};
        for(int f=0;f<2;f++) {
            CHECK(h3_gpu_begin(gpu));
            CHECK(h3_gpu_copy_f32(gpu,out,0,tensors[f*2],0,4));
            CHECK(h3_gpu_submit(gpu));
            CHECK(h3_gpu_tensor_read_f32(out,values,4));CHECK(!memcmp(values,expected,sizeof(values)));
            CHECK(h3_gpu_begin(gpu));
            CHECK(h3_gpu_cast_bf16_to_f32(gpu,out,tensors[f*2+1],4));
            CHECK(h3_gpu_submit(gpu));
            CHECK(h3_gpu_tensor_read_f32(out,values,4));CHECK(!memcmp(values,expected,sizeof(values)));
        }
        h3_gpu_tensor_free(out);
        for(int i=3;i>=0;i--) h3_gpu_tensor_free(tensors[i]);
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(gpu,&stats));CHECK(stats.live_bytes==0);
    }
    CHECK(!h3_gpu_tensor_load_f32(gpu,aligned,headers[0].file_size-2,1));
    CHECK(!h3_gpu_tensor_load_f32(gpu,aligned,INT64_MAX,1));
    h3_gpu_tensor *buffer=h3_gpu_tensor_new_bf16(gpu,4);CHECK(buffer);
    CHECK(!h3_gpu_tensor_read_file_bf16(buffer,aligned,headers[0].file_size-2,4,error,sizeof(error)));
    CHECK(!h3_gpu_tensor_stream_file_bf16(buffer,aligned,headers[0].file_size-2,4,error,sizeof(error)));
    h3_gpu_tensor_free(buffer);
    CHECK(truncate(unaligned,(off_t)(headers[1].file_size-1))==0);
    unsigned char bytes[8];
    CHECK(!h3_st_read_data(&headers[1],h3_st_find(&headers[1],"y"),bytes,8,error,sizeof(error)));
    h3_st_free_header(&headers[0]);h3_st_free_header(&headers[1]);h3_gpu_free(gpu);
    puts("PASS bounds, mixed ownership, Metal operations and repeated release");
#undef CHECK
}
int main(int argc,char **argv) {
    char error[4096];
    if(argc==4&&!strcmp(argv[1],"state-check")) {
        h3_av_state *a=h3_av_state_load(argv[2],error,sizeof(error));if(!a) die(error);
        h3_av_state *b=h3_av_state_load(argv[3],error,sizeof(error));if(!b) die(error);
        if(a->info.video_elements!=b->info.video_elements || a->info.audio_elements!=b->info.audio_elements ||
           memcmp(a->video,b->video,(size_t)a->info.video_elements*sizeof(float)) ||
           memcmp(a->audio,b->audio,(size_t)a->info.audio_elements*sizeof(float))) die("AV latent bytes differ");
        puts("PASS exact video and audio latent bytes");h3_av_state_free(a);h3_av_state_free(b);return 0;
    }
    if(argc>=2&&!strcmp(argv[1],"shader")) {
        char *path=h3_shader_resolve(argc>2?argv[2]:NULL,error,sizeof(error));
        if(!path) die(error);
        puts(path);free(path);return 0;
    }
    if(argc==4&&!strcmp(argv[1],"storage-check")) { storage_check(argv[2],argv[3]); return 0; }
    if(argc<3) die("usage: bugfix1_probe shader [PATH] | header FILE | load FILE TENSOR EXPECT_MAPPED OUTPUT");
    h3_st_header header;
    if(!h3_st_read_header(argv[2],&header,error,sizeof(error))) die(error);
    if(!strcmp(argv[1],"header")) {
        printf("%zu\n",header.tensor_count);h3_st_free_header(&header);return 0;
    }
    if(argc!=6) die("invalid load arguments");
    const h3_st_tensor *entry=h3_st_find(&header,argv[3]);if(!entry) die("tensor missing");
    size_t elements=(size_t)h3_st_tensor_elements(entry),bytes=(size_t)(entry->data_end-entry->data_begin);
    h3_gpu *gpu=h3_gpu_create(NULL,error,sizeof(error));if(!gpu) die(error);
    uint64_t before=resident(),foot_before=footprint();double start=now();
    h3_weight_store *store=NULL;
    h3_gpu_tensor *tensor=NULL;
    if(!strcmp(argv[1],"weight")) {
        char *directory=strdup(argv[2]);char *slash=strrchr(directory,'/');
        if(slash) *slash='\0';else strcpy(directory,".");
        store=h3_weight_store_open(directory,error,sizeof(error));free(directory);
        if(!store) die(error);
        tensor=entry->dtype==H3_DTYPE_BF16?
            h3_weight_load_bf16(store,gpu,entry->name,entry->ndim,entry->shape,error,sizeof(error)):
            h3_weight_load_f32(store,gpu,entry->name,entry->ndim,entry->shape,error,sizeof(error));
        if(!tensor) die(error);
    } else tensor=entry->dtype==H3_DTYPE_BF16?
        h3_gpu_tensor_load_bf16(gpu,argv[2],entry->file_offset,elements):
        entry->dtype==H3_DTYPE_F32?h3_gpu_tensor_load_f32(gpu,argv[2],entry->file_offset,elements):NULL;
    if(!tensor) die(h3_gpu_error(gpu));
    double load=now()-start;int mapped=h3_gpu_tensor_is_file_mapped(tensor);
    if(atoi(argv[4])>=0&&mapped!=atoi(argv[4])) die("unexpected mapping eligibility");
    uint64_t loaded=resident(),foot_loaded=footprint();
    void *values=malloc(bytes?bytes:1);if(!values) die("out of memory");
    int ok=entry->dtype==H3_DTYPE_BF16?h3_gpu_tensor_read_bf16(tensor,values,elements):h3_gpu_tensor_read_f32(tensor,values,elements);
    if(!ok) die("cannot read tensor");
    double ready=now()-start;
    FILE *f=fopen(argv[5],"wb");if(!f||fwrite(values,1,bytes,f)!=bytes||fclose(f)) die("cannot write tensor bytes");
    free(values);
    malloc_zone_pressure_relief(NULL,0);
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    uint64_t touched=resident(),foot_touched=footprint();h3_gpu_tensor_free(tensor);malloc_zone_pressure_relief(NULL,0);uint64_t released=resident(),foot_released=footprint();
    printf("{\"bytes\":%zu,\"mapped\":%d,\"load_seconds\":%.9f,\"rss_before\":%llu,\"rss_loaded\":%llu,\"rss_touched\":%llu,\"rss_released\":%llu,\"peak_rss\":%ld,\"ready_seconds\":%.9f,\"footprint_before\":%llu,\"footprint_loaded\":%llu,\"footprint_touched\":%llu,\"footprint_released\":%llu}\n",
           bytes,mapped,load,(unsigned long long)before,(unsigned long long)loaded,(unsigned long long)touched,(unsigned long long)released,usage.ru_maxrss,ready,(unsigned long long)foot_before,(unsigned long long)foot_loaded,(unsigned long long)foot_touched,(unsigned long long)foot_released);
    h3_weight_store_free(store);h3_gpu_free(gpu);h3_st_free_header(&header);return 0;
}
