#include "src/gpu.h"
#include "src/device.h"
#include "src/execution.h"
#include <pthread.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static char error[1024];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}} while(0)
typedef struct {h3_gpu_tensor *tensor;const char *path;int generation,ok;} upload;
static void *prefetch(void *opaque){upload *u=opaque;u->ok=h3_gpu_tensor_stream_file_bf16(u->tensor,u->path,(uint64_t)u->generation*2048,1024,error,sizeof(error));return NULL;}
int main(void) {
    h3_device_info info;CHECK(h3_device_query(&info,error,sizeof(error)));
    h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    int cuda=!strcmp(info.backend,"cuda");
    CHECK(!strcmp(h3_gpu_backend_name(g),cuda?"CUDA":"Metal"));
    CHECK(h3_gpu_prefers_device_sampler(g,0)==(cuda || h3_gpu_is_m5(g)));
    CHECK(h3_gpu_prefers_device_sampler(g,1)==cuda);
    if(!strcmp(info.backend,"cuda")) {
        const char *configured=getenv("H3_CUDA_WEIGHT_MODE");char *saved=configured?strdup(configured):NULL;
        CHECK(!configured||saved);
        const char *modes[]={"auto","resident","stream","auto"};
        const uint64_t weights[]={UINT64_MAX,UINT64_MAX,0,0};
        const int expected[]={1,-1,1,0};
        for(unsigned i=0;i<4;i++) {
            CHECK(!setenv("H3_CUDA_WEIGHT_MODE",modes[i],1));
            h3_gpu *probe=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(probe);
            CHECK(h3_gpu_plan_weights(probe,weights[i],0)==expected[i]);
            if(expected[i]<0)CHECK(strstr(h3_gpu_error(probe),"resident weights need"));
            /* A later environment edit belongs to the next context. */
            CHECK(!setenv("H3_CUDA_WEIGHT_MODE",i==2?"resident":"stream",1));
            CHECK(h3_gpu_plan_weights(probe,weights[i],0)==expected[i]);
            h3_gpu_free(probe);
        }
        if(saved){CHECK(!setenv("H3_CUDA_WEIGHT_MODE",saved,1));free(saved);}else unsetenv("H3_CUDA_WEIGHT_MODE");
    }

    float values[1024],readback[1024];for(unsigned i=0;i<1024;i++)values[i]=(float)i/16;
    float ones[32];for(unsigned i=0;i<32;i++)ones[i]=1;
    h3_gpu_tensor *scale=h3_gpu_tensor_from_f32(g,ones,32);CHECK(scale);
    for(int kind=H3_GPU_HOST_VISIBLE;kind<=H3_GPU_HOST_PINNED;kind++) {
        h3_gpu_tensor *t=h3_gpu_tensor_alloc(g,1024,H3_GPU_F32,(h3_gpu_storage)kind);CHECK(t);
        CHECK(h3_gpu_tensor_write_f32(t,values,1024));
        void *host=h3_gpu_tensor_contents(t);
        CHECK(host || (kind==H3_GPU_DEVICE_ONLY&&!strcmp(info.backend,"cuda")));
        if(host)CHECK(!memcmp(host,values,sizeof(values)));
        CHECK(!h3_gpu_tensor_write_f32_range(t,1023,values,2));
        CHECK(h3_gpu_tensor_read_f32(t,readback,1024)&&!memcmp(readback,values,sizeof(values)));
        h3_gpu_tensor *copied=h3_gpu_tensor_new_f32(g,1024);CHECK(copied);
        CHECK(h3_gpu_begin(g)&&h3_gpu_scale_add_f32(g,t,t,t,scale,32,32)&&
              h3_gpu_copy_f32(g,copied,0,t,0,1024)&&h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_f32(copied,readback,1024));
        for(unsigned i=0;i<1024;i++)CHECK(readback[i]==2*values[i]);
        if(host){host=h3_gpu_tensor_contents(t);CHECK(host);for(unsigned i=0;i<1024;i++)CHECK(((float*)host)[i]==2*values[i]);}
        h3_gpu_tensor_free(copied);
        h3_gpu_tensor_free(t);
    }
    h3_gpu_tensor_free(scale);
    h3_gpu_tensor *a=h3_gpu_tensor_from_f32(g,values,1024),*b=h3_gpu_tensor_new_f32(g,1024);CHECK(a&&b);
    h3_gpu_event *event=h3_gpu_event_new(g);CHECK(event&&h3_gpu_begin(g));
    CHECK(h3_gpu_copy_f32(g,b,3,a,7,1000)&&h3_gpu_event_record(g,event)&&h3_gpu_event_wait(g,event)&&h3_gpu_submit(g));
    CHECK(h3_gpu_tensor_read_f32_range(b,3,readback,1000)&&!memcmp(readback,values+7,4000));
    h3_gpu_event_free(event);h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);
    const char *tmp=getenv("TMPDIR");if(!tmp)tmp="/tmp";
    char path[4096];CHECK(snprintf(path,sizeof(path),"%s/h3-cuda-stream-XXXXXX",tmp)<(int)sizeof(path));
    int fd=mkstemp(path);CHECK(fd>=0);uint16_t payload[65][1024];
    for(unsigned generation=0;generation<65;generation++)for(unsigned i=0;i<1024;i++)payload[generation][i]=(uint16_t)(0x3f00+generation*128+(i%128));
    CHECK(write(fd,payload,sizeof(payload))==(ssize_t)sizeof(payload)&&!close(fd));
    h3_gpu_tensor *slots[2]={h3_gpu_tensor_new_bf16(g,1024),h3_gpu_tensor_new_bf16(g,1024)},*out=h3_gpu_tensor_new_bf16(g,64*1024);CHECK(slots[0]&&slots[1]&&out);
    if(cuda) {
        CHECK(h3_gpu_tensor_write_bf16(slots[0],payload[0],1024));
        CHECK(h3_gpu_begin(g)&&h3_gpu_copy_bf16(g,out,0,slots[0],0,1024));
        CHECK(h3_gpu_tensor_read_file_bf16(slots[0],path,2048,1024,error,sizeof(error)));
        CHECK(h3_gpu_copy_bf16(g,out,1024,slots[0],0,1024));
        CHECK(h3_gpu_tensor_stream_file_bf16(slots[0],path,4096,1024,error,sizeof(error)));
        CHECK(h3_gpu_copy_bf16(g,out,2048,slots[0],0,1024)&&h3_gpu_submit(g));
        uint16_t transitions[3][1024];CHECK(h3_gpu_tensor_read_bf16(out,&transitions[0][0],3072));
        CHECK(!memcmp(transitions,payload,sizeof(transitions)));
        puts("ok: resident overwrite and first streaming transition preserve in-flight reads");
    }
    uint16_t got[64][1024];
    for(int repeat=0;repeat<1;repeat++) {
    upload first={slots[0],path,0,0};prefetch(&first);CHECK(first.ok&&h3_gpu_begin(g));
    for(int i=0;i<64;i++) {
        upload next={slots[(i+1)%2],path,i+1,0};pthread_t worker;CHECK(!pthread_create(&worker,NULL,prefetch,&next));
        CHECK(h3_gpu_copy_bf16(g,out,(size_t)i*1024,slots[i%2],0,1024));
        if(!strcmp(info.backend,"cuda"))CHECK(h3_gpu_continue(g));
        /* Metal's pre-existing shared-memory upload API requires the caller
         * to finish reading a slot before a CPU prefetch overwrites it. */
        else CHECK(h3_gpu_submit(g)&&h3_gpu_begin(g));
        CHECK(!pthread_join(worker,NULL)&&next.ok);
    }
    CHECK(h3_gpu_submit(g));
    CHECK(h3_gpu_tensor_read_bf16(out,&got[0][0],64*1024)&&!memcmp(got,payload,sizeof(got)));
    }
    for(int i=0;i<16;i++) {
        upload next={slots[1],path,i+1,0};pthread_t worker;
        CHECK(h3_gpu_begin(g)&&h3_gpu_copy_bf16(g,out,0,slots[0],0,1024));
        CHECK(!pthread_create(&worker,NULL,prefetch,&next));
        h3_gpu_cancel(g);
        CHECK(!pthread_join(worker,NULL)&&next.ok);
        CHECK(h3_gpu_begin(g)&&h3_gpu_copy_bf16(g,out,0,slots[1],0,1024)&&h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_bf16(out,&got[0][0],1024)&&!memcmp(got,payload[i+1],2048));
    }

    // Teardown must drain a pending read before either allocation is released.
    CHECK(h3_gpu_begin(g)&&h3_gpu_copy_bf16(g,out,0,slots[1],0,1024));
    h3_gpu_tensor_free(slots[0]);h3_gpu_tensor_free(slots[1]);h3_gpu_tensor_free(out);CHECK(!unlink(path));
    h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats)&&stats.live_bytes==0);h3_gpu_free(g);
    puts("ok: storage modes, range copies, fences, 64 asynchronous slot handoffs, 16 cancellations and in-flight teardown");return 0;
}
