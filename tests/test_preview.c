#include "src/media/preview.h"
#include "src/denoise/dit.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t checks;
static char error[512];
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error); exit(1); } } while (0)
static unsigned rng=72;
static float random_float(void) { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return (float)(rng&65535)/32768.0f-1; }
static float fp(uint16_t b) { uint32_t bits=(uint32_t)b<<16; float f; memcpy(&f,&bits,4); return f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec+(double)t.tv_nsec*1e-9; }

static void numerical(void) {
    h3_preview_mode mode;
    CHECK(h3_preview_mode_parse(NULL,&mode,error,sizeof(error)) && mode==H3_PREVIEW_DENOISED);
    CHECK(h3_preview_mode_parse("",&mode,error,sizeof(error)) && mode==H3_PREVIEW_DENOISED);
    CHECK(h3_preview_mode_parse("denoised",&mode,error,sizeof(error)) && mode==H3_PREVIEW_DENOISED);
    CHECK(h3_preview_mode_parse("noisy",&mode,error,sizeof(error)) && mode==H3_PREVIEW_NOISY);
    for (int i=0;i<3;i++) {
        const char *invalid[]={"DENOISED","denoised ","bad"};
        CHECK(!h3_preview_mode_parse(invalid[i],&mode,error,sizeof(error)));
        CHECK(strstr(error,"H3_PREVIEW_MODE"));
    }
    double max=0,sq=0,norm=0;
    for (int i=0;i<100000;i++) {
        float magnitude=ldexpf(1.0f,(i%21)-10);
        float x=random_float()*magnitude, v=random_float()*magnitude;
        float sigma=.001f+.999f*(random_float()+1)*.5f;
        float next=sigma*.999f*(random_float()+1)*.5f;
        float updated=x;
        CHECK(h3_euler_velocity_step(&updated,&v,1,sigma,next));
        float estimate=v;
        h3_preview_denoised_f32(&estimate,&updated,1,next);
        float independent=fmaf(sigma,v,x);
        double delta=(double)estimate-independent;
        max=fmax(max,fabs(delta)); sq+=delta*delta; norm+=(double)independent*independent;
        CHECK(fabs(delta)<=8*FLT_EPSILON*(fabs(x)+fabs(sigma*v)+1e-20));
        CHECK(updated==fmaf(sigma-next,v,x));
    }
    printf("CPU identity: max %.9g RMSE %.9g relative-L2 %.9g\n",max,sqrt(sq/100000),sqrt(sq/norm));
    uint32_t bits[]={0x80000000,0,0x7fc01234,0xffc05678,0x3f800000};
    float state[5],velocity[5];memcpy(state,bits,sizeof(bits));
    for(int i=0;i<5;i++)velocity[i]=NAN;
    h3_preview_denoised_f32(velocity,state,5,0);
    CHECK(!memcmp(velocity,state,sizeof(state)));
    memset(velocity,0,sizeof(velocity));h3_preview_denoised_f32(velocity,state,5,.75f);
    CHECK(!memcmp(velocity,state,sizeof(state)));
    enum { N=24*42*32*32, REPS=100 };
    float *x=malloc(N*4),*v=malloc(N*4);CHECK(x&&v);
    for(int i=0;i<N;i++){x[i]=random_float();v[i]=random_float();}
    double start=now();for(int i=0;i<REPS;i++)h3_preview_denoised_f32(v,x,N,.5f);
    printf("CPU preview: %u elements %.9g seconds/construction, no allocation\n",N,(now()-start)/(double)REPS);
    free(x);free(v);
}

static void gpu_case(h3_gpu *gpu,int kind,float sigma,float ratio) {
    enum { T=17,H=4,W=6,N=24*T*H*W,ROWS=N/96,OFFSET=193,EXTRA=19 };
    float sample[N+OFFSET+EXTRA],packed[N],want[N],got[N],after[N+OFFSET+EXTRA];
    uint16_t last[N],previous[N],read[N];uint32_t classes[ROWS];
    float strengths[]={0,.125f,.5f,1};
    for(int i=0;i<N+OFFSET+EXTRA;i++)sample[i]=random_float()*4;
    for(int r=0;r<ROWS;r++)classes[r]=(uint32_t)(kind==1 ? (r<12*(H/2)*(W/2)?0:3) : r%4);
    for(int i=0;i<N;i++) {
        last[i]=(uint16_t)(0x3e00+(i*37)%900+(i%2?0x8000:0));
        previous[i]=(uint16_t)(0x3d00+(i*47)%700+(i%3?0x8000:0));
        if(kind && strengths[classes[i/96]]==0) {
            uint32_t bits=i%2?0x80000000:0x7fc01234;
            memcpy(sample+OFFSET+i,&bits,4);last[i]=previous[i]=0x7fc1;
        }
        float v=fp(last[i]);
        if(!kind || ratio!=0)v=fmaf(ratio,v-fp(previous[i]),v);
        float mask=kind?strengths[classes[i/96]]:1;
        if(kind)v*=mask;
        packed[i]=sigma==0 || mask==0 || v==0 ? sample[OFFSET+i] : fmaf(sigma,v,sample[OFFSET+i]);
    }
    CHECK(h3_dit_unpatchify_video(packed,24,T,H,W,want,N));
    h3_gpu_tensor *state=h3_gpu_tensor_from_f32(gpu,sample,N+OFFSET+EXTRA);
    h3_gpu_tensor *a=h3_gpu_tensor_from_bf16(gpu,last,N),*b=h3_gpu_tensor_from_bf16(gpu,previous,N);
    /* A BF16 arena is deliberately reused as raw F32 scratch. */
    h3_gpu_tensor *scratch=h3_gpu_tensor_new_bf16(gpu,2*N);
    h3_gpu_tensor *map=h3_gpu_tensor_from_u32(gpu,classes,ROWS),*masks=h3_gpu_tensor_from_f32(gpu,strengths,4);
    CHECK(state&&a&&b&&scratch&&map&&masks);
    h3_gpu_stats before,stats;CHECK(h3_gpu_get_stats(gpu,&before));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_video_preview_bf16(gpu,scratch,state,OFFSET,a,b,kind?map:NULL,kind?masks:NULL,T,H,W,sigma,ratio));
    CHECK(h3_gpu_submit(gpu));CHECK(h3_gpu_get_stats(gpu,&stats));
    CHECK(before.allocated_bytes==stats.allocated_bytes && before.live_bytes==stats.live_bytes);
    CHECK(h3_gpu_video_preview_read(scratch,got,N));
    CHECK(h3_gpu_tensor_read_f32(state,after,N+OFFSET+EXTRA));CHECK(!memcmp(sample,after,sizeof(sample)));
    CHECK(h3_gpu_tensor_read_bf16(a,read,N));CHECK(!memcmp(last,read,sizeof(last)));
    CHECK(h3_gpu_tensor_read_bf16(b,read,N));CHECK(!memcmp(previous,read,sizeof(previous)));
    double max=0,sq=0,norm=0;size_t finite=0;
    for(int i=0;i<N;i++) {
        CHECK(!memcmp(got+i,want+i,4));
        if(isfinite(want[i])) {double d=(double)got[i]-want[i];max=fmax(max,fabs(d));sq+=d*d;norm+=(double)want[i]*want[i];finite++;}
    }
    printf("GPU preview: kind=%d sigma=%.3g ratio=%.3g max=%.9g RMSE=%.9g relative-L2=%.9g\n",kind,(double)sigma,(double)ratio,max,sqrt(sq/(double)finite),sqrt(sq/fmax(norm,1e-30)));
    CHECK(!h3_gpu_video_preview_bf16(gpu,state,state,OFFSET,a,b,NULL,NULL,T,H,W,sigma,ratio));
    CHECK(!h3_gpu_video_preview_bf16(gpu,a,state,OFFSET,a,b,NULL,NULL,T,H,W,sigma,ratio));
    CHECK(!h3_gpu_video_preview_bf16(gpu,scratch,state,SIZE_MAX,a,b,NULL,NULL,T,H,W,sigma,ratio));
    CHECK(!h3_gpu_video_preview_bf16(gpu,scratch,state,OFFSET,a,b,map,NULL,T,H,W,sigma,ratio));
    CHECK(!h3_gpu_video_preview_bf16(gpu,scratch,state,OFFSET,a,b,NULL,NULL,T,H+1,W,sigma,ratio));
    CHECK(!h3_gpu_video_preview_bf16(gpu,scratch,state,OFFSET,a,b,NULL,NULL,UINT32_MAX,H,W,sigma,ratio));
    CHECK(!h3_gpu_video_preview_bf16(gpu,scratch,state,OFFSET,a,b,NULL,NULL,T,H,W,NAN,ratio));
    CHECK(!h3_gpu_video_preview_bf16(gpu,scratch,state,OFFSET,a,b,NULL,NULL,T,H,W,sigma,INFINITY));
    CHECK(!h3_gpu_video_preview_read(scratch,got,N+1));
    h3_gpu_tensor_free(state);h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);h3_gpu_tensor_free(scratch);h3_gpu_tensor_free(map);h3_gpu_tensor_free(masks);
}

int main(void) {
    numerical();
    h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(gpu);
    for(int kind=0;kind<3;kind++)for(int s=0;s<4;s++)for(int r=0;r<3;r++)
        gpu_case(gpu,kind,(float)s/3,(float)r*.5f);
    h3_gpu_free(gpu);printf("ok: %zu preview numerical, parity, isolation and range checks\n",checks);
    return 0;
}
