#include "src/upscale/upscale_network.h"
#include "src/upscale/upscale_gpu.h"
#include "src/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int trace(const char *name,const float *data,int c,int t,int h,int w,void *opaque) {
    char path[4096];if(snprintf(path,sizeof(path),"%s/%s.f32",(const char*)opaque,name)>=(int)sizeof(path))return 1;
    FILE *f=fopen(path,"wb");if(!f)return 1;
    size_t n=(size_t)c*t*h*w;int ok=fwrite(data,4,n,f)==n;
    if(fclose(f))ok=0;return !ok;
}
static int no_memory(uint64_t *bytes,void *opaque) {(void)opaque;*bytes=0;return 1;}
static int cancel(const char *phase,int completed,int total,void *opaque) {
    (void)phase;(void)total;return completed>=*(int *)opaque;
}
int main(int argc,char **argv) {
    if(argc!=4){fprintf(stderr,"usage: upscale_network WEIGHTS FIXTURES NEW_OUTPUT_DIRECTORY\n");return 2;}
    char error[1024]={0};if(mkdir(argv[3],0700)){perror("output directory must be new");return 2;}
    h3_upscale_model *m=h3_upscale_model_load(argv[1],error,sizeof(error));
    if(!m){fprintf(stderr,"%s\n",error);return 1;}
    const char *names[]={"constant","channel-ramp","impulse-edge","random","real"};
    int times[]={1,2,27,27,27},heights[]={2,3,3,3,3},widths[]={3,4,4,4,4};
    int ok=1;
    for(int i=0;i<5&&ok;i++) {
        char path[4096],out[4096];snprintf(path,sizeof(path),"%s/%s/input.f32",argv[2],names[i]);
        size_t count=(size_t)24*times[i]*heights[i]*widths[i];float *input=malloc(count*4),*copy=malloc(count*4);
        FILE *f=fopen(path,"rb");ok=f&&input&&copy&&fread(input,4,count,f)==count&&fgetc(f)==EOF;
        if(f)fclose(f);
        snprintf(out,sizeof(out),"%s/%s",argv[3],names[i]);if(mkdir(out,0700))ok=0;
        float *result=NULL;
        if(ok) {
            memcpy(copy,input,count*4);
            result=h3_upscale_volume(m,input,times[i],heights[i],widths[i],heights[i]*2,widths[i]*2,
                NULL,NULL,trace,out,error,sizeof(error));
            ok=result&&!memcmp(input,copy,count*4);
        }
        if(ok) {
            char ep[4096],isolated[4096];float emb[64];
            snprintf(ep,sizeof(ep),"%s/%s/bf16/embed.f32",argv[2],names[i]);FILE *ef=fopen(ep,"rb");
            ok=ef&&fread(emb,4,64,ef)==64;if(ef)fclose(ef);
            snprintf(isolated,sizeof(isolated),"%s/%s/isolated",argv[3],names[i]);if(mkdir(isolated,0700))ok=0;
            for(int layer=0;layer<38&&ok;layer++) {
                char name[64],prior[64],ip[4096];int h=heights[i],w=widths[i];
                if(layer<18) {
                    snprintf(name,sizeof(name),"in_blocks.%d",layer);
                    if(layer)snprintf(prior,sizeof(prior),"in_blocks.%d",layer-1);else snprintf(prior,sizeof(prior),"conv_in");
                } else if(layer<36) {
                    snprintf(name,sizeof(name),"out_blocks.%d",layer-18);
                    snprintf(prior,sizeof(prior),layer==18?"in_blocks.%d":"out_blocks.%d",layer==18?17:layer-19);
                    if(layer>18){h*=2;w*=2;}
                } else {
                    snprintf(name,sizeof(name),"%s",layer==36?"norm_out":"conv_out");
                    snprintf(prior,sizeof(prior),"%s",layer==36?"out_blocks.17":"norm_out");h*=2;w*=2;
                }
                size_t ni=(size_t)512*times[i]*h*w;float *v=malloc(ni*4);
                snprintf(ip,sizeof(ip),"%s/%s/bf16/%s.f32",argv[2],names[i],prior);FILE *pf=fopen(ip,"rb");
                ok=v&&pf&&fread(v,4,ni,pf)==ni;if(pf)fclose(pf);
                float *y=ok?h3_upscale_probe_layer(m,name,v,emb,times[i],h,w,error,sizeof(error)):NULL;
                if(layer==18){h*=2;w*=2;}
                ok=y&&!trace(name,y,layer==37?24:512,times[i],h,w,isolated);
                free(v);free(y);
            }
        }
        free(input);free(copy);free(result);
    }
    /* Reuse after cancellation/admission failure and interleave independently
     * owned GPU contexts. These are bounded latent probes, no DiT evaluations. */
    if(ok) {
        float input[24*2*3];for(size_t i=0;i<sizeof(input)/sizeof(*input);i++)input[i]=.25f;
        float *base=h3_upscale_volume(m,input,1,2,3,4,6,NULL,NULL,NULL,NULL,error,sizeof(error));
        ok=base!=NULL;
        for(int boundary=0;boundary<=40&&ok;boundary+=20) {
            float *v=h3_upscale_volume(m,input,1,2,3,4,6,cancel,&boundary,NULL,NULL,error,sizeof(error));
            ok=v==NULL;free(v);
        }
        input[0]=NAN;float *bad=h3_upscale_volume(m,input,1,2,3,4,6,NULL,NULL,NULL,NULL,error,sizeof(error));
        ok=ok&&!bad;free(bad);input[0]=.25f;
        h3_memory_set_test_query(no_memory,NULL);
        bad=h3_upscale_volume(m,input,1,2,3,4,6,NULL,NULL,NULL,NULL,error,sizeof(error));
        h3_memory_set_test_query(NULL,NULL);ok=ok&&!bad;free(bad);
        h3_upscale_model *other=h3_upscale_model_load(argv[1],error,sizeof(error));
        if(!other)ok=0;
        for(int turn=0;turn<3&&ok;turn++) {
            float *v=h3_upscale_volume(turn==1?other:m,input,1,2,3,4,6,NULL,NULL,NULL,NULL,error,sizeof(error));
            ok=v&&base&&!memcmp(v,base,24*4*6*sizeof(float));free(v);
        }
        h3_upscale_model_free(other);free(base);
        size_t counts[4];
        ok=ok&&!h3_upscale_gpu_counts(H3_UP_COLUMNS,(h3_upscale_gpu_shape){.time=27,.height=24,.width=42,
            .channels=512,.rows=UINT32_MAX,.kernel_time=3,.kernel_space=3},counts);
    }
    h3_upscale_stats stats;h3_upscale_model_stats(m,&stats);
    if(!ok)fprintf(stderr,"FAIL: %s\n",error);
    else printf("{\"passed\":true,\"cases\":5,\"load_seconds\":%.9g,\"forward_seconds\":%.9g,\"peak_gpu_bytes\":%llu,\"workspace_bytes\":%llu,\"convolutions\":%llu,\"convolution_tiles\":%llu}\n",
        stats.load_seconds,stats.forward_seconds,(unsigned long long)stats.peak_gpu_bytes,
        (unsigned long long)stats.workspace_bytes,(unsigned long long)stats.convolutions,
        (unsigned long long)stats.convolution_tiles);
    h3_upscale_model_free(m);return !ok;
}
