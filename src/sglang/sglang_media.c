/* Reference-only byte preprocessing. The Lanczos coefficient recipe and
 * fixed-point rounding follow Pillow 11.3.0 src/libImaging/Resample.c.
 * Copyright notices and MIT-CMU license: third_party/pillow/LICENSE. */
#include "src/sglang/sglang.h"
#include "src/media/ffmpeg.h"
#include "src/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <dlfcn.h>
#endif

typedef struct { int size; int *bounds; int32_t *weights; } resample_axis;
static void axis_free(resample_axis *a) { free(a->bounds);free(a->weights); }
static double sinc(double x) {
    if(x==0)return 1;
    x*=3.14159265358979323846;return sin(x)/x;
}
static int axis_make(int in,int out,resample_axis *a) {
    double scale=(double)in/out,fs=fmax(scale,1),support=3*fs;
    a->size=(int)ceil(support)*2+1;
    if((size_t)out*a->size> (64u<<20)/sizeof(double))return 0;
    a->bounds=calloc((size_t)out*2,sizeof(int));
    a->weights=calloc((size_t)out*a->size,sizeof(int32_t));
    double *k=malloc((size_t)a->size*sizeof(double));
    if(!a->bounds||!a->weights||!k){free(k);return 0;}
    for(int p=0;p<out;p++) {
        double center=(p+.5)*scale,sum=0,ss=1/fs;
        int lo=(int)(center-support+.5),hi=(int)(center+support+.5);
        if (lo < 0)
            lo = 0;
        if (hi > in)
            hi = in;
        a->bounds[p*2]=lo;a->bounds[p*2+1]=hi-lo;
        for(int j=0;j<hi-lo;j++) {
            double x=(j+lo-center+.5)*ss;
            k[j]=x>=-3&&x<3?sinc(x)*sinc(x/3):0;sum+=k[j];
        }
        for(int j=0;j<hi-lo;j++) {
            double v=sum!=0?k[j]/sum:k[j];
            a->weights[(size_t)p*a->size+j]=(int32_t)(v<0?v*(1<<22)-.5:v*(1<<22)+.5);
        }
    }
    free(k);return 1;
}
static uint8_t clip(int64_t sum) {
    int64_t value=sum>>22;return (uint8_t)(value<0?0:value>255?255:value);
}
int h3_sglang_resize_rgb(const uint8_t *in,int iw,int ih,int ow,int oh,uint8_t **out) {
    if (!out)
        return 0;
    *out = NULL;
    if(!in||iw<1||ih<1||ow<1||oh<1||iw>32768||ih>32768||ow>8192||oh>8192||
       (uint64_t)iw*ih*3>(128u<<20)||(uint64_t)ow*ih*3>(128u<<20)||(uint64_t)ow*oh*3>(128u<<20))return 0;
    resample_axis x={0},y={0};uint8_t *temp=NULL,*dst=NULL;int ok=0;
    if(!axis_make(iw,ow,&x)||!axis_make(ih,oh,&y))goto done;
    temp=malloc((size_t)ow*ih*3);dst=malloc((size_t)ow*oh*3);if(!temp||!dst)goto done;
    if(iw==ow)memcpy(temp,in,(size_t)ow*ih*3);
    else for(int row=0;row<ih;row++)for(int col=0;col<ow;col++)for(int c=0;c<3;c++) {
        int64_t sum=1<<21;
        for(int j=0;j<x.bounds[col*2+1];j++)sum+=(int64_t)in[((size_t)row*iw+x.bounds[col*2]+j)*3+c]*x.weights[(size_t)col*x.size+j];
        temp[((size_t)row*ow+col)*3+c]=clip(sum);
    }
    if(ih==oh)memcpy(dst,temp,(size_t)ow*oh*3);
    else for(int row=0;row<oh;row++)for(int col=0;col<ow;col++)for(int c=0;c<3;c++) {
        int64_t sum=1<<21;
        for(int j=0;j<y.bounds[row*2+1];j++)sum+=(int64_t)temp[((size_t)(y.bounds[row*2]+j)*ow+col)*3+c]*y.weights[(size_t)row*y.size+j];
        dst[((size_t)row*ow+col)*3+c]=clip(sum);
    }
    *out=dst;dst=NULL;ok=1;
 done:
    axis_free(&x);axis_free(&y);free(temp);free(dst);return ok;
}

/* Stable TurboJPEG v2 ABI; only the RGB decoder is used. Keeping this library
 * private to preprocessing leaves the protected FFmpeg/fast path intact. */
static int read_jpeg(FILE *f,uint8_t **rgb,int expected_w,int expected_h,char *error,size_t size) {
#ifdef __linux__
    int ok=0;void *handle=NULL;uint8_t *data=NULL,*pixels=NULL;
    const char *path=getenv("H3_SGLANG_JPEG_LIBRARY");
    void *library=dlopen(path&&*path?path:"libturbojpeg.so.0",RTLD_NOW|RTLD_LOCAL);
    if(!library){snprintf(error,size,"reference JPEG decoding requires libturbojpeg.so.0 (or H3_SGLANG_JPEG_LIBRARY)");return 0;}
    void *(*init)(void);
    int (*header)(void*,const unsigned char*,unsigned long,int*,int*,int*,int*);
    int (*decode)(void*,const unsigned char*,unsigned long,unsigned char*,int,int,int,int,int);
    int (*destroy)(void*);
    /* POSIX dlsym returns a function address in a void pointer. Copy its
     * representation without an ISO C object/function pointer conversion. */
#define LOAD_JPEG_SYMBOL(function, symbol) do { \
    void *address=dlsym(library,symbol); \
    _Static_assert(sizeof(function)==sizeof(address),"POSIX function pointer size"); \
    memcpy(&function,&address,sizeof(function)); \
} while(0)
    LOAD_JPEG_SYMBOL(init,"tjInitDecompress");
    LOAD_JPEG_SYMBOL(header,"tjDecompressHeader3");
    LOAD_JPEG_SYMBOL(decode,"tjDecompress2");
    LOAD_JPEG_SYMBOL(destroy,"tjDestroy");
#undef LOAD_JPEG_SYMBOL
    if(!init||!header||!decode||!destroy)goto done;
    if (fseek(f, 0, SEEK_END))
        goto done;
    long bytes = ftell(f);
    if(bytes<=0||bytes>(128<<20)||fseek(f,0,SEEK_SET))goto done;
    data=malloc((size_t)bytes);if(!data||fread(data,1,(size_t)bytes,f)!=(size_t)bytes)goto done;
    handle=init();if(!handle)goto done;
    int w=0,h=0,subsampling=0,color=0;
    if(header(handle,data,(unsigned long)bytes,&w,&h,&subsampling,&color)||w!=expected_w||h!=expected_h)goto done;
    pixels=malloc((size_t)w*h*3);if(!pixels)goto done;
    /* TJPF_RGB=0, TJFLAG_ACCURATEDCT=4096. Fancy chroma upsampling stays on. */
    if(decode(handle,data,(unsigned long)bytes,pixels,w,0,h,0,4096))goto done;
    *rgb=pixels;pixels=NULL;ok=1;
 done:
     if (handle && destroy)
         destroy(handle);
     free(data);
     free(pixels);
     dlclose(library);
     if (!ok && !*error)
         snprintf(error, size, "reference JPEG decode failed or source geometry changed");
     return ok;
#else
    (void)f;(void)rgb;(void)expected_w;(void)expected_h;
    snprintf(error,size,"reference JPEG preprocessing requires Linux");return 0;
#endif
}
int h3_sglang_read_reference_image(const char *path,int iw,int ih,int ow,int oh,float **out,char *error,size_t size) {
    *out=NULL;uint8_t *rgb=NULL,*resized=NULL;float *values=NULL;int ok=0;
    if(iw<1||ih<1||ow<1||oh<1||(uint64_t)iw*ih*3>(128u<<20)||(uint64_t)ow*oh*3>(128u<<20)) {
        snprintf(error,size,"reference image exceeds bounded preprocessing geometry");return 0;
    }
    size_t source=(size_t)iw*ih*3,count=(size_t)ow*oh*3;
    if(!h3_memory_check(source*5+count*5,"reference image preprocessing",error,size))return 0;
    FILE *f=fopen(path,"rb");if(!f){snprintf(error,size,"cannot open reference image");return 0;}
    int first=fgetc(f),second=fgetc(f);
    if(first==255&&second==216)ok=read_jpeg(f,&rgb,iw,ih,error,size);
    else {
        ok=h3_ffmpeg_read_image_f32(path,iw,ih,H3_IMAGE_FIT_STRETCH,&values,error,size);
        if(ok) {
            rgb=malloc(source);ok=rgb!=NULL;
            if(ok)for(size_t p=0;p<source/3;p++)for(size_t c=0;c<3;c++)rgb[p*3+c]=(uint8_t)roundf(values[c*(source/3)+p]*255.f);
        }
        free(values);values=NULL;
    }
    fclose(f);
    if (ok)
        ok = h3_sglang_resize_rgb(rgb, iw, ih, ow, oh, &resized);
    free(rgb);
    if(ok){values=malloc(count*sizeof(float));ok=values!=NULL;}
    if(ok)for(size_t p=0;p<count/3;p++)for(size_t c=0;c<3;c++)values[c*(count/3)+p]=(float)resized[p*3+c]/255.f;
    free(resized);if(ok)*out=values;else {free(values);if(!*error)snprintf(error,size,"reference image preprocessing failed");}return ok;
}

/* Torchaudio's default 44100 -> 32000 polyphase Hann sinc. Kernel preparation
 * uses F64 after the phase term's pinned FP32 division; each coefficient is
 * rounded to FP32 before waveform convolution. No GPU or Python dependency. */
int h3_sglang_read_soundtrack(const char *path,int max_frames,float **pcm,int *samples,char *error,size_t size) {
    *pcm=NULL;*samples=0;
    if(max_frames<1||max_frames>362)return 0;
    int cap=(int)fmin(15*44100,ceil((double)max_frames*44100/24)),source_samples=0;
    float *source=NULL;
    if(!h3_ffmpeg_read_soundtrack_44100(path,cap,&source,&source_samples,error,size))return 0;
    const int phases=320,stride=441,width=9,length=459;
    int count=(int)(((int64_t)source_samples*phases+stride-1)/stride);
    float *out=malloc((size_t)count*2*sizeof(float));float *kernel=malloc((size_t)phases*length*sizeof(float));
    if(!out||!kernel){free(source);free(out);free(kernel);snprintf(error,size,"cannot allocate reference soundtrack resampler");return 0;}
    const double base=320*.99,pi=3.14159265358979323846;
    for(int p=0;p<phases;p++)for(int j=0;j<length;j++) {
        double t=((double)(-(float)p/(float)phases)+(double)(j-width)/stride)*base;
        t=fmin(6,fmax(-6,t));double window=cos(t*pi/6/2);window*=window;t*=pi;
        kernel[p*length+j]=(float)((t==0?1:sin(t)/t)*(window*(base/stride)));
    }
    for(int channel=0;channel<2;channel++)for(int i=0;i<count;i++) {
        int block=i/phases,phase=i%phases;float sum=0;
        for(int j=0;j<length;j++) {
            int index=block*stride+j-width;
            if(index>=0&&index<source_samples)sum=fmaf(source[(size_t)channel*source_samples+index],kernel[phase*length+j],sum);
        }
        out[(size_t)channel*count+i]=sum;
    }
    free(kernel);free(source);*pcm=out;*samples=count;return 1;
}
