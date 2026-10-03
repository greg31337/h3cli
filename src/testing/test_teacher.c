#include "src/testing/test_teacher.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_values(const char *directory,int step,const char *name,float *out,size_t count) {
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/step-%03d-%s-latent.f32",directory,step,name);
    if(n<0 || n>=(int)sizeof(path))return 0;
    FILE *file=fopen(path,"rb");if(!file)return 0;
    int ok=fread(out,sizeof(*out),count,file)==count && fgetc(file)==EOF && !ferror(file);
    if(fclose(file))ok=0;
    for(size_t i=0;ok&&i<count;i++)if(!isfinite(out[i]))ok=0;
    return ok;
}
static int load(const char *directory,int step,int first,int last,float *video,size_t nv,
                float *audio,size_t na,char *error,size_t size) {
    int ok=directory&&*directory&&step>=first&&step<=last&&video&&audio&&nv&&na&&
        nv<=SIZE_MAX/sizeof(float)&&na<=SIZE_MAX/sizeof(float);
    float *v=ok?malloc(nv*sizeof(float)):NULL,*a=ok?malloc(na*sizeof(float)):NULL;
    ok=ok&&v&&a&&read_values(directory,step,"video",v,nv)&&read_values(directory,step,"audio",a,na);
    if(ok){memcpy(video,v,nv*sizeof(float));memcpy(audio,a,na*sizeof(float));}
    else if(error&&size)snprintf(error,size,"missing, malformed or nonfinite teacher input at evaluation %d",step+1);
    free(v);free(a);return ok;
}
int h3_test_teacher_load(const char *directory,int step,float *video,size_t nv,
                         float *audio,size_t na,char *error,size_t size) {
    return load(directory,step,1,5,video,nv,audio,na,error,size);
}
int h3_test_teacher_load_sglang(const char *directory,int step,float *video,size_t nv,
                               float *audio,size_t na,char *error,size_t size) {
    return load(directory,step,0,49,video,nv,audio,na,error,size);
}
