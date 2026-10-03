#include "src/metal/ane.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
static uint16_t bf(float f){uint32_t b;memcpy(&b,&f,4);b+=0x7fff+((b>>16)&1);return (uint16_t)(b>>16);}
static float unbf(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
int main(int argc,char **argv) {
    int k=argc>1?atoi(argv[1]):1024,n=argc>2?atoi(argv[2]):1024,rows=argc>3?atoi(argv[3]):257;
    int chunk=argc>4?atoi(argv[4]):256;
    if(k<1||n<1||rows<1)return 2;
    char error[4096]={0};h3_ane_stats s={0};
    h3_ane *a=h3_ane_create(chunk,k,n,1024,getenv("H3_ANE_CACHE_DIR"),&s,error,sizeof(error));
    if(!a){fprintf(stderr,"ANE unavailable: %s (plan %d/%d)\n",error,s.ane_matmuls,s.total_matmuls);return 1;}
    fprintf(stderr,"ANE graph: load %.3fs cache %d planned %d/%d matmuls, arrays %.1f MiB\n",s.load_seconds,s.cache_hit,s.ane_matmuls,s.total_matmuls,(double)s.memory_bytes/1048576.);
    uint16_t *x=malloc((size_t)rows*k*2),*w=malloc((size_t)n*k*2),*y=malloc((size_t)rows*n*2);
    if(!x||!w||!y)return 2;
    for(size_t i=0;i<(size_t)rows*k;i++)x[i]=bf(sinf((float)(i%997)*.17f)*3.f);
    for(size_t i=0;i<(size_t)n*k;i++)w[i]=bf(cosf((float)(i%991)*.11f)*.03f);
    int ok=h3_ane_predict(a,x,w,y,rows,&s,error,sizeof(error));
    if(!ok){fprintf(stderr,"prediction failed: %s\n",error);return 1;}
    double num=0,den=0;unsigned probes=0;
    for(int r=0;r<rows;r+=MAX(1,rows/7))for(int c=0;c<n;c+=MAX(1,n/31)) {
        double ref=0;for(int j=0;j<k;j++)ref+=(double)unbf(x[(size_t)r*k+j])*unbf(w[(size_t)c*k+j]);
        double d=unbf(y[(size_t)r*n+c])-ref;num+=d*d;den+=ref*ref;probes++;
    }
    double rel=sqrt(num/fmax(den,1e-30));
    printf("{\"rows\":%d,\"k\":%d,\"n\":%d,\"chunk\":%d,\"relative_l2\":%.9g,\"probes\":%u,\"pack_seconds\":%.6f,\"predict_seconds\":%.6f,\"unpack_seconds\":%.6f,\"pass\":%s}\n",rows,k,n,chunk,rel,probes,s.pack_seconds,s.predict_seconds,s.unpack_seconds,rel<.01?"true":"false");
    h3_ane_free(a);free(x);free(w);free(y);return rel<.01?0:1;
}
