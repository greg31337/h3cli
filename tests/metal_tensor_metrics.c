/* Streaming BF16 metrics for captured QKV, projection and block boundaries.
 * Compile with: mkdir -p bin && cc -O3 tests/metal_tensor_metrics.c -lm -o bin/metal_tensor_metrics */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static float fp(uint16_t x){uint32_t u=(uint32_t)x<<16;float f;memcpy(&f,&u,4);return f;}
int main(int argc,char **argv){
    if(argc!=3&&argc!=4)return 2;
    int f32=argc==4&&!strcmp(argv[3],"f32");if(argc==4&&!f32)return 2;
    FILE *a=fopen(argv[1],"rb"),*b=fopen(argv[2],"rb");if(!a||!b)return 2;
    union {uint16_t bf[65536];float f[65536];} x,y;
    double xx=0,yy=0,xy=0,dd=0,mx=0,peak=0;size_t n=0,element_bytes=f32?4:2;
    for(;;){size_t nx=fread(&x,element_bytes,65536,a),ny=fread(&y,element_bytes,65536,b);if(nx!=ny)return 2;if(!nx)break;
        for(size_t i=0;i<nx;i++){double f=f32?x.f[i]:fp(x.bf[i]),g=f32?y.f[i]:fp(y.bf[i]),d=f-g;if(!isfinite(f)||!isfinite(g))return 2;
            xx+=f*f;yy+=g*g;xy+=f*g;dd+=d*d;mx=fmax(mx,fabs(d));peak=fmax(peak,fabs(f));}n+=nx;
    }
    if(!n||ferror(a)||ferror(b))return 2;fclose(a);fclose(b);
    printf("{\"elements\":%zu,\"relative_l2\":%.12g,\"cosine\":%.12g,\"max_abs\":%.12g,\"peak\":%.12g}\n",
        n,sqrt(dd/fmax(xx,1e-30)),xx*yy>0?xy/sqrt(xx*yy):1,mx,peak);return 0;
}
