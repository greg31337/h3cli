/* Linux high-quality RGB resize: separable Lanczos-3 with antialiasing.
 * Darwin retains its original vImage implementation in src/media/resize_metal.c. */
#include "src/host.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
static double lanczos(double x) {
    x=fabs(x);
    if(x<1e-12) return 1;
    if(x>=3) return 0;
    x*=3.14159265358979323846;
    return sin(x)*sin(x/3)/(x*x/3);
}
int h3_resize_rgb24_high_quality(const uint8_t *input,int frames,
    int iw,int ih,int ow,int oh,uint8_t **output) {
    if(output)*output=NULL;
    if(!input||!output||frames<1||iw<1||ih<1||ow<1||oh<1)return 0;
    size_t area=(size_t)ow*(size_t)oh;
    if(area>SIZE_MAX/3/(size_t)frames || (size_t)ow>SIZE_MAX/3/(size_t)ih/sizeof(float))return 0;
    uint8_t *dst=malloc(area*3*(size_t)frames);
    if(!dst)return 0;
    if(iw==ow&&ih==oh){memcpy(dst,input,area*3*(size_t)frames);*output=dst;return 1;}
    float *tmp=malloc((size_t)ow*(size_t)ih*3*sizeof(float));
    if(!tmp){free(dst);return 0;}
    double sx=fmax(1,(double)iw/ow),sy=fmax(1,(double)ih/oh);
    for(int f=0;f<frames;f++) {
        const uint8_t *src=input+(size_t)f*(size_t)iw*(size_t)ih*3;
        for(int y=0;y<ih;y++)for(int x=0;x<ow;x++) {
            double center=((double)x+0.5)*iw/ow-0.5,sum[3]={0},norm=0;
            for(int k=(int)ceil(center-3*sx);k<=floor(center+3*sx);k++){
                double w=lanczos((center-k)/sx);int clamped=k<0?0:k>=iw?iw-1:k;
                for(int c=0;c<3;c++)sum[c]+=w*src[((size_t)y*(size_t)iw+(size_t)clamped)*3+(size_t)c];
                norm+=w;
            }
            for(int c=0;c<3;c++)tmp[((size_t)y*(size_t)ow+(size_t)x)*3+(size_t)c]=(float)(sum[c]/norm);
        }
        for(int y=0;y<oh;y++)for(int x=0;x<ow;x++) {
            double center=((double)y+0.5)*ih/oh-0.5,sum[3]={0},norm=0;
            for(int k=(int)ceil(center-3*sy);k<=floor(center+3*sy);k++){
                double w=lanczos((center-k)/sy);int clamped=k<0?0:k>=ih?ih-1:k;
                for(int c=0;c<3;c++)sum[c]+=w*tmp[((size_t)clamped*(size_t)ow+(size_t)x)*3+(size_t)c];
                norm+=w;
            }
            for(int c=0;c<3;c++)dst[(((size_t)f*(size_t)oh+(size_t)y)*(size_t)ow+(size_t)x)*3+(size_t)c]=(uint8_t)fmin(255,fmax(0,round(sum[c]/norm)));
        }
    }
    free(tmp);*output=dst;return 1;
}
