/* Source-independent replay of the pre-change renderer's initial PCG draws.
 * Build against each tree's src/host.c. This does not access models or a GPU;
 * full-render equivalence is checked independently by the fast campaign. */
#include "src/host.h"
#include <stdio.h>
#include <stdlib.h>

static int save(const char *root,const char *name,const void *data,size_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",root,name);
    if(n<0||(size_t)n>=sizeof(path))return 0;
    FILE *f=fopen(path,"wbx");if(!f)return 0;
    int ok=fwrite(data,1,bytes,f)==bytes;if(fclose(f))ok=0;return ok;
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    const int frames[]={124,362};
    for(int c=0;c<2;c++) {
        h3_temporal_shape t=h3_temporal(frames[c]);int w,h;
        h3_latent_canvas(640,480,&w,&h);
        size_t counts[]={24u*(size_t)t.video_t*h*w,32u*2u*(size_t)t.audio_t};
        for(int m=0;m<2;m++) {
            if(!counts[m]||counts[m]>(32u<<20)/sizeof(float))return 1;
            float *data=malloc(counts[m]*sizeof(float));if(!data)return 1;
            h3_rng rng={0};h3_rng_seed(&rng,42);h3_rng_fill_normal(&rng,data,counts[m]);
            char name[80];snprintf(name,sizeof(name),"%d-%s.f32",frames[c],m?"audio":"video");
            int ok=save(argv[1],name,data,counts[m]*sizeof(float));free(data);if(!ok)return 1;
            char record[512];int n=snprintf(record,sizeof(record),
                "{\"frames\":%d,\"video_t\":%d,\"audio_t\":%d,\"elements\":%zu,\"seed\":42,\"state\":%llu,\"increment\":%llu,\"spare\":%.9g,\"has_spare\":%d}\n",
                t.frame_count,t.video_t,t.audio_t,counts[m],(unsigned long long)rng.state,
                (unsigned long long)rng.increment,rng.spare,rng.has_spare);
            if(n<0||(size_t)n>=sizeof(record))return 1;
            snprintf(name,sizeof(name),"%d-%s.json",frames[c],m?"audio":"video");
            if(!save(argv[1],name,record,(size_t)n))return 1;
        }
    }
    return 0;
}
