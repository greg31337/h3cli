#include "src/denoise/dit.h"
#include "src/internal.h"
#include "src/weights/safetensors.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
static char error[512];
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s (%s)\n",__FILE__,__LINE__,#x,error); exit(1); } } while (0)
static void *read_tensor(h3_st_header *f, const char *name, h3_dtype dtype, size_t count) {
    const h3_st_tensor *t=h3_st_find(f,name); CHECK(t && t->dtype==dtype && h3_st_tensor_elements(t)==count);
    size_t bytes=(size_t)(t->data_end-t->data_begin);
    void *p=malloc(bytes); CHECK(p && h3_st_read_data(f,t,p,bytes,error,sizeof(error))); return p;
}
static void layout_case(const char *directory, const char *name) {
    char path[4096]; snprintf(path,sizeof(path),"%s/layout-%s.safetensors",directory,name);
    h3_st_header f; CHECK(h3_st_read_header(path,&f,error,sizeof(error)));
    const h3_st_tensor *rt=h3_st_find(&f,"x.references"); CHECK(rt && rt->ndim==2 && rt->shape[1]==5);
    size_t n=(size_t)rt->shape[0]; CHECK(n>0 && n<=3);
    int *r=read_tensor(&f,"x.references",H3_DTYPE_I32,n*5);
    int *s=read_tensor(&f,"x.spec",H3_DTYPE_I32,6);
    h3_layout_ref refs[3];
    for (size_t i=0;i<n;i++) refs[i]=(h3_layout_ref){r[5*i],r[5*i+1],r[5*i+2],r[5*i+3],r[5*i+4]};
    h3_layout_spec spec={s[0],s[1],s[2],s[3],s[4],s[5],NULL,0,refs,n};
    h3_layout l; CHECK(h3_layout_build(&spec,&l,error,sizeof(error)));
    double *positions=read_tensor(&f,"x.positions",H3_DTYPE_F64,l.seq_len*3);
    for (size_t i=0;i<l.seq_len;i++) {
        CHECK(fabs(l.positions[i].t-positions[i*3])<1e-12);
        CHECK(fabs(l.positions[i].h-positions[i*3+1])<1e-12);
        CHECK(fabs(l.positions[i].w-positions[i*3+2])<1e-12);
    }
    int64_t *counts=read_tensor(&f,"x.counts",H3_DTYPE_I64,4);
    CHECK(l.img_cond_rows==(size_t)counts[0] && l.audio_cond_rows==(size_t)counts[1]);
    CHECK(l.img_target_rows==(size_t)counts[2] && l.audio_target_rows==(size_t)counts[3]);
    int64_t *vi=read_tensor(&f,"x.video_indices",H3_DTYPE_I64,l.img_cond_rows+l.img_target_rows);
    int64_t *ai=read_tensor(&f,"x.audio_indices",H3_DTYPE_I64,l.audio_cond_rows+l.audio_target_rows);
    size_t v=0,a=0;
    for (size_t i=1;i<l.segment_count;i++) {
        h3_segment seg=l.segments[i];
        for (size_t j=seg.start;j<seg.stop;j++) {
            if (seg.kind==H3_SEG_VIDEO || seg.kind==H3_SEG_REF_IMAGE || seg.kind==H3_SEG_COND) CHECK(vi[v++]==(int64_t)j);
            else CHECK(ai[a++]==(int64_t)j);
        }
    }
    double *cursor=read_tensor(&f,"x.cursor",H3_DTYPE_F64,1);
    int64_t *start=read_tensor(&f,"x.target_start",H3_DTYPE_I64,1);
    CHECK((size_t)*start==(size_t)s[0]+l.img_cond_rows+l.audio_cond_rows);
    CHECK(fabs(l.positions[*start].t-*cursor)<1e-12);
    /* Test the real production augmentation with a guard on either side.
     * Every independent visual block must reset its request RNG, including
     * the last three temporal positions added by the released 56-frame path. */
    size_t nv=l.img_cond_rows*96,na=l.audio_cond_rows*32;
    float *video=malloc((nv+2)*4), *audio=malloc((na+2)*4); CHECK(video && audio);
    for (size_t i=0;i<nv+2;i++) video[i]=2.0f;
    for (size_t i=0;i<na+2;i++) audio[i]=3.0f;
    h3_params p=H3_PARAMS_DEFAULT; p.reference_count=n; p.seed=72;
    CHECK(h3_augment_conditions(&p,1,96,64,refs,video+1,nv,audio+1,na));
    CHECK(video[0]==2 && video[nv+1]==2 && audio[0]==3 && audio[na+1]==3);
    v=1;a=1;
    for (size_t i=0;i<n;i++) {
        h3_rng rng; h3_rng_seed(&rng,72);
        size_t span=refs[i].kind==H3_LAYOUT_REF_AUDIO?0:(size_t)refs[i].latent_t*refs[i].latent_h*refs[i].latent_w/4*96;
        for (size_t j=0;j<span;j++) CHECK(fabsf(video[v++]- (0.999f*2.0f+0.001f*h3_rng_normal(&rng)))<1e-6f);
        h3_rng_seed(&rng,73);
        for (size_t j=0;j<(size_t)refs[i].audio_t*64;j++) CHECK(fabsf(audio[a++]- (0.999f*3.0f+0.001f*h3_rng_normal(&rng)))<1e-6f);
    }
    CHECK(v==nv+1 && a==na+1);
    if (!strcmp(name,"released56")) CHECK(l.img_cond_rows==17*4);
    free(video);free(audio);free(r);free(s);free(positions);free(counts);free(vi);free(ai);free(cursor);free(start);
    h3_layout_free(&l); h3_st_free_header(&f);
}
int main(int argc,char **argv) {
    if (argc==5 && !strcmp(argv[1],"--noise")) {
        size_t n=(size_t)strtoull(argv[2],NULL,10);
        h3_rng rng; h3_rng_seed(&rng,strtoull(argv[3],NULL,10));
        FILE *f=fopen(argv[4],"wb"); CHECK(f);
        for (size_t i=0;i<n;i++) { float value=h3_rng_normal(&rng); CHECK(fwrite(&value,4,1,f)==1); }
        CHECK(!fclose(f)); return 0;
    }
    const char *dir=argc>1?argv[1]:"tests/fixtures/refvideo-layout";
    const char *cases[]={"released56","image-video","video-audio","two-videos","three-videos","audio-video-image"};
    for (size_t i=0;i<sizeof(cases)/sizeof(*cases);i++) layout_case(dir,cases[i]);
    printf("ok: %d official Ref2VA layout, RoPE, row allocation and augmentation checks\n",checks);
    return 0;
}
