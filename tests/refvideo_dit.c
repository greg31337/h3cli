#include "src/denoise/dit.h"
#include "src/weights/safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static char error[512];
static void die(const char *m) { fprintf(stderr,"FAIL: %s %s\n",m,error); exit(1); }
static void *read_tensor(h3_st_header *f,const char *key,h3_dtype dtype,size_t count) {
    const h3_st_tensor *t=h3_st_find(f,key);
    if (!t || t->dtype!=dtype || h3_st_tensor_elements(t)!=count) die(key);
    size_t bytes=(size_t)(t->data_end-t->data_begin); void *v=malloc(bytes?bytes:1);
    if (!v || !h3_st_read_data(f,t,v,bytes,error,sizeof(error))) die(key); return v;
}
static void save(const char *base,const char *name,const float *values,size_t n) {
    char path[4096]; snprintf(path,sizeof(path),"%s.%s",base,name); FILE *f=fopen(path,"wb");
    if (!f || fwrite(values,4,n,f)!=n || fclose(f)) die("write");
}
static int progress(const char *phase,int done,int total,void *opaque) {
    (void)opaque;
    if (!done || done==total || done%10==0) fprintf(stderr,"%s %d/%d\n",phase,done,total);
    return 0;
}
int main(int argc,char **argv) {
    if (argc!=6) die("usage: refvideo_dit WEIGHTS FIXTURE OUTPUT STEPS FULL");
    setenv("H3_DISABLE_FUSED_MLP","1",1); setenv("H3_DIT_F32_FINAL","1",1);
    h3_st_header f; if (!h3_st_read_header(argv[2],&f,error,sizeof(error))) die("fixture");
    int *s=read_tensor(&f,"x.spec",H3_DTYPE_I32,6);
    const h3_st_tensor *r=h3_st_find(&f,"x.references");
    if (!r || r->ndim!=2 || r->shape[1]!=5) die("references");
    size_t nr=(size_t)r->shape[0]; int *geometry=read_tensor(&f,"x.references",H3_DTYPE_I32,nr*5);
    h3_layout_ref *refs=calloc(nr,sizeof(*refs)); if (!refs) die("allocation");
    for (size_t i=0;i<nr;i++) refs[i]=(h3_layout_ref){geometry[i*5],geometry[i*5+1],geometry[i*5+2],geometry[i*5+3],geometry[i*5+4]};
    h3_layout_spec spec={s[0],s[1],s[2],s[3],s[4],s[5],NULL,0,refs,nr};
    h3_layout l; if (!h3_layout_build(&spec,&l,error,sizeof(error))) die("layout");
    double *positions=read_tensor(&f,"x.positions",H3_DTYPE_F64,l.seq_len*3);
    for (size_t i=0;i<l.seq_len;i++)
        if (fabs(l.positions[i].t-positions[i*3])>1e-12 || fabs(l.positions[i].h-positions[i*3+1])>1e-12 ||
            fabs(l.positions[i].w-positions[i*3+2])>1e-12) die("oracle layout differs");
    size_t vc=(size_t)24*s[1]*s[2]*s[3], ac=(size_t)32*2*s[4];
    float *video=read_tensor(&f,"x.video",H3_DTYPE_F32,vc), *audio=read_tensor(&f,"x.audio",H3_DTYPE_F32,ac);
    float *cv=read_tensor(&f,"x.condition_video",H3_DTYPE_F32,l.img_cond_rows*96);
    float *ca=read_tensor(&f,"x.condition_audio",H3_DTYPE_F32,l.audio_cond_rows*32);
    uint16_t *tv=read_tensor(&f,"x.text",H3_DTYPE_BF16,(size_t)s[0]*5120);
    int64_t *tags=read_tensor(&f,"x.tags",H3_DTYPE_I64,l.seq_len);
    uint8_t *text_tags=malloc((size_t)s[0]); if (!text_tags) die("allocation");
    for (int i=0;i<s[0];i++) text_tags[i]=(uint8_t)tags[i];
    h3_text_embedding text={(size_t)s[0],5120,tv,{0},text_tags,NULL};
    h3_sigma_schedule sigmas; int steps=atoi(argv[4]);
    if (!h3_serving_schedule_build(steps,&sigmas)) die("schedule");
    float *sv=read_tensor(&f,"x.sigmas_video",H3_DTYPE_F32,(size_t)steps+1);
    float *sa=read_tensor(&f,"x.sigmas_audio",H3_DTYPE_F32,(size_t)steps+1);
    for (int i=0;i<=steps;i++) if (fabsf(sv[i]-sigmas.video[i])>2e-7f || fabsf(sa[i]-sigmas.audio[i])>2e-7f) die("oracle schedule differs");
    h3_dit *dit=h3_dit_load_conditioned(argv[1],"src/metal/shaders.metal",&text,&l,&sigmas,
        50,1,0,0,1.0f, 1,1,1,1,1,1,1,1,0,0,0,
        cv,l.img_cond_rows*96,ca,l.audio_cond_rows*32,1,progress,NULL,error,sizeof(error));
    if (!dit) die("load");
    float *pv=malloc(vc*4),*pa=malloc(ac*4); if (!pv || !pa) die("allocation");
    if (!h3_dit_forward(dit,0,video,audio,pv,pa,error,sizeof(error))) die("forward");
    save(argv[3],"video_velocity",pv,vc);save(argv[3],"audio_velocity",pa,ac);
    if (atoi(argv[5])) {
        if (!h3_dit_reset_run(dit,cv,l.img_cond_rows*96,ca,l.audio_cond_rows*32,error,sizeof(error)) ||
            !h3_dit_denoise_euler(dit,video,audio,1,progress,NULL,error,sizeof(error))) die("denoise");
        save(argv[3],"video_final",video,vc);save(argv[3],"audio_final",audio,ac);
    }
    h3_dit_free(dit);h3_layout_free(&l);h3_st_free_header(&f);
    free(s);free(geometry);free(refs);free(positions);free(video);free(audio);free(cv);free(ca);free(tv);free(tags);free(text_tags);free(sv);free(sa);free(pv);free(pa);
    return 0;
}
