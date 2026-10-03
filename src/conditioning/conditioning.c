#include "src/sglang/sglang.h"
#include "src/conditioning/conditioning.h"
#include "src/denoise/attention.h"
#include "src/sampling/av_state.h"
#include "src/digest.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { HEADER=128, ENTRY=96, MAX_RECORDS=80, U8=1, BF16=2, U32=3, U64=4, F32=5 };
#define MAX_BYTES (UINT64_C(8)<<30)
static const unsigned char magic[8]={'H','3','C','O','N','D',0,0};
static _Thread_local const h3_conditioning *active;
const h3_conditioning *h3_conditioning_current(void){return active;}
const h3_conditioning *h3_conditioning_exchange(const h3_conditioning *s){const h3_conditioning *p=active;active=s;return p;}
static int fail(char *e,size_t n,const char *m){if(e&&n)snprintf(e,n,"h3cond: %s",m);return 0;}
static void put(unsigned char *p,uint64_t v,unsigned n){for(unsigned i=0;i<n;i++)p[i]=(unsigned char)(v>>(8*i));}
static uint64_t get(const unsigned char *p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;i++)v|=(uint64_t)p[i]<<(8*i);return v;}
static void hex(FILE *f,const char *name,const uint8_t digest[32]){
    fprintf(f,"%s=",name);for(int i=0;i<32;i++)fprintf(f,"%02x",digest[i]);fputc('\n',f);
}
static int source(FILE *f,const char *name,const char *path){
    uint8_t d[32]={0};uint64_t bytes=0;
    if(path&&!h3_sampler_source_fingerprint(path,d,&bytes))return 0;
    fprintf(f,"%s.present=%d\n%s.bytes=%llu\n",name,path!=NULL,name,(unsigned long long)bytes);
    hex(f,name,d);return 1;
}
int h3_conditioning_image_geometry(const h3_params *p) {
    if(!p || !p->references || (p->reference_image_size!=H3_REFERENCE_IMAGE_HIGH &&
        (p->reference_image_size!=H3_REFERENCE_IMAGE_MAX || p->_arithmetic_recipe)))return 0;
    for(size_t i=0;i<p->reference_count;i++)if(p->references[i].kind==H3_REFERENCE_IMAGE)return 2;
    return 0;
}
char *h3_conditioning_identity(const char *root,const char *transformer,const char *prompt,
    const h3_params *p,const uint8_t av[32],const h3_device_info *device,char *e,size_t n){
    if(!root||!prompt||!p||!av||!device){fail(e,n,"missing conditioning identity inputs");return NULL;}
    char *result=NULL;size_t bytes=0;FILE *f=open_memstream(&result,&bytes);
    if(!f){fail(e,n,"cannot allocate identity");return NULL;}
    int mode=p->reference_count!=0;uint8_t d[32];
    fprintf(f,"format=2\nimplementation=1\ncompatibility=2\nmode=%s\nwidth=%d\nheight=%d\nrender-width=%d\nrender-height=%d\nframes=%d\naudio-rate=32000\naudio-channels=2\naudio-latent-fps=40\nimage-size=%d\n",
        mode?"Ref2VA":"FL2VA",p->width,p->height,p->render_width?p->render_width:p->width,
        p->render_height?p->render_height:p->height,h3_align_frame_count(p->frames),
        p->reference_image_size);
    if(h3_conditioning_image_geometry(p))fprintf(f,"reference-image-geometry=%d\n",h3_conditioning_image_geometry(p));
    /* First anchors used stretch before this policy; reject those old caches.
     * Last-only conditioning already used cover and keeps its identity. */
    if(p->first_frame)fprintf(f,"first-frame-fit=cover-v1\n");
    h3_sampler_hash(prompt,strlen(prompt),d);hex(f,"prompt-sha256",d);
    h3_sampler_hash("",0,d);hex(f,"negative-prompt-sha256",d);hex(f,"av-signature",av);
    if(!h3_sampler_model_metadata_effective(root,transformer,mode,d,e,n))goto failed;
    hex(f,"model-signature",d);
    const char *components[]={"transformer","text_encoder","tokenizer","processor","video_vae","audio_vae"};
    for(size_t i=0;i<sizeof(components)/sizeof(*components);i++){
        if(!h3_sampler_component_metadata(root,transformer,mode,components[i],d,e,n))goto failed;
        hex(f,components[i],d);
    }
    if(p->_arithmetic_recipe)fprintf(f,"sglang-reference=%d\n",H3_SGLANG_VERSION);
    /* Conservatively retain all arithmetic environment settings using the
     * checkpoint filter. Diagnostics/test controls are intentionally excluded. */
    char *env=h3_sampler_environment();if(!env)goto failed;
    h3_sampler_hash(env,strlen(env),d);hex(f,"arithmetic-environment",d);free(env);
    fprintf(f,"conditioning-platform=%s\nconditioning-device=%s\nconditioning-architecture=%s\nconditioning-family=%d\n",
        device->backend,device->name,device->architecture,device->apple_gpu_family);
    if(getenv("H3_SHADER_PATH")&&!source(f,"shader-override",getenv("H3_SHADER_PATH")))goto failed;
    if(!source(f,"first-frame",p->first_frame)||!source(f,"last-frame",p->last_frame))goto failed;
    fprintf(f,"reference-count=%zu\n",p->reference_count);
    for(size_t i=0;i<p->reference_count;i++){
        const h3_reference *r=p->references+i;char name[80];
        fprintf(f,"reference.%zu.kind=%d\nreference.%zu.embedded-audio=%d\n",i,r->kind,i,r->include_embedded_audio);
        snprintf(name,sizeof(name),"reference.%zu.media",i);if(!source(f,name,r->path))goto failed;
        snprintf(name,sizeof(name),"reference.%zu.audio",i);if(!source(f,name,r->audio_path))goto failed;
    }
    if(fclose(f)){free(result);fail(e,n,"cannot finalize identity");return NULL;}
    return result;
failed:
    fclose(f);free(result);if(e&&n&&!*e)fail(e,n,"cannot hash conditioning sources");return NULL;
}
int h3_conditioning_identity_matches(const char *a,const char *b,char *e,size_t n){
    if(!a||!b)return fail(e,n,"missing identity");
    while(*a&&*b){
        size_t x=strcspn(a,"\n"),y=strcspn(b,"\n");
        if(x!=y||memcmp(a,b,x)){
            size_t label=strcspn(b,"=\n");
            if(e&&n)snprintf(e,n,"h3cond: %.*s mismatch; regenerate the cache with --save-conditioning for these inputs",(int)(label>100?100:label),b);
            return 0;
        }
        a+=x+(a[x]=='\n');b+=y+(b[y]=='\n');
    }
    return (*a||*b)?fail(e,n,"identity fields differ; regenerate conditioning cache"):1;
}
void h3_conditioning_schedule_key(const h3_sigma_schedule *s,const h3_layout *l,int released,uint8_t key[32]){
    /* Encode scalars explicitly. Include the full schedule, prefix, bridge,
     * reference topology and arithmetic recipe, never a step-count-only key. */
    unsigned char b[32+8*(H3_MAX_STEPS+1)+sizeof(h3_bridge_profile)]={0};
    put(b,1,4);put(b+4,(uint32_t)s->steps,4);put(b+8,(uint32_t)released,4);
    put(b+12,l->img_cond_rows!=0,4);put(b+16,l->audio_cond_rows!=0,4);
    put(b+20,(uint32_t)l->prefix.video_prefix_t,4);put(b+24,(uint32_t)l->prefix.audio_prefix_t,4);
    put(b+28,l->bridge!=NULL,4);size_t used=32;
    for(int i=0;i<=s->steps;i++){uint32_t v,a;memcpy(&v,s->video+i,4);memcpy(&a,s->audio+i,4);put(b+used,v,4);put(b+used+4,a,4);used+=8;}
    if(l->bridge){
        const h3_bridge_profile *p=l->bridge;
        const int fields[]={p->prefix.video_prefix_t,p->prefix.audio_prefix_t,p->context_frames,
            p->video_bridge_t,p->video_exact_t,p->bridge_frames,p->audio_bridge_t,p->audio_exact_t,p->type,p->class_count};
        for(size_t i=0;i<sizeof(fields)/sizeof(*fields);i++){put(b+used,(uint32_t)fields[i],4);used+=4;}
        uint32_t bits;memcpy(&bits,&p->max_strength,4);put(b+used,bits,4);used+=4;
        memcpy(b+used,p->active,sizeof(p->active));used+=sizeof(p->active);
        for(size_t i=0;i<H3_TARGET_ROW_CLASSES;i++){memcpy(&bits,p->class_mask+i,4);put(b+used,bits,4);used+=4;}
        memcpy(b+used,p->video_classes,sizeof(p->video_classes));used+=sizeof(p->video_classes);
        memcpy(b+used,p->audio_classes,sizeof(p->audio_classes));used+=sizeof(p->audio_classes);
    }
    h3_sampler_hash(b,used,key);
}

int h3_conditioning_request_matches(const h3_conditioning *s,const h3_params *p,char *e,size_t n){
    if(!s||!p||s->reference_count!=p->reference_count||
       (s->reference_count&&(!s->references||!p->references)))
        return fail(e,n,"reference descriptor count differs from request");
    size_t anchors=(p->first_frame!=NULL)+(p->last_frame!=NULL);
    if(s->keyframe_count!=anchors||(s->reference_count&&anchors))
        return fail(e,n,"keyframe descriptors differ from request");
    size_t anchor=0;
    if(p->first_frame&&s->keyframes[anchor++]!=0)return fail(e,n,"first-frame index mismatch");
    if(p->last_frame&&s->keyframes[anchor++]!=h3_align_frame_count(p->frames)-1)return fail(e,n,"last-frame index mismatch");
    int w,h;h3_latent_canvas(p->render_width?p->render_width:p->width,p->render_height?p->render_height:p->height,&w,&h);
    uint64_t video=(uint64_t)anchors*(unsigned)(w/2)*(unsigned)(h/2)*96,audio=0;
    for(size_t i=0;i<s->reference_count;i++){
        const h3_layout_ref *r=s->references+i;
        h3_layout_ref_kind kind=p->references[i].kind==H3_REFERENCE_IMAGE?H3_LAYOUT_REF_IMAGE:
            p->references[i].kind==H3_REFERENCE_AUDIO?H3_LAYOUT_REF_AUDIO:H3_LAYOUT_REF_VIDEO;
        if(r->kind!=kind||(kind==H3_LAYOUT_REF_IMAGE&&r->latent_t!=1))return fail(e,n,"reference kind or image time dimension mismatch");
        if(p->adaptive_cache||p->cuda_attention==H3_ATTENTION_SUBBLOCK){
            int has_audio=p->references[i].kind==H3_REFERENCE_AUDIO||p->references[i].kind==H3_REFERENCE_VIDEO_AUDIO||
                (p->references[i].kind==H3_REFERENCE_VIDEO&&p->references[i].include_embedded_audio);
            if(has_audio!=(r->audio_t>0))return fail(e,n,"reference soundtrack presence differs from request");
        }
        if(kind!=H3_LAYOUT_REF_AUDIO)video+=(uint64_t)r->latent_t*(unsigned)(r->latent_h/2)*(unsigned)(r->latent_w/2)*96;
        audio+=(uint64_t)r->audio_t*2*32;
    }
    if(video!=s->video_elements||audio!=s->audio_elements||s->conditioned!=(video||audio))
        return fail(e,n,"raw conditioning sizes disagree with reference/keyframe geometry");
    return 1;
}

typedef struct {uint32_t id,type,rank;uint64_t shape[3],bytes;const void *data;} record;
static unsigned item_size(unsigned type){return type==U8?1:type==BF16?2:(type==U32||type==F32)?4:type==U64?8:0;}
static int add(record *r,size_t *count,unsigned id,unsigned type,unsigned rank,uint64_t a,uint64_t b,uint64_t c,const void *data){
    if(!a)return 1;
    unsigned unit=item_size(type);uint64_t bytes=unit;
    uint64_t dims[3]={a,b,c};if(*count==MAX_RECORDS||!data||!unit||rank<1||rank>3)return 0;
    for(unsigned i=0;i<rank;i++){if(!dims[i]||dims[i]>MAX_BYTES/bytes)return 0;bytes*=dims[i];}
    r[*count]=(record){id,type,rank,{a,b,c},bytes,data};(*count)++;return 1;
}
int h3_conditioning_save(const h3_conditioning *s,const char *path,char *e,size_t n){
    uint32_t endian=1;if(*(uint8_t *)&endian!=1)return fail(e,n,"little-endian host required");
    if(!s||!s->identity||strlen(s->identity)>(1u<<20)||!s->text.tokens||s->text.width!=H3_TEXT_HIDDEN_SIZE||!path||!*path||
        s->reference_count>4096||(s->reference_count&&!s->references)||s->prepared.count>H3_PREPARED_MAX||
        s->keyframe_count>2||(s->conditioned!=0&&s->conditioned!=1))
        return fail(e,n,"invalid conditioning payload");
    unsigned char prepared_seen[150]={0};
    for(size_t i=0;i<s->prepared.count;i++){
        const h3_prepared_tensor *t=s->prepared.tensors+i;
        if((t->id!=1&&t->id!=2&&(t->id<100||t->id>=150))||
           prepared_seen[t->id]||!t->values||!t->elements||
           (t->id==1&&t->elements!=s->text.tokens*5376))
            return fail(e,n,"invalid or duplicate prepared conditioning tensor");
        prepared_seen[t->id]=1;
    }
    if(s->has_schedule){
        if(!prepared_seen[1]||!prepared_seen[2])return fail(e,n,"incomplete schedule conditioning tensors");
        for(unsigned id=100;id<150;id++)if(!prepared_seen[id])return fail(e,n,"schedule conditioning caches require all 50 blocks");
    }
    record r[MAX_RECORDS]={0};size_t count=0;uint32_t flags[4]={(uint32_t)s->conditioned,(uint32_t)s->keyframe_count,(uint32_t)s->keyframes[0],(uint32_t)s->keyframes[1]};
    uint32_t *refs=calloc(s->reference_count?s->reference_count*5:1,4);
    if(!refs)return fail(e,n,"cannot allocate reference descriptors");
    for(size_t i=0;i<s->reference_count;i++){const h3_layout_ref *v=s->references+i;uint32_t row[5]={(uint32_t)v->kind,(uint32_t)v->latent_t,(uint32_t)v->latent_h,(uint32_t)v->latent_w,(uint32_t)v->audio_t};memcpy(refs+i*5,row,20);}
#define ADD(id,type,rank,a,b,c,data) do{if(!add(r,&count,id,type,rank,a,b,c,data))goto bad;}while(0)
    ADD(1,U8,1,strlen(s->identity),0,0,s->identity);
    ADD(2,BF16,2,s->text.tokens,s->text.width,0,s->text.values);
    if(s->text.tags)ADD(3,U8,1,s->text.tokens,0,0,s->text.tags);
    ADD(4,F32,1,s->video_elements,0,0,s->video);ADD(5,F32,1,s->audio_elements,0,0,s->audio);
    ADD(6,U32,2,s->reference_count,5,0,refs);ADD(7,U32,1,4,0,0,flags);
    if(s->text.diagnostics){const h3_text_diagnostics *d=s->text.diagnostics;
        ADD(8,U32,1,d->tokens,0,0,d->ids);ADD(9,U32,2,3,d->tokens,0,d->positions);
        ADD(10,U64,2,d->span_count,2,0,d->spans);
    }
    if(s->has_schedule)ADD(11,U8,1,32,0,0,s->schedule_key);
    for(size_t i=0;i<s->prepared.count;i++){
        const h3_prepared_tensor *t=s->prepared.tensors+i;
        if(t->id!=1&&!s->has_schedule)continue;
        ADD(1000+t->id,BF16,1,t->elements,0,0,t->values);
    }
#undef ADD
    unsigned char header[HEADER]={0},table[MAX_RECORDS*ENTRY]={0};
    uint64_t offset=HEADER+count*ENTRY;
    for(size_t i=0;i<count;i++){
        unsigned char *t=table+i*ENTRY;put(t,r[i].id,4);put(t+4,r[i].type,4);put(t+8,r[i].rank,4);put(t+12,1,4);
        for(unsigned j=0;j<3;j++)put(t+16+8*j,r[i].shape[j],8);
        put(t+40,offset,8);put(t+48,r[i].bytes,8);h3_sampler_hash(r[i].data,(size_t)r[i].bytes,t+56);
        if (r[i].bytes > MAX_BYTES - offset)
            goto bad;
        offset += r[i].bytes;
    }
    memcpy(header,magic,8);put(header+8,2,4);put(header+12,HEADER,4);put(header+16,offset,8);put(header+24,count,4);put(header+28,ENTRY,4);
    h3_sampler_hash(table,count*ENTRY,header+32);h3_sampler_hash(header,64,header+64);
    char *tmp=NULL;if(asprintf(&tmp,"%s.XXXXXX",path)<0)goto bad;
    int fd=mkstemp(tmp);FILE *f=fd>=0?fdopen(fd,"wb"):NULL;int ok=f!=NULL;
    if(!f&&fd>=0)close(fd);
    if(f){ok=fwrite(header,1,HEADER,f)==HEADER&&fwrite(table,ENTRY,count,f)==count;
        for(size_t i=0;ok&&i<count;i++)ok=fwrite(r[i].data,1,(size_t)r[i].bytes,f)==r[i].bytes;
        if (fflush(f) || fsync(fd))
            ok = 0;
        if (fclose(f))
            ok = 0;
    }
    if (ok)
        ok = rename(tmp, path) == 0;
    unlink(tmp);
    free(tmp);
    free(refs);
    return ok?1:fail(e,n,"cannot atomically write conditioning file");
bad:free(refs);return fail(e,n,"invalid or oversized conditioning record");
}

void h3_conditioning_free(h3_conditioning *s){
    if (!s)
        return;
    free(s->identity);
    h3_text_embedding_free(&s->text);
    free(s->video);
    free(s->audio);
    free(s->references);
    h3_prepared_cache_free(&s->prepared);
    free(s);
}
static void *copy(const void *p,size_t bytes){void *v=malloc(bytes+1);if(v){memcpy(v,p,bytes);((unsigned char *)v)[bytes]=0;}return v;}
h3_conditioning *h3_conditioning_load(const char *path,char *e,size_t n){
    if(!path||!*path){fail(e,n,"conditioning path is empty");return NULL;}
    uint32_t endian=1;if(*(uint8_t *)&endian!=1){fail(e,n,"little-endian host required");return NULL;}
    int fd=open(path,O_RDONLY);struct stat st;
    if(fd<0){fail(e,n,"cannot open conditioning file");return NULL;}
    if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<HEADER||(uint64_t)st.st_size>MAX_BYTES){close(fd);fail(e,n,"invalid conditioning file size/type");return NULL;}
    size_t bytes=(size_t)st.st_size;unsigned char *m=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(m==MAP_FAILED){fail(e,n,"cannot map conditioning file");return NULL;}
    h3_conditioning *s=NULL;uint8_t hash[32];size_t count=(size_t)get(m+24,4);
    h3_sampler_hash(m,64,hash);
    if(memcmp(m,magic,8)||get(m+8,4)!=2||get(m+12,4)!=HEADER||get(m+16,8)!=bytes||get(m+28,4)!=ENTRY||
        !count||count>MAX_RECORDS||HEADER+count*ENTRY>bytes||memcmp(hash,m+64,32))goto bad;
    for(size_t i=96;i<HEADER;i++)if(m[i])goto bad;
    h3_sampler_hash(m+HEADER,count*ENTRY,hash);if(memcmp(hash,m+32,32))goto bad;
    uint64_t offset=HEADER+count*ENTRY;unsigned seen[1200]={0};record r[MAX_RECORDS]={0};
    for(size_t i=0;i<count;i++){
        const unsigned char *t=m+HEADER+i*ENTRY;unsigned id=(unsigned)get(t,4),type=(unsigned)get(t+4,4),rank=(unsigned)get(t+8,4);
        unsigned unit=item_size(type);uint64_t length=unit;
        if(id>=1200||!id||seen[id]++||!unit||rank<1||rank>3||get(t+12,4)!=1||get(t+40,8)!=offset)goto bad;
        if(!(id<=11||id==1001||id==1002||(id>=1100&&id<1150)))goto bad;
        for(unsigned j=0;j<3;j++){uint64_t d=get(t+16+8*j,8);r[i].shape[j]=d;if(j<rank){if(!d||d>MAX_BYTES/length)goto bad;length*=d;}else if(d)goto bad;}
        if(length!=get(t+48,8)||offset>bytes||length>bytes-offset)goto bad;
        for(unsigned j=88;j<ENTRY;j++)if(t[j])goto bad;
        h3_sampler_hash(m+offset,(size_t)length,hash);if(memcmp(hash,t+56,32))goto bad;
        r[i].id=id;r[i].type=type;r[i].rank=rank;r[i].bytes=length;r[i].data=m+offset;offset+=length;
    }
    if(offset!=bytes||!seen[1]||!seen[2]||!seen[7]||seen[8]!=seen[9])goto bad;
    s=calloc(1,sizeof(*s));if(!s)goto bad;
    for(size_t i=0;i<count;i++){
        record *v=r+i;size_t a=(size_t)v->shape[0],b=(size_t)v->shape[1];
        #define U(j) ((uint32_t)get((const unsigned char *)v->data+4*(j),4))
        switch(v->id){
        case 1:if(v->type!=U8||v->rank!=1||a>(1u<<20)||memchr(v->data,0,a))goto bad;s->identity=copy(v->data,a);if(!s->identity)goto bad;break;
        case 2:if(v->type!=BF16||v->rank!=2||b!=H3_TEXT_HIDDEN_SIZE||a>1048576)goto bad;s->text.tokens=a;s->text.width=b;s->text.values=copy(v->data,(size_t)v->bytes);if(!s->text.values)goto bad;break;
        case 3:if(v->type!=U8||v->rank!=1)goto bad;s->text.tags=copy(v->data,a);if(!s->text.tags)goto bad;break;
        case 4:case 5:{if(v->type!=F32||v->rank!=1)goto bad;float *value=copy(v->data,(size_t)v->bytes);if(!value)goto bad;for(size_t j=0;j<a;j++)if(!isfinite(value[j])){free(value);goto bad;}if(v->id==4){s->video=value;s->video_elements=a;}else{s->audio=value;s->audio_elements=a;}break;}
        case 6:if(v->type!=U32||v->rank!=2||b!=5||a>4096)goto bad;
            s->references=calloc(a,sizeof(*s->references));if(!s->references)goto bad;s->reference_count=a;
            for(size_t j=0;j<a;j++){for(unsigned k=0;k<5;k++)if(U(j*5+k)>INT32_MAX)goto bad;
                s->references[j]=(h3_layout_ref){(h3_layout_ref_kind)U(j*5),(int)U(j*5+1),(int)U(j*5+2),(int)U(j*5+3),(int)U(j*5+4)};
                const h3_layout_ref *ref=s->references+j;
                if(ref->kind<H3_LAYOUT_REF_IMAGE||ref->kind>H3_LAYOUT_REF_VIDEO||ref->audio_t>1000000||
                    (ref->kind!=H3_LAYOUT_REF_AUDIO&&(ref->latent_t<1||ref->latent_t>1000000||ref->latent_h<2||ref->latent_w<2||ref->latent_h%2||ref->latent_w%2||ref->latent_h>16384||ref->latent_w>16384))||
                    (ref->kind==H3_LAYOUT_REF_AUDIO&&(!ref->audio_t||ref->latent_t||ref->latent_h||ref->latent_w)))goto bad;
            }break;
        case 7:if(v->type!=U32||v->rank!=1||a!=4||U(0)>1||U(1)>2||U(2)>INT32_MAX||U(3)>INT32_MAX)goto bad;
            s->conditioned=(int)U(0);s->keyframe_count=U(1);s->keyframes[0]=(int)U(2);s->keyframes[1]=(int)U(3);break;
        case 8:case 9:case 10:
            if (!s->text.diagnostics)
                s->text.diagnostics = calloc(1, sizeof(*s->text.diagnostics));
            if (!s->text.diagnostics)
                goto bad;
            if(v->id==8){if(v->type!=U32||v->rank!=1)goto bad;s->text.diagnostics->tokens=a;s->text.diagnostics->ids=copy(v->data,(size_t)v->bytes);if(!s->text.diagnostics->ids)goto bad;}
            else if(v->id==9){if(v->type!=U32||v->rank!=2||a!=3)goto bad;s->text.diagnostics->positions=copy(v->data,(size_t)v->bytes);if(!s->text.diagnostics->positions)goto bad;}
            else{if(v->type!=U64||v->rank!=2||b!=2)goto bad;s->text.diagnostics->span_count=a;s->text.diagnostics->spans=copy(v->data,(size_t)v->bytes);if(!s->text.diagnostics->spans)goto bad;}break;
        case 11:if(v->type!=U8||v->rank!=1||a!=32)goto bad;s->has_schedule=1;memcpy(s->schedule_key,v->data,32);break;
        default:{if(v->type!=BF16||v->rank!=1||s->prepared.count>=H3_PREPARED_MAX)goto bad;
            h3_prepared_tensor *p=&s->prepared.tensors[s->prepared.count++];p->id=v->id-1000;p->elements=a;p->values=copy(v->data,(size_t)v->bytes);if(!p->values)goto bad;s->prepared.version=1;break;}
        }
    }
    #undef U
    for(size_t i=0;i<count;i++){
        if(r[i].id==3){if(r[i].shape[0]!=s->text.tokens)goto bad;for(size_t j=0;j<s->text.tokens;j++)if(s->text.tags[j]>1)goto bad;}
        if(r[i].id==8&&r[i].shape[0]!=s->text.tokens)goto bad;
        if(r[i].id==9&&r[i].shape[1]!=s->text.tokens)goto bad;
    }
    if(seen[10]&&!seen[8])goto bad;
    if(s->text.diagnostics)for(size_t i=0;i<s->text.diagnostics->span_count;i++){
        const uint64_t *span=s->text.diagnostics->spans+2*i;
        if(span[0]>s->text.tokens||span[1]>s->text.tokens-span[0])goto bad;
    }
    if(seen[1001]&&!h3_prepared_find(&s->prepared,1,s->text.tokens*5376))goto bad;
    if(s->has_schedule){
        if(!seen[1001]||!seen[1002])goto bad;
        for(unsigned id=1100;id<1150;id++)if(!seen[id])goto bad;
    }
    for(size_t i=0;i<s->text.tokens*s->text.width;i++)if((s->text.values[i]&0x7f80)==0x7f80)goto bad;
    for(size_t i=0;i<s->prepared.count;i++){h3_prepared_tensor *p=s->prepared.tensors+i;
        if(p->id!=1&&!s->has_schedule)goto bad;
        for(size_t j=0;j<p->elements;j++)if((p->values[j]&0x7f80)==0x7f80)goto bad;
    }
    munmap(m,bytes);return s;
bad:munmap(m,bytes);h3_conditioning_free(s);fail(e,n,"invalid version, shape, record or checksum in conditioning file");return NULL;
}
