#include "src/server/sglang.h"
#include "src/h3.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
static int fail(char *e,size_t n,const char *field,const char *why){snprintf(e,n,"%s: %s",field,why);return 0;}
static int in(const char *name,const char *list){
    size_t n=strlen(name);for(const char *p=list;*p;){const char *q=strchr(p,'|');size_t len=q?(size_t)(q-p):strlen(p);
        if (n == len && !memcmp(name, p, n))
            return 1;
        p = q ? q + 1 : p + len;
    }
    return 0;
}
static int equal(const sj_value *a,const sj_value *b){sj_value *ac=sj_clone(a),*bc=sj_clone(b);sj_sort(ac);sj_sort(bc);char *x=sj_dump(ac),*y=sj_dump(bc);sj_free(ac);sj_free(bc);int ok=x&&y&&!strcmp(x,y);free(x);free(y);return ok;}
static int flatten(sj_value *out,const sj_value *input,int depth,char *error,size_t size){
    if(!input||input->type!=LJ_OBJECT||depth>2)return fail(error,size,"body","expected bounded object wrapper");
    for(size_t i=0;i<input->count;i++){
        const sj_value *v=input->items[i];
        if(in(v->key,"extra_body|extra_json|extra_params")){
            sj_value *parsed=NULL;const sj_value *child=v;
            if(v->type==LJ_STRING){parsed=sj_parse(v->text,strlen(v->text),error,size);child=parsed;}
            int ok=flatten(out,child,depth+1,error,size);sj_free(parsed);if(!ok)return 0;
        }else{
            const sj_value *old=sj_get(out,v->key);
            if(old){if(!equal(old,v))return fail(error,size,v->key,"conflicting transport aliases");}
            else if(!sj_add(out,v->key,sj_clone(v)))return fail(error,size,"body","out of memory");
        }
    }return 1;
}
static const char *strings="prompt|model|task|quality|input_reference|reference_url|video_path|video_url|task_type|size|generator_device|negative_prompt|output_path|output_mode|output_quality|frame_interpolation_model_path|upscaling_model_path|perf_dump_path|attention_backend_override";
static const char *ints="n|num_outputs_per_prompt|width|height|num_frames|fps|num_inference_steps|max_sequence_length|frame_interpolation_exp|upscaling_scale|num_profiled_timesteps";
static const char *numbers="seconds|guidance_scale|guidance_scale_2|true_cfg_scale|flow_shift|audio_flow_shift|audio_guidance_scale|frame_interpolation_scale|output_compression|cfg_gate_step|imgvid_cond_noise_aug_for_inference|audio_cond_noise_aug_for_inference";
static const char *bools="enhance_prompt|enable_teacache|enable_cache_dit|use_diffusion_decoder|enable_frame_interpolation|enable_upscaling|profile|profile_all_stages";
static int typecheck(const sj_value *b,char *e,size_t n){
    for(size_t i=0;i<b->count;i++){
        const sj_value *v=b->items[i];const char *k=v->key;int valid=1;
        if(!strcmp(k,"h3cli")){if(v->type!=LJ_STRING)return fail(e,n,k,"must be a string");continue;}
        if(!strcmp(k,"seed")){
            if(v->type==LJ_NULL)continue;
            if(v->type!=LJ_NUMBER&&v->type!=LJ_ARRAY)return fail(e,n,k,"expected integer or integer list");
            if(v->type==LJ_ARRAY){for(size_t j=0;j<v->count;j++){uint64_t x;if(!sj_u64(v->items[j],&x))return fail(e,n,k,"expected uint64 list");}}
            else {uint64_t x;if(!sj_u64(v,&x))return fail(e,n,k,"expected uint64");}continue;
        }
        if(in(k,strings))valid=v->type==LJ_STRING;
        else if(in(k,ints)){int64_t x;valid=sj_i64(v,&x);}
        else if(in(k,numbers))valid=v->type==LJ_NUMBER&&isfinite(v->number);
        else if(in(k,bools))valid=v->type==LJ_BOOL;
        else if(!strcmp(k,"conditions"))valid=v->type==LJ_ARRAY;
        else if(in(k,"target|diffusers_kwargs|cache_dit_params|skip_softmax_params"))valid=v->type==LJ_OBJECT;
        else return fail(e,n,k,"unknown request field; use h3cli for native flags");
        if(!valid&&v->type!=LJ_NULL)return fail(e,n,k,"invalid JSON type");
    }return 1;
}
static int inactive(const sj_value *v){return !v||v->type==LJ_NULL||(v->type==LJ_BOOL&&v->number==0);}
static int unsupported(const sj_value *b,char *e,size_t n){
    const char *never[]={"guidance_scale","guidance_scale_2","true_cfg_scale","negative_prompt","audio_guidance_scale",
        "cache_dit_params","skip_softmax_params","cfg_gate_step","attention_backend_override","diffusers_kwargs","max_sequence_length",
        "frame_interpolation_model_path","upscaling_model_path","perf_dump_path","num_profiled_timesteps","output_compression",
        "imgvid_cond_noise_aug_for_inference","audio_cond_noise_aug_for_inference","fps","num_frames","video_path","video_url","task_type"};
    for(size_t i=0;i<sizeof(never)/sizeof(*never);i++)if(!inactive(sj_get(b,never[i])))return fail(e,n,never[i],"unsupported SGLang control");
    const char *flags[]={"enhance_prompt","enable_teacache","enable_cache_dit","use_diffusion_decoder","enable_frame_interpolation","enable_upscaling","profile_all_stages"};
    for(size_t i=0;i<sizeof(flags)/sizeof(*flags);i++)if(!inactive(sj_get(b,flags[i])))return fail(e,n,flags[i],"unsupported SGLang control");
    const char *keys[]={"flow_shift","audio_flow_shift","frame_interpolation_exp","frame_interpolation_scale","upscaling_scale"};
    const double fixed[]={12,3,1,1,4};
    for(size_t i=0;i<5;i++){const sj_value *v=sj_get(b,keys[i]);if(v&&v->type!=LJ_NULL&&v->number!=fixed[i])return fail(e,n,keys[i],"unsupported non-default value");}
    const char *mode=sj_field(b,"output_mode"),*rng=sj_field(b,"generator_device");
    if(mode&&strcmp(mode,"decoded_files"))return fail(e,n,"output_mode","only decoded_files is supported");
    if(rng&&strcmp(rng,"cuda"))return fail(e,n,"generator_device","RNG device override is unsupported");
    return 1;
}
static int add(h3_request *r,const char *k,const char *v,char *e,size_t n){return h3_request_add(r,k,v,NULL,H3_SOURCE_SGLANG,e,n);}
static int integer(h3_request *r,const char *key,int64_t v,char *e,size_t n){char b[32];snprintf(b,sizeof(b),"%"PRId64,v);return add(r,key,b,e,n);}
static int has_group(const h3_request *r,const char *group){for(size_t i=0;i<r->count;i++)if(!strcmp(r->values[i].option->group,group))return 1;return 0;}
static int owned_by(const h3_request *native,const char *name){return h3_request_get(native,name)!=NULL;}
static int conditions(h3_submission *s,const h3_request *native,int operation_changed,char *e,size_t n){
    const sj_value *list=sj_get(s->transport,"conditions");if(list&&list->type==LJ_NULL)list=NULL;
    int refs=has_group(native,"references"),anchors=owned_by(native,"first-frame")||owned_by(native,"last-frame");
    unsigned seen=0;size_t count=0;
    if(list)for(size_t i=0;i<list->count;i++){
        const sj_value *c=list->items[i];if(c->type!=LJ_OBJECT)return fail(e,n,"conditions","entries must be objects");
        for(size_t j=0;j<c->count;j++)if(!in(c->items[j]->key,"type|uri|role|frame_index|start_time_seconds"))return fail(e,n,"conditions","unknown condition field");
        const char *type=sj_field(c,"type"),*uri=sj_field(c,"uri"),*role=sj_field(c,"role");
        if(!type||!uri||!role)return fail(e,n,"conditions","type, uri and role must be strings");
        int keyframe=!strcmp(role,"keyframe");const sj_value *index=sj_get(c,"frame_index"),*start=sj_get(c,"start_time_seconds");int64_t frame=0;if(index&&index->type==LJ_NULL)index=NULL;
        if(index&&!sj_i64(index,&frame))return fail(e,n,"conditions.frame_index","expected integer");
        if(start&&start->type!=LJ_NULL&&start->type!=LJ_NUMBER)return fail(e,n,"conditions.start_time_seconds","expected number");
        const char *flag=NULL;
        if(keyframe){if(!index||strcmp(type,"image")||(frame!=0&&frame!=-1))return fail(e,n,"conditions","only image keyframes 0/-1 are supported");flag=frame==0?"first-frame":"last-frame";}
        else if(!strcmp(role,"reference")){
            if(!strcmp(type,"image"))flag="ref-image";else if(!strcmp(type,"video"))flag="ref-video";
            else if(!strcmp(type,"video_audio"))flag="ref-video";else if(!strcmp(type,"audio"))flag="ref-audio";
            else return fail(e,n,"conditions.type","unsupported media type");
            if(index)return fail(e,n,"conditions.frame_index","not allowed for reference media");
        }else return fail(e,n,"conditions.role","expected keyframe or reference");
        if(operation_changed||refs||(keyframe?owned_by(native,flag):anchors))continue;
        if(start&&start->type!=LJ_NULL&&start->number!=0)return fail(e,n,"conditions.start_time_seconds","reference seeking is unsupported");
        if(!*uri)return fail(e,n,"conditions.uri","empty input");
        if(keyframe){unsigned bit=frame==0?1u:2u;if(seen&bit||(frame==0&&(seen&2u)))return fail(e,n,"conditions","duplicate or misordered keyframes");seen|=bit;}
        if(!add(&s->request,flag,uri,e,n))return 0;
        /* Keep the stronger video_audio requirement per condition, even in mixed lists. */
        if(!strcmp(type,"video_audio")){s->needs_embedded_audio=1;s->required_audio[s->request.count-1]=1;}
        count++;
    }
    const char *generic=sj_field(s->transport,"input_reference"),*url=sj_field(s->transport,"reference_url");
    if(!operation_changed&&!refs&&!owned_by(native,"first-frame")){
        if(generic&&url&&strcmp(generic,url))return fail(e,n,"input_reference","conflicting reference aliases");
        generic=generic?generic:url;
        if(generic&&*generic){if(count)return fail(e,n,"input_reference","ambiguous with canonical conditions");if(!add(&s->request,"first-frame",generic,e,n))return 0;}
    }
    if(!operation_changed&&!refs&&!anchors&&s->canonical){
        if(!strcmp(s->task,"t2va")&&count)return fail(e,n,"conditions","t2va requires no conditions");
        if(strcmp(s->task,"t2va")&&!count&&!generic)return fail(e,n,"conditions","task requires inputs");
        if(!strcmp(s->task,"fl2va")&&count&&(!seen||count!=(size_t)((seen&1u)!=0)+(size_t)((seen&2u)!=0)))return fail(e,n,"conditions","fl2va requires keyframes");
    }return 1;
}
void h3_submission_free(h3_submission *s){h3_request_free(&s->request);sj_free(s->transport);free(s->model);memset(s,0,sizeof(*s));}
int h3_submission_parse(const sj_value *body,h3_submission *s,char *e,size_t n){
    memset(s,0,sizeof(*s));s->variants=1;s->transport=sj_object();h3_request native={0};int ok=0;
    if(!s->transport||!flatten(s->transport,body,0,e,n)||!typecheck(s->transport,e,n))goto done;
    const char *flags=sj_field(s->transport,"h3cli");
    if (flags && !h3_request_parse_string(&native, flags, e, n))
        goto done;
    for (size_t i = 0; i < native.count; i++) {
        if (!strcmp(native.values[i].option->operations, "process")) {
            fail(e, n, native.values[i].option->name, "process/startup option is not a job flag");
            goto done;
        }
    }
    s->native_flags = native.count != 0;
    if(!unsupported(s->transport,e,n))goto done;
    const char *task=sj_field(s->transport,"task");s->canonical=task!=NULL;
    const char *native_operation=h3_request_operation(&native);
    int operation_changed=strcmp(native_operation,"generate")&&strcmp(native_operation,"still");
    if(task&&!in(task,"t2va|fl2va|ref2va")&&!operation_changed){fail(e,n,"task","expected t2va, fl2va or ref2va");goto done;}
    if(!task&&!s->native_flags){fail(e,n,"task","required for SGLang-only requests");goto done;}
    snprintf(s->task,sizeof(s->task),"%s",task&&in(task,"t2va|fl2va|ref2va")?task:"t2va");
    const char *model=sj_field(s->transport,"model");s->model=strdup(model?model:"h3cli");if(!s->model)goto done;
    const char *p=sj_field(s->transport,"prompt");
    if(p&&!operation_changed&&!owned_by(&native,"prompt")&&!add(&s->request,"prompt",p,e,n))goto done;
    const char *quality=sj_field(s->transport,"quality");
    int native_quality=owned_by(&native,"quality");
    if(quality&&!operation_changed&&!native_quality){
        if(!in(quality,"lossless|extra-high|high")){fail(e,n,"quality","expected lossless, extra-high or high; native preview levels use h3cli");goto done;}
        if(!add(&s->request,"quality",quality,e,n))goto done;
    }
    const char *output_quality=sj_field(s->transport,"output_quality");
    if(output_quality&&!has_group(&native,"encoding")) {
        if(!add(&s->request,"output-quality",output_quality,e,n))goto done;
    }
    const sj_value *steps=sj_get(s->transport,"num_inference_steps");
    if(steps&&steps->type!=LJ_NULL&&!operation_changed&&!native_quality&&!owned_by(&native,"steps")){
        int64_t x;if(!sj_i64(steps,&x)||x<2||x>1001){fail(e,n,"num_inference_steps","expected 2..1001 sigma points");goto done;}
        if(!integer(&s->request,"steps",x-1,e,n))goto done;
    }
    const sj_value *count=sj_get(s->transport,"n"),*count2=sj_get(s->transport,"num_outputs_per_prompt");
    if (count && count->type == LJ_NULL)
        count = NULL;
    if (count2 && count2->type == LJ_NULL)
        count2 = NULL;
    int64_t variants=1;if(count&&!sj_i64(count,&variants))goto done;
    if(count2){int64_t v;if(!sj_i64(count2,&v))goto done;if(count&&v!=variants){fail(e,n,"n","inconsistent variant aliases");goto done;}variants=v;}
    if(variants<1||variants>10){fail(e,n,"n","must be 1..10");goto done;}s->variants=(int)variants;
    if(!conditions(s,&native,operation_changed,e,n))goto done;
    if(!operation_changed){
        const sj_value *target=sj_get(s->transport,"target");if(target&&target->type==LJ_NULL)target=NULL;
        if(target)for(size_t i=0;i<target->count;i++)if(!in(target->items[i]->key,"short_edge|aspect_ratio|duration_seconds")){fail(e,n,"target","unknown target field");goto done;}
        if(s->canonical&&!target&&!(owned_by(&native,"width")&&owned_by(&native,"height")&&has_group(&native,"timing"))){fail(e,n,"target","required when canonical geometry/timing is not overridden");goto done;}
        s->width_needed=!owned_by(&native,"width")&&target;
        s->height_needed=!owned_by(&native,"height")&&target;
        s->duration_needed=!has_group(&native,"timing")&&target&&strcmp(native_operation,"still");
        if(!target){const char *keys[]={"width","height","seconds"};for(size_t i=0;i<3;i++){
            const sj_value *v=sj_get(s->transport,keys[i]);if(v&&v->type!=LJ_NULL&&!owned_by(&native,keys[i])&&!(i==2&&has_group(&native,"timing")))if(!add(&s->request,keys[i],v->text,e,n))goto done;}}
    }
    if(!owned_by(&native,"output")){p=sj_field(s->transport,"output_path");if(p&&!add(&s->request,"output",p,e,n))goto done;}
    if(!inactive(sj_get(s->transport,"profile"))&&!owned_by(&native,"profile")&&!add(&s->request,"profile",NULL,e,n))goto done;
    for(size_t i=0;i<native.count;i++){h3_option_value *v=&native.values[i];if(!h3_request_add(&s->request,v->option->name,v->value,v->second,H3_SOURCE_NATIVE,e,n))goto done;}
    const h3_option_value *seed=h3_request_get(&native,"seed");const sj_value *seed_json=sj_get(s->transport,"seed");uint64_t base=42;
    if(seed){errno=0;base=strtoull(seed->value,NULL,10);}
    else if(seed_json&&seed_json->type==LJ_NUMBER){if(!sj_u64(seed_json,&base))goto done;}
    if(!seed&&seed_json&&seed_json->type==LJ_ARRAY){
        if(seed_json->count!=(size_t)s->variants){fail(e,n,"seed","list length must match variants");goto done;}
        for(int i=0;i<s->variants;i++)if(!sj_u64(seed_json->items[i],&s->seeds[i]))goto done;
    }else{if(base>UINT64_MAX-(uint64_t)(s->variants-1)){fail(e,n,"seed","variant seed overflow");goto done;}
        for(int i=0;i<s->variants;i++)s->seeds[i]=base+(uint64_t)i;}
    // Only explicitly supplied seeds become execution overrides; checkpoints own omitted seeds.
    if(!operation_changed&&!seed&&seed_json&&seed_json->type!=LJ_NULL){char b[32];snprintf(b,sizeof(b),"%"PRIu64,s->seeds[0]);if(!add(&s->request,"seed",b,e,n))goto done;}
    if(!strcmp(h3_request_operation(&s->request),"generate")){
        if(has_group(&s->request,"references"))strcpy(s->task,"ref2va");
        else if(owned_by(&s->request,"first-frame")||owned_by(&s->request,"last-frame"))strcpy(s->task,"fl2va");else strcpy(s->task,"t2va");
    }else if(strcmp(h3_request_operation(&s->request),"still"))s->width_needed=s->height_needed=s->duration_needed=0;
    ok=1;
done:h3_request_free(&native);if(!ok){if(!*e)fail(e,n,"request","invalid value or out of memory");h3_submission_free(s);}return ok;
}
/* Python's round(x) uses ties-to-even independently of the ambient C rounding mode. */
static double round_even(double x){double f=floor(x),d=x-f;return d>0.5?f+1:d<0.5?f:fmod(f,2)==0?f:f+1;}
int h3_submission_geometry(h3_submission *s,int iw,int ih,double audio,char *e,size_t n){
    const sj_value *target=sj_get(s->transport,"target");if(!s->width_needed&&!s->height_needed&&!s->duration_needed)return 1;
    if(s->width_needed||s->height_needed){
        const sj_value *edge=sj_get(target,"short_edge");int64_t short_edge;
        if(!sj_i64(edge,&short_edge)||short_edge<=0||short_edge>INT_MAX)return fail(e,n,"target.short_edge","expected positive integer");
        const char *aspect=sj_field(target,"aspect_ratio");if(!aspect)return fail(e,n,"target.aspect_ratio","required");
        double ratio;int a=0,b=0;char extra;
        if(!strcmp(aspect,"auto")){if(!strcmp(s->task,"fl2va")){if(iw<=0||ih<=0)return fail(e,n,"target.aspect_ratio","first effective anchor geometry unavailable");ratio=(double)iw/ih;}else ratio=16.0/9.0;}
        else {
            const char *colon=strchr(aspect,':');char *end=NULL;errno=0;
            if(!colon||colon==aspect||strlen(aspect)>31)return fail(e,n,"target.aspect_ratio","expected W:H or auto");
            long x=strtol(aspect,&end,10);if(errno||end!=colon||x<=0||x>INT_MAX)return fail(e,n,"target.aspect_ratio","invalid width component");
            errno=0;long y=strtol(colon+1,&end,10);if(errno||!* (colon+1)||*end||y<=0||y>INT_MAX)return fail(e,n,"target.aspect_ratio","invalid height component");
            a=(int)x;b=(int)y;ratio=(double)a/b;
        }
        if(s->canonical&&strcmp(s->task,"fl2va")&&strcmp(aspect,"auto")&&!in(aspect,"21:9|16:9|4:3|1:1|3:4|9:16"))return fail(e,n,"target.aspect_ratio","not in the pinned SGLang task ratio set");
        if(ratio<0.25||ratio>4)return fail(e,n,"target.aspect_ratio","must be within 1:4..4:1");
        double w=ratio>=1?(double)short_edge*ratio:(double)short_edge,h=ratio>=1?(double)short_edge:(double)short_edge/ratio;
        if(w*h>768.0*1344){double scale=sqrt((768.0*1344)/(w*h));w*=scale;h*=scale;}
        int width=(int)fmax(32,round_even(w/32)*32),height=(int)fmax(32,round_even(h/32)*32);
        const sj_value *assertion;int64_t asserted;const char *keys[]={"width","height"};int dims[]={width,height},needed[]={s->width_needed,s->height_needed};
        const char *size=sj_field(s->transport,"size");int sw=0,sh=0;
        if(size&&*size&&sscanf(size,"%dx%d%c",&sw,&sh,&extra)!=2)return fail(e,n,"size","expected WIDTHxHEIGHT");
        for(int i=0;i<2;i++)if(needed[i]){
            assertion=sj_get(s->transport,keys[i]);if(assertion&&assertion->type!=LJ_NULL&&(!sj_i64(assertion,&asserted)||asserted!=dims[i]))return fail(e,n,keys[i],"contradicts target");
            if(size&&*size&&(i?sh:sw)!=dims[i])return fail(e,n,"size","contradicts target");
            if(!integer(&s->request,keys[i],dims[i],e,n))return 0;
        }
    }
    if(s->duration_needed){
        const sj_value *duration=sj_get(target,"duration_seconds");double seconds;
        if(!duration||duration->type==LJ_NULL)seconds=audio;
        else {if(duration->type!=LJ_NUMBER)return fail(e,n,"target.duration_seconds","expected number");seconds=duration->number;}
        if(!isfinite(seconds)||seconds<4||seconds>15)return fail(e,n,"target.duration_seconds","must be 4..15 seconds (or a qualifying audio reference)");
        const sj_value *assertion=sj_get(s->transport,"seconds");if(assertion&&assertion->type!=LJ_NULL&&assertion->number!=seconds)return fail(e,n,"seconds","contradicts target duration");
        int frames=(int)round_even(seconds*24);frames+=(5-frames%17+17)%17;
        if(!integer(&s->request,"frames",frames,e,n))return 0;
    }s->width_needed=s->height_needed=s->duration_needed=0;return 1;
}
sj_value *h3_request_json(const h3_request *r){
    sj_value *o=sj_object(),*a=sj_array();if(!o||!a){sj_free(o);sj_free(a);return NULL;}
    sj_add(o,"schema",sj_int(1));sj_add(o,"options",a);
    for(size_t i=0;i<r->count;i++){
        const h3_option_value *x=&r->values[i];sj_value *v=sj_object();sj_add(v,"name",sj_string(x->option->name));
        sj_add(v,"source",sj_int(x->source));sj_add(v,"value",x->value?sj_string(x->value):sj_bool(1));
        if (x->second)
            sj_add(v, "second", sj_string(x->second));
        if (!sj_add(a, NULL, v)) {
            sj_free(o);
            return NULL;
        }
    }return o;
}
int h3_request_from_json(const sj_value *v,h3_request *r,char *e,size_t n){
    int64_t schema;const sj_value *a=sj_get(v,"options");
    if(!sj_i64(sj_get(v,"schema"),&schema)||schema!=1||!a||a->type!=LJ_ARRAY||a->count>512)return fail(e,n,"IPC","unsupported request schema");
    for(size_t i=0;i<a->count;i++){
        const sj_value *x=a->items[i],*value=sj_get(x,"value");const char *name=sj_field(x,"name");int64_t source;
        if(!name||!sj_i64(sj_get(x,"source"),&source)||source<0||source>H3_SOURCE_MANAGED)return fail(e,n,"IPC","invalid option descriptor");
        const h3_option_descriptor *d=h3_option_find(name);if(!d)return fail(e,n,"IPC","unknown option");
        if((d->arity&&(!value||value->type!=LJ_STRING))||(!d->arity&&(!value||value->type!=LJ_BOOL||value->number==0)))return fail(e,n,"IPC","invalid typed value");
        if(!h3_request_add(r,name,d->arity?value->text:NULL,sj_field(x,"second"),(h3_option_source)source,e,n))return 0;
    }return 1;
}
sj_value *h3_option_schema(void){
    sj_value *a=sj_array();const char *types[]={"boolean","integer","uint64","number","string","enum"};
    const char *roles[]={"none","media","state","model","output","cache"};
    for(size_t i=0;i<h3_cli_option_count;i++){
        const h3_option_descriptor *d=&h3_cli_options[i];sj_value *v=sj_object();sj_add(v,"name",sj_string(d->name));
        sj_add(v,"type",sj_string(types[d->type]));sj_add(v,"arity",sj_int(d->arity));sj_add(v,"path_role",sj_string(roles[d->path]));
        sj_add(v,"repeatable",sj_bool(d->repeatable));sj_add(v,"group",sj_string(d->group));sj_add(v,"choices",sj_string(d->choices));
        sj_add(v,"default",sj_string(d->default_value));sj_add(v,"minimum",sj_real(d->minimum));sj_add(v,"maximum",d->type==H3_OPT_U64?sj_uint(UINT64_MAX):sj_real(d->maximum));
        sj_add(v,"default_policy",sj_string(d->default_value?"fresh CLI default; presets, device policy or restored state may override":"omitted; native operation/device/state supplies the effective value"));
        sj_add(v,"platform",sj_string(d->platform));sj_add(v,"operations",sj_string(d->operations));
#ifdef __APPLE__
        int available=strcmp(d->platform,"cuda")!=0;
#else
        int available=strcmp(d->platform,"metal")!=0;
#endif
        sj_add(v,"available",sj_bool(available));sj_add(v,"mapping_available",sj_bool(1));
        if(d->id>0&&d->id<128){char alias[2]={(char)d->id,0};sj_add(v,"short",sj_string(alias));}
        if(!sj_add(a,NULL,v)){sj_free(a);return NULL;}
    }return a;
}

sj_value *h3_submission_schema(void){
    sj_value *v=sj_object(),*properties=sj_object();sj_add(v,"$schema",sj_string("https://json-schema.org/draft/2020-12/schema"));sj_add(v,"type",sj_string("object"));sj_add(v,"additionalProperties",sj_bool(0));sj_add(v,"properties",properties);
    const char *lists[]={strings,ints,numbers,bools,"target|diffusers_kwargs|cache_dit_params|skip_softmax_params","conditions","seed","h3cli"};
    const char *types[]={"string","integer","number","boolean","object","array","integer","string"};
    for(size_t i=0;i<8;i++){char *copy=strdup(lists[i]);if(!copy)continue;char *save=NULL;
        for(char *key=strtok_r(copy,"|",&save);key;key=strtok_r(NULL,"|",&save)){sj_value *field=sj_object();sj_value *allowed_types=sj_array();sj_add(allowed_types,NULL,sj_string(types[i]));if(i==6)sj_add(allowed_types,NULL,sj_string("array"));if(i!=7)sj_add(allowed_types,NULL,sj_string("null"));sj_add(field,"type",allowed_types);
            if(i==7)sj_add(field,"maxLength",sj_int(65536));
            sj_add(properties,key,field);}free(copy);}
    sj_add(v,"description",sj_string("SGLang H3 compatible subset. Unsupported active controls return a field-specific error. h3cli is the sole native request extension; SDK wrappers are flattened before validation."));
    return v;
}
