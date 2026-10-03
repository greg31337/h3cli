#include "src/weights/residency.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int bad(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s);return 0;}
static int add(uint64_t a,uint64_t b,uint64_t *v){if(a>UINT64_MAX-b)return 0;*v=a+b;return 1;}
static int mul(uint64_t a,uint64_t b,uint64_t *v){if(a&&b>UINT64_MAX/a)return 0;*v=a*b;return 1;}
const char *h3_weight_mode_name(int m){return m==H3_WEIGHTS_RESIDENT?"resident":m==H3_WEIGHTS_STREAM?"stream":"auto";}
static int number(const char *name,uint64_t *v,char *e,size_t n){
    const char *s=getenv(name);if(!s)return 1;
    if(!*s||*s<'0'||*s>'9')return bad(e,n,"invalid CUDA weight test cap");
    char *end;errno=0;unsigned long long x=strtoull(s,&end,10);
    if (errno || *end)
        return bad(e, n, "invalid CUDA weight test cap");
    *v = x;
    return 1;
}
int h3_weight_options_read(h3_weight_options *o,int force,char *e,size_t n){
    *o=(h3_weight_options){.max_resident=-1};const char *s=getenv("H3_CUDA_WEIGHT_MODE");
    if(s&&strcmp(s,"auto")){
        if(!strcmp(s,"resident"))o->mode=H3_WEIGHTS_RESIDENT;
        else if(!strcmp(s,"stream"))o->mode=H3_WEIGHTS_STREAM;
        else return bad(e,n,"CUDA weight mode must be auto, resident, or stream");
    }
    if(force&&o->mode==H3_WEIGHTS_RESIDENT)return bad(e,n,"--ssd-streaming conflicts with CUDA resident weight mode");
    if(force)o->mode=H3_WEIGHTS_STREAM;
    uint64_t cap=50;
    if(!number("H3_TEST_CUDA_RESIDENT_BLOCKS",&cap,e,n)||cap>50)return bad(e,n,"H3_TEST_CUDA_RESIDENT_BLOCKS must be 0..50");
    if(getenv("H3_TEST_CUDA_RESIDENT_BLOCKS"))o->max_resident=(int)cap;
    return number("H3_TEST_CUDA_CAPACITY_BYTES",&o->capacity_limit,e,n)&&
           number("H3_CUDA_TEST_MEMORY_BUDGET",&o->allocation_limit,e,n);
}
int h3_weight_options_equal(const h3_weight_options *a,const h3_weight_options *b){
    return a->mode==b->mode&&a->max_resident==b->max_resident&&
        a->capacity_limit==b->capacity_limit&&a->allocation_limit==b->allocation_limit;
}
int h3_weight_retry_cap(unsigned n){return n?(int)((n-1)/2):-1;}
int h3_weight_packed_size(uint32_t rows,uint32_t columns,int mode,
    uint64_t *scale,uint64_t *global,uint64_t *bytes){
    uint64_t payload,scales,end;
    if(!rows||!columns||(mode!=1&&mode!=2)||!scale||!global||!bytes||
       (mode==2&&columns%16)||!mul(rows,columns,&payload))return 0;
    if(mode==2)payload/=2;
    if (!add(payload, 255, &end))
        return 0;
    *scale = end / 256 * 256;
    scales=mode==1?4:(((uint64_t)rows+127)/128*128)*(((uint64_t)columns/16+3)/4*4);
    if(!add(*scale,scales,&end)||!add(end,255,&end))return 0;
    *global=end/256*256;return add(*global,16,bytes);
}
int h3_weight_plan_build(h3_weight_plan *p,h3_weight_options o,uint64_t mask,
    uint64_t block,uint64_t free_bytes,uint64_t live,uint64_t future,uint64_t reserve,char *e,size_t n){
    if(!p||!mask||(mask>>50)||!block||o.mode<0||o.mode>2||o.max_resident< -1||o.max_resident>50)
        return bad(e,n,"invalid weight planner input");
    *p=(h3_weight_plan){.options=o,.active_mask=mask,.live_bytes=live,
        .future_bytes=future,.reserve_bytes=reserve,.block_bytes=block};
    for(uint64_t m=mask;m;m&=m-1)p->active_count++;
    if(o.capacity_limit){uint64_t capped=o.capacity_limit>live?o.capacity_limit-live:0;if(capped<free_bytes)free_bytes=capped;}
    p->free_bytes=free_bytes;
    uint64_t fixed,full,slots;
    if(!add(future,reserve,&fixed)||!mul(block,p->active_count,&full)||!mul(block,2,&slots))
        return bad(e,n,"weight planner byte overflow");
    uint64_t budget=free_bytes>=fixed?free_bytes-fixed:0;
    int full_allowed=o.max_resident<0||(unsigned)o.max_resident>=p->active_count;
    if(o.mode!=H3_WEIGHTS_STREAM&&full_allowed&&free_bytes>=fixed&&full<=budget){
        p->resident_count=p->active_count;p->resident_mask=mask;p->resident_bytes=full;p->effective_mode=H3_WEIGHTS_RESIDENT;return 1;
    }
    if(o.mode==H3_WEIGHTS_RESIDENT)return bad(e,n,"resident weights need more capacity than available after future allocations/reserve or test cap");
    if(free_bytes<fixed||slots>budget)return bad(e,n,"insufficient CUDA capacity even for two-slot streaming plus future allocations/reserve");
    p->slot_bytes=slots;p->effective_mode=H3_WEIGHTS_STREAM;
    uint64_t count=(budget-slots)/block;
    if(count>=p->active_count)count=p->active_count-1;
    if(o.max_resident>=0&&count>(unsigned)o.max_resident)count=(unsigned)o.max_resident;
    if(o.mode==H3_WEIGHTS_STREAM)count=0;
    p->resident_count=(unsigned)count;p->resident_bytes=count*block;
    if(count)p->effective_mode=H3_WEIGHTS_AUTO;
    for(unsigned i=0;count&&i<50;i++)if(mask&(UINT64_C(1)<<i)){p->resident_mask|=UINT64_C(1)<<i;count--;}
    return 1;
}
