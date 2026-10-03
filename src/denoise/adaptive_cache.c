#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "src/denoise/adaptive_cache.h"
#include "src/sampling/bridge.h"
#include <limits.h>
#include <errno.h>
#include <locale.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <xlocale.h>
#endif
#include <math.h>
#include <stdio.h>
#include <string.h>
const char *h3_adaptive_name(int mode) {
    return mode==0?"off":mode==1?"conservative":mode==2?"aggressive":"invalid";
}
int h3_adaptive_parse(const char *text,int *mode) {
    if(!text||!mode)return 0;
    for(int i=0;i<3;i++)if(!strcmp(text,h3_adaptive_name(i))){*mode=i;return 1;}
    return 0;
}
unsigned h3_adaptive_recipe(int mode,int quant,int references) {
    return mode?(references?H3_ADAPTIVE_REFERENCE_VERSION:
        quant?H3_ADAPTIVE_QUANT_VERSION:H3_ADAPTIVE_VERSION):0;
}
unsigned h3_adaptive_execution_recipe(int mode,int quant,int references,int continuation) {
    return mode&&continuation?H3_ADAPTIVE_CONTINUATION_VERSION:h3_adaptive_recipe(mode,quant,references);
}
int h3_adaptive_regions_build(const h3_layout *l,h3_adaptive_regions *out,char *e,size_t n) {
    h3_adaptive_regions p={0};
    if(!l||!out||!l->segments||!l->segment_count||l->segment_count>1000||
       !l->seq_len||l->seq_len>UINT32_MAX||l->frozen_audio||
       l->signature[1]<1||l->signature[1]>107||l->signature[4]<1||l->signature[4]>604||
       l->signature[2]<2||l->signature[3]<2||l->signature[2]%2||l->signature[3]%2||
       l->prefix.video_prefix_t<1||l->prefix.video_prefix_t>=l->signature[1]||
       l->prefix.audio_prefix_t<1||l->prefix.audio_prefix_t>=l->signature[4])goto bad;
    if(l->bridge&&(!h3_bridge_profile_valid(l->bridge)||
       l->bridge->prefix.video_prefix_t!=l->prefix.video_prefix_t||
       l->bridge->prefix.audio_prefix_t!=l->prefix.audio_prefix_t))goto bad;
    uint64_t spatial=(uint64_t)(l->signature[2]/2)*(unsigned)(l->signature[3]/2);
    if(spatial>UINT32_MAX/(unsigned)l->signature[1])goto bad;
    p.rows=(uint32_t)l->seq_len;p.spatial=(uint32_t)spatial;
    p.video_t=(unsigned)l->signature[1];p.audio_t=(unsigned)l->signature[4];
    size_t end=0;int video=0,audio=0;
    for(size_t i=0;i<l->segment_count;i++) {
        const h3_segment *s=&l->segments[i];
        if(s->start!=end||s->stop<=s->start||s->stop>l->seq_len||
           s->kind<H3_SEG_TEXT||s->kind>H3_SEG_VIDEO)goto bad;
        if(s->kind==H3_SEG_VIDEO){if(video++)goto bad;p.video_start=(unsigned)s->start;p.video_rows=(unsigned)(s->stop-s->start);}
        if(s->kind==H3_SEG_AUDIO){if(audio++)goto bad;p.audio_start=(unsigned)s->start;p.audio_rows=(unsigned)(s->stop-s->start);}
        end=s->stop;
    }
    if(end!=l->seq_len||video!=1||audio!=1||p.video_rows!=p.spatial*p.video_t||p.audio_rows!=2*p.audio_t||
       l->img_target_rows!=p.video_rows||l->audio_target_rows!=p.audio_rows)goto bad;
    p.counts[0]=p.rows;
    for(unsigned t=0;t<p.video_t;t++) {
        unsigned slot=1;
        if(t<(unsigned)l->prefix.video_prefix_t) {
            slot=0;
            if(l->bridge) {unsigned c=l->bridge->video_classes[t];
                if(c>=H3_TARGET_ROW_CLASSES)goto bad;
                if(l->bridge->class_mask[c]>0) {
                    if(c==H3_ROW_GENERATED_VIDEO)slot=12; /* unit-strength bridge prefix */
                    else {
                        if(c<H3_ROW_BRIDGE_VIDEO_FIRST||c>=H3_ROW_BRIDGE_AUDIO_FIRST)goto bad;
                        slot=3+c-H3_ROW_BRIDGE_VIDEO_FIRST;
                    }
                }
            }
        }
        p.video[t]=(uint8_t)slot;if(slot)p.counts[slot]+=p.spatial;
    }
    for(unsigned t=0;t<p.audio_t;t++) {
        unsigned slot=2;
        if(t<(unsigned)l->prefix.audio_prefix_t) {
            slot=0;
            if(l->bridge) {unsigned c=l->bridge->audio_classes[t];
                if(c>=H3_TARGET_ROW_CLASSES)goto bad;
                if(l->bridge->class_mask[c]>0) {
                    if(c==H3_ROW_GENERATED_AUDIO)slot=22;
                    else {
                        if(c<H3_ROW_BRIDGE_AUDIO_FIRST)goto bad;
                        slot=13+c-H3_ROW_BRIDGE_AUDIO_FIRST;
                    }
                }
            }
        }
        p.audio[t]=(uint8_t)slot;if(slot)p.counts[slot]+=2;
    }
    if(!p.counts[1]||!p.counts[2])goto bad;
    *out=p;return 1;
bad:if(e&&n)snprintf(e,n,"invalid adaptive continuation target/class layout");return 0;
}
int h3_adaptive_regions_score(const h3_adaptive_regions *p,const float scores[H3_ADAPTIVE_REGIONS],float *score) {
    if(!p||!scores||!score||!p->counts[0]||!p->counts[1]||!p->counts[2])return 0;
    float maximum=0;
    for(unsigned i=0;i<H3_ADAPTIVE_REGIONS;i++) {
        if(!isfinite(scores[i])||scores[i]<0||(!p->counts[i]&&scores[i]!=0))return 0;
        if(p->counts[i])maximum=fmaxf(maximum,scores[i]);
    }
    *score=maximum;return 1;
}
int h3_adaptive_threshold_parse(const char *text,float *threshold) {
    if(!text||!*text||!threshold)return 0;
    const char *p=text;int digits=0;
    while(*p>='0'&&*p<='9'){p++;digits++;}
    if(*p=='.'){p++;while(*p>='0'&&*p<='9'){p++;digits++;}}
    if(!digits)return 0;
    if(*p=='e'||*p=='E'){
        p++;if(*p=='+'||*p=='-')p++;
        const char *start=p;while(*p>='0'&&*p<='9')p++;
        if(p==start)return 0;
    }
    if(*p)return 0;
    /* Enforce the decimal upper bound before rounding, even for a spelling
     * infinitesimally above one that strtod would round down to exactly one. */
    int integer_digits=0,index=0,first=-1,first_digit=0,tail=0,fraction=0;
    p=text;
    for(;*p&&*p!='e'&&*p!='E';p++){
        if(*p=='.'){fraction=1;continue;}
        if(!fraction)integer_digits++;
        if(first<0&&*p!='0'){first=index;first_digit=*p;}
        else if(first>=0&&*p!='0')tail=1;
        index++;
    }
    int exponent=0,sign=1;
    if(*p){p++;if(*p=='-'||*p=='+'){if(*p=='-')sign=-1;p++;}
        for(;*p;p++)if(exponent<1000000)exponent=exponent*10+(*p-'0');}
    int64_t point=(int64_t)integer_digits+(int64_t)sign*exponent-first;
    if(first>=0&&(point>1||(point==1&&(first_digit!='1'||tail))))return 0;
    locale_t locale=newlocale(LC_NUMERIC_MASK,"C",(locale_t)0);
    if(!locale)return 0;
    char *end;errno=0;double exact=strtod_l(text,&end,locale);
    int ok=!errno&&!*end&&isfinite(exact)&&exact>=0&&exact<=1;
    errno=0;float value=strtof_l(text,&end,locale);
    ok=ok&&!errno&&!*end&&isfinite(value)&&value>=0&&value<=1;
    freelocale(locale);if(ok)*threshold=value;return ok;
}
int h3_adaptive_max_hits_parse(const char *text,int *max_hits) {
    if(!text||!*text||!max_hits)return 0;
    unsigned value=0;
    for(const char *p=text;*p;p++){
        if(*p<'0'||*p>'9'||value>16)return 0;
        value=value*10+(unsigned)(*p-'0');
    }
    if(value<1||value>16)return 0;
    *max_hits=(int)value;return 1;
}
int h3_adaptive_score(unsigned recipe,const float scores[3],float *score) {
    if(!scores||!score||recipe<1||recipe>3)return 0;
    for(int i=0;i<3;i++)if(!isfinite(scores[i])||scores[i]<0)return 0;
    *score=recipe==H3_ADAPTIVE_REFERENCE_VERSION?
        fmaxf(scores[0],fmaxf(scores[1],scores[2])):scores[0];
    return 1;
}
int h3_adaptive_warmup(int configured) { return configured?configured:4; }
uint64_t h3_adaptive_budget(uint64_t bytes) { return bytes?bytes:H3_ADAPTIVE_DEFAULT_BYTES; }
int h3_adaptive_mib_parse(const char *text,uint64_t *bytes) {
    if(!text||!*text||!bytes)return 0;
    uint64_t value=0,limit=(uint64_t)SIZE_MAX/1048576;
    for(const unsigned char *p=(const unsigned char *)text;*p;p++) {
        if(*p<'0'||*p>'9'||value>limit/10||
           (value==limit/10&&(uint64_t)(*p-'0')>limit%10))return 0;
        value=value*10+(*p-'0');
    }
    if(!value)return 0;
    *bytes=value*1048576;return 1;
}
int h3_adaptive_plan_size(int mode,size_t rows,size_t columns,
    h3_adaptive_plan *p,char *e,size_t n) {
    if(p)memset(p,0,sizeof(*p));
    if(!p||mode<0||mode>2)goto fail;
    if(!mode)return 1;
    if(!rows||!columns||rows>SIZE_MAX/columns)goto fail;
    size_t elements=rows*columns;
    /* Fixed 256 CTAs x 6 FP32 partials plus six final scalar sums. */
    size_t scratch=(256*6+6)*sizeof(float);
    if(elements>(SIZE_MAX-scratch)/6)goto fail;
    *p=(h3_adaptive_plan){elements,elements*2,scratch,elements*6+scratch,elements*4};return 1;
fail:if(e&&n)snprintf(e,n,"adaptive cache shape is invalid or overflows platform size");return 0;
}
int h3_adaptive_plan_admit(const h3_adaptive_plan *p,uint64_t budget,char *e,size_t n) {
    budget=h3_adaptive_budget(budget);
    if(!p||budget>SIZE_MAX){if(e&&n)snprintf(e,n,"adaptive cache ceiling exceeds platform size");return 0;}
    if(p->bytes<=budget)return 1;
    if(e&&n)snprintf(e,n,"adaptive cache requires %llu bytes, ceiling %llu bytes; minimum --adaptive-cache-max-mib %llu",
        (unsigned long long)p->bytes,(unsigned long long)budget,
        (unsigned long long)(p->bytes/1048576+(p->bytes%1048576!=0)));
    return 0;
}
int h3_adaptive_plan_budget(int mode,size_t rows,size_t columns,uint64_t budget,
    h3_adaptive_plan *p,char *e,size_t n) {
    return h3_adaptive_plan_size(mode,rows,columns,p,e,n)&&h3_adaptive_plan_admit(p,budget,e,n);
}
int h3_adaptive_plan_make(int mode,size_t rows,size_t columns,
    h3_adaptive_plan *p,char *e,size_t n) {
    return h3_adaptive_plan_budget(mode,rows,columns,0,p,e,n);
}
int h3_adaptive_plan_recipe(int mode,unsigned recipe,size_t rows,size_t columns,uint64_t budget,
    h3_adaptive_plan *p,char *e,size_t n) {
    if(!h3_adaptive_plan_size(mode,rows,columns,p,e,n))return 0;
    if(mode&&recipe==H3_ADAPTIVE_CONTINUATION_VERSION) {
        size_t scratch=257*2*H3_ADAPTIVE_REGIONS*sizeof(float);
        if(p->elements>(SIZE_MAX-scratch)/6){if(e&&n)snprintf(e,n,"adaptive continuation workspace overflow");return 0;}
        p->scratch_bytes=scratch;p->bytes=p->elements*6+scratch;
    } else if(mode&&(recipe<1||recipe>3)){if(e&&n)snprintf(e,n,"invalid adaptive recipe");return 0;}
    return h3_adaptive_plan_admit(p,budget,e,n);
}
int h3_adaptive_decide(int mode,const h3_adaptive_history *h,int step,
    int steps,int warmup,unsigned phase,float score,float threshold,int max_hits,const char **reason) {
    const char *why=NULL;
    warmup=h3_adaptive_warmup(warmup);
    if(!h||mode<1||mode>2||step<0||steps<1||step>=steps||phase>1||
       warmup<2||warmup>16||!isfinite(threshold)||threshold<0||threshold>1||
       max_hits<1||max_hits>16||!isfinite(score)||score<0){if(reason)*reason="invalid";return -1;}
    if(!h->ready)why="empty";
    else if(step!=h->last_step+1)why="discontinuity";
    else if(step<warmup)why="warmup";
    else if(step==steps-1)why="final";
    else if(phase!=h->phase)why="attention-phase";
    else if(h->streak>=(unsigned)max_hits)why="streak";
    else if(!(score<threshold))why="score";
    if (reason)
        *reason = why ? why : "hit";
    return why != NULL;
}
void h3_adaptive_commit(h3_adaptive_history *h,int step,unsigned phase,int refresh) {
    h->ready=1;h->last_step=step;h->phase=phase;
    if(refresh){h->last_refresh=step;h->streak=0;}else h->streak++;
}
