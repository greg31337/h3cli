#include "src/denoise/sol.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void h3_sol_layout_free(h3_sol_layout *p) {
    if(p){free(p->query);free(p->key);memset(p,0,sizeof(*p));}
}
int h3_sol_layout_build(const h3_layout *l,unsigned qb,unsigned kb,h3_sol_layout *out,char *e,size_t n) {
    if(!out)return 0;
    memset(out,0,sizeof(*out));
    const char *why="invalid SOL packed layout";
    if(!l||!l->segments||!l->segment_count||!l->seq_len||l->seq_len>10000000||
       (qb!=32&&qb!=64)||(kb!=32&&kb!=64&&kb!=128)||
       l->signature[1]<1||l->signature[2]<2||l->signature[3]<2||
       l->signature[2]%2||l->signature[3]%2||l->prefix.video_prefix_t<0||
       l->prefix.video_prefix_t>=l->signature[1]||l->prefix.audio_prefix_t<0||
       l->prefix.audio_prefix_t>=l->signature[4])goto fail;
    size_t spatial=(size_t)(l->signature[2]/2)*(l->signature[3]/2);
    if(spatial>SIZE_MAX/(size_t)l->signature[1])goto fail;
    size_t expected=spatial*(size_t)l->signature[1],cursor=0;unsigned videos=0;
    out->sequence=(uint32_t)l->seq_len;
    out->query_blocks=(out->sequence+qb-1)/qb;
    out->key_blocks=(out->sequence+kb-1)/kb;
    out->query=calloc(out->query_blocks,sizeof(*out->query));
    out->key=calloc(out->key_blocks,sizeof(*out->key));
    if(!out->query||!out->key){why="cannot allocate SOL protection metadata";goto fail;}
    for(size_t s=0;s<l->segment_count;s++) {
        const h3_segment *g=&l->segments[s];
        if(g->start!=cursor||g->stop<g->start||g->stop>l->seq_len||
           g->kind<H3_SEG_TEXT||g->kind>H3_SEG_VIDEO)goto fail;
        if(g->kind==H3_SEG_VIDEO) {
            if(++videos!=1||g->stop-g->start!=expected)goto fail;
        }
        for(size_t r=g->start;r<g->stop;r++) {
            int frame=g->kind==H3_SEG_VIDEO?(int)((r-g->start)/spatial):-1;
            /* All audio rows are conservative exact sinks/queries, including
             * generated audio and its full continuation prefix. All reference
             * kinds (including video packed as REF_IMAGE) are protected too. */
            unsigned protected_row=frame<0||frame<l->prefix.video_prefix_t||
                frame==0||frame==l->signature[1]-1;
            out->protected_rows+=protected_row;
            h3_sol_block *blocks[2]={&out->query[r/qb],&out->key[r/kb]};
            for(unsigned j=0;j<2;j++) {
                h3_sol_block *b=blocks[j];b->protect|=protected_row;
                if(!b->rows)b->first_frame=b->last_frame=frame;
                else {if(frame<b->first_frame)b->first_frame=frame;if(frame>b->last_frame)b->last_frame=frame;}
                b->rows++;
            }
        }
        cursor=g->stop;
    }
    if(cursor!=l->seq_len||videos!=1)goto fail;
    return 1;
fail:
    h3_sol_layout_free(out);if(e&&n)snprintf(e,n,"%s",why);return 0;
}
