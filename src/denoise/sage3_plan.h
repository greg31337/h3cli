#ifndef H3_SAGE3_PLAN_H
#define H3_SAGE3_PLAN_H
#include "src/geometry.h"
#include "src/denoise/attention.h"

/* Same shape-only plan as Sage3 plan 1. Shared with host admission so maximum
 * shapes can be checked without allocating Q/K/V or a correction matrix. */
#define H3_SAGE_RESERVED_BYTES (UINT64_C(256)+(UINT64_C(4)<<20))
typedef struct {
    uint32_t padded_rows, partials, query_rows, head_group;
    uint64_t per_head_bytes, workspace_bytes, input_elements;
} h3_sage3_plan;
static inline uint32_t h3_sage_reduce_partials(uint32_t rows) {
    if(rows<64)return 1;
    uint32_t values=(rows+3)/4,small=(values+15)/16,large=(values+255)/256;
    if(values<256)return 1;
    if(small>37)small=37;
    return small>large?small:large;
}
static inline int h3_sage3_plan_build(uint64_t rows,uint32_t heads,
                                     uint64_t capacity,h3_sage3_plan *out) {
    if(!out||!h3_canvas_sequence_valid(rows,1)||heads!=56||
       capacity>H3_ATTENTION_WORKSPACE_BYTES||capacity<=H3_SAGE_RESERVED_BYTES)return 0;
    uint64_t padded=(rows+127)/128*128,partials=h3_sage_reduce_partials((uint32_t)rows);
    uint64_t available=capacity-H3_SAGE_RESERVED_BYTES;
    uint64_t fixed=padded*400+partials*128*8+8192;
    if(fixed>=available)return 0;
    uint64_t blocks=(available-fixed)/(128*600+padded*4);
    if(blocks>padded/128)blocks=padded/128;
    if(!blocks)return 0;
    uint64_t query=blocks*128,per_head=fixed+query*600+blocks*padded*4;
    uint64_t group=available/per_head;
    if(group>heads)group=heads;
    if(!group)return 0;
    *out=(h3_sage3_plan){(uint32_t)padded,(uint32_t)partials,(uint32_t)query,(uint32_t)group,
        per_head,H3_SAGE_RESERVED_BYTES+per_head*group,rows*heads*128};
    return 1;
}
#endif
