#include "src/weights/residency.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    h3_weight_options o={.max_resident=-1};h3_weight_plan p;char e[256];
    /* Three noncontiguous blocks, 100 bytes each; 100 fixed bytes. */
    const uint64_t mask=(1ull<<0)|(1ull<<7)|(1ull<<49);
    assert(h3_weight_plan_build(&p,o,mask,100,400,900,40,60,e,sizeof(e)));
    assert(p.resident_mask==mask&&p.resident_bytes==300&&!p.slot_bytes);
    assert(h3_weight_plan_build(&p,o,mask,100,399,900,40,60,e,sizeof(e)));
    assert(!p.resident_count&&p.slot_bytes==200); /* Two-slot tax, no unsafe one-slot shortcut. */
    o.max_resident=2;
    assert(h3_weight_plan_build(&p,o,mask,100,500,900,40,60,e,sizeof(e)));
    assert(p.resident_count==2&&p.resident_mask==129&&p.slot_bytes==200);
    assert(h3_weight_plan_build(&p,o,mask,100,400,900,40,60,e,sizeof(e)));
    assert(p.resident_count==1&&p.resident_mask==1);
    o.mode=H3_WEIGHTS_RESIDENT;
    assert(!h3_weight_plan_build(&p,o,mask,100,10000,0,40,60,e,sizeof(e)));
    o.max_resident=-1;
    assert(!h3_weight_plan_build(&p,o,mask,100,399,0,40,60,e,sizeof(e)));
    o.mode=H3_WEIGHTS_STREAM;
    assert(h3_weight_plan_build(&p,o,mask,100,300,0,40,60,e,sizeof(e))&&!p.resident_count);
    assert(!h3_weight_plan_build(&p,o,mask,100,299,0,40,60,e,sizeof(e)));
    o.mode=H3_WEIGHTS_AUTO;o.capacity_limit=1200;
    assert(h3_weight_plan_build(&p,o,mask,100,9999,800,40,60,e,sizeof(e))&&p.resident_count==3);
    assert(h3_weight_plan_build(&p,o,mask,100,9999,801,40,60,e,sizeof(e))&&!p.resident_count);
    assert(!h3_weight_plan_build(&p,o,mask,100,9999,1201,40,60,e,sizeof(e)));
    o.capacity_limit=0;
    assert(!h3_weight_plan_build(&p,o,mask,UINT64_MAX,UINT64_MAX,0,0,0,e,sizeof(e)));
    assert(!h3_weight_plan_build(&p,o,mask,1,UINT64_MAX,0,UINT64_MAX,1,e,sizeof(e)));
    assert(!h3_weight_plan_build(&p,o,1ull<<50,1,999,0,0,0,e,sizeof(e)));
    assert(!h3_weight_plan_build(&p,o,0,1,999,0,0,0,e,sizeof(e)));
    int count=50,retries=0;
    while(count>0){int next=h3_weight_retry_cap((unsigned)count);assert(next>=0&&next<count);count=next;retries++;}
    assert(retries<=7&&h3_weight_retry_cap(0)==-1);
    uint64_t scale,global,bytes;
    assert(h3_weight_packed_size(128,64,1,&scale,&global,&bytes)&&scale==8192&&global==8448&&bytes==8464);
    assert(h3_weight_packed_size(128,64,2,&scale,&global,&bytes)&&scale==4096&&global==4608&&bytes==4624);
    assert(h3_weight_packed_size(129,64,2,&scale,&global,&bytes)&&scale==4352&&global==5376&&bytes==5392);
    assert(!h3_weight_packed_size(128,63,2,&scale,&global,&bytes));
    assert(!h3_weight_packed_size(0,64,1,&scale,&global,&bytes));
    unsetenv("H3_TEST_CUDA_RESIDENT_BLOCKS");unsetenv("H3_TEST_CUDA_CAPACITY_BYTES");unsetenv("H3_CUDA_TEST_MEMORY_BUDGET");
    setenv("H3_CUDA_WEIGHT_MODE","resident",1);
    assert(!h3_weight_options_read(&o,1,e,sizeof(e)));
    setenv("H3_CUDA_WEIGHT_MODE","auto",1);
    assert(h3_weight_options_read(&o,1,e,sizeof(e))&&o.mode==H3_WEIGHTS_STREAM);
    setenv("H3_TEST_CUDA_RESIDENT_BLOCKS","-1",1);assert(!h3_weight_options_read(&o,0,e,sizeof(e)));
    setenv("H3_TEST_CUDA_RESIDENT_BLOCKS","51",1);assert(!h3_weight_options_read(&o,0,e,sizeof(e)));
    setenv("H3_TEST_CUDA_RESIDENT_BLOCKS","0",1);assert(h3_weight_options_read(&o,0,e,sizeof(e))&&o.max_resident==0);
    setenv("H3_TEST_CUDA_CAPACITY_BYTES","18446744073709551616",1);assert(!h3_weight_options_read(&o,0,e,sizeof(e)));
    puts("ok: weight capacity boundaries, live accounting, slot tax, gaps, caps, overflow and bounded retries");
    return 0;
}
