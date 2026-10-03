/* Check the private cache copy that preserves preview-independent checkpoints. */
#include "../src/engine.c"
#include <assert.h>
int main(void) {
    uint16_t values[]={1,2,3,4,5,6};uint8_t tags[]={1,0,1};
    uint32_t ids[]={10,20,30},positions[]={0,1,2,3,4,5,6,7,8};uint64_t spans[]={1,1};
    h3_text_diagnostics diagnostics={3,1,ids,positions,spans};
    h3_text_embedding input={3,2,values,{0},tags,&diagnostics},out={0};
    h3_ctx ctx={0};ctx.cache_enabled=1;
    assert(h3_conditioning_cache_store(&ctx,"fixture",&input,NULL,0,NULL,0,NULL,0,0));
    ids[0]=99;positions[0]=99;spans[0]=99;values[0]=99;tags[0]=0;
    float *v,*a;size_t nv,na,nrefs;h3_layout_ref *refs;int conditioned;
    assert(h3_conditioning_cache_load(&ctx,&out,&v,&nv,&a,&na,&refs,&nrefs,&conditioned));
    assert(out.diagnostics && out.diagnostics->ids[0]==10 && out.diagnostics->positions[0]==0 && out.diagnostics->spans[0]==1);
    assert(out.values[0]==1 && out.tags[0]==1 && !nv && !na && !nrefs);
    h3_cache_info info;h3_cache_get_info(&ctx,&info);
    assert(info.embedding_bytes==sizeof(values)+sizeof(tags)+sizeof(diagnostics)+sizeof(ids)+sizeof(positions)+sizeof(spans));
    h3_cache_clear(&ctx);assert(out.diagnostics->ids[0]==10);h3_text_embedding_free(&out);
    diagnostics.positions=NULL;diagnostics.span_count=0;diagnostics.spans=NULL;
    assert(h3_text_embedding_copy(&out,&input));assert(!out.diagnostics->positions && !out.diagnostics->spans);h3_text_embedding_free(&out);
    diagnostics.tokens=4;assert(!h3_text_embedding_copy(&out,&input));assert(!out.values && !out.diagnostics);
    puts("ok: checkpoint diagnostics survive cache copy/clear without aliasing");return 0;
}
