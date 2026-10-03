"""Test-only Qwen presentation/layer capture; never compiled into production."""
from pathlib import Path
from source_tree import source_path
HELPER=r'''
static void scaling_dump(const char *name, const void *data, size_t bytes) {
    const char *dir=getenv("H3_TEST_QWEN_DUMP"); if(!dir || !data) return;
    char path[4096];snprintf(path,sizeof(path),"%s/%s",dir,name);
    FILE *f=fopen(path,"wb");if(!f || fwrite(data,1,bytes,f)!=bytes || fclose(f)) abort();
}
static void scaling_tensor(const char *name, h3_gpu_tensor *t) {
    if(!getenv("H3_TEST_QWEN_DUMP")) return;
    size_t n=h3_gpu_tensor_elements(t);uint16_t *data=malloc(n*2);
    if(!data || !h3_gpu_tensor_read_bf16(t,data,n)) abort();
    scaling_dump(name,data,n*2);free(data);
}
'''
def instrument(directory):
    p=source_path(directory,'text_encoder.c');s=p.read_text();needle='enum {';s=s.replace(needle,HELPER+'\n'+needle,1)
    needle='    if (!h3_memory_checkpoint(progress && progress(0, TEXT_LAYERS, progress_opaque),'
    assert s.count(needle)==1
    s=s.replace(needle,r'''    uint64_t capture_spec[]={token_count,span_count,position_ids!=NULL,tags!=NULL};
    scaling_dump("spec.u64",capture_spec,sizeof(capture_spec));
    scaling_dump("ids.u32",token_ids,token_count*4);
    scaling_dump("positions.u32",position_ids,token_count*3*4);
    scaling_dump("tags.u8",tags,token_count);
    for(size_t i=0;i<span_count;i++) {
        char name[64];uint64_t spec[]={spans[i].start,spans[i].tokens};
        snprintf(name,sizeof(name),"span-%zu.u64",i);scaling_dump(name,spec,sizeof(spec));
        snprintf(name,sizeof(name),"vision-%zu.bf16",i);scaling_dump(name,spans[i].embeddings,spans[i].tokens*5120*2);
        for(int j=0;j<3;j++) {snprintf(name,sizeof(name),"deepstack-%zu-%d.bf16",i,j);scaling_dump(name,spans[i].deepstack[j],spans[i].tokens*5120*2);}
    }
    if(getenv("H3_TEST_QWEN_CAPTURE_ONLY")) {fail(error,error_size,"Qwen presentation captured");return 0;}
'''+needle)
    needle='    int prefetch_threads = text_prefetch_threads();';assert s.count(needle)==1
    s=s.replace(needle,'    scaling_tensor("layer-00.bf16", hidden);\n'+needle)
    needle='        if (!h3_memory_checkpoint(\n                progress && progress(layer + 1, TEXT_LAYERS, progress_opaque),';assert s.count(needle)==1
    s=s.replace(needle,r'''        if(layer<5 || layer==9 || layer==24 || layer==39 || layer==49) {
            char name[64];snprintf(name,sizeof(name),"layer-%02d.bf16",layer+1);scaling_tensor(name,hidden);
        }
        if(layer==0) {
            scaling_tensor("first-query.bf16",query);scaling_tensor("first-key.bf16",key);
            scaling_tensor("first-value.bf16",value);scaling_tensor("first-attention.bf16",attention_heads);
        }
'''+needle);p.write_text(s)
