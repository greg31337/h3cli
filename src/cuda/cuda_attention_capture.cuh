/* Explicit test-only capture of complete short-sequence post-RoPE Q/K/V.
 * No tensor capture occurs unless the caller supplies an existing directory.
 * H3_TEST_* diagnostics do not alter checkpoint arithmetic identity. */
static bool attention_capture_selected(const char *text,int value) {
    if(!text||!strcmp(text,"all"))return true;
    const char *p=text;
    while(*p){char *end=nullptr;long n=strtol(p,&end,10);if(end==p)return false;if(n==value)return true;if(*end!=',')return false;p=end+1;}
    return false;
}
static int attention_capture(h3_gpu *g,const h3_gpu_tensor *q,const h3_gpu_tensor *k,
    const h3_gpu_tensor *v,unsigned seq,unsigned heads,unsigned dim,unsigned block,int step) {
    const char *dir=getenv("H3_TEST_ATTENTION_CAPTURE_DIR");if(!dir||!*dir)return 1;
    unsigned max_seq=4096;
    const char *limit=getenv("H3_TEST_ATTENTION_CAPTURE_MAX_SEQUENCE");
    if(limit){char *end=nullptr;unsigned long value=strtoul(limit,&end,10);
        if(!end||*end||value<1||value>131072||!getenv("H3_TEST_ATTENTION_CAPTURE_BLOCKS")||!getenv("H3_TEST_ATTENTION_CAPTURE_STEPS"))
            return h3_gpu_set_error(g,"bounded attention capture requires max sequence <=131072 and explicit blocks/steps");
        max_seq=(unsigned)value;}
    if(!seq||heads!=56||dim!=128||seq>max_seq||!attention_capture_selected(getenv("H3_TEST_ATTENTION_CAPTURE_BLOCKS"),(int)block)||
       !attention_capture_selected(getenv("H3_TEST_ATTENTION_CAPTURE_STEPS"),step))return 1;
    size_t elements=(size_t)seq*heads*dim;
    if(!h3_gpu_require_bf16(g,q,elements,"capture Q")||!h3_gpu_require_bf16(g,k,elements,"capture K")||
       !h3_gpu_require_bf16(g,v,elements,"capture V"))return 0;
    std::string path=std::string(dir)+"/step-"+std::to_string(step)+"-block-"+std::to_string(block)+"-s-"+std::to_string(seq)+".qkv";
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0)return h3_gpu_set_error(g,"attention capture cannot create %s: %s",path.c_str(),strerror(errno));
    FILE *f=fdopen(fd,"wb");if(!f){close(fd);unlink(path.c_str());return h3_gpu_set_error(g,"attention capture fdopen failed");}
    bool ok=true;std::vector<uint16_t> data;
    try{data.resize((size_t)seq*heads*dim);}catch(...){ok=false;}
    const h3_gpu_tensor *tensors[]={q,k,v};
    for(auto *tensor:tensors)if(ok){
        ok=h3_gpu_tensor_read_bf16(tensor,data.data(),data.size())!=0;
        if(ok)ok=fwrite(data.data(),2,data.size(),f)==data.size();
    }
    if(fclose(f))ok=false;
    if(!ok){unlink(path.c_str());return h3_gpu_set_error(g,"attention capture failed for %s",path.c_str());}
    fprintf(stderr,"h3cli: captured attention QKV step=%d block=%u S=%u H=%u D=%u path=%s\n",step,block,seq,heads,dim,path.c_str());
    return 1;
}
