#include "src/log.h"
/* Native narrow DiT projections. Original implementation of the cuBLASLt
 * documented tensorwide FP8 and tiled VEC16 UE4M3 NVFP4 contracts. */
extern "C" {
#include "src/digest.h"
#include "src/weights/quant_cache.h"
}
#if CUDART_VERSION >= 12080
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <sys/file.h>

static size_t quant_up(size_t n,size_t alignment){return (n+alignment-1)/alignment*alignment;}
static size_t quant_scale_bytes(unsigned rows,unsigned k,int mode) {
    return mode==H3_QUANT_FP8?4:quant_up(rows,128)*quant_up(k/16,4);
}
static size_t quant_bytes(unsigned rows,unsigned k,int mode,size_t *scale,size_t *global) {
    uint64_t s,g,bytes;
    if(!h3_weight_packed_size(rows,k,mode,&s,&g,&bytes))return 0;
    *scale=(size_t)s;*global=(size_t)g;return (size_t)bytes;
}
__host__ __device__ static size_t quant_sf(unsigned row,unsigned block,unsigned k) {
    unsigned tiles=(k/16+3)/4;
    return ((size_t)(row/128)*tiles+block/4)*512+(row%32)*16+((row%128)/32)*4+block%4;
}
__global__ static void quant_amax(const __nv_bfloat16 *x,size_t count,float *maximum,unsigned *fault) {
    float value=0;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(size_t)blockDim.x*gridDim.x) {
        float v=__bfloat162float(x[i]);
        if(!isfinite(v))atomicExch(fault,1u);
        value=fmaxf(value,fabsf(v));
    }
    for(int offset=16;offset;offset/=2)value=fmaxf(value,__shfl_down_sync(0xffffffff,value,offset));
    if(!(threadIdx.x%32))atomicMax((unsigned*)maximum,__float_as_uint(value));
}
__global__ static void quant_finish_scale(float *global,const float *weight_global,int mode) {
    float scale=fmaxf(global[0]/(mode==H3_QUANT_FP8?448.f:2688.f),1.e-20f);
    global[0]=scale;global[1]=mode==H3_QUANT_NVFP4&&weight_global?scale*weight_global[0]:1.f;global[2]=0;
}
__global__ static void quant_fp8_pack(const __nv_bfloat16 *x,unsigned char *out,float *scale,
                                     const float *global,unsigned rows,unsigned padded,unsigned k) {
    if(!blockIdx.x&&!threadIdx.x)*scale=*global;
    size_t count=(size_t)padded*k;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(size_t)blockDim.x*gridDim.x) {
        float v=i<(size_t)rows*k?__bfloat162float(x[i]) / *global:0;
        out[i]=__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3);
    }
}
__global__ static void quant_fp4_pack(const __nv_bfloat16 *x,unsigned char *out,unsigned char *scales,
                                     const float *global,unsigned rows,unsigned padded,unsigned k) {
    size_t blocks=(size_t)padded*(k/16);
    for(size_t b=(size_t)blockIdx.x*blockDim.x+threadIdx.x;b<blocks;b+=(size_t)blockDim.x*gridDim.x) {
        unsigned row=(unsigned)(b/(k/16)),col=(unsigned)(b%(k/16));float values[16],maximum=0;
        for(unsigned j=0;j<16;j++) {
            values[j]=row<rows?__bfloat162float(x[b*16+j]):0;
            maximum=fmaxf(maximum,fabsf(values[j]));
        }
        unsigned char sf=__nv_cvt_float_to_fp8(fmaxf(maximum/(6.f * *global),0.001953125f),__NV_SATFINITE,__NV_E4M3);
        __nv_fp8_e4m3 decoded;decoded.__x=sf;
        float reciprocal=1.f/((float)decoded * *global);
        scales[quant_sf(row,col,k)]=sf;
        for(unsigned j=0;j<8;j++) {
            unsigned lo=__nv_cvt_float_to_fp4(values[j*2]*reciprocal,__NV_E2M1,cudaRoundNearest);
            unsigned hi=__nv_cvt_float_to_fp4(values[j*2+1]*reciprocal,__NV_E2M1,cudaRoundNearest);
            out[b*8+j]=(unsigned char)(lo|(hi<<4));
        }
    }
}
/* Optional diagnostics are a separate kernel: the normal packing/GEMM path
 * performs no extra scan or host synchronization. Counts include weights on
 * cold preparation and activations, so warm/cold totals are distinguished. */
__global__ static void quant_diagnostics(const __nv_bfloat16 *source,const unsigned char *packed,
    unsigned rows,unsigned k,int mode,size_t scales,size_t global,unsigned long long *counts) {
    unsigned clipped=0,zeroed=0;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<(size_t)rows*k;i+=(size_t)blockDim.x*gridDim.x) {
        float x=__bfloat162float(source[i]),factor=*(const float*)(packed+global);unsigned value;
        if(mode==H3_QUANT_FP8)value=packed[i]&127;
        else {
            value=(packed[i/2]>>((i%2)*4))&7;
            __nv_fp8_e4m3 sf;sf.__x=packed[scales+quant_sf((unsigned)(i/k),(unsigned)(i%k)/16,k)];factor*=(float)sf;
        }
        clipped+=fabsf(x/factor)>(mode==H3_QUANT_FP8?448.f:6.f);
        zeroed+=x!=0&&value==0;
    }
    if(clipped)atomicAdd(counts+1,(unsigned long long)clipped);
    if(zeroed)atomicAdd(counts+2,(unsigned long long)zeroed);
}
static int quant_pack(h3_gpu *g,const void *source,void *packed,unsigned rows,unsigned padded,unsigned k,
                      int mode,size_t scale,size_t global,const float *weight_global=nullptr) {
    float *gs=(float*)((char*)packed+global);
    if(!checked(g,cudaMemsetAsync(gs,0,16,g->compute),"quant scale reset"))return 0;
    unsigned grid=(unsigned)std::min<size_t>(4096,((size_t)rows*k+255)/256);
    quant_amax<<<grid,256,0,g->compute>>>((const __nv_bfloat16*)source,(size_t)rows*k,gs,(unsigned*)g->quant_fault->data);
    quant_finish_scale<<<1,1,0,g->compute>>>(gs,weight_global,mode);
    if(mode==H3_QUANT_FP8)
        quant_fp8_pack<<<grid,256,0,g->compute>>>((const __nv_bfloat16*)source,(unsigned char*)packed,(float*)((char*)packed+scale),gs,rows,padded,k);
    else {
        if(!checked(g,cudaMemsetAsync((char*)packed+scale,0,quant_scale_bytes(padded,k,mode),g->compute),"quant block scale padding"))return 0;
        quant_fp4_pack<<<grid,256,0,g->compute>>>((const __nv_bfloat16*)source,(unsigned char*)packed,(unsigned char*)packed+scale,gs,rows,padded,k);
    }
    if(g->quant_diagnostics)quant_diagnostics<<<grid,256,0,g->compute>>>((const __nv_bfloat16*)source,(const unsigned char*)packed,rows,k,mode,scale,global,(unsigned long long*)g->quant_fault->data);
    g->stats.direct_dispatches+=3;
    return checked(g,cudaGetLastError(),"quantize projection input/weight");
}
int h3_quant_preflight(int mode,char *error,size_t size) {
    if(!mode)return 1;
    int device=0;cudaDeviceProp prop={};
    if(mode<1||mode>2||cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&prop,device)!=cudaSuccess) {
        snprintf(error,size,"cannot query CUDA quantization device");return 0;
    }
    if(prop.major!=12||prop.minor!=0) {
        snprintf(error,size,"cuda-denoise-quant currently requires SM120 kernels (RTX 5090 / RTX PRO Blackwell) (device SM%d%d)",prop.major,prop.minor);return 0;
    }
    if(cublasLtGetVersion()<120804) {
        snprintf(error,size,"cuda-denoise-quant requires cuBLASLt 12.8 Update 1 or newer (linked %zu)",cublasLtGetVersion());return 0;
    }
    return 1;
}
static int quant_mkdir(const std::string &path) {
    if(path.empty())return 0;
    for(size_t i=1;i<=path.size();i++)if(i==path.size()||path[i]=='/') {
        std::string part=path.substr(0,i);
        if(mkdir(part.c_str(),0755)&&errno!=EEXIST)return 0;
        struct stat st;if(stat(part.c_str(),&st)||!S_ISDIR(st.st_mode))return 0;
    }
    return 1;
}
int h3_gpu_quant_configure(h3_gpu *g,int mode,const char *cache,uint64_t weights,uint64_t activations,int force_stream) {
    if(g && g->policy.active && mode!=g->policy.projection_precision)
        return h3_gpu_set_error(g,"quantization differs from captured CUDA request policy");
    if(!g||!h3_quant_preflight(mode,g->error,sizeof(g->error)))return 0;
    if(!mode)return 1; /* BF16 never creates or accesses packed-weight caches. */
    if(g->quant_mode)return h3_gpu_set_error(g,"quantized GPU context already configured");
    const char *home=getenv("HOME");
    g->quant_cache=cache?cache:std::string(home?home:".")+"/.cache/h3/denoise-quant";
    if(!quant_mkdir(g->quant_cache))return h3_gpu_set_error(g,"cannot create quantization cache %s: %s",g->quant_cache.c_str(),strerror(errno));
    g->quant_mode=mode;
    g->quant_verify=h3_quant_verify();
    const uint64_t bf16_block=2ull*(4ull*7168*5376+3ull*14336*5376);
    size_t scale,global;
    uint64_t packed_block=quant_bytes(3*7168,5376,mode,&scale,&global)+
        quant_bytes(5376,7168,mode,&scale,&global)+quant_bytes(2*14336,5376,mode,&scale,&global)+
        quant_bytes(5376,14336,mode,&scale,&global);
    if(weights%bf16_block)return h3_gpu_set_error(g,"invalid packed DiT core geometry");
    uint64_t packed=(weights/bf16_block)*packed_block;
    /* Adaptive requests require all 50 blocks. Account for the BF16 probe
     * before admission, in addition to the already allocated cache tensors. */
    if(g->policy.active&&g->policy.adaptive_cache) {
        packed+=bf16_block-packed_block;
    }
    if(force_stream&&g->weight_options.mode==H3_WEIGHTS_RESIDENT)return h3_gpu_set_error(g,"--ssd-streaming conflicts with CUDA resident weight mode");
    /* Largest conversion operand, packed stream buffer and activation/output scratch. */
    int stream=h3_gpu_plan_weights(g,packed+(1ull<<30),activations+(2ull<<30));
    if(stream<0)return 0;
    g->quant_streaming=stream!=0||force_stream;
    g->quant_diagnostics=getenv("H3_QUANT_DIAGNOSTICS")&&!strcmp(getenv("H3_QUANT_DIAGNOSTICS"),"1");
    g->quant_fault=h3_gpu_tensor_alloc(g,6,H3_GPU_U32,H3_GPU_DEVICE_ONLY);
    if(!g->quant_fault||!checked(g,cudaMemset(g->quant_fault->data,0,24),"quant finite flag"))return 0;
    H3_VERBOSE("h3cli: DiT quantization=%s recipe=%d native cuBLASLt=%zu SM120 W%dA%d BF16-output FP32-accumulation weights=%s cache=%s verification=%s\n",
        h3_quant_name(mode),h3_quant_execution_recipe(mode,g->policy.adaptive_cache,g->policy.attention),cublasLtGetVersion(),mode==1?8:4,mode==1?8:4,
        g->quant_streaming?"compressed-stream":"compressed-resident",g->quant_cache.c_str(),
        g->quant_verify?"sha256-strict":"metadata-v2 (no weight hashing)");
    if(g->policy.adaptive_cache)H3_VERBOSE("h3cli: adaptive quantization: block 0 projections BF16, blocks 1-49 %s, cache BF16, score FP32; packing recipe %d\n",h3_quant_name(mode),H3_QUANT_VERSION);
    return 1;
}
static void quant_hash(const void *data,size_t bytes,unsigned char digest[32]) {
    h3_sha256_ctx state;h3_sha256_init_fast(&state);
    auto *p=(const unsigned char*)data;
    while(bytes){size_t chunk=std::min<size_t>(bytes,1u<<30);h3_sha256_update(&state,p,(uint32_t)chunk);bytes-=chunk;p+=chunk;}
    h3_sha256_final(digest,&state);
}
static void quant_put(unsigned char *p,uint64_t v,unsigned n){for(unsigned i=0;i<n;i++)p[i]=(unsigned char)(v>>(i*8));}
static uint64_t quant_get(const unsigned char *p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;i++)v|=(uint64_t)p[i]<<(i*8);return v;}
static bool quant_io(int fd,void *data,size_t bytes,bool write) {
    auto *p=(unsigned char*)data;
    while(bytes){ssize_t n=write ? ::write(fd,p,bytes) : ::read(fd,p,bytes);if(n<0&&errno==EINTR)continue;if(n<=0)return false;p+=n;bytes-=(size_t)n;}return true;
}
static bool quant_source_unchanged(const char *path,const struct stat &before) {
    struct stat after;
    return !stat(path,&after)&&before.st_dev==after.st_dev&&before.st_ino==after.st_ino&&before.st_size==after.st_size&&
        before.st_mtim.tv_sec==after.st_mtim.tv_sec&&before.st_mtim.tv_nsec==after.st_mtim.tv_nsec&&
        before.st_ctim.tv_sec==after.st_ctim.tv_sec&&before.st_ctim.tv_nsec==after.st_ctim.tv_nsec;
}
static h3_gpu_tensor *quant_finish_weight(h3_gpu *g,std::vector<unsigned char> &packed,
    unsigned n,unsigned k,size_t scale,size_t global,bool hit,double start);
static h3_gpu_tensor *quant_load_strict(h3_gpu *g,const char *path,uint64_t offset,unsigned n,unsigned k) {
    if(!g||!g->quant_mode||!n||!k||n%128||k%128||(uint64_t)n*k>UINT32_MAX) {
        h3_gpu_set_error(g,"quantized weight shape must be nonzero and a multiple of 128");return nullptr;
    }
    double start=now();size_t scale,global,bytes=quant_bytes(n,k,g->quant_mode,&scale,&global);
    size_t source_bytes=(size_t)n*k*2;
    int fd=open(path,O_RDONLY);struct stat st;
    if(fd<0||fstat(fd,&st)||offset>(uint64_t)st.st_size||source_bytes>(uint64_t)st.st_size-offset) {
        if(fd>=0)close(fd);h3_gpu_set_error(g,"invalid quantized source range: %s",path);return nullptr;
    }
    const struct stat source_status=st;
    size_t page=(size_t)sysconf(_SC_PAGESIZE),delta=offset%page;
    void *mapping=mmap(nullptr,source_bytes+delta,PROT_READ,MAP_PRIVATE,fd,(off_t)(offset-delta));close(fd);
    if(mapping==MAP_FAILED){h3_gpu_set_error(g,"cannot map quantized source: %s",strerror(errno));return nullptr;}
    unsigned char source_hash[32];quant_hash((char*)mapping+delta,source_bytes,source_hash);
    char hex[65];for(int i=0;i<32;i++)snprintf(hex+i*2,3,"%02x",source_hash[i]);
    std::string artifact=g->quant_cache+"/v"+std::to_string(H3_QUANT_VERSION)+"-"+h3_quant_name(g->quant_mode)+"-"+std::to_string(n)+"x"+std::to_string(k)+"-"+hex+".h3q";
    int lock=open((artifact+".lock").c_str(),O_CREAT|O_RDWR,0600);
    if(lock<0||flock(lock,LOCK_EX)){if(lock>=0)close(lock);munmap(mapping,source_bytes+delta);h3_gpu_set_error(g,"cannot lock quantization artifact: %s",strerror(errno));return nullptr;}
    std::vector<unsigned char> packed;
    try {packed.resize(bytes);}catch(const std::bad_alloc&) {
        munmap(mapping,source_bytes+delta);flock(lock,LOCK_UN);close(lock);
        h3_gpu_set_error(g,"host allocation failed for %zu-byte packed weight",bytes);return nullptr;
    }
    unsigned char header[128]={},hash[32];
    bool ok=true,hit=false;
    fd=open(artifact.c_str(),O_RDONLY);
    if(fd>=0) {
        ok=!fstat(fd,&st)&&(uint64_t)st.st_size==128+bytes&&quant_io(fd,header,128,false)&&
            !memcmp(header,"H3QWT001",8)&&quant_get(header+8,4)==H3_QUANT_VERSION&&
            quant_get(header+12,4)==(unsigned)g->quant_mode&&quant_get(header+16,4)==n&&quant_get(header+20,4)==k&&
            quant_get(header+24,8)==bytes&&quant_get(header+32,8)==scale&&quant_get(header+40,8)==global&&
            !memcmp(header+48,source_hash,32)&&quant_io(fd,packed.data(),bytes,false);
        close(fd);
        if(ok){quant_hash(packed.data(),bytes,hash);ok=!memcmp(header+80,hash,32);}
        for(unsigned i=112;ok&&i<128;i++)ok=header[i]==0;
        if(!ok)h3_gpu_set_error(g,"corrupt/incompatible quantization artifact %s; remove it and prepare again",artifact.c_str());
        hit=ok;
    } else if(errno!=ENOENT) {ok=false;h3_gpu_set_error(g,"cannot read quantization artifact: %s",strerror(errno));}
    if(ok&&!hit) {
        auto *source=h3_gpu_tensor_from_bf16(g,(const uint16_t*)((char*)mapping+delta),(size_t)n*k);
        auto *destination=h3_gpu_tensor_alloc(g,bytes,H3_GPU_I8,H3_GPU_DEVICE_ONLY);
        ok=source&&destination&&checked(g,cudaMemsetAsync(destination->data,0,bytes,g->compute),"packed weight initialization")&&quant_pack(g,source->data,destination->data,n,n,k,g->quant_mode,scale,global)&&
            checked(g,cudaStreamSynchronize(g->compute),"prepare quantized weight")&&
            checked(g,cudaMemcpy(packed.data(),destination->data,bytes,cudaMemcpyDeviceToHost),"download packed artifact");
        h3_gpu_tensor_free(source);h3_gpu_tensor_free(destination);
        if(ok)ok=h3_gpu_synchronize(g);
        if(ok&&!quant_source_unchanged(path,source_status)) {
            ok=false;h3_gpu_set_error(g,"source weight changed during quantization; retry with immutable model files");
        }
        if(ok) {
            memset(header,0,sizeof(header));memcpy(header,"H3QWT001",8);
            quant_put(header+8,H3_QUANT_VERSION,4);quant_put(header+12,(unsigned)g->quant_mode,4);
            quant_put(header+16,n,4);quant_put(header+20,k,4);quant_put(header+24,bytes,8);
            quant_put(header+32,scale,8);quant_put(header+40,global,8);memcpy(header+48,source_hash,32);
            quant_hash(packed.data(),bytes,header+80);
            std::string temporary=artifact+".tmp.XXXXXX";std::vector<char> temp(temporary.begin(),temporary.end());temp.push_back(0);
            fd=mkstemp(temp.data());
            ok=fd>=0;
            if(ok){ok=quant_io(fd,header,128,true)&&quant_io(fd,packed.data(),bytes,true)&&!fsync(fd);if(close(fd))ok=false;}
            if(ok)ok=rename(temp.data(),artifact.c_str())==0;
            if(!ok){unlink(temp.data());h3_gpu_set_error(g,"cannot commit quantization artifact: %s",strerror(errno));}
        }
    }
    munmap(mapping,source_bytes+delta);flock(lock,LOCK_UN);close(lock);
    if(!ok)return nullptr;
    if(!quant_source_unchanged(path,source_status)) {
        h3_gpu_set_error(g,"source weight changed during artifact verification; retry with immutable model files");return nullptr;
    }
    return quant_finish_weight(g,packed,n,k,scale,global,hit,start);
}
static h3_gpu_tensor *quant_finish_weight(h3_gpu *g,std::vector<unsigned char> &packed,
    unsigned n,unsigned k,size_t scale,size_t global,bool hit,double start) {
    size_t bytes=packed.size();
    float gs;memcpy(&gs,packed.data()+global,4);
    if(!std::isfinite(gs)||gs<=0){h3_gpu_set_error(g,"invalid quantized weight global scale");return nullptr;}
    if(g->quant_mode==H3_QUANT_FP8) {
        float ss;memcpy(&ss,packed.data()+scale,4);
        if(ss!=gs){h3_gpu_set_error(g,"invalid FP8 weight scale");return nullptr;}
        /* Detect E4M3 NaN bytes eight at a time. Shapes are multiples of 128;
         * the zero-byte test is endian independent and retains strict loading. */
        for(size_t i=0;i<(size_t)n*k;i+=8) {
            uint64_t word;memcpy(&word,packed.data()+i,8);
            word=(word&UINT64_C(0x7f7f7f7f7f7f7f7f))^UINT64_C(0x7f7f7f7f7f7f7f7f);
            if((word-UINT64_C(0x0101010101010101))&~word&UINT64_C(0x8080808080808080)) {
                h3_gpu_set_error(g,"nonfinite FP8 weight");return nullptr;
            }
        }
    }else {
        for(size_t i=0;i<quant_scale_bytes(n,k,g->quant_mode);i++)if(!packed[scale+i]||packed[scale+i]>=127){h3_gpu_set_error(g,"invalid NVFP4 weight block scale");return nullptr;}
    }
    h3_gpu_tensor *result=nullptr;
    if(g->quant_streaming) {
        result=new(std::nothrow) h3_gpu_tensor;
        if(result){result->owner=g;result->quant_host=std::move(packed);result->dtype=H3_GPU_I8;}
    }else {
        result=h3_gpu_tensor_alloc(g,bytes,H3_GPU_I8,H3_GPU_DEVICE_ONLY);
        if(result&&!transfer(result,0,packed.data(),bytes,H3_GPU_I8,true)){h3_gpu_tensor_free(result);result=nullptr;}
    }
    if(!result)return nullptr;
    result->elements=(size_t)n*k;result->quant_mode=g->quant_mode;
    result->quant_n=n;result->quant_k=k;result->quant_scale=scale;result->quant_global=global;
    if(hit)g->quant_cache_hits++;else g->quant_cache_misses++;
    if(g->profiling)fprintf(stderr,"h3cli: quant weight %s %ux%u %s bytes=%zu time=%.6fs\n",h3_quant_name(g->quant_mode),n,k,hit?"cache-hit":"prepared",bytes,now()-start);
    return result;
}
h3_gpu_tensor *h3_gpu_quant_load(h3_gpu *g,const char *path,uint64_t offset,unsigned n,unsigned k) {
    if(!g||!g->quant_mode||!n||!k||n%128||k%128||(uint64_t)n*k>UINT32_MAX) {
        h3_gpu_set_error(g,"quantized weight shape must be nonzero and a multiple of 128");return nullptr;
    }
    if(g->quant_verify)return quant_load_strict(g,path,offset,n,k);
    double start=now();size_t scale,global,bytes=quant_bytes(n,k,g->quant_mode,&scale,&global);
    h3_quant_cache *cache=h3_quant_cache_open(g->quant_cache.c_str(),path,offset,n,k,
        g->quant_mode,bytes,scale,global,g->error,sizeof(g->error));
    if(!cache)return nullptr;
    std::vector<unsigned char> packed;
    try {packed.resize(bytes);}catch(const std::bad_alloc&) {
        h3_quant_cache_close(cache);h3_gpu_set_error(g,"host allocation failed for %zu-byte packed weight",bytes);return nullptr;
    }
    int status=h3_quant_cache_read(cache,packed.data(),g->error,sizeof(g->error));
    bool ok=status>=0;
    if(ok&&!status) {
        /* Only a cache miss opens the original tensor. Preparation, publication,
         * and future lookup all avoid hashing source and packed weight bytes. */
        size_t source_bytes=(size_t)n*k*2,page=(size_t)sysconf(_SC_PAGESIZE),delta=offset%page;
        int fd=open(cache->source,O_RDONLY);struct stat st;
        void *mapping=MAP_FAILED;
        if(fd>=0&&!fstat(fd,&st)&&st.st_dev==cache->stamp.st_dev&&st.st_ino==cache->stamp.st_ino&&
           st.st_size==cache->stamp.st_size&&h3_quant_cache_unchanged(cache,g->error,sizeof(g->error)))
            mapping=mmap(nullptr,source_bytes+delta,PROT_READ,MAP_PRIVATE,fd,(off_t)(offset-delta));
        if(fd>=0)close(fd);
        if(mapping==MAP_FAILED) {
            h3_gpu_set_error(g,"cannot map metadata-cache source: %s",path);ok=false;
        }else {
            auto *source=h3_gpu_tensor_from_bf16(g,(const uint16_t*)((char*)mapping+delta),(size_t)n*k);
            auto *destination=h3_gpu_tensor_alloc(g,bytes,H3_GPU_I8,H3_GPU_DEVICE_ONLY);
            ok=source&&destination&&checked(g,cudaMemsetAsync(destination->data,0,bytes,g->compute),"packed weight initialization")&&
                quant_pack(g,source->data,destination->data,n,n,k,g->quant_mode,scale,global)&&
                checked(g,cudaStreamSynchronize(g->compute),"prepare quantized weight")&&
                checked(g,cudaMemcpy(packed.data(),destination->data,bytes,cudaMemcpyDeviceToHost),"download packed artifact");
            h3_gpu_tensor_free(source);h3_gpu_tensor_free(destination);munmap(mapping,source_bytes+delta);
            if(ok)ok=h3_gpu_synchronize(g)&&h3_quant_cache_write(cache,packed.data(),g->error,sizeof(g->error));
        }
    }
    if(ok)ok=h3_quant_cache_unchanged(cache,g->error,sizeof(g->error));
    h3_quant_cache_close(cache);
    return ok?quant_finish_weight(g,packed,n,k,scale,global,status==1,start):nullptr;
}
static int quant_arena(h3_gpu *g,h3_gpu_tensor **tensor,size_t bytes,h3_gpu_dtype dtype) {
    if(*tensor&&(*tensor)->bytes>=bytes)return 1;
    h3_gpu_tensor_free(*tensor);*tensor=h3_gpu_tensor_alloc(g,(bytes+item_size(dtype)-1)/item_size(dtype),dtype,H3_GPU_DEVICE_ONLY);
    return *tensor!=nullptr;
}
static int quant_gemm(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,const h3_gpu_tensor *w,
                       unsigned rows,unsigned k,unsigned n) {
    if(!launch_ready(g)||!w||!rows||w->owner!=g||w->quant_k!=k||w->quant_n!=n||
        !h3_gpu_require_bf16(g,in,(size_t)rows*k,"quant input")||!h3_gpu_require_bf16(g,out,(size_t)rows*n,"quant output"))return 0;
    unsigned padded=(unsigned)quant_up(rows,128);size_t scale,global,bytes=quant_bytes(padded,k,w->quant_mode,&scale,&global);
    if(!quant_arena(g,&g->quant_activation,bytes,H3_GPU_I8))return 0;
    const void *weight=w->data;
    if(!w->quant_host.empty()) {
        if(!quant_arena(g,&g->quant_stream,w->quant_host.size(),H3_GPU_I8))return 0;
        if(!checked(g,cudaMemcpyAsync(g->quant_stream->data,w->quant_host.data(),w->quant_host.size(),cudaMemcpyHostToDevice,g->compute),"packed weight stream"))return 0;
        g->stats.h2d_bytes+=w->quant_host.size();g->stats.streamed_bytes+=w->quant_host.size();weight=g->quant_stream->data;
    }else weight=tensor_pointer(w);
    void *input=g->quant_activation->data,*output=tensor_pointer(out);
    if(padded!=rows){if(!quant_arena(g,&g->quant_output,(size_t)padded*n*2,H3_GPU_BF16))return 0;output=g->quant_output->data;}
    profile_scope timing(g,0);
    {profile_scope conversion(g,7);
     if(!quant_pack(g,tensor_pointer(in),input,rows,padded,k,w->quant_mode,scale,global,(const float*)((const char*)weight+w->quant_global)))return 0;}
    auto key=std::make_tuple(padded,k,n,w->quant_mode);
    auto found=g->quant_plans.find(key);
    if(found==g->quant_plans.end()) {
        if(g->quant_plans.size()>=128)g->quant_plans.clear();
        auto plan=std::make_unique<cuda_gemm_plan>();auto &p=*plan;
        cublasOperation_t trans=CUBLAS_OP_T;cublasLtPointerMode_t pointer=CUBLASLT_POINTER_MODE_DEVICE;
        cudaDataType_t dtype=w->quant_mode==1?CUDA_R_8F_E4M3:CUDA_R_4F_E2M1;
        cublasLtMatmulMatrixScale_t scaling=w->quant_mode==1?CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F:CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
#define QLT(call) do{if(status==CUBLAS_STATUS_SUCCESS){status=(call);if(status!=CUBLAS_STATUS_SUCCESS)fprintf(stderr,"h3cli: quant descriptor %s failed status=%d\n",#call,(int)status);}}while(0)
        QLT(cublasLtMatmulDescCreate(&p.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        QLT(cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        QLT(cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_POINTER_MODE,&pointer,sizeof(pointer)));
        QLT(cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&scaling,sizeof(scaling)));
        QLT(cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&scaling,sizeof(scaling)));
        const void *weight_scale=(const char*)weight+w->quant_scale,*activation_scale=(const char*)input+scale;
        QLT(cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&weight_scale,sizeof(weight_scale)));
        QLT(cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&activation_scale,sizeof(activation_scale)));
        QLT(cublasLtMatrixLayoutCreate(&p.a,dtype,k,n,k));QLT(cublasLtMatrixLayoutCreate(&p.b,dtype,k,padded,k));
        QLT(cublasLtMatrixLayoutCreate(&p.c,CUDA_R_16BF,n,padded,n));
        cublasLtMatmulPreference_t preference=nullptr;QLT(cublasLtMatmulPreferenceCreate(&preference));
        QLT(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&g->workspace_bytes,sizeof(g->workspace_bytes)));
        cublasLtMatmulHeuristicResult_t candidates[8]={};int count=0;
        QLT(cublasLtMatmulAlgoGetHeuristic(g->lt,p.operation,p.a,p.b,p.c,p.c,preference,8,candidates,&count));
        if(preference)cublasLtMatmulPreferenceDestroy(preference);
        int chosen=-1;
        for(int i=0;i<count;i++){cublasLtMatmulHeuristicResult_t check={};if(candidates[i].state==CUBLAS_STATUS_SUCCESS&&
            cublasLtMatmulAlgoCheck(g->lt,p.operation,p.a,p.b,p.c,p.c,&candidates[i].algo,&check)==CUBLAS_STATUS_SUCCESS&&check.state==CUBLAS_STATUS_SUCCESS&&check.workspaceSize<=g->workspace_bytes){chosen=i;break;}}
        if(status!=CUBLAS_STATUS_SUCCESS||chosen<0)return h3_gpu_set_error(g,"native %s GEMM %ux%ux%u unavailable: cuBLASLt status=%d candidates=%d",h3_quant_name(w->quant_mode),padded,n,k,(int)status,count);
        p.algorithm=candidates[chosen].algo;
        if(g->profiling)fprintf(stderr,"h3cli: native %s GEMM plan rows=%u N=%u K=%u candidates=%d workspace=%zu\n",h3_quant_name(w->quant_mode),padded,n,k,count,g->workspace_bytes);
        found=g->quant_plans.emplace(key,std::move(plan)).first;
#undef QLT
    }
    auto &p=*found->second;const void *ws=(const char*)weight+w->quant_scale,*as=(const char*)input+scale;
    const float *gs=(const float*)((char*)input+global);
    cublasStatus_t status=cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&ws,sizeof(ws));
    if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&as,sizeof(as));
    if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmul(g->lt,p.operation,gs+1,weight,p.a,input,p.b,gs+2,output,p.c,output,p.c,&p.algorithm,g->workspace,g->workspace_bytes,g->compute);
    if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,"native %s GEMM failed: cuBLASLt status=%d",h3_quant_name(w->quant_mode),(int)status);
    if(padded!=rows&&!checked(g,cudaMemcpyAsync(out->data,output,(size_t)rows*n*2,cudaMemcpyDeviceToDevice,g->compute),"quant output tail"))return 0;
    g->quant_calls++;g->stats.linear_dispatches++;return launch_status(g,"native quantized projection");
}
#else
int h3_quant_preflight(int mode,char *error,size_t size){if(!mode)return 1;snprintf(error,size,"cuda-denoise-quant requires CUDA 12.8 Update 1 or newer");return 0;}
int h3_gpu_quant_configure(h3_gpu *g,int mode,const char*,uint64_t,uint64_t,int){return h3_quant_preflight(mode,g->error,sizeof(g->error));}
h3_gpu_tensor *h3_gpu_quant_load(h3_gpu*,const char*,uint64_t,unsigned,unsigned){return nullptr;}
static int quant_gemm(h3_gpu *g,h3_gpu_tensor*,const h3_gpu_tensor*,const h3_gpu_tensor*,unsigned,unsigned,unsigned){return h3_gpu_set_error(g,"native quantization unavailable in this build");}
#endif
