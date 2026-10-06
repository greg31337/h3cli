#include "src/log.h"
/* A separate cuBLAS 13 handle for the pinned reference recipe. Other GPU
 * contexts keep their own linked cuBLAS handles and symbol resolution. */
#include <dlfcn.h>
struct h3_sglang_blas {
    void *library=nullptr;
    cublasHandle_t handle=nullptr;
    decltype(&cublasCreate) create=nullptr;
    decltype(&cublasDestroy) destroy=nullptr;
    decltype(&cublasSetStream) stream=nullptr;
    decltype(&cublasSetMathMode) math=nullptr;
    decltype(&cublasSetWorkspace) workspace=nullptr;
    decltype(&cublasGetVersion) get_version=nullptr;
    decltype(&cublasSgemm) sgemm=nullptr;
    decltype(&cublasSgemmStridedBatched) sgemm_batched=nullptr;
    using gemm_fn=cublasStatus_t(*)(cublasHandle_t,cublasOperation_t,cublasOperation_t,
        int,int,int,const void*,const void*,cudaDataType_t,int,const void*,cudaDataType_t,int,
        const void*,void*,cudaDataType_t,int,cublasComputeType_t,cublasGemmAlgo_t);
    gemm_fn gemm=nullptr;
    cublasLtHandle_t lt=nullptr;
#define H3_SG_LT_FUNCTIONS(X) \
    X(cublasLtCreate) X(cublasLtDestroy) X(cublasLtMatmulDescCreate) \
    X(cublasLtMatmulDescDestroy) X(cublasLtMatmulDescSetAttribute) \
    X(cublasLtMatrixLayoutCreate) X(cublasLtMatrixLayoutDestroy) \
    X(cublasLtMatmulPreferenceCreate) X(cublasLtMatmulPreferenceDestroy) \
    X(cublasLtMatmulPreferenceSetAttribute) X(cublasLtMatmulAlgoGetHeuristic) X(cublasLtMatmul)
#define H3_SG_LT_FIELD(name) decltype(&name) name##_ref=nullptr;
    H3_SG_LT_FUNCTIONS(H3_SG_LT_FIELD)
#undef H3_SG_LT_FIELD
    using plan_key=std::tuple<unsigned,unsigned,unsigned,int,unsigned,unsigned,unsigned,unsigned,unsigned>;
    std::map<plan_key,cublasLtMatmulAlgo_t> algorithms;
    int version=0;
    ~h3_sglang_blas(){if(lt && cublasLtDestroy_ref)cublasLtDestroy_ref(lt);if(handle && destroy)destroy(handle);if(library)dlclose(library);}
    bool initialize(cudaStream_t execution,void *scratch,size_t bytes,char *error,size_t size) {
        const char *path=getenv("H3_SGLANG_CUBLAS_LIBRARY");
        if(!path || !*path)path="libcublas.so.13";
        library=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
        if(!library){snprintf(error,size,"SGLang reference needs cuBLAS 13: %s; set H3_SGLANG_CUBLAS_LIBRARY to its absolute library path",dlerror());return false;}
#define H3_SG_SYMBOL(field,name) do { field=reinterpret_cast<decltype(field)>(dlsym(library,name)); \
    if(!field){snprintf(error,size,"reference cuBLAS missing %s",name);return false;} } while(0)
        H3_SG_SYMBOL(create,"cublasCreate_v2");H3_SG_SYMBOL(destroy,"cublasDestroy_v2");
        H3_SG_SYMBOL(stream,"cublasSetStream_v2");H3_SG_SYMBOL(math,"cublasSetMathMode");
        H3_SG_SYMBOL(workspace,"cublasSetWorkspace_v2");H3_SG_SYMBOL(get_version,"cublasGetVersion_v2");
        H3_SG_SYMBOL(gemm,"cublasGemmEx");
        H3_SG_SYMBOL(sgemm,"cublasSgemm_v2");
        H3_SG_SYMBOL(sgemm_batched,"cublasSgemmStridedBatched");
#define H3_SG_LT_SYMBOL(name) H3_SG_SYMBOL(name##_ref,#name);
        H3_SG_LT_FUNCTIONS(H3_SG_LT_SYMBOL)
#undef H3_SG_LT_SYMBOL
#undef H3_SG_LT_FUNCTIONS
#undef H3_SG_SYMBOL
        if(create(&handle)!=CUBLAS_STATUS_SUCCESS || stream(handle,execution)!=CUBLAS_STATUS_SUCCESS ||
           math(handle,CUBLAS_DEFAULT_MATH)!=CUBLAS_STATUS_SUCCESS ||
           workspace(handle,scratch,bytes)!=CUBLAS_STATUS_SUCCESS ||
           get_version(handle,&version)!=CUBLAS_STATUS_SUCCESS || version!=130101 ||
           cublasLtCreate_ref(&lt)!=CUBLAS_STATUS_SUCCESS) {
            snprintf(error,size,"reference recipe requires pinned cuBLAS 130101 (found %d)",version);return false;
        }
        H3_VERBOSE("h3cli: SGLang reference cuBLAS=%d workspace=%zu partial-reduction=BF16-permitted\n",version,bytes);
        return true;
    }
    static unsigned alignment(const void *ptr) {
        unsigned result=256;while((uintptr_t)ptr%result)result/=2;return result;
    }
    /* Pinned torch addmm: one Lt heuristic, bias epilogue, FP32 compute,
     * and pointer-alignment preferences. This handle/cache is reference-only. */
    cublasStatus_t biased(void *out,const void *input,const void *weight,const void *bias,
        unsigned rows,unsigned k,unsigned n,cudaDataType_t dtype,void *scratch,size_t bytes,cudaStream_t execution,unsigned heuristic_rows=0) {
        /* The installed oracle gives Lt 1 MiB, independently of the 32 MiB
         * ordinary cuBLAS workspace. A larger heuristic budget changes K
         * splitting and BF16 reduction on the real condition projection. */
        bytes=std::min<size_t>(bytes,1u<<20);
        cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,hb=nullptr,hc=nullptr;
        cublasLtMatmulPreference_t pref=nullptr;cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
        cublasOperation_t trans=CUBLAS_OP_T,normal=CUBLAS_OP_N;
        cublasLtEpilogue_t ep=CUBLASLT_EPILOGUE_BIAS;
        float alpha=1,beta=0;
        unsigned aa=alignment(weight),ba=alignment(input),ca=alignment(out),da=alignment(bias);
        plan_key key={rows,k,n,(int)dtype,aa,ba,ca,da,heuristic_rows};cublasLtMatmulAlgo_t algo={};
#define H3_SG_LT(call) do {status=(call);if(status!=CUBLAS_STATUS_SUCCESS)goto done;}while(0)
        H3_SG_LT(cublasLtMatmulDescCreate_ref(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        H3_SG_LT(cublasLtMatmulDescSetAttribute_ref(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        H3_SG_LT(cublasLtMatmulDescSetAttribute_ref(op,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
        H3_SG_LT(cublasLtMatmulDescSetAttribute_ref(op,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep)));
        H3_SG_LT(cublasLtMatmulDescSetAttribute_ref(op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)));
        H3_SG_LT(cublasLtMatrixLayoutCreate_ref(&a,dtype,k,n,k));
        H3_SG_LT(cublasLtMatrixLayoutCreate_ref(&b,dtype,k,rows,k));
        H3_SG_LT(cublasLtMatrixLayoutCreate_ref(&c,dtype,n,rows,n));
        {
            auto found=algorithms.find(key);
            if(found!=algorithms.end())algo=found->second;
            else {
                H3_SG_LT(cublasLtMatmulPreferenceCreate_ref(&pref));
                H3_SG_LT(cublasLtMatmulPreferenceSetAttribute_ref(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
                H3_SG_LT(cublasLtMatmulPreferenceSetAttribute_ref(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,&aa,sizeof(aa)));
                H3_SG_LT(cublasLtMatmulPreferenceSetAttribute_ref(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,&ba,sizeof(ba)));
                H3_SG_LT(cublasLtMatmulPreferenceSetAttribute_ref(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,&ca,sizeof(ca)));
                H3_SG_LT(cublasLtMatmulPreferenceSetAttribute_ref(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,&da,sizeof(da)));
                cublasLtMatmulHeuristicResult_t result={};int count=0;
                if(heuristic_rows && heuristic_rows!=rows) {
                    H3_SG_LT(cublasLtMatrixLayoutCreate_ref(&hb,dtype,k,heuristic_rows,k));
                    H3_SG_LT(cublasLtMatrixLayoutCreate_ref(&hc,dtype,n,heuristic_rows,n));
                }
                H3_SG_LT(cublasLtMatmulAlgoGetHeuristic_ref(lt,op,a,hb?hb:b,hc?hc:c,hc?hc:c,pref,1,&result,&count));
                if(!count){status=CUBLAS_STATUS_NOT_SUPPORTED;goto done;}
                algo=result.algo;if(algorithms.size()>=128)algorithms.clear();algorithms.emplace(key,algo);
            }
        }
        H3_SG_LT(cublasLtMatmul_ref(lt,op,&alpha,weight,a,input,b,&beta,out,c,out,c,&algo,scratch,bytes,execution));
    done:
        if(pref)cublasLtMatmulPreferenceDestroy_ref(pref);
        if(hb)cublasLtMatrixLayoutDestroy_ref(hb);if(hc)cublasLtMatrixLayoutDestroy_ref(hc);
        if(a)cublasLtMatrixLayoutDestroy_ref(a);if(b)cublasLtMatrixLayoutDestroy_ref(b);if(c)cublasLtMatrixLayoutDestroy_ref(c);
        if(op)cublasLtMatmulDescDestroy_ref(op);
#undef H3_SG_LT
        return status;
    }
};
