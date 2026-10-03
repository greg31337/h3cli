/* Standalone linked-cuDNN diagnostic. Inputs are bounded retained FP32
 * operation captures; no model loading and no renderer dispatch changes. */
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#define CU(x) do{cudaError_t s=(x);if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));return 1;}}while(0)
#define DN(x) do{cudnnStatus_t s=(x);if(s!=CUDNN_STATUS_SUCCESS){fprintf(stderr,"%s\n",cudnnGetErrorString(s));return 1;}}while(0)
static bool read(const char *root,const char *suffix,std::vector<float>&data){char p[4096];snprintf(p,sizeof(p),"%s/%s.bin",root,suffix);FILE*f=fopen(p,"rb");if(!f)return false;bool ok=fread(data.data(),4,data.size(),f)==data.size()&&fgetc(f)==EOF;fclose(f);return ok;}
int main(int argc,char**argv){
 if(argc!=4&&argc!=14){fprintf(stderr,"usage: conv_probe CAPTURE dec_in_proj|decoder.conv_pre OUTDIR\n");return 2;}
 bool pre=std::string(argv[2])=="decoder.conv_pre";
 if(argc==4&&!pre&&std::string(argv[2])!="dec_in_proj")return 2;
 int n=2,ci=pre?2048:32,co=pre?1024:2048,len=207,k=pre?7:1,pad=k/2,stride=1,dilation=1,transpose=0,groups=1;
 if(argc==14){n=atoi(argv[4]);ci=atoi(argv[5]);co=atoi(argv[6]);len=atoi(argv[7]);k=atoi(argv[8]);stride=atoi(argv[9]);pad=atoi(argv[10]);dilation=atoi(argv[11]);transpose=atoi(argv[12]);groups=atoi(argv[13]);}
 if(n<1||ci<1||co<1||len<1||k<1||stride<1||pad<0||dilation<1||groups<1||ci%groups||co%groups)return 2;
 int olen=transpose?(len-1)*stride-2*pad+k:(len+2*pad-dilation*(k-1)-1)/stride+1;
 if(olen<1||(size_t)n*ci*len>(128u<<20)||(size_t)n*co*olen>(128u<<20)||(size_t)ci*co*k/groups>(128u<<20))return 2;
 std::vector<float>x(n*ci*len),w(co*ci*k/groups),y(n*co*olen),target(y.size()),bias(co,0);char name[128];
 snprintf(name,sizeof(name),"%s.input",argv[2]);if(!read(argv[1],name,x))return 2;
 snprintf(name,sizeof(name),"%s.weight",argv[2]);if(!read(argv[1],name,w))return 2;
 if(argc==14){snprintf(name,sizeof(name),"%s.output",argv[2]);if(!read(argv[1],name,target))return 2;snprintf(name,sizeof(name),"%s.bias",argv[2]);read(argv[1],name,bias);}
 void *xp=nullptr,*wp=nullptr,*yp=nullptr;CU(cudaMalloc(&xp,x.size()*4));CU(cudaMalloc(&wp,w.size()*4));CU(cudaMalloc(&yp,y.size()*4));
 CU(cudaMemcpy(xp,x.data(),x.size()*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(wp,w.data(),w.size()*4,cudaMemcpyHostToDevice));
 cudnnHandle_t h;DN(cudnnCreate(&h));cudnnTensorDescriptor_t xd,yd;cudnnFilterDescriptor_t wd;cudnnConvolutionDescriptor_t cd;
 DN(cudnnCreateTensorDescriptor(&xd));DN(cudnnCreateTensorDescriptor(&yd));DN(cudnnCreateFilterDescriptor(&wd));DN(cudnnCreateConvolutionDescriptor(&cd));
 DN(cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,n,ci,len,1));DN(cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,n,co,olen,1));
 DN(cudnnSetFilter4dDescriptor(wd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,transpose?ci:co,(transpose?co:ci)/groups,k,1));DN(cudnnSetConvolution2dDescriptor(cd,pad,0,stride,1,dilation,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));DN(cudnnSetConvolutionMathType(cd,CUDNN_FMA_MATH));
 DN(cudnnSetConvolutionGroupCount(cd,groups));
 float alpha=1,beta=0;printf("cuDNN %zu\n",cudnnGetVersion());
 cudnnConvolutionFwdAlgoPerf_t choices[8];int found=0;
 if(!transpose)DN(cudnnGetConvolutionForwardAlgorithm_v7(h,xd,wd,cd,yd,8,&found,choices));
 for(int j=0;j<found;j++)printf("heuristic %d algorithm %d status %d math %d determinism %d workspace %zu time %.6f\n",j,int(choices[j].algo),int(choices[j].status),int(choices[j].mathType),int(choices[j].determinism),choices[j].memory,choices[j].time);
 for(int a=0;a<(transpose?6:8);a++){size_t bytes=0;auto algo=(cudnnConvolutionFwdAlgo_t)a;auto s=transpose?cudnnGetConvolutionBackwardDataWorkspaceSize(h,wd,xd,cd,yd,(cudnnConvolutionBwdDataAlgo_t)a,&bytes):cudnnGetConvolutionForwardWorkspaceSize(h,xd,wd,cd,yd,algo,&bytes);if(s!=CUDNN_STATUS_SUCCESS||bytes>(512u<<20))continue;
  void*scratch=nullptr;if(bytes)CU(cudaMalloc(&scratch,bytes));s=transpose?cudnnConvolutionBackwardData(h,&alpha,wd,wp,xd,xp,cd,(cudnnConvolutionBwdDataAlgo_t)a,scratch,bytes,&beta,yd,yp):cudnnConvolutionForward(h,&alpha,xd,xp,wd,wp,cd,algo,scratch,bytes,&beta,yd,yp);
  if(s==CUDNN_STATUS_SUCCESS){CU(cudaMemcpy(y.data(),yp,y.size()*4,cudaMemcpyDeviceToHost));char file[4096];
   if(argc==4){snprintf(file,sizeof(file),"%s/algorithm-%d.f32",argv[3],a);FILE*f=fopen(file,"wbx");if(!f||fwrite(y.data(),4,y.size(),f)!=y.size()||fclose(f))return 2;}
   else {double dd=0,bb=0,aa=0,ab=0,max_abs=0;size_t diff=0;bool finite=true;
    for(size_t j=0;j<y.size();j++){float actual=y[j]+bias[(j/olen)%co];double delta=(double)actual-target[j];diff+=actual!=target[j];finite=finite&&std::isfinite(actual);dd+=delta*delta;bb+=(double)target[j]*target[j];aa+=(double)actual*actual;ab+=(double)actual*target[j];max_abs=std::max(max_abs,std::abs(delta));}
    snprintf(file,sizeof(file),"%s/algorithm-%d.json",argv[3],a);FILE*f=fopen(file,"wx");if(!f)return 2;
    fprintf(f,"{\"algorithm\":%d,\"workspace\":%zu,\"elements\":%zu,\"different\":%zu,\"finite\":%s,\"max_abs\":%.17g,\"relative_l2\":%.17g,\"cosine\":%.17g}\n",a,bytes,y.size(),diff,finite?"true":"false",max_abs,sqrt(dd/std::max(bb,1e-30)),ab/std::max(sqrt(aa*bb),1e-30));if(fclose(f))return 2;
   }
   printf("algorithm %d workspace %zu complete\n",a,bytes);}
  else printf("algorithm %d status %d\n",a,int(s));if(scratch)CU(cudaFree(scratch));
 }
 DN(cudnnDestroyConvolutionDescriptor(cd));DN(cudnnDestroyFilterDescriptor(wd));DN(cudnnDestroyTensorDescriptor(xd));DN(cudnnDestroyTensorDescriptor(yd));DN(cudnnDestroy(h));CU(cudaFree(xp));CU(cudaFree(wp));CU(cudaFree(yp));return 0;
}
