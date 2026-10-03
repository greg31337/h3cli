/* Small real-operation cuDNN diagnostic; no production dispatch changes. */
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#define CU(x) do{auto s=(x);if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));return 1;}}while(0)
#define DN(x) do{auto s=(x);if(s!=CUDNN_STATUS_SUCCESS){fprintf(stderr,"%s\n",cudnnGetErrorString(s));return 1;}}while(0)
static bool read(const char *root,const char *name,std::vector<float>&v){char p[4096];snprintf(p,sizeof(p),"%s/%s.f32",root,name);FILE*f=fopen(p,"rb");if(!f)return false;bool ok=fread(v.data(),4,v.size(),f)==v.size()&&fgetc(f)==EOF;fclose(f);return ok;}
int main(int argc,char**argv){
 if(argc!=3&&argc!=14)return 2;
 int n=1,ci=atoi(argv[2]),co=128,d=3,h=258,w=258,kd=3,kh=3,kw=3,sd=1,sh=1,sw=1;
 if(argc==14){n=atoi(argv[2]);ci=atoi(argv[3]);d=atoi(argv[4]);h=atoi(argv[5]);w=atoi(argv[6]);co=atoi(argv[7]);kd=atoi(argv[8]);kh=atoi(argv[9]);kw=atoi(argv[10]);sd=atoi(argv[11]);sh=atoi(argv[12]);sw=atoi(argv[13]);}
 if(n<1||ci<1||co<1||kd<1||kh<1||kw<1||sd<1||sh<1||sw<1||d<kd||h<kh||w<kw)return 2;
 int od=(d-kd)/sd+1,oh=(h-kh)/sh+1,ow=(w-kw)/sw+1;
 std::vector<float>x((size_t)n*ci*d*h*w),weight((size_t)co*ci*kd*kh*kw),y((size_t)n*co*od*oh*ow),target(y.size()),bias(co);
 if(!read(argv[1],"input",x)||!read(argv[1],"weight",weight)||!read(argv[1],"output",target)||!read(argv[1],"bias",bias))return 2;
 void *xp,*wp,*yp;CU(cudaMalloc(&xp,x.size()*4));CU(cudaMalloc(&wp,weight.size()*4));CU(cudaMalloc(&yp,y.size()*4));CU(cudaMemcpy(xp,x.data(),x.size()*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(wp,weight.data(),weight.size()*4,cudaMemcpyHostToDevice));
 cudnnHandle_t handle;DN(cudnnCreate(&handle));cudnnTensorDescriptor_t xd,yd;cudnnFilterDescriptor_t wd;cudnnConvolutionDescriptor_t cd;
 DN(cudnnCreateTensorDescriptor(&xd));DN(cudnnCreateTensorDescriptor(&yd));DN(cudnnCreateFilterDescriptor(&wd));DN(cudnnCreateConvolutionDescriptor(&cd));
 int xs[5]={n,ci,d,h,w},xt[5]={ci*d*h*w,d*h*w,h*w,w,1},ys[5]={n,co,od,oh,ow},yt[5]={co*od*oh*ow,od*oh*ow,oh*ow,ow,1},ws[5]={co,ci,kd,kh,kw},pad[3]={0,0,0},step[3]={sd,sh,sw},dilation[3]={1,1,1};
 DN(cudnnSetTensorNdDescriptor(xd,CUDNN_DATA_FLOAT,5,xs,xt));DN(cudnnSetTensorNdDescriptor(yd,CUDNN_DATA_FLOAT,5,ys,yt));DN(cudnnSetFilterNdDescriptor(wd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,5,ws));DN(cudnnSetConvolutionNdDescriptor(cd,3,pad,step,dilation,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
 printf("cuDNN %zu\n",cudnnGetVersion());float alpha=1,beta=0;
 for(int math=0;math<4;math++){DN(cudnnSetConvolutionMathType(cd,(cudnnMathType_t)math));for(int a=0;a<8;a++){
  size_t bytes=0;auto s=cudnnGetConvolutionForwardWorkspaceSize(handle,xd,wd,cd,yd,(cudnnConvolutionFwdAlgo_t)a,&bytes);if(s!=CUDNN_STATUS_SUCCESS||bytes>(512u<<20))continue;
  void *scratch=nullptr;if(bytes)CU(cudaMalloc(&scratch,bytes));s=cudnnConvolutionForward(handle,&alpha,xd,xp,wd,wp,cd,(cudnnConvolutionFwdAlgo_t)a,scratch,bytes,&beta,yd,yp);
  if(s==CUDNN_STATUS_SUCCESS){CU(cudaMemcpy(y.data(),yp,y.size()*4,cudaMemcpyDeviceToHost));size_t diff=0;double dd=0,bb=0,mx=0;
   for(size_t i=0;i<y.size();i++){float v=y[i]+bias[(i/(od*oh*ow))%co];double delta=(double)v-target[i];diff+=v!=target[i];dd+=delta*delta;bb+=(double)target[i]*target[i];mx=std::max(mx,std::abs(delta));}
   printf("math %d algorithm %d workspace %zu different %zu max_abs %.17g relative_l2 %.17g\n",math,a,bytes,diff,mx,sqrt(dd/bb));fflush(stdout);
  }if(scratch)CU(cudaFree(scratch));
 }}
 DN(cudnnDestroyConvolutionDescriptor(cd));DN(cudnnDestroyFilterDescriptor(wd));DN(cudnnDestroyTensorDescriptor(xd));DN(cudnnDestroyTensorDescriptor(yd));DN(cudnnDestroy(handle));CU(cudaFree(xp));CU(cudaFree(wp));CU(cudaFree(yp));return 0;
}
