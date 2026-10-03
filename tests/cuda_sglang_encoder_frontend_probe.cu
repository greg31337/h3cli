/* Bounded frontend replay of the first differing video encoder convolution.
 * Reads one retained real-operation fixture; never scans model weights. */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cudnn_frontend.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#define CU(x) do{auto s=(x);if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));return 1;}}while(0)
#define FE(x) do{auto s=(x);if(!s.is_good()){fprintf(stderr,"%s\n",s.get_message().c_str());return 1;}}while(0)
static bool read(const char *root,const char *name,std::vector<float>&v){char p[4096];snprintf(p,sizeof(p),"%s/%s.f32",root,name);FILE*f=fopen(p,"rb");if(!f)return false;bool ok=fread(v.data(),4,v.size(),f)==v.size()&&fgetc(f)==EOF;fclose(f);return ok;}
int main(int argc,char**argv){
 if(argc!=2)return 2;namespace fe=cudnn_frontend;
 int64_t n=1,ci=256,co=256,d=3,hgt=65,width=65,k=3,od=1,oh=32,ow=32;
 std::vector<float>x(n*ci*d*hgt*width),w(co*ci*k*k*k),y(n*co*od*oh*ow),target(y.size()),bias(co);
 if(!read(argv[1],"input",x)||!read(argv[1],"weight",w)||!read(argv[1],"output",target)||!read(argv[1],"bias",bias))return 2;
 void *xp,*wp,*yp;CU(cudaMalloc(&xp,x.size()*4));CU(cudaMalloc(&wp,w.size()*4));CU(cudaMalloc(&yp,y.size()*4));
 CU(cudaMemcpy(xp,x.data(),x.size()*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(wp,w.data(),w.size()*4,cudaMemcpyHostToDevice));
 cudnnHandle_t h;if(cudnnCreate(&h)!=CUDNN_STATUS_SUCCESS)return 1;
 auto g=std::make_shared<fe::graph::Graph>();g->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT).set_intermediate_data_type(fe::DataType_t::FLOAT);
 auto X=g->tensor(fe::graph::Tensor_attributes().set_uid(1).set_dim({n,ci,d,hgt,width}).set_stride({ci*d*hgt*width,d*hgt*width,hgt*width,width,1}));
 auto W=g->tensor(fe::graph::Tensor_attributes().set_uid(2).set_dim({co,ci,k,k,k}).set_stride({ci*k*k*k,k*k*k,k*k,k,1}));
 auto Y=g->conv_fprop(X,W,fe::graph::Conv_fprop_attributes().set_padding({0,0,0}).set_stride({2,2,2}).set_dilation({1,1,1}));
 Y->set_output(true).set_uid(3).set_dim({n,co,od,oh,ow}).set_stride({co*od*oh*ow,od*oh*ow,oh*ow,ow,1});
 FE(g->validate());FE(g->build_operation_graph(h));FE(g->create_execution_plans({fe::HeurMode_t::A,fe::HeurMode_t::FALLBACK}));
 g->deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC}).deselect_workspace_greater_than(512u<<20);
 std::unordered_map<int64_t,void*> ptr={{1,xp},{2,wp},{3,yp}};
 for(int i=0;i<g->get_execution_plan_count();i++){
  auto s=g->build_plan_at_index(h,i);if(!s.is_good()){printf("plan %d unsupported\n",i);continue;}
  std::string name;FE(g->get_plan_name_at_index(i,name));printf("plan %d name %s\n",i,name.c_str());
  int64_t bytes=0;FE(g->get_workspace_size_plan_at_index(i,bytes));if(bytes<0||bytes>(512u<<20))continue;
  void *scratch=nullptr;if(bytes)CU(cudaMalloc(&scratch,bytes));s=g->execute_plan_at_index(h,ptr,scratch,i);
  if(s.is_good()){CU(cudaMemcpy(y.data(),yp,y.size()*4,cudaMemcpyDeviceToHost));size_t diff=0;double dd=0,bb=0,max_abs=0;
   for(size_t j=0;j<y.size();j++){float v=y[j]+bias[(j/(od*oh*ow))%co];double d=(double)v-target[j];diff+=v!=target[j];dd+=d*d;bb+=(double)target[j]*target[j];max_abs=std::max(max_abs,std::abs(d));}
   printf("plan %d workspace %lld different %zu max_abs %.17g relative_l2 %.17g\n",i,(long long)bytes,diff,max_abs,sqrt(dd/bb));
  }else printf("plan %d execution failed: %s\n",i,s.get_message().c_str());
  if(scratch)CU(cudaFree(scratch));
 }
 CU(cudaFree(xp));CU(cudaFree(wp));CU(cudaFree(yp));cudnnDestroy(h);return 0;
}
