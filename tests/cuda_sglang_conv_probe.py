#!/usr/bin/env python3
"""Bounded diagnostic: pinned cuDNN convolution candidates on real vision pixels."""
import argparse,ctypes as C,json,os
from pathlib import Path
import numpy as np
from cuda_sglang_compare import oracle,metric

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--audio',choices=('dec_in_proj','decoder.conv_pre'));p.add_argument('--capture',type=Path,required=True);p.add_argument('--model',type=Path,default=Path('models/MiniMax-H3'));p.add_argument('--cudnn-library',default=os.environ.get('H3_SGLANG_CUDNN_LIBRARY','libcudnn.so.9'));a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
 import torch
 from safetensors import safe_open
 if a.audio:
  captured=a.capture
  x=torch.from_numpy(oracle(captured,a.audio+'.input').copy()).cuda().unsqueeze(-1)
  w=torch.from_numpy(oracle(captured,a.audio+'.weight').copy()).cuda().unsqueeze(-1)
  with safe_open(a.model/'FL2VA/audio_vae/model.safetensors',framework='pt',device='cpu') as f:b=f.get_tensor(a.audio+'.bias').cuda()
  yshape=(x.shape[0],w.shape[0],x.shape[2],1);pad=[w.shape[2]//2,0];stride=[1,1];dtype=0;target=oracle(captured,a.audio+'.output').reshape(-1)
 else:
  root=a.model/'FL2VA/text_encoder';mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];v={}
  for k in mapping:
   if not k.endswith(('visual.patch_embed.proj.weight','visual.patch_embed.proj.bias')):continue
   with safe_open(root/mapping[k],framework='pt',device='cpu') as f:v[k.rsplit('.',1)[-1]]=f.get_tensor(k).cuda()
  captured=a.capture
  x=torch.from_numpy(oracle(captured,'patch_embed.input').copy()).cuda().bfloat16().reshape(-1,3,2,16,16);w=v['weight'];b=v['bias'];target=oracle(captured,'patch_embed.output').reshape(-1)
  yshape=(len(x),1152,1,1,1);pad=[0,0,0];stride=[2,16,16];dtype=9
 lib=C.CDLL(a.cudnn_library)
 P=C.c_void_p;I=C.c_int;Z=C.c_size_t
 def fn(name,args):
  f=getattr(lib,name);f.argtypes=args;f.restype=I;return f
 def check(s):
  if s:raise RuntimeError('cuDNN status '+str(s))
 create=fn('cudnnCreate',[C.POINTER(P)]);setstream=fn('cudnnSetStream',[P,P]);h=P();check(create(C.byref(h)));check(setstream(h,P(torch.cuda.current_stream().cuda_stream)))
 tensor=fn('cudnnCreateTensorDescriptor',[C.POINTER(P)]);set_tensor=fn('cudnnSetTensorNdDescriptor',[P,I,I,C.POINTER(I),C.POINTER(I)])
 def desc(shape):
  d=P();check(tensor(C.byref(d)));strides=[int(np.prod(shape[i+1:])) for i in range(len(shape))];check(set_tensor(d,dtype,len(shape),(I*len(shape))(*shape),(I*len(shape))(*strides)));return d
 xd=desc(tuple(x.shape));yd=desc(yshape);wd=P();cd=P();dims=len(x.shape);spatial=dims-2
 check(fn('cudnnCreateFilterDescriptor',[C.POINTER(P)])(C.byref(wd)));check(fn('cudnnSetFilterNdDescriptor',[P,I,I,I,C.POINTER(I)])(wd,dtype,0,dims,(I*dims)(*w.shape)))
 check(fn('cudnnCreateConvolutionDescriptor',[C.POINTER(P)])(C.byref(cd)));check(fn('cudnnSetConvolutionNdDescriptor',[P,I,C.POINTER(I),C.POINTER(I),C.POINTER(I),I,I])(cd,spatial,(I*spatial)(*pad),(I*spatial)(*stride),(I*spatial)(*[1]*spatial),1,0))
 math=fn('cudnnSetConvolutionMathType',[P,I]);workspace=fn('cudnnGetConvolutionForwardWorkspaceSize',[P,P,P,P,P,I,C.POINTER(Z)])
 forward=fn('cudnnConvolutionForward',[P,P,P,P,P,P,P,I,P,Z,P,P,P]);alpha=C.c_float(1);beta=C.c_float(0);rows=[]
 for mode in ((3,) if a.audio else (0,1,2)):
  check(math(cd,mode))
  for algo in range(8):
   size=Z();status=workspace(h,xd,wd,cd,yd,algo,C.byref(size))
   if status or size.value>512*1024**2:continue
   scratch=torch.empty(max(1,size.value),device='cuda',dtype=torch.uint8);y=torch.empty(yshape,device='cuda',dtype=x.dtype)
   status=forward(h,C.byref(alpha),xd,P(x.data_ptr()),wd,P(w.data_ptr()),cd,algo,P(scratch.data_ptr()),size,C.byref(beta),yd,P(y.data_ptr()))
   if status:continue
   z=y+b.reshape(1,-1,*[1]*spatial);m=dict(math=mode,algorithm=algo,workspace=size.value,**metric(z.float().cpu().numpy().reshape(-1),target));rows.append(m);print(json.dumps(m),flush=True)
 (a.out/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
 for name,d in [('Tensor',xd),('Tensor',yd),('Filter',wd),('Convolution',cd)]:check(fn('cudnnDestroy'+name+'Descriptor',[P])(d))
 check(fn('cudnnDestroy',[P])(h))
if __name__=='__main__':main()
