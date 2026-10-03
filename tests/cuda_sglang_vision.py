#!/usr/bin/env python3
"""Replay only the installed SGLang vision tower on retained real input pixels."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 import torch
 from safetensors import safe_open
 from sglang.srt.runtime_context import get_parallel
 from sglang.multimodal_gen.runtime.server_args import set_global_server_args
 from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
 from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision import Qwen3VLVisionTransformer
 from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
 from cuda_sglang_compare import oracle
 from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
 init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method='file://'+str((a.out/'distributed-init').resolve()),backend='nccl')
 initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
 set_global_server_args(SimpleNamespace(attention_backend='torch_sdpa',attention_backend_config=None,kv_gather_degree=1))
 cfg=Qwen3VLVisionConfig(**json.loads((a.model/'config.json').read_text())['vision_config'])
 with get_parallel().override(tp_rank=0,tp_size=1):
  with torch.device('cpu'):model=Qwen3VLVisionTransformer(cfg)
 mapping=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map'];weights={}
 for filename in sorted({file for key,file in mapping.items() if key.startswith(('visual.','model.visual.'))}):
  with safe_open(a.model/filename,framework='pt',device='cpu') as f:
   for key in f.keys():
    name=key.removeprefix('model.')
    if not name.startswith('visual.'):continue
    name=name.removeprefix('visual.').replace('.attn.qkv.','.attn.qkv_proj.')
    weights[name]=f.get_tensor(key)
 model.load_state_dict(weights,strict=True,assign=True);del weights;model.eval().requires_grad_(False).to('cuda')
 records={};used=0
 def save(name,value):
  nonlocal used
  if name in records:return
  if isinstance(value,(tuple,list)):value=value[0]
  if not isinstance(value,torch.Tensor):return
  t=value.detach().contiguous().cpu();raw=t.reshape(-1).view(torch.uint8).numpy().tobytes();used+=len(raw)
  if used>512*1024**2:raise ValueError('vision capture exceeds 512 MiB')
  (a.out/(name+'.bin')).write_bytes(raw);j=dict(dtype=str(t.dtype),shape=list(t.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());records[name]=j;(a.out/(name+'.json')).write_text(json.dumps(j)+'\n')
 handles=[]
 for name,module in model.named_modules():
  if not (name in ('patch_embed','merger') or name.startswith('deepstack_merger_list.') and name.count('.')==1 or name.startswith('blocks.0.') or name in ('blocks.0','blocks.8','blocks.16','blocks.24','blocks.26')):continue
  def pre(m,args,name=name):save(name+'.input',args)
  def post(m,args,out,name=name):save(name+'.output',out)
  handles.extend((module.register_forward_pre_hook(pre),module.register_forward_hook(post)))
 original=model._interpolate_position_embeddings
 def interpolate(grid):
  out=original(grid);save('position',out);return out
 model._interpolate_position_embeddings=interpolate
 first_attn=model.blocks[0].attn
 old_attention=first_attn._packed_attention
 def attention(q,k,v,metadata):
  for name,value in [('query',q),('key',k),('value',v)]:save('attention.'+name,value)
  result=old_attention(q,k,v,metadata);save('attention.output',result);return result
 first_attn._packed_attention=attention
 def rotary_inputs(m,args,kwargs):
  cos,sin=kwargs['position_embeddings'];save('rope.cos',cos);save('rope.sin',sin)
 first_attn.register_forward_pre_hook(rotary_inputs,with_kwargs=True)
 x=torch.from_numpy(oracle(a.capture,'text.encode_ids.kwargs.pixel_values').copy()).to('cuda')
 grid=torch.from_numpy(oracle(a.capture,'text.encode_ids.kwargs.image_grid_thw').copy()).to('cuda')
 with torch.inference_mode(),set_forward_context(current_timestep=0,attn_metadata=None):out=model(x,grid)
 save('merged',out.pooler_output)
 for i,value in enumerate(out.deepstack_features):save('deepstack-'+str(i),value)
 (a.out/'result.json').write_text(json.dumps(dict(implementation='installed SGLang vision tower',torch=torch.__version__,records=records),indent=2)+'\n')
 torch.distributed.destroy_process_group()
 print('vision tower capture complete',a.out)
if __name__=='__main__':main()
