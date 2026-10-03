#!/usr/bin/env python3
"""Pinned reference soundtrack encoder, with bounded operation captures."""
import argparse, hashlib, json
from pathlib import Path
from types import SimpleNamespace


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('model','input','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 import torch
 from safetensors.torch import load_file
 from sglang.multimodal_gen.runtime.server_args import set_global_server_args
 from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEConfig
 from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3AudioVAE
 from sglang.multimodal_gen.runtime.loader.utils import adopt_plain_weight_norm_state
 from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import minimax_h3_encode_reference_audio_rows
 set_global_server_args(SimpleNamespace(attention_backend='torch_sdpa',attention_backend_config=None,kv_gather_degree=1))
 config=json.loads((a.model/'config.json').read_text());c=MiniMaxH3AudioVAEConfig();c.update_model_arch(config)
 with torch.device('cpu'):model=MiniMaxH3AudioVAE(c)
 weights=load_file(str(a.model/'model.safetensors'),device='cpu');adopt_plain_weight_norm_state(model,weights)
 model.load_state_dict(weights,strict=True,assign=True);del weights
 model.eval().requires_grad_(False);model.encoder.to('cuda');model.pre_block.to('cuda');model.mean_proj.to('cuda')
 records={};used=0
 def save(name,t):
  nonlocal used
  if name in records:return
  if isinstance(t,(tuple,list)):
   if not t:return
   t=t[0]
  if not isinstance(t,torch.Tensor):return
  value=t.detach().contiguous().cpu();raw=value.reshape(-1).view(torch.uint8).numpy().tobytes();used+=len(raw)
  if used>768*1024**2:raise ValueError('audio encoder capture exceeds 768 MiB')
  (a.out/(name+'.bin')).write_bytes(raw)
  meta=dict(dtype=str(t.dtype),shape=list(t.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());records[name]=meta
  (a.out/(name+'.json')).write_text(json.dumps(meta)+'\n')
 for name,module in model.named_modules():
  if not (name in ('encoder.block.0','encoder.block.1','encoder.block.2','encoder.block.3','encoder.block.4','encoder.block.5','encoder.block.6','encoder.block.7','encoder','pre_block','mean_proj') or
          name in {f'encoder.block.1.block.0.block.{i}' for i in range(4)} or
          name in ('pre_block.norm1','pre_block.norm3','pre_block.proj','pre_block.attn','pre_block.attn.attn',
                   'pre_block.attn.proj','pre_block.norm2','pre_block.mlp','pre_block.mlp.norm',
                   'pre_block.mlp.w0','pre_block.mlp.w1','pre_block.mlp.w2')):continue
  def pre(m,args,name=name):
   save(name+'.input',args)
   if name=='pre_block.attn.attn':
    for label,tensor in zip(('q','k','v'),args):save(name+'.'+label,tensor)
  def post(m,args,out,name=name):save(name+'.output',out)
  module.register_forward_pre_hook(pre);module.register_forward_hook(post)
 result=minimax_h3_encode_reference_audio_rows(model,str(a.input),c.arch_config,material_chain='video.reference_preserve',max_duration_seconds=124/24)
 save('rows',result['rows']);torch.cuda.synchronize()
 report=dict(kind='pinned SGLang audio encoder diagnostic',torch=torch.__version__,records=records,ref_audio_t=result['ref_audio_t'],duration_seconds=result['duration_seconds'])
 (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(a.out,flush=True)
if __name__=='__main__':main()
