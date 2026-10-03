"""Read-only tensor snapshots of the installed oracle; no package files changed.

The import hook wraps loaded modules, so spawned workers retain instrumentation
without importing torch before SGLang configures its runtime. All saves preserve
dtype. Instrumented runs are excluded from performance qualification.
"""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(os.environ.get("H3_SGLANG_CAPTURE", "."))
LIMIT = 8 * 1024**3
used = 0
step = -1
selected = {0}
scope = ""
prefix = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3."


def save(name, value):
    global used
    import torch
    ROOT.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    path = ROOT / (name + ".json")
    if path.exists():
        return
    if isinstance(value, torch.Tensor):
        count=value.numel()*value.element_size()
        if used+count>LIMIT:
            raise RuntimeError("oracle capture exceeds explicit 8 GiB budget")
        tensor=value.detach().contiguous().cpu()
        raw=tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        with (ROOT/(name+".bin")).open("xb") as f:f.write(raw)
        meta=dict(dtype=str(value.dtype),shape=list(value.shape),stride=list(value.stride()),
                  bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        path.write_text(json.dumps(meta,indent=2)+"\n");used+=count
    elif isinstance(value,dict):
        for key,item in value.items():save(name+"."+str(key),item)
    elif isinstance(value,(tuple,list)):
        if all(isinstance(v,(float,int,str,bool)) or v is None for v in value):
            path.write_text(json.dumps(value)+"\n")
        else:
            for i,item in enumerate(value):save(name+f".{i}",item)
    elif value is None or isinstance(value,(float,int,str,bool)):
        path.write_text(json.dumps(value)+"\n")


def stage_hook(module, clsname, label):
    cls=getattr(module,clsname)
    old=cls.forward
    @functools.wraps(old)
    def wrapped(self,batch,*args,**kwargs):
        encoder=getattr(self,"text_encoder",None) if label=="text" else None
        encode=getattr(encoder,"encode_ids",None)
        handles=[]
        if encoder is not None and not os.environ.get('H3_SGLANG_CAPTURE_TRAJECTORY_ONLY') and (not os.environ.get('H3_TEST_SGLANG_INPUTS_ONLY') or
                                    os.environ.get('H3_SGLANG_CAPTURE_QWEN_INPUTS')):
            for name,child in encoder.named_modules():
                layers=r"[0-9]+" if os.environ.get("H3_SGLANG_CAPTURE_QWEN_ALL")=="1" else r"0|25|49"
                match=re.fullmatch(r"model.language_model.layers.("+layers+r")(.*)",name)
                if not match or match[2] not in ("",".input_layernorm",".post_attention_layernorm",".self_attn.q_proj",".self_attn.k_proj",".self_attn.v_proj",".self_attn.q_norm",".self_attn.k_norm",".self_attn.o_proj",".mlp.gate_proj",".mlp.up_proj",".mlp.down_proj"):continue
                label_name=f"qwen-{int(match[1])}."+(match[2].lstrip(".") or "block")
                def before(child,a,k,name=label_name):
                    global scope
                    if name.endswith(".block"):
                        scope=name.split(".")[0];save(name+".input",a)
                def after(child,a,out,name=label_name):
                    global scope
                    save(name+".output",out)
                    if name.endswith(".block"):scope=""
                handles.append(child.register_forward_pre_hook(before,with_kwargs=True))
                handles.append(child.register_forward_hook(after))
        if encode is not None:
            @functools.wraps(encode)
            def encode_ids(*a,**kw):
                save("text.encode_ids.args",a);save("text.encode_ids.kwargs",kw)
                return encode(*a,**kw)
            encoder.encode_ids=encode_ids
        try:out=old(self,batch,*args,**kwargs)
        finally:
            if encode is not None:encoder.encode_ids=encode
            for handle in handles:handle.remove()
        save(label+".extra",batch.extra)
        save(label+".prompt_embeds",getattr(batch,"prompt_embeds",None))
        return out
    cls.forward=wrapped


def loop_hook(module):
    old=module.minimax_h3_denoise_loop
    @functools.wraps(old)
    def wrapped(**kw):
        global selected,step,scope
        n=len(kw["sigmas_video"])-1
        selection=os.environ.get("H3_SGLANG_CAPTURE_STEPS","0")
        selected=set() if selection=='none' else {int(x) for x in selection.split(",")}
        if not selected.issubset(range(n)):raise ValueError("invalid capture step")
        for name in ("sigmas_video","sigmas_audio","initial_video_rows","initial_audio_rows","keyframe_cond_rows","audio_ref_rows"):
            save(name,kw.get(name))
        fields=("static_kwargs","seq_len","img_pos","audio_pos","video_target_start","audio_target_start",
                "update_mask","audio_update_mask","img_cond_seq_idx","img_target_seq_idx",
                "audio_target_seq_idx","audio_ref_seq_idx","cond_row_idx","audio_ref_row_idx")
        save("positive",{k:v for k,v in vars(kw["positive"]).items() if k in fields})
        if os.environ.get("H3_TEST_SGLANG_INPUTS_ONLY"):
            save('input_capture_complete',True)
            raise RuntimeError('test-only reference input capture complete; no denoising executed')
        model=kw["model"];forward=kw.get("model_forward");callback=kw.get("on_step")
        handles=[]
        if hasattr(model,"named_modules"):
            names={"blocks.0","blocks.0.norm1","blocks.0.attn.qkv_proj","blocks.0.attn.out_proj",
                   "blocks.0.norm2","blocks.0.mlp.fc1","blocks.0.mlp.fc2","blocks.25","blocks.49",
                   "time_embedder.proj_in","time_embedder.proj_out","blocks.0.adaln_proj.linear",
                   "final_layer.norm","final_layer.audio_out","final_layer.video_out"}
            if os.environ.get('H3_TEST_SGLANG_BLOCK_ONLY'):
                names={name for name in names if name.startswith('blocks.0')}
            for name,child in model.named_modules():
                if name not in names:continue
                def before(child,args,kwargs,name=name):
                    global scope
                    if step in selected:
                        if name=="blocks.0":
                            scope="block0";save(f"step-{step:03d}.{name}.input",args)
                            save(f"step-{step:03d}.{name}.kwargs",{k:v for k,v in kwargs.items() if k in ("adaln_params","combined_indices")})
                        elif name in ("blocks.0.attn.qkv_proj","blocks.0.mlp.fc1","blocks.0.norm2",
                                      "time_embedder.proj_in","time_embedder.proj_out","blocks.0.adaln_proj.linear","final_layer.audio_out"):
                            save(f"step-{step:03d}.{name}.input",args)
                def after(child,args,out,name=name):
                    global scope
                    if step in selected and not (os.environ.get('H3_TEST_SGLANG_BLOCK_ONLY') and name=='blocks.0.mlp.fc1'):
                        save(f"step-{step:03d}.{name}.output",out)
                    if name=="blocks.0":scope=""
                    if name=="blocks.0" and os.environ.get('H3_TEST_SGLANG_BLOCK_ONLY'):
                        save('first_block_capture_complete',True)
                        raise RuntimeError('test-only first block capture complete; no sampler update executed')
                handles.append(child.register_forward_pre_hook(before,with_kwargs=True))
                handles.append(child.register_forward_hook(after))
        def capture_forward(model,fk,i):
            global step
            step=i
            if i in selected:save(f"step-{i:03d}.forward_kwargs",{k:v for k,v in fk.items() if k in ("x","audio_x","unique_timesteps","inverse_indices","block_combined_indices")})
            out=forward(model,fk,i) if forward else model(**fk)
            if i in selected or os.environ.get('H3_SGLANG_CAPTURE_VELOCITIES')=='1':save(f"step-{i:03d}.velocity",out)
            return out
        def capture_step(i,video,audio):
            save(f"step-{i:03d}.video",video);save(f"step-{i:03d}.audio",audio)
            if callback:callback(i,video,audio)
        kw["model_forward"]=capture_forward;kw["on_step"]=capture_step
        try:return old(**kw)
        finally:
            for handle in handles:handle.remove()
            step=-1;scope=""
    module.minimax_h3_denoise_loop=wrapped


def attention_hook(module):
    old=module.scaled_dot_product_attention
    @functools.wraps(old)
    def wrapped(query,key,value,*args,**kwargs):
        q,k,v=query,key,value
        capture=(scope=="block0" and step in selected) or scope.startswith(("qwen-","refiner-"))
        name=scope if scope.startswith(("qwen-","refiner-")) else f"step-{step:03d}"
        if capture:
            save(name+".attention.q",q);save(name+".attention.k",k);save(name+".attention.v",v)
            save(name+".attention.options",kwargs)
        out=old(q,k,v,*args,**kwargs)
        if capture:save(name+".attention.output",out)
        return out
    module.scaled_dot_product_attention=wrapped


def refiner_hook(module):
    cls=module.MiniMaxH3DiTModel;old=cls.refine_prompt_embeds
    @functools.wraps(old)
    def wrapped(self,*args,**kwargs):
        global scope
        if os.environ.get('H3_SGLANG_CAPTURE_TRAJECTORY_ONLY'):
            result=old(self,*args,**kwargs)
            if os.environ.get('H3_TEST_SGLANG_BLOCK_ONLY'):save('refiner.final',result)
            return result
        if os.environ.get('H3_TEST_SGLANG_INPUTS_ONLY'):
            result=old(self,*args,**kwargs);save('refiner.final',result);return result
        handles=[]
        for name,child in self.named_modules():
            if not (name=="condition_proj" or name.startswith("token_refiner.")):continue
            if not (name.endswith(("norm1","norm2","qkv_proj","out_proj","fc1","fc2","final_norm")) or name in ("condition_proj","token_refiner.blocks.0","token_refiner.blocks.1")):continue
            def before(child,a,name=name):
                global scope
                if name in ("token_refiner.blocks.0","token_refiner.blocks.1"):scope="refiner-"+name[-1]
                save("refiner."+name+".input",a)
            def after(child,a,out,name=name):
                global scope
                save("refiner."+name+".output",out)
                if name in ("token_refiner.blocks.0","token_refiner.blocks.1"):scope=""
            handles.append(child.register_forward_pre_hook(before));handles.append(child.register_forward_hook(after))
        try:
            result=old(self,*args,**kwargs);save("refiner.final",result);return result
        finally:
            for handle in handles:handle.remove()
            scope=""
    cls.refine_prompt_embeds=wrapped


def varlen_hook(module):
    cls=module.SDPAImpl
    old=cls.forward_varlen
    @functools.wraps(old)
    def wrapped(self,q,k,v,**kwargs):
        if scope=="block0" and step in selected:
            save(f"step-{step:03d}.attention.segments",kwargs)
            save(f"step-{step:03d}.attention.scale",self.softmax_scale)
        return old(self,q,k,v,**kwargs)
    cls.forward_varlen=wrapped


HOOKS={
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3":refiner_hook,
    prefix+"stages.text_encoding":lambda m:stage_hook(m,"MiniMaxH3TextEncodingStage","text"),
    prefix+"stages.latent_preparation":lambda m:stage_hook(m,"MiniMaxH3LatentPreparationStage","latents"),
    prefix+"stages.timestep_preparation":lambda m:stage_hook(m,"MiniMaxH3TimestepPreparationStage","timesteps"),
    prefix+"denoise_loop":loop_hook,
    "torch.nn.functional":attention_hook,
    "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa":varlen_hook,
}


class Loader(importlib.abc.Loader):
    def __init__(self,original,name):self.original,self.name=original,name
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module);HOOKS[self.name](module)
    def __getattr__(self,key):return getattr(self.original,key)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in HOOKS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec and spec.loader:spec.loader=Loader(spec.loader,fullname)
        return spec


def install():sys.meta_path.insert(0,Finder())
