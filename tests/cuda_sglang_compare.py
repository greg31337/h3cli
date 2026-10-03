#!/usr/bin/env python3
"""Compare retained oracle tensors to native captures, without using the GPU.

This is a diagnostic boundary comparison. Teacher-forced and free-running
measurements are explicitly separate; no MP4 parity follows from this report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def oracle(root, name):
    meta=json.loads((root/(name+".json")).read_text())
    if not isinstance(meta,dict):return np.array(meta)
    path=root/(name+".bin")
    raw=path.read_bytes()
    if len(raw)!=meta["bytes"] or hashlib.sha256(raw).hexdigest()!=meta["sha256"]:
        raise ValueError(f"stale or corrupt capture: {name}")
    dtype=meta["dtype"].removeprefix("torch.")
    if dtype=="bfloat16":
        x=(np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
    else:x=np.frombuffer(raw,dtype=np.dtype(dtype))
    return x.reshape(meta["shape"])


def native(root,name,dtype="<f4"):
    x=np.fromfile(root/name,dtype=dtype)
    if dtype=="<u2":return (x.astype(np.uint32)<<16).view(np.float32)
    return x


def native_block(root, step, name, rows):
    widths={"qkv":3*7168,"query":7168,"key":7168,"value":7168,"attention":7168}
    width=widths.get(name,5376)
    x=native(root,f"step-{step:03d}.block-0.{name}.bf16","<u2")
    count=rows*width
    # Before logical-size capture was added, the two aliases exported their
    # entire QKV backing allocation. Only its documented live prefix is valid.
    if name in ("attention","mod-mlp") and x.size==rows*3*7168:x=x[:count]
    if x.size!=count:raise ValueError(f"{name}: {x.size} elements, expected {count}")
    return x.reshape(rows,width)


def metric(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return {"compatible":False,"native_shape":list(a.shape),"oracle_shape":list(b.shape)}
    if not np.isfinite(a).all() or not np.isfinite(b).all():return {"compatible":True,"finite":False}
    aa=bb=ab=dd=0.;max_abs=0.;different=0;near_zero_abs=0.
    a=a.reshape(-1);b=b.reshape(-1)
    for start in range(0,a.size,1<<20):
        x=a[start:start+(1<<20)].astype(np.float64);y=b[start:start+(1<<20)].astype(np.float64)
        d=x-y;aa+=float(x@x);bb+=float(y@y);ab+=float(x@y);dd+=float(d@d)
        max_abs=max(max_abs,float(np.max(np.abs(d),initial=0)))
        different+=int(np.count_nonzero(d))
        near_zero_abs=max(near_zero_abs,float(np.max(np.abs(d[np.abs(y)<1e-6]),initial=0)))
    return dict(compatible=True,finite=True,elements=a.size,exact=different==0,different=different,
                max_abs=max_abs,near_zero_max_abs=near_zero_abs,rmse=float(np.sqrt(dd/max(a.size,1))),
                relative_l2=float(np.sqrt(dd/max(bb,1e-30))),cosine=1.0 if aa==0 and bb==0 else float(ab/max(np.sqrt(aa*bb),1e-30)),
                reference_rms=float(np.sqrt(bb/max(a.size,1))))


def pack_video(x):
    # Canonical [24,T,30,40] -> [T,15,20,24,2,2].
    return x.reshape(24,-1,15,2,20,2).transpose(1,2,4,0,3,5).reshape(-1,96)


def pack_audio(x):return x.reshape(32,2,-1).transpose(1,2,0).reshape(-1,32)


def oracle_target(root, name, modality):
    """Remove only the oracle's explicitly recorded immutable condition rows."""
    value = oracle(root, name)
    prefix = json.loads((root / f'positive.{modality}_target_start.json').read_text())
    if not isinstance(prefix, int) or prefix < 0 or prefix >= value.shape[0]:
        raise ValueError('invalid oracle target range')
    return value[prefix:]


def compare(args):
    o=args.oracle/"capture";n=args.native/"capture";results=[]
    contract_path=Path(__file__).with_name("cuda_sglang_contract.json")
    contract=json.loads(contract_path.read_text())
    def add(label,native_fn,oracle_fn,kind="arithmetic"):
        try:row=dict(name=label,kind=kind,**metric(native_fn(),oracle_fn()))
        except (OSError,ValueError,KeyError) as e:row=dict(name=label,kind=kind,error=str(e))
        if row.get("compatible") and row.get("finite"):
            if kind=="exact":row["passed"]=row["exact"]
            else:
                gate=contract["gates"]["trajectory" if kind=="trajectory" else "full_velocity" if kind=="velocity" else "operation"]
                row["passed"]=row["relative_l2"]<=gate["relative_l2_max"] and row["cosine"]>=gate["cosine_min"]
                if "absolute_max_per_reference_rms" in gate:
                    row["absolute_limit"]=gate["absolute_floor"]+gate["absolute_max_per_reference_rms"]*row["reference_rms"]
                    row["passed"] &= row["max_abs"]<=row["absolute_limit"]
        else:row["passed"]=False
        row["passed"]=bool(row["passed"])
        results.append(row)
    add("tokens",lambda:native(n,"tokens.u32","<u4"),lambda:oracle(o,"text.encode_ids.args.0").reshape(-1),"exact")
    text_rows=len(native(n,"tokens.u32","<u4"))
    qwen={"input":"block.input.0","input-norm":"input_layernorm.output",
          "q-projection":"self_attn.q_proj.output","k-projection":"self_attn.k_proj.output",
          "v-projection":"self_attn.v_proj.output","q-norm":"self_attn.q_norm.output",
          "k-norm":"self_attn.k_norm.output","projection":"self_attn.o_proj.output",
          "post-norm":"post_attention_layernorm.output","mlp-gate":"mlp.gate_proj.output",
          "mlp-up":"mlp.up_proj.output","mlp-down":"mlp.down_proj.output","output":"block.output"}
    for layer in (range(50) if args.qwen_all else (0,25,49)):
        for key,value in qwen.items():
            add(f"qwen-{layer}.{key}",lambda k=key,l=layer:native(n,f"qwen-{l}.{k}.bf16","<u2").reshape(text_rows,-1),
                lambda v=value,l=layer:oracle(o,f"qwen-{l}.{v}").reshape(text_rows,-1))
            if key=='k-norm':
                for key,value in (("query","q"),("key","k"),("attention","output")):
                    add(f"qwen-{layer}.{key}",lambda k=key,l=layer:native(n,f"qwen-{l}.{k}.bf16","<u2").reshape(text_rows,-1),
                        lambda v=value,l=layer:oracle(o,f"qwen-{l}.attention.{v}")[0].transpose(1,0,2).reshape(text_rows,-1))
    add("text",lambda:native(n,"text.bf16","<u2").reshape(-1,5120),lambda:oracle(o,"text.prompt_embeds.0"))
    if args.refiner:
        def ref_native(name,width=5376):return native(n,'step-000.block-0.'+name+'.bf16','<u2').reshape(text_rows,width)
        add('condition-proj',lambda:ref_native('condition-proj'),lambda:oracle(o,'refiner.condition_proj.output.0').reshape(text_rows,5376))
        for layer in (0,1):
            for key,value,width in [('input','input.0',5376),('norm1','norm1.output',5376),('projection','attn.out_proj.output.0',5376),('residual','norm2.input.0',5376),('norm2','norm2.output',5376),('output','output',5376)]:
                add(f'refiner-{layer}.{key}',lambda k=key,l=layer,w=width:ref_native(f'refiner-{l}.{k}',w),
                    lambda v=value,l=layer,w=width:oracle(o,f'refiner.token_refiner.blocks.{l}.{v}').reshape(text_rows,w))
                if key=='norm1':
                    add(f'refiner-{layer}.qkv',lambda l=layer:ref_native(f'refiner-{l}.qkv',3*7168).reshape(text_rows,56,3,128).transpose(0,2,1,3).reshape(text_rows,-1),
                        lambda l=layer:oracle(o,f'refiner.token_refiner.blocks.{l}.attn.qkv_proj.output.0').reshape(text_rows,-1))
                    for key,value in (('query','q'),('key','k'),('value','v'),('attention','output')):
                        add(f'refiner-{layer}.{key}',lambda k=key,l=layer:ref_native(f'refiner-{l}.{k}',7168).reshape(text_rows,56,128),
                            lambda v=value,l=layer:oracle(o,f'refiner-{l}.attention.{v}')[0].transpose(1,0,2))
        add('refined-text',lambda:ref_native('refined-text'),lambda:oracle(o,'refiner.final').reshape(text_rows,5376))
    for modality,pack in (("video",pack_video),("audio",pack_audio)):
        add("initial-"+modality,lambda m=modality,p=pack:p(native(n,f"initial-{m}.f32")),
            lambda m=modality:oracle_target(o,f"initial_{m}_rows",m),"exact")
        add("sigma-"+modality,lambda m=modality:native(n,f"sigmas-{m}.f32"),
            lambda m=modality:oracle(o,f"sigmas_{m}").astype(np.float32),"exact")
    positions=native(n,"positions.f64","<f8").reshape(-1,3);rows=len(positions)
    # Native has no trailing padding. Remove only the explicitly recorded,
    # separate padding segment, and require the live segment length to match.
    bounds=oracle(o,"positive.static_kwargs.packed_seq_params.cu_seqlens_q").reshape(-1)
    live=int(bounds[1])
    if live!=rows:raise ValueError(f"packed live rows disagree: {rows} vs {live}")
    add("positions",lambda:positions.astype(np.float32),
        lambda:oracle(o,"positive.static_kwargs.img_position_ids").reshape(-1,3)[:live],"exact")
    mapping={"input":"blocks.0.input.0","mod-attention":"blocks.0.attn.qkv_proj.input.0",
             "projection":"blocks.0.attn.out_proj.output.0","attention-residual":"blocks.0.norm2.input.0",
             "mod-mlp":"blocks.0.mlp.fc1.input.0","output":"blocks.0.output"}
    for step in args.steps:
        for key,value in mapping.items():
            if key not in ("input","mod-attention"):continue
            add(f"step-{step}.{key}",lambda k=key,s=step:native_block(n,s,k,rows),
                lambda v=value,s=step:oracle(o,f"step-{s:03d}.{v}")[:live])
        add(f"step-{step}.qkv",lambda s=step:native_block(n,s,"qkv",rows).reshape(rows,56,3,128).transpose(0,2,1,3).reshape(rows,-1),
            lambda s=step:oracle(o,f"step-{s:03d}.blocks.0.attn.qkv_proj.output.0")[:live])
        for key,value in (("query","q"),("key","k"),("value","v"),("attention","output")):
            add(f"step-{step}.{key}",lambda k=key,s=step:native_block(n,s,k,rows).reshape(rows,56,128),
                lambda v=value,s=step:oracle(o,f"step-{s:03d}.attention.{v}")[0].transpose(1,0,2))
        for key,value in mapping.items():
            if key in ("input","mod-attention"):continue
            add(f"step-{step}.{key}",lambda k=key,s=step:native_block(n,s,k,rows),
                lambda v=value,s=step:oracle(o,f"step-{s:03d}.{v}")[:live])
        for index,(modality,pack) in enumerate((("video",pack_video),("audio",pack_audio))):
            add(f"step-{step}.{modality}-velocity",lambda m=modality,p=pack,s=step:p(native(args.native/"steps",f"step-{s+1:03d}-{m}-velocity.f32")),
                lambda j=index,s=step:oracle_target(o,f"step-{s:03d}.velocity.{j}","audio") if j==1 else oracle(o,f"step-{s:03d}.velocity.{j}"),"velocity")
    for path in sorted((args.native/"steps").glob("step-*-video-latent.f32")):
        step=int(path.name.split("-")[1])-1
        for modality,pack in (("video",pack_video),("audio",pack_audio)):
            add(f"step-{step}.{modality}-state",lambda m=modality,p=pack,s=step:p(native(args.native/"steps",f"step-{s+1:03d}-{m}-latent.f32")),
                lambda m=modality,s=step:oracle_target(o,f"step-{s:03d}.{m}",m),"trajectory")
    report={"comparison":"teacher_forced_boundary_diagnostic" if any(x.startswith("H3_TEST_NATIVE_TEACHER_DIR=") for x in json.loads((args.native/"command.json").read_text()).get("overrides",[])) else "free_running_boundary_diagnostic","native":str(args.native),"oracle":str(args.oracle),
            "live_rows":rows,"results":results,"parity_qualified":False,
            "contract_sha256":hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "first_failing_boundary":next((r["name"] for r in results if not r["passed"]),None)}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for row in results:print(json.dumps(row))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle",type=Path,required=True);p.add_argument("--native",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,nargs="+",default=[0])
    p.add_argument("--qwen-all",action="store_true");p.add_argument("--refiner",action="store_true")
    compare(p.parse_args())
