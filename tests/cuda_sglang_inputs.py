#!/usr/bin/env python3
"""Compare conditioned input captures before spending full denoising runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cuda_sglang_compare import oracle, native, metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('oracle',type=Path);p.add_argument('native',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();o=a.oracle/'capture';n=a.native/'capture';results=[]
    path=Path(__file__).with_name('cuda_sglang_contract.json');gate=json.loads(path.read_text())['gates']['operation']
    def add(name,left,right,exact=False):
        try:r=dict(name=name,**metric(left(),right()))
        except (OSError,ValueError,KeyError) as exc:r=dict(name=name,error=str(exc))
        r['passed']=bool(r.get('exact')) if exact else bool(r.get('finite') and r.get('compatible') and
            r['relative_l2']<=gate['relative_l2_max'] and r['cosine']>=gate['cosine_min'] and
            r['max_abs']<=gate['absolute_floor']+gate['absolute_max_per_reference_rms']*r['reference_rms'])
        results.append(r)
    add('tokens',lambda:native(n,'tokens.u32','<u4'),lambda:oracle(o,'text.encode_ids.args.0').reshape(-1),True)
    add('text',lambda:native(n,'text.bf16','<u2').reshape(-1,5120),lambda:oracle(o,'text.prompt_embeds.0'))
    for modality in ('video','audio'):
        add('sigma-'+modality,lambda m=modality:native(n,'sigmas-'+m+'.f32'),
            lambda m=modality:oracle(o,'sigmas_'+m).astype(np.float32),True)
    positions=native(n,'positions.f64','<f8').reshape(-1,3)
    live=int(oracle(o,'positive.static_kwargs.packed_seq_params.cu_seqlens_q').reshape(-1)[1])
    add('positions',lambda:positions.astype(np.float32),lambda:oracle(o,'positive.static_kwargs.img_position_ids').reshape(-1,3)[:live],True)
    def condition(name,width):
        info=json.loads((o/(name+'.json')).read_text())
        return np.zeros((0,width),dtype=np.float32) if info is None else oracle(o,name).reshape(-1,width)
    add('condition-video',lambda:native(n,'condition-video.f32').reshape(-1,96),lambda:condition('keyframe_cond_rows',96))
    add('condition-audio',lambda:native(n,'condition-audio.f32').reshape(-1,32),lambda:condition('audio_ref_rows',32))
    result=dict(comparison='conditioned_input_diagnostic',oracle=str(a.oracle),native=str(a.native),
        contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results,
        passed=all(r['passed'] for r in results),first_failing_boundary=next((r['name'] for r in results if not r['passed']),None))
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
