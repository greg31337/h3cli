#!/usr/bin/env python3
"""Compare six identical-input noise levels; retain independent trajectories separately."""
import argparse
import json
from pathlib import Path
import subprocess
from metal_native_bench import ROOT, sha


def load(path):
    r=json.loads((path/'record.json').read_text())
    if not r['evaluation_contract_pass'] or sha(path/'result.h3av')!=r['av_sha256']:
        raise ValueError('Incomplete or changed run: '+str(path))
    if r['evaluations']!=6 or r['geometry']!=[640,480,243] or r['blocks']!=50:
        raise ValueError('Expected six evaluations, 243 frames, all 50 blocks')
    for name,digest in r.get('step_tensor_sha256',{}).items():
        if sha(path/'steps'/name)!=digest:raise ValueError('Changed tensor: '+name)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','m1','fp16','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--independent',type=Path,help='Optional independent FP16 B6 with step captures')
    a=p.parse_args()
    if a.output.exists():p.error('use a fresh output directory')
    a.output.mkdir(parents=True)
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'tensor_metrics.c').write_bytes((ROOT/'tests/metal_tensor_metrics.c').read_bytes())
    paths={'reference':a.reference,'m1':a.m1,'fp16':a.fp16}
    if a.independent:paths['independent-fp16']=a.independent
    runs={k:load(v) for k,v in paths.items()};ref=runs['reference']
    if ref.get('teacher_from'):raise ValueError('Reference must be an independent trajectory')
    if a.independent and runs['independent-fp16'].get('teacher_from'):
        raise ValueError('Independent trajectory must not use teacher inputs')
    for key in ('conditioning_sha256_after','conditioning_identity','seed','prompt','sampler','weight_precision','case'):
        if any(r[key]!=ref[key] for r in runs.values()):raise ValueError('Workload mismatch: '+key)
    for name in ('m1','fp16'):
        r=runs[name]
        if not r.get('teacher_input_pass') or Path(r['teacher_from']).resolve()!=a.reference.resolve():
            raise ValueError('Missing matched teacher input: '+name)
        if len(r['teacher_steps'])!=6:raise ValueError('Incomplete noise-level coverage')
        for tensor,digest in r['teacher_input_sha256'].items():
            if sha(a.reference/'steps'/tensor)!=digest:raise ValueError('Changed reference capture')
    if runs['m1']['teacher_steps']!=runs['fp16']['teacher_steps']:raise ValueError('Noise-level mismatch')
    tool=ROOT/'bin/metal_tensor_metrics'
    tool.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['cc','-O3',str(ROOT/'tests/metal_tensor_metrics.c'),'-lm','-o',str(tool)],check=True)
    result={'version':1,'timing_comparable':False,'script_sha256':sha(__file__),
            'tensor_metric_source_sha256':sha(a.output/'tensor_metrics.c'),
            'record_sha256':{k:sha(v/'record.json') for k,v in paths.items()},'steps':[],
            'scope':'Matched reference inputs at all six sigmas, including BF16 native A and mixed FP16. Independent diffusion trajectories are separate evidence.'}
    for s in range(1,7):
        row={'step':s,'noise':runs['m1']['teacher_steps'][s-1],'inputs':{},'metrics':{}}
        for modality in ('video','audio'):
            filename=f'step-{s:03d}-{modality}-input.f32'
            hashes=[sha(paths[k]/'steps'/filename) for k in ('m1','fp16')]
            if hashes[0]!=hashes[1]:raise ValueError('Candidate input mismatch: '+filename)
            if s>1 and hashes[0]!=sha(a.reference/'steps'/f'step-{s-1:03d}-{modality}-latent.f32'):
                raise ValueError('Teacher input differs from prior reference latent')
            row['inputs'][modality]=hashes[0]
            if a.independent:
                digest=sha(a.independent/'steps'/filename)
                row.setdefault('independent_inputs',{})[modality]={'sha256':digest,'same_as_teacher':digest==hashes[0]}
                expected=hashes[0] if s==1 else sha(a.independent/'steps'/f'step-{s-1:03d}-{modality}-latent.f32')
                if digest!=expected:raise ValueError('Independent trajectory input does not follow its own prior latent')
        for name in (('m1','fp16','independent-fp16') if a.independent else ('m1','fp16')):
            row['metrics'][name]={}
            for tensor in ('video-velocity','audio-velocity','video-latent','audio-latent'):
                filename=f'step-{s:03d}-{tensor}.f32';left=a.reference/'steps'/filename;right=paths[name]/'steps'/filename
                if left.stat().st_size!=right.stat().st_size:raise ValueError('Tensor shape mismatch')
                command=[str(tool.resolve()),str(left.resolve()),str(right.resolve()),'f32']
                metrics=json.loads(subprocess.run(command,capture_output=True,text=True,check=True).stdout)
                row['metrics'][name][tensor]={'metrics':metrics,'reference_sha256':sha(left),'candidate_sha256':sha(right)}
        result['steps'].append(row)
    r=runs['fp16']
    if not r.get('range_capture_pass') or not r.get('mixed_range_pass'):raise ValueError('Missing range/score coverage')
    result['ranges']={stage:{'minimum':min(x['minimum'] for x in r['ranges'] if x['stage']==stage),
        'maximum':max(x['maximum'] for x in r['ranges'] if x['stage']==stage),
        'nonfinite':sum(x['nonfinite'] for x in r['ranges'] if x['stage']==stage),
        'over_fp16':sum(x['over_fp16'] for x in r['ranges'] if x['stage']==stage),
        'below_half_normal':sum(x['below_half_normal'] for x in r['ranges'] if x['stage']==stage)}
        for stage in sorted({x['stage'] for x in r['ranges']})}
    result['scores']={'records':len(r['scores']),'scope':r['scores'][0]['scope'],
        'minimum':min(x['minimum'] for x in r['scores']),'maximum':max(x['maximum'] for x in r['scores']),
        'sum_minimum':min(x['sum_minimum'] for x in r['scores']),'sum_maximum':max(x['sum_maximum'] for x in r['scores']),
        'nonfinite':sum(x['nonfinite'] for x in r['scores']),
        'zero_exp':sum(x['zero_exp'] for x in r['scores']),
        'below_half_normal_exp':sum(x['below_half_normal_exp'] for x in r['scores'])}
    result['mixed']={'records':len(r['mixed']),'recovered_heads':sum(x['recovered_heads'] for x in r['mixed']),
        'invalid_heads':sum(x['invalid_heads'] for x in r['mixed']),
        'underflows':sum(x['underflows'] for x in r['mixed'])}
    result['input_contract_pass']=True
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'input_contract_pass':True,'ranges':result['ranges'],'scores':result['scores'],'mixed':result['mixed']},indent=2))


if __name__=='__main__':main()
